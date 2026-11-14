import json
import re
import struct
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO = "GabrieleMagrini/FRED"
FRAME_S = 0.033333
FRAME_US = 33333


def zip_url(revision: str, split: str, seq: str) -> str:
    return f"https://huggingface.co/datasets/{REPO}/resolve/{revision}/{split}/{seq}.zip"


def list_sequences(revision: str) -> list[tuple[str, str]]:
    out = []
    for split in ["train", "test"]:
        url = f"https://huggingface.co/api/datasets/{REPO}/tree/{revision}/{split}"
        items = requests.get(url, timeout=60).json()
        out += [(split, Path(x["path"]).stem) for x in items if x["path"].endswith(".zip")]
    return out


def _get(url: str, rng: str, retries: int = 4) -> requests.Response:
    for k in range(retries):
        try:
            r = requests.get(url, headers={"Range": f"bytes={rng}"}, timeout=120)
            if r.status_code in (200, 206):
                return r
        except requests.RequestException:
            pass
        time.sleep(2 ** k)
    raise RuntimeError(f"range request failed: {url} {rng}")


def remote_zip_entries(url: str, tail: int = 3_000_000) -> dict[str, tuple[int, int, int, int]]:
    r = _get(url, f"-{tail}")
    buf = r.content
    total = int(r.headers["Content-Range"].split("/")[1])
    off0 = total - len(buf)
    eocd = buf.rfind(b"PK\x05\x06")
    n, _, cd_off = struct.unpack("<HII", buf[eocd + 10:eocd + 20])
    if cd_off == 0xFFFFFFFF or n == 0xFFFF:
        loc = buf.rfind(b"PK\x06\x07", 0, eocd)
        z = struct.unpack("<Q", buf[loc + 8:loc + 16])[0] - off0
        n, _, cd_off = struct.unpack("<QQQ", buf[z + 32:z + 56])
    p = cd_off - off0
    if p < 0:
        if tail >= total:
            raise RuntimeError("central directory not found")
        return remote_zip_entries(url, tail * 4)
    entries = {}
    for _ in range(n):
        method, = struct.unpack("<H", buf[p + 10:p + 12])
        csize, usize = struct.unpack("<II", buf[p + 20:p + 28])
        fl, el, cl = struct.unpack("<HHH", buf[p + 28:p + 34])
        lho, = struct.unpack("<I", buf[p + 42:p + 46])
        name = buf[p + 46:p + 46 + fl].decode("utf-8", "replace")
        extra = buf[p + 46 + fl:p + 46 + fl + el]
        q = 0
        while q + 4 <= len(extra):
            hid, hsz = struct.unpack("<HH", extra[q:q + 4])
            if hid == 1:
                d, k = extra[q + 4:q + 4 + hsz], 0
                if usize == 0xFFFFFFFF:
                    usize, = struct.unpack("<Q", d[k:k + 8])
                    k += 8
                if csize == 0xFFFFFFFF:
                    csize, = struct.unpack("<Q", d[k:k + 8])
                    k += 8
                if lho == 0xFFFFFFFF:
                    lho, = struct.unpack("<Q", d[k:k + 8])
            q += 4 + hsz
        entries[name] = (method, csize, usize, lho)
        p += 46 + fl + el + cl
    return entries


def remote_zip_read(url: str, entry: tuple[int, int, int, int]) -> bytes:
    method, csize, _, lho = entry
    b = _get(url, f"{lho}-{lho + 30 + 4096 + csize}").content
    fl, el = struct.unpack("<HH", b[26:30])
    data = b[30 + fl + el:30 + fl + el + csize]
    return zlib.decompress(data, -15) if method == 8 else data


def parse_coordinates(text: str) -> pd.DataFrame:
    rows = []
    for line in text.splitlines():
        if ": " not in line:
            continue
        t, rest = line.strip().split(": ", 1)
        parts = [s.strip() for s in rest.split(",")]
        try:
            x1, y1, x2, y2 = map(float, parts[:4])
            rows.append((float(t), x1, y1, x2, y2, int(float(parts[4])), ",".join(parts[5:]).strip()))
        except (ValueError, IndexError):
            continue
    df = pd.DataFrame(rows, columns=["t", "x1", "y1", "x2", "y2", "id", "cls"])
    df["frame"] = (np.round(df["t"] / FRAME_S) - 1).astype(int)
    return df


def frame_number(name: str) -> int:
    return int(re.findall(r"(\d+)\.png$", name)[0])


def single_drone_runs(df: pd.DataFrame) -> list[tuple[int, int, int]]:
    per_frame = df.groupby("frame")["id"].nunique()
    frames = np.sort(per_frame[per_frame == 1].index.to_numpy())
    frames = frames[frames >= 0]
    if len(frames) == 0:
        return []
    ids = df[df["frame"].isin(frames)].groupby("frame")["id"].first().reindex(frames).to_numpy()
    breaks = np.where((np.diff(frames) != 1) | (np.diff(ids) != 0))[0]
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, len(frames) - 1]
    return [(int(frames[s]), int(frames[e] - frames[s] + 1), int(ids[s])) for s, e in zip(starts, ends)]


def sequence_stats(revision: str, split: str, seq: str) -> dict:
    url = zip_url(revision, split, seq)
    try:
        ent = remote_zip_entries(url)
        n_rgb = sum(1 for k in ent if re.match(rf"{seq}/RGB/.*\.jpg$", k))
        n_ev = sum(1 for k in ent if re.match(rf"{seq}/Event/Frames/.*\.png$", k))
        df = parse_coordinates(remote_zip_read(url, ent[f"{seq}/coordinates.txt"]).decode("utf-8", "replace"))
        runs = single_drone_runs(df)
        start, length, drone = max(runs, key=lambda r: r[1]) if runs else (-1, 0, -1)
        clip = df[(df["frame"] >= start) & (df["frame"] < start + length)]
        w, h = clip["x2"] - clip["x1"], clip["y2"] - clip["y1"]
        cls = clip["cls"].mode().iloc[0] if len(clip) else ""
        return dict(split=split, seq=seq, zip_bytes=sum(v[1] for v in ent.values()), n_rgb=n_rgb, n_evframes=n_ev,
                    n_ids=int(df["id"].nunique()), max_simultaneous=int(df.groupby("frame")["id"].nunique().max()),
                    run_start=start, run_length=length, run_drone=drone, drone_class=cls,
                    box_w_median=float(w.median()), box_h_median=float(h.median()),
                    has_extra=int(any("_extra/" in k for k in ent)), error="")
    except Exception as e:
        return dict(split=split, seq=seq, error=repr(e)[:200])


def build_index(revision: str, workers: int = 16) -> pd.DataFrame:
    todo = list_sequences(revision)
    with ThreadPoolExecutor(workers) as ex:
        rows = list(ex.map(lambda s: sequence_stats(revision, *s), todo))
    return pd.DataFrame(rows).sort_values(["split", "seq"], key=lambda c: c.map(lambda v: (len(str(v)), str(v))))


def download(url: str, dest: Path, retries: int = 4) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    for k in range(retries):
        try:
            with requests.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 22):
                        f.write(chunk)
            tmp.rename(dest)
            return dest
        except (requests.RequestException, OSError):
            time.sleep(2 ** k)
    raise RuntimeError(f"download failed: {url}")


def extract_clip(zip_path: Path, seq: str, start: int, length: int, out: Path, h5_dest: Path) -> dict:
    from PIL import Image

    out.mkdir(parents=True, exist_ok=True)
    (out / "rgb").mkdir(exist_ok=True)
    (out / "evframe").mkdir(exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        rgb = sorted(k for k in names if re.match(rf"{seq}/RGB/[^/]+\.jpg$", k))
        evf = sorted((k for k in names if re.match(rf"{seq}/Event/Frames/[^/]+\.png$", k)), key=frame_number)
        coords = parse_coordinates(z.read(f"{seq}/coordinates.txt").decode("utf-8", "replace"))
        for i in range(length):
            f = start + i
            (out / "rgb" / f"{i:05d}.jpg").write_bytes(z.read(rgb[f]))
            (out / "evframe" / f"{i:05d}.png").write_bytes(z.read(evf[f]))
        h5_dest.parent.mkdir(parents=True, exist_ok=True)
        with z.open(f"{seq}/Event/events.hdf5") as src, open(h5_dest, "wb") as dst:
            while chunk := src.read(1 << 22):
                dst.write(chunk)
    clip = coords[(coords["frame"] >= start) & (coords["frame"] < start + length)].copy()
    clip = clip.sort_values("frame").drop_duplicates("frame")
    clip.insert(0, "i", clip["frame"] - start)
    clip[["i", "frame", "x1", "y1", "x2", "y2", "id", "cls"]].to_csv(out / "boxes.csv", index=False)
    size = Image.open(out / "rgb" / "00000.jpg").size
    meta = {"seq": seq, "start": start, "length": length, "rgb_size": list(size),
            "frame_ts_us": [frame_number(evf[start + i]) for i in range(length)],
            "rgb_names": [Path(rgb[start + i]).name for i in range(0, length, max(1, length // 4))]}
    (out / "meta.json").write_text(json.dumps(meta))
    return meta
