import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from boxprop import params
from boxprop.ecf import read_events
from boxprop.events import evframe_mask, histogram, mask_iou, render, time_surface

MARGIN_US = 200_000


def to_dict(ev: np.ndarray) -> dict[str, np.ndarray]:
    return {"t": ev["t"].astype(np.int64), "x": ev["x"].astype(np.int16), "y": ev["y"].astype(np.int16),
            "p": ev["p"].astype(np.int8)}


def represent_clip(clip: Path, h5: Path, out: Path, p: dict) -> dict:
    meta = json.loads((clip / "meta.json").read_text())
    ts = np.asarray(meta["frame_ts_us"], dtype=np.int64)
    ev = to_dict(read_events(h5, int(ts[0]) - MARGIN_US, int(ts[-1]) + 1))
    reps = {f"hist{w}": ("histogram", w * 1000) for w in p["hist_windows_ms"]}
    if p.get("ts_tau_ms"):
        reps[f"ts{p['ts_tau_ms']}"] = ("time_surface", p["ts_tau_ms"] * 1000)
    for name in reps:
        (out / name).mkdir(parents=True, exist_ok=True)
    for i, t in enumerate(ts):
        for name, (kind, v) in reps.items():
            rep = histogram(ev, int(t), v) if kind == "histogram" else time_surface(ev, int(t), v)
            Image.fromarray(render(rep, kind, p["count_clip"])).save(out / name / f"{i:05d}.png", compress_level=1)
    check = np.linspace(0, len(ts) - 1, p["align_frames"]).round().astype(int)
    scores = []
    for i in check:
        m = evframe_mask(np.asarray(Image.open(clip / "evframe" / f"{i:05d}.png").convert("RGB")))
        scores.append(mask_iou(histogram(ev, int(ts[i]), 33333).sum(axis=0) > 0, m))
    return {"seq": meta["seq"], "n_events": int(len(ev["t"])), "events_per_s": float(len(ev["t"]) / ((ts[-1] - ts[0] + MARGIN_US) / 1e6)),
            "align_iou_mean": float(np.mean(scores)), "align_iou_min": float(np.min(scores))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/clips")
    ap.add_argument("--events", default="data/events")
    ap.add_argument("--reps", default="data/reps")
    ap.add_argument("--out", default="results/representations.csv")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    p = params.load()["represent"]
    clips = sorted(Path(args.clips).iterdir(), key=lambda c: int(c.name))
    with ProcessPoolExecutor(args.workers) as ex:
        rows = list(ex.map(represent_clip, clips, [Path(args.events) / f"{c.name}.hdf5" for c in clips],
                           [Path(args.reps) / c.name for c in clips], [p] * len(clips)))
    for r in rows:
        print(r, flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)


if __name__ == "__main__":
    main()
