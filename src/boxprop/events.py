import numpy as np

from boxprop.boxes import H, W

FRAME_US = 33333
EVFRAME_BG = np.array([30, 37, 52], dtype=np.uint8)


def window(ev: dict[str, np.ndarray], t0: int, t1: int) -> dict[str, np.ndarray]:
    i0, i1 = np.searchsorted(ev["t"], [t0, t1])
    return {k: v[i0:i1] for k, v in ev.items()}


def histogram(ev: dict[str, np.ndarray], t_end: int, window_us: int) -> np.ndarray:
    w = window(ev, t_end - window_us, t_end)
    keep = (w["x"] >= 0) & (w["x"] < W) & (w["y"] >= 0) & (w["y"] < H)
    ch = 1 - (w["p"][keep] > 0).astype(np.int64)
    idx = (ch * H + w["y"][keep].astype(np.int64)) * W + w["x"][keep].astype(np.int64)
    return np.bincount(idx, minlength=2 * H * W).reshape(2, H, W).astype(np.float32)


def time_surface(ev: dict[str, np.ndarray], t_end: int, tau_us: int, horizon_us: int | None = None) -> np.ndarray:
    horizon_us = horizon_us or 5 * tau_us
    w = window(ev, t_end - horizon_us, t_end)
    keep = (w["x"] >= 0) & (w["x"] < W) & (w["y"] >= 0) & (w["y"] < H)
    last = np.full((2, H, W), -np.inf)
    ch = 1 - (w["p"][keep] > 0).astype(int)
    last[ch, w["y"][keep], w["x"][keep]] = w["t"][keep]
    return np.exp((last - t_end) / tau_us).astype(np.float32)


def render(rep: np.ndarray, kind: str, count_clip: float = 3.0) -> np.ndarray:
    if kind == "histogram":
        n = np.clip(rep / count_clip, 0, 1)
    else:
        n = np.clip(rep, 0, 1)
    img = np.stack([n[0], n[1], np.maximum(n[0], n[1])], axis=-1)
    return (255 * img).round().astype(np.uint8)


def evframe_mask(img: np.ndarray) -> np.ndarray:
    return (img != EVFRAME_BG).any(axis=-1)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 0.0


def estimate_offset(ev: dict[str, np.ndarray], frame_ts: list[int], masks: list[np.ndarray],
                    windows_us: list[int], offsets_us: range) -> dict:
    best = {"score": -1.0}
    for win in windows_us:
        for off in offsets_us:
            s = []
            for ts, m in zip(frame_ts, masks):
                h = histogram(ev, ts + off, win).sum(axis=0) > 0
                s.append(mask_iou(h, m))
            score = float(np.mean(s))
            if score > best["score"]:
                best = {"score": score, "window_us": int(win), "offset_us": int(off)}
    return best
