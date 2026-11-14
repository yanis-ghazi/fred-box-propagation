import numpy as np
import pandas as pd

from boxprop.boxes import area, center, iou

SIGNALS = ["fb", "mask_conf", "obj_score", "jump", "size", "empty"]


def _boxes(df: pd.DataFrame, prefix: str = "") -> np.ndarray:
    return df[[f"{prefix}x1", f"{prefix}y1", f"{prefix}x2", f"{prefix}y2"]].to_numpy(dtype=float)


def jump_signal(boxes: np.ndarray, scale: float = 1.0) -> np.ndarray:
    c = center(boxes)
    a = area(boxes)
    d = np.zeros(len(boxes))
    prev_c, prev_a = c[0], a[0]
    for t in range(1, len(boxes)):
        if np.isfinite(c[t]).all() and np.isfinite(prev_c).all() and prev_a > 0:
            d[t] = np.linalg.norm(c[t] - prev_c) / np.sqrt(prev_a)
        if np.isfinite(c[t]).all():
            prev_c, prev_a = c[t], a[t]
    return 1 - np.exp(-d / scale)


def size_signal(boxes: np.ndarray, scale: float = np.log(2)) -> np.ndarray:
    a = area(boxes)
    s = np.zeros(len(boxes))
    prev = a[0]
    for t in range(1, len(boxes)):
        if a[t] > 0 and prev > 0:
            s[t] = abs(np.log(a[t] / prev))
        if a[t] > 0:
            prev = a[t]
    return 1 - np.exp(-s / scale)


def signals(fwd: pd.DataFrame, bwd: pd.DataFrame) -> pd.DataFrame:
    f = fwd.sort_values("frame").reset_index(drop=True)
    b = bwd.set_index("frame").reindex(f["frame"]).reset_index()
    fb_ = _boxes(f)
    out = pd.DataFrame({"frame": f["frame"]})
    out["fb"] = 1 - iou(fb_, _boxes(b))
    out["mask_conf"] = 1 - f["pred_iou"].clip(0, 1).fillna(0).to_numpy()
    out["obj_score"] = 1 - 1 / (1 + np.exp(-f["obj_logit"].fillna(-20).to_numpy()))
    out["jump"] = jump_signal(fb_)
    out["size"] = size_signal(fb_)
    out["empty"] = (~np.isfinite(fb_).all(axis=1)).astype(float)
    return out


def combine(sig: pd.DataFrame, cols: list[str] | None = None) -> np.ndarray:
    cols = cols or SIGNALS
    motion = sig[["jump", "size"]].max(axis=1) if {"jump", "size"} <= set(cols) else None
    parts = [sig[c].to_numpy() for c in cols if c not in ("jump", "size")]
    if motion is not None:
        parts.append(motion.to_numpy())
    return np.mean(parts, axis=0)


def auroc(score: np.ndarray, positive: np.ndarray) -> float:
    s = np.asarray(score, dtype=float)
    y = np.asarray(positive, dtype=bool)
    n_pos, n_neg = y.sum(), (~y).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pd.Series(s).rank(method="average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def loso_logistic(df: pd.DataFrame, cols: list[str], target: str, group: str) -> np.ndarray:
    from sklearn.linear_model import LogisticRegression

    pred = np.full(len(df), np.nan)
    for g in df[group].unique():
        test = (df[group] == g).to_numpy()
        y = df.loc[~test, target].to_numpy()
        if len(np.unique(y)) < 2:
            continue
        model = LogisticRegression(max_iter=1000, class_weight="balanced")
        model.fit(df.loc[~test, cols].to_numpy(), y)
        pred[test] = model.predict_proba(df.loc[test, cols].to_numpy())[:, 1]
    return pred
