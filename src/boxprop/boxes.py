import numpy as np

W, H = 1280, 720


def clip_box(box: np.ndarray, width: int = W, height: int = H) -> np.ndarray:
    b = np.asarray(box, dtype=float).copy()
    b[..., [0, 2]] = b[..., [0, 2]].clip(0, width)
    b[..., [1, 3]] = b[..., [1, 3]].clip(0, height)
    return b


def area(box: np.ndarray) -> np.ndarray:
    b = np.asarray(box, dtype=float)
    return (b[..., 2] - b[..., 0]).clip(0) * (b[..., 3] - b[..., 1]).clip(0)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ix = (np.minimum(a[..., 2], b[..., 2]) - np.maximum(a[..., 0], b[..., 0])).clip(0)
    iy = (np.minimum(a[..., 3], b[..., 3]) - np.maximum(a[..., 1], b[..., 1])).clip(0)
    inter = ix * iy
    union = area(a) + area(b) - inter
    out = np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)
    return np.nan_to_num(out, nan=0.0)


def center(box: np.ndarray) -> np.ndarray:
    b = np.asarray(box, dtype=float)
    return np.stack([(b[..., 0] + b[..., 2]) / 2, (b[..., 1] + b[..., 3]) / 2], axis=-1)


def mask_to_box(mask: np.ndarray) -> np.ndarray:
    ys = np.flatnonzero(mask.any(axis=1))
    xs = np.flatnonzero(mask.any(axis=0))
    if len(xs) == 0:
        return np.full(4, np.nan)
    return np.array([xs[0], ys[0], xs[-1] + 1, ys[-1] + 1], dtype=float)


def nan_box() -> np.ndarray:
    return np.full(4, np.nan)
