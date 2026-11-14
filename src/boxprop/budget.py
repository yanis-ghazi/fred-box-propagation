from typing import Callable, Iterable

import numpy as np
import pandas as pd

from boxprop.boxes import iou

RunFn = Callable[[int, np.ndarray, int], np.ndarray]


def greedy_pick(order: Iterable[int], n: int, min_gap: int, taken: Iterable[int] = (0,)) -> list[int]:
    fixed = list(taken)
    chosen: list[int] = []
    if n <= 0:
        return chosen
    for f in order:
        f = int(f)
        if all(abs(f - c) >= min_gap for c in fixed + chosen):
            chosen.append(f)
            if len(chosen) == n:
                break
    return sorted(chosen)


def fixed_keyframes(length: int, n: int) -> list[int]:
    if n <= 0:
        return []
    return sorted({int(round(i * length / (n + 1))) for i in range(1, n + 1)} - {0})


def random_keyframes(length: int, n: int, rng: np.random.Generator, min_gap: int = 1) -> list[int]:
    return greedy_pick(rng.permutation(np.arange(1, length)), n, min_gap)


def score_keyframes(score: np.ndarray, n: int, min_gap: int) -> list[int]:
    s = np.nan_to_num(np.asarray(score, dtype=float), nan=np.inf)
    order = np.argsort(-s, kind="stable")
    return greedy_pick(order[order > 0], n, min_gap)


def oracle_keyframes(ious: np.ndarray, n: int, min_gap: int) -> list[int]:
    order = np.argsort(np.asarray(ious, dtype=float), kind="stable")
    return greedy_pick(order[order > 0], n, min_gap)


def segments(keyframes: Iterable[int], length: int) -> list[tuple[int, int]]:
    starts = [0] + sorted(int(k) for k in keyframes)
    ends = starts[1:] + [length]
    return list(zip(starts, ends))


def required_runs(plans: Iterable[Iterable[int]], length: int) -> dict[int, int]:
    need: dict[int, int] = {}
    for keyframes in plans:
        for s, e in segments(keyframes, length)[1:]:
            need[s] = max(need.get(s, 0), e - s)
    return dict(sorted(need.items()))


def assemble(base: np.ndarray, runs: dict[int, np.ndarray], keyframes: Iterable[int], gt: np.ndarray) -> np.ndarray:
    length = len(base)
    out = np.array(base, dtype=float, copy=True)
    for s, e in segments(keyframes, length)[1:]:
        out[s:e] = runs[s][: e - s]
    for k in [0] + list(keyframes):
        out[k] = gt[k]
    return out


def quality(boxes: np.ndarray, gt: np.ndarray, thr: float = 0.5) -> dict:
    v = iou(boxes, gt)
    return {"miou": float(v.mean()), "iou50": float((v >= thr).mean())}


def plan_all(length: int, budgets: Iterable[int], score: np.ndarray, base_iou: np.ndarray, min_gap: int,
             random_seeds: Iterable[int]) -> list[dict]:
    plans = []
    for n in budgets:
        plans.append({"strategy": "fixed", "budget": n, "seed": -1, "keyframes": fixed_keyframes(length, n)})
        for seed in random_seeds:
            rng = np.random.default_rng(seed * 1000 + n)
            plans.append({"strategy": "random", "budget": n, "seed": seed,
                          "keyframes": random_keyframes(length, n, rng, min_gap)})
        plans.append({"strategy": "score", "budget": n, "seed": -1, "keyframes": score_keyframes(score, n, min_gap)})
        plans.append({"strategy": "oracle", "budget": n, "seed": -1,
                      "keyframes": oracle_keyframes(base_iou, n, min_gap)})
    return plans


def simulate(base: np.ndarray, gt: np.ndarray, plans: list[dict], run: RunFn,
             keep: Callable[[dict], bool] | None = None) -> tuple[pd.DataFrame, dict, dict]:
    length = len(base)
    need = required_runs([p["keyframes"] for p in plans], length)
    runs = {s: np.asarray(run(s, gt[s], n), dtype=float) for s, n in need.items()}
    rows, kept = [], {}
    for p in plans:
        boxes = assemble(base, runs, p["keyframes"], gt)
        rows.append({k: v for k, v in p.items() if k != "keyframes"} | quality(boxes, gt) |
                    {"n_corrections": len(p["keyframes"]), "keyframes": " ".join(map(str, p["keyframes"]))})
        if keep is not None and keep(p):
            kept[(p["strategy"], p["budget"], p["seed"])] = boxes
    return pd.DataFrame(rows), {"n_runs": len(runs), "frames_propagated": int(sum(need.values()))}, kept
