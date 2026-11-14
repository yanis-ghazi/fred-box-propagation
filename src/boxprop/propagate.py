import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from boxprop import params
from boxprop.boxes import clip_box, iou
from boxprop.budget import plan_all, simulate
from boxprop.drift import combine, signals

BOX = ["x1", "y1", "x2", "y2"]


def clip_frames(clip: Path, rep: str, reps: Path = Path("data/reps")) -> list[Path]:
    d = clip / rep if (clip / rep).exists() else reps / clip.name / rep
    return sorted(d.glob("*.jpg")) + sorted(d.glob("*.png"))


def gt_boxes(clip: Path, length: int, size: tuple[int, int]) -> np.ndarray:
    b = pd.read_csv(clip / "boxes.csv").set_index("i").reindex(range(length))
    return clip_box(b[BOX].to_numpy(dtype=float), *size)


def last_valid(df: pd.DataFrame) -> np.ndarray | None:
    ok = df.dropna(subset=BOX)
    return None if ok.empty else ok.iloc[-1][BOX].to_numpy(dtype=float)


def run_clip(prop, clip: Path, reps: Path, rep: str, p_sim: dict | None) -> tuple[pd.DataFrame, pd.DataFrame | None, dict]:
    paths = clip_frames(clip, rep, reps)
    n = len(paths)
    meta = json.loads((clip / "meta.json").read_text())
    gt = gt_boxes(clip, n, tuple(meta["rgb_size"]))
    t_enc = prop.encode(paths)
    t0 = time.time()
    fwd = prop.run(0, gt[0], n)
    t_fwd = time.time() - t0
    end_box = last_valid(fwd)
    t0 = time.time()
    bwd = prop.run(n - 1, end_box if end_box is not None else gt[0], n, reverse=True)
    t_bwd = time.time() - t0
    out = fwd.copy()
    for c in BOX + ["pred_iou", "obj_logit", "mask_area"]:
        out[f"bwd_{c}"] = bwd.set_index("frame").reindex(out["frame"])[c].to_numpy()
    for k, c in enumerate(BOX):
        out[f"gt_{c}"] = gt[:, k]
    out["iou"] = iou(out[BOX].to_numpy(dtype=float), gt)
    sig = signals(fwd, bwd)
    out = out.merge(sig, on="frame")
    out["score"] = combine(sig)
    timing = {"seq": meta["seq"], "rep": rep, "frames": n, "encode_s": t_enc, "forward_s": t_fwd, "backward_s": t_bwd}
    sim = None
    if p_sim is not None:
        plans = plan_all(n, p_sim["budgets"], out["score"].to_numpy(), out["iou"].to_numpy(), p_sim["min_gap"],
                         p_sim["random_seeds"])
        plans.append({"strategy": "none", "budget": 0, "seed": -1, "keyframes": []})
        t0 = time.time()
        sim, info, kept = simulate(out[BOX].to_numpy(dtype=float), gt, plans, prop.run_boxes,
                                   keep=lambda q: q["strategy"] in ("score", "fixed") and q["budget"] in (1, 4))
        timing |= {"simulate_s": time.time() - t0} | info
        sim.insert(0, "rep", rep)
        sim.insert(0, "seq", meta["seq"])
        for (strat, budget, _), boxes in kept.items():
            out[f"iou_{strat}{budget}"] = iou(boxes, gt)
            for k, c in enumerate(BOX):
                out[f"{strat}{budget}_{c}"] = boxes[:, k]
    return out, sim, timing


def worker(gpu: int, clips: list[str], args: dict) -> tuple[list, list, dict]:
    import torch

    from boxprop.sam2prop import Propagator

    torch.cuda.set_device(gpu)
    p = params.load()
    pp, ps = p["propagate"], p["simulate"]
    prop = Propagator(pp["model"], args["ckpt_dir"], device=f"cuda:{gpu}", fp16=pp["fp16"])
    out_dir = Path(args["out"])
    sims, timings, check = [], [], {}
    for name in clips:
        clip = Path(args["clips"]) / name
        for rep in pp["representations"]:
            df, sim, timing = run_clip(prop, clip, Path(args["reps"]), rep,
                                       ps if rep in ps["representations"] else None)
            df.to_csv(out_dir / "propagation" / f"{name}_{rep}.csv", index=False, float_format="%.4f")
            timings.append(timing | {"gpu": gpu})
            if sim is not None:
                sims.append(sim)
            if args["check_fp32"] and not check and pp["fp16"]:
                ref = Propagator(pp["model"], args["ckpt_dir"], device=f"cuda:{gpu}", fp16=False)
                n = min(100, len(df))
                ref.encode(clip_frames(clip, rep, Path(args["reps"]))[:n])
                b32 = ref.run(0, df[[f"gt_{c}" for c in BOX]].to_numpy(dtype=float)[0], n)[BOX].to_numpy(dtype=float)
                agree = iou(df[BOX].to_numpy(dtype=float)[:n], b32)
                check = {"seq": name, "rep": rep, "frames": n, "fp16_vs_fp32_iou_mean": float(agree.mean()),
                         "fp16_vs_fp32_iou_min": float(agree.min())}
                del ref
                torch.cuda.empty_cache()
            print(name, rep, f"miou={df['iou'].mean():.3f}", {k: round(v, 1) if isinstance(v, float) else v
                                                             for k, v in timing.items()}, flush=True)
    return sims, timings, check


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/clips")
    ap.add_argument("--reps", default="data/reps")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--gpus", type=int, default=None)
    args = ap.parse_args()
    import multiprocessing as mp

    import torch

    out_dir = Path(args.out)
    (out_dir / "propagation").mkdir(parents=True, exist_ok=True)
    clips = sorted((c.name for c in Path(args.clips).iterdir()), key=int)
    n_gpu = args.gpus or max(1, torch.cuda.device_count())
    from boxprop.sam2prop import Propagator

    Propagator.fetch_checkpoint(params.load()["propagate"]["model"], args.ckpt_dir)
    shared = {"clips": args.clips, "reps": args.reps, "out": args.out, "ckpt_dir": args.ckpt_dir}
    parts = [clips[k::n_gpu] for k in range(n_gpu)]
    with ProcessPoolExecutor(n_gpu, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(worker, range(n_gpu), parts, [shared | {"check_fp32": k == 0} for k in range(n_gpu)]))
    sims = [s for r in res for s in r[0]]
    timings = [t for r in res for t in r[1]]
    pd.DataFrame(timings).sort_values(["seq", "rep"]).to_csv(out_dir / "timing.csv", index=False, float_format="%.3f")
    pd.concat(sims).to_csv(out_dir / "simulation.csv", index=False, float_format="%.5f")
    checks = [r[2] for r in res if r[2]]
    (out_dir / "precision_check.json").write_text(json.dumps(checks[0] if checks else {}, indent=1))


if __name__ == "__main__":
    main()
