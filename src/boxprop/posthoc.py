import argparse
import json
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from boxprop import params
from boxprop.budget import fixed_keyframes, score_keyframes, simulate
from boxprop.propagate import BOX, clip_frames, gt_boxes


def plans_for(df: pd.DataFrame, budgets: list[int], min_gap: int) -> list[dict]:
    n = len(df)
    conf = 1 - df["pred_iou"].clip(0, 1).fillna(0).to_numpy()
    plans = [{"strategy": "none", "budget": 0, "seed": -1, "keyframes": []}]
    for b in budgets:
        plans.append({"strategy": "fixed", "budget": b, "seed": -1, "keyframes": fixed_keyframes(n, b)})
        plans.append({"strategy": "mask_conf", "budget": b, "seed": -1, "keyframes": score_keyframes(conf, b, min_gap)})
    return plans


def worker(gpu: int, clips: list[str], args: dict) -> list[pd.DataFrame]:
    import torch

    from boxprop.sam2prop import Propagator

    torch.cuda.set_device(gpu)
    p = params.load()
    prop = Propagator(p["propagate"]["model"], args["ckpt_dir"], device=f"cuda:{gpu}", fp16=p["propagate"]["fp16"])
    out = []
    for name in clips:
        clip = Path(args["clips"]) / name
        base = pd.read_csv(Path(args["results"]) / "propagation" / f"{name}_{args['rep']}.csv")
        paths = clip_frames(clip, args["rep"], Path(args["reps"]))
        meta = json.loads((clip / "meta.json").read_text())
        gt = gt_boxes(clip, len(paths), tuple(meta["rgb_size"]))
        prop.encode(paths)
        sim, info, _ = simulate(base[BOX].to_numpy(dtype=float), gt,
                                plans_for(base, p["simulate"]["budgets"], p["simulate"]["min_gap"]), prop.run_boxes)
        sim.insert(0, "rep", args["rep"])
        sim.insert(0, "seq", name)
        out.append(sim)
        print(name, info, flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", default="data/clips")
    ap.add_argument("--reps", default="data/reps")
    ap.add_argument("--results", default="results")
    ap.add_argument("--rep", default="evframe")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    args = ap.parse_args()
    import torch

    from boxprop.sam2prop import Propagator

    Propagator.fetch_checkpoint(params.load()["propagate"]["model"], args.ckpt_dir)
    clips = sorted((c.name for c in Path(args.clips).iterdir()), key=int)
    n_gpu = max(1, torch.cuda.device_count())
    shared = {"clips": args.clips, "reps": args.reps, "results": args.results, "rep": args.rep, "ckpt_dir": args.ckpt_dir}
    with ProcessPoolExecutor(n_gpu, mp_context=mp.get_context("spawn")) as ex:
        res = list(ex.map(worker, range(n_gpu), [clips[k::n_gpu] for k in range(n_gpu)], [shared] * n_gpu))
    sim = pd.concat([s for r in res for s in r])
    res_dir = Path(args.results)
    sim.to_csv(res_dir / "posthoc_simulation.csv", index=False, float_format="%.5f")
    per_seq = sim.groupby(["strategy", "budget", "seq"])[["miou", "iou50"]].mean().reset_index()
    summary = per_seq.groupby(["strategy", "budget"]).agg(miou=("miou", "mean"), miou_sd=("miou", "std"),
                                                          iou50=("iou50", "mean"), sequences=("seq", "nunique"))
    summary.to_csv(res_dir / "posthoc_summary.csv", float_format="%.4f")
    print(summary.to_string())


if __name__ == "__main__":
    main()
