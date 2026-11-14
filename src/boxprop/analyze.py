import argparse
from pathlib import Path

import pandas as pd


def correction_effect(results: Path, rep: str, budget: int, window: int) -> pd.DataFrame:
    sim = pd.read_csv(results / "simulation.csv", dtype={"seq": str})
    rows = []
    for seq in sorted(sim["seq"].unique(), key=int):
        p = pd.read_csv(results / "propagation" / f"{seq}_{rep}.csv")
        for strat in ["fixed", "score"]:
            k = int(str(sim[(sim["seq"] == seq) & (sim["rep"] == rep) & (sim["strategy"] == strat) &
                            (sim["budget"] == budget)]["keyframes"].iloc[0]).split()[0])
            after = slice(k + 1, k + 1 + window)
            rows.append({"seq": seq, "strategy": strat, "keyframe": k, "iou_at_keyframe_uncorrected": p["iou"].iloc[k],
                         "iou_after_uncorrected": p["iou"].iloc[after].mean(),
                         "iou_after_corrected": p[f"iou_{strat}{budget}"].iloc[after].mean()})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--window", type=int, default=30)
    args = ap.parse_args()
    res = Path(args.results)
    per_clip = correction_effect(res, "evframe", 1, args.window)
    per_clip.to_csv(res / "correction_effect_per_clip.csv", index=False, float_format="%.4f")
    summary = per_clip.groupby("strategy")[["keyframe", "iou_at_keyframe_uncorrected", "iou_after_uncorrected",
                                            "iou_after_corrected"]].mean()
    summary.to_csv(res / "correction_effect.csv", float_format="%.4f")
    print(summary.to_string())


if __name__ == "__main__":
    main()
