import argparse
from pathlib import Path

import pandas as pd

from boxprop import params


def select(index: pd.DataFrame, n: int, clip_length: int, skip: int, split: str, max_per_class: int) -> pd.DataFrame:
    df = index[index["error"].fillna("") == ""].copy()
    df = df[(df["max_simultaneous"] == 1) & (df["has_extra"] == 0) & (df["run_length"] >= clip_length + skip)]
    df["seq_num"] = df["seq"].astype(int)
    df["preferred"] = (df["split"] == split).astype(int)
    df = df.sort_values(["preferred", "seq_num"], ascending=[False, True])
    picked, per_class = [], {}
    for _, r in df.iterrows():
        if per_class.get(r["drone_class"], 0) >= max_per_class:
            continue
        picked.append(r)
        per_class[r["drone_class"]] = per_class.get(r["drone_class"], 0) + 1
        if len(picked) == n:
            break
    out = pd.DataFrame(picked)
    out["clip_start"] = out["run_start"] + skip
    out["clip_length"] = clip_length
    return out[["split", "seq", "drone_class", "run_start", "run_length", "clip_start", "clip_length",
                "box_w_median", "box_h_median"]].reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", default="data/index.csv")
    ap.add_argument("--out", default="data/selection.csv")
    args = ap.parse_args()
    p = params.load()["select"]
    sel = select(pd.read_csv(args.index, dtype={"seq": str}), p["n_sequences"], p["clip_length"],
                 p["skip_frames"], p["split"], p["max_per_class"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    sel.to_csv(args.out, index=False)
    print(sel.to_string())


if __name__ == "__main__":
    main()
