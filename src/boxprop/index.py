import argparse
from pathlib import Path

from boxprop import params
from boxprop.fred import build_index


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/index.csv")
    args = ap.parse_args()
    p = params.load()
    df = build_index(p["fred"]["revision"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    ok = df[df["error"].fillna("") == ""]
    print(f"{len(df)} sequences, {len(ok)} indexed, {int((ok['max_simultaneous'] == 1).sum())} with one drone")


if __name__ == "__main__":
    main()
