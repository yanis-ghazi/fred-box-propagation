import argparse
import shutil
from pathlib import Path

import pandas as pd

from boxprop import params
from boxprop.fred import download, extract_clip, zip_url


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", default="data/selection.csv")
    ap.add_argument("--out", default="data/clips")
    ap.add_argument("--events", default="data/events")
    ap.add_argument("--tmp", default="data/tmp")
    args = ap.parse_args()
    rev = params.load()["fred"]["revision"]
    sel = pd.read_csv(args.selection, dtype={"seq": str})
    tmp = Path(args.tmp)
    for _, r in sel.iterrows():
        z = download(zip_url(rev, r["split"], r["seq"]), tmp / f"{r['seq']}.zip")
        meta = extract_clip(z, r["seq"], int(r["clip_start"]), int(r["clip_length"]), Path(args.out) / r["seq"],
                            Path(args.events) / f"{r['seq']}.hdf5")
        z.unlink()
        print(r["seq"], meta["start"], meta["length"], meta["rgb_size"], flush=True)
    shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    main()
