import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from PIL import Image

from boxprop import params
from boxprop.drift import SIGNALS, auroc, combine, loso_logistic

BOX = ["x1", "y1", "x2", "y2"]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"
STRATEGY_COLORS = {"fixed": "#2a78d6", "random": "#eb6834", "score": "#1baf7a", "oracle": "#eda100"}
STRATEGY_LABELS = {"fixed": "fixed interval", "random": "random", "score": "drift score", "oracle": "oracle (uses reference)"}
REP_LABELS = {"rgb": "RGB", "evframe": "FRED event frames\n(33 ms)", "hist10": "polarity histogram\n10 ms",
              "hist50": "polarity histogram\n50 ms", "ts20": "time surface\nτ = 20 ms"}
GT_COLOR, PRED_COLOR = "#1baf7a", "#eb6834"


def style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
        "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10, "axes.titlesize": 11,
        "axes.titleweight": "regular", "legend.frameon": False,
    })


def load_propagation(prop_dir: Path) -> pd.DataFrame:
    frames = []
    for f in sorted(prop_dir.glob("*.csv")):
        seq, rep = f.stem.split("_", 1)
        frames.append(pd.read_csv(f).assign(seq=seq, rep=rep))
    return pd.concat(frames, ignore_index=True)


def quality_table(df: pd.DataFrame, thr: float) -> pd.DataFrame:
    g = df.groupby(["seq", "rep"])
    q = g.agg(miou=("iou", "mean"), iou50=("iou", lambda v: float((v >= thr).mean())),
              empty=("empty", "mean")).reset_index()
    first_fail = g.apply(lambda d: int(d.loc[d["iou"] < thr, "frame"].min()) if (d["iou"] < thr).any() else -1,
                         include_groups=False)
    return q.merge(first_fail.rename("first_failure").reset_index(), on=["seq", "rep"])


def drift_table(df: pd.DataFrame, reps: list[str], thr: float) -> pd.DataFrame:
    rows = []
    for rep in reps:
        d = df[(df["rep"] == rep) & (df["frame"] > 0)].copy()
        y = (d["iou"] < thr).to_numpy()
        for s in SIGNALS:
            rows.append({"rep": rep, "signal": s, "auroc": auroc(d[s].to_numpy(), y)})
        rows.append({"rep": rep, "signal": "combined (equal weights)", "auroc": auroc(d["score"].to_numpy(), y)})
        d["target"] = y.astype(int)
        pred = loso_logistic(d, SIGNALS, "target", "seq")
        ok = np.isfinite(pred)
        rows.append({"rep": rep, "signal": "logistic, leave one sequence out", "auroc": auroc(pred[ok], y[ok])})
        for r in rows[-len(SIGNALS) - 2:]:
            r |= {"frames": int(len(y)), "failures": int(y.sum())}
    return pd.DataFrame(rows)


def simulation_summary(sim: pd.DataFrame) -> pd.DataFrame:
    per_seq = sim.groupby(["rep", "strategy", "budget", "seq"])[["miou", "iou50", "n_corrections"]].mean().reset_index()
    out = per_seq.groupby(["rep", "strategy", "budget"]).agg(
        miou=("miou", "mean"), miou_sd=("miou", "std"), iou50=("iou50", "mean"), iou50_sd=("iou50", "std"),
        corrections=("n_corrections", "mean"), sequences=("seq", "nunique")).reset_index()
    return out


def fig_corrections(summary: pd.DataFrame, reps: list[str], path: Path) -> None:
    fig, axes = plt.subplots(2, len(reps), figsize=(5.2 * len(reps), 7.2), sharex=True, sharey="row")
    axes = np.atleast_2d(axes).reshape(2, len(reps))
    for j, rep in enumerate(reps):
        s = summary[summary["rep"] == rep]
        base = s[s["strategy"] == "none"]
        for i, metric in enumerate(["miou", "iou50"]):
            ax = axes[i, j]
            for strat, color in STRATEGY_COLORS.items():
                d = s[s["strategy"] == strat].sort_values("budget")
                x = np.r_[0, d["budget"].to_numpy()]
                y = np.r_[base[metric].to_numpy()[:1], d[metric].to_numpy()]
                ax.plot(x, y, color=color, lw=2, marker="o", ms=5, ls="--" if strat == "oracle" else "-",
                        label=STRATEGY_LABELS[strat])
            ax.set_xscale("symlog", linthresh=1)
            ax.set_xticks([0, 1, 2, 4, 8, 16])
            ax.set_xticklabels(["0", "1", "2", "4", "8", "16"])
            if i == 0:
                ax.set_title(REP_LABELS[rep].replace("\n", " "))
            if i == 1:
                ax.set_xlabel("corrections per clip")
            if j == 0:
                ax.set_ylabel("mean IoU" if metric == "miou" else "frames with IoU ≥ 0.5")
    axes[0, -1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def crop(img: np.ndarray, box: np.ndarray, size: int = 160) -> tuple[np.ndarray, int, int]:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    h, w = img.shape[:2]
    x0 = int(np.clip(cx - size / 2, 0, w - size))
    y0 = int(np.clip(cy - size / 2, 0, h - size))
    return img[y0:y0 + size, x0:x0 + size], x0, y0


def draw_box(ax, box: np.ndarray, x0: int, y0: int, color: str, ls: str = "-") -> None:
    if np.isfinite(box).all():
        ax.add_patch(Rectangle((box[0] - x0, box[1] - y0), box[2] - box[0], box[3] - box[1], fill=False,
                               ec=color, lw=1.6, ls=ls))


def frame_path(clips: Path, seq: str, rep: str, i: int) -> Path:
    for d in (clips / seq / rep, clips.parent / "reps" / seq / rep):
        for ext in (".jpg", ".png"):
            if (d / f"{i:05d}{ext}").exists():
                return d / f"{i:05d}{ext}"
    raise FileNotFoundError(f"{seq}/{rep}/{i}")


def fig_representations(df: pd.DataFrame, q: pd.DataFrame, reps: list[str], clips: Path, path: Path) -> None:
    seq = q[q["rep"] == "rgb"].sort_values("miou").iloc[len(q[q["rep"] == "rgb"]) // 2]["seq"]
    d = df[(df["seq"] == seq) & (df["rep"] == "rgb")]
    i = int(d["frame"].iloc[len(d) // 2])
    gt = d[d["frame"] == i][[f"gt_{c}" for c in BOX]].to_numpy()[0]
    fig = plt.figure(figsize=(12.5, 6.4))
    gs = fig.add_gridspec(2, len(reps), height_ratios=[1, 1.15], hspace=0.35)
    for j, rep in enumerate(reps):
        ax = fig.add_subplot(gs[0, j])
        img = np.asarray(Image.open(frame_path(clips, seq, rep, i)).convert("RGB"))
        c, x0, y0 = crop(img, gt)
        ax.imshow(c)
        draw_box(ax, gt, x0, y0, GT_COLOR)
        ax.set_title(REP_LABELS[rep], fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
    ax = fig.add_subplot(gs[1, :])
    rng = np.random.default_rng(0)
    for j, rep in enumerate(reps):
        v = q[q["rep"] == rep]["miou"].to_numpy()
        ax.scatter(j + rng.uniform(-0.12, 0.12, len(v)), v, s=22, color="#2a78d6", alpha=0.75, lw=0)
        ax.hlines(np.median(v), j - 0.28, j + 0.28, color=INK, lw=2)
    ax.set_xticks(range(len(reps)))
    ax.set_xticklabels([REP_LABELS[r] for r in reps], fontsize=9)
    ax.set_ylabel("mean IoU per clip")
    ax.set_ylim(0, 1)
    fig.text(0.01, 0.97, f"sequence {seq}, frame {i} of the clip; reference box in green. "
             "Below: one dot per clip, bar at the median.", color=INK2, fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def pick_drift_example(df: pd.DataFrame, sim: pd.DataFrame, rep: str) -> tuple[str, int] | None:
    cands = []
    for seq, d in df[df["rep"] == rep].groupby("seq"):
        s = sim[(sim["seq"] == seq) & (sim["rep"] == rep) & (sim["strategy"] == "score") & (sim["budget"] == 1)]
        if s.empty or (d["iou"] >= 0.5).all():
            continue
        k = int(str(s["keyframes"].iloc[0]).split()[0]) if str(s["keyframes"].iloc[0]).strip() else None
        if k is None:
            continue
        gain = s["miou"].iloc[0] - d["iou"].mean()
        cands.append((gain, seq, k))
    if not cands:
        return None
    _, seq, k = max(cands)
    return seq, k


def fig_drift(df: pd.DataFrame, sim: pd.DataFrame, rep: str, clips: Path, path: Path) -> dict | None:
    pick = pick_drift_example(df, sim, rep)
    if pick is None:
        return None
    seq, k = pick
    d = df[(df["seq"] == seq) & (df["rep"] == rep)].sort_values("frame")
    fail = d[(d["iou"] < 0.5)]["frame"]
    f_fail = int(fail.min()) if len(fail) else k
    fig = plt.figure(figsize=(12.5, 6.6))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.1, 1], hspace=0.4)
    ax = fig.add_subplot(gs[0, :])
    ax.plot(d["frame"], d["iou"], color="#2a78d6", lw=2, label="IoU of the propagated box")
    ax.plot(d["frame"], d["score"], color="#eb6834", lw=1.5, label="drift score")
    ax.axvline(k, color=INK2, lw=1, ls="--")
    ax.text(k + 3, 0.96, "correction chosen by the score", color=INK2, fontsize=9, va="top")
    ax.plot(d["frame"], d["iou_score1"], color="#1baf7a", lw=2, label="IoU with that correction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0, d["frame"].max())
    ax.set_xlabel("frame in clip")
    ax.legend(loc="lower left", ncols=3)
    shown = [max(0, f_fail - 30), f_fail, min(k + 15, int(d["frame"].max())), min(k + 60, int(d["frame"].max()))]
    titles = ["before the drift", "first frame with IoU < 0.5", "after the correction", "later"]
    for j, (i, t) in enumerate(zip(shown, titles)):
        a = fig.add_subplot(gs[1, j])
        img = np.asarray(Image.open(frame_path(clips, seq, rep, i)).convert("RGB"))
        row = d[d["frame"] == i].iloc[0]
        gt = row[[f"gt_{c}" for c in BOX]].to_numpy(dtype=float)
        pred = row[BOX].to_numpy(dtype=float)
        if j >= 2:
            pred = row[[f"score1_{c}" for c in BOX]].to_numpy(dtype=float)
        c, x0, y0 = crop(img, gt, 200)
        a.imshow(c)
        draw_box(a, gt, x0, y0, GT_COLOR)
        draw_box(a, pred, x0, y0, PRED_COLOR, "--")
        a.set_title(f"{t}\nframe {i}", fontsize=9)
        a.set_xticks([])
        a.set_yticks([])
        a.grid(False)
    fig.text(0.01, 0.97, f"sequence {seq}, {REP_LABELS[rep].replace(chr(10), ' ')}. Crops: reference box in green, "
             "SAM 2.1 box dashed orange.", color=INK2, fontsize=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return {"seq": seq, "rep": rep, "keyframe": k, "first_failure": f_fail}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--clips", default="data/clips")
    ap.add_argument("--figures", default="figures")
    args = ap.parse_args()
    p = params.load()
    thr = p["simulate"]["iou_threshold"]
    res, figs = Path(args.results), Path(args.figures)
    figs.mkdir(parents=True, exist_ok=True)
    style()
    df = load_propagation(res / "propagation")
    reps = [r for r in p["propagate"]["representations"] if r in set(df["rep"])]
    q = quality_table(df, thr)
    q.to_csv(res / "quality_per_clip.csv", index=False, float_format="%.4f")
    qs = q.groupby("rep").agg(miou=("miou", "mean"), miou_median=("miou", "median"), iou50=("iou50", "mean"),
                              clips_failing=("first_failure", lambda v: int((v >= 0).sum())),
                              clips=("seq", "nunique")).reindex(reps).reset_index()
    qs.to_csv(res / "quality_by_representation.csv", index=False, float_format="%.4f")
    dt = drift_table(df, p["simulate"]["representations"], thr)
    dt.to_csv(res / "drift_auroc.csv", index=False, float_format="%.4f")
    sim = pd.read_csv(res / "simulation.csv", dtype={"seq": str})
    ss = simulation_summary(sim)
    ss.to_csv(res / "simulation_summary.csv", index=False, float_format="%.4f")
    fig_corrections(ss, p["simulate"]["representations"], figs / "corrections.png")
    fig_representations(df, q, reps, Path(args.clips), figs / "representations.png")
    example = fig_drift(df, sim, p["simulate"]["representations"][0], Path(args.clips), figs / "drift_correction.png")
    metrics = {
        "quality": {r["rep"]: {"miou": round(r["miou"], 4), "iou50": round(r["iou50"], 4)} for _, r in qs.iterrows()},
        "drift_auroc": {f"{r['rep']}/{r['signal']}": round(r["auroc"], 4) for _, r in dt.iterrows()},
        "simulation": {f"{r['rep']}/{r['strategy']}/{int(r['budget'])}": round(r["miou"], 4) for _, r in ss.iterrows()},
        "drift_example": example,
    }
    (res / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(qs.to_string())
    print(dt.to_string())
    print(ss.to_string())


if __name__ == "__main__":
    main()
