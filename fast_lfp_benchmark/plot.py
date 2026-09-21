"""
Rate–distortion figure: each panel is one axis of the benchmark against rate
(stored floats per second, log scale). PCA is a curve over ranks; lfpack and
channel binning are points. Mean ± SEM over held-out probes.

    python -m fast_lfp_benchmark.plot [--out-dir DIR]
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = os.path.expanduser("~/Downloads/lfp-brain-state/fast_lfp_results")

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
COLOR = {"pca": "#2a78d6", "lfpack": "#eb6834", "bin": "#1baf7a"}   # validated, fixed order
MARKER = {"pca": "o", "lfpack": "s", "bin": "D"}                     # shape is the secondary encoding

PANELS = [
    ("r2", "Broadband R²", "reconstruction"),
    ("coh_delta", "Coherence, delta 1–4 Hz", "frequency"),
    ("coh_beta", "Coherence, beta 15–30 Hz", "frequency"),
    ("coh_gamma", "Coherence, gamma 30–90 Hz", "frequency"),
    ("env_corr_gamma", "Gamma envelope correlation", "temporal"),
    ("beh_retained", "Behavioural R² retained", "behaviour"),
]


def family(name):
    return "pca" if name.startswith("pca") else "lfpack" if name.startswith("lfpack") else "bin" if name.startswith("bin") else None


def main(out_dir=OUT_DIR):
    out = Path(out_dir)
    df = pd.read_csv(out / "probe_metrics.csv")
    df = df[df.candidate != "raw"]
    numeric = df.select_dtypes("number").columns
    stats = df.groupby("candidate")[list(numeric)].agg(["mean", "sem"])
    n_probes = df.pid.nunique()
    rates = stats[("floats_per_s", "mean")]
    xlim = (rates.min() / 1.8, rates.max() * 1.8)

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK_2,
                         "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2})
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.6), facecolor=SURFACE)

    for ax, (metric, title, axis_name) in zip(axes.flat, PANELS):
        ax.set_facecolor(SURFACE)
        ax.set_xscale("log")
        ax.grid(True, which="major", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.set_title(title, loc="left", color=INK, fontsize=10, fontweight="bold")

        if metric not in stats.columns.get_level_values(0) or stats[(metric, "mean")].isna().all():
            ax.text(0.5, 0.5, "raw signal does not predict\nbehaviour in these sessions",
                    transform=ax.transAxes, ha="center", va="center", color=INK_2)
            ax.set_xticks([]); ax.set_yticks([])
            continue

        for fam in ("pca", "lfpack", "bin"):
            names = [c for c in stats.index if family(c) == fam]
            if not names:
                continue
            sub = stats.loc[names].sort_values(("floats_per_s", "mean"))
            x = sub[("floats_per_s", "mean")].to_numpy()
            y = sub[(metric, "mean")].to_numpy()
            e = sub[(metric, "sem")].fillna(0).to_numpy()
            ax.errorbar(x, y, yerr=e, color=COLOR[fam], marker=MARKER[fam], markersize=6,
                        markeredgecolor=SURFACE, markeredgewidth=1.2, linewidth=1.6,
                        elinewidth=0.9, capsize=0, zorder=3,
                        label={"pca": "Spatial PCA (rank sweep)", "lfpack": "lfpack", "bin": "Channel bin ×16"}[fam])

            # selective direct labels, in ink, never on every point
            for name, xi, yi in zip(sub.index, x, y):
                label = None
                if fam == "pca" and name in ("pca1", "pca8", "pca64"):
                    label = "r=" + name[3:]
                elif fam == "lfpack":
                    label = name.split("_")[1]
                elif fam == "bin":
                    label = "bin16"
                if label and np.isfinite(yi):
                    # PCA labels sit above-left, lfpack left, bin16 below-right, so the
                    # three families never label the same spot.
                    dx, dy, ha, va = {"pca": (-6, 5, "right", "bottom"),
                                      "lfpack": (-8, 0, "right", "center"),
                                      "bin": (7, -7, "left", "top")}[fam]
                    ax.annotate(label, (xi, yi), xytext=(dx, dy), textcoords="offset points",
                                fontsize=7.5, color=INK_2, ha=ha, va=va)

        ax.set_xlim(*xlim)
        ax.set_xlabel("stored floats per second (log)")
        if metric in ("r2", "beh_retained"):
            ax.axhline(0, color=INK_2, linewidth=0.8, zorder=2)

    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9,
               labelcolor=INK, bbox_to_anchor=(0.985, 0.995))
    fig.text(0.012, 0.975, "Fast LFP representations: rate vs. fidelity", fontsize=13,
             fontweight="bold", color=INK, va="top")
    fig.text(0.012, 0.935, f"Mean ± SEM over {n_probes} held-out probes · 384 ch at 500 Hz = 192,000 floats/s uncompressed",
             fontsize=8.5, color=INK_2, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    path = out / "rate_distortion.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    print("figure:", path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=OUT_DIR)
    main(ap.parse_args().out_dir)
