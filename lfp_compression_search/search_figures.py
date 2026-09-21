"""
Figures for the method search, from run_search.py outputs only.

    /opt/miniconda3/envs/lfp-brain-state/bin/python -m lfp_compression_search.search_figures

fig_search_ranking.png   search set: per-probe behaviour-r gain over PCA 16 for every
                         method, hollow if the method breaks the waveform constraint
fig_confirmation.png     confirmation set: the pre-selected winner and wave12+amp4 vs PCA 16
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import pandas as pd

from lfp_compression_pilot import pilot_config as C

ROOT = Path(C.RESULTS_DIR) / "search"
FIG = ROOT / "figures"
SURFACE, INK, INK_2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1"
ORANGE, GRAY, BLUE = "#eb6834", "#8a8a86", "#2a78d6"
LABEL = {
    "wave12+amp4_smooth": "12 wave + 4 amp, 200 ms RMS",
    "wave12+amp4_multiband": "12 wave + 4 amp, 4 bands",
    "wave12+amp4": "12 wave + 4 amp, 40 ms (previous)",
    "joint_pca_16": "Joint PCA (wave + 4 amp bands)",
    "bandpower_pca_16": "Band-power PCA (no waveform)",
    "autoencoder_16": "MLP autoencoder",
    "delay_pca_16": "Delay-embedded PCA",
    "spatiotemporal_pca_16": "Spatiotemporal PCA (40 ms windows)",
}


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=10)


def ranking():
    paired = pd.read_csv(ROOT / "search" / "paired.csv")
    summ = pd.read_csv(ROOT / "search" / "method_summary.csv").set_index("method")
    winner = json.load(open(ROOT / "search" / "winner.json"))["winner"]
    order = summ.sort_values("median_d_beh_r").index.tolist()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.0), facecolor=SURFACE, sharey=True,
                                  gridspec_kw={"width_ratios": [2.4, 1.2]})
    for a in (ax, ax2):
        style(a)
        a.axvline(0, color=INK, linewidth=1)
    rng = np.random.default_rng(0)
    for i, m in enumerate(order):
        g = paired[paired.method == m]
        ok = summ.loc[m, "median_d_wave_r2"] >= -0.02
        col = ORANGE if m.startswith("wave12") else GRAY
        ax.scatter(g.beh_r_mean, i + rng.uniform(-0.15, 0.15, len(g)), s=26,
                   facecolor=col if ok else "none", edgecolor=col, linewidth=1.2, zorder=3)
        med = summ.loc[m, "median_d_beh_r"]
        ax.plot([med, med], [i - 0.3, i + 0.3], color=INK, linewidth=2.2, zorder=4)
        wmed = summ.loc[m, "median_d_wave_r2"]
        ax2.barh(i, max(wmed, -0.3), height=0.5, color=col if ok else "none", edgecolor=col, linewidth=1.2)
        ax2.text(max(wmed, -0.3) - 0.008, i, f"{wmed:+.3f}", ha="right", va="center", fontsize=9, color=INK)
    ax2.axvline(-0.02, color=INK_2, linestyle="--", linewidth=1)
    ax2.text(-0.017, -0.55, "limit −0.02", ha="left", va="center", fontsize=9, color=INK_2)
    ax2.set_ylim(-0.8, len(order) - 0.5)
    lo, hi = paired.beh_r_mean.min(), paired.beh_r_mean.max()
    ax.set_xlim(min(lo, 0) - 0.02, hi + 0.02)
    ax2.set_xlim(-0.42, 0.01)
    ax.set_yticks(range(len(order)), [LABEL[m] + ("  ★" if m == winner else "") for m in order])
    ax.set_xlabel("Behaviour r: method − PCA 16 (each dot = probe)")
    ax2.set_xlabel("Waveform R²: median method − PCA 16")
    ax.set_title("Behaviour gain", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax2.set_title("Waveform cost (bars cut at −0.3)", loc="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.01, "Search set: 12 probes, 6 sessions. Filled = within the waveform limit; hollow = breaks it. "
             "Black tick = median. ★ = winner by the pre-declared rule.", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "fig_search_ranking.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def confirmation():
    f = ROOT / "confirm" / "paired.csv"
    if not f.exists():
        return
    paired = pd.read_csv(f)
    verdict = json.load(open(ROOT / "confirm" / "verdict.json"))
    sess = paired.groupby(["method", "eid"])[["beh_r_mean", "r2_wave", "r2_env_gamma"]].mean().reset_index()
    methods = [verdict["winner"]] + [m for m in sess.method.unique() if m != verdict["winner"]]
    panels = [("beh_r_mean", "Δ behaviour r"), ("r2_wave", "Δ waveform R²"), ("r2_env_gamma", "Δ gamma amplitude R²")]
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), facecolor=SURFACE, sharey=True)
    for ax, (col, title) in zip(axes, panels):
        style(ax)
        ax.axvline(0, color=INK, linewidth=1)
        for i, m in enumerate(methods):
            g = paired[paired.method == m]
            s = sess[sess.method == m]
            yy = len(methods) - 1 - i
            ax.scatter(g[col], np.full(len(g), yy) + 0.12, s=20, color=ORANGE, alpha=0.45, edgecolor="none")
            ax.scatter(s[col], np.full(len(s), yy) - 0.12, s=46, marker="D", color=ORANGE, edgecolor=INK, linewidth=0.7)
            ax.text(0.5, yy + 0.32, f"median {g[col].median():+.3f} · {int((s[col] > 0).sum())}/{len(s)} sessions higher",
                    transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData),
                    ha="center", fontsize=8.5, color=INK)
        ax.set_title(title, loc="left", fontsize=11, color=INK)
        ax.set_ylim(-0.5, len(methods) - 0.35)
    axes[0].set_yticks(range(len(methods)), [LABEL[m] for m in reversed(methods)])
    fig.text(0.01, 0.01, f"Confirmation set: {int(sess.eid.nunique())} new sessions. Small dots: probes; diamonds: session means. "
             f"Verdict: {'CONFIRMED' if verdict['confirmed'] else 'NOT confirmed'} ({verdict['rule']}).", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(FIG / "fig_confirmation.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelcolor": INK_2})
    FIG.mkdir(parents=True, exist_ok=True)
    ranking()
    confirmation()
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
