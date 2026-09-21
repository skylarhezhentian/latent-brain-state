"""
Figures for the morning deck, drawn only from the CSVs the runners wrote.

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    /opt/miniconda3/envs/lfp-brain-state/bin/python -m lfp_compression_pilot.make_figures

Writes PNGs to RESULTS_DIR/figures/. Each figure makes one point:
  fig1_tradeoff.png       spending some of the 16 dims on amplitude: what it costs, what it buys
  fig2_scorecard.png      every axis, pilot probe, fold-level points
  fig3_lfpack.png         what lfpack keeps, by band, and what representations built on it keep
  fig4_replication.png    paired per-probe differences on the other probes
"""
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import pandas as pd

from . import pilot_config as C

RES = Path(C.RESULTS_DIR)
PILOT = RES / f"pilot_{C.PILOT_PID[:8]}"
FIG = RES / "figures"

SURFACE, INK, INK_2, GRID = "#ffffff", "#0b0b0b", "#52514e", "#e6e5e1"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"   # validated order


def color_of(method):
    if method.startswith("waveform_pca"):
        return BLUE
    if method.startswith("wave") or method.startswith("amplitude"):
        return ORANGE
    if method.startswith("spatial"):
        return VIOLET
    return INK_2


LABEL = {
    "waveform_pca_16": "PCA 16 (current)",
    "waveform_pca_20": "PCA 20",
    "waveform_pca_8": "PCA 8",
    "spatial_avg_16": "Spatial avg 16",
    "wave12+amp4": "C2: 12 wave + 4 amp",
    "wave16+amp4": "C1: 16 wave + 4 amp",
    "wave14+amp2": "14 wave + 2 amp",
    "wave8+amp8": "8 wave + 8 amp",
    "wave4+amp12": "4 wave + 12 amp",
    "amplitude_pca_cv16": "inner-CV split (run 1 spec)",
    "reference_1152": "Reference (1152 feat.)",
}


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=10)


def base_rc():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11, "axes.labelcolor": INK_2})


OFFSETS_LEFT = {"waveform_pca_16": (-10, 10, "right"), "wave14+amp2": (-10, 0, "right"),
                "wave12+amp4": (-10, 6, "right"), "wave8+amp8": (10, 10, "left"),
                "wave4+amp12": (12, -14, "left"), "waveform_pca_20": (10, -12, "left"), "wave16+amp4": (10, 0, "left")}
# Right panel: PCA 16 and 14+2 sit on top of each other, so they share one label.
OFFSETS_RIGHT = {"waveform_pca_16": (10, -14, "left"), "wave14+amp2": None,
                 "wave12+amp4": (0, -14, "center"), "wave8+amp8": (-10, 0, "right"),
                 "wave4+amp12": (10, 6, "left"), "waveform_pca_20": (10, 10, "left"), "wave16+amp4": (10, 0, "left")}
RIGHT_TEXT = {"waveform_pca_16": "PCA 16 (current) ≈ 14 wave + 2 amp"}


def fig1_tradeoff(df, plain=False):
    """plain=True writes fig1_tradeoff_plain.png with no C1/C2 names (for the short deck)."""
    global LABEL
    saved = LABEL
    if plain:
        LABEL = {k: v.split(": ", 1)[-1] for k, v in LABEL.items()}
    try:
        _fig1_tradeoff(df, "fig1_tradeoff_plain.png" if plain else "fig1_tradeoff.png")
    finally:
        LABEL = saved


def _fig1_tradeoff(df, filename):
    raw = df[df.source == "raw"].groupby("method").mean(numeric_only=True)
    order16 = ["waveform_pca_16", "wave14+amp2", "wave12+amp4", "wave8+amp8", "wave4+amp12"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), facecolor=SURFACE)
    for ax, ycol, ylab in ((axes[0], "r2_env_gamma", "Gamma amplitude R² (held-out)"),
                           (axes[1], "beh_r_mean", "Behaviour decoding r (held-out)")):
        style(ax)
        x = raw.loc[order16, "r2_wave"].to_numpy()
        y = raw.loc[order16, ycol].to_numpy()
        ax.plot(x, y, color=ORANGE, linewidth=2, zorder=2)
        offsets = OFFSETS_LEFT if ycol == "r2_env_gamma" else OFFSETS_RIGHT
        for m, xi, yi in zip(order16, x, y):
            c = BLUE if m == "waveform_pca_16" else ORANGE
            ax.scatter([xi], [yi], s=70, color=c, edgecolor=SURFACE, linewidth=1.5, zorder=3,
                       marker="o" if m == "waveform_pca_16" else "s")
            if offsets[m] is None:
                continue
            dx, dy, ha = offsets[m]
            text = RIGHT_TEXT.get(m, LABEL[m]) if offsets is OFFSETS_RIGHT else LABEL[m]
            ax.annotate(text, (xi, yi), xytext=(dx, dy), textcoords="offset points",
                        ha=ha, va="center", fontsize=9.5, color=INK)
        for m, mk in (("waveform_pca_20", "o"), ("wave16+amp4", "D")):
            ax.scatter([raw.loc[m, "r2_wave"]], [raw.loc[m, ycol]], s=70, marker=mk, color=color_of(m),
                       edgecolor=INK, linewidth=0.8, zorder=3)
            dx, dy, ha = offsets[m]
            ax.annotate(LABEL[m] + " (20 dims)", (raw.loc[m, "r2_wave"], raw.loc[m, ycol]), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va="center", fontsize=9.5, color=INK)
        ax.set_xlim(0.815, 1.04)
        ax.set_xlabel("Waveform R² (25 Hz, held-out)")
        ax.set_ylabel(ylab)
    axes[0].set_title("What amplitude dimensions buy", loc="left", fontsize=13, color=INK, fontweight="bold")
    axes[1].set_title("…and what they do to behaviour", loc="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, 0.005, "Pilot probe c4b5a9fa (NYU-37), raw source, mean of 4 held-out 25 s blocks. "
             "Orange line: 16 dims split between waveform and amplitude. Right panel: note the narrow y range.", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / filename, dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig2_scorecard(df):
    raw = df[df.source == "raw"]
    methods = ["waveform_pca_16", "waveform_pca_20", "spatial_avg_16", "wave12+amp4", "wave16+amp4"]
    panels = [("r2_wave", "Waveform R²"), ("r2_env_beta", "Beta amplitude R²"), ("r2_env_gamma", "Gamma amplitude R²"),
              ("coh_alpha", "Alpha coherence"), ("beh_r_mean", "Behaviour r")]
    fig, axes = plt.subplots(1, len(panels), figsize=(14, 4.2), facecolor=SURFACE, sharey=True)
    ypos = {m: i for i, m in enumerate(reversed(methods))}
    for ax, (col, title) in zip(axes, panels):
        style(ax)
        ax.grid(axis="y", visible=False)
        for m in methods:
            v = raw[raw.method == m][col].to_numpy()
            ax.scatter(v, np.full(len(v), ypos[m]), s=18, color=color_of(m), alpha=0.55, zorder=2, edgecolor="none")
            ax.scatter([v.mean()], [ypos[m]], s=90, color=color_of(m), edgecolor=INK, linewidth=0.8, zorder=3,
                       marker="D" if m.startswith("wave1") and "amp" in m else "o")
            ax.annotate(f"{abs(v.mean()) if abs(v.mean()) < 0.005 else v.mean():.2f}", (v.mean(), ypos[m]), xytext=(0, 9), textcoords="offset points",
                        ha="center", fontsize=8.5, color=INK)
        if col == "beh_r_mean":
            sh = raw[raw.method.isin(methods)]["shuf_r_mean"].mean()
            ax.axvline(sh, color=INK_2, linestyle="--", linewidth=1)
            ax.annotate("shuffled\ncontrol", (sh, len(methods) - 0.6), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=INK_2, va="top")
        ax.set_title(title, fontsize=11, color=INK, loc="left")
        ax.set_ylim(-0.7, len(methods) - 0.3)
    axes[0].set_yticks(list(ypos.values()), [LABEL[m] for m in ypos])
    fig.text(0.01, 0.01, "Small dots: 4 held-out folds. Large markers: mean. Pilot probe, raw source.",
             fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(FIG / "fig2_scorecard.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig3_lfpack(df, codec):
    bands = ["delta", "theta", "alpha", "beta", "gamma"]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6), facecolor=SURFACE)
    ax = axes[0]
    style(ax)
    names = {"lfpack_default": "lfpack default (no Cadzow)", "lfpack_cadzow_default": "Cadzow + lfpack default",
             "lfpack_cadzow_aggressive": "Cadzow + lfpack aggressive"}
    markers = {"lfpack_default": "o", "lfpack_cadzow_default": "s", "lfpack_cadzow_aggressive": "^"}
    for _, r in codec.iterrows():
        y = [r[f"coh250_{b}"] for b in bands]
        ax.plot(range(len(bands)), y, color=AQUA, linewidth=1.8, marker=markers[r.source], markersize=7,
                markeredgecolor=INK, markeredgewidth=0.6)
        ax.annotate(f"{names[r.source]}  ({r.storage_bytes_per_s:,.0f} B/s)", (len(bands) - 1, y[-1]),
                    xytext=(6, 0), textcoords="offset points", fontsize=9, color=INK, va="center")
    ax.set_xticks(range(len(bands)), bands)
    ax.set_ylim(0, 1.02)
    ax.set_xlim(-0.2, len(bands) + 2.4)
    ax.set_ylabel("Coherence with raw (250 Hz)")
    ax.set_title("lfpack decoded vs raw, by band", loc="left", fontsize=13, color=INK, fontweight="bold")

    ax = axes[1]
    style(ax)
    srcs = ["raw", "lfpack_default", "lfpack_cadzow_default", "lfpack_cadzow_aggressive"]
    src_label = {"raw": "raw", "lfpack_default": "lfpack, no Cadzow", "lfpack_cadzow_default": "Cadzow + lfpack default",
                 "lfpack_cadzow_aggressive": "Cadzow + lfpack aggressive"}
    metrics = [("r2_wave", "Waveform R²"), ("r2_env_gamma", "Gamma amp. R²"), ("beh_r_mean", "Behaviour r")]
    width = 0.2
    for i, (col, _) in enumerate(metrics):
        for j, s in enumerate(srcs):
            v = df[(df.source == s) & (df.method == "wave16+amp4")][col].mean()
            ax.bar(i + (j - 1.5) * width, max(v, 0), width=width * 0.9, color=BLUE if s == "raw" else AQUA,
                   alpha=1.0 if s == "raw" else 0.45 + 0.18 * j, edgecolor=SURFACE)
            ax.annotate(f"{v:.2f}", (i + (j - 1.5) * width, max(v, 0)), xytext=(0, 2), textcoords="offset points",
                        ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(range(len(metrics)), [m[1] for m in metrics])
    ax.set_ylim(0, 1.05)
    handles = [plt.Rectangle((0, 0), 1, 1, color=BLUE if s == "raw" else AQUA, alpha=1.0 if s == "raw" else 0.45 + 0.18 * j)
               for j, s in enumerate(srcs)]
    ax.legend(handles, [src_label[s] for s in srcs], frameon=False, fontsize=9, loc="upper right")
    ax.set_title("C1 built from raw vs lfpack-decoded data", loc="left", fontsize=13, color=INK, fontweight="bold")
    fig.text(0.01, 0.005, "Pilot probe, 100 s. Left: whole recording. Right: mean of 4 held-out folds; targets always from raw.",
             fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(FIG / "fig3_lfpack.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig4_replication(paired, summary):
    metrics = [("r2_wave", "Waveform R²"), ("r2_env_gamma", "Gamma amplitude R²"), ("beh_r_mean", "Behaviour r")]
    comps = ["C2 vs current (16 dims)", "C1 vs PCA20 (20 dims)"]
    fig, axes = plt.subplots(1, len(metrics), figsize=(13, 4.2), facecolor=SURFACE, sharey=True)
    rng = np.random.default_rng(0)
    for ax, (col, title) in zip(axes, metrics):
        style(ax)
        ax.grid(axis="y", visible=False)
        ax.axvline(0, color=INK, linewidth=1)
        for i, comp in enumerate(comps):
            g = paired[paired.comparison == comp]
            y = np.full(len(g), len(comps) - 1 - i) + rng.uniform(-0.12, 0.12, len(g))
            ax.scatter(g[col], y, s=30, color=ORANGE, edgecolor=INK, linewidth=0.5, zorder=3)
            s = summary[(summary.comparison == comp) & (summary.metric == col)].iloc[0]
            ax.text(0.5, len(comps) - 1 - i + 0.3, f"median {s.median_diff:+.3f}  ·  {s.n_improved}/{s.n_probes} probes higher",
                    transform=matplotlib.transforms.blended_transform_factory(ax.transAxes, ax.transData),
                    ha="center", va="bottom", fontsize=9, color=INK)
        ax.set_title(f"Δ {title}", fontsize=11, color=INK, loc="left")
        ax.set_ylim(-0.4, len(comps) - 1 + 0.6)
    axes[0].set_yticks(range(len(comps)), list(reversed(comps)))
    n_mice = int(summary.n_mice.max())
    fig.text(0.01, 0.01, f"Each dot: one probe (candidate minus comparator, mean of 4 held-out folds). "
             f"{int(summary.n_probes.max())} probes, {n_mice} mice; pilot probe excluded.", fontsize=9, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(FIG / "fig4_replication.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    base_rc()
    FIG.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(PILOT / "fold_metrics.csv")
    fig1_tradeoff(df)
    fig1_tradeoff(df, plain=True)
    fig2_scorecard(df)
    if (PILOT / "codec_fidelity.csv").exists():
        fig3_lfpack(df, pd.read_csv(PILOT / "codec_fidelity.csv"))
    rep = RES / "replication"
    if (rep / "paired.csv").exists():
        fig4_replication(pd.read_csv(rep / "paired.csv"), pd.read_csv(rep / "paired_summary.csv"))
    print("figures in", FIG, sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
