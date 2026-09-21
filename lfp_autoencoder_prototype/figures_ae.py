"""
Figures for the autoencoder prototype.

    /opt/miniconda3/envs/lfp-brain-state/bin/python -m lfp_autoencoder_prototype.figures_ae

fig_ae_compare.png   48-number representations side by side: behaviour r, waveform R², amplitude depth pattern R²
fig_ae_training.png  training and validation loss per fold
fig_ae_latents.png   one held-out insertion: the per-depth latent over depth × time, with whisker motion
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .cache_bank import AE_CACHE
from .train import AE_RESULTS

FIG = AE_RESULTS / "figures"
INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#ffffff", "#e6e5e1"
BLUE, ORANGE, AQUA, GRAY, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#8a8a86", "#4a3aa7"
LABEL = {"waveform_pca_48": "PCA 48", "spatial_avg_48": "Spatial average 48", "waveform+gamma_48": "Waveform + gamma per depth (no PCA)",
         "group_pca_2x24": "Group PCA 2×24 (per insertion)", "group_pca_pooled_48": "Group PCA 2×24 (pooled)",
         "group_wave+amp1": "Waveform + 1 amp PC per depth", "ae_global_48": "AE, global latent (48)", "ae_depth_48": "AE, 2 latents per depth (48)",
         "waveform_pca_16": "PCA 16", "wave12+bank4": "12 wave + 4 bank PCs (16)"}
COLOR = {"waveform_pca_48": BLUE, "spatial_avg_48": GRAY, "waveform+gamma_48": AQUA, "group_pca_2x24": ORANGE, "group_pca_pooled_48": ORANGE,
         "group_wave+amp1": ORANGE, "ae_global_48": VIOLET, "ae_depth_48": VIOLET, "waveform_pca_16": BLUE, "wave12+bank4": BLUE}


def style(ax, grid="x"):
    ax.set_facecolor(SURFACE)
    ax.grid(axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=10)


def fig_compare():
    summ = pd.read_csv(AE_RESULTS / "summary_ae_compare.csv", index_col=0)
    order = [m for m in summ.index if m not in ("waveform_pca_16", "wave12+bank4")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.5 * len(order) + 1.5), facecolor=SURFACE, sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (col, title, xmax) in zip(axes, [("beh_r_mean", "Behaviour r (mean of 5)", 0.8), ("r2_wave", "Waveform R²", 1.1),
                                              ("r2_amp_spatial", "Amplitude depth pattern R²", 0.7)]):
        style(ax)
        v = summ.loc[order, col]
        ax.barh(y, v.clip(lower=0), height=0.62, color=[COLOR[m] for m in order])
        for yi, val in zip(y, v):
            ax.text(max(val, 0) + xmax * 0.015, yi, f"{val:.2f}", va="center", fontsize=9.5, color=INK)
        ax.set_xlim(0, xmax)
        ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", color=INK)
    axes[0].set_yticks(y, [LABEL.get(m, m) for m in order], fontsize=10)
    n = int(summ["n"].min())
    fig.text(0.01, 0.01, f"All 48 numbers at 25 Hz. Median over {n} insertions. Autoencoders and pooled group PCA were trained on other sessions "
             "(5 session folds);\nper-insertion methods fit on each insertion's training time. Same held-out readout for all.", fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.09, 1, 1))
    fig.savefig(FIG / "fig_ae_compare.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_training():
    logs = pd.concat([pd.read_csv(f) for f in sorted(AE_RESULTS.glob("training_log_fold*.csv"))], ignore_index=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), facecolor=SURFACE, sharey=True)
    for ax, model in zip(axes, ["ae_depth_48", "ae_global_48"]):
        style(ax, grid="y")
        for fold, g in logs[logs.model == model].groupby("fold"):
            ax.plot(g.epoch, g.train_loss, color=GRAY, linewidth=1, alpha=0.7)
            ax.plot(g.epoch, g.val_loss, color=VIOLET, linewidth=1.6)
        ax.set_title(LABEL[model], loc="left", fontsize=11.5, fontweight="bold", color=INK)
        ax.set_xlabel("epoch", color=INK_2)
    axes[0].set_ylabel("view-weighted MSE (z-scored features)", color=INK_2)
    axes[1].legend(handles=[plt.Line2D([], [], color=GRAY, label="train"), plt.Line2D([], [], color=VIOLET, label="validation sessions")],
                   frameon=False, fontsize=9.5)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ae_training.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_latents():
    pp = pd.read_csv(AE_RESULTS / "per_probe_ae_compare.csv")
    g = pp[pp.method == "ae_depth_48"].sort_values("beh_r_mean")
    pid = g.pid.iloc[len(g) // 2]                                  # median insertion, not a cherry-pick
    with np.load(AE_RESULTS / "latents" / f"{pid}.npz") as z:
        Z = z["ae_depth_48"].reshape(2, 24, -1)
    with np.load(AE_CACHE / f"{pid}.npz") as z:
        B, names, regions = z["behavior"], [str(x) for x in z["behavior_names"]], [str(x) for x in z["regions"]]
    t0, t1 = 500, 1250                                             # 30 s shown
    t = np.arange(t0, t1) * 0.04
    fig, axes = plt.subplots(3, 1, figsize=(12, 6.2), facecolor=SURFACE, sharex=True, gridspec_kw={"height_ratios": [2, 2, 1]})
    for k in range(2):
        ax = axes[k]
        zz = Z[k, :, t0:t1]
        lim = np.percentile(np.abs(zz), 98)
        ax.imshow(zz, aspect="auto", cmap="RdBu_r", vmin=-lim, vmax=lim, extent=(t[0], t[-1], 23.5, -0.5))
        ax.set_ylabel(f"latent {k + 1}\ndepth group", color=INK_2)
        ax.set_yticks([0, 12, 23], [f"{regions[0]} (tip)", regions[12], f"{regions[23]} (top)"], fontsize=8.5)
    j = names.index("whisker_me_left") if "whisker_me_left" in names else 0
    ax = axes[2]
    style(ax, grid="y")
    ax.plot(t, B[t0:t1, j], color=INK, linewidth=1)
    ax.set_ylabel("whisker ME", color=INK_2)
    ax.set_xlabel("time in window (s)", color=INK_2)
    axes[0].set_title(f"Held-out insertion {pid[:8]}: the 48 latents are 2 maps over depth × time", loc="left", fontsize=12, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ae_latents.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    FIG.mkdir(parents=True, exist_ok=True)
    fig_compare()
    fig_training()
    fig_latents()
    print("figures:", sorted(p.name for p in FIG.glob("*.png")))


if __name__ == "__main__":
    main()
