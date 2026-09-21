"""
Decoding r for every behaviour under every method, as heatmaps.

    /opt/miniconda3/envs/lfp-brain-state/bin/python -m lfp_compression_search.behaviour_r_heatmap

Panels (each from its own run, never mixed):
  A  search set: the 9 representations (12 probes, 6 mice)
  B  confirmation set: the pre-selected winner vs PCA 16 (12 probes, 6 new mice)
  C  24-channel spatial-averaging baseline vs PCA 24 / PCA 16 (all 24 probes), when available
Cell = median over probes of held-out Pearson r between decoded and measured behaviour.
All medians are positive here, so r and |r| coincide.
Writes RESULTS_DIR/search/figures/fig_behaviour_r_heatmap.png and behaviour_r_table.csv.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from lfp_compression_pilot import pilot_config as C

ROOT = Path(C.RESULTS_DIR) / "search"
BEH = [("beh_r_wheel_speed", "Wheel\nspeed"), ("beh_r_whisker_me_left", "Whisker\n(left cam)"),
       ("beh_r_whisker_me_right", "Whisker\n(right cam)"), ("beh_r_body_me", "Body\nmotion"), ("beh_r_mean", "Mean")]
LABEL = {
    "wave12+amp4_smooth": "12 wave + 4 amp (200 ms)", "wave12+amp4_multiband": "12 wave + 4 amp (4 bands)",
    "wave12+amp4": "12 wave + 4 amp (40 ms)", "joint_pca_16": "Joint PCA", "bandpower_pca_16": "Band-power PCA",
    "autoencoder_16": "MLP autoencoder", "waveform_pca_16": "PCA 16 (current)", "waveform_pca_24": "PCA 24",
    "delay_pca_16": "Delay-embedded PCA", "spatiotemporal_pca_16": "Spatiotemporal PCA", "spatial_avg_24": "Spatial avg 24 (baseline)",
}
# Sequential single-hue ramp (validated blue ramp, light → dark).
RAMP = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])
VMIN, VMAX = 0.2, 0.8
INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#ffffff"


def load(pattern, method_filter=None):
    files = sorted(ROOT.glob(pattern))
    if not files:
        return None
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    for c in ("beh_r_body_me",):
        if c not in d:
            d[c] = np.nan
    if "beh_r_mean" not in d:
        d["beh_r_mean"] = d[[b for b, _ in BEH[:-1]]].mean(axis=1)
    pp = d.groupby(["pid", "method"])[[b for b, _ in BEH]].mean().reset_index()
    if method_filter:
        pp = pp[pp.method.isin(method_filter)]
    med = pp.groupby("method")[[b for b, _ in BEH]].median()
    n = pp.groupby("method")[[b for b, _ in BEH]].count().max()
    return med.sort_values("beh_r_mean", ascending=True), n, pp.pid.nunique()


def panel(ax, med, n, title):
    im = ax.imshow(med.to_numpy(), cmap=RAMP, vmin=VMIN, vmax=VMAX, aspect="auto")
    for i in range(med.shape[0]):
        for j in range(med.shape[1]):
            v = med.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if v > 0.55 else INK, fontweight="bold" if j == med.shape[1] - 1 else "normal")
    ax.set_yticks(range(med.shape[0]), [LABEL.get(m, m) for m in med.index], fontsize=10)
    ax.set_xticks(range(len(BEH)), [f"{lab}\n(n={int(n[b])})" for b, lab in BEH], fontsize=9)
    ax.xaxis.tick_top()
    ax.tick_params(length=0, colors=INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK, pad=46)
    return im


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    panels = []
    a = load("search/probes/*.csv")
    panels.append((a, f"A  Search set: {a[2]} probes, 6 mice"))
    b = load("confirm/probes/*.csv")
    panels.append((b, f"B  Confirmation: {b[2]} probes, 6 new mice"))
    c = load("baseline24/*/*.csv")
    if c is not None:
        panels.append((c, f"C  Spatial-averaging baseline: {c[2]} probes"))
    heights = [p[0][0].shape[0] for p in panels]
    fig, axes = plt.subplots(len(panels), 1, figsize=(8.6, 0.5 * sum(heights) + 2.1 * len(panels)), facecolor=SURFACE,
                             gridspec_kw={"height_ratios": heights, "hspace": 0.9})
    axes = np.atleast_1d(axes)
    rows = []
    for ax, ((med, n, npid), title) in zip(axes, panels):
        im = panel(ax, med, n, title)
        rows.append(med.assign(panel=title.split("  ")[0]))
    cb = fig.colorbar(im, ax=axes.tolist(), orientation="vertical", fraction=0.03, pad=0.02)
    cb.set_label("held-out decoding r (median over probes)", color=INK_2)
    fig.text(0.01, 0.004, "Linear ridge decoding with ±320 ms context, 4 contiguous held-out folds per probe. "
             "Body motion is missing for some public sessions (see n).", fontsize=8.5, color=INK_2)
    out = ROOT / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "fig_behaviour_r_heatmap.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    pd.concat(rows).to_csv(ROOT / "behaviour_r_table.csv")
    print("wrote", out / "fig_behaviour_r_heatmap.png")


if __name__ == "__main__":
    main()
