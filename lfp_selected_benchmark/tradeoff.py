"""
Why amplitude, how to split the budget, and why more dimensions stop helping.

Two sweeps on the selected insertions, scored with the benchmark protocol (evaluate.py):

  tradeoff    16 numbers split between k waveform PCs and 16-k PCs of the amplitude views
              (k = 16, 14, 12, 8, 4, 0), plus 12 + 4 with RMS-only and PSD-only amplitude PCs
  saturation  waveform-only PCA with 4, 8, 32, 64 PCs (16 is in the main run), and
              12 waveform PCs + m amplitude PCs with m = 1, 2, 8, 16, 32 (m = 4 is in tradeoff)

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 $PY -m lfp_selected_benchmark.run_selected --worker --extra tradeoff
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 $PY -m lfp_selected_benchmark.run_selected --worker --extra saturation
    $PY -m lfp_selected_benchmark.tradeoff        # aggregate + figures (separate from the main numbers)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lfp_compression_pilot.pilot_representations import Representation, WaveformPCA, top_pcs
from .feature_bank import Bank
from .run_selected import RESULTS


class WaveAmpSplit(Representation):
    """k spatial waveform PCs + m PCs of the z-scored amplitude views (RMS and/or PSD on 24 depths)."""

    def __init__(self, k, m, views=("rms", "psd")):
        self.k, self.m, self.views = k, m, tuple(views)
        self.dims = k + m
        self.name = f"split_w{k}_{'+'.join(self.views)}{m}"

    def fit(self, F, train):
        if self.k:
            self.wave = WaveformPCA(self.k).fit(F, train)
        if self.m:
            self.bank = Bank(self.views).fit(F, train)
            self.W = top_pcs(self.bank.transform(F)[:, train], self.m)
        return self

    def transform(self, F):
        parts = []
        if self.k:
            parts.append(self.wave.transform(F))
        if self.m:
            parts.append(self.W.T @ self.bank.transform(F))
        return np.concatenate(parts, axis=0)


def tradeoff_methods():
    return ([WaveAmpSplit(k, 16 - k) for k in (16, 14, 12, 8, 4, 0)]
            + [WaveAmpSplit(12, 4, ("rms",)), WaveAmpSplit(12, 4, ("psd",))])


def saturation_methods():
    return [WaveAmpSplit(k, 0) for k in (4, 8, 32, 64)] + [WaveAmpSplit(12, m) for m in (1, 2, 8, 16, 32)]


INK, INK_2, SURFACE, GRID = "#0b0b0b", "#52514e", "#ffffff", "#e6e5e1"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"


def load(sub):
    files = sorted((RESULTS / sub).glob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    beh = [c for c in df.columns if c.startswith("beh_r_") and not c.startswith("beh_r2_")]
    pp = df.groupby(["pid", "eid", "method"])[beh + ["r2_wave", "r2_env_gamma", "r2_amp_global", "r2_amp_spatial"]].mean().reset_index()
    pp["beh_r_mean"] = pp[beh].mean(axis=1)
    have = pp.groupby("pid").method.nunique()
    pp = pp[pp.pid.isin(have[have == have.max()].index)]
    return pp, beh


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=10)


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig_dir = RESULTS / "figures"
    pp, beh = load("probes_tradeoff")
    med = pp.groupby("method")[["beh_r_mean", "r2_wave", "r2_env_gamma", "r2_amp_spatial"] + beh].median()
    med["n"] = pp.groupby("method").pid.nunique()
    med.to_csv(RESULTS / "tradeoff_summary.csv")
    base = pp[pp.method == "split_w16_rms+psd0"].set_index("pid")
    split = [f"split_w{k}_rms+psd{16 - k}" for k in (16, 14, 12, 8, 4, 0)]
    amp_dims = [16 - k for k in (16, 14, 12, 8, 4, 0)]

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), facecolor=SURFACE)
    ax = axes[0]
    style(ax)
    ax.plot(amp_dims, med.loc[split, "r2_wave"], marker="o", color=BLUE, linewidth=2, label="waveform R²")
    ax.plot(amp_dims, med.loc[split, "r2_env_gamma"], marker="s", color=ORANGE, linewidth=2, label="gamma amplitude R² (40 ms)")
    for x, v in zip(amp_dims, med.loc[split, "r2_wave"]):
        ax.annotate(f"{v:.2f}", (x, v), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.5, color=BLUE)
    ax.set_xticks(amp_dims, [f"{16 - a}+{a}" for a in amp_dims])
    ax.set_xlabel("waveform PCs + amplitude PCs (16 total)", color=INK_2)
    ax.set_ylim(-0.02, 1.08)
    ax.set_title("What each split reconstructs", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=9, loc="center left")

    ax = axes[1]
    style(ax)
    gains = []
    for m in split:
        g = pp[pp.method == m].set_index("pid")
        c = g.index.intersection(base.index)
        gains.append((g.loc[c, "beh_r_mean"] - base.loc[c, "beh_r_mean"]).median())
    ax.plot(amp_dims, gains, marker="o", color=INK, linewidth=2)
    for x, v in zip(amp_dims, gains):
        ax.annotate(f"{v:+.2f}", (x, v), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8.5, color=INK)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set_xticks(amp_dims, [f"{16 - a}+{a}" for a in amp_dims])
    ax.set_xlabel("waveform PCs + amplitude PCs (16 total)", color=INK_2)
    ax.set_ylabel("behaviour r gain over 16 waveform PCs", color=INK_2)
    ax.set_title("What each split does for behaviour", loc="left", fontsize=12, fontweight="bold", color=INK)

    ax = axes[2]
    style(ax)
    views = ["split_w16_rms+psd0", "split_w12_rms4", "split_w12_psd4", "split_w12_rms+psd4"]
    labels = ["16 waveform", "12 + 4 RMS", "12 + 4 PSD", "12 + 4 RMS+PSD"]
    vals = []
    for m in views:
        g = pp[pp.method == m].set_index("pid")
        c = g.index.intersection(base.index)
        vals.append((g.loc[c, "beh_r_mean"] - base.loc[c, "beh_r_mean"]).median())
    ax.bar(range(4), vals, color=[BLUE, ORANGE, AQUA, INK], width=0.62)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.005, f"{v:+.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(4), labels, rotation=15, ha="right")
    ax.set_title("Which amplitude view", loc="left", fontsize=12, fontweight="bold", color=INK)
    ax.set_ylabel("behaviour r gain over 16 waveform PCs", color=INK_2)
    fig.text(0.01, 0.01, f"Median over {pp.pid.nunique()} selected insertions. Gains are paired per insertion. "
             "Preliminary: absolute decoding values are being re-verified.", fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(fig_dir / "fig_tradeoff_50.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print(med.loc[split + ["split_w12_rms4", "split_w12_psd4"], ["n", "beh_r_mean", "r2_wave", "r2_env_gamma", "r2_amp_spatial"]].round(3))
    import json
    today = {"n_insertions": int(pp.pid.nunique()),
             "split": {f"{16 - a}+{a}": {"gain_beh_r": round(float(g), 3), "r2_wave": round(float(med.loc[m, "r2_wave"]), 3),
                                          "r2_env_gamma": round(float(med.loc[m, "r2_env_gamma"]), 3),
                                          "r2_amp_spatial": round(float(med.loc[m, "r2_amp_spatial"]), 3)}
                       for a, m, g in zip(amp_dims, split, gains)},
             "view": {lab: {"gain_beh_r": round(float(v), 3), "r2_env_gamma": round(float(med.loc[m, "r2_env_gamma"]), 3),
                            "r2_wave": round(float(med.loc[m, "r2_wave"]), 3)} for lab, m, v in zip(labels, views, vals)}}
    json.dump(today, open(RESULTS / "today_numbers.json", "w"), indent=1)

    sat_files = list((RESULTS / "probes_saturation").glob("*.csv"))
    if sat_files:
        sp, _ = load("probes_saturation")
        # Same insertions for every point: restrict the tradeoff rows to insertions in the saturation sweep.
        both = pd.concat([sp, pp[pp.method.isin(["split_w16_rms+psd0", "split_w12_rms+psd4"]) & pp.pid.isin(sp.pid)]])
        sm = both.groupby("method")[["beh_r_mean", "r2_wave"]].median()
        sm["n"] = both.groupby("method").pid.nunique()
        sm.to_csv(RESULTS / "saturation_summary.csv")
        fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.0), facecolor=SURFACE, sharey=True)
        wk = [k for k in (4, 8, 16, 32, 64) if f"split_w{k}_rms+psd0" in sm.index]
        wm = [m for m in (1, 2, 4, 8, 16, 32) if f"split_w12_rms+psd{m}" in sm.index]
        for ax, xs, names, col, title, xl in (
                (a1, wk, [f"split_w{k}_rms+psd0" for k in wk], BLUE, "Waveform PCs only", "number of waveform PCs"),
                (a2, wm, [f"split_w12_rms+psd{m}" for m in wm], ORANGE, "12 waveform PCs + m amplitude PCs", "number of amplitude PCs (m)")):
            style(ax)
            ys = sm.loc[names, "beh_r_mean"].to_numpy()
            ax.plot(xs, ys, marker="o", color=col, linewidth=2)
            for x, y in zip(xs, ys):
                ax.annotate(f"{y:.2f}", (x, y), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=9, color=INK)
            ax.set_xscale("log", base=2)
            ax.set_xticks(xs, [str(x) for x in xs])
            ax.set_xlabel(xl, color=INK_2)
            ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
        a1.set_ylabel("median behaviour r", color=INK_2)
        a1.set_ylim(0, 0.72)
        fig.text(0.01, 0.01, f"Median over {int(sm['n'].min())} insertions. Preliminary: same pipeline whose absolute values are being verified.",
                 fontsize=8.5, color=INK_2)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        fig.savefig(fig_dir / "fig_saturation_50.png", dpi=200, facecolor=SURFACE)
        plt.close(fig)
        print(sm.round(3))


if __name__ == "__main__":
    main()
