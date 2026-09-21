"""
All five autoencoder designs against PCA and the linear per-depth models, one protocol.

Every row is 48 numbers per 40 ms bin, scored by lfp_selected_benchmark.evaluate.evaluate_probe on the
same 50 insertions (4 blocked folds, 1 s gaps, ridge readout, +/-320 ms lags, 30 s shift control).
The autoencoders and the pooled/weighted group PCAs are fit on other sessions (5 session folds);
the per-insertion methods are fit on each insertion's training blocks.

  A  ae_depth_48         CNN, 2 latents per depth                       train.py
  B  ae_global_48        CNN, 48 global latents                         train.py
  C  ae_masked_48        A + depth groups hidden during training        train_variants.py (fork)
  D  ae_anatomy_48       A + a learned embedding of each depth's region train_variants.py (fork)
  E  ae_filterbank_48    A's decoder, learned band-pass filters on raw  train_variants.py (fork)
  F  ae_waveamp_48       waveform stored as is + 1 learned amplitude latent per depth (fork)
  W  group_pca_weighted_48  group PCA with the AE's view weights: its linear twin (fork)

    $PY -m lfp_autoencoder_prototype.compare_models --evaluate C D E F    # workers; skip tags already scored
    $PY -m lfp_autoencoder_prototype.compare_models

Writes ~/Downloads/lfp-brain-state/ae_results/compare/: per_probe_models.csv, summary_models.csv,
paired_models.csv, fig_models.png, fig_models_paired.png.
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lfp_selected_benchmark.run_selected import RESULTS
from .train import AE_RESULTS

OUT = AE_RESULTS / "compare"
SOURCES = ([AE_RESULTS / "probes"] + [AE_RESULTS / "variants" / f"probes_{t}" for t in ("C", "D", "E", "F", "W", "WF")]
           + [AE_RESULTS / "compare" / f"probes_{t}" for t in "CDEF"])
VARIANT_KEY = {"C": "ae_masked_48", "D": "ae_anatomy_48", "E": "ae_filterbank_48", "F": "ae_waveamp_48"}
BENCH = ["waveform_pca_48", "spatial_avg_48", "group_pca_2x24", "group_wave+amp1", "waveform_pca_16", "wave12+bank4"]
LABEL = {
    "waveform_pca_48": "PCA 48 (current method)",
    "spatial_avg_48": "Spatial average 48",
    "waveform+gamma_48": "Waveform + gamma per depth (no PCA)",
    "group_pca_2x24": "Group PCA 2×24, per insertion",
    "group_pca_pooled_48": "Group PCA 2×24, other sessions",
    "group_pca_weighted_48": "W  Group PCA, AE view weights",
    "group_wave+amp1_pooled": "WF Waveform + 1 amp PC per depth, other sessions",
    "group_wave+amp1": "Waveform + 1 amp PC per depth",
    "ae_depth_48": "A  CNN, 2 latents per depth",
    "ae_global_48": "B  CNN, global latent",
    "ae_masked_48": "C  CNN, masked depths",
    "ae_anatomy_48": "D  CNN, anatomy-conditioned",
    "ae_filterbank_48": "E  CNN, learned filterbank",
    "ae_waveamp_48": "F  CNN, waveform + learned amplitude",
}
FAMILY = {"waveform_pca_48": "global", "spatial_avg_48": "global", "waveform+gamma_48": "handmade",
          "group_pca_2x24": "linear", "group_pca_pooled_48": "linear", "group_pca_weighted_48": "linear",
          "group_wave+amp1": "linear", "group_wave+amp1_pooled": "linear"}
COLOR = {"global": "#2a78d6", "handmade": "#1baf7a", "linear": "#eb6834", "ae": "#4a3aa7", "reference": "#9a9892"}
FAMILY.update({"waveform_pca_pooled_48": "global", "spatial_avg_24": "global", "bank_456": "reference"})
AES = ["ae_depth_48", "ae_global_48", "ae_masked_48", "ae_anatomy_48", "ae_filterbank_48", "ae_waveamp_48"]
BASELINES = ["group_pca_weighted_48", "group_wave+amp1_pooled", "group_pca_pooled_48", "group_wave+amp1", "waveform_pca_48"]
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def load():
    frames = [pd.read_csv(f) for d in SOURCES if d.exists() for f in sorted(d.glob("*.csv"))]
    # the same model may be scored in two places (fork's variants/ and compare/): keep one copy per fold
    new = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["pid", "method", "fold"])
    beh = [c for c in new.columns if c.startswith("beh_r_") and not c.startswith("beh_r2_")]
    cols = beh + ["r2_wave", "r2_env_gamma", "r2_amp_global", "r2_amp_spatial", "r2_wave_spatial"] + \
        [c.replace("beh_r_", "shuf_r_") for c in beh]
    pp = new.groupby(["pid", "eid", "method"])[cols].mean().reset_index()
    pp["beh_r_mean"] = pp[beh].mean(axis=1)
    pp["shuf_r_mean"] = pp[[c.replace("beh_r_", "shuf_r_") for c in beh]].mean(axis=1)
    bench = pd.read_csv(RESULTS / "per_probe.csv")
    old = bench[bench.method.isin(BENCH)][["pid", "eid", "method"] + cols + ["beh_r_mean", "shuf_r_mean"]]
    pp = pd.concat([pp, old], ignore_index=True)
    counts = pp.groupby("method").pid.nunique()
    full = counts[counts == counts.max()].index
    pending = {m: int(counts.get(m, 0)) for m in AES + ["group_pca_weighted_48"] if counts.get(m, 0) < counts.max()}
    pp = pp[pp.method.isin(full)]
    return pp, beh, pending


def paired(pp):
    rows = []
    for base in BASELINES:
        b = pp[pp.method == base].set_index("pid")
        if b.empty:
            continue
        for m in sorted(set(pp.method) - {base}):
            g = pp[pp.method == m].set_index("pid")
            c = g.index.intersection(b.index)
            for col in ("beh_r_mean", "r2_wave", "r2_amp_spatial"):
                d = g.loc[c, col] - b.loc[c, col]
                s = d.groupby(g.loc[c, "eid"]).mean()
                rows.append({"baseline": base, "method": m, "metric": col, "median_diff": d.median(),
                             "sessions_higher": int((s > 0).sum()), "n_sessions": len(s)})
    return pd.DataFrame(rows)


def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=9.5)


def fig_models(summ, n):
    order = [m for m in ["waveform_pca_48", "spatial_avg_48", "waveform+gamma_48", "group_pca_2x24",
                         "group_pca_pooled_48", "group_pca_weighted_48", "group_wave+amp1", "group_wave+amp1_pooled"]
             + AES if m in summ.index]
    fig, axes = plt.subplots(1, 3, figsize=(15, 0.46 * len(order) + 1.9), facecolor=SURFACE, sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (col, title, lim) in zip(axes, [("beh_r_mean", "Behaviour r (mean of 5)", 0.8),
                                            ("r2_wave", "Waveform R²", 1.1),
                                            ("r2_amp_spatial", "Amplitude depth pattern R²", 0.7)]):
        style(ax)
        v = summ.loc[order, col].to_numpy()
        colors = [COLOR["ae"] if m in AES else COLOR[FAMILY[m]] for m in order]
        ax.barh(y, np.clip(v, 0, None), color=colors, height=0.66)
        for yi, vi in zip(y, v):
            ax.text(max(vi, 0) + lim * 0.012, yi, f"{vi:.2f}", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, lim)
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=INK)
    axes[0].set_yticks(y, [LABEL[m] for m in order], fontsize=10)
    fig.text(0.01, 0.012, f"All 48 numbers per 40 ms. Median over {n} selected insertions. Purple = autoencoders "
             "and orange = linear per-depth models, all fit on other sessions except 'per insertion' and\n"
             "'waveform + 1 amp PC'. Same held-out folds and ridge readout for every row. Bars clipped at 0.",
             fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "fig_models.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)


def fig_paired(pp):
    """Each AE minus its linear twin W and minus the best linear model, per session."""
    bases = [b for b in ("group_pca_weighted_48", "group_wave+amp1") if b in set(pp.method)]
    aes = [m for m in AES if m in set(pp.method)]
    if not bases or not aes:
        return
    fig, axes = plt.subplots(1, len(bases), figsize=(6.2 * len(bases), 0.62 * len(aes) + 1.6), facecolor=SURFACE,
                             sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    y = np.arange(len(aes))[::-1]
    rows = []
    for ax, base in zip(axes[0], bases):
        style(ax)
        b = pp[pp.method == base].groupby("eid").beh_r_mean.mean()
        for yi, m in zip(y, aes):
            s = (pp[pp.method == m].groupby("eid").beh_r_mean.mean() - b).dropna()
            ax.scatter(s, yi + rng.uniform(-0.15, 0.15, len(s)), s=15, color=COLOR["ae"], alpha=0.55, edgecolor="none")
            ax.plot([s.median()] * 2, [yi - 0.3, yi + 0.3], color=INK, linewidth=2.2)
            rows.append({"baseline": base, "method": m, "median_session_diff": s.median(),
                         "sessions_higher": int((s > 0).sum()), "n_sessions": len(s)})
            ax.text(1.02, yi, f"{s.median():+.3f}  ({int((s > 0).sum())}/{len(s)})", transform=ax.get_yaxis_transform(),
                    va="center", fontsize=9, color=INK)
        ax.axvline(0, color=INK, linewidth=1)
        ax.set_xlabel(f"behaviour r: model minus {LABEL[base].replace('W  ', '')}", color=INK_2)
    axes[0][0].set_yticks(y, [LABEL[m] for m in aes], fontsize=10)
    fig.text(0.01, 0.01, "Each dot = one session (mean of its insertions). Text: median difference (sessions where the model "
             "is higher / sessions).", fontsize=8.5, color=INK_2)
    fig.tight_layout(rect=(0, 0.06, 0.9, 1))
    fig.savefig(OUT / "fig_models_paired.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(OUT / "paired_sessions_models.csv", index=False)


def evaluate(tags):
    """Score the fork's held-out latents with the unchanged benchmark protocol, into compare/probes_<tag>/."""
    from pathlib import Path
    from lfp_compression_pilot.pilot_data import find_probe, load_raw
    from lfp_selected_benchmark.evaluate import evaluate_probe
    from lfp_selected_benchmark.extract_selected import CACHE_DIR
    from lfp_selected_benchmark.run_selected import PIDS_CSV
    from .evaluate_ae import Precomputed
    for tag in tags:
        key, lat_dir, sub_dir = VARIANT_KEY[tag], AE_RESULTS / "variants" / f"latents_{tag}", OUT / f"probes_{tag}"
        sub_dir.mkdir(parents=True, exist_ok=True)
        for pid in sorted(pd.read_csv(PIDS_CSV).pid):
            out, lat = sub_dir / f"{pid[:8]}.csv", lat_dir / f"{pid}.npz"
            if out.exists() or not lat.exists():
                continue
            try:
                os.close(os.open(str(out) + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            except FileExistsError:
                continue
            raw, meta, _ = load_raw(pid, CACHE_DIR)
            path, _ = find_probe(pid, CACHE_DIR)
            with np.load(path) as z:
                B = z["behavior"]
            with np.load(lat) as z:
                reps, fold = [Precomputed(key, z[key])], int(z["fold"])
            df = pd.DataFrame(evaluate_probe(raw, B, meta["behavior_names"], reps))
            df.insert(0, "pid", pid)
            df.insert(1, "eid", meta["eid"])
            df.insert(2, "ae_fold", fold)
            df.to_csv(out, index=False)
            Path(str(out) + ".lock").unlink(missing_ok=True)
            print(f"[{tag} {pid[:8]}] done", flush=True)


def main():
    import sys
    if "--evaluate" in sys.argv:
        evaluate(sys.argv[sys.argv.index("--evaluate") + 1:] or list("CDEF"))
        return
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    pp, beh, pending = load()
    n = pp.pid.nunique()
    pp.to_csv(OUT / "per_probe_models.csv", index=False)
    summ = pp.groupby("method")[["beh_r_mean"] + beh + ["r2_wave", "r2_env_gamma", "r2_amp_global", "r2_amp_spatial",
                                                        "shuf_r_mean"]].median()
    summ.insert(0, "n", pp.groupby("method").pid.nunique())
    summ.to_csv(OUT / "summary_models.csv")
    pr = paired(pp)
    pr.to_csv(OUT / "paired_models.csv", index=False)
    fig_models(summ, n)
    fig_paired(pp)
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(summ[["n", "beh_r_mean", "r2_wave", "r2_amp_spatial", "shuf_r_mean"]].sort_values("beh_r_mean"))
        for base in ("group_pca_weighted_48", "group_wave+amp1"):
            q = pr[(pr.baseline == base) & pr.method.isin(AES)]
            if len(q):
                print(f"\nvs {base}:")
                print(q.pivot_table(index="method", columns="metric", values=["median_diff", "sessions_higher"]).round(3))
    fig_purpose()
    if pending:
        print("\nnot yet complete (excluded):", pending)



PURPOSE_LABEL = dict(LABEL, bank_456="Full feature bank (456)", waveform_pca_pooled_48="PCA 48, other sessions",
                     spatial_avg_24="Spatial average 24")


def fig_purpose(path=None):
    """fig_purpose.png from the fork's purpose_summary.csv: the tests tied to tasks 2, 3 and 5."""
    path = path or AE_RESULTS / "purpose" / "purpose_summary.csv"
    if not path.exists():
        return
    t = pd.read_csv(path).set_index("method")
    order = [m for m in ["waveform_pca_48", "waveform_pca_pooled_48", "spatial_avg_24", "group_pca_2x24",
                         "group_pca_pooled_48", "group_pca_weighted_48", "group_wave+amp1", "group_wave+amp1_pooled"]
             + AES + ["bank_456"] if m in t.index]
    panels = [(c, title, ref) for c, title, ref in [
        ("cca_top5", "Task 2 · shared between simultaneous\nprobes (held-out CCA, top 5)", "cca_null_top5"),
        ("states_nmi_within", "Task 3 · 3-state HMM agrees with the\nAkella-style reference (NMI)", None),
        ("transfer_full_r", "Task 5 · readout trained on other\nsessions, applied unchanged (r)", None)]
        if c in t.columns]
    # The first transfer test mixed encoders (each insertion's latents came from its own fold's model), which
    # confounds it for the CNNs; show transfer only once the fold-consistent version exists.
    if not list(path.parent.glob("transfer_consistent*.csv")):
        panels = [pnl for pnl in panels if pnl[0] != "transfer_full_r"]
    cca_file = path.parent / "cca.csv"
    hp = pd.read_csv(cca_file).groupby("method").hp1hz_top5.median() if cca_file.exists() else None
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(max(5.2 * len(panels), 10), 0.42 * len(order) + 2.4),
                             facecolor=SURFACE, sharey=True, squeeze=False)
    y = np.arange(len(order))[::-1]
    for ax, (col, title, ref) in zip(axes[0], panels):
        style(ax)
        v = t.loc[order, col].to_numpy(dtype=float)
        colors = [COLOR["ae"] if m in AES else COLOR[FAMILY.get(m, "reference")] for m in order]
        ax.barh(y, np.clip(np.nan_to_num(v), 0, None), color=colors, height=0.64)
        for yi, vi, m in zip(y, v, order):
            # A and B have no saved weights, so the fold-consistent transfer test cannot score them
            why = "no saved weights" if m in ("ae_depth_48", "ae_global_48") else "not run"
            label = f"–  {why}" if np.isnan(vi) else f"{0.0 if abs(vi) < 0.005 else vi:.2f}"
            ax.text(max(np.nan_to_num(vi), 0) + 0.012, yi, label, va="center", fontsize=8.5,
                    color=INK_2 if np.isnan(vi) else INK)
        if ref and ref in t.columns:
            rv = t.loc[order, ref].to_numpy(dtype=float)
            ax.scatter(rv, y, marker="|", s=160, color=INK, zorder=3, label="circular-shift null")
        if col == "cca_top5" and hp is not None:
            hv = np.array([hp.get(m, np.nan) for m in order])
            ax.scatter(hv, y, marker="o", s=26, facecolor=SURFACE, edgecolor=INK, zorder=4, label="after 1 Hz high-pass")
        if col == "states_nmi_within" and "akella_ref" in t.index:
            ceil = float(t.loc["akella_ref", col])
            ax.axvline(ceil, color=INK, linestyle="--", linewidth=1.1)
            ax.text(ceil + 0.01, y[-1] - 0.9, f"reference HMM\nrefit on itself: {ceil:.2f}", fontsize=8, color=INK, va="bottom")
        if col in ("cca_top5",):
            ax.legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=2)
        ax.set_xlim(0, 1.0 if col != "states_nmi_within" else 0.6)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", color=INK)
    axes[0][0].set_yticks(y, [PURPOSE_LABEL.get(m, m) for m in order], fontsize=9.5)
    fig.text(0.01, 0.01, "Tests by the fork session (purpose_tests.py, state_tests.py); medians over 50 insertions (HMM, "
             "transfer) or 24 simultaneous probe pairs (CCA). Transfer uses one encoder per fold, so readout and\ntest "
             "share coordinates. Rows marked 'other sessions' and all CNNs are fit leave-session-out. "
             "The HMM reference is built from bank features, so bank-derived rows\nhave a home advantage over "
             "waveform-only rows. Bars clipped at 0.", fontsize=8.3, color=INK_2)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(OUT / "fig_purpose.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("figure: fig_purpose.png")


if __name__ == "__main__":
    main()
