"""
Collect every result on the five autoencoder designs (A-E) and their PCA-family comparators into
tables and figures, and write the standards/tests table. Every number in the report is read from a
result file here; nothing is typed by hand.

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    KMP_DUPLICATE_LIB_OK=TRUE /opt/miniconda3/envs/lfp-brain-state/bin/python \
        -m lfp_autoencoder_prototype.model_report [--purpose DIR] [--ab-probes DIR] [--out DIR]

Inputs (read only)
  selected_benchmark_results/per_probe.csv            PCA 16/48, spatial avg, group PCA, wave+amp, full bank
  ae_results/probes/                                  A, B, pooled group PCA (evaluate_ae.py, 50-epoch run)
  ae_results/variants/probes_{C,D,E,W}/               C, D, E, weighted group PCA (evaluate_variants.py)
  ae_results/variants/{heldout,imputation,cost}_*.csv  train_variants.py
  ae_results/purpose/                                 purpose_tests.py, state_tests.py
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from .train import AE_RESULTS

ROOT = AE_RESULTS.parent
BENCH = ROOT / "selected_benchmark_results" / "per_probe.csv"
V = AE_RESULTS / "variants"
BEH = ["body_me", "pupil_diameter", "wheel_speed", "whisker_me_left", "whisker_me_right"]
METRICS = ["beh_r_mean", "r2_wave", "r2_env_gamma", "r2_amp_global", "r2_amp_spatial", "shuf_r_mean"]

# column order and labels for every table; letters are the design options
M = {
    "waveform_pca_16": "PCA 16 (current)",
    "waveform_pca_48": "PCA 48",
    "waveform_pca_pooled_48": "PCA 48 pooled",
    "group_pca_pooled_48": "Group PCA pooled",
    "group_pca_weighted_48": "W: linear twin",
    "group_wave+amp1": "Wave+amp per depth",
    "group_wave+amp1_pooled": "WF: F's linear twin",
    "ae_depth_48": "A: per-depth AE",
    "ae_global_48": "B: global AE",
    "ae_masked_48": "C: masked AE",
    "ae_anatomy_48": "D: anatomy AE",
    "ae_filterbank_48": "E: filterbank AE",
    "ae_waveamp_48": "F: wave + learned amp",
    "bank_wave+rms+psd": "Full bank 456",
    "bank_456": "Full bank 456",
}
CORE = ["waveform_pca_48", "group_pca_pooled_48", "group_pca_weighted_48", "group_wave+amp1_pooled",
        "ae_depth_48", "ae_global_48", "ae_masked_48", "ae_anatomy_48", "ae_filterbank_48", "ae_waveamp_48"]
LEARNED = ["ae_depth_48", "ae_global_48", "ae_masked_48", "ae_anatomy_48", "ae_filterbank_48", "ae_waveamp_48"]
DEPTH_INDEXED = {"group_pca_pooled_48", "group_pca_weighted_48", "group_wave+amp1", "ae_depth_48",
                 "ae_masked_48", "ae_anatomy_48", "ae_filterbank_48", "group_pca_2x24"}
POOLED = {"group_pca_pooled_48", "group_pca_weighted_48", "waveform_pca_pooled_48"} | set(LEARNED)
INK, INK2, GRID, SURF = "#1b1b1b", "#5a5955", "#e6e4df", "#fcfcfb"


def per_probe(ab_probes):
    """One row per (insertion, method): mean over the 4 folds, then mean over behaviours."""
    frames = [pd.read_csv(f) for f in sorted(Path(ab_probes).glob("*.csv"))]
    for tag in ("C", "D", "E", "F", "W", "WF"):                # C-F are also scored by the other session in
        frames += [pd.read_csv(f) for f in sorted((V / f"probes_{tag}").glob("*.csv"))]      # ae_results/compare/;
        frames += [pd.read_csv(f) for f in sorted((AE_RESULTS / "compare" / f"probes_{tag}").glob("*.csv"))]   # same protocol
    new = pd.concat(frames, ignore_index=True).drop_duplicates(["pid", "method", "fold"])
    cols = [f"beh_r_{b}" for b in BEH] + [f"shuf_r_{b}" for b in BEH] + METRICS[1:-1]
    pp = new.groupby(["pid", "eid", "method"])[cols].mean().reset_index()
    bench = pd.read_csv(BENCH)
    keep = ["waveform_pca_16", "waveform_pca_48", "group_pca_2x24", "group_wave+amp1", "bank_wave+rms+psd"]
    pp = pd.concat([pp[~pp.method.isin(keep)], bench[bench.method.isin(keep)][["pid", "eid", "method"] + cols]],
                   ignore_index=True)
    pp["beh_r_mean"] = pp[[f"beh_r_{b}" for b in BEH]].mean(axis=1)
    pp["shuf_r_mean"] = pp[[f"shuf_r_{b}" for b in BEH]].mean(axis=1)
    return pp


def paired(pp, a, b, col):
    """Session-level paired comparison: per-session mean difference (a - b)."""
    A = pp[pp.method == a].set_index("pid")
    B = pp[pp.method == b].set_index("pid")
    c = A.index.intersection(B.index)
    if len(c) == 0:
        return None
    d = (A.loc[c, col] - B.loc[c, col])
    s = d.groupby(A.loc[c, "eid"]).mean()
    p = wilcoxon(s).pvalue if len(s) >= 6 and np.any(s != 0) else np.nan
    return {"a": a, "b": b, "metric": col, "median_diff": float(d.median()), "sessions_higher": int((s > 0).sum()),
            "n_sessions": int(len(s)), "n_insertions": int(len(c)), "wilcoxon_p": float(p)}


def fmt(x, nd=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "–"
    return f"{x:+.{nd}f}".replace("+", "") if x >= 0 else f"{x:.{nd}f}".replace("-", "−")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ab-probes", default=str(AE_RESULTS / "probes"))
    ap.add_argument("--purpose", default=str(AE_RESULTS / "purpose"))
    ap.add_argument("--out", default=str(ROOT / "model_purpose_2026-09-21"))
    a = ap.parse_args()
    out, P = Path(a.out), Path(a.purpose)
    (out / "tables").mkdir(parents=True, exist_ok=True)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    N = {}                                             # every number the report uses

    # ---------------------------------------------------------------- benchmark protocol
    pp = per_probe(a.ab_probes)
    pp.to_csv(out / "tables" / "per_insertion_benchmark.csv", index=False)
    methods = [m for m in M if m in set(pp.method)]
    summ = pp.groupby("method")[METRICS + [f"beh_r_{b}" for b in BEH]].median().loc[methods]
    summ.insert(0, "n_insertions", pp.groupby("method").pid.nunique().loc[methods])
    summ.to_csv(out / "tables" / "benchmark_summary.csv")
    N["bench"] = summ.round(4).to_dict(orient="index")

    rows = []
    for base in ("waveform_pca_48", "group_pca_weighted_48", "group_pca_pooled_48", "group_wave+amp1",
                 "group_wave+amp1_pooled", "ae_depth_48"):
        for m in methods:
            if m == base:
                continue
            for col in ("beh_r_mean", "r2_wave", "r2_amp_spatial", "r2_env_gamma"):
                r = paired(pp, m, base, col)
                if r:
                    rows.append(r)
    pr = pd.DataFrame(rows)
    pr.to_csv(out / "tables" / "paired_session_level.csv", index=False)
    N["paired"] = {f"{r.a}|{r.b}|{r.metric}": {"d": round(r.median_diff, 4), "k": r.sessions_higher, "n": r.n_sessions,
                                                "p": r.wilcoxon_p} for r in pr.itertuples()}

    # 15-epoch vs 50-epoch A and B: does lower reconstruction loss help the readouts?
    e15 = AE_RESULTS / "run_e15" / "probes"
    if e15.exists() and Path(a.ab_probes).resolve() != e15.resolve():
        old = per_probe(e15)
        old = old[old.method.isin(["ae_depth_48", "ae_global_48"])].copy()
        old["method"] = old.method + "_e15"
        both = pd.concat([pp, old], ignore_index=True)
        N["e15_vs_e50"] = {}
        for m in ("ae_depth_48", "ae_global_48"):
            for col in ("beh_r_mean", "r2_wave", "r2_amp_spatial", "r2_env_gamma"):
                r = paired(both, m, m + "_e15", col)
                if r:
                    N["e15_vs_e50"][f"{m}|{col}"] = {"e15": float(old[old.method == m + "_e15"][col].median()),
                                                     "e50": float(pp[pp.method == m][col].median()),
                                                     "d": r["median_diff"], "k": r["sessions_higher"], "n": r["n_sessions"]}

    # ---------------------------------------------------------------- held-out reconstruction, imputation, cost
    held = pd.concat([pd.read_csv(f) for f in V.glob("heldout_*_fold*.csv")], ignore_index=True) if list(V.glob("heldout_*")) else pd.DataFrame()
    if len(held):
        held = held.drop_duplicates(["pid", "method"])
        hs = held.groupby("method")[["waveform", "rms", "psd"]].median()
        hs.to_csv(out / "tables" / "heldout_reconstruction.csv")
        N["heldout"] = hs.round(4).to_dict(orient="index")
    if (V / "heldout_static_vs_dynamic.csv").exists():
        sd_ = pd.read_csv(V / "heldout_static_vs_dynamic.csv")
        sds = sd_.groupby("method")[[c for c in sd_.columns if c.endswith(("_raw", "_dynamic"))]].median()
        sds.to_csv(out / "tables" / "heldout_static_vs_dynamic.csv")
        N["heldout_sd"] = sds.round(4).to_dict(orient="index")
    imp = pd.concat([pd.read_csv(f) for f in list(V.glob("imputation_fold*.csv")) + list(V.glob("imputation_linear_baseline.csv"))],
                    ignore_index=True) if list(V.glob("imputation_*")) else pd.DataFrame()
    if len(imp):
        ip = imp.groupby(["pid", "mask", "method"])[["r2_waveform", "r2_rms", "r2_psd"]].mean().reset_index()
        isum = ip.groupby(["mask", "method"])[["r2_waveform", "r2_rms", "r2_psd"]].median()
        isum.to_csv(out / "tables" / "imputation.csv")
        N["imputation"] = {f"{k[0]}|{k[1]}": v for k, v in isum.round(4).to_dict(orient="index").items()}
        win = {}
        for mask, g in ip.groupby("mask"):
            for view in ("r2_waveform", "r2_rms", "r2_psd"):
                w = g.pivot(index="pid", columns="method", values=view)
                base = "depth_interp_shrunk" if "depth_interp_shrunk" in w else "depth_interp"
                win[f"{mask}|{view}"] = {"ae_better": int((w.masked_ae > w[base]).sum()), "n": int(len(w)), "baseline": base}
        N["imputation_wins_rms"] = win
    cost = pd.concat([pd.read_csv(f) for f in V.glob("cost_*_fold*.csv")], ignore_index=True) if list(V.glob("cost_*")) else pd.DataFrame()
    ab_log = pd.concat([pd.read_csv(f) for f in sorted(AE_RESULTS.glob("training_log_fold*.csv"))], ignore_index=True)
    ab_cost = ab_log.groupby(["model", "fold"]).agg(epochs_run=("epoch", "max"), train_s=("epoch_s", "sum")).reset_index()
    ab_cost["epochs_run"] += 1
    ab_params = json.load(open(sorted(AE_RESULTS.glob("train_meta_fold*.json"))[0]))["params"]
    ab_cost["params"] = ab_cost.model.map(ab_params)
    allcost = pd.concat([ab_cost, cost], ignore_index=True)
    cs = allcost.groupby("model").agg(params=("params", "first"), epochs_run=("epochs_run", "median"),
                                      train_min_per_fold=("train_s", lambda s: s.median() / 60))
    if "encode_s_per_100s" in allcost:
        cs["encode_s_per_100s"] = allcost.groupby("model").encode_s_per_100s.median()
    logs = pd.concat([ab_log] + [pd.read_csv(f) for f in V.glob("training_log_*_fold*.csv")], ignore_index=True)
    be = logs.loc[logs.groupby(["model", "fold"]).val_loss.idxmin()][["model", "fold", "epoch"]].rename(columns={"epoch": "best_epoch"})
    last = logs.groupby(["model", "fold"]).epoch.max().rename("last_epoch").reset_index()
    be = be.merge(last, on=["model", "fold"])
    be["best_in_last_5"] = be.best_epoch >= be.last_epoch - 4
    be.to_csv(out / "tables" / "training_convergence.csv", index=False)
    N["convergence"] = {m: {"median_best_epoch": float(g.best_epoch.median()), "median_last_epoch": float(g.last_epoch.median()),
                            "folds_best_in_last_5": int(g.best_in_last_5.sum()), "n_folds": int(len(g))}
                        for m, g in be.groupby("model")}
    cs.to_csv(out / "tables" / "cost.csv")
    N["cost"] = cs.round(4).to_dict(orient="index")
    N["ab_epochs_cap"] = int(json.load(open(sorted(AE_RESULTS.glob("train_meta_fold*.json"))[0]))["epochs"])

    # ---------------------------------------------------------------- learned filterbank (E)
    fb = []
    for f in sorted((V / "models").glob("ae_filterbank_48_fold*.pt")):
        import torch
        sd = torch.load(f, map_location="cpu")
        c = sd["fb.log_c"].exp().clamp(0.5, 120).numpy()
        w = sd["fb.log_s"].exp().clamp(0.3, 60).numpy()
        fb += [{"fold": int(f.stem[-1]), "filter": i, "centre_hz": float(ci), "width_hz": float(wi)} for i, (ci, wi) in enumerate(zip(c, w))]
    if fb:
        fbd = pd.DataFrame(fb)
        fbd.to_csv(out / "tables" / "filterbank_learned.csv", index=False)
        fs = fbd.groupby("filter").agg(centre_med=("centre_hz", "median"), centre_min=("centre_hz", "min"),
                                       centre_max=("centre_hz", "max"), width_med=("width_hz", "median"),
                                       width_min=("width_hz", "min"), width_max=("width_hz", "max"), n_folds=("fold", "nunique"))
        N["filterbank"] = {str(k): v for k, v in fs.round(2).to_dict(orient="index").items()}

    # ---------------------------------------------------------------- purpose tests
    # zero-shot transfer: 'consistent' = one fold's encoder for every insertion (consistent_latents.py);
    # 'mixed' = each insertion encoded by its own fold's model (purpose_tests.py), which asks whether
    # separately trained encoders agree, an identifiability question rather than a test of R4
    within = pp.groupby("method").beh_r_mean.median()
    for tag, files in (("transfer_mixed", sorted(P.glob("transfer_[0-9].csv")) or sorted(P.glob("transfer.csv"))),
                       ("transfer", sorted(P.glob("transfer_consistent_*.csv")))):
        if not files:
            continue
        tr = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        tp = tr.groupby(["method", "readout", "pid"]).r.mean().reset_index()
        ts = tp.groupby(["method", "readout"]).r.median().unstack()
        ts.to_csv(out / "tables" / f"{tag}_zero_shot.csv")
        N[tag] = ts.round(4).to_dict(orient="index")
        N[f"{tag}_ratio"] = {m: round(float(ts.loc[m, "full"] / within[m]), 3) for m in ts.index
                             if m in within and np.isfinite(ts.loc[m, "full"])}
    # methods with no trained encoder (per-insertion or fixed fits) have no encoder-mixing issue: their
    # transfer_mixed value IS their zero-shot transfer
    for key in ("transfer", "transfer_ratio"):
        N.setdefault(key, {})
        for m, v in N.get(key.replace("transfer", "transfer_mixed"), {}).items():
            if m not in N[key] and m not in LEARNED:
                N[key][m] = v
    if (P / "cca.csv").exists():
        cc = pd.read_csv(P / "cca.csv")
        cs2 = cc.groupby("method")[[c for c in ("cc1", "cc_top5", "null_top5", "n_shared_dims", "hp1hz_top5", "hp1hz_null_top5") if c in cc]].median()
        cs2.to_csv(out / "tables" / "cca_simultaneous_probes.csv")
        N["cca"] = cs2.round(4).to_dict(orient="index")
        N["cca_npairs"] = int(cc.eid.nunique())
    for name in ("timescale", "interpret"):
        if (P / f"{name}.csv").exists():
            N[f"has_{name}"] = True
    if (P / "timescale.csv").exists():
        t = pd.read_csv(P / "timescale.csv").groupby("method").acf_1e_s.median()
        t.to_csv(out / "tables" / "timescale.csv")
        N["timescale"] = t.round(3).to_dict()
    swf = sorted(P.glob("states_within_*.csv")) or sorted(P.glob("states_within.csv"))
    if swf:
        sw = pd.concat([pd.read_csv(f) for f in swf], ignore_index=True)
        ss = sw.groupby("method")[[c for c in sw.columns if c.startswith(("nmi", "ari")) or c in ("eta2", "dwell_s")]].median()
        ss.to_csv(out / "tables" / "states_within.csv")
        N["states_within"] = ss.round(4).to_dict(orient="index")
    for tag, files in (("states_pooled_mixed", sorted(P.glob("states_pooled_[0-9].csv")) or sorted(P.glob("states_pooled.csv"))),
                       ("states_pooled", sorted(P.glob("states_pooled_consistent_*.csv")))):
        if not files:
            continue
        sp = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
        sps = sp.groupby("method")[["nmi_akella_ref", "eta2", "dwell_s"]].median()
        sps.to_csv(out / "tables" / f"{tag}.csv")
        N[tag] = sps.round(4).to_dict(orient="index")
    N.setdefault("states_pooled", {})
    for m, v in N.get("states_pooled_mixed", {}).items():
        if m not in N["states_pooled"] and m not in LEARNED:
            N["states_pooled"][m] = v
    if (P / "interpret.csv").exists():
        it = pd.read_csv(P / "interpret.csv")
        feats = [c for c in it.columns if c not in ("pid", "method", "channel")]
        prof = {}
        for (m, c), g in it.groupby(["method", "channel"]):
            A = g[feats].to_numpy()
            ref = A.mean(axis=0)
            s = np.sign(A @ ref)[:, None]                  # align each insertion's sign to the mean profile
            prof[f"{m}|{c}"] = (A * s).mean(axis=0)
        pf = pd.DataFrame(prof, index=feats).T
        pf.to_csv(out / "tables" / "latent_feature_profiles.csv")
        folds = pd.read_csv(out / "tables" / "per_insertion_benchmark.csv")[["pid"]].drop_duplicates()
        N["interpret_max_abs"] = {k: round(float(np.abs(v).max()), 3) for k, v in prof.items()}

    # one row per method for other documents (column meanings in the module docstring of purpose_tests.py)
    summ_rows = {}
    for m in sorted(set(N.get("transfer", {})) | set(N.get("transfer_mixed", {})) | set(N.get("cca", {})) | set(N.get("states_within", {}))):
        r = {}
        if m in N.get("transfer", {}):
            r["transfer_full_r"] = N["transfer"][m].get("full")
            r["transfer_pooled_r"] = N["transfer"][m].get("pooled")
            r["transfer_ratio"] = N.get("transfer_ratio", {}).get(m)
        if m in N.get("transfer_mixed", {}):
            r["transfer_full_r_mixed_encoders"] = N["transfer_mixed"][m].get("full")
        if m in N.get("cca", {}):
            r.update(cca_top5=N["cca"][m]["cc_top5"], cca_null_top5=N["cca"][m]["null_top5"], cca_shared_dims=N["cca"][m]["n_shared_dims"])
        if m in N.get("states_within", {}):
            sw_ = N["states_within"][m]
            r.update(states_nmi_within=sw_.get("nmi_akella_ref"), states_eta2=sw_.get("eta2"), states_dwell_s=sw_.get("dwell_s"))
        if m in N.get("states_pooled", {}):
            r["states_nmi_pooled"] = N["states_pooled"][m].get("nmi_akella_ref")
        if m in N.get("states_pooled_mixed", {}):
            r["states_nmi_pooled_mixed_encoders"] = N["states_pooled_mixed"][m].get("nmi_akella_ref")
        if m in N.get("timescale", {}):
            r["acf_1e_s"] = N["timescale"][m]
        summ_rows[m] = r
    if summ_rows:
        ps = pd.DataFrame(summ_rows).T
        ps.index.name = "method"
        ps.to_csv(out / "tables" / "purpose_summary.csv")
        ps.to_csv(P / "purpose_summary.csv")
    json.dump(N, open(out / "tables" / "numbers.json", "w"), indent=1, default=float)
    figures(pp, out, N)
    purpose_figures(out, N)
    print("report tables and figures ->", out)
    return N


def figures(pp, out, N):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": INK2,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2})
    methods = [m for m in CORE if m in set(pp.method)]
    cols = [("beh_r_mean", "Behaviour r (mean of 5)"), ("r2_wave", "Waveform R²"),
            ("r2_amp_spatial", "Amplitude depth-pattern R²"), ("r2_env_gamma", "Gamma envelope R² (40 ms)")]
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.6), facecolor=SURF, sharey=True)
    for ax, (c, title) in zip(axes, cols):
        ax.set_facecolor(SURF)
        data = [pp[pp.method == m][c].dropna().to_numpy() for m in methods]
        y = np.arange(len(methods))[::-1]
        ax.boxplot(data, positions=y, vert=False, widths=0.55, showfliers=False,
                   medianprops={"color": INK, "linewidth": 1.6}, boxprops={"color": INK2}, whiskerprops={"color": INK2},
                   capprops={"color": INK2})
        for yi, d, m in zip(y, data, methods):
            col = "#c2562b" if m in LEARNED else "#2a6fb0" if m.startswith("group") else "#8a8a8a"
            ax.scatter(d, yi + np.random.default_rng(0).uniform(-0.18, 0.18, len(d)), s=6, color=col, alpha=0.45, lw=0)
            ax.text(1.02, yi, f"{np.median(d):.2f}", transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=INK)
        ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold", pad=18)
        ax.grid(axis="x", color=GRID)
        if c == "r2_env_gamma":
            ax.set_xlim(-0.15, 0.9)                        # two outlier insertions below -0.15 are clipped
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(np.arange(len(methods))[::-1])
    axes[0].set_yticklabels([M[m] for m in methods])
    fig.text(0.01, 0.98, "Six autoencoder designs vs PCA-family comparators — benchmark protocol, 48 numbers per 40 ms",
             fontsize=12, fontweight="bold", color=INK, va="top")
    fig.text(0.01, 0.93, f"Each dot = one held-out insertion (n = {pp[pp.method == 'waveform_pca_48'].pid.nunique()}); "
             "numbers = medians. Orange = learned, blue = group (per-depth) PCA, grey = waveform PCA.",
             fontsize=8.5, color=INK2, va="top")
    fig.tight_layout(rect=(0, 0, 0.97, 0.9))
    fig.savefig(out / "figures" / "fig_models_benchmark.png", dpi=170, facecolor=SURF)
    plt.close(fig)

    # paired differences vs the linear twin and vs PCA 48
    for base, fname in (("group_pca_weighted_48", "fig_paired_vs_linear_twin.png"), ("waveform_pca_48", "fig_paired_vs_pca48.png")):
        ms = [m for m in methods if m != base]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), facecolor=SURF, sharey=True)
        for ax, (c, title) in zip(axes, cols[:3]):
            ax.set_facecolor(SURF)
            B = pp[pp.method == base].set_index("pid")
            for i, m in enumerate(ms[::-1]):
                A = pp[pp.method == m].set_index("pid")
                ix = A.index.intersection(B.index)
                d = (A.loc[ix, c] - B.loc[ix, c]).groupby(A.loc[ix, "eid"]).mean()
                col = "#c2562b" if m in LEARNED else "#2a6fb0"
                ax.scatter(d, np.full(len(d), i) + np.random.default_rng(1).uniform(-0.2, 0.2, len(d)), s=10, color=col, alpha=0.6, lw=0)
                ax.plot([d.median()] * 2, [i - 0.3, i + 0.3], color=INK, lw=2)
                ax.text(1.02, i, f"{(d > 0).sum()}/{len(d)}", transform=ax.get_yaxis_transform(), va="center", fontsize=8, color=INK)
            ax.axvline(0, color=INK2, lw=0.8)
            ax.set_title(title, loc="left", fontsize=10, color=INK, fontweight="bold", pad=16)
            ax.grid(axis="x", color=GRID)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[0].set_yticks(range(len(ms)))
        axes[0].set_yticklabels([M[m] for m in ms[::-1]])
        fig.text(0.01, 0.98, f"Session-level difference from {M[base]} (dots = sessions; bar = median; right = sessions higher)",
                 fontsize=11.5, fontweight="bold", color=INK, va="top")
        fig.tight_layout(rect=(0, 0, 0.97, 0.92))
        fig.savefig(out / "figures" / fname, dpi=170, facecolor=SURF)
        plt.close(fig)



def purpose_figures(out, N):
    """Transfer, CCA and HMM-state panels; learned filterbank; imputation."""
    order = [m for m in ["waveform_pca_48", "waveform_pca_pooled_48", "group_pca_pooled_48", "group_pca_weighted_48",
                         "ae_depth_48", "ae_global_48", "ae_masked_48", "ae_anatomy_48", "ae_filterbank_48", "ae_waveamp_48"]]
    lab = dict(M, group_pca_weighted_48="W = WF: linear twin")
    panels = [("Zero-shot transfer r (readout trained on\nother sessions; one encoder per fold)", lambda m: get(N, "transfer", m, "full"), None),
              ("Cross-probe CCA, top-5 r\n(hatched: after 1 Hz high-pass)", lambda m: get(N, "cca", m, "cc_top5"),
               lambda m: get(N, "cca", m, "hp1hz_top5")),
              ("HMM state agreement (NMI)\nwith Akella-style reference", lambda m: get(N, "states_within", m, "nmi_akella_ref"), None),
              ("Pooled HMM across animals\n(NMI, held-out sessions)", lambda m: get(N, "states_pooled", m, "nmi_akella_ref"), None)]
    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.8), facecolor=SURF, sharey=True)
    y = np.arange(len(order))[::-1]
    for ax, (title, f, f2_) in zip(axes, panels):
        ax.set_facecolor(SURF)
        for yi, m in zip(y, order):
            v = f(m)
            if v is None:
                if m in ("ae_depth_48", "ae_global_48"):
                    ax.text(0.01, yi, "n/a: no saved weights", va="center", fontsize=7.5, color=INK2)
                continue
            col = "#c2562b" if m in LEARNED else "#2a6fb0" if m.startswith("group") else "#8a8a8a"
            ax.barh(yi, v, color=col, height=0.62)
            if f2_ is not None and f2_(m) is not None:
                ax.barh(yi, f2_(m), color="none", edgecolor=INK, hatch="///", height=0.62, lw=0.6)
            ax.text(v + 0.01, yi, f"{v:.2f}", va="center", fontsize=8, color=INK)
        ax.set_title(title, loc="left", fontsize=9.5, color=INK, fontweight="bold")
        ax.grid(axis="x", color=GRID)
        ax.set_xlim(0, 1.0)
        if title.startswith("HMM state agreement") and get(N, "states_within", "akella_ref", "nmi_akella_ref") is not None:
            c_ = get(N, "states_within", "akella_ref", "nmi_akella_ref")
            ax.axvline(c_, color=INK, ls="--", lw=0.9)
            ax.text(c_ + 0.01, y[-1] - 0.45, f"refit ceiling {c_:.2f}", fontsize=7.5, color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([lab.get(m, m) for m in order])
    fig.text(0.01, 0.98, "Tests of what the representation is for: shared coordinates, cross-region structure, brain states",
             fontsize=12, fontweight="bold", color=INK, va="top")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out / "figures" / "fig_purpose_tests.png", dpi=170, facecolor=SURF)
    plt.close(fig)

    fb = out / "tables" / "filterbank_learned.csv"
    if fb.exists():
        d = pd.read_csv(fb)
        fig, ax = plt.subplots(figsize=(9, 3.6), facecolor=SURF)
        ax.set_facecolor(SURF)
        for (lo, hi), name in zip([(1, 4), (4, 8), (8, 12), (15, 30), (30, 90)], ["delta", "theta", "alpha", "beta", "gamma"]):
            ax.axvspan(lo, hi, color="#e9e6df" if name in ("delta", "alpha", "gamma") else "#f4f2ed", zorder=0)
            ax.text(np.sqrt(lo * hi), 5.6, name, ha="center", fontsize=8, color=INK2)
        init = np.logspace(np.log10(2), np.log10(80), 8)
        ax.scatter(init, np.full(8, -0.6), marker="|", s=120, color=INK2, label="initial centres")
        for fold, g in d.groupby("fold"):
            ax.errorbar(g.centre_hz, np.full(len(g), fold), xerr=g.width_hz, fmt="o", color="#c2562b", ms=4, lw=1, capsize=0)
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50, 100])
        ax.set_xticklabels(["1", "2", "5", "10", "20", "50", "100"])
        ax.set_yticks(range(5))
        ax.set_yticklabels([f"fold {k}" for k in range(5)])
        ax.set_ylim(-1.2, 6.2)
        ax.set_xlabel("Hz (dot = learned centre, bar = ± learned width)")
        ax.set_title("E: learned band-pass filters, trained from raw 250 Hz LFP in 5 independent folds", loc="left",
                     fontsize=10.5, fontweight="bold", color=INK)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="lower right")
        fig.tight_layout()
        fig.savefig(out / "figures" / "fig_filterbank_learned.png", dpi=170, facecolor=SURF)
        plt.close(fig)

    imp = out / "tables" / "imputation.csv"
    if imp.exists():
        d = pd.read_csv(imp)
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), facecolor=SURF, sharey=True)
        names = {"masked_ae": "C: masked AE", "depth_interp": "linear interpolation", "depth_interp_shrunk": "shrunk interpolation"}
        cols = {"masked_ae": "#c2562b", "depth_interp": "#b8b6b0", "depth_interp_shrunk": "#2a6fb0"}
        for ax, mask in zip(axes, ["random25", "block3"]):
            ax.set_facecolor(SURF)
            g = d[d["mask"] == mask].set_index("method")
            x = np.arange(3)
            for i, meth in enumerate(["depth_interp", "depth_interp_shrunk", "masked_ae"]):
                if meth in g.index:
                    ax.bar(x + (i - 1) * 0.26, g.loc[meth, ["r2_waveform", "r2_rms", "r2_psd"]], width=0.26, color=cols[meth], label=names[meth])
            ax.axhline(0, color=INK2, lw=0.8)
            ax.set_xticks(x)
            ax.set_xticklabels(["waveform", "RMS", "PSD"])
            ax.set_title({"random25": "random 25 % of depths hidden", "block3": "3-group block (0.96 mm) hidden"}[mask],
                         loc="left", fontsize=10, fontweight="bold", color=INK)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
        axes[0].set_ylabel("R² on hidden groups (median)")
        axes[1].legend(frameon=False, fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "figures" / "fig_imputation.png", dpi=170, facecolor=SURF)
        plt.close(fig)


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


if __name__ == "__main__":
    main()
