"""
Collect every number the deck quotes into one JSON, straight from the result CSVs.

    /opt/miniconda3/envs/lfp-brain-state/bin/python -m lfp_compression_pilot.export_deck_numbers

The deck builder reads only this file, and verify_outputs.py recomputes each value
from the CSVs and checks the deck text against it, so a slide cannot quote a
number the outputs do not contain.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import pilot_config as C

RES = Path(C.RESULTS_DIR)
PILOT = RES / f"pilot_{C.PILOT_PID[:8]}"


def r(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def collect():
    df = pd.read_csv(PILOT / "fold_metrics.csv")
    man = json.load(open(PILOT / "manifest.json"))
    raw = df[df.source == "raw"].groupby("method")
    mean, sd = raw.mean(numeric_only=True), raw.std(numeric_only=True)
    cols = ["r2_wave", "r2_env_beta", "r2_env_gamma", "coh_delta", "coh_theta", "coh_alpha",
            "forecast200ms_r2_wave", "forecast200ms_r2_env_gamma", "beh_r_mean", "shuf_r_mean",
            "beh_r2_mean", "storage_bytes_per_s", "floats_per_s", "cr_vs_raw_float32_250hz",
            "fit_s", "transform_s", "peak_traced_mb", "n_params"]
    out = {"pilot": {"pid": man["pid"], "subject": man["subject"], "regions": man["regions_cosmos"],
                     "behaviour": man["behaviour"], "duration_s": C.DURATION_S, "n_folds": C.N_FOLDS},
           "raw": {m: {c: r(mean.loc[m, c]) for c in cols if c in mean} for m in mean.index},
           "raw_sd": {m: {c: r(sd.loc[m, c]) for c in cols if c in sd} for m in sd.index}}

    lf = df[df.source != "raw"].groupby(["source", "method"]).mean(numeric_only=True)
    out["on_lfpack"] = {f"{s}|{m}": {c: r(lf.loc[(s, m), c]) for c in ("r2_wave", "r2_env_gamma", "r2_env_beta", "beh_r_mean")}
                        for s, m in lf.index}
    codec = pd.read_csv(PILOT / "codec_fidelity.csv")
    out["codec"] = {row.source: {k: r(v, 4) if isinstance(v, (int, float, np.floating)) else v
                                 for k, v in row.items() if k not in ("pid", "source")}
                    for _, row in codec.iterrows()}
    costs = PILOT / "costs.csv"
    if costs.exists():
        cdf = pd.read_csv(costs)
        out["costs"] = {row.method: {k: (r(v, 6) if isinstance(v, (int, float, np.floating)) else v)
                                     for k, v in row.items() if k != "method"} for _, row in cdf.iterrows()}
    rep = pd.read_csv(PILOT / "reproducibility.csv")
    out["halves"] = {f"{row.method}|{row.part}": {"similarity": r(row.similarity), "random": r(row.random_expected)}
                     for _, row in rep.iterrows()}
    sel = PILOT / "amplitude_split_selection.csv"
    if sel.exists():
        out["cv_split_k_wave"] = sorted(pd.read_csv(sel).k_wave.unique().tolist())

    repl = RES / "replication" / "paired_summary.csv"
    if repl.exists():
        s = pd.read_csv(repl)
        out["replication"] = {f"{row.comparison}|{row.metric}": {
            "n_probes": int(row.n_probes), "n_mice": int(row.n_mice), "median": r(row.median_diff),
            "min": r(row.min_diff), "max": r(row.max_diff), "n_improved": int(row.n_improved),
            "p": r(row.wilcoxon_p, 4)} for _, row in s.iterrows()}
        ss = pd.read_csv(RES / "replication" / "session_summary.csv")
        out["replication_sessions"] = {f"{row.comparison}|{row.metric}": {
            "n_sessions": int(row.n_sessions), "median": r(row.median_diff),
            "n_improved": int(row.n_sessions_improved), "p": r(row.sign_test_p, 4)} for _, row in ss.iterrows()}
        # Strictest view: drop the pilot's own session (its sibling probe shares the animal).
        paired = pd.read_csv(RES / "replication" / "paired.csv")
        pilot_eid = man["eid"]
        indep = paired[paired.eid != pilot_eid].groupby(["comparison", "eid"])[["r2_wave", "r2_env_beta", "r2_env_gamma", "beh_r_mean"]].mean()
        out["replication_excl_pilot_session"] = {
            f"{c}|{m}": {"n_sessions": int(len(g)), "n_improved": int((g[m] > 0).sum()), "median": r(g[m].median())}
            for c, g in indep.groupby(level=0) for m in ("r2_wave", "r2_env_beta", "r2_env_gamma", "beh_r_mean")}
        tol = {}
        for c, g in paired.groupby("comparison"):
            tol[c] = {"n_within_0.02_waveform_loss": int((g.r2_wave >= -0.02).sum()), "n_probes": int(len(g)),
                      "worst_waveform_loss": r(-g.r2_wave.min())}
        out["replication_waveform_tolerance"] = tol
        pp = pd.read_csv(RES / "replication" / "per_probe.csv")
        out["replication_meta"] = {"n_probes": int(pp.pid.nunique()), "n_sessions": int(pp.eid.nunique()),
                                   "n_mice": int(pp.subject.nunique()), "mice": sorted(pp.subject.unique().tolist())}
        rawpp = pp.groupby("method").mean(numeric_only=True)
        out["replication_means"] = {m: {c: r(rawpp.loc[m, c]) for c in ("r2_wave", "r2_env_beta", "r2_env_gamma", "beh_r_mean", "shuf_r_mean")}
                                    for m in rawpp.index}
    # Numbers quoted from Alon's slides for context. Not produced by this pilot.
    out["quoted_from_alon_slides"] = {"rms_envelope_behaviour_r_8sep": 0.46, "signed_pca_behaviour_r_8sep": 0.28,
                                      "pca_r2_8_to_12_dims_31aug": 0.8}
    return out


def main():
    out = collect()
    path = RES / "deck_numbers.json"
    json.dump(out, open(path, "w"), indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
