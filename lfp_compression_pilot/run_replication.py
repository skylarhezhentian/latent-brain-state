"""
Replicate the pre-registered comparisons on every other cached probe.

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
      /opt/miniconda3/envs/lfp-brain-state/bin/python -m lfp_compression_pilot.run_replication

Each probe runs through run_pilot.evaluate_probe in replication mode (raw source,
primary methods + spatial_avg_16), then per-probe means over folds are compared as
paired differences. The pilot probe is excluded: it was used to develop the
candidate, so it cannot also confirm it.

Writes to OUT_DIR/replication/:
  per_probe.csv        one row per (probe, method): fold-mean metrics
  paired.csv           one row per (probe, comparison): candidate minus comparator
  paired_summary.csv   per comparison and metric: n probes, n mice, median, n improved, Wilcoxon p
  session_summary.csv  the same at the level of sessions: probes recorded together share the
                       animal and its behaviour, so sessions (not probes) are the independent units.
                       Exact two-sided sign test over sessions.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon

from . import pilot_config as C
from .pilot_data import list_cached_probes
from .run_pilot import evaluate_probe

COMPARISONS = {                      # (candidate, comparator), fixed in PREREGISTRATION.md
    "C2 vs current (16 dims)": ("wave12+amp4", "waveform_pca_16"),
    "C1 vs PCA20 (20 dims)": ("wave16+amp4", "waveform_pca_20"),
    "C1 vs current": ("wave16+amp4", "waveform_pca_16"),
}
METRICS = {                          # metric -> direction that counts as improvement
    "r2_wave": +1, "r2_env_beta": +1, "r2_env_gamma": +1, "coh_alpha": +1,
    "forecast200ms_r2_env_gamma": +1, "forecast200ms_r2_wave": +1, "beh_r_mean": +1,
}


def aggregate(fold_csvs, out):
    df = pd.concat([pd.read_csv(f) for f in fold_csvs], ignore_index=True)
    df = df[df.source == "raw"]
    per_probe = df.groupby(["pid", "subject", "eid", "method"]).mean(numeric_only=True).reset_index()
    per_probe.to_csv(out / "per_probe.csv", index=False)

    paired = []
    for label, (cand, comp) in COMPARISONS.items():
        a = per_probe[per_probe.method == cand].set_index("pid")
        b = per_probe[per_probe.method == comp].set_index("pid")
        for pid in a.index.intersection(b.index):
            paired.append({"comparison": label, "pid": pid, "subject": a.loc[pid, "subject"], "eid": a.loc[pid, "eid"],
                           **{m: a.loc[pid, m] - b.loc[pid, m] for m in METRICS}})
    paired = pd.DataFrame(paired)
    paired.to_csv(out / "paired.csv", index=False)

    rows = []
    for label, g in paired.groupby("comparison", sort=False):
        for m, sign in METRICS.items():
            d = g[m].dropna().to_numpy()
            p = wilcoxon(d).pvalue if len(d) >= 5 and np.any(d != 0) else np.nan
            rows.append({"comparison": label, "metric": m, "n_probes": len(d),
                         "n_mice": g.subject.nunique(), "median_diff": float(np.median(d)),
                         "min_diff": float(d.min()), "max_diff": float(d.max()),
                         "n_improved": int((sign * d > 0).sum()), "wilcoxon_p": p})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "paired_summary.csv", index=False)

    sess_rows = []
    by_session = paired.groupby(["comparison", "eid", "subject"], sort=False)[list(METRICS)].mean().reset_index()
    for label, g in by_session.groupby("comparison", sort=False):
        for m, sign in METRICS.items():
            d = g[m].dropna().to_numpy()
            k = int((sign * d > 0).sum())
            sess_rows.append({"comparison": label, "metric": m, "n_sessions": len(d), "n_mice": g.subject.nunique(),
                              "median_diff": float(np.median(d)), "n_sessions_improved": k,
                              "sign_test_p": binomtest(k, len(d), 0.5).pvalue if len(d) else np.nan})
    pd.DataFrame(sess_rows).to_csv(out / "session_summary.csv", index=False)
    return per_probe, paired, summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=C.RESULTS_DIR)
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    root = Path(args.out_dir)
    out = root / "replication"
    out.mkdir(parents=True, exist_ok=True)

    probes = [(pid, meta) for pid, meta in list_cached_probes() if pid != C.PILOT_PID]
    print(f"{len(probes)} replication probes from {len({m['eid'] for _, m in probes})} sessions")
    csvs = []
    for i, (pid, _) in enumerate(probes):
        d = root / f"replication_{pid[:8]}"
        if not args.aggregate_only and not (d / "fold_metrics.csv").exists():
            print(f"--- {i + 1}/{len(probes)} ---")
            evaluate_probe(pid, d, replication=True)
        if (d / "fold_metrics.csv").exists():
            csvs.append(d / "fold_metrics.csv")

    per_probe, paired, summary = aggregate(csvs, out)
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(summary.to_string(index=False))
    print("saved to:", out)


if __name__ == "__main__":
    main()
