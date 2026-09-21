"""
Evaluate the pre-registered methods on one probe (PREREGISTRATION.md, run 2).

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
      /opt/miniconda3/envs/lfp-brain-state/bin/python -m lfp_compression_pilot.run_pilot

Options:
  --pid PID        probe to evaluate (default pilot_config.PILOT_PID)
  --replication    raw source and primary methods only (used by run_replication.py)
  --out-dir DIR

Same 100 s, same preprocessing, same targets, same 4 contiguous folds (1 s gaps),
same ridge readout for every method and source. Nothing learned ever sees test bins.

Writes to OUT_DIR:
  fold_metrics.csv            one row per (fold, source, method)
  summary.csv                 mean and SD over folds
  codec_fidelity.csv          each lfpack variant vs raw at 250 Hz, with measured costs
  reproducibility.csv         bases fit on disjoint halves of the recording
  amplitude_split_selection.csv  inner-CV choices of the run-1 variant (pilot only)
  manifest.json               probe, interval, subject, regions, settings, versions
"""
import argparse
import io
import json
import os
import platform
import time
import tracemalloc
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import scipy
import sklearn

from . import pilot_config as C
from .pilot_data import behavior_25hz, load_sources, targets_from_raw
from .pilot_metrics import (behavior_metrics, codec_fidelity, fit_readout, forecast_r2, inner_folds,
                            outer_folds, r2_weighted, readout_metrics, subspace_similarity,
                            waveform_band_coherence)
from .pilot_representations import AmplitudePCA, Features, Reference, SpatialAverage, WaveformPCA, build

TARGET_LABELS = ("wave", "env_beta", "env_gamma")


def choose_amplitude_split(F, targets, train, d):
    """Run-1 specification, kept as a reported variant: inner-CV choice of the split (criterion: mean R² over 3 targets)."""
    scores = {}
    for share in C.AMP_SHARE_GRID:
        k = int(np.clip(round(d * (1 - share)), 1, d - 1))
        vals = []
        for tr, val in inner_folds(train):
            Z = AmplitudePCA(d, k).fit(F, tr).transform(F)
            for label in TARGET_LABELS:
                predict = fit_readout(Z, targets[label], tr)
                vals.append(r2_weighted(targets[label].T[val].astype(np.float64), predict(val)))
        scores[k] = float(np.mean(vals))
    return max(scores, key=scores.get), scores


def evaluate(rep, F, targets, B, B_shuf, beh_names, train, test):
    tracemalloc.start()
    t0 = time.perf_counter()
    rep.fit(F, train)
    t1 = time.perf_counter()
    Z = rep.transform(F)
    t2 = time.perf_counter()
    peak_mb = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()

    row = {"fit_s": t1 - t0, "transform_s": t2 - t1, "peak_traced_mb": peak_mb}
    for label in TARGET_LABELS:
        m, P, Yt = readout_metrics(Z, targets[label], train, test, label)
        row.update(m)
        if label == "wave":
            row.update(waveform_band_coherence(Yt, P))
    for label in TARGET_LABELS:
        row.update(forecast_r2(Z, targets[label], train, test, label))
    if isinstance(rep, Reference):
        # 1152-feature ceiling: behaviour read out without lag context (5760 lagged
        # features would dominate the runtime); no shuffle control needed.
        row.update(behavior_metrics(Z, B, beh_names, train, test, prefix="beh", lags=(0,)))
    else:
        row.update(behavior_metrics(Z, B, beh_names, train, test, prefix="beh"))
        row.update(behavior_metrics(Z, B_shuf, beh_names, train, test, prefix="shuf"))

    buf = io.BytesIO()
    np.savez_compressed(buf, Z=Z.astype(np.float32))
    row["storage_bytes_per_s"] = buf.getbuffer().nbytes / C.DURATION_S
    return row


def methods_for(source, replication):
    reps = [build(n, k, m) for n, (k, m) in C.PRIMARY_METHODS.items()]
    if source == "raw" and not replication:
        reps += [SpatialAverage(16), WaveformPCA(8)]
        reps += [build(n, k, m) for n, (k, m) in C.TRADEOFF_METHODS.items()]
    elif source == "raw":
        reps += [SpatialAverage(16)]
    else:
        reps = [r for r in reps if r.name in ("waveform_pca_16", "wave16+amp4", "wave12+amp4")]
    return reps


def evaluate_probe(pid, out_dir, replication=False, verbose=True):
    tic = time.time()
    sources, lfpack_info, meta, cosmos = load_sources(pid, with_lfpack=not replication)
    targets = targets_from_raw(sources["raw"])
    n_bins = targets["wave"].shape[1]
    B, beh_names, subject = behavior_25hz(meta, n_bins)
    B_shuf = np.roll(B, int(C.SHUFFLE_SHIFT_S * C.FS_REP), axis=0)
    feats = {s: Features.from_source(x) for s, x in sources.items()}
    regions = sorted(set(cosmos) - {"void", "root"})
    if verbose:
        print(f"[{pid[:8]}] subject {subject} | regions {regions} | behaviour {beh_names} | sources {list(sources)}")

    codec_rows = [{"pid": pid, "source": s, **lfpack_info[s], **codec_fidelity(sources["raw"], sources[s]),
                   "cr_vs_raw_float32_250hz": C.RAW_FLOAT32_BYTES_PER_S_250HZ / lfpack_info[s]["storage_bytes_per_s"]}
                  for s in sources if s != "raw"]

    rows, selections = [], []
    for fold, (train, test) in enumerate(outer_folds(n_bins)):
        for src, F in feats.items():
            reps = methods_for(src, replication)
            if src == "raw" and not replication:
                k, scores = choose_amplitude_split(F, targets, train, 16)
                selections.append({"pid": pid, "fold": fold, "dims": 16, "k_wave": k,
                                   **{f"inner_score_k{kk}": v for kk, v in scores.items()}})
                reps += [AmplitudePCA(16, k, name="amplitude_pca_cv16"), Reference()]
            for rep in reps:
                row = {"pid": pid, "subject": subject, "eid": meta["eid"], "fold": fold, "source": src,
                       "method": rep.name, "dims": rep.dims, "k_wave": getattr(rep, "k", rep.dims),
                       "floats_per_s": rep.floats_per_s, "n_params": rep.n_params,
                       "bytes_per_s_float32": rep.floats_per_s * 4}
                row["cr_vs_raw_float32_250hz"] = C.RAW_FLOAT32_BYTES_PER_S_250HZ / row["bytes_per_s_float32"]
                row.update(evaluate(rep, F, targets, B, B_shuf, beh_names, train, test))
                rows.append(row)
        if verbose:
            print(f"[{pid[:8]}] fold {fold + 1}/{C.N_FOLDS} done ({time.time() - tic:.0f} s)")

    df = pd.DataFrame(rows)
    for pre in ("beh", "shuf"):
        for stat in ("r2", "r"):
            cols = [f"{pre}_{stat}_{n}" for n in beh_names if f"{pre}_{stat}_{n}" in df]
            df[f"{pre}_{stat}_mean"] = df[cols].mean(axis=1) if cols else np.nan

    # Reproducibility within the recording: bases fit on disjoint halves, 1 s apart.
    half = n_bins // 2
    idx = np.arange(n_bins)
    A, Bh = idx[idx < half - C.GAP_BINS], idx[idx >= half + C.GAP_BINS]
    F = feats["raw"]
    n_env = C.N_CHANNELS * len(C.ENVELOPE_BANDS)
    rep_rows = []
    for name, (k, m) in C.PRIMARY_METHODS.items():
        ra, rb = build(name, k, m).fit(F, A), build(name, k, m).fit(F, Bh)
        wa, wb = (ra.W, rb.W) if m == 0 else (ra.wave.W, rb.wave.W)
        rep_rows.append({"pid": pid, "method": name, "part": "waveform basis", "rank": k,
                         "similarity": subspace_similarity(wa, wb), "random_expected": k / C.N_CHANNELS})
        if m:
            rep_rows.append({"pid": pid, "method": name, "part": "amplitude basis", "rank": m,
                             "similarity": subspace_similarity(ra.We, rb.We), "random_expected": m / n_env})

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "fold_metrics.csv", index=False)
    if selections:
        pd.DataFrame(selections).to_csv(out / "amplitude_split_selection.csv", index=False)
    if codec_rows:
        pd.DataFrame(codec_rows).to_csv(out / "codec_fidelity.csv", index=False)
    pd.DataFrame(rep_rows).to_csv(out / "reproducibility.csv", index=False)
    numeric = df.select_dtypes("number").columns.drop("fold")
    summary = df.groupby(["source", "method"], sort=False)[list(numeric)].agg(["mean", "std"])
    summary.columns = [f"{a}__{b}" for a, b in summary.columns]
    summary.to_csv(out / "summary.csv")
    json.dump({
        "pid": pid, "eid": meta["eid"], "subject": subject, "regions_cosmos": regions,
        "interval_main_clock_s": [meta["t0_main_s"], meta["t0_main_s"] + C.DURATION_S],
        "behaviour": beh_names, "n_bins_25hz": int(n_bins), "replication_mode": replication,
        "settings": {k: getattr(C, k) for k in dir(C) if k.isupper() and not k.endswith("_DIR")},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
                     "sklearn": sklearn.__version__, "pandas": pd.__version__},
        "runtime_s": time.time() - tic,
    }, open(out / "manifest.json", "w"), indent=1, default=str)
    if verbose:
        print(f"[{pid[:8]}] done in {time.time() - tic:.0f} s -> {out}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", default=C.PILOT_PID)
    ap.add_argument("--replication", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    tag = "replication" if args.replication else "pilot"
    out = args.out_dir or os.path.join(C.RESULTS_DIR, f"{tag}_{args.pid[:8]}")
    df = evaluate_probe(args.pid, out, replication=args.replication)
    cols = ["r2_wave", "r2_env_beta", "r2_env_gamma", "coh_alpha", "forecast200ms_r2_wave",
            "forecast200ms_r2_env_gamma", "beh_r_mean", "shuf_r_mean", "floats_per_s"]
    with pd.option_context("display.width", 250, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(df.groupby(["source", "method"], sort=False)[cols].mean().to_string())


if __name__ == "__main__":
    main()
