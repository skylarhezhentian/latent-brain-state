"""
Stage 1 search and confirmation (SEARCH_PROTOCOL.md).

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
    PY=/opt/miniconda3/envs/lfp-brain-state/bin/python
    $PY -m lfp_compression_search.run_search --set search [--shard 0/3]   # evaluate the 12 search probes
    $PY -m lfp_compression_search.run_search --set search --aggregate-only # summary + winner
    $PY -m lfp_compression_search.run_search --set confirm                 # PCA 16, winner, wave12+amp4 on fresh sessions

Per-probe evaluation is the pilot's (run_pilot.evaluate): raw source, same folds,
readout, targets, behaviour context and shuffled control. Results are written per
probe, so interrupted runs resume and shards can run in parallel.

Writes RESULTS_DIR/search/<set>/:
  probes/<pid8>.csv       fold-level metrics for every method
  per_probe.csv           fold means
  paired.csv              method minus waveform_pca_16, per probe
  method_summary.csv      medians and counts per method (probe level)
  session_summary.csv     the same at session level
  winner.json (search) / verdict.json (confirm)
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from lfp_compression_pilot import pilot_config as C
from lfp_compression_pilot.pilot_data import behavior_25hz, list_cached_probes, load_raw, targets_from_raw
from lfp_compression_pilot.pilot_metrics import outer_folds
from lfp_compression_pilot.run_pilot import evaluate
from .search_methods import SearchFeatures, stage1_methods
from lfp_compression_pilot.pilot_representations import AmplitudePCA, WaveformPCA

CACHES = {"search": C.CACHE_DIR, "confirm": os.path.expanduser("~/Downloads/lfp-brain-state/fast_lfp_cache_confirm")}
ROOT = Path(C.RESULTS_DIR) / "search"
BASELINE = "waveform_pca_16"
METRICS = ["beh_r_mean", "r2_wave", "r2_env_beta", "r2_env_gamma", "coh_alpha", "shuf_r_mean"]
MAX_WAVE_LOSS = 0.02


def methods_for(set_name):
    if set_name == "search":
        return stage1_methods()
    winner = json.load(open(ROOT / "search" / "winner.json"))["winner"]
    registry = {m.name: m for m in stage1_methods()}
    chosen = [WaveformPCA(16), registry[winner]]
    if winner != "wave12+amp4":
        chosen.append(AmplitudePCA(16, 12))
    return chosen


def evaluate_probe(pid, meta, cache_dir, methods, out_csv):
    tic = time.time()
    raw, meta, cosmos = load_raw(pid, cache_dir)
    F = SearchFeatures(raw)
    targets = targets_from_raw(raw)
    n_bins = targets["wave"].shape[1]
    B, names, subject = behavior_25hz(meta, n_bins)
    B_shuf = np.roll(B, int(C.SHUFFLE_SHIFT_S * C.FS_REP), axis=0)
    rows = []
    for fold, (train, test) in enumerate(outer_folds(n_bins)):
        for rep in methods:
            row = {"pid": pid, "eid": meta["eid"], "subject": subject, "fold": fold, "method": rep.name,
                   "dims": rep.dims, "n_params": rep.n_params,
                   "regions": ",".join(sorted(set(cosmos) - {"void", "root"}))}
            row.update(evaluate(rep, F, targets, B, B_shuf, names, train, test))
            rows.append(row)
    df = pd.DataFrame(rows)
    for pre in ("beh", "shuf"):
        cols = [f"{pre}_r_{n}" for n in names]
        df[f"{pre}_r_mean"] = df[cols].mean(axis=1)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[{pid[:8]}] {subject} done in {time.time() - tic:.0f} s", flush=True)


def aggregate(set_name):
    out = ROOT / set_name
    df = pd.concat([pd.read_csv(f) for f in sorted((out / "probes").glob("*.csv"))], ignore_index=True)
    per_probe = df.groupby(["pid", "eid", "subject", "method"])[METRICS + ["fit_s", "transform_s"]].mean().reset_index()
    per_probe.to_csv(out / "per_probe.csv", index=False)
    base = per_probe[per_probe.method == BASELINE].set_index("pid")
    paired = []
    for m, g in per_probe[per_probe.method != BASELINE].groupby("method"):
        g = g.set_index("pid")
        for pid in g.index.intersection(base.index):
            paired.append({"method": m, "pid": pid, "eid": g.loc[pid, "eid"], "subject": g.loc[pid, "subject"],
                           **{k: g.loc[pid, k] - base.loc[pid, k] for k in METRICS}})
    paired = pd.DataFrame(paired)
    paired.to_csv(out / "paired.csv", index=False)

    def summarise(frame, unit):
        rows = []
        for m, g in frame.groupby("method"):
            rows.append({"method": m, f"n_{unit}": len(g),
                         "median_d_beh_r": g.beh_r_mean.median(), f"{unit}_beh_up": int((g.beh_r_mean > 0).sum()),
                         "median_d_wave_r2": g.r2_wave.median(), "worst_d_wave_r2": g.r2_wave.min(),
                         "median_d_beta_r2": g.r2_env_beta.median(), "median_d_gamma_r2": g.r2_env_gamma.median(),
                         "median_d_alpha_coh": g.coh_alpha.median()})
        return pd.DataFrame(rows).sort_values("median_d_beh_r", ascending=False)

    probe_summary = summarise(paired, "probes")
    session_summary = summarise(paired.groupby(["method", "eid"])[METRICS].mean().reset_index(), "sessions")
    probe_summary.to_csv(out / "method_summary.csv", index=False)
    session_summary.to_csv(out / "session_summary.csv", index=False)
    return per_probe, paired, probe_summary, session_summary


def select_winner(probe_summary):
    ok = probe_summary[probe_summary.median_d_wave_r2 >= -MAX_WAVE_LOSS]
    ok = ok[ok.median_d_beh_r > 0]
    if ok.empty:
        return {"winner": None, "rule": "no method gained behaviour r within the waveform constraint"}
    best = ok.sort_values("median_d_beh_r", ascending=False).iloc[0]
    return {"winner": best.method, "median_d_beh_r": float(best.median_d_beh_r),
            "median_d_wave_r2": float(best.median_d_wave_r2),
            "rule": "highest median behaviour-r gain vs waveform_pca_16 among methods with median waveform-R2 loss <= 0.02"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=list(CACHES), required=True)
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    out = ROOT / args.set
    i, n = map(int, args.shard.split("/"))

    if not args.aggregate_only:
        probes = list_cached_probes(CACHES[args.set])
        methods = methods_for(args.set)
        print(f"{args.set}: {len(probes)} probes, methods {[m.name for m in methods]}, shard {i}/{n}", flush=True)
        for j, (pid, meta) in enumerate(probes):
            f = out / "probes" / f"{pid[:8]}.csv"
            if j % n == i and not f.exists():
                evaluate_probe(pid, meta, CACHES[args.set], methods, f)
        if n > 1:
            return

    per_probe, paired, probe_summary, session_summary = aggregate(args.set)
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.float_format", "{:+.3f}".format):
        print("\n=== per probe (method − PCA 16) ===")
        print(probe_summary.to_string(index=False))
        print("\n=== per session ===")
        print(session_summary.to_string(index=False))

    if args.set == "search":
        w = select_winner(probe_summary)
        json.dump(w, open(out / "winner.json", "w"), indent=1)
        print("\nWINNER:", w)
    else:
        winner = json.load(open(ROOT / "search" / "winner.json"))["winner"]
        s = session_summary.set_index("method").loc[winner]
        confirmed = bool(s.sessions_beh_up >= 5 and s.median_d_wave_r2 >= -MAX_WAVE_LOSS)
        v = {"winner": winner, "n_sessions": int(s.n_sessions), "sessions_beh_up": int(s.sessions_beh_up),
             "median_d_beh_r_sessions": float(s.median_d_beh_r), "median_d_wave_r2_sessions": float(s.median_d_wave_r2),
             "confirmed": confirmed,
             "rule": "behaviour r higher in >= 5 of 6 sessions and median waveform-R2 loss <= 0.02"}
        json.dump(v, open(out / "verdict.json", "w"), indent=1)
        print("\nVERDICT:", v)


if __name__ == "__main__":
    main()
