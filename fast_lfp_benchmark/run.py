"""
Score every candidate representation on the cached sessions.

    python -m fast_lfp_benchmark.run
    python -m fast_lfp_benchmark.run --cache-dir DIR --out-dir DIR

PCA is fit without the session it is scored on (leave-one-session-out), so no
candidate sees its test probe during fitting. With a single cached session the
runner falls back to leave-one-probe-out and says so.

Outputs in --out-dir:
  probe_metrics.csv          one row per (probe, candidate), every metric
  summary.csv                mean over probes per candidate
  pca_reproducibility.csv    subspace similarity of PCA fit on disjoint session halves
  ranking_reproducibility.csv  does the ordering of candidates replicate across halves?
"""
import argparse
import itertools
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from . import metrics as M
from .representations import FS, N_CHANNELS, ChannelBin, Lfpack, Raw, SpatialPCA

PCA_RANKS = (1, 2, 4, 8, 16, 24, 32, 64)
NATIVE_FLOATS_PER_S = N_CHANNELS * 2500
CACHE_DIR = os.path.expanduser("~/Downloads/lfp-brain-state/fast_lfp_cache")
OUT_DIR = os.path.expanduser("~/Downloads/lfp-brain-state/fast_lfp_results")
RANKING_METRICS = ("r2", "coh_delta", "coh_gamma", "env_corr_beta", "env_corr_gamma", "beh_retained")


@dataclass
class Entry:
    eid: str
    pid: str
    path: Path
    meta: dict


def index_cache(cache_dir):
    entries = []
    for sj in sorted(Path(cache_dir).glob("*/session.json")):
        meta = json.load(open(sj))
        for pid in meta["pids"]:
            f = sj.parent / f"{pid}.npz"
            if f.exists():
                entries.append(Entry(meta["eid"], pid, f, meta))
    return entries


def load(entry):
    """Probes are loaded one at a time: each holds three 384 x T float arrays."""
    with np.load(entry.path) as z:
        d = {k: z[k] for k in z.files}
    # Chunked decimation can leave a sample or two of rounding at the end; trim so
    # simultaneous probes share exactly duration x fs samples.
    n = int(round(entry.meta["duration_s"] * entry.meta["fs"]))
    for k, v in d.items():
        if k.endswith("_uV"):
            d[k] = v[:, :n]
    d.update(eid=entry.eid, pid=entry.pid,
             behavior_names=entry.meta["behavior_names"],
             behavior_step_s=entry.meta["behavior_step_s"])
    return d


def score_probe(probe, candidates):
    x = probe["x_uV"]
    xs = M.subset(x)
    env = M.band_envelopes(xs, FS)
    B, names, step = probe["behavior"], probe["behavior_names"], probe["behavior_step_s"]
    n_bins = min(B.shape[0], int(env["delta"].shape[1] / (M.ENV_FS * step)))
    raw_beh = M.behavior_r2(M.behavior_features(env, n_bins, step), B, names)

    rows = []
    for rep in candidates:
        x_hat = rep.reconstruct(probe)
        if rep.name == "raw":
            xs_hat, env_hat, beh = xs, env, raw_beh
        else:
            xs_hat = M.subset(x_hat)
            env_hat = M.band_envelopes(xs_hat, FS)
            beh = M.behavior_r2(M.behavior_features(env_hat, n_bins, step), B, names)

        fps = rep.floats_per_second(probe)
        row = {
            "eid": probe["eid"], "pid": probe["pid"], "candidate": rep.name,
            "floats_per_s": fps, "compression_vs_native": NATIVE_FLOATS_PER_S / fps,
            "n_params": rep.n_params, "shared_latent": rep.shared_latent,
        }
        row.update(M.reconstruction(x, x_hat))
        row.update(M.frequency(xs, xs_hat, FS))
        row.update(M.temporal(env, env_hat))
        row.update(beh)

        # Fraction of the behavioural information in the raw signal that survives,
        # averaged over behaviours the raw signal actually predicts (R² > 0.05).
        ratios = [beh[k] / raw_beh[k] for k in raw_beh
                  if np.isfinite(raw_beh[k]) and raw_beh[k] > 0.05 and np.isfinite(beh[k])]
        row["beh_retained"] = float(np.mean(ratios)) if ratios else np.nan
        rows.append(row)
    return rows


def half_splits(sessions):
    """All ways to split sessions into two equal halves (each unordered split once)."""
    k = len(sessions) // 2
    seen, out = set(), []
    for a in itertools.combinations(sessions, k):
        b = tuple(s for s in sessions if s not in a)[:k]
        key = frozenset([frozenset(a), frozenset(b)])
        if key not in seen:
            seen.add(key)
            out.append((set(a), set(b)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    entries = index_cache(args.cache_dir)
    sessions = sorted({e.eid for e in entries})
    print(f"{len(entries)} probes from {len(sessions)} sessions")
    if not entries:
        raise SystemExit(f"nothing cached in {args.cache_dir}; run fast_lfp_benchmark.extract first")
    by_session = len(sessions) > 1
    if not by_session:
        print("only one session cached: PCA falls back to leave-one-PROBE-out, reproducibility skipped")

    tic = time.time()
    covs = {e.pid: SpatialPCA.covariance(load(e)["x_uV"])[0] for e in entries}
    print(f"channel covariances in {time.time() - tic:.0f} s")

    def fit_pcas(keep):
        cov = sum(covs[e.pid] for e in entries if keep(e))
        return [SpatialPCA(r).fit_from_covariance(cov) for r in PCA_RANKS]

    rows = []
    for i, e in enumerate(entries):
        tic = time.time()
        if by_session:
            pcas = fit_pcas(lambda o: o.eid != e.eid)
        else:
            pcas = fit_pcas(lambda o: o.pid != e.pid)
        candidates = [Raw(), ChannelBin(16), *pcas, Lfpack("default"), Lfpack("aggressive")]
        rows += score_probe(load(e), candidates)
        print(f"[{i + 1}/{len(entries)}] {e.eid[:8]} {e.pid[:8]} scored in {time.time() - tic:.0f} s")

    df = pd.DataFrame(rows)
    df.to_csv(out / "probe_metrics.csv", index=False)
    order = df.groupby("candidate")["floats_per_s"].mean().sort_values(ascending=False).index
    summary = df.drop(columns=["eid", "pid"]).groupby("candidate").mean(numeric_only=True).loc[order]
    summary.to_csv(out / "summary.csv")

    if by_session and len(sessions) >= 2:
        splits = half_splits(sessions)
        rep_rows = []
        for a, b in splits:
            pa = fit_pcas(lambda o: o.eid in a)
            pb = fit_pcas(lambda o: o.eid in b)
            for r, ua, ub in zip(PCA_RANKS, pa, pb):
                rep_rows.append({"rank": r, "similarity": M.subspace_similarity(ua.components, ub.components)})
        pca_rep = pd.DataFrame(rep_rows).groupby("rank")["similarity"].agg(["mean", "min"])
        pca_rep["random_subspace_expected"] = [r / N_CHANNELS for r in pca_rep.index]
        pca_rep.to_csv(out / "pca_reproducibility.csv")

        cand = df[df.candidate != "raw"]
        rank_rows = []
        for a, b in splits:
            for metric in RANKING_METRICS:
                ma = cand[cand.eid.isin(a)].groupby("candidate")[metric].mean()
                mb = cand[cand.eid.isin(b)].groupby("candidate")[metric].mean()
                both = pd.concat([ma, mb], axis=1).dropna()
                if len(both) >= 3:
                    rank_rows.append({"metric": metric, "spearman": spearmanr(both.iloc[:, 0], both.iloc[:, 1])[0]})
        rank_rep = pd.DataFrame(rank_rows).groupby("metric")["spearman"].agg(["mean", "min"]).loc[
            [m for m in RANKING_METRICS if m in set(r["metric"] for r in rank_rows)]
        ]
        rank_rep.to_csv(out / "ranking_reproducibility.csv")
    else:
        pca_rep = rank_rep = None

    show = ["floats_per_s", "compression_vs_native", "r2", "rmse_uV",
            "coh_delta", "coh_theta", "coh_beta", "coh_gamma", "pow_db_gamma",
            "env_corr_delta", "env_corr_gamma", "tau_ratio_gamma", "beh_retained"]
    with pd.option_context("display.width", 250, "display.max_columns", None,
                           "display.float_format", "{:.3f}".format):
        print(f"\n=== SUMMARY: mean over {len(entries)} held-out probes ===")
        print(summary[show].to_string())
        beh_cols = [c for c in summary.columns if c.startswith("beh_r2_")]
        print("\n=== BEHAVIOUR R² (cross-validated) ===")
        print(summary[beh_cols].to_string())
        if pca_rep is not None:
            print(f"\n=== PCA SUBSPACE REPRODUCIBILITY ({len(splits)} session half-splits) ===")
            print(pca_rep.to_string())
            print("\n=== DOES THE RANKING OF CANDIDATES REPLICATE ACROSS HALVES? (Spearman) ===")
            print(rank_rep.to_string())
    from .plot import main as plot_main
    plot_main(out)
    print("\nsaved to:", out)


if __name__ == "__main__":
    main()
