"""
Score the autoencoder latents with exactly the benchmark protocol (lfp_selected_benchmark.evaluate),
next to a representation that uses no PCA at all.

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 /opt/miniconda3/envs/lfp-brain-state/bin/python \
        -m lfp_autoencoder_prototype.evaluate_ae --worker        # run several in parallel
    ... -m lfp_autoencoder_prototype.evaluate_ae --aggregate

Representations scored here (all 48 numbers at 25 Hz):
  ae_depth_48            per-depth-latent autoencoder, trained on other sessions
  ae_global_48           global-latent autoencoder, trained on other sessions
  group_pca_pooled_48    group PCA 2×24 with loadings fit on other sessions (linear counterpart)
  waveform+gamma_48      no PCA: 24 group waveforms + 24 group gamma log-RMS (200 ms), hand-picked
Compared in the aggregate with benchmark rows already computed on the same insertions:
  waveform_pca_48, spatial_avg_48, group_pca_2x24 (per-insertion fit), group_wave+amp1.
"""
import argparse
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from lfp_compression_pilot.pilot_data import find_probe, load_raw
from lfp_compression_pilot.pilot_representations import Representation
from lfp_selected_benchmark.evaluate import evaluate_probe
from lfp_selected_benchmark.extract_selected import CACHE_DIR
from lfp_selected_benchmark.feature_bank import N_GROUPS
from lfp_selected_benchmark.run_selected import PIDS_CSV, RESULTS
from .train import AE_RESULTS

SUB = AE_RESULTS / "probes"
COMPARE = ["waveform_pca_48", "spatial_avg_48", "group_pca_2x24", "group_wave+amp1", "waveform_pca_16", "wave12+bank4"]


class Precomputed(Representation):
    """Latents computed by a model that never saw this session; nothing is fit here."""

    def __init__(self, name, Z):
        self.name, self.Z, self.dims = name, Z, Z.shape[0]

    def transform(self, F):
        return self.Z[:, : F.wave25.shape[1]]


class WaveformGamma(Representation):
    """No PCA anywhere: each depth keeps its mean waveform and its 200 ms gamma log-RMS."""
    name, dims = "waveform+gamma_48", 2 * N_GROUPS

    def transform(self, F):
        return np.concatenate([F.wave_groups, F.rms("gamma", 5)], axis=0)


def worker():
    SUB.mkdir(parents=True, exist_ok=True)
    for pid in sorted(pd.read_csv(PIDS_CSV).pid):
        out = SUB / f"{pid[:8]}.csv"
        lat = AE_RESULTS / "latents" / f"{pid}.npz"
        if out.exists() or not lat.exists():
            continue
        try:
            os.close(os.open(str(out) + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        except FileExistsError:
            continue
        tic = time.time()
        raw, meta, _ = load_raw(pid, CACHE_DIR)
        path, _ = find_probe(pid, CACHE_DIR)
        with np.load(path) as z:
            B = z["behavior"]
        with np.load(lat) as z:
            reps = [Precomputed(k, z[k]) for k in ("ae_depth_48", "ae_global_48", "group_pca_pooled_48")] + [WaveformGamma()]
            fold = int(z["fold"])
        df = pd.DataFrame(evaluate_probe(raw, B, meta["behavior_names"], reps))
        df.insert(0, "pid", pid)
        df.insert(1, "eid", meta["eid"])
        df.insert(2, "ae_fold", fold)
        df.to_csv(out, index=False)
        Path(str(out) + ".lock").unlink(missing_ok=True)
        print(f"[{pid[:8]}] fold {fold} done in {time.time() - tic:.0f} s", flush=True)


def aggregate():
    new = pd.concat([pd.read_csv(f) for f in sorted(SUB.glob("*.csv"))], ignore_index=True)
    bench = pd.read_csv(RESULTS / "per_probe.csv")
    beh = [c for c in new.columns if c.startswith("beh_r_") and not c.startswith("beh_r2_")]
    cols = beh + ["r2_wave", "r2_env_gamma", "r2_amp_global", "r2_amp_spatial", "r2_wave_spatial"] + [c.replace("beh_r_", "shuf_r_") for c in beh]
    pp_new = new.groupby(["pid", "eid", "method"])[cols].mean().reset_index()
    pp_new["beh_r_mean"] = pp_new[beh].mean(axis=1)
    pp_new["shuf_r_mean"] = pp_new[[c.replace("beh_r_", "shuf_r_") for c in beh]].mean(axis=1)
    pp_old = bench[bench.method.isin(COMPARE) & bench.pid.isin(pp_new.pid)][["pid", "eid", "method"] + cols + ["beh_r_mean", "shuf_r_mean"]]
    pp = pd.concat([pp_new, pp_old], ignore_index=True)
    pp.to_csv(AE_RESULTS / "per_probe_ae_compare.csv", index=False)
    order = ["waveform_pca_48", "spatial_avg_48", "waveform+gamma_48", "group_pca_2x24", "group_pca_pooled_48",
             "group_wave+amp1", "ae_global_48", "ae_depth_48", "waveform_pca_16", "wave12+bank4"]
    order = [m for m in order if m in set(pp.method)]
    summ = pp.groupby("method")[["beh_r_mean"] + beh + ["r2_wave", "r2_amp_global", "r2_amp_spatial", "shuf_r_mean"]].median().loc[order]
    summ.insert(0, "n", pp.groupby("method").pid.nunique().loc[order])
    summ.to_csv(AE_RESULTS / "summary_ae_compare.csv")
    rows = []
    for base in ("group_pca_pooled_48", "group_pca_2x24", "waveform_pca_48", "group_wave+amp1"):
        b = pp[pp.method == base].set_index("pid")
        for m in order:
            if m == base:
                continue
            g = pp[pp.method == m].set_index("pid")
            c = g.index.intersection(b.index)
            for col in ("beh_r_mean", "r2_wave", "r2_amp_spatial"):
                d = g.loc[c, col] - b.loc[c, col]
                s = d.groupby(g.loc[c, "eid"]).mean()
                rows.append({"baseline": base, "method": m, "metric": col, "median_diff": d.median(),
                             "sessions_higher": int((s > 0).sum()), "n_sessions": len(s)})
    pd.DataFrame(rows).to_csv(AE_RESULTS / "paired_ae_compare.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(summ)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.worker:
        worker()
    if a.aggregate:
        aggregate()


if __name__ == "__main__":
    main()
