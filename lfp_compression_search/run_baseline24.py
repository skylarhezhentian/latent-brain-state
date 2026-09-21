"""
Baseline requested by the mentor: spatial averaging into 24 channel groups
(384 / 16, the repository's own compression), compared with PCA at the same
24 dimensions and with PCA 16, on every cached probe (search + confirmation).

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
      /opt/miniconda3/envs/lfp-brain-state/bin/python -m lfp_compression_search.run_baseline24 [--shard 0/2]

Same protocol and code path as run_search.py. Writes
RESULTS_DIR/search/baseline24/<set>/<pid8>.csv and baseline24/per_probe.csv.
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import pandas as pd

from lfp_compression_pilot.pilot_data import list_cached_probes
from lfp_compression_pilot.pilot_representations import SpatialAverage, WaveformPCA
from .run_search import CACHES, ROOT, evaluate_probe

OUT = ROOT / "baseline24"


def methods():
    return [SpatialAverage(24), WaveformPCA(24), WaveformPCA(16)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--aggregate-only", action="store_true")
    args = ap.parse_args()
    i, n = map(int, args.shard.split("/"))
    if not args.aggregate_only:
        jobs = [(s, pid, meta) for s in ("search", "confirm") for pid, meta in list_cached_probes(CACHES[s])]
        for j, (s, pid, meta) in enumerate(jobs):
            f = OUT / s / f"{pid[:8]}.csv"
            if j % n == i and not f.exists():
                evaluate_probe(pid, meta, CACHES[s], methods(), f)
        if n > 1:
            return
    frames = []
    for s in ("search", "confirm"):
        for f in sorted((OUT / s).glob("*.csv")):
            d = pd.read_csv(f)
            d["set"] = s
            frames.append(d)
    df = pd.concat(frames, ignore_index=True)
    cols = [c for c in df.columns if c.startswith("beh_r_")] + ["r2_wave", "r2_env_gamma", "shuf_r_mean"]
    per_probe = df.groupby(["set", "pid", "eid", "subject", "method"])[cols].mean().reset_index()
    per_probe.to_csv(OUT / "per_probe.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.float_format", "{:.3f}".format):
        print(per_probe.groupby("method")[cols].median().to_string())
        print("probes:", per_probe.pid.nunique())


if __name__ == "__main__":
    main()
