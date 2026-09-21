"""
Cache the per-depth feature bank for every selected insertion, so training never
recomputes it.

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 /opt/miniconda3/envs/lfp-brain-state/bin/python \
        -m lfp_autoencoder_prototype.cache_bank [--shard 0/3]

Writes ~/Downloads/lfp-brain-state/ae_cache/<pid>.npz with
  X         [24 depths, 19 features, 2500 bins] float32: group waveform, 9 RMS, 9 PSD (feature_bank.py)
  behavior  [2500, 5] and behavior_names
  eid, regions (majority Cosmos label per depth)
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from lfp_compression_pilot.pilot_data import find_probe, load_raw
from lfp_selected_benchmark.evaluate import group_regions
from lfp_selected_benchmark.extract_selected import CACHE_DIR
from lfp_selected_benchmark.feature_bank import BankFeatures, GroupPCA
from lfp_selected_benchmark.run_selected import PIDS_CSV

AE_CACHE = Path(os.path.expanduser("~/Downloads/lfp-brain-state/ae_cache"))
FEATURE_NAMES = (["waveform"] + [f"rms_{b}_{s * 40}ms" for b, sc in
                 {"delta": (25,), "theta": (25,), "alpha": (5, 25), "beta": (5, 25), "gamma": (1, 5, 25)}.items() for s in sc]
                 + [f"psd_{lo}-{hi}Hz" for lo, hi in zip((1, 2, 4, 8, 13, 20, 30, 50), (2, 4, 8, 13, 20, 30, 50, 100))] + ["psd_slope"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    AE_CACHE.mkdir(parents=True, exist_ok=True)
    pids = sorted(pd.read_csv(PIDS_CSV).pid)
    for j, pid in enumerate(pids):
        out = AE_CACHE / f"{pid}.npz"
        if j % n != i or out.exists():
            continue
        raw, meta, cosmos = load_raw(pid, CACHE_DIR)
        path, _ = find_probe(pid, CACHE_DIR)
        with np.load(path) as z:
            B = z["behavior"]
        X = GroupPCA(1)._per_group(BankFeatures(raw)).astype(np.float32)
        np.savez(out, X=X, behavior=B[: X.shape[2]], behavior_names=np.array(meta["behavior_names"]),
                 eid=meta["eid"], regions=np.array(group_regions(cosmos)), feature_names=np.array(FEATURE_NAMES))
        print(f"[{pid[:8]}] X {X.shape}", flush=True)


if __name__ == "__main__":
    main()
