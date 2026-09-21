"""
Cache a channel subset of the raw 250 Hz LFP for the filterbank autoencoder (option E), which
reads the signal itself instead of the hand-made feature bank.

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 /opt/miniconda3/envs/lfp-brain-state/bin/python \
        -m lfp_autoencoder_prototype.cache_raw [--shard 0/2]

Writes ~/Downloads/lfp-brain-state/ae_cache_raw/<pid>.npz with
  raw   [96, 25000 + 2 * PAD] float32: 4 channels per depth group (offsets 2, 6, 10, 14 of each
        16-channel group), divided by one scalar per insertion (the SD over these channels and
        all time), reflect-padded by PAD samples at both ends so every training window has context
  scale the scalar, in µV
"""
import argparse
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from lfp_compression_pilot.pilot_data import load_raw
from lfp_selected_benchmark.extract_selected import CACHE_DIR
from lfp_selected_benchmark.feature_bank import GROUP, N_GROUPS
from lfp_selected_benchmark.run_selected import PIDS_CSV

AE_CACHE_RAW = Path(os.path.expanduser("~/Downloads/lfp-brain-state/ae_cache_raw"))
OFFSETS = (2, 6, 10, 14)
PAD = 250                     # 1 s at 250 Hz
N_SAMPLES = 25000             # 2500 bins x 10 samples


def channels():
    return np.array([g * GROUP + o for g in range(N_GROUPS) for o in OFFSETS])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    i, n = map(int, a.shard.split("/"))
    AE_CACHE_RAW.mkdir(parents=True, exist_ok=True)
    for j, pid in enumerate(sorted(pd.read_csv(PIDS_CSV).pid)):
        out = AE_CACHE_RAW / f"{pid}.npz"
        if j % n != i or out.exists():
            continue
        raw, _, _ = load_raw(pid, CACHE_DIR)
        x = raw[channels(), :N_SAMPLES].astype(np.float64)
        assert x.shape[1] == N_SAMPLES, x.shape
        scale = float(x.std())
        x = np.pad(x / scale, ((0, 0), (PAD, PAD)), mode="reflect").astype(np.float32)
        np.savez(out, raw=x, scale=scale)
        print(f"[{pid[:8]}] raw {x.shape} scale {scale:.1f} uV", flush=True)


if __name__ == "__main__":
    main()
