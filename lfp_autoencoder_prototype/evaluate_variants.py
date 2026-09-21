"""
Score options C, D, E and the view-weighted linear twin with exactly the benchmark protocol
(lfp_selected_benchmark.evaluate.evaluate_probe: 4 contiguous folds with 1 s gaps, ridge readouts,
±320 ms behaviour lags, 30 s shift control), as evaluate_ae.py does for A and B.

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 /opt/miniconda3/envs/lfp-brain-state/bin/python \
        -m lfp_autoencoder_prototype.evaluate_variants --worker [--tags C D W E]   # several in parallel

  C  ae_masked_48            D  ae_anatomy_48            E  ae_filterbank_48
  F  ae_waveamp_48           (stored waveform + 1 learned amplitude latent per depth)
  W  group_pca_weighted_48   (purpose_tests.py --weighted): linear twin of A-E
  WF group_wave+amp1_pooled  (purpose_tests.py --weighted): linear twin of F

Writes ~/Downloads/lfp-brain-state/ae_results/variants/probes_<tag>/<pid[:8]>.csv.
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
from lfp_selected_benchmark.evaluate import evaluate_probe
from lfp_selected_benchmark.extract_selected import CACHE_DIR
from lfp_selected_benchmark.run_selected import PIDS_CSV
from .evaluate_ae import Precomputed
from .purpose_tests import BASE2
from .train import AE_RESULTS

V = AE_RESULTS / "variants"
SOURCES = {"C": (V / "latents_C", "ae_masked_48"), "D": (V / "latents_D", "ae_anatomy_48"),
           "E": (V / "latents_E", "ae_filterbank_48"), "F": (V / "latents_F", "ae_waveamp_48"),
           "W": (BASE2, "group_pca_weighted_48"), "WF": (BASE2, "group_wave+amp1_pooled")}


def worker(tags):
    for tag in tags:
        lat_dir, key = SOURCES[tag]
        sub = V / f"probes_{tag}"
        sub.mkdir(parents=True, exist_ok=True)
        for pid in sorted(pd.read_csv(PIDS_CSV).pid):
            out, lat = sub / f"{pid[:8]}.csv", lat_dir / f"{pid}.npz"
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
                reps, fold = [Precomputed(key, z[key])], int(z["fold"])
            df = pd.DataFrame(evaluate_probe(raw, B, meta["behavior_names"], reps))
            df.insert(0, "pid", pid)
            df.insert(1, "eid", meta["eid"])
            df.insert(2, "ae_fold", fold)
            df.to_csv(out, index=False)
            Path(str(out) + ".lock").unlink(missing_ok=True)
            print(f"[{tag} {pid[:8]}] fold {fold} done in {time.time() - tic:.0f} s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--tags", nargs="+", default=["W", "C", "D", "E"])
    a = ap.parse_args()
    if a.worker:
        worker(a.tags)


if __name__ == "__main__":
    main()
