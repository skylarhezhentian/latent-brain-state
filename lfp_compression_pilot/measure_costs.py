"""
Runtime, memory and storage of every method on the pilot probe, with one
accounting rule for all of them.

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 \
      /opt/miniconda3/envs/lfp-brain-state/bin/python -m lfp_compression_pilot.measure_costs

Everything starts from the same shared 250 Hz signal (100 s, 384 channels).
- Representations, encode = everything needed to produce Z from that signal:
  40 ms binning, band-pass + RMS envelopes (amplitude methods only), projection.
  Fit = learning the bases on 75 s. Median of 3 repeats.
- lfpack, encode = Cadzow (when used) + SVD/wavelet compression, measured once in
  pilot_lfpack.py on the same 100 s (decoded arrays are cached, so it is read back).
- Memory = peak of Python/NumPy allocations traced by tracemalloc during the call.
- Storage = numpy.savez_compressed float32 of what a method must keep, per second.

Writes RESULTS_DIR/pilot_<pid8>/costs.csv. Single thread, Intel i5 laptop.
"""
import io
import json
import os
import time
import tracemalloc
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from . import pilot_config as C
from .pilot_data import bin_mean, load_raw
from .pilot_representations import AmplitudePCA, Features, SpatialAverage, build

REPEATS = 3


def traced(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    return out, dt, peak


def median_traced(fn):
    runs = [traced(fn) for _ in range(REPEATS)]
    return runs[-1][0], float(np.median([r[1] for r in runs])), float(max(r[2] for r in runs))


def npz_bytes(**arrays):
    buf = io.BytesIO()
    np.savez_compressed(buf, **{k: np.asarray(v, dtype=np.float32) for k, v in arrays.items()})
    return buf.getbuffer().nbytes


def main():
    raw, meta, _ = load_raw(C.PILOT_PID)
    out = Path(C.RESULTS_DIR) / f"pilot_{C.PILOT_PID[:8]}"
    duration = raw.shape[1] / C.FS
    n_bins = raw.shape[1] // C.BIN
    train = np.arange(int(0.75 * n_bins))
    F_full = Features.from_source(raw)
    rows = [{"method": "raw 250 Hz (reference)", "storage_bytes_per_s": npz_bytes(x=raw) / duration,
             "keeps": "full signal, 250 Hz"}]

    methods = [SpatialAverage(16)] + [build(n, k, m) for n, (k, m) in C.PRIMARY_METHODS.items()]
    for rep in methods:
        amplitude = isinstance(rep, AmplitudePCA)

        def features():
            # What this method needs from the 250 Hz signal before projecting.
            return Features.from_source(raw) if amplitude else Features(raw, bin_mean(raw), None)

        _, t_fit, m_fit = median_traced(lambda: rep.fit(features(), train))
        Z, t_enc, m_enc = median_traced(lambda: rep.transform(features()))
        params = {}
        if hasattr(rep, "W"):
            params.update(W=rep.W, mu=rep.mu)
        if hasattr(rep, "wave"):
            params.update(W=rep.wave.W, mu=rep.wave.mu, We=rep.We, e_mu=rep.e_mu, e_sd=rep.e_sd)
        rows.append({"method": rep.name, "fit_s": t_fit, "fit_peak_traced_mb": m_fit,
                     "encode_s_per_s": t_enc / duration, "encode_peak_traced_mb": m_enc,
                     "storage_bytes_per_s": npz_bytes(Z=Z) / duration,
                     "model_bytes": npz_bytes(**params) if params else 0, "keeps": f"{rep.dims} dims, 25 Hz"})

    info = json.load(open(Path(C.PILOT_CACHE_DIR) / f"lfpack250_{C.PILOT_PID}.json"))
    for name, v in info.items():
        rows.append({"method": name, "encode_s_per_s": v["encode_s_per_s"] + v["cadzow_s_per_s"],
                     "encode_peak_traced_mb": max(v["encode_peak_traced_mb"], v["cadzow_peak_traced_mb"]),
                     "decode_s_per_s": v["decode_s_per_s"], "storage_bytes_per_s": v["storage_bytes_per_s"],
                     "model_bytes": 0, "keeps": "384 ch, 250 Hz"})

    df = pd.DataFrame(rows)
    ref = df.loc[0, "storage_bytes_per_s"]
    df["compression_vs_raw_250hz_npz"] = ref / df["storage_bytes_per_s"]
    df.to_csv(out / "costs.csv", index=False)
    with pd.option_context("display.width", 220, "display.max_columns", None, "display.float_format", "{:.4g}".format):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
