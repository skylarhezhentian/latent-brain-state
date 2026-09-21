"""
Runtime, memory and storage of the search winner, with the same accounting as
lfp_compression_pilot/measure_costs.py (pilot probe, shared 250 Hz signal, 100 s;
encode = everything needed to produce Z from that signal; fit on 75 s; median of 3;
tracemalloc peak; numpy.savez_compressed float32 storage).

    /opt/miniconda3/envs/lfp-brain-state/bin/python -m lfp_compression_search.measure_search_costs

Writes RESULTS_DIR/search/costs.csv.
"""
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from lfp_compression_pilot import pilot_config as C
from lfp_compression_pilot.measure_costs import median_traced, npz_bytes
from lfp_compression_pilot.pilot_data import load_raw
from .search_methods import SearchFeatures, stage1_methods

ROOT = Path(C.RESULTS_DIR) / "search"


def main():
    raw, _, _ = load_raw(C.PILOT_PID)
    duration = raw.shape[1] / C.FS
    train = np.arange(int(0.75 * (raw.shape[1] // C.BIN)))
    winner = json.load(open(ROOT / "search" / "winner.json"))["winner"]
    registry = {m.name: m for m in stage1_methods()}
    rows = [{"method": "raw 250 Hz (reference)", "storage_bytes_per_s": npz_bytes(x=raw) / duration}]
    for name in ("waveform_pca_16", "wave12+amp4", winner):
        rep = registry[name]
        # Fresh features inside each call, so envelope computation is charged to encode and fit.
        _, t_fit, m_fit = median_traced(lambda: rep.fit(SearchFeatures(raw), train))
        Z, t_enc, m_enc = median_traced(lambda: rep.transform(SearchFeatures(raw)))
        rows.append({"method": name, "fit_s": t_fit, "fit_peak_traced_mb": m_fit,
                     "encode_s_per_s": t_enc / duration, "encode_peak_traced_mb": m_enc,
                     "storage_bytes_per_s": npz_bytes(Z=Z) / duration})
    df = pd.DataFrame(rows)
    df["compression_vs_raw_250hz_npz"] = df.storage_bytes_per_s.iloc[0] / df.storage_bytes_per_s
    df.to_csv(ROOT / "costs.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
