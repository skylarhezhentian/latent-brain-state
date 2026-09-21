"""
Every number the short deck quotes about the search, straight from run_search.py and
measure_search_costs.py outputs.

    /opt/miniconda3/envs/lfp-brain-state/bin/python -m lfp_compression_search.export_search_numbers

Writes RESULTS_DIR/search/search_numbers.json.
"""
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from lfp_compression_pilot import pilot_config as C

ROOT = Path(C.RESULTS_DIR) / "search"


def r(x, nd=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), nd)


def rows(df, key="method", nd=3):
    return {row[key]: {k: (r(v, nd) if isinstance(v, (int, float, np.floating, np.integer)) else v)
                       for k, v in row.items() if k != key} for _, row in df.iterrows()}


def collect():
    out = {}
    for s in ("search", "confirm"):
        d = ROOT / s
        pp = pd.read_csv(d / "per_probe.csv")
        paired = pd.read_csv(d / "paired.csv")
        out[s] = {
            "n_probes": int(pp.pid.nunique()), "n_sessions": int(pp.eid.nunique()), "n_mice": int(pp.subject.nunique()),
            "mice": sorted(pp.subject.unique().tolist()),
            "probe_summary": rows(pd.read_csv(d / "method_summary.csv")),
            "session_summary": rows(pd.read_csv(d / "session_summary.csv")),
            "absolute_medians": rows(pp.groupby("method")[["beh_r_mean", "shuf_r_mean", "r2_wave", "r2_env_gamma"]]
                                     .median().reset_index()),
            "probes_wave_loss_over_0.02": {m: int((g.r2_wave < -0.02).sum()) for m, g in paired.groupby("method")},
        }
    out["winner"] = json.load(open(ROOT / "search" / "winner.json"))
    out["verdict"] = json.load(open(ROOT / "confirm" / "verdict.json"))
    out["mice_overlap"] = len(set(out["search"]["mice"]) & set(out["confirm"]["mice"]))
    # Conservative behaviour gain on the confirmation set: subtract the extra shuffled-control
    # correlation the winner shows over PCA 16 (a 30 s shift may leave slow shared drift).
    w = out["winner"]["winner"]
    am = out["confirm"]["absolute_medians"]
    out["confirm_conservative_gain"] = r(out["confirm"]["probe_summary"][w]["median_d_beh_r"]
                                         - (am[w]["shuf_r_mean"] - am["waveform_pca_16"]["shuf_r_mean"]))
    out["n_methods"] = len(out["search"]["probe_summary"]) + 1          # + the PCA 16 baseline
    out["costs"] = rows(pd.read_csv(ROOT / "costs.csv"), nd=6)
    return out


def main():
    json.dump(collect(), open(ROOT / "search_numbers.json", "w"), indent=1)
    print("wrote", ROOT / "search_numbers.json")


if __name__ == "__main__":
    main()
