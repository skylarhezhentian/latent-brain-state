"""
Controls for the decoding numbers: is a held-out r of 0.6-0.7 real, and where does it come from?

Alon (15 Sep): "the correlation coefficient you get between some representations and behavior
sometimes reach ~0.7 - this is higher than any correlation we have seen before, and I would like
to be cautious before reporting these numbers". Each check below removes one possible explanation.

  reimpl    same features and folds, but the readout is sklearn RidgeCV instead of our own ridge.
            A bug in our readout or in the r computation would show up as a disagreement.
  align     cross-correlation between probe-wide gamma amplitude and each behaviour over +/-2 s.
            A misaligned clock would put the peak far from zero lag.
  lags      +/-320 ms of context (default) vs causal-only vs zero lag: how much of r is context?
  gap       1 s (default) vs 5 s excluded around each test block: is anything leaking across folds?
  swap      behaviour taken from other sessions (3 partners), and the same behaviour circularly
            shifted by 30 s. Both are nulls: whatever they reach is the floor, not zero, because
            every IBL session runs the same task and behaviour is strongly autocorrelated.
  outside   decode from depth groups whose Cosmos label is void/root (outside the brain) and from
            three matched in-brain subsets (adjacent to the void, deepest, and middle). Movement or
            muscle artefact reaches the electrodes outside the brain too.
  alonlike  probe-wide RMS envelope only, zero lag: the setting closest to the r = 0.46 on slide 24.

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
    PY=/opt/miniconda3/envs/lfp-brain-state/bin/python
    $PY -m lfp_selected_benchmark.controls --worker      # start one per core
    $PY -m lfp_selected_benchmark.controls --aggregate

Outputs in ~/Downloads/lfp-brain-state/selected_benchmark_results/:
  probes_controls/<pid8>.csv   one row per (check, rep, variant, behaviour, fold)
  controls_summary.csv         median r over insertions per (check, rep, variant, behaviour)
  controls_align.csv           peak cross-correlation lag per insertion and behaviour
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
from lfp_compression_pilot.pilot_data import find_probe, load_raw
from lfp_compression_pilot.pilot_metrics import lagged, outer_folds
from lfp_compression_pilot.pilot_representations import Representation
from .evaluate import group_regions, ridge_predict
from .extract_selected import CACHE_DIR
from .feature_bank import Bank, BankFeatures, N_GROUPS, WaveBank, _zscore_fit
from .run_selected import PIDS_CSV, RESULTS, claim

CAUSAL_LAGS = (-8, -4, 0)          # -320, -160, 0 ms
ZERO_LAG = (0,)
GAP_5S = 125                       # bins
# Only 'void' is outside the brain. In the Cosmos mapping 'root' collects fiber tracts (191 regions),
# ventricles (12) and parent-level labels, i.e. white matter INSIDE the brain. Runs before 20 Sep used
# {"void", "root"}, so their "outside" rows mostly compare white vs grey matter; superseded by
# neural_validity.py, which separates grey matter, white matter and void.
OUTSIDE_LABELS = {"void"}
ALIGN_MAX_LAG = 50                 # +/- 2 s in 40 ms bins


class GroupSubset(Representation):
    """The feature bank restricted to a subset of depth groups (used for the in/outside check)."""

    def __init__(self, groups, name, views=("wave", "rms", "psd")):
        self.groups = list(groups)
        self.views = tuple(views)
        self.name = name
        self.dims = len(self.groups) * (1 * ("wave" in views) + 9 * ("rms" in views) + 9 * ("psd" in views))
        self.n_params = 2 * self.dims

    def _stack(self, F):
        blocks = []
        if "wave" in self.views:
            blocks.append(F.wave_groups[self.groups])
        if "rms" in self.views:
            # rms_block is feature-major: row = feature * N_GROUPS + group
            rows = [f * N_GROUPS + g for f in range(9) for g in self.groups]
            blocks.append(F.rms_block[rows])
        if "psd" in self.views:
            # psd_block is group-major: row = group * 9 + bin
            rows = [g * 9 + k for g in self.groups for k in range(9)]
            blocks.append(F.psd_block[rows])
        return np.concatenate(blocks, axis=0)

    def fit(self, F, train):
        self.mu, self.sd = _zscore_fit(self._stack(F), train)
        return self

    def transform(self, F):
        return (self._stack(F) - self.mu) / self.sd


class ProbeRMS(Representation):
    """Probe-wide RMS envelope only: the 9 log-RMS features averaged over all depths."""

    name, dims = "probe_rms_envelope", 9

    def __init__(self):
        self.n_params = 2 * self.dims

    def _stack(self, F):
        return F.rms_block.reshape(9, N_GROUPS, -1).mean(axis=1)

    def fit(self, F, train):
        self.mu, self.sd = _zscore_fit(self._stack(F), train)
        return self

    def transform(self, F):
        return (self._stack(F) - self.mu) / self.sd


def decode_rows(F, B, names, rep, lags, gap, check, variant, sklearn_too=False):
    """Held-out Pearson r per behaviour and fold for one (representation, lags, gap) setting."""
    n = B.shape[0]
    rows = []
    for fold, (train, test) in enumerate(outer_folds(n, gap=gap)):
        rep.fit(F, train)
        Z = lagged(rep.transform(F), lags)
        P = ridge_predict(Z, B.T, train, test)
        for j, bn in enumerate(names):
            y, p = B[test, j], P[:, j]
            ok = np.isfinite(y) & np.isfinite(p)
            r = float(np.corrcoef(y[ok], p[ok])[0, 1]) if ok.sum() > 50 and np.std(y[ok]) > 0 and np.std(p[ok]) > 0 else np.nan
            rows.append({"check": check, "rep": rep.name, "variant": variant, "readout": "ours",
                         "behaviour": bn, "fold": fold, "r": r, "dims": rep.dims})
        if sklearn_too:
            from sklearn.linear_model import RidgeCV
            X = Z.T.astype(np.float64)
            for j, bn in enumerate(names):
                y = B[:, j].astype(np.float64)
                ok = np.isfinite(y)
                tr, te = train[ok[train]], test[ok[test]]
                if len(tr) < 100 or len(te) < 50:
                    continue
                mu, sd = X[tr].mean(0), X[tr].std(0)
                sd = np.where(sd > 1e-9, sd, 1.0)
                m = RidgeCV(alphas=np.logspace(-2, 6, 9)).fit((X[tr] - mu) / sd, y[tr])
                p = m.predict((X[te] - mu) / sd)
                r = float(np.corrcoef(y[te], p)[0, 1]) if np.std(p) > 0 else np.nan
                rows.append({"check": check, "rep": rep.name, "variant": variant, "readout": "sklearn",
                             "behaviour": bn, "fold": fold, "r": r, "dims": rep.dims})
    return rows


def align_rows(F, B, names):
    """Cross-correlation of probe-wide 200 ms gamma log-RMS with each behaviour, +/-2 s."""
    amp = F.rms("gamma", 5).mean(axis=0)
    n = min(len(amp), B.shape[0])
    a = amp[:n] - amp[:n].mean()
    rows = []
    for j, bn in enumerate(names):
        y = B[:n, j].astype(float)
        ok = np.isfinite(y)
        if ok.sum() < 200 or np.std(y[ok]) == 0:
            continue
        yz = np.where(ok, y - y[ok].mean(), 0.0)
        best_lag, best_r = 0, -2.0
        for lag in range(-ALIGN_MAX_LAG, ALIGN_MAX_LAG + 1):
            # positive lag = behaviour follows the LFP amplitude
            s = slice(max(0, lag), n + min(0, lag))
            t = slice(max(0, -lag), n - max(0, lag))
            m = ok[s]
            if m.sum() < 200:
                continue
            r = float(np.corrcoef(a[t][m], yz[s][m])[0, 1])
            if np.isfinite(r) and r > best_r:
                best_lag, best_r = lag, r
        rows.append({"check": "align", "behaviour": bn, "peak_lag_ms": best_lag * 40, "peak_r": best_r})
    return rows


def run_one(pid, row, out, partner_pids):
    tic = time.time()
    raw, meta, cosmos = load_raw(pid, CACHE_DIR)
    path, _ = find_probe(pid, CACHE_DIR)
    with np.load(path) as z:
        B = z["behavior"]
    names = meta["behavior_names"]
    F = BankFeatures(raw)
    n = F.wave25.shape[1]
    B = B[:n].astype(float)

    rows = align_rows(F, B, names)
    regions = group_regions(cosmos)
    small, full = WaveBank(12, 4), Bank(("wave", "rms", "psd"))

    for rep, sk in ((small, True), (full, False)):
        rows += decode_rows(F, B, names, rep, C.BEHAVIOR_LAGS, C.GAP_BINS, "lags", "acausal_320ms", sklearn_too=sk)
        rows += decode_rows(F, B, names, rep, CAUSAL_LAGS, C.GAP_BINS, "lags", "causal_320ms")
        rows += decode_rows(F, B, names, rep, ZERO_LAG, C.GAP_BINS, "lags", "zero_lag")
        rows += decode_rows(F, B, names, rep, C.BEHAVIOR_LAGS, GAP_5S, "gap", "gap_5s")

    # nulls. Behaviour from other sessions (the full bank only for the first partner, it is the
    # expensive one), and the same behaviour circularly shifted by 30 s.
    for i, pp in enumerate(partner_pids):
        ppath, _ = find_probe(pp, CACHE_DIR)
        with np.load(ppath) as z:
            Bp = z["behavior"].astype(float)
        if Bp.shape[0] < n:
            continue
        rows += decode_rows(F, Bp[:n], names, small, C.BEHAVIOR_LAGS, C.GAP_BINS, "swap", f"other_session_{i}")
        if i == 0:
            rows += decode_rows(F, Bp[:n], names, full, C.BEHAVIOR_LAGS, C.GAP_BINS, "swap", f"other_session_{i}")
    B_shift = np.roll(B, int(C.SHUFFLE_SHIFT_S * C.FS_REP), axis=0)
    rows += decode_rows(F, B_shift, names, small, C.BEHAVIOR_LAGS, C.GAP_BINS, "swap", "circ_shift_30s")
    rows += decode_rows(F, B_shift, names, full, C.BEHAVIOR_LAGS, C.GAP_BINS, "swap", "circ_shift_30s")

    # outside the brain vs three matched in-brain subsets of the same size
    outside = [g for g, lab in enumerate(regions) if lab.lower() in OUTSIDE_LABELS]
    inside = [g for g, lab in enumerate(regions) if lab.lower() not in OUTSIDE_LABELS]
    if 1 <= len(outside) <= len(inside) // 2:
        k = len(outside)
        mid = len(inside) // 2
        subsets = {"outside_brain": outside,              # groups labelled void/root
                   "in_brain_adjacent": inside[-k:],      # in-brain groups closest to the surface
                   "in_brain_deep": inside[:k],           # at the probe tip
                   "in_brain_middle": inside[mid - k // 2: mid - k // 2 + k]}
        for variant, groups in subsets.items():
            rows += decode_rows(F, B, names, GroupSubset(groups, f"{variant}_{k}g"),
                                C.BEHAVIOR_LAGS, C.GAP_BINS, "outside", variant)

    rows += decode_rows(F, B, names, ProbeRMS(), ZERO_LAG, C.GAP_BINS, "alonlike", "probe_rms_zero_lag")
    rows += decode_rows(F, B, names, ProbeRMS(), C.BEHAVIOR_LAGS, C.GAP_BINS, "alonlike", "probe_rms_lagged")

    df = pd.DataFrame(rows)
    df.insert(0, "pid", pid)
    df.insert(1, "eid", meta["eid"])
    df.insert(2, "lab", row.lab)
    df["n_outside_groups"] = len(outside)
    df["regions"] = json.dumps(regions)
    df.to_csv(out, index=False)
    Path(str(out) + ".lock").unlink(missing_ok=True)
    print(f"[{pid[:8]}] {row.lab} controls in {time.time() - tic:.0f} s", flush=True)


N_PARTNERS = 3


def partners(sel, n=N_PARTNERS):
    """Pair every insertion with n insertions from other sessions, for the swap control."""
    pids, eids = list(sel.index), list(sel.eid)
    out = {}
    for i, pid in enumerate(pids):
        picks = []
        for k in range(1, len(pids)):
            j = (i + k * 7) % len(pids)          # stride 7 so the partners are not neighbours
            if eids[j] != eids[i] and pids[j] not in picks:
                picks.append(pids[j])
            if len(picks) == n:
                break
        out[pid] = picks
    return out


def worker():
    sel = pd.read_csv(PIDS_CSV).set_index("pid")
    pair = partners(sel)
    for pid, row in sel.iterrows():
        if not (Path(CACHE_DIR) / row.eid / f"{pid}.npz").exists():
            continue
        out = claim(pid, "probes_controls")
        if out is not None:
            run_one(pid, row, out, pair[pid])


def aggregate():
    files = sorted((RESULTS / "probes_controls").glob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"{df.pid.nunique()} insertions")

    al = df[df.check == "align"]
    al.to_csv(RESULTS / "controls_align.csv", index=False)

    dec = df[df.check != "align"].copy()
    per_probe = dec.groupby(["pid", "check", "rep", "variant", "readout", "behaviour"]).r.mean().reset_index()
    wide = per_probe.groupby(["check", "rep", "variant", "readout", "behaviour"]).r.median().unstack("behaviour")
    # The benchmark's headline convention: mean over behaviours per insertion, then median over
    # insertions. (The mean of per-behaviour medians differs by up to 0.03 and is kept for reference.)
    per_ins = per_probe.groupby(["pid", "check", "rep", "variant", "readout"]).r.mean().reset_index()
    wide["median_of_insertion_means"] = per_ins.groupby(["check", "rep", "variant", "readout"]).r.median()
    wide["mean_of_behaviour_medians"] = wide[[c for c in wide.columns if c != "median_of_insertion_means"]].mean(axis=1)
    wide["n_insertions"] = per_probe.groupby(["check", "rep", "variant", "readout", "behaviour"]).pid.nunique().unstack("behaviour").min(axis=1)
    wide.to_csv(RESULTS / "controls_summary.csv")
    with pd.option_context("display.width", 220, "display.float_format", "{:+.3f}".format):
        print(wide.round(3).to_string())
    print("\npeak cross-correlation lag (ms), median over insertions:")
    print(al.groupby("behaviour").agg(median_lag_ms=("peak_lag_ms", "median"), median_peak_r=("peak_r", "median"),
                                      frac_within_200ms=("peak_lag_ms", lambda s: float((s.abs() <= 200).mean()))).round(3).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.worker:
        worker()
    if a.aggregate:
        aggregate()
