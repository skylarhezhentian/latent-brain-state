"""
Which brain regions carry which behaviour? A spatial read-out that only a
location-preserving representation allows.

For each insertion, the 19 per-depth features (waveform, 9 RMS, 9 PSD) are averaged over
the depth groups whose majority Cosmos label is a given region (regions with >= 2 groups),
so every region gets the same 19 features whatever its extent. Each behaviour is then
decoded from one region alone, with the same protocol as evaluate.py.

    /opt/miniconda3/envs/lfp-brain-state/bin/python -m lfp_selected_benchmark.regions --worker [--wait]
    /opt/miniconda3/envs/lfp-brain-state/bin/python -m lfp_selected_benchmark.regions --aggregate
Writes selected_benchmark_results/probes_regions/<pid8>.csv, regions_by_behaviour.csv, figures/fig_regions.png.
"""
import argparse
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
from .evaluate import group_regions, ridge_predict
from .extract_selected import CACHE_DIR
from .feature_bank import BankFeatures, GroupPCA, N_GROUPS
from .run_selected import PIDS_CSV, RESULTS

SUB = "probes_regions"
SKIP = {"void", "root", "NA", "nan"}


def region_rows(raw, B, names, cosmos):
    F = BankFeatures(raw)
    per_group = GroupPCA(1)._per_group(F)                          # [24, 19, T]
    labels = np.array(group_regions(cosmos))
    n = per_group.shape[2]
    B = B[:n]
    rows = []
    regions = [(r, np.where(labels == r)[0]) for r in dict.fromkeys(labels) if r not in SKIP]
    regions = [(r, idx) for r, idx in regions if len(idx) >= 2] + [("whole probe", np.arange(N_GROUPS))]
    for region, idx in regions:
        Z = per_group[idx].mean(axis=0)                             # [19, T]
        for fold, (train, test) in enumerate(outer_folds(n)):
            mu = Z[:, train].mean(axis=1, keepdims=True)
            sd = Z[:, train].std(axis=1, keepdims=True)
            Zs = (Z - mu) / np.where(sd > 1e-6, sd, 1.0)
            P = ridge_predict(lagged(Zs), B.T, train, test)
            row = {"region": region, "n_groups": len(idx), "fold": fold}
            for j, bn in enumerate(names):
                y, p = B[test, j], P[:, j]
                ok = np.isfinite(y) & np.isfinite(p)
                good = ok.sum() > 50 and np.std(y[ok]) > 0 and np.std(p[ok]) > 0
                row[f"beh_r_{bn}"] = float(np.corrcoef(y[ok], p[ok])[0, 1]) if good else np.nan
            rows.append(row)
    return rows


def worker(wait):
    sel = pd.read_csv(PIDS_CSV).set_index("pid")
    outdir = RESULTS / SUB
    outdir.mkdir(parents=True, exist_ok=True)
    while True:
        pending = [p for p in sel.index if not (outdir / f"{p[:8]}.csv").exists()]
        if not pending:
            return
        ran = False
        for pid in pending:
            try:
                path, _ = find_probe(pid, CACHE_DIR)
            except FileNotFoundError:
                continue
            out = outdir / f"{pid[:8]}.csv"
            try:
                fd = os.open(str(out) + ".lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.close(fd)
            except FileExistsError:
                continue
            tic = time.time()
            raw, meta, cosmos = load_raw(pid, CACHE_DIR)
            with np.load(path) as z:
                B = z["behavior"]
            df = pd.DataFrame(region_rows(raw, B, meta["behavior_names"], cosmos))
            df.insert(0, "pid", pid)
            df.insert(1, "eid", meta["eid"])
            df.to_csv(out, index=False)
            Path(str(out) + ".lock").unlink(missing_ok=True)
            print(f"[{pid[:8]}] regions {sorted(set(df.region))} done in {time.time() - tic:.0f} s", flush=True)
            ran = True
        if not ran:
            if not wait:
                return
            time.sleep(60)


def aggregate():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    df = pd.concat([pd.read_csv(f) for f in sorted((RESULTS / SUB).glob("*.csv"))], ignore_index=True)
    beh = [c for c in df.columns if c.startswith("beh_r_")]
    pp = df.groupby(["pid", "eid", "region"])[beh].mean().reset_index()
    pp["beh_r_mean"] = pp[beh].mean(axis=1)
    cols = beh + ["beh_r_mean"]
    med = pp.groupby("region")[cols].median()
    n = pp.groupby("region").pid.nunique()
    tab = med.assign(n_insertions=n).sort_values("beh_r_mean", ascending=False)
    tab.to_csv(RESULTS / "regions_by_behaviour.csv")
    keep = tab[(tab.n_insertions >= 5) | (tab.index == "whole probe")]
    order = ["whole probe"] + [r for r in keep.index if r != "whole probe"]
    M = keep.loc[order, cols]
    ramp = LinearSegmentedColormap.from_list("blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#0d366b"])
    fig, ax = plt.subplots(figsize=(9.5, 0.5 * len(order) + 1.9), facecolor="white")
    ax.imshow(M.to_numpy(), cmap=ramp, vmin=0.1, vmax=0.7, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=10.5, color="white" if v > 0.5 else "#0b0b0b",
                        fontweight="bold" if j == M.shape[1] - 1 else "normal")
    nice = {"beh_r_wheel_speed": "Wheel\nspeed", "beh_r_whisker_me_left": "Whisker\nleft cam", "beh_r_whisker_me_right": "Whisker\nright cam",
            "beh_r_body_me": "Body\nmotion", "beh_r_pupil_diameter": "Pupil\ndiameter", "beh_r_mean": "Mean"}
    ax.set_xticks(range(len(cols)), [nice.get(c, c) for c in cols], fontsize=9.5)
    ax.set_yticks(range(len(order)), [f"{r}  (n={int(keep.loc[r, 'n_insertions'])})" for r in order], fontsize=10.5)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.plot([-0.5, M.shape[1] - 0.5], [0.5, 0.5], color="#0b0b0b", linewidth=1.5)
    fig.text(0.01, 0.01, "Median held-out r when behaviour is decoded from one Cosmos region's 19 averaged features "
             "(waveform, RMS, PSD).\nRegions present on ≥ 5 insertions. Same folds and readout as the main benchmark.",
             fontsize=9, color="#52514e")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    (RESULTS / "figures").mkdir(exist_ok=True)
    fig.savefig(RESULTS / "figures" / "fig_regions.png", dpi=200, facecolor="white")
    with pd.option_context("display.width", 200, "display.float_format", "{:.3f}".format):
        print(tab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--wait", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.worker:
        worker(a.wait)
    if a.aggregate:
        aggregate()


if __name__ == "__main__":
    main()
