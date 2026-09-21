"""
Is the behavioural information in the representation neural?

Current source density (CSD) is the second spatial difference of the signed potential,
CSD_g = phi_(g-1) - 2 phi_g + phi_(g+1). It is ~0 wherever no current enters or leaves the
tissue locally, so it removes far-field signal - volume-conducted activity from elsewhere in the
brain and muscle activity alike. Outside the brain there are no local sources at all.

  Prediction if the behavioural signal is local and neural:
    in-brain CSD band power decodes behaviour; out-of-brain CSD band power does not.
  Per band, the out-of-brain potential distinguishes the two far-field explanations:
    muscle (EMG) is broadband and strongest at high frequencies (gamma);
    volume-conducted brain activity follows the 1/f spectrum (strongest at low frequencies).

Group potential = mean of 16 channels (320 um); CSD uses neighbouring groups, so it is computed
only for groups whose two neighbours are on the same side of the brain surface.
Band amplitude = log10 RMS of the band-passed signal, 200 ms smoothing, one value per 40 ms bin.
Decoding = the benchmark's protocol (4 blocked folds, 1 s gaps, +/-320 ms lags, ridge).

    $PY -m lfp_selected_benchmark.neural_validity --worker     # one per free core
    $PY -m lfp_selected_benchmark.neural_validity --aggregate
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
import scipy.signal
from scipy.ndimage import uniform_filter1d

from lfp_compression_pilot import pilot_config as C
from lfp_compression_pilot.pilot_data import find_probe, load_raw
from lfp_compression_pilot.pilot_representations import Representation
from .controls import decode_rows
from .evaluate import group_regions
from .extract_selected import CACHE_DIR
from .feature_bank import GROUP, N_GROUPS, _zscore_fit
from .run_selected import PIDS_CSV, RESULTS, claim

BANDS = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 12), "beta": (15, 30), "gamma": (30, 90)}
SMOOTH_BINS = 5


class Fixed(Representation):
    """A precomputed feature matrix [p, T25], z-scored on training bins."""

    def __init__(self, M, name):
        self.M, self.name, self.dims = M, name, M.shape[0]
        self.n_params = 2 * self.dims

    def fit(self, F, train):
        self.mu, self.sd = _zscore_fit(self.M, train)
        return self

    def transform(self, F):
        return (self.M - self.mu) / self.sd


def band_amplitude(sig, band):
    """log10 RMS of the band-passed signal, 40 ms bins smoothed over 200 ms: [groups, T25]."""
    sos = scipy.signal.butter(4, BANDS[band], btype="bandpass", fs=C.FS, output="sos")
    xf = scipy.signal.sosfiltfilt(sos, sig, axis=1)
    g, t = xf.shape
    p = (xf[:, : t - t % C.BIN].reshape(g, -1, C.BIN) ** 2).mean(axis=2)
    p = uniform_filter1d(p, size=SMOOTH_BINS, axis=1, mode="nearest")
    return np.log10(np.sqrt(p) + C.LOG_EPS_UV)


def run_one(pid, row, out):
    tic = time.time()
    raw, meta, cosmos = load_raw(pid, CACHE_DIR)
    path, _ = find_probe(pid, CACHE_DIR)
    with np.load(path) as z:
        B = z["behavior"].astype(float)
    names = meta["behavior_names"]
    x = raw.astype(np.float64)
    phi = x[: N_GROUPS * GROUP].reshape(N_GROUPS, GROUP, -1).mean(axis=1)          # group potentials
    csd = np.full_like(phi, np.nan)
    csd[1:-1] = phi[:-2] - 2 * phi[1:-1] + phi[2:]                                  # second spatial difference
    regions = group_regions(cosmos)
    # grey matter = a named Cosmos region; white = 'root' (fiber tracts, ventricles); void = outside the brain
    cls = np.array(["void" if r.lower() == "void" else "white" if r.lower() == "root" else "grey" for r in regions])
    interior = lambda c: [g for g in range(1, N_GROUPS - 1) if all(cls[g - 1:g + 2] == c)]
    grey, white, void = interior("grey"), interior("white"), interior("void")

    def grey_matched(k):
        start = max(0, len(grey) // 2 - k // 2)
        return grey[start:start + k]

    amp = {s: {b: band_amplitude(np.nan_to_num(sig), b) for b in BANDS} for s, sig in (("potential", phi), ("csd", csd))}
    n = amp["potential"]["delta"].shape[1]
    B = B[:n]

    sets = {"grey_all": grey}
    if white:
        sets["white"] = white
        sets["grey_matched_white"] = grey_matched(len(white))
    if void:
        sets["void"] = void
        sets["grey_matched_void"] = grey_matched(len(void))

    rows = []
    for signal in ("potential", "csd"):
        for set_name, groups in sets.items():
            if not groups:
                continue
            for band in list(BANDS) + ["all"]:
                M = (np.concatenate([amp[signal][b][groups] for b in BANDS], axis=0) if band == "all"
                     else amp[signal][band][groups])
                for r in decode_rows(None, B, names, Fixed(M, f"{signal}_{band}"), C.BEHAVIOR_LAGS, C.GAP_BINS,
                                     signal, f"{set_name}|{band}"):
                    r["n_groups"] = len(groups)
                    rows.append(r)
    df = pd.DataFrame(rows)
    df.insert(0, "pid", pid)
    df.insert(1, "eid", meta["eid"])
    df[["set", "band"]] = df.variant.str.split("|", expand=True)
    df["regions"] = json.dumps(regions)
    df.to_csv(out, index=False)
    Path(str(out) + ".lock").unlink(missing_ok=True)
    print(f"[{pid[:8]}] interior groups: grey {len(grey)}, white {len(white)}, void {len(void)}, {time.time() - tic:.0f} s", flush=True)


def worker():
    sel = pd.read_csv(PIDS_CSV).set_index("pid")
    for pid, row in sel.iterrows():
        if (Path(CACHE_DIR) / row.eid / f"{pid}.npz").exists():
            out = claim(pid, "probes_validity")
            if out is not None:
                run_one(pid, row, out)


def aggregate():
    files = sorted((RESULTS / "probes_validity").glob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pp = df.groupby(["pid", "check", "set", "band"]).r.mean().reset_index()       # check = signal
    tab = pp.groupby(["set", "band", "check"]).r.median().unstack("check")
    tab["n_insertions"] = pp.groupby(["set", "band"]).pid.nunique()
    order = list(BANDS) + ["all"]
    set_order = ["grey_all", "grey_matched_white", "white", "grey_matched_void", "void"]
    tab = tab.reindex(pd.MultiIndex.from_product([set_order, order])).dropna(how="all")
    tab.to_csv(RESULTS / "neural_validity_summary.csv")
    print(f"{df.pid.nunique()} insertions. Median over insertions of held-out r (mean of 5 behaviours):")
    with pd.option_context("display.float_format", "{:+.3f}".format):
        print(tab.to_string())
    w = pp.pivot_table(index="pid", columns=["check", "set", "band"], values="r")
    print("\npaired comparisons, all bands:")
    for a_set, b_set in (("grey_matched_white", "white"), ("grey_matched_void", "void")):
        for sig in ("potential", "csd"):
            ka, kb = (sig, a_set, "all"), (sig, b_set, "all")
            if ka in w.columns and kb in w.columns:
                d = (w[ka] - w[kb]).dropna()
                print(f"  {sig:9s} {a_set} minus {b_set}: median {d.median():+.3f}, higher in {int((d > 0).sum())}/{len(d)}")
    for sig_a, sig_b in (("csd", "potential"),):
        ka, kb = (sig_a, "grey_all", "all"), (sig_b, "grey_all", "all")
        d = (w[ka] - w[kb]).dropna()
        print(f"  grey matter, CSD minus potential: median {d.median():+.3f}, CSD higher in {int((d > 0).sum())}/{len(d)}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--aggregate", action="store_true")
    a = ap.parse_args()
    if a.worker:
        worker()
    if a.aggregate:
        aggregate()
        figure()


def figure():
    """fig_neural_validity.png: grey-matter CSD vs potential by band; void vs matched grey matter."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#9a9892"
    files = sorted((RESULTS / "probes_validity").glob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pp = df.groupby(["pid", "check", "set", "band"]).r.mean().reset_index()
    w = pp.pivot_table(index="pid", columns=["check", "set", "band"], values="r")

    def style(ax):
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK_2, labelsize=9)

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6), facecolor=SURFACE, gridspec_kw={"width_ratios": [1.6, 1, 1]})

    ax = axes[0]
    style(ax)
    bands = list(BANDS) + ["all"]
    x = np.arange(len(bands))
    for off, sig, c, lab in ((-0.19, "potential", GREY, "potential (includes far-field)"),
                             (0.19, "csd", BLUE, "CSD (local sources only)")):
        v = [w[(sig, "grey_all", b)].median() for b in bands]
        ax.bar(x + off, v, width=0.36, color=c, label=lab)
        for xi, vi in zip(x + off, v):
            ax.text(xi, vi + 0.008, f"{vi:.2f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x, ["delta", "theta", "alpha", "beta", "gamma", "all bands"])
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("held-out r, mean of 5 behaviours", color=INK_2)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("A  Grey matter: removing the far field keeps the signal", loc="left", fontsize=11,
                 fontweight="bold", color=INK)

    ax = axes[1]
    style(ax)
    d = (w[("csd", "grey_all", "all")] - w[("potential", "grey_all", "all")]).dropna()
    rng = np.random.default_rng(0)
    ax.scatter(rng.uniform(-0.12, 0.12, len(d)), d, s=18, color=BLUE, alpha=0.6, edgecolor="none")
    ax.plot([-0.2, 0.2], [d.median()] * 2, color=INK, linewidth=2.5)
    ax.axhline(0, color=INK_2, linewidth=0.9)
    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.text(0.24, d.median(), f"median {d.median():+.3f}\nCSD higher in {int((d > 0).sum())}/{len(d)}",
            va="center", fontsize=9, color=INK)
    ax.set_ylabel("CSD minus potential (all bands), per insertion", color=INK_2)
    ax.set_title("B  Per insertion", loc="left", fontsize=11, fontweight="bold", color=INK)

    ax = axes[2]
    style(ax)
    have = [p for p in w.index if not np.isnan(w.loc[p].get(("potential", "void", "all"), np.nan))]
    for sig, c, xs in (("potential", GREY, (0, 1)), ("csd", BLUE, (2.2, 3.2))):
        for p in have:
            ax.plot(xs, [w.loc[p, (sig, "grey_matched_void", "all")], w.loc[p, (sig, "void", "all")]],
                    color=c, marker="o", linewidth=1.3, markersize=5, alpha=0.85)
    ax.set_xticks([0, 1, 2.2, 3.2], ["grey\nmatter", "outside\nbrain", "grey\nmatter", "outside\nbrain"], fontsize=8.5)
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("held-out r, all bands", color=INK_2)
    ax.set_title(f"C  Outside the brain ({len(have)} insertions)", loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.text(0.5, 0.66, "potential", ha="center", fontsize=9, color=GREY)
    ax.text(2.7, 0.66, "CSD", ha="center", fontsize=9, color=BLUE)

    fig.text(0.01, 0.015, f"{df.pid.nunique()} insertions. CSD = second spatial difference of the 320 µm group potential; "
             "band amplitude = log RMS, 200 ms.\nGrey matter = named Cosmos region; outside = 'void' "
             "(groups with same-class neighbours only). Same folds, lags and ridge readout as the benchmark.",
             fontsize=8.3, color=INK_2)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(RESULTS / "figures" / "fig_neural_validity.png", dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("figure: fig_neural_validity.png")
