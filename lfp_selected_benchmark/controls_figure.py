"""
One figure and one table for the verification checks (controls.py).

    $PY -m lfp_selected_benchmark.controls_figure

Panels:
  A  from the current representation (probe-wide RMS envelope, no time context), one change at a time
  B  robustness: independent ridge implementation, time context, fold gaps
  C  nulls: behaviour from other sessions, and circularly shifted behaviour
  D  alignment: peak cross-correlation lag between gamma amplitude and behaviour

The earlier "outside the brain" panel was removed on 20 Sep: it counted Cosmos 'root' (fiber tracts,
ventricles) as outside. Grey vs white matter vs void is in neural_validity.py.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .run_selected import RESULTS

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, GREEN, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9892"
CURRENT_R = 0.46       # earlier result with the current RMS envelope (31 Aug slides, slide 24)


def style(ax, grid="y"):
    ax.set_facecolor(SURFACE)
    ax.grid(True, axis=grid, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK_2, labelsize=9)


def per_probe(df):
    """Mean over behaviours and folds, one value per (insertion, setting)."""
    d = df[df.check != "align"]
    return d.groupby(["pid", "check", "rep", "variant", "readout"]).r.mean().reset_index()


def bars(ax, labels, values, colors, title, ylim=None, fmt="{:.2f}"):
    ax.bar(range(len(values)), values, color=colors, width=0.62)
    for i, v in enumerate(values):
        ax.text(i, v + 0.012 * (1 if v >= 0 else -1), fmt.format(v), ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9, color=INK)
    ax.set_xticks(range(len(labels)), labels, fontsize=8.5, rotation=18, ha="right")
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK)
    if ylim:
        ax.set_ylim(*ylim)


def main():
    files = sorted((RESULTS / "probes_controls").glob("*.csv"))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    pp = per_probe(df)
    n = pp.pid.nunique()
    med = pp.groupby(["check", "rep", "variant", "readout"]).r.median()

    plt.rcParams.update({"font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(12, 8.2), facecolor=SURFACE)
    gs = fig.add_gridspec(2, 2, hspace=0.58, wspace=0.22, left=0.07, right=0.985, top=0.85, bottom=0.1)

    # A - from the current representation to the full feature bank, one change at a time
    # (the readout key "ours" in the result files means the benchmark's own ridge code)
    ax = fig.add_subplot(gs[0, 0])
    style(ax)
    ladder = [("probe-wide RMS,\nno time context", ("alonlike", "probe_rms_envelope", "probe_rms_zero_lag", "ours"), GREY),
              ("probe-wide RMS,\n±320 ms context", ("alonlike", "probe_rms_envelope", "probe_rms_lagged", "ours"), GREY),
              ("24 depths,\n16 numbers", ("lags", "wave12+bank4", "acausal_320ms", "ours"), BLUE),
              ("24 depths,\n456 features", ("lags", "bank_wave+rms+psd", "acausal_320ms", "ours"), BLUE)]
    bars(ax, [l for l, _, _ in ladder], [med.loc[k] for _, k, _ in ladder], [c for _, _, c in ladder],
         "A  One change at a time", ylim=(0, 0.72))
    ax.axhline(CURRENT_R, color=ORANGE, linewidth=1.4, linestyle="--",
               label=f"Current RMS envelope, earlier analysis: {CURRENT_R:.2f}")
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    ax.set_ylabel("held-out r, mean of 5 behaviours", color=INK_2, fontsize=9)

    # B - robustness
    ax = fig.add_subplot(gs[0, 1])
    style(ax)
    rob = [("benchmark\nridge", ("lags", "wave12+bank4", "acausal_320ms", "ours")),
           ("scikit-learn\nridge", ("lags", "wave12+bank4", "acausal_320ms", "sklearn")),
           ("past-only\ncontext", ("lags", "wave12+bank4", "causal_320ms", "ours")),
           ("no time\ncontext", ("lags", "wave12+bank4", "zero_lag", "ours")),
           ("5 s gaps\nbetween folds", ("gap", "wave12+bank4", "gap_5s", "ours"))]
    bars(ax, [l for l, _ in rob], [med.loc[k] for _, k in rob], [BLUE, GREEN, BLUE, BLUE, BLUE],
         "B  Alternative analysis choices, 16-number representation", ylim=(0, 0.72))

    # C - nulls
    ax = fig.add_subplot(gs[1, 0])
    style(ax)
    nulls = [("matched\nbehaviour", ("lags", "wave12+bank4", "acausal_320ms", "ours"), BLUE)]
    for i in range(3):
        k = ("swap", "wave12+bank4", f"other_session_{i}", "ours")
        if k in med.index:
            nulls.append((f"behaviour from\nsession {i + 1}", k, GREY))
    nulls.append(("behaviour\nshifted 30 s", ("swap", "wave12+bank4", "circ_shift_30s", "ours"), GREY))
    bars(ax, [l for l, _, _ in nulls], [med.loc[k] for _, k, _ in nulls], [c for _, _, c in nulls],
         "C  Shuffled controls, 16-number representation", ylim=(-0.1, 0.72))
    ax.axhline(0, color=INK_2, linewidth=0.8)

    # E - alignment
    ax = fig.add_subplot(gs[1, 1])
    style(ax, grid="both")
    al = df[df.check == "align"]
    names = {"whisker_me_left": "Whisker motion", "wheel_speed": "Wheel speed", "body_me": "Body motion",
             "pupil_diameter": "Pupil diameter"}
    for beh, c in zip(names, [ORANGE, BLUE, GREEN, GREY]):
        v = al[al.behaviour == beh].peak_lag_ms
        if len(v):
            ax.scatter(v, np.full(len(v), names[beh]), s=14, color=c, alpha=0.6, edgecolor="none")
    ax.axvline(0, color=INK, linewidth=1)
    ax.axvspan(-200, 200, color=GRID, alpha=0.8, zorder=0)
    ax.set_xlabel("lag of peak correlation, gamma amplitude vs behaviour (ms)", color=INK_2, fontsize=9)
    ax.set_title("D  Timing of LFP relative to video", loc="left", fontsize=11, fontweight="bold", color=INK)
    ax.tick_params(labelsize=8.5)

    fig.text(0.07, 0.955, "Checks on the behaviour decoding values", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.07, 0.905, f"{n} insertions. Ridge readout, 4 held-out 25 s blocks per insertion. Value = median over "
             "insertions of the mean r over 5 behaviours.\n16-number representation = 12 waveform PCs + 4 amplitude PCs. "
             "Grey band in D: ±200 ms.", fontsize=9, color=INK_2)
    path = RESULTS / "figures" / "fig_controls.png"
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    print("figure:", path, "| insertions:", n)


if __name__ == "__main__":
    main()
