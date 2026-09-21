"""
Write docs/STANDARDS_AND_THEORY.md: what the representation is for, the theory behind each
design, the catalogue of standards and tests, and the results matrix for PCA vs the five
autoencoder designs. Every result number is read from model_report.py's numbers.json or from the
benchmark's own result files; the theory text contains no hand-typed results.

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    /opt/miniconda3/envs/lfp-brain-state/bin/python -m lfp_autoencoder_prototype.write_standards [--report DIR]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .train import AE_RESULTS

ROOT = AE_RESULTS.parent
REPO = Path(__file__).resolve().parents[1]
COLS = ["waveform_pca_48", "group_pca_pooled_48", "group_pca_weighted_48", "group_wave+amp1_pooled",
        "ae_depth_48", "ae_global_48", "ae_masked_48", "ae_anatomy_48", "ae_filterbank_48", "ae_waveamp_48"]
HEAD = ["PCA 48", "Group PCA (pooled)", "**W** twin of A–E", "**WF** twin of F",
        "**A** per-depth", "**B** global", "**C** masked", "**D** anatomy", "**E** filterbank", "**F** wave + amp"]
LEARNED = {"ae_depth_48", "ae_global_48", "ae_masked_48", "ae_anatomy_48", "ae_filterbank_48", "ae_waveamp_48"}
DEPTH_INDEXED = {"group_pca_pooled_48", "group_pca_weighted_48", "group_wave+amp1", "group_wave+amp1_pooled",
                 "ae_depth_48", "ae_masked_48", "ae_anatomy_48", "ae_filterbank_48", "ae_waveamp_48"}
TWIN = lambda m: "group_wave+amp1_pooled" if m == "ae_waveamp_48" else "group_pca_weighted_48"
POOLED = {"group_pca_pooled_48", "group_pca_weighted_48"} | LEARNED
TOL = 0.02


def f2(x, nd=2):
    if x is None or not np.isfinite(x):
        return "–"
    s = f"{x:.{nd}f}"
    return s.replace("-", "−")


def get(d, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def cell(v, ok, nd=2):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    txt = v if isinstance(v, str) else f2(v, nd)
    return txt if ok is None else (f"{txt} ✓" if ok else f"{txt} ✗")


def row_rel(N, fetch, higher=True, extra=None, nd=2):
    """Pass = not worse than the model's linear twin (W; WF for F) by more than TOL. Linear columns: no mark."""
    out = []
    for m in COLS:
        v = fetch(m)
        w = fetch(TWIN(m))
        if v is None or w is None or m not in LEARNED:
            ok = None
        else:
            ok = (v >= w - TOL) if higher else (v <= w + TOL)
            if extra is not None:
                ok = ok and extra(m, v)
        out.append(cell(v, ok, nd))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=str(ROOT / "model_purpose_2026-09-21"))
    ap.add_argument("--out", default=str(REPO / "docs" / "STANDARDS_AND_THEORY.md"))
    a = ap.parse_args()
    rep = Path(a.report)
    N = json.load(open(rep / "tables" / "numbers.json"))
    B = N["bench"]
    nv = pd.read_csv(ROOT / "selected_benchmark_results" / "neural_validity_summary.csv")
    nv.columns = ["set", "band", "csd", "potential", "n"]
    nv = nv[nv.band == "all"].set_index("set")
    al = pd.read_csv(ROOT / "selected_benchmark_results" / "aliasing_by_insertion.csv").drop(columns="pid").median()
    fc = json.load(open(AE_RESULTS / "purpose" / "feature_cost.json"))
    have = lambda m: m in B
    L = []
    w = L.append

    n_ins = int(B["waveform_pca_48"]["n_insertions"])
    missing = [m for m in COLS if not have(m)]
    w("# What the representation is for: theory, standards and tests for PCA and six autoencoder designs\n")
    w(f"Skylar Tian · generated from result files by `lfp_autoencoder_prototype/write_standards.py` · Alon's {n_ins} "
      "selected insertions (26 sessions, 9 labs), 48 numbers per 40 ms unless stated.\n")
    status = (f"All {len(COLS)} columns of the results matrix are complete." if not missing else
              f"Columns still missing when this was generated: {', '.join(missing)} (training/evaluation in progress); "
              "their cells show –.")
    w(f"> **Status.** {status} A and B come from the 50-epoch retrain when `ae_results/probes` holds it "
      f"(epoch cap in the files read here: {N.get('ab_epochs_cap')}). Results are **completed** measurements on held-out "
      "sessions unless a section says *preliminary* or *proposed*. Nothing here was used to tune the models.\n")

    # ------------------------------------------------------------------ 1 purpose
    w("## 1. The purpose, stated as requirements\n")
    w("Task 1 builds the measurement layer that tasks 2–5 stand on. A representation is fit for purpose only if it "
      "serves those tasks, so the requirements come from them, not from what is easy to score:\n")
    w("| # | Requirement | Which project task needs it | Why, neuroscientifically |")
    w("|---|---|---|---|")
    reqs = [
        ("R1", "Compact and fast (≤ 48 numbers / 40 ms; encodes 100 s in seconds on a CPU)", "All: 750 insertions × hours", "Brain-wide scale; the raw LF band is 384 ch × 2.5 kHz"),
        ("R2", "Keeps state-relevant spectral content: band power delta→gamma and the aperiodic slope", "3 (organisation of state), 5 (atlas)", "Brain states are spectral: desynchronisation lowers low-frequency power and raises gamma with arousal and locomotion; hippocampal theta tracks running; the aperiodic exponent tracks E/I balance"),
        ("R3", "Keeps location: every number tied to a depth that maps to a region", "2 (local vs global), 4 (anatomy), 5 (atlas)", "High-frequency power is local; low-frequency potential is volume-conducted from elsewhere"),
        ("R4", "Shared coordinates: the same number means the same thing in every insertion and animal", "5 (across-animal atlas), 3 (similarity across animals)", "An atlas pools animals; per-insertion axes (sign, rotation) cannot be pooled"),
        ("R5", "Keeps both the component shared between regions and the private one", "2 (does a brain-wide latent exist?)", "Global arousal signals coexist with local circuit dynamics"),
        ("R6", "Keeps timescales: no smearing of fast dynamics, no invented slow ones", "3 (timescales, discrete vs continuous)", "State dwell times run from ~0.1 s (gamma bursts) to minutes (arousal)"),
        ("R7", "Learned without behaviour", "3, 5: states are defined from neural data, then related to behaviour", "Otherwise the state→behaviour link is circular"),
        ("R8", "Represents neural signal, not artefact", "All", "Movement artefacts, EMG and far-field potentials also correlate with behaviour"),
        ("R9", "Interpretable, stable axes", "4, 5: relate to physiology; compare across retrainings", "A latent that means 'broadband power at this depth' can be related to anatomy; an arbitrary mixture cannot"),
    ]
    for r in reqs:
        w("| " + " | ".join(r) + " |")
    w("")

    # ------------------------------------------------------------------ 2 theory
    pca = B["waveform_pca_48"]
    w("## 2. Theory\n")
    w("### 2.1 Why waveform PCA cannot meet R2–R4, however well it reconstructs\n")
    w(f"Waveform PCA 48 keeps waveform R² **{f2(pca['r2_wave'])}** yet recovers amplitude depth pattern R² "
      f"**{f2(pca['r2_amp_spatial'])}** and gamma-envelope R² **{f2(pca['r2_env_gamma'])}**. That is not a tuning problem; it "
      "follows from three facts:\n")
    w("1. **Power is quadratic in the signal.** Band power is the smoothed square of a band-passed signal. Every PCA "
      "coordinate is a linear function of the signal, and so is every ridge readout of those coordinates, with or "
      "without time lags. No linear function equals a positive quadratic form, so amplitude, the quantity brain states "
      "are defined by (R2), is out of reach of any linear readout of waveform PCs.")
    w(f"2. **The 25 Hz waveform has no beta or gamma, and some of what it has is aliased.** A 40 ms boxcar keeps "
      f"< 12.5 Hz. The other session measured that **{al['alpha' + chr(10) + '8–12.5 Hz'] * 100:.1f}%** of the 8–12.5 Hz power in "
      f"the benchmark waveform is aliased content from above 13 Hz (theta {al['theta' + chr(10) + '4–8 Hz'] * 100:.1f}%, delta "
      f"{al['delta' + chr(10) + '1–4 Hz'] * 100:.1f}%; median of 50 insertions; `lfp_selected_benchmark/aliasing.py`).")
    w("3. **PCA keeps variance, and LFP variance is low-frequency and far-field.** With a ~1/f spectrum, most "
      "variance is slow, and slow potentials are volume-conducted over millimetres (Kajikawa & Schroeder 2011; "
      "Buzsáki, Anastassiou & Koch 2012). The top PCs are therefore smooth, probe-wide modes. Local, high-frequency "
      "activity carries little variance and is the first thing dropped (R3). Per-insertion PCs are also defined only up "
      "to sign and rotation, so they share no coordinates across animals (R4).")
    w("\nSo PCA is the right answer to *compress the waveform* and the wrong one to *represent brain state*. This is "
      "why our gains come from changing the **input** (log band power per depth) before changing the **model**.\n")

    w("### 2.2 Why per-depth, multi-scale log band power is the right substrate\n")
    w("- **State is spectral.** Arousal and locomotion desynchronise cortex: low-frequency power falls and gamma rises "
      "(Niell & Stryker 2010; Vinck et al. 2015; McGinley et al. 2015). Akella et al. (2025) define their three states "
      "from exactly such band envelopes.")
    w("- **Log.** Power is multiplicative and roughly log-normal. On a log scale the 1/f spectrum becomes additive, so "
      "each band contributes comparably instead of delta dominating.")
    w("- **At least one cycle per window.** Resolving frequency f needs roughly 1/f seconds (time–frequency "
      "uncertainty). Hence delta and theta at 1 s, beta at 200 ms and 1 s, gamma down to 40 ms.")
    w("- **Aperiodic slope.** The 1/f exponent tracks excitation/inhibition balance (Gao, Peterson & Voytek 2017) and "
      "has to be separated from oscillatory peaks (Donoghue et al. 2020).")
    g, gp = nv.loc["grey_all"], nv
    w(f"- **Locality.** The current-source density (second spatial difference) removes far-field potential "
      f"(Mitzdorf 1985; Pesaran et al. 2018). The other session found that grey-matter CSD band power decodes "
      f"behaviour at r = **{f2(g.csd, 3)}** vs **{f2(g.potential, 3)}** for the potential ({int(g.n)} insertions; "
      "`neural_validity.py`), so the behavioural signal is at least as strong in the local component. White-matter "
      f"groups decode about as well as matched grey groups ({f2(gp.loc['white'].potential)} vs "
      f"{f2(gp.loc['grey_matched_white'].potential)}; {int(gp.loc['white'].n)} insertions), consistent with spread "
      f"from nearby grey matter. Out-of-brain groups decode less ({f2(gp.loc['void'].potential)} vs "
      f"{f2(gp.loc['grey_matched_void'].potential)}), but on only {int(gp.loc['void'].n)} insertions.")
    w("- **Caveat that shapes the tests.** Movement drives activity brain-wide (Stringer et al. 2019; Musall et al. "
      "2019), and artefacts correlate with movement too. High behaviour decoding is expected almost anywhere and is not, "
      "on its own, evidence of a good state representation. That is why behaviour r is one row of the table below, "
      "not the criterion.\n")

    w("### 2.3 What the autoencoders optimise, and when that matches the purpose\n")
    w("All five designs minimise the same loss: mean squared error on the z-scored 19-feature bank at 24 depths. Views "
      "are weighted equally (waveform ⅓, nine RMS ⅓, nine PSD ⅓), there are 48 latents per 40 ms, and behaviour is "
      "never seen (R7 ✓).\n")
    w("- **The linear twin.** A linear autoencoder trained with squared error spans the PCA subspace of its input "
      "(Baldi & Hornik 1989). The linear, context-free minimiser of *our* loss is therefore group PCA on features scaled "
      "by √(view weight), fit on the training sessions: **W**. Plain group PCA gives the waveform 1/19 of the variance "
      "instead of ⅓, so it answers a different objective. Comparing A with plain group PCA mixes a change of objective "
      "with a change of model; comparing A with **W** isolates what the network adds (nonlinearity plus ±240 ms and "
      "±320 µm of context).")
    w("- **What the dominant covariance is.** On held-out insertions, W's two latents per depth are the waveform and "
      "broadband power (all bands moving together). Broadband power is the classic desynchronisation/arousal axis. "
      "Without view weighting, the second axis is instead an RMS-vs-PSD contrast (section 6.4).")
    w("- **What MSE does not know.** Reconstruction keeps variance, not state relevance. Slow drifts, artefacts and "
      "far-field signals survive if they are large. This is why the neural-validity checks and the "
      "shared-coordinate tests are needed on top of reconstruction.")
    w("- **Identifiability.** An autoencoder's latents are defined only up to an invertible reparametrisation the "
      "decoder can undo (see Locatello et al. 2019 on the general problem). Within one trained encoder, coordinates are "
      "shared across insertions (R4). Across retrainings they are not, so an atlas needs a frozen, versioned encoder or "
      "a canonicalised latent.\n")
    w("**Hypothesis behind each design, and what would falsify it**\n")
    w("| Design | Inductive bias | Hypothesis | Falsified if |")
    w("|---|---|---|---|")
    w("| **A** per-depth | same computation at every depth (shared weights), ±240 ms, ±320 µm | nonlinearity + context add information a linear per-depth summary misses | A ≤ W on behaviour, depth pattern, transfer and states |")
    w("| **B** global | depths flattened into 48 shared latents | a global summary averages noise across depths | (not a spatial representation by construction: fails R3 whatever it scores) |")
    w("| **C** masked | hide 25% of depth groups in training, reconstruct all | learning spatial predictability helps infer unrecorded depths (a small version of 'infer unrecorded regions') | C ≤ linear interpolation in depth on hidden groups |")
    w("| **D** anatomy | learned embedding of the Cosmos region per depth, in encoder and decoder | explaining region identity frees the latent to encode state | D ≤ A on transfer, states and behaviour |")
    w("| **E** filterbank | 8 Gaussian band-pass filters learned on the raw 250 Hz signal (LEAF/SincNet-style); no hand-made features in the input | our band choices are not optimal, and a learned front end recovers the state information itself | E ≤ A everywhere, or learned bands wander between folds |")
    w("| **F** wave + amp | waveform stored as is (24); 1 learned amplitude latent per depth from the 18 RMS/PSD features | spend nonlinear capacity only where the physics needs it (amplitude), since the sub-12.5 Hz waveform is spatially smooth | F ≤ WF (stored waveform + 1 pooled amplitude PC per depth) |")
    w("")

    # ------------------------------------------------------------------ 3 standards catalogue
    w("## 3. The standards and tests\n")
    w("Criteria were fixed before the C, D, E, W and purpose-test results existed, though after the baseline "
      "benchmark and the 15-epoch A/B results. F and its twin WF were added during the run, after W's first result. The comparator for any learned model is **W** (does the network earn "
      f"its complexity?); tolerance ±{TOL} for R²/r metrics. The comparator for the project question is **PCA 48** "
      "(does it improve on the current method at the same budget?).\n")
    w("| # | Standard | Requirement | Test | Metric | Pass criterion |")
    w("|---|---|---|---|---|---|")
    cat = [
        ("T1", "Compact", "R1", "numbers stored per second", "dims × 25 Hz", "≤ 1,200 floats/s (48 dims)"),
        ("T2", "Waveform fidelity", "R1, R6", "ridge from representation to 384-ch 25 Hz waveform, 4 blocked folds", "R² (pooled)", "≥ 0.90"),
        ("T3", "Fast amplitude", "R2, R6", "ridge to 40 ms gamma log-RMS envelope", "R²", "≥ W − 0.02"),
        ("T4", "Probe-wide amplitude", "R2", "ridge to 200 ms beta/gamma log-RMS per group", "R²", "≥ W − 0.02"),
        ("T5", "Amplitude depth pattern", "R3", "same, minus the probe mean (where on the probe)", "R²", "≥ W − 0.02 and ≥ PCA 48 + 0.2"),
        ("T6", "Behavioural information", "R2 (proxy)", "ridge with ±320 ms lags → 5 behaviours", "mean r", "≥ W − 0.02 (paired sessions reported)"),
        ("T7", "Null control", "validity", "behaviour circularly shifted 30 s", "mean r", "|r| ≤ 0.03"),
        ("T8", "Shared coordinates", "R4", "zero-shot: readout trained on the other 25 sessions, applied unchanged", "mean r", "≥ W − 0.02"),
        ("T9", "Depth-invariant transfer", "R3 + R4", "zero-shot with a readout that only sees mean and SD over depths", "mean r", "reported (depth-indexed only)"),
        ("T10", "Cross-region shared component", "R5", "cross-validated CCA between the 2 simultaneous probes (24 pairs)", "mean of top-5 held-out canonical r", "≥ W − 0.02 and > shift null"),
        ("T11", "State structure", "R2, R6", "3-state HMM vs Akella-style reference HMM (per insertion)", "NMI", "≥ W − 0.02"),
        ("T12", "States across animals", "R4", "one HMM fit on other sessions, decoded on the held-out one", "NMI with pooled reference", "≥ W − 0.02"),
        ("T13", "Timescale", "R6", "1/e autocorrelation time of the dimensions", "s (median)", "reported"),
        ("T14", "Location", "R3", "structural: are numbers tied to depths?", "yes/no", "yes"),
        ("T15", "Behaviour-free", "R7", "structural", "yes/no", "yes"),
        ("T16", "Interpretable axis", "R9", "max |corr| of a latent channel with one bank feature (held-out)", "|r|", "≥ 0.5"),
        ("T17", "Spatial imputation", "R3, research goal", "hide depth groups of held-out insertions (C only)", "R² on hidden groups", "> depth interpolation"),
        ("T18", "Cost", "R1", "params; CPU train min/fold; s to encode 100 s (+13 s for the feature bank where needed)", "–", "train ≤ 60 min/fold"),
    ]
    for c in cat:
        w("| " + " | ".join(c) + " |")
    w("")

    # ------------------------------------------------------------------ 4 results matrix
    w("## 4. Results matrix\n")
    w(f"Medians over held-out insertions (T1–T9, T11–T13, T16) or simultaneous pairs (T10). ✓/✗ = criterion met or "
      "not. Relative criteria compare each learned model with its linear twin (W for A–E, WF for F); linear columns are "
      "the references and carry no relative mark. – = not applicable or not available.\n")
    w("| Test | " + " | ".join(HEAD) + " |")
    w("|---|" + "---|" * len(COLS))
    bget = lambda k: (lambda m: get(B, m, k))
    w("| T1 floats/s | " + " | ".join("1,200 ✓" for _ in COLS) + " |")
    w("| T2 waveform R² | " + " | ".join(cell(get(B, m, "r2_wave"), None if get(B, m, "r2_wave") is None else get(B, m, "r2_wave") >= 0.9) for m in COLS) + " |")
    w("| T3 gamma env R² | " + " | ".join(row_rel(N, bget("r2_env_gamma"))) + " |")
    w("| T4 amplitude R² | " + " | ".join(row_rel(N, bget("r2_amp_global"))) + " |")
    p48 = get(B, "waveform_pca_48", "r2_amp_spatial")
    w("| T5 depth-pattern R² | " + " | ".join(row_rel(N, bget("r2_amp_spatial"), extra=lambda m, v: v >= p48 + 0.2)) + " |")
    w("| T6 behaviour r | " + " | ".join(row_rel(N, bget("beh_r_mean"))) + " |")
    w("| T7 shift null r | " + " | ".join(cell(get(B, m, "shuf_r_mean"), None if get(B, m, "shuf_r_mean") is None else abs(get(B, m, "shuf_r_mean")) <= 0.03) for m in COLS) + " |")
    w("| T8 zero-shot r (one encoder per fold) | " + " | ".join(row_rel(N, lambda m: get(N, "transfer", m, "full"))) + " |")
    w("| T8m zero-shot r, mixed fold encoders† | " + " | ".join(cell(get(N, "transfer_mixed", m, "full"), None) for m in COLS) + " |")
    w("| T9 depth-invariant zero-shot r | " + " | ".join(cell(get(N, "transfer", m, "pooled"), None) for m in COLS) + " |")
    null = lambda m, v: v > (get(N, "cca", m, "null_top5") or 0)
    w("| T10 CCA top-5 (null) | " + " | ".join(
        (lambda c: c if get(N, "cca", m, "null_top5") is None else f"{c} ({f2(get(N, 'cca', m, 'null_top5'))})")(x)
        for m, x in zip(COLS, row_rel(N, lambda m: get(N, "cca", m, "cc_top5"), extra=null))) + " |")
    w("| T11 state NMI | " + " | ".join(row_rel(N, lambda m: get(N, "states_within", m, "nmi_akella_ref"))) + " |")
    w("| T12 pooled state NMI (one encoder per fold) | " + " | ".join(row_rel(N, lambda m: get(N, "states_pooled", m, "nmi_akella_ref"))) + " |")
    w("| T10b CCA top-5 after 1 Hz high-pass | " + " | ".join(cell(get(N, "cca", m, "hp1hz_top5"), None) for m in COLS) + " |")
    ak = get(N, "states_within", "akella_ref", "nmi_akella_ref")
    w("| T13 timescale (s)* | " + " | ".join(cell(get(N, "timescale", m), None) for m in COLS) + " |")
    w("| T14 location | " + " | ".join("yes ✓" if m in DEPTH_INDEXED else "no ✗" for m in COLS) + " |")
    w("| T15 behaviour-free | " + " | ".join("yes ✓" for _ in COLS) + " |")
    def interp(m):
        vals = [v for k, v in N.get("interpret_max_abs", {}).items() if k.split("|")[0] == m]
        return max(vals) if vals else None
    w("| T16 max \\|r\\| latent↔feature | " + " | ".join(cell(interp(m), None if interp(m) is None else interp(m) >= 0.5) for m in COLS) + " |")
    iw = N.get("imputation_wins_rms", {})
    imp_cell = lambda m: ("–" if m != "ae_masked_48" or not iw else
                          " · ".join(f"{k.split('|')[1][3:]} {v['ae_better']}/{v['n']}" for k, v in sorted(iw.items()) if k.startswith("random25")))
    w("| T17 imputation, random 25 %: insertions where C > shrunk interpolation (per view) | " + " | ".join(imp_cell(m) for m in COLS) + " |")
    cost = N.get("cost", {})
    def cost_cell(m):
        c = cost.get(m)
        if c is None:
            return "linear: s" if m not in LEARNED else "–"
        ok = c.get("train_min_per_fold") is not None and c["train_min_per_fold"] <= 60
        return f"{int(c['params']):,} p · {f2(c['train_min_per_fold'], 0)} min" + (" ✓" if ok else " ✗")
    cost_cell_ = cost_cell
    cost_cell = lambda m: "linear (seconds)" if m not in LEARNED else cost_cell_(m)
    w("| T18 params · train/fold | " + " | ".join(cost_cell(m) for m in COLS) + " |")
    w("")
    w(f"† T8m trains the readout on insertions encoded by the other folds' models and tests on the held-out fold's own "
      "model, so it asks whether separately trained encoders agree (identifiability, R9), not whether one encoder gives "
      "shared coordinates (T8). For learned models, T8, T9 and T12 re-encode every insertion with one fold's model "
      "(`consistent_latents.py`). That is impossible for A and B, which were trained without saving weights, hence –. "
      "Methods with no trained encoder use their own fit per insertion (PCA 48) or per fold, so T8 and T8m coincide for "
      "them.\n")
    if ak is not None:
        w(f"HMM ceiling: refitting the same 3-state HMM on the reference features themselves (a PCA rotation and "
          f"re-standardisation, different local optimum) agrees with the reference at NMI **{f2(ak)}**. Read T11 against "
          "that ceiling, not against 1.\n")
    w(f"Feature cost, one 100 s insertion, 1 thread, measured while other jobs ran (upper bounds): computing the "
      f"456-feature bank takes **{fc['feature_bank_456_s']:.1f} s** (needed by group PCA, W, wave+amp, A–D); waveform "
      f"PCA 48 fit + encode **{fc['waveform_pca_48_fit_and_encode_s']:.2f} s**. Encoding with a trained AE adds "
      + ", ".join(f"{k} {v['encode_s_per_100s']:.2f} s" for k, v in cost.items()
                  if v.get("encode_s_per_100s") is not None and np.isfinite(v["encode_s_per_100s"]))
      + " (E reads raw and needs no bank; A and B were not timed because train.py saves no weights, but they share "
      "C/D's trunk). Training times are CPU minutes per fold with 5–8 jobs sharing 4 physical cores.\n")

    # ------------------------------------------------------------------ 5 paired statistics
    w("## 5. Paired, session-level comparisons\n")
    w("Per-session mean difference (simultaneous probes averaged, so sessions are the independent unit). "
      "k/n = sessions where the model is higher; p = two-sided Wilcoxon signed-rank over sessions, uncorrected and "
      "descriptive only.\n")
    w("| Model | vs | Δ behaviour r | k/n | p | Δ depth-pattern R² | k/n | Δ waveform R² | k/n |")
    w("|---|---|---|---|---|---|---|---|---|")
    P = N["paired"]
    for m in ["ae_depth_48", "ae_global_48", "ae_masked_48", "ae_anatomy_48", "ae_filterbank_48", "ae_waveamp_48",
              "group_wave+amp1_pooled", "group_pca_pooled_48"]:
        for base in ("group_pca_weighted_48", "group_wave+amp1_pooled", "waveform_pca_48", "ae_depth_48"):
            if m == base or f"{m}|{base}|beh_r_mean" not in P:
                continue
            if base == "ae_depth_48" and m not in ("ae_masked_48", "ae_anatomy_48", "ae_filterbank_48", "ae_global_48", "ae_waveamp_48"):
                continue
            if base == "group_wave+amp1_pooled" and m != "ae_waveamp_48":
                continue
            b_, s_, wv = P[f"{m}|{base}|beh_r_mean"], P[f"{m}|{base}|r2_amp_spatial"], P[f"{m}|{base}|r2_wave"]
            lab = {"group_pca_weighted_48": "W", "group_wave+amp1_pooled": "WF", "waveform_pca_48": "PCA 48", "ae_depth_48": "A"}[base]
            w(f"| {m} | {lab} | {f2(b_['d'], 3)} | {b_['k']}/{b_['n']} | {f2(b_['p'], 3)} | {f2(s_['d'], 3)} | {s_['k']}/{s_['n']} | "
              f"{f2(wv['d'], 3)} | {wv['k']}/{wv['n']} |")
    w("")

    # ------------------------------------------------------------------ 6 model-specific
    w("## 6. Model-specific results\n")
    if "heldout_sd" in N:
        w("### 6.1 Held-out reconstruction of the feature bank: static vs dynamic\n")
        w("'raw' = fraction of variance reconstructed; 'dynamic' = the same after removing each (feature, depth) "
          "mean residual on the held-out insertion, i.e. ignoring a wrong static depth profile. The benchmark's ridge "
          "readouts fit intercepts, so only the dynamic part affects sections 4–5. Medians over held-out insertions.\n")
        w("| Model | waveform raw | RMS raw | PSD raw | waveform dyn | RMS dyn | PSD dyn |")
        w("|---|---|---|---|---|---|---|")
        for m, v in N["heldout_sd"].items():
            w(f"| {m} | {f2(v['waveform_raw'])} | {f2(v['rms_raw'])} | {f2(v['psd_raw'])} | {f2(v['waveform_dynamic'])} | "
              f"{f2(v['rms_dynamic'])} | {f2(v['psd_dynamic'])} |")
        w("")
    if "heldout_sd" in N:
        w("A and B were trained by `train.py`, which does not save weights, so their decoder reconstruction is not in "
          "this table. The pooled group PCA row reconstructs from its 2 components per depth.\n")
    if "imputation" in N:
        w("### 6.2 C: inferring hidden depths\n")
        w("Hidden groups on held-out insertions: four 3-group blocks (0.96 mm) and four random 25% masks per insertion. "
          "R² over the hidden groups only, per view:\n")
        w("| Mask | Method | waveform | RMS | PSD |")
        w("|---|---|---|---|---|")
        for k, v in sorted(N["imputation"].items()):
            mk, me = k.split("|")
            w(f"| {mk} | {me} | {f2(v['r2_waveform'])} | {f2(v['r2_rms'])} | {f2(v['r2_psd'])} |")
        w("")
    if "filterbank" in N:
        w("### 6.3 E: what the learned filters converge to\n")
        w("Initial centres were log-spaced 2–80 Hz. Median (range across folds) of the learned centre and width:\n")
        w("| Filter | centre Hz | width Hz |")
        w("|---|---|---|")
        for i, v in N["filterbank"].items():
            w(f"| {i} | {f2(v['centre_med'], 1)} ({f2(v['centre_min'], 1)}–{f2(v['centre_max'], 1)}) | "
              f"{f2(v['width_med'], 1)} ({f2(v['width_min'], 1)}–{f2(v['width_max'], 1)}) |")
        w(f"\nFolds with a trained filterbank: {int(max(v['n_folds'] for v in N['filterbank'].values()))} of 5.\n")
        w("")
    if "interpret_max_abs" in N:
        w("### 6.4 What the latent channels track\n")
        w("Correlation, over depths × time on held-out insertions, of each depth-indexed latent channel with each bank "
          "feature, with signs aligned across insertions. The full profiles are in `tables/model_purpose/latent_feature_profiles.csv`; "
          "the maximum |r| per channel is:\n")
        w("| Model · channel | max \\|r\\| |")
        w("|---|---|")
        for k, v in N["interpret_max_abs"].items():
            w(f"| {k.replace('|', ' · ')} | {f2(v)} |")
        w("")
    return L, a, N


def finish(L, a):
    import shutil
    out = Path(a.out)
    figs = out.parent / "figures" / "model_purpose"
    figs.mkdir(parents=True, exist_ok=True)
    for f in sorted((Path(a.report) / "figures").glob("*.png")):
        shutil.copy2(f, figs / f.name)
    tabs = out.parent / "tables" / "model_purpose"                   # the tables the write-up cites
    tabs.mkdir(parents=True, exist_ok=True)
    for f in sorted((Path(a.report) / "tables").glob("*")):
        shutil.copy2(f, tabs / f.name)
    out.write_text("\n".join(L) + "\n")
    shutil.copy2(out, Path(a.report) / out.name)                      # a copy next to the tables and figures
    print("wrote", out, "and", Path(a.report) / out.name)


if __name__ == "__main__":
    L, a, N = main()
    from .write_standards_tail import tail
    L += tail(N)
    finish(L, a)
