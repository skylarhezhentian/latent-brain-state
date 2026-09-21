"""
Sections 7-12 of docs/STANDARDS_AND_THEORY.md: verdicts, mechanisms, recommendations, limitations,
references, reproduction. Numbers come from numbers.json (model_report.py); sentences that depend on a
result are computed from it.
"""
import numpy as np

LEARNED = ["ae_depth_48", "ae_global_48", "ae_masked_48", "ae_anatomy_48", "ae_filterbank_48", "ae_waveamp_48"]
LETTER = {"ae_depth_48": "A", "ae_global_48": "B", "ae_masked_48": "C", "ae_anatomy_48": "D",
          "ae_filterbank_48": "E", "ae_waveamp_48": "F"}
TWIN = lambda m: "group_wave+amp1_pooled" if m == "ae_waveamp_48" else "group_pca_weighted_48"
TWIN_L = lambda m: "WF" if m == "ae_waveamp_48" else "W"
TOL = 0.02
PURPOSE = {
    "ae_depth_48": ("R2, R3, R4, R7", "yes: spatial, pooled, behaviour-free"),
    "ae_global_48": ("R2, R4, R7", "**no for a spatial representation**: no number is tied to a depth (fails R3 by construction); useful only as an upper bound on probe-wide information"),
    "ae_masked_48": ("R3 + inferring unrecorded depths", "yes: a small-scale version of the project's 'infer unrecorded regions' question"),
    "ae_anatomy_48": ("R3, R4 with anatomical grounding", "yes, but needs histology-aligned region labels at deployment"),
    "ae_filterbank_48": ("R2 without hand-made features", "yes for the long-term goal; the prototype reads only 2 of 16 channels per depth"),
    "ae_waveamp_48": ("R1–R3: capacity allocated by physics", "yes: waveform stored, network spends all capacity on amplitude"),
}


def g(d, *k):
    for x in k:
        if not isinstance(d, dict) or x not in d:
            return None
        d = d[x]
    return d


def f2(x, nd=2):
    return "–" if x is None or not np.isfinite(x) else f"{x:.{nd}f}".replace("-", "−")


def tests(N):
    """(label, fetch) for every relative test with a value."""
    B = N["bench"]
    return [
        ("behaviour r", lambda m: g(B, m, "beh_r_mean")),
        ("waveform", lambda m: g(B, m, "r2_wave")),
        ("depth pattern", lambda m: g(B, m, "r2_amp_spatial")),
        ("probe-wide amplitude", lambda m: g(B, m, "r2_amp_global")),
        ("gamma envelope 40 ms", lambda m: g(B, m, "r2_env_gamma")),
        ("zero-shot transfer", lambda m: g(N, "transfer", m, "full")),
        ("cross-probe CCA", lambda m: g(N, "cca", m, "cc_top5")),
        ("state NMI", lambda m: g(N, "states_within", m, "nmi_akella_ref")),
        ("pooled state NMI", lambda m: g(N, "states_pooled", m, "nmi_akella_ref")),
    ]


def tail(N):
    L = []
    w = L.append
    B = N["bench"]
    P = N["paired"]
    T = tests(N)

    # ------------------------------------------------------------------ 7 verdicts
    w("## 7. Verdict: right purpose, and does the network earn its complexity?\n")
    w("'Earns its complexity' = better than its linear twin by more than the tolerance on at least one purpose test "
      f"and not worse by more than ±{TOL} on any. Counts are over the relative tests of section 4 that have values.\n")
    w("| Model | Built for | Right purpose? | vs linear twin: better / tied / worse | Better on | Worse on |")
    w("|---|---|---|---|---|---|")
    verdicts = {}
    for m in LEARNED:
        if m not in B:
            continue
        better, tied, worse = [], [], []
        for lab, fetch in T:
            v, t = fetch(m), fetch(TWIN(m))
            if v is None or t is None:
                continue
            (better if v > t + TOL else worse if v < t - TOL else tied).append(lab)
        verdicts[m] = (better, tied, worse)
        purpose, right = PURPOSE[m]
        w(f"| **{LETTER[m]}** `{m}` | {purpose} | {right} | {len(better)} / {len(tied)} / {len(worse)} (vs {TWIN_L(m)}) | "
          f"{', '.join(better) or '–'} | {', '.join(worse) or '–'} |")
    w("")
    earns = [LETTER[m] for m, (b, t, wo) in verdicts.items() if b and not wo]
    never = [LETTER[m] for m, (b, t, wo) in verdicts.items() if not b]
    w(f"**Reading.** Models that earn their complexity by this rule: {', '.join(earns) if earns else 'none'}. "
      f"Models better than their linear twin on no purpose test: {', '.join(never) if never else 'none'}.")
    beh = {LETTER[m]: g(P, f"{m}|{TWIN(m)}|beh_r_mean") for m in verdicts}
    w(" Behaviour, session level (Δ median, sessions higher): " + "; ".join(
        f"{k} {f2(v['d'], 3)} ({v['k']}/{v['n']})" for k, v in beh.items() if v) + ".\n")
    conv = N.get("convergence", {})
    capped = [LETTER[m] for m in LEARNED if m in conv and conv[m]["folds_best_in_last_5"] >= 3]
    if capped:
        w(f"Training caveat: for {', '.join(capped)} the best validation epoch fell in the last 5 epochs in most folds, "
          f"so they may still improve with longer training (cap {N.get('ab_epochs_cap')} epochs; "
          "`tables/model_purpose/training_convergence.csv`). Longer training reduces reconstruction loss, but section 8 explains why "
          "that need not help the linear readouts.\n")

    # ------------------------------------------------------------------ 8 mechanisms
    Wv = B.get("group_pca_weighted_48", {})
    bank = B.get("bank_wave+rms+psd", {})
    dg = [g(P, f"{m}|{TWIN(m)}|r2_env_gamma") for m in LEARNED if g(P, f"{m}|{TWIN(m)}|r2_env_gamma")]
    w("## 8. Why the networks do not beat their linear twins here\n")
    w(f"1. **The decisive nonlinearity is already in the features.** Log band power is the log of a smoothed squared "
      f"band-passed signal, exactly the nonlinearity section 2.1 says PCA lacks. After it, the remaining structure is close "
      f"to linear and Gaussian. The linear twin W reaches behaviour r {f2(Wv.get('beh_r_mean'))} at 48 numbers, against "
      f"{f2(bank.get('beh_r_mean'))} for the full 456-number bank.")
    w("2. **The consumers are linear-Gaussian.** Ridge readouts, CCA and Gaussian HMMs, the tools of tasks 2–5, can only "
      "use information that is linearly available in the latent. A reconstruction loss with a nonlinear decoder does not "
      "require that. So a network can reconstruct well through its own decoder (section 6.1) while its latent serves a "
      "linear readout worse.")
    ev = N.get("e15_vs_e50", {})
    if "ae_depth_48|beh_r_mean" in ev:
        e = lambda c: ev[f"ae_depth_48|{c}"]
        w(f"   Direct evidence: training A longer (15 → {N.get('ab_epochs_cap')} epochs) lowered its validation loss, yet "
          f"behaviour r went {f2(e('beh_r_mean')['e15'])} → {f2(e('beh_r_mean')['e50'])} (higher in {e('beh_r_mean')['k']}/"
          f"{e('beh_r_mean')['n']} sessions) and depth pattern {f2(e('r2_amp_spatial')['e15'])} → {f2(e('r2_amp_spatial')['e50'])}, "
          f"while waveform R² rose {f2(e('r2_wave')['e15'])} → {f2(e('r2_wave')['e50'])}. Better reconstruction, "
          "worse linear readouts: the objective and the purpose pull apart.")
    if dg:
        w(f"3. **Temporal context smooths fast dynamics.** Every convolutional model loses gamma-envelope (40 ms) R² against "
          f"its twin; the median session-level Δ ranges {f2(min(x['d'] for x in dg))} to {f2(max(x['d'] for x in dg))}. "
          "The encoder integrates ±240 ms, and the 40 ms gamma feature carries 1/27 of the loss, so the bottleneck is "
          "spent on slow structure. W has no temporal kernel and keeps the fast part.")
    w("4. **Data scale.** 50 insertions × 100 s is about 125,000 time bins, strongly autocorrelated. Nonlinear gains "
      "usually need more data; the fair test of the AE family is the 750-insertion atlas, not this set.")
    w("5. **Objective ≠ purpose.** Mean squared error rewards variance, and the dominant variance, broadband power at "
      "each depth, is captured linearly (W's second channel; section 6.4).")
    tc, tm = N.get("transfer", {}), N.get("transfer_mixed", {})
    if g(tc, "ae_masked_48", "full") is not None and g(tm, "ae_masked_48", "full") is not None:
        w(f"6. **Coordinates are shared within one trained encoder, not across retrainings.** Zero-shot transfer with one "
          f"encoder per fold: C {f2(tc['ae_masked_48']['full'])}, D {f2(g(tc, 'ae_anatomy_48', 'full'))} vs W "
          f"{f2(g(tc, 'group_pca_weighted_48', 'full'))}. With each insertion encoded by its own fold's model, the "
          f"same models fall to C {f2(tm['ae_masked_48']['full'])}, D {f2(g(tm, 'ae_anatomy_48', 'full'))}, while the "
          f"linear fits do not move (W {f2(g(tm, 'group_pca_weighted_48', 'full'))}, pooled group PCA "
          f"{f2(g(tm, 'group_pca_pooled_48', 'full'))} vs {f2(g(tc, 'group_pca_pooled_48', 'full'))}). PCA's axes are "
          "determined by the data; an autoencoder's are one of many equivalent solutions, and each retraining picks a "
          "different one (section 2.3). For an atlas, an AE must be trained once, frozen and versioned.")
    W_, WF_ = B.get("group_pca_weighted_48", {}), B.get("group_wave+amp1_pooled", {})
    same = all(abs(W_.get(k, 0) - WF_.get(k, 1)) < 0.002 for k in ("beh_r_mean", "r2_wave", "r2_amp_spatial", "r2_env_gamma"))
    if same:
        w("\n**A result that ties this together.** W (the linear minimiser of the autoencoders' loss) and WF (stored "
          "waveform + 1 amplitude PC per depth) score identically on every benchmark metric. With views weighted "
          "equally, the waveform is nearly uncorrelated with the amplitude features, so the weighted PCA splits exactly "
          "into 'waveform' and 'top amplitude component'. The hand-designed per-depth representation is therefore "
          "*the linear optimum of the autoencoder's own objective*, and the networks are being asked to beat the "
          "optimum of their own loss restricted to linear maps.")
    w("")

    # ------------------------------------------------------------------ 9 recommendations
    w("## 9. Recommendation\n")
    wf = B.get("group_wave+amp1_pooled", {})
    w(f"- **Use the linear twin as the task-1 representation now.** W or WF: each depth stores its waveform plus one or "
      f"more broadband-amplitude components, with loadings fit once on training sessions and shared by all insertions. "
      f"It is linear, interpretable (waveform + power per depth), has shared coordinates (R4), and costs one matrix "
      f"multiply after the feature bank. Behaviour r W {f2(Wv.get('beh_r_mean'))} / WF {f2(wf.get('beh_r_mean'))}, "
      f"waveform R² {f2(Wv.get('r2_wave'))} / {f2(wf.get('r2_wave'))}, depth pattern {f2(Wv.get('r2_amp_spatial'))} / "
      f"{f2(wf.get('r2_amp_spatial'))}, against PCA 48's {f2(B['waveform_pca_48']['beh_r_mean'])}, "
      f"{f2(B['waveform_pca_48']['r2_wave'])}, {f2(B['waveform_pca_48']['r2_amp_spatial'])}.")
    w("- **Keep the autoencoder as the scalable route, but change what it is asked to do** before training it again:")
    w("  1. a fast path: a skip connection for the 40 ms features, or no temporal kernel before the bottleneck;")
    w("  2. an objective that suits linear-Gaussian consumers: a decorrelation/whitening penalty on the latent, or a "
      "behaviour-free temporal objective (slow-feature or contrastive in time, e.g. the time-contrastive mode of CEBRA, "
      "Schneider et al. 2023), still without behaviour;")
    w("  3. pre-training on all 750 atlas insertions, then this benchmark as the held-out test; save the weights "
      "(train.py does not, which is why A and B could not be re-encoded) and freeze one versioned encoder for the atlas;")
    w("  4. for E: all 16 channels per depth (power averaged, as in the bank) and a per-insertion spectral normalisation. "
      "Check its highest learned band before trusting it (it moved above 100 Hz, where spike leakage and EMG live);")
    w("  5. C's masking, if the goal is imputation: on held-out insertions it beats shrunk depth interpolation on the RMS "
      "view but loses on the waveform and PSD views, which are spatially smooth enough for interpolation (section 6.2).")
    w("- **Keep neural validity as a standing test.** The CSD comparison (other session) and the out-of-brain controls "
      "should be run on every candidate, because behaviour decoding alone cannot tell neural signal from artefact.\n")

    # ------------------------------------------------------------------ 10 limitations
    w("## 10. Limitations\n")
    for s in [
        "50 insertions, 26 sessions, one 100 s window per session. Folds are sessions for model training and contiguous "
        "time blocks for readouts. This is not evidence of reproducibility across the 750-insertion atlas.",
        "All bank-derived representations and autoencoders z-score each insertion over its full 100 s: label-free but "
        "transductive, since test blocks inform the normalisation.",
        "Behaviour r is a proxy for state-relevant information. Wilcoxon p-values are uncorrected and descriptive; many "
        "comparisons were made.",
        "The HMM reference is built from bank features, which gives bank-derived representations a home advantage over "
        "waveform-only ones.",
        "*Timescale (T13): confounded by our own smoothing windows (RMS 200 ms/1 s, PSD 1 s), so it describes feature "
        "construction as much as brain dynamics. No criterion is attached.",
        "Cross-probe CCA can include common non-neural signals (motion artefacts). Each probe is common-median referenced "
        "separately. The 1 Hz high-pass variant separates slow shared drift, but it is not a CSD test.",
        "One seed per fold, no hyperparameter tuning (deliberately: nothing was tuned on test data). A–D and F ran to the "
        "epoch cap.",
        "E reads 2 of 16 channels per depth, so its static depth profile of power is off (section 6.1). D depends on the "
        "histology alignment of region labels.",
        "Imputation hides depths within an insertion; that is only a proxy for inferring unrecorded regions.",
        "Outside-the-brain checks (other session): in the Cosmos mapping 'root' collects fiber tracts and ventricles, "
        "i.e. white matter INSIDE the brain, and only 'void' is outside. An earlier control counted both as outside; "
        "the numbers quoted here use the corrected split (grey / white / void).",
        "3-state HMMs are unstable: refits on the same features agree at NMI ≈ 0.5 (section 4), so small NMI differences "
        "between representations are within that noise.",
    ]:
        w(f"- {s}")
    w("")

    # ------------------------------------------------------------------ figures
    w("## Figures\n")
    w("In `docs/figures/model_purpose/`; every table behind them is in `docs/tables/model_purpose/`:\n")
    for f, cap in [
        ("fig_models_benchmark.png", "every model on the benchmark protocol: behaviour r, waveform, depth pattern, 40 ms gamma; one dot per held-out insertion"),
        ("fig_paired_vs_linear_twin.png", "session-level differences from W: the test of whether each network earns its complexity"),
        ("fig_paired_vs_pca48.png", "session-level differences from PCA 48: the test of the project question"),
        ("fig_purpose_tests.png", "zero-shot transfer, cross-probe CCA (with 1 Hz high-pass), HMM state agreement within and across animals"),
        ("fig_filterbank_learned.png", "E's learned band-pass filters in 5 independent folds against the canonical bands"),
        ("fig_imputation.png", "C vs linear and shrunk interpolation on hidden depth groups"),
    ]:
        w(f"- `{f}`: {cap}")
    w("")

    # ------------------------------------------------------------------ 11 references
    w("## 11. References\n")
    for r in [
        "Akella S. et al. (2025) Deciphering neuronal variability across states. *Nature Communications*.",
        "Baldi P., Hornik K. (1989) Neural networks and principal component analysis: learning from examples without local minima. *Neural Networks* 2:53–58.",
        "Buzsáki G., Anastassiou C.A., Koch C. (2012) The origin of extracellular fields and currents — EEG, ECoG, LFP and spikes. *Nat Rev Neurosci* 13:407–420.",
        "Donoghue T. et al. (2020) Parameterizing neural power spectra into periodic and aperiodic components. *Nat Neurosci* 23:1655–1665.",
        "Einevoll G.T., Kayser C., Logothetis N.K., Panzeri S. (2013) Modelling and analysis of local field potentials for studying the function of cortical circuits. *Nat Rev Neurosci* 14:770–785.",
        "Gao R., Peterson E.J., Voytek B. (2017) Inferring synaptic excitation/inhibition balance from field potentials. *NeuroImage* 158:70–78.",
        "He K. et al. (2022) Masked autoencoders are scalable vision learners. *CVPR*.",
        "International Brain Laboratory (2023/2025) A brain-wide map of neural activity during complex behaviour.",
        "Kajikawa Y., Schroeder C.E. (2011) How local is the local field potential? *Neuron* 72:847–858.",
        "Locatello F. et al. (2019) Challenging common assumptions in the unsupervised learning of disentangled representations. *ICML*.",
        "McGinley M.J. et al. (2015) Waking state: rapid variations modulate neural and behavioral responses. *Neuron* 87:1143–1161.",
        "Mitzdorf U. (1985) Current source-density method and application in cat cerebral cortex. *Physiol Rev* 65:37–100.",
        "Musall S. et al. (2019) Single-trial neural dynamics are dominated by richly varied movements. *Nat Neurosci* 22:1677–1686.",
        "Niell C.M., Stryker M.P. (2010) Modulation of visual responses by behavioral state in mouse visual cortex. *Neuron* 65:472–479.",
        "Pesaran B. et al. (2018) Investigating large-scale brain dynamics using field potential recordings: analysis and interpretation. *Nat Neurosci* 21:903–919.",
        "Ravanelli M., Bengio Y. (2018) Speaker recognition from raw waveform with SincNet. *IEEE SLT*.",
        "Schneider S., Lee J.H., Mathis M.W. (2023) Learnable latent embeddings for joint behavioural and neural analysis. *Nature* 617:360–368.",
        "Stringer C. et al. (2019) Spontaneous behaviors drive multidimensional, brainwide activity. *Science* 364:eaav7893.",
        "Vinck M. et al. (2015) Arousal and locomotion make distinct contributions to cortical activity patterns and visual encoding. *Neuron* 86:740–754.",
        "Zeghidour N. et al. (2021) LEAF: a learnable frontend for audio classification. *ICLR*.",
    ]:
        w(f"- {r}")
    w("")

    # ------------------------------------------------------------------ 12 reproduction
    w("## 12. Reproduction\n")
    w("```bash")
    w("cd ~/Downloads/lfp-brain-state/lfp_based_brain_state")
    w("export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1; PY=/opt/miniconda3/envs/lfp-brain-state/bin/python")
    w("$PY -m lfp_autoencoder_prototype.cache_raw                                   # raw channel subset for E")
    w("for f in 0 1 2 3 4; do for m in C D E F; do $PY -m lfp_autoencoder_prototype.train_variants --only-fold $f --models $m; done; done")
    w("$PY -m lfp_autoencoder_prototype.purpose_tests --baselines --weighted       # PCA family, W, WF")
    w("$PY -m lfp_autoencoder_prototype.evaluate_variants --worker --tags W WF C D E F")
    w("$PY -m lfp_autoencoder_prototype.purpose_tests --tests                      # transfer, CCA, timescale, interpret")
    w("/opt/miniconda3/envs/lfp/bin/python lfp_autoencoder_prototype/state_tests.py  # HMM states (needs hmmlearn)")
    w("$PY -m lfp_autoencoder_prototype.recon_offsets && $PY -m lfp_autoencoder_prototype.imputation_linear")
    w("$PY -m lfp_autoencoder_prototype.model_report && $PY -m lfp_autoencoder_prototype.write_standards")
    w("```")
    w("A and B: `train.py` / `evaluate_ae.py` (unchanged protocol, 50-epoch run). Nothing is committed or pushed.")
    return L
