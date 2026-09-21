# What the representation is for: theory, standards and tests for PCA and six autoencoder designs

Skylar Tian · generated from result files by `lfp_autoencoder_prototype/write_standards.py` · Using 50 selected insertions (26 sessions, 9 labs), 48 numbers per 40 ms unless stated.

> **Status.** All 10 columns of the results matrix are complete. A and B come from the 50-epoch retrain when `ae_results/probes` holds it (epoch cap in the files read here: 50). Results are **completed** measurements on held-out sessions unless a section says *preliminary* or *proposed*. Nothing here was used to tune the models.

## 1. The purpose, stated as requirements

Task 1 builds the measurement layer that tasks 2–5 stand on. A representation is fit for purpose only if it serves those tasks, so the requirements come from them, not from what is easy to score:

| # | Requirement | Which project task needs it | Why, neuroscientifically |
|---|---|---|---|
| R1 | Compact and fast (≤ 48 numbers / 40 ms; encodes 100 s in seconds on a CPU) | All: 750 insertions × hours | Brain-wide scale; the raw LF band is 384 ch × 2.5 kHz |
| R2 | Keeps state-relevant spectral content: band power delta→gamma and the aperiodic slope | 3 (organisation of state), 5 (atlas) | Brain states are spectral: desynchronisation lowers low-frequency power and raises gamma with arousal and locomotion; hippocampal theta tracks running; the aperiodic exponent tracks E/I balance |
| R3 | Keeps location: every number tied to a depth that maps to a region | 2 (local vs global), 4 (anatomy), 5 (atlas) | High-frequency power is local; low-frequency potential is volume-conducted from elsewhere |
| R4 | Shared coordinates: the same number means the same thing in every insertion and animal | 5 (across-animal atlas), 3 (similarity across animals) | An atlas pools animals; per-insertion axes (sign, rotation) cannot be pooled |
| R5 | Keeps both the component shared between regions and the private one | 2 (does a brain-wide latent exist?) | Global arousal signals coexist with local circuit dynamics |
| R6 | Keeps timescales: no smearing of fast dynamics, no invented slow ones | 3 (timescales, discrete vs continuous) | State dwell times run from ~0.1 s (gamma bursts) to minutes (arousal) |
| R7 | Learned without behaviour | 3, 5: states are defined from neural data, then related to behaviour | Otherwise the state→behaviour link is circular |
| R8 | Represents neural signal, not artefact | All | Movement artefacts, EMG and far-field potentials also correlate with behaviour |
| R9 | Interpretable, stable axes | 4, 5: relate to physiology; compare across retrainings | A latent that means 'broadband power at this depth' can be related to anatomy; an arbitrary mixture cannot |

## 2. Theory

### 2.1 Why waveform PCA cannot meet R2–R4, however well it reconstructs

Waveform PCA 48 keeps waveform R² **0.99** yet recovers amplitude depth pattern R² **−0.01** and gamma-envelope R² **0.04**. That is not a tuning problem; it follows from three facts:

1. **Power is quadratic in the signal.** Band power is the smoothed square of a band-passed signal. Every PCA coordinate is a linear function of the signal, and so is every ridge readout of those coordinates, with or without time lags. No linear function equals a positive quadratic form, so amplitude, the quantity brain states are defined by (R2), is out of reach of any linear readout of waveform PCs.
2. **The 25 Hz waveform has no beta or gamma, and some of what it has is aliased.** A 40 ms boxcar keeps < 12.5 Hz. The other session measured that **14.9%** of the 8–12.5 Hz power in the benchmark waveform is aliased content from above 13 Hz (theta 2.1%, delta 0.2%; median of 50 insertions; `lfp_selected_benchmark/aliasing.py`).
3. **PCA keeps variance, and LFP variance is low-frequency and far-field.** With a ~1/f spectrum, most variance is slow, and slow potentials are volume-conducted over millimetres (Kajikawa & Schroeder 2011; Buzsáki, Anastassiou & Koch 2012). The top PCs are therefore smooth, probe-wide modes. Local, high-frequency activity carries little variance and is the first thing dropped (R3). Per-insertion PCs are also defined only up to sign and rotation, so they share no coordinates across animals (R4).

So PCA is the right answer to *compress the waveform* and the wrong one to *represent brain state*. This is why our gains come from changing the **input** (log band power per depth) before changing the **model**.

### 2.2 Why per-depth, multi-scale log band power is the right substrate

- **State is spectral.** Arousal and locomotion desynchronise cortex: low-frequency power falls and gamma rises (Niell & Stryker 2010; Vinck et al. 2015; McGinley et al. 2015). Akella et al. (2025) define their three states from exactly such band envelopes.
- **Log.** Power is multiplicative and roughly log-normal. On a log scale the 1/f spectrum becomes additive, so each band contributes comparably instead of delta dominating.
- **At least one cycle per window.** Resolving frequency f needs roughly 1/f seconds (time–frequency uncertainty). Hence delta and theta at 1 s, beta at 200 ms and 1 s, gamma down to 40 ms.
- **Aperiodic slope.** The 1/f exponent tracks excitation/inhibition balance (Gao, Peterson & Voytek 2017) and has to be separated from oscillatory peaks (Donoghue et al. 2020).
- **Locality.** The current-source density (second spatial difference) removes far-field potential (Mitzdorf 1985; Pesaran et al. 2018). The other session found that grey-matter CSD band power decodes behaviour at r = **0.568** vs **0.548** for the potential (50 insertions; `neural_validity.py`), so the behavioural signal is at least as strong in the local component. White-matter groups decode about as well as matched grey groups (0.47 vs 0.46; 11 insertions), consistent with spread from nearby grey matter. Out-of-brain groups decode less (0.23 vs 0.39), but on only 4 insertions.
- **Caveat that shapes the tests.** Movement drives activity brain-wide (Stringer et al. 2019; Musall et al. 2019), and artefacts correlate with movement too. High behaviour decoding is expected almost anywhere and is not, on its own, evidence of a good state representation. That is why behaviour r is one row of the table below, not the criterion.

### 2.3 What the autoencoders optimise, and when that matches the purpose

All five designs minimise the same loss: mean squared error on the z-scored 19-feature bank at 24 depths. Views are weighted equally (waveform ⅓, nine RMS ⅓, nine PSD ⅓), there are 48 latents per 40 ms, and behaviour is never seen (R7 ✓).

- **The linear twin.** A linear autoencoder trained with squared error spans the PCA subspace of its input (Baldi & Hornik 1989). The linear, context-free minimiser of *our* loss is therefore group PCA on features scaled by √(view weight), fit on the training sessions: **W**. Plain group PCA gives the waveform 1/19 of the variance instead of ⅓, so it answers a different objective. Comparing A with plain group PCA mixes a change of objective with a change of model; comparing A with **W** isolates what the network adds (nonlinearity plus ±240 ms and ±320 µm of context).
- **What the dominant covariance is.** On held-out insertions, W's two latents per depth are the waveform and broadband power (all bands moving together). Broadband power is the classic desynchronisation/arousal axis. Without view weighting, the second axis is instead an RMS-vs-PSD contrast (section 6.4).
- **What MSE does not know.** Reconstruction keeps variance, not state relevance. Slow drifts, artefacts and far-field signals survive if they are large. This is why the neural-validity checks and the shared-coordinate tests are needed on top of reconstruction.
- **Identifiability.** An autoencoder's latents are defined only up to an invertible reparametrisation the decoder can undo (see Locatello et al. 2019 on the general problem). Within one trained encoder, coordinates are shared across insertions (R4). Across retrainings they are not, so an atlas needs a frozen, versioned encoder or a canonicalised latent.

**Hypothesis behind each design, and what would falsify it**

| Design | Inductive bias | Hypothesis | Falsified if |
|---|---|---|---|
| **A** per-depth | same computation at every depth (shared weights), ±240 ms, ±320 µm | nonlinearity + context add information a linear per-depth summary misses | A ≤ W on behaviour, depth pattern, transfer and states |
| **B** global | depths flattened into 48 shared latents | a global summary averages noise across depths | (not a spatial representation by construction: fails R3 whatever it scores) |
| **C** masked | hide 25% of depth groups in training, reconstruct all | learning spatial predictability helps infer unrecorded depths (a small version of 'infer unrecorded regions') | C ≤ linear interpolation in depth on hidden groups |
| **D** anatomy | learned embedding of the Cosmos region per depth, in encoder and decoder | explaining region identity frees the latent to encode state | D ≤ A on transfer, states and behaviour |
| **E** filterbank | 8 Gaussian band-pass filters learned on the raw 250 Hz signal (LEAF/SincNet-style); no hand-made features in the input | our band choices are not optimal, and a learned front end recovers the state information itself | E ≤ A everywhere, or learned bands wander between folds |
| **F** wave + amp | waveform stored as is (24); 1 learned amplitude latent per depth from the 18 RMS/PSD features | spend nonlinear capacity only where the physics needs it (amplitude), since the sub-12.5 Hz waveform is spatially smooth | F ≤ WF (stored waveform + 1 pooled amplitude PC per depth) |

## 3. The standards and tests

Criteria were fixed before the C, D, E, W and purpose-test results existed, though after the baseline benchmark and the 15-epoch A/B results. F and its twin WF were added during the run, after W's first result. The comparator for any learned model is **W** (does the network earn its complexity?); tolerance ±0.02 for R²/r metrics. The comparator for the project question is **PCA 48** (does it improve on the current method at the same budget?).

| # | Standard | Requirement | Test | Metric | Pass criterion |
|---|---|---|---|---|---|
| T1 | Compact | R1 | numbers stored per second | dims × 25 Hz | ≤ 1,200 floats/s (48 dims) |
| T2 | Waveform fidelity | R1, R6 | ridge from representation to 384-ch 25 Hz waveform, 4 blocked folds | R² (pooled) | ≥ 0.90 |
| T3 | Fast amplitude | R2, R6 | ridge to 40 ms gamma log-RMS envelope | R² | ≥ W − 0.02 |
| T4 | Probe-wide amplitude | R2 | ridge to 200 ms beta/gamma log-RMS per group | R² | ≥ W − 0.02 |
| T5 | Amplitude depth pattern | R3 | same, minus the probe mean (where on the probe) | R² | ≥ W − 0.02 and ≥ PCA 48 + 0.2 |
| T6 | Behavioural information | R2 (proxy) | ridge with ±320 ms lags → 5 behaviours | mean r | ≥ W − 0.02 (paired sessions reported) |
| T7 | Null control | validity | behaviour circularly shifted 30 s | mean r | |r| ≤ 0.03 |
| T8 | Shared coordinates | R4 | zero-shot: readout trained on the other 25 sessions, applied unchanged | mean r | ≥ W − 0.02 |
| T9 | Depth-invariant transfer | R3 + R4 | zero-shot with a readout that only sees mean and SD over depths | mean r | reported (depth-indexed only) |
| T10 | Cross-region shared component | R5 | cross-validated CCA between the 2 simultaneous probes (24 pairs) | mean of top-5 held-out canonical r | ≥ W − 0.02 and > shift null |
| T11 | State structure | R2, R6 | 3-state HMM vs Akella-style reference HMM (per insertion) | NMI | ≥ W − 0.02 |
| T12 | States across animals | R4 | one HMM fit on other sessions, decoded on the held-out one | NMI with pooled reference | ≥ W − 0.02 |
| T13 | Timescale | R6 | 1/e autocorrelation time of the dimensions | s (median) | reported |
| T14 | Location | R3 | structural: are numbers tied to depths? | yes/no | yes |
| T15 | Behaviour-free | R7 | structural | yes/no | yes |
| T16 | Interpretable axis | R9 | max |corr| of a latent channel with one bank feature (held-out) | |r| | ≥ 0.5 |
| T17 | Spatial imputation | R3, research goal | hide depth groups of held-out insertions (C only) | R² on hidden groups | > depth interpolation |
| T18 | Cost | R1 | params; CPU train min/fold; s to encode 100 s (+13 s for the feature bank where needed) | – | train ≤ 60 min/fold |

## 4. Results matrix

Medians over held-out insertions (T1–T9, T11–T13, T16) or simultaneous pairs (T10). ✓/✗ = criterion met or not. Relative criteria compare each learned model with its linear twin (W for A–E, WF for F); linear columns are the references and carry no relative mark. – = not applicable or not available.

| Test | PCA 48 | Group PCA (pooled) | **W** twin of A–E | **WF** twin of F | **A** per-depth | **B** global | **C** masked | **D** anatomy | **E** filterbank | **F** wave + amp |
|---|---|---|---|---|---|---|---|---|---|---|
| T1 floats/s | 1,200 ✓ | 1,200 ✓ | 1,200 ✓ | 1,200 ✓ | 1,200 ✓ | 1,200 ✓ | 1,200 ✓ | 1,200 ✓ | 1,200 ✓ | 1,200 ✓ |
| T2 waveform R² | 0.99 ✓ | 0.07 ✗ | 0.96 ✓ | 0.96 ✓ | 0.86 ✗ | 0.70 ✗ | 0.82 ✗ | 0.88 ✗ | 0.84 ✗ | 0.96 ✓ |
| T3 gamma env R² | 0.04 | 0.52 | 0.45 | 0.45 | 0.22 ✗ | 0.31 ✗ | 0.21 ✗ | 0.24 ✗ | 0.12 ✗ | 0.23 ✗ |
| T4 amplitude R² | 0.05 | 0.99 | 0.81 | 0.81 | 0.59 ✗ | 0.91 ✓ | 0.72 ✗ | 0.65 ✗ | 0.50 ✗ | 0.47 ✗ |
| T5 depth-pattern R² | −0.01 | 0.45 | 0.35 | 0.35 | 0.26 ✗ | 0.36 ✓ | 0.30 ✗ | 0.29 ✗ | 0.15 ✗ | 0.13 ✗ |
| T6 behaviour r | 0.34 | 0.58 | 0.58 | 0.58 | 0.55 ✗ | 0.61 ✓ | 0.57 ✓ | 0.56 ✗ | 0.55 ✗ | 0.54 ✗ |
| T7 shift null r | 0.00 ✓ | −0.01 ✓ | −0.01 ✓ | −0.01 ✓ | −0.01 ✓ | −0.01 ✓ | −0.00 ✓ | −0.01 ✓ | −0.01 ✓ | 0.00 ✓ |
| T8 zero-shot r (one encoder per fold) | −0.00 | 0.51 | 0.44 | 0.43 | – | – | 0.44 ✓ | 0.42 ✓ | 0.32 ✗ | 0.39 ✗ |
| T8m zero-shot r, mixed fold encoders† | −0.00 | 0.51 | 0.44 | 0.44 | 0.17 | 0.23 | −0.04 | 0.05 | −0.02 | 0.08 |
| T9 depth-invariant zero-shot r | – | 0.33 | 0.30 | 0.30 | – | – | 0.36 | 0.34 | 0.32 | 0.34 |
| T10 CCA top-5 (null) | 0.83 (0.06) | 0.61 (0.10) | 0.72 (0.10) | 0.72 (0.10) | 0.65 ✗ (0.06) | 0.74 ✓ (0.08) | 0.68 ✗ (0.07) | 0.68 ✗ (0.06) | 0.68 ✗ (0.07) | 0.64 ✗ (0.04) |
| T11 state NMI | 0.16 | 0.28 | 0.30 | 0.31 | 0.29 ✓ | 0.31 ✓ | 0.32 ✓ | 0.29 ✓ | 0.21 ✗ | 0.24 ✗ |
| T12 pooled state NMI (one encoder per fold) | 0.03 | 0.11 | 0.07 | 0.07 | – | – | 0.07 ✓ | 0.06 ✓ | 0.05 ✗ | 0.06 ✓ |
| T10b CCA top-5 after 1 Hz high-pass | 0.78 | 0.43 | 0.61 | 0.60 | 0.57 | 0.63 | 0.58 | 0.59 | 0.64 | 0.52 |
| T13 timescale (s)* | 0.12 | 0.44 | 0.20 | 0.20 | 0.08 | 0.28 | 0.08 | 0.08 | 0.12 | 0.08 |
| T14 location | no ✗ | yes ✓ | yes ✓ | yes ✓ | yes ✓ | no ✗ | yes ✓ | yes ✓ | yes ✓ | yes ✓ |
| T15 behaviour-free | yes ✓ | yes ✓ | yes ✓ | yes ✓ | yes ✓ | yes ✓ | yes ✓ | yes ✓ | yes ✓ | yes ✓ |
| T16 max \|r\| latent↔feature | – | 0.72 ✓ | 1.00 ✓ | 1.00 ✓ | 0.52 ✓ | – | 0.38 ✗ | 0.24 ✗ | 0.53 ✓ | 1.00 ✓ |
| T17 imputation, random 25 %: insertions where C > shrunk interpolation (per view) | – | – | – | – | – | – | psd 1/50 · rms 50/50 · waveform 4/50 | – | – | – |
| T18 params · train/fold | linear (seconds) | linear (seconds) | linear (seconds) | linear (seconds) | 16,197 p · 28 min ✓ | 71,995 p · 27 min ✓ | 16,221 p · 20 min ✓ | 16,437 p · 20 min ✓ | 16,407 p · 25 min ✓ | 16,099 p · 17 min ✓ |

† T8m trains the readout on insertions encoded by the other folds' models and tests on the held-out fold's own model, so it asks whether separately trained encoders agree (identifiability, R9), not whether one encoder gives shared coordinates (T8). For learned models, T8, T9 and T12 re-encode every insertion with one fold's model (`consistent_latents.py`). That is impossible for A and B, which were trained without saving weights, hence –. Methods with no trained encoder use their own fit per insertion (PCA 48) or per fold, so T8 and T8m coincide for them.

HMM ceiling: refitting the same 3-state HMM on the reference features themselves (a PCA rotation and re-standardisation, different local optimum) agrees with the reference at NMI **0.48**. Read T11 against that ceiling, not against 1.

Feature cost, one 100 s insertion, 1 thread, measured while other jobs ran (upper bounds): computing the 456-feature bank takes **13.4 s** (needed by group PCA, W, wave+amp, A–D); waveform PCA 48 fit + encode **0.36 s**. Encoding with a trained AE adds ae_anatomy_48 0.09 s, ae_filterbank_48 0.38 s, ae_masked_48 0.08 s, ae_waveamp_48 0.06 s (E reads raw and needs no bank; A and B were not timed because train.py saves no weights, but they share C/D's trunk). Training times are CPU minutes per fold with 5–8 jobs sharing 4 physical cores.

## 5. Paired, session-level comparisons

Per-session mean difference (simultaneous probes averaged, so sessions are the independent unit). k/n = sessions where the model is higher; p = two-sided Wilcoxon signed-rank over sessions, uncorrected and descriptive only.

| Model | vs | Δ behaviour r | k/n | p | Δ depth-pattern R² | k/n | Δ waveform R² | k/n |
|---|---|---|---|---|---|---|---|---|
| ae_depth_48 | W | −0.026 | 1/26 | 0.000 | −0.089 | 1/26 | −0.083 | 0/26 |
| ae_depth_48 | PCA 48 | 0.216 | 25/26 | 0.000 | 0.283 | 26/26 | −0.124 | 0/26 |
| ae_global_48 | W | 0.016 | 23/26 | 0.000 | 0.030 | 19/26 | −0.239 | 0/26 |
| ae_global_48 | PCA 48 | 0.255 | 26/26 | 0.000 | 0.381 | 26/26 | −0.287 | 0/26 |
| ae_global_48 | A | 0.047 | 26/26 | 0.000 | 0.117 | 24/26 | −0.166 | 0/26 |
| ae_masked_48 | W | −0.019 | 5/26 | 0.001 | −0.053 | 6/26 | −0.092 | 0/26 |
| ae_masked_48 | PCA 48 | 0.224 | 25/26 | 0.000 | 0.309 | 26/26 | −0.166 | 0/26 |
| ae_masked_48 | A | 0.015 | 20/26 | 0.000 | 0.028 | 20/26 | −0.010 | 9/26 |
| ae_anatomy_48 | W | −0.024 | 4/26 | 0.000 | −0.076 | 2/26 | −0.067 | 0/26 |
| ae_anatomy_48 | PCA 48 | 0.217 | 25/26 | 0.000 | 0.300 | 26/26 | −0.101 | 0/26 |
| ae_anatomy_48 | A | 0.003 | 16/26 | 0.468 | 0.024 | 15/26 | 0.023 | 17/26 |
| ae_filterbank_48 | W | −0.043 | 4/26 | 0.000 | −0.183 | 0/26 | −0.097 | 0/26 |
| ae_filterbank_48 | PCA 48 | 0.187 | 26/26 | 0.000 | 0.171 | 26/26 | −0.131 | 0/26 |
| ae_filterbank_48 | A | −0.014 | 10/26 | 0.075 | −0.097 | 4/26 | 0.009 | 17/26 |
| ae_waveamp_48 | W | −0.030 | 2/26 | 0.000 | −0.182 | 0/26 | 0.000 | 19/26 |
| ae_waveamp_48 | WF | −0.030 | 2/26 | 0.000 | −0.182 | 0/26 | 0.000 | 19/26 |
| ae_waveamp_48 | PCA 48 | 0.191 | 24/26 | 0.000 | 0.151 | 26/26 | −0.023 | 0/26 |
| ae_waveamp_48 | A | −0.001 | 10/26 | 0.208 | −0.100 | 2/26 | 0.084 | 26/26 |
| group_wave+amp1_pooled | W | −0.000 | 13/26 | 0.803 | −0.000 | 10/26 | 0.000 | 21/26 |
| group_wave+amp1_pooled | PCA 48 | 0.240 | 25/26 | 0.000 | 0.361 | 26/26 | −0.023 | 0/26 |
| group_pca_pooled_48 | W | −0.008 | 9/26 | 0.247 | 0.104 | 26/26 | −0.859 | 0/26 |
| group_pca_pooled_48 | PCA 48 | 0.236 | 25/26 | 0.000 | 0.468 | 26/26 | −0.904 | 0/26 |

## 6. Model-specific results

### 6.1 Held-out reconstruction of the feature bank: static vs dynamic

'raw' = fraction of variance reconstructed; 'dynamic' = the same after removing each (feature, depth) mean residual on the held-out insertion, i.e. ignoring a wrong static depth profile. The benchmark's ridge readouts fit intercepts, so only the dynamic part affects sections 4–5. Medians over held-out insertions.

| Model | waveform raw | RMS raw | PSD raw | waveform dyn | RMS dyn | PSD dyn |
|---|---|---|---|---|---|---|
| ae_anatomy_48 | 0.96 | 0.80 | 0.79 | 0.97 | 0.88 | 0.83 |
| ae_filterbank_48 | 0.99 | 0.06 | 0.65 | 0.99 | 0.71 | 0.79 |
| ae_masked_48 | 0.96 | 0.77 | 0.76 | 0.97 | 0.86 | 0.82 |
| ae_waveamp_48 | 1.00 | 0.70 | 0.68 | 1.00 | 0.83 | 0.75 |
| group_pca_pooled_48 | 0.00 | 0.60 | 0.64 | 0.00 | 0.88 | 0.72 |

A and B were trained by `train.py`, which does not save weights, so their decoder reconstruction is not in this table. The pooled group PCA row reconstructs from its 2 components per depth.

### 6.2 C: inferring hidden depths

Hidden groups on held-out insertions: four 3-group blocks (0.96 mm) and four random 25% masks per insertion. R² over the hidden groups only, per view:

| Mask | Method | waveform | RMS | PSD |
|---|---|---|---|---|
| block3 | depth_interp | 0.64 | −0.68 | 0.37 |
| block3 | depth_interp_shrunk | 0.64 | −0.22 | 0.32 |
| block3 | masked_ae | 0.47 | 0.08 | 0.01 |
| random25 | depth_interp | 0.78 | −0.84 | 0.77 |
| random25 | depth_interp_shrunk | 0.78 | −0.09 | 0.77 |
| random25 | masked_ae | 0.74 | 0.35 | 0.56 |

### 6.3 E: what the learned filters converge to

Initial centres were log-spaced 2–80 Hz. Median (range across folds) of the learned centre and width:

| Filter | centre Hz | width Hz |
|---|---|---|
| 0 | 1.5 (1.3–1.5) | 1.0 (1.0–1.1) |
| 1 | 3.7 (2.9–3.9) | 1.3 (1.0–1.4) |
| 2 | 5.7 (4.5–6.3) | 2.4 (1.9–2.6) |
| 3 | 9.8 (8.8–10.2) | 3.0 (2.9–3.0) |
| 4 | 21.4 (19.1–23.2) | 6.9 (6.0–7.2) |
| 5 | 34.4 (31.6–36.2) | 9.9 (9.1–10.3) |
| 6 | 48.8 (45.5–63.4) | 19.2 (16.1–20.2) |
| 7 | 104.7 (76.9–111.0) | 27.8 (25.2–30.4) |

Folds with a trained filterbank: 5 of 5.


### 6.4 What the latent channels track

Correlation, over depths × time on held-out insertions, of each depth-indexed latent channel with each bank feature, with signs aligned across insertions. The full profiles are in `tables/model_purpose/latent_feature_profiles.csv`; the maximum |r| per channel is:

| Model · channel | max \|r\| |
|---|---|
| ae_anatomy_48 · 0 | 0.23 |
| ae_anatomy_48 · 1 | 0.24 |
| ae_depth_48 · 0 | 0.52 |
| ae_depth_48 · 1 | 0.33 |
| ae_filterbank_48 · 0 | 0.53 |
| ae_filterbank_48 · 1 | 0.50 |
| ae_masked_48 · 0 | 0.38 |
| ae_masked_48 · 1 | 0.34 |
| ae_waveamp_48 · 0 | 1.00 |
| ae_waveamp_48 · 1 | 0.38 |
| group_pca_2x24 · 0 | 0.74 |
| group_pca_2x24 · 1 | 0.41 |
| group_pca_pooled_48 · 0 | 0.72 |
| group_pca_pooled_48 · 1 | 0.65 |
| group_pca_weighted_48 · 0 | 1.00 |
| group_pca_weighted_48 · 1 | 0.72 |
| group_wave+amp1_pooled · 0 | 1.00 |
| group_wave+amp1_pooled · 1 | 0.72 |

## 7. Verdict: right purpose, and does the network earn its complexity?

'Earns its complexity' = better than its linear twin by more than the tolerance on at least one purpose test and not worse by more than ±0.02 on any. Counts are over the relative tests of section 4 that have values.

| Model | Built for | Right purpose? | vs linear twin: better / tied / worse | Better on | Worse on |
|---|---|---|---|---|---|
| **A** `ae_depth_48` | R2, R3, R4, R7 | yes: spatial, pooled, behaviour-free | 0 / 1 / 6 (vs W) | – | behaviour r, waveform, depth pattern, probe-wide amplitude, gamma envelope 40 ms, cross-probe CCA |
| **B** `ae_global_48` | R2, R4, R7 | **no for a spatial representation**: no number is tied to a depth (fails R3 by construction); useful only as an upper bound on probe-wide information | 3 / 2 / 2 (vs W) | behaviour r, probe-wide amplitude, cross-probe CCA | waveform, gamma envelope 40 ms |
| **C** `ae_masked_48` | R3 + inferring unrecorded depths | yes: a small-scale version of the project's 'infer unrecorded regions' question | 0 / 4 / 5 (vs W) | – | waveform, depth pattern, probe-wide amplitude, gamma envelope 40 ms, cross-probe CCA |
| **D** `ae_anatomy_48` | R3, R4 with anatomical grounding | yes, but needs histology-aligned region labels at deployment | 0 / 3 / 6 (vs W) | – | behaviour r, waveform, depth pattern, probe-wide amplitude, gamma envelope 40 ms, cross-probe CCA |
| **E** `ae_filterbank_48` | R2 without hand-made features | yes for the long-term goal; the prototype reads only 2 of 16 channels per depth | 0 / 0 / 9 (vs W) | – | behaviour r, waveform, depth pattern, probe-wide amplitude, gamma envelope 40 ms, zero-shot transfer, cross-probe CCA, state NMI, pooled state NMI |
| **F** `ae_waveamp_48` | R1–R3: capacity allocated by physics | yes: waveform stored, network spends all capacity on amplitude | 0 / 2 / 7 (vs WF) | – | behaviour r, depth pattern, probe-wide amplitude, gamma envelope 40 ms, zero-shot transfer, cross-probe CCA, state NMI |

**Reading.** Models that earn their complexity by this rule: none. Models better than their linear twin on no purpose test: A, C, D, E, F.
 Behaviour, session level (Δ median, sessions higher): A −0.026 (1/26); B 0.016 (23/26); C −0.019 (5/26); D −0.024 (4/26); E −0.043 (4/26); F −0.030 (2/26).

Training caveat: for A, B, C, D, F the best validation epoch fell in the last 5 epochs in most folds, so they may still improve with longer training (cap 50 epochs; `tables/model_purpose/training_convergence.csv`). Longer training reduces reconstruction loss, but section 8 explains why that need not help the linear readouts.

## 8. Why the networks do not beat their linear twins here

1. **The decisive nonlinearity is already in the features.** Log band power is the log of a smoothed squared band-passed signal, exactly the nonlinearity section 2.1 says PCA lacks. After it, the remaining structure is close to linear and Gaussian. The linear twin W reaches behaviour r 0.58 at 48 numbers, against 0.62 for the full 456-number bank.
2. **The consumers are linear-Gaussian.** Ridge readouts, CCA and Gaussian HMMs, the tools of tasks 2–5, can only use information that is linearly available in the latent. A reconstruction loss with a nonlinear decoder does not require that. So a network can reconstruct well through its own decoder (section 6.1) while its latent serves a linear readout worse.
   Direct evidence: training A longer (15 → 50 epochs) lowered its validation loss, yet behaviour r went 0.58 → 0.55 (higher in 1/26 sessions) and depth pattern 0.35 → 0.26, while waveform R² rose 0.68 → 0.86. Better reconstruction, worse linear readouts: the objective and the purpose pull apart.
3. **Temporal context smooths fast dynamics.** Every convolutional model loses gamma-envelope (40 ms) R² against its twin; the median session-level Δ ranges −0.30 to −0.11. The encoder integrates ±240 ms, and the 40 ms gamma feature carries 1/27 of the loss, so the bottleneck is spent on slow structure. W has no temporal kernel and keeps the fast part.
4. **Data scale.** 50 insertions × 100 s is about 125,000 time bins, strongly autocorrelated. Nonlinear gains usually need more data; the fair test of the AE family is the 750-insertion atlas, not this set.
5. **Objective ≠ purpose.** Mean squared error rewards variance, and the dominant variance, broadband power at each depth, is captured linearly (W's second channel; section 6.4).
6. **Coordinates are shared within one trained encoder, not across retrainings.** Zero-shot transfer with one encoder per fold: C 0.44, D 0.42 vs W 0.44. With each insertion encoded by its own fold's model, the same models fall to C −0.04, D 0.05, while the linear fits do not move (W 0.44, pooled group PCA 0.51 vs 0.51). PCA's axes are determined by the data; an autoencoder's are one of many equivalent solutions, and each retraining picks a different one (section 2.3). For an atlas, an AE must be trained once, frozen and versioned.

**A result that ties this together.** W (the linear minimiser of the autoencoders' loss) and WF (stored waveform + 1 amplitude PC per depth) score identically on every benchmark metric. With views weighted equally, the waveform is nearly uncorrelated with the amplitude features, so the weighted PCA splits exactly into 'waveform' and 'top amplitude component'. The hand-designed per-depth representation is therefore *the linear optimum of the autoencoder's own objective*, and the networks are being asked to beat the optimum of their own loss restricted to linear maps.

## 9. Recommendation

- **Use the linear twin as the task-1 representation now.** W or WF: each depth stores its waveform plus one or more broadband-amplitude components, with loadings fit once on training sessions and shared by all insertions. It is linear, interpretable (waveform + power per depth), has shared coordinates (R4), and costs one matrix multiply after the feature bank. Behaviour r W 0.58 / WF 0.58, waveform R² 0.96 / 0.96, depth pattern 0.35 / 0.35, against PCA 48's 0.34, 0.99, −0.01.
- **Keep the autoencoder as the scalable route, but change what it is asked to do** before training it again:
  1. a fast path: a skip connection for the 40 ms features, or no temporal kernel before the bottleneck;
  2. an objective that suits linear-Gaussian consumers: a decorrelation/whitening penalty on the latent, or a behaviour-free temporal objective (slow-feature or contrastive in time, e.g. the time-contrastive mode of CEBRA, Schneider et al. 2023), still without behaviour;
  3. pre-training on all 750 atlas insertions, then this benchmark as the held-out test; save the weights (train.py does not, which is why A and B could not be re-encoded) and freeze one versioned encoder for the atlas;
  4. for E: all 16 channels per depth (power averaged, as in the bank) and a per-insertion spectral normalisation. Check its highest learned band before trusting it (it moved above 100 Hz, where spike leakage and EMG live);
  5. C's masking, if the goal is imputation: on held-out insertions it beats shrunk depth interpolation on the RMS view but loses on the waveform and PSD views, which are spatially smooth enough for interpolation (section 6.2).
- **Keep neural validity as a standing test.** The CSD comparison (other session) and the out-of-brain controls should be run on every candidate, because behaviour decoding alone cannot tell neural signal from artefact.

## 10. Limitations

- 50 insertions, 26 sessions, one 100 s window per session. Folds are sessions for model training and contiguous time blocks for readouts. This is not evidence of reproducibility across the 750-insertion atlas.
- All bank-derived representations and autoencoders z-score each insertion over its full 100 s: label-free but transductive, since test blocks inform the normalisation.
- Behaviour r is a proxy for state-relevant information. Wilcoxon p-values are uncorrected and descriptive; many comparisons were made.
- The HMM reference is built from bank features, which gives bank-derived representations a home advantage over waveform-only ones.
- *Timescale (T13): confounded by our own smoothing windows (RMS 200 ms/1 s, PSD 1 s), so it describes feature construction as much as brain dynamics. No criterion is attached.
- Cross-probe CCA can include common non-neural signals (motion artefacts). Each probe is common-median referenced separately. The 1 Hz high-pass variant separates slow shared drift, but it is not a CSD test.
- One seed per fold, no hyperparameter tuning (deliberately: nothing was tuned on test data). A–D and F ran to the epoch cap.
- E reads 2 of 16 channels per depth, so its static depth profile of power is off (section 6.1). D depends on the histology alignment of region labels.
- Imputation hides depths within an insertion; that is only a proxy for inferring unrecorded regions.
- Outside-the-brain checks (other session): in the Cosmos mapping 'root' collects fiber tracts and ventricles, i.e. white matter INSIDE the brain, and only 'void' is outside. An earlier control counted both as outside; the numbers quoted here use the corrected split (grey / white / void).
- 3-state HMMs are unstable: refits on the same features agree at NMI ≈ 0.5 (section 4), so small NMI differences between representations are within that noise.

## Figures

In `docs/figures/model_purpose/`; every table behind them is in `docs/tables/model_purpose/`:

- `fig_models_benchmark.png`: every model on the benchmark protocol: behaviour r, waveform, depth pattern, 40 ms gamma; one dot per held-out insertion
- `fig_paired_vs_linear_twin.png`: session-level differences from W: the test of whether each network earns its complexity
- `fig_paired_vs_pca48.png`: session-level differences from PCA 48: the test of the project question
- `fig_purpose_tests.png`: zero-shot transfer, cross-probe CCA (with 1 Hz high-pass), HMM state agreement within and across animals
- `fig_filterbank_learned.png`: E's learned band-pass filters in 5 independent folds against the canonical bands
- `fig_imputation.png`: C vs linear and shrunk interpolation on hidden depth groups

## 11. References

- Akella S. et al. (2025) Deciphering neuronal variability across states. *Nature Communications*.
- Baldi P., Hornik K. (1989) Neural networks and principal component analysis: learning from examples without local minima. *Neural Networks* 2:53–58.
- Buzsáki G., Anastassiou C.A., Koch C. (2012) The origin of extracellular fields and currents — EEG, ECoG, LFP and spikes. *Nat Rev Neurosci* 13:407–420.
- Donoghue T. et al. (2020) Parameterizing neural power spectra into periodic and aperiodic components. *Nat Neurosci* 23:1655–1665.
- Einevoll G.T., Kayser C., Logothetis N.K., Panzeri S. (2013) Modelling and analysis of local field potentials for studying the function of cortical circuits. *Nat Rev Neurosci* 14:770–785.
- Gao R., Peterson E.J., Voytek B. (2017) Inferring synaptic excitation/inhibition balance from field potentials. *NeuroImage* 158:70–78.
- He K. et al. (2022) Masked autoencoders are scalable vision learners. *CVPR*.
- International Brain Laboratory (2023/2025) A brain-wide map of neural activity during complex behaviour.
- Kajikawa Y., Schroeder C.E. (2011) How local is the local field potential? *Neuron* 72:847–858.
- Locatello F. et al. (2019) Challenging common assumptions in the unsupervised learning of disentangled representations. *ICML*.
- McGinley M.J. et al. (2015) Waking state: rapid variations modulate neural and behavioral responses. *Neuron* 87:1143–1161.
- Mitzdorf U. (1985) Current source-density method and application in cat cerebral cortex. *Physiol Rev* 65:37–100.
- Musall S. et al. (2019) Single-trial neural dynamics are dominated by richly varied movements. *Nat Neurosci* 22:1677–1686.
- Niell C.M., Stryker M.P. (2010) Modulation of visual responses by behavioral state in mouse visual cortex. *Neuron* 65:472–479.
- Pesaran B. et al. (2018) Investigating large-scale brain dynamics using field potential recordings: analysis and interpretation. *Nat Neurosci* 21:903–919.
- Ravanelli M., Bengio Y. (2018) Speaker recognition from raw waveform with SincNet. *IEEE SLT*.
- Schneider S., Lee J.H., Mathis M.W. (2023) Learnable latent embeddings for joint behavioural and neural analysis. *Nature* 617:360–368.
- Stringer C. et al. (2019) Spontaneous behaviors drive multidimensional, brainwide activity. *Science* 364:eaav7893.
- Vinck M. et al. (2015) Arousal and locomotion make distinct contributions to cortical activity patterns and visual encoding. *Neuron* 86:740–754.
- Zeghidour N. et al. (2021) LEAF: a learnable frontend for audio classification. *ICLR*.

## 12. Reproduction

```bash
cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1; PY=/opt/miniconda3/envs/lfp-brain-state/bin/python
$PY -m lfp_autoencoder_prototype.cache_raw                                   # raw channel subset for E
for f in 0 1 2 3 4; do for m in C D E F; do $PY -m lfp_autoencoder_prototype.train_variants --only-fold $f --models $m; done; done
$PY -m lfp_autoencoder_prototype.purpose_tests --baselines --weighted       # PCA family, W, WF
$PY -m lfp_autoencoder_prototype.evaluate_variants --worker --tags W WF C D E F
$PY -m lfp_autoencoder_prototype.purpose_tests --tests                      # transfer, CCA, timescale, interpret
/opt/miniconda3/envs/lfp/bin/python lfp_autoencoder_prototype/state_tests.py  # HMM states (needs hmmlearn)
$PY -m lfp_autoencoder_prototype.recon_offsets && $PY -m lfp_autoencoder_prototype.imputation_linear
$PY -m lfp_autoencoder_prototype.model_report && $PY -m lfp_autoencoder_prototype.write_standards
```
A and B: `train.py` / `evaluate_ae.py` (unchanged protocol, 50-epoch run). Nothing is committed or pushed.
