# Skylar branch: a fast, meaningful LFP representation (task 1)

Status as of 20 Sep 2026. Everything here is uncommitted work in the `Skylar` branch's working tree.

## What the representation is for

It is the measurement layer for the Latent Brain State project: the input to the brain-state models of
tasks 2–5 (shared vs private dynamics across simultaneous probes, discrete vs continuous states and
their timescales, mechanisms of global control, a brain-wide dynamic atlas). "Compression is for
dimension and meaning": few numbers, each interpretable. So it must keep

1. **band-limited power at each location** — the state variables of the LFP-HMM literature
   (e.g. Akella et al. 2025: θ/β/γ Hilbert envelopes by depth → Gaussian HMM);
2. **their slow dynamics** (seconds for arousal, ~100 ms for movement-related gamma);
3. **where on the probe they came from** (global vs local, the atlas);
4. **the signed low-frequency waveform** (current source density, phase, cross-probe coupling);
5. and it must **transfer to sessions it has never seen**.

Waveform reconstruction is a guardrail, not the goal. Behaviour decoding is a validation, not the
goal, and it must be checked against non-neural explanations.

## The work, in order

| Stage | Package | Question | Answer |
|---|---|---|---|
| Pilot (13–14 Sep) | `lfp_compression_pilot/` | One insertion, 100 s: raw vs lfpack vs PCA 16 vs amplitude-augmented PCA | PCA keeps zero beta/gamma amplitude; 4 amplitude numbers restore it at ~0 waveform cost. lfpack is a storage codec (needs 250 Hz + Cadzow), not a representation |
| Search (14 Sep) | `lfp_compression_search/` | 9 candidates on 12 probes with a pre-declared rule, confirmed on 6 new mice | 12 waveform PCs + 4 amplitude PCs (200 ms RMS) wins; +0.162 behaviour r in 6/6 new sessions |
| Alon's 50 insertions (14–15 Sep) | `lfp_selected_benchmark/` | Full benchmark: 17 methods, 5 behaviours, spatial metrics, matched budgets, regions, locality | Amplitude methods 0.48–0.62 vs PCA 0.30–0.34; PCA recovers no amplitude depth pattern (−0.01); per-depth methods 0.33–0.46 |
| Budget sweeps (19 Sep) | `lfp_selected_benchmark/tradeoff.py` | How should 16 numbers be split? Why doesn't more budget help? | 2 amplitude numbers give +0.26; waveform PCs plateau at 0.35 (32 = 64); amplitude saturates by ~8 |
| Verification (20 Sep) | `lfp_selected_benchmark/controls.py` | Are the decoding values real? | Independent readout 0.591 vs 0.592; causal 0.59; zero lag 0.56; 5 s gaps 0.59; nulls −0.02..+0.01; aligned (median lag 40 ms). Setting closest to Alon's: 0.47 vs his 0.46 |
| Neural validity (20 Sep) | `lfp_selected_benchmark/neural_validity.py` | Is the behavioural signal local and neural? | Grey-matter CSD ≥ potential (0.57 vs 0.55; 39/50; every band). Outside the brain much lower (0.23 vs 0.39; 4 insertions) |
| Aliasing (20 Sep) | `lfp_selected_benchmark/aliasing.py` | Is the 25 Hz waveform clean? | 14.9% of its 8–12.5 Hz power is aliased from >13 Hz (boxcar bin mean) |
| Autoencoders (15–21 Sep) | `lfp_autoencoder_prototype/` | Five CNN designs vs PCA and linear per-depth twins, trained leave-session-out | see "Five models" below |

## Why each design choice (the theory)

| Choice | Reason |
|---|---|
| 25 Hz, 40 ms bins | The current representation's rate; behaviour timescales ~50 ms. Anything faster than 12.5 Hz must be kept as amplitude, not waveform |
| Amplitude (band RMS) features | Brain states are slow modulations of fast rhythms; the envelope of a fast rhythm is slow and can be sampled at 25 Hz |
| log amplitude | Multiplicative gain changes become additive; power becomes roughly Gaussian (ridge readout, Gaussian HMM) |
| RMS windows ≥ one cycle of the band | Time–frequency uncertainty: a window shorter than one cycle cannot estimate the band's power |
| PSD shape + 1/f slope | Spectral shape separates oscillatory states; the aperiodic exponent tracks excitation/inhibition balance and arousal |
| 24 groups × 16 channels (320 µm) | Neighbouring channels are highly redundant (volume conduction); averaging raises SNR; regional state, not laminar detail |
| Group PCA with loadings shared across depths | Prior: features covary the same way at every depth. Supported: loadings fit on other sessions work as well as per-insertion ones (0.579 vs 0.580) |
| CNN: shared per-depth encoder, ±240 ms time, ±320 µm depth | Weight sharing = the same prior, nonlinear; short kernels because the data are small (50 × 100 s) |
| Contiguous test blocks + 1 s gaps; circular-shift null | Autocorrelated time series leak across random splits |
| Current source density | Second spatial derivative of the potential; removes far-field signal (volume conduction, muscle) |

## Known flaws and open issues

- **Aliasing in the 25 Hz waveform** (both the current method and ours): the 40 ms bin mean is the only step without a
  proper anti-aliasing filter (the 2500 → 500 → 250 Hz decimations are zero-phase FIR); fix with a low-pass before sampling.
- **Referencing**: `destripe_lfp` is called with every channel labelled good, so there is no bad-channel interpolation, and
  the spatial step is a common median reference over all 384 channels. The probe-wide shared component is partly removed
  before any analysis; the CSD results are reference-free and unaffected.
- **1/f slope** is a plain log–log fit over 2–40 Hz, so oscillatory peaks bias it; a periodic/aperiodic
  separation (specparam) is the principled version.
- **Autoencoder normalisation** uses each held-out insertion's full 100 s (label-free, but transductive).
- **Tissue labels**: Cosmos 'root' = fiber tracts and ventricles (inside the brain), only 'void' is outside;
  white-matter comparisons are impure because 'root' also includes parent-level grey-matter labels.
- **Regions table** compares different insertions, not regions within an animal.
- **100 s per insertion**, one window; behaviour from video and wheel only (paw states not public).
- The CSD is coarse (320 µm); per-channel CSD is the next step for laminar questions.

## Six autoencoders vs their linear twins (21 Sep)

All 48 numbers per 40 ms, trained leave-session-out (5 session folds), scored with the benchmark protocol on
the 50 insertions (`lfp_autoencoder_prototype/compare_models.py`; options C–F by the fork session).

| Model | Behaviour r | Waveform R² | Depth pattern R² | vs W, sessions higher |
|---|---|---|---|---|
| W: group PCA with the AE's view weights (linear twin) | 0.583 | 0.962 | 0.345 | — |
| WF: waveform + 1 amplitude PC per depth, other sessions | 0.583 | 0.962 | 0.345 | — |
| A: CNN, 2 latents per depth | 0.551 | 0.858 | 0.258 | 1/26 |
| B: CNN, 48 global latents | 0.606 | 0.699 | 0.363 | 23/26 |
| C: A + masked depths | 0.568 | 0.816 | 0.295 | 5/26 |
| D: A + region embedding | 0.556 | 0.878 | 0.293 | 4/26 |
| E: learned filterbank on raw signal | 0.546 | 0.843 | 0.154 | 4/26 |
| F: waveform stored + learned amplitude | 0.542 | 0.963 | 0.132 | 2/26 |

- W learns [waveform, broadband amplitude] per depth, i.e. the hand-designed hybrid (WF), from the AE's
  own objective.
- Only B beats the twins on behaviour, by a small margin, at the cost of waveform fidelity and of latents
  that no longer belong to depths.
- The CNNs reconstruct the feature bank better than linear group PCA, but the advantage is the static depth
  profile; on the time-varying part linear group PCA is as good (RMS 0.88 vs 0.83–0.88). Training A from 15
  to 50 epochs raised waveform R² (0.68 → 0.86) and lowered behaviour (0.58 → 0.55): the reconstruction
  objective is not aligned with the goal.
- E did not train (early stop at 13–23 epochs; its front end reconstructs RMS at 0.06).

**Tests of what the representation is for** (fork session, `purpose_tests.py`, `state_tests.py`):
- Task 2: held-out CCA between the 24 simultaneous probe pairs is highest for waveform PCA 48 (0.83; 0.78
  after a 1 Hz high-pass); the hybrid keeps 0.72. The waveform carries what probes share.
- Task 3: a 3-state HMM on the hybrid agrees with an Akella-style reference at NMI 0.30, waveform PCA at
  0.16; the reference refit on itself reaches only 0.48, so 100 s HMMs are unstable.
- Task 5: a behaviour readout trained on other sessions and applied unchanged (one encoder per fold) reaches
  0.51 for pooled group PCA and 0.44 for the hybrid; per-insertion fits reach only 0.27–0.30. One trained CNN
  transfers as well (C 0.44), but mixing separately trained CNNs drops it to −0.04: an atlas would need one
  frozen, versioned encoder. States fit across animals agree poorly with every representation (NMI ≤ 0.14).

Full theory, standards and results matrix: [`docs/STANDARDS_AND_THEORY.md`](STANDARDS_AND_THEORY.md).

**Recommendation.** Adopt waveform + one amplitude component per depth, loadings shared across sessions, as
the task-1 representation (with an anti-aliased waveform). Revisit CNNs only with a goal-aligned objective
and the full 750-insertion dataset.

## Reproduction

Environment: conda env `lfp-brain-state` (Python 3.11, ibllib, ONE, iblatlas, torch, scikit-learn).
Each package's README has its exact commands; run them from the repository root. Data and results live
outside the repository, in `~/Downloads/lfp-brain-state/` (`selected_pids_cache/`, `ae_cache/`,
`selected_benchmark_results/`, `ae_results/`). Alon's 50 insertions: `data/selected_pids.csv` in this repository (the code falls back to `~/Downloads/selected_pids.csv`).

## Figures

`docs/figures/` holds the figures for the 21 Sep meeting (`docs/figures/model_purpose/`: the fork's model figures); the script with captions and talking points is
in `~/Downloads/lfp-brain-state/alon_meeting_2026-09-21/SCRIPT.md`.
