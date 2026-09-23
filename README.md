# LFP-Based Brain State Analysis

This repository contains code for analyzing brain states from large-scale LFP recordings.

The **`Skylar` branch** adds task 1 of the Latent Brain State project: choosing a fast, compact LFP
representation for the brain-state models of tasks 2–5, and testing it. Everything below the
installation section describes the branch additions; the original installation instructions are
unchanged.

---

## Installation

### 1. Install IBL Ephys Atlas Tools

First, clone and install the IBL Ephys Atlas tools repository:

```bash
git clone https://github.com/int-brain-lab/ibleatools.git
cd ibleatools
pip install -e .
```

> **Note:** Depending on your IDE (e.g., PyCharm or VS Code), you may need to mark the `src` directory as a source root.

---

### 2. Install Project Dependencies

Return to the root directory of this project:

```bash
cd ../
```

Then install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Verifying the Installation

After installation, you should be able to import the main dependencies without errors:

```python
import numpy as np
import pandas as pd
import torch
from iblatlas.atlas import AllenAtlas
```

If these imports succeed, the installation was completed successfully.

### Notes for the `Skylar` branch

- `requirements.txt` pins a CUDA build of torch; on a Mac, install the CPU build instead.
- The pilot's lfpack comparison also needs `lfpack` (IBL's LFP codec). Nothing else is required beyond
  `requirements.txt`.
- Run every command from the repository root; the packages import each other by name.

---

## What the branch adds

| Folder | Role |
|---|---|
| `data/selected_pids.csv` | The 50 selected insertions (26 sessions, 9 labs) and their 100 s analysis windows |
| `fast_lfp_benchmark/` | First benchmark: raw vs channel binning vs spatial PCA vs lfpack, rate–distortion |
| `lfp_compression_pilot/` | Controlled pilot on one insertion, then 11 more: PCA vs amplitude-augmented PCA vs lfpack |
| `lfp_compression_search/` | Search over 9 candidate representations with a pre-declared rule, confirmed on 6 new mice |
| `lfp_selected_benchmark/` | Main benchmark on the 50 insertions, budget sweeps, verification controls, neural-validity (CSD) test, aliasing check |
| `lfp_autoencoder_prototype/` | Six convolutional autoencoders (A–F), their linear counterparts, and tests tied to tasks 2, 3 and 5 |
| `docs/PROGRESS.md` | Research log: what was done, why, results, open issues |
| `docs/STANDARDS_AND_THEORY.md` | Standards, theory and full results matrix for PCA and the six autoencoders |
| `docs/figures/` | Figures for the 21 Sep 2026 meeting (`docs/figures/model_purpose/`: autoencoder figures) |

Each package has its own README with exact commands.

## The representation, in one paragraph

Every 40 ms (25 Hz), each of 24 depth groups (16 adjacent channels, 320 µm) is summarised by 19 features:
the group-mean signed waveform, log RMS amplitude of 5 frequency bands at up to 3 time scales, and the
shape of the power spectrum (8 log-spaced bands plus the 1/f slope). The recommended compact form keeps,
per depth, the waveform plus one amplitude component whose loadings are fit once and shared by all
insertions: 48 numbers per 40 ms.

## The six autoencoders

All six keep 48 numbers per 40 ms, share one encoder design (convolutions over time, ±240 ms, and over
neighbouring depths, ±320 µm), reconstruct the 19-feature bank at 24 depths, and are trained with whole
sessions held out (5 folds by session).

| | Model | Code | Trained by | What it changes |
|---|---|---|---|---|
| A | `DepthLatentAE` | [`models.py`](lfp_autoencoder_prototype/models.py#L44) | `train.py` | 2 latents per depth |
| B | `GlobalLatentAE` | [`models.py`](lfp_autoencoder_prototype/models.py#L63) | `train.py` | 48 latents shared by all depths |
| C | `MaskedDepthAE` | [`variants.py`](lfp_autoencoder_prototype/variants.py#L68) | `train_variants.py` | whole depth groups hidden during training |
| D | `AnatomyAE` | [`variants.py`](lfp_autoencoder_prototype/variants.py#L91) | `train_variants.py` | learned embedding of each depth's brain region |
| E | `FilterbankAE` | [`variants.py`](lfp_autoencoder_prototype/variants.py#L155) | `train_variants.py` | learned band-pass filters on raw signal instead of hand-made features |
| F | `WaveAmpAE` | [`variants.py`](lfp_autoencoder_prototype/variants.py#L110) | `train_variants.py` | waveform stored as is; network learns only the amplitude part |

Shared building blocks (`conv`, `trunk`, `untrunk`) are in [`models.py`](lfp_autoencoder_prototype/models.py#L21).
Linear counterparts: **W** = group PCA with the autoencoders' feature weights; **WF** = waveform + 1 amplitude
PC per depth, fit on other sessions (`purpose_tests.py --weighted`). `evaluate_ae.py` and
`evaluate_variants.py` score the models with the benchmark protocol; `compare_models.py` puts all of them
side by side.

## Main results (50 insertions)

| Representation (48 numbers per 40 ms unless noted) | Behaviour r | Waveform R² | Amplitude depth pattern R² |
|---|---|---|---|
| Waveform PCA 48 (current method) | 0.34 | 0.99 | −0.01 |
| Waveform + 1 amplitude PC per depth, other sessions (WF) | 0.58 | 0.96 | 0.35 |
| Autoencoder B (global latent) | 0.61 | 0.70 | 0.36 |
| Autoencoders A, C, D, E, F | 0.54–0.57 | 0.82–0.96 | 0.13–0.30 |
| Full 456-feature bank | 0.62 | 0.93 | 1 (by construction) |

- Behaviour r = held-out Pearson correlation between decoded and measured behaviour (wheel speed, whisker
  motion ×2, body motion, pupil), ridge readout, 4 contiguous 25 s test blocks per insertion.
- The decoding values survive an independent ridge implementation, past-only context, 5 s gaps between
  folds, and shuffled controls (−0.02 to +0.01); the current probe-wide RMS envelope reproduces the earlier
  0.46 (0.47 here).
- In grey matter, current source density band power decodes behaviour as well as or better than the
  potential (higher in 39 of 50 insertions): the behavioural signal is local.
- Recommendation: waveform + one amplitude component per depth with shared loadings. Details, caveats and
  theory: `docs/PROGRESS.md` and `docs/STANDARDS_AND_THEORY.md`.

## Where data and results go

Caches and results are written outside the repository. To move them, change these constants:

| Constant | File | Default |
|---|---|---|
| `CACHE_DIR` | `lfp_selected_benchmark/extract_selected.py` | `~/Downloads/lfp-brain-state/selected_pids_cache` |
| `RESULTS` | `lfp_selected_benchmark/run_selected.py` | `~/Downloads/lfp-brain-state/selected_benchmark_results` |
| `AE_CACHE`, `AE_CACHE_RAW` | `lfp_autoencoder_prototype/cache_bank.py`, `cache_raw.py` | `~/Downloads/lfp-brain-state/ae_cache`, `ae_cache_raw` |
| `AE_RESULTS` | `lfp_autoencoder_prototype/train.py` | `~/Downloads/lfp-brain-state/ae_results` |
| `CACHE_DIR`, `RESULTS_DIR` | `lfp_compression_pilot/pilot_config.py` | `~/Downloads/lfp-brain-state/...` |
| `CACHE_DIR`, `OUT_DIR` | `fast_lfp_benchmark/run.py` | `~/Downloads/lfp-brain-state/...` |

Raw LFP is streamed from the IBL public server (openalyx) by the extract scripts.

## Order to reproduce

1. `lfp_selected_benchmark/extract_selected.py` — stream and cache the 50 insertions.
2. `lfp_selected_benchmark/run_selected.py` (+ `--extra` passes) — main benchmark; then `figures.py`.
3. `lfp_selected_benchmark/tradeoff.py`, `controls.py`, `neural_validity.py`, `aliasing.py` — sweeps and checks.
4. `lfp_autoencoder_prototype/cache_bank.py`, `cache_raw.py` — feature caches for the autoencoders.
5. `train.py` (A, B) and `train_variants.py` (C–F), one process per fold; `evaluate_ae.py`, `evaluate_variants.py`.
6. `purpose_tests.py`, `state_tests.py` — tests for tasks 2, 3, 5; `compare_models.py` — tables and figures.

Exact commands are in each package's README.

## Known issues

- The 25 Hz waveform is a 40 ms bin mean, which aliases: about 15% of its 8–12.5 Hz power comes from above
  13 Hz. Low-pass before sampling to fix.
- The 1/f slope is a straight-line fit that ignores oscillatory peaks.
- `destripe_lfp` is called with every channel labelled good: no bad-channel interpolation, and a common
  median reference over each probe.
- Autoencoder inputs are normalised over each insertion's full 100 s (label-free, but transductive).
- A and B were trained without saving weights, so they cannot be re-encoded for the cross-session tests.
- 100 s per insertion, one window; states estimated with 3-state HMMs are unstable at this length.
