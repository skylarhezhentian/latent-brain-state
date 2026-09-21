# Autoencoder prototype

Tests the proposed compact spatiotemporal autoencoder against its linear counterpart and
against a representation with no PCA, all at 48 numbers per 40 ms, on Alon's 50 insertions.

| File | Role |
|---|---|
| `cache_bank.py` | caches the 24-depth × 19-feature bank per insertion (waveform, 9 RMS, 9 PSD) |
| `models.py` | `DepthLatentAE` (2 latents per depth) and `GlobalLatentAE` (48 shared latents); shared convolutional front end |
| `train.py` | leave-session-out training (5 session folds), early stopping on 2 held-out training sessions, pooled group PCA baseline; `--mask p` hides whole depth groups (masked-AE variant) |
| `evaluate_ae.py` | scores latents with the benchmark protocol; adds `waveform+gamma_48` (no PCA) |
| `figures_ae.py` | comparison, training curves, latent maps for the median insertion |

Priors built in, because the data are small (50 × 2,500 bins): the same computation at every
depth (1×1 and depth convolutions share weights), short temporal kernels (±240 ms), neighbouring-depth
mixing only (±320 µm), a loss that weights waveform, RMS and PSD views equally, and behaviour kept out
of training entirely.

```bash
cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
PY=/opt/miniconda3/envs/lfp-brain-state/bin/python
for k in 0 1 2; do $PY -m lfp_autoencoder_prototype.cache_bank --shard $k/3 & done; wait
for f in 0 1 2 3 4; do $PY -m lfp_autoencoder_prototype.train --only-fold $f & done; wait     # ~15 min per fold, CPU
for k in 0 1 2 3; do $PY -m lfp_autoencoder_prototype.evaluate_ae --worker & done; wait
$PY -m lfp_autoencoder_prototype.evaluate_ae --aggregate
$PY -m lfp_autoencoder_prototype.figures_ae
```

Requires the outputs of `lfp_selected_benchmark` (selected_pids_cache and selected_benchmark_results).
Results: `~/Downloads/lfp-brain-state/ae_results/`. Nothing is committed or pushed.

**20–21 Sep: 50-epoch retrain and the six-model comparison.** The 15-epoch run is archived in
`ae_results/run_e15/`. Retrain with `train.py --epochs 50 --patience 6` (one process per fold), then clear
`ae_results/probes/` (use `find ... -delete`; a zsh glob that matches nothing aborts `rm`) and re-run
`evaluate_ae`. `compare_models.py` puts A–F, the linear twins W/WF and the benchmark baselines on the same
50 insertions:

```bash
$PY -m lfp_autoencoder_prototype.compare_models --evaluate C D E F     # scores the fork's latents (one per core)
$PY -m lfp_autoencoder_prototype.compare_models                        # tables + fig_models, fig_models_paired, fig_purpose
```

Writes `ae_results/compare/`: `summary_models.csv`, `paired_models.csv` (per insertion),
`paired_sessions_models.csv` (per session, as in the figure), `probes_{C,D,E,F}/`.

## Design options C–F, linear twins W/WF, and purpose tests

Each option changes one thing in A (same trunk, 48 numbers per 40 ms, same folds, windows, optimiser,
50-epoch cap, patience 6). The write-up with theory, the standards catalogue and the results matrix is
`docs/STANDARDS_AND_THEORY.md`.

| File | Role |
|---|---|
| `variants.py` | C `MaskedDepthAE` (hide 25 % of depth groups, reconstruct all), D `AnatomyAE` (Cosmos-region embedding; `root` = white matter and `void` = outside get separate tokens), E `FilterbankAE` (8 learned Gaussian band-pass filters on 2 raw 250 Hz channels per depth; no hand-made features in the input), F `WaveAmpAE` (waveform stored, 1 learned amplitude latent per depth) |
| `cache_raw.py` | raw channel subset (4 per depth, 1 s reflect padding) for E |
| `train_variants.py` | trains C–F with train.py's protocol; saves weights, held-out reconstruction, C's imputation test, cost |
| `evaluate_variants.py` | benchmark protocol (unchanged `evaluate_probe`) for C, D, E, F, W, WF |
| `purpose_tests.py` | PCA-family latents (`--baselines`), linear twins W and WF (`--weighted`), and tests beyond the benchmark (`--tests`): zero-shot transfer across sessions, CCA between simultaneous probes (+1 Hz high-pass), timescale, latent↔feature correlations |
| `state_tests.py` | 3-state Gaussian HMM agreement with an Akella-style reference, per insertion and pooled across animals (runs in the `lfp` env, which has hmmlearn) |
| `recon_offsets.py` | held-out reconstruction split into static (depth profile) and dynamic parts |
| `imputation_linear.py` | shrunk depth interpolation, a fair linear baseline for C's imputation |
| `model_report.py`, `write_standards.py`, `write_standards_tail.py` | tables, figures and `docs/STANDARDS_AND_THEORY.md`; every number is read from result files |

W = pooled group PCA on features scaled by √(view weight): the linear, context-free minimiser of the
autoencoders' own loss (Baldi & Hornik 1989), so "A vs W" isolates what the network adds. WF = stored
waveform + 1 pooled amplitude PC per depth: F's linear twin. Reproduction order: section 12 of the write-up.
Outputs: `ae_results/variants/`, `ae_results/purpose/`, `~/Downloads/lfp-brain-state/model_purpose_2026-09-21/`.
