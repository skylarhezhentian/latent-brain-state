# Fast LFP representation benchmark

> **Status (14 Sep 2026): exploratory v0, partly superseded.** `extract.py` is still the
> data extractor used by `lfp_compression_pilot` (it provides the destriped `x_uV` arrays).
> Its lfpack arrays are **not valid**: they compress the 2500 Hz signal, while IBL runs
> lfpack on 250 Hz data with 2048-sample chunks. Use `lfp_compression_pilot/pilot_lfpack.py`
> for lfpack, and `lfp_compression_pilot` for all reported comparisons.


Scores candidate LFP representations on the five axes of the task, on
**simultaneous two-probe sessions**, so the same cache serves the later
shared-vs-private and dimensionality work.

| Axis | Metric | Question it answers |
|---|---|---|
| Reconstruction | `r2`, `rmse_uV` | How much of the signal survives? |
| Frequency content | `coh_<band>`, `pow_db_<band>` | Which bands survive, with the right phase and power? |
| Temporal information | `env_corr_<band>`, `tau_ratio_<band>` | Are fast amplitude dynamics and their timescales preserved? |
| Behavioural information | `beh_r2_<behaviour>`, `beh_retained` | Does behaviour decode from it as well as from raw LFP? |
| Reproducibility | `pca_reproducibility.csv`, `ranking_reproducibility.csv` | Is the basis, and the ordering of candidates, the same in independent sessions? |

All are plotted against **rate** (stored floats per second). Broadband R² alone
is misleading for LFP: power falls as 1/f, so it mostly measures delta.

## Run

```bash
cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 /opt/miniconda3/envs/lfp-brain-state/bin/python -m fast_lfp_benchmark.extract --sessions 6 --duration 120
KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 /opt/miniconda3/envs/lfp-brain-state/bin/python -m fast_lfp_benchmark.run
```

Extraction streams from the public IBL server (sessions that are not public are
skipped), about 10 minutes per session. Scoring reads the cache in
`~/Downloads/lfp-brain-state/fast_lfp_cache` and writes CSVs and
`rate_distortion.png` to `~/Downloads/lfp-brain-state/fast_lfp_results`.

## Candidates

| Name | What it is | Shared latent? |
|---|---|---|
| `raw` | the destriped reference; scores perfectly, a sanity check | — |
| `bin16` | average of 16 adjacent channels, the repo's current compression | yes |
| `pcaR` | rank-R spatial PCA, fit leave-one-session-out | yes |
| `lfpack_default`, `lfpack_aggressive` | IBL's codec (adaptive SVD + wavelet packets) at its two archive levels | no — basis changes every 0.8 s chunk |

"Shared latent" separates compression for **storage** from compression for
**meaning**: only a shared basis gives coordinates that mean the same thing
across time, probes and animals.

## Adding a candidate (e.g. the convolutional autoencoder)

Subclass `Representation` in `representations.py`:

```python
class ConvAE(Representation):
    shared_latent = True

    def __init__(self, checkpoint):
        self.model = load_model(checkpoint)                  # your code
        self.name = "conv_ae"
        self.n_params = sum(p.numel() for p in self.model.parameters())

    def reconstruct(self, probe):                            # probe["x_uV"]: [384, T] µV at 500 Hz
        return decode(encode(probe["x_uV"]))                 # same shape, µV

    def floats_per_second(self, probe):
        return latent_channels * latent_rate_hz
```

then add it to `candidates` in `run.py`. If it needs fitting, fit it without the
held-out session, as `SpatialPCA` does.

## Known limitations of v0

- The input is decimated to 500 Hz; lfpack's rate is measured on the native
  2500 Hz signal it compresses, which is conservative against lfpack.
- lfpack compresses the destriped signal, not the Cadzow-denoised signal IBL
  archives, so its published error figures do not transfer.
- Spatial PCA indexes channels by depth, not anatomy: across probes the basis
  mixes different regions.
- Motion energy is missing for some public sessions; behaviour columns are
  skipped per session when absent.
- 6 sessions give 10 distinct session half-splits: enough to see whether an
  effect replicates, not to estimate it precisely.
