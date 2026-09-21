# Pilot protocol and decisions log

Written before the results it governs. Timestamps are local (EDT).

## Run 1 — exploratory, SUPERSEDED (2026-09-14 ~00:05)

Probe c4b5a9fa (NYU-37, CNU + Isocortex), 100 s, 4 contiguous folds.
Kept in `lfp_compression_pilot_results/SUPERSEDED_run1_lfpack2500hz_pilot_c4b5a9fa`
for the record; **not used for any claim**, for two reasons found on inspection:

1. **lfpack was applied incorrectly.** It compressed the 2500 Hz signal, but IBL's
   production pipeline (`lfpack.compress_bin_to_h5`) compresses 250 Hz data in
   2048-sample chunks (8.2 s); at 2500 Hz the same chunks span 0.8 s and the
   thresholds are mistuned. Its numbers there are unrepresentative of lfpack.
2. **The forecast metric was ill-posed**: predicting 200 ms ahead from a single
   instant, it scored ~0 for every method including the 1152-feature reference.

What run 1 did show, and which shaped the revision below: the originally specified
candidate (inner-CV choice of the waveform/amplitude split, criterion = mean R² over
waveform, beta, gamma) chose 4 waveform + 12 amplitude dimensions at d = 16, losing
waveform R² (0.947 → 0.827) and behaviour r (0.432 → 0.382). The criterion weights
amplitude 2:1. Meanwhile waveform PCs 9–16 added only ~0.04 waveform R².

## Run 2 — protocol fixed at 2026-09-14 07:16 before running

**lfpack, as in production.** Input = the same shared 250 Hz signal every method gets
(destripe = 0.5 Hz high-pass + CAR at 2500 Hz, FIR decimation to 250 Hz). Compression
in 2048-sample chunks with 128-sample guard bands, levels default (ε150, α28) and
aggressive (ε450, α96). Cadzow denoising is off by default in `compress_bin_to_h5`;
a Cadzow variant (640-sample chunks, 64 halo, rank 5, as `run_cadzow_checkpoint`) is
included only if it runs in reasonable time.

**Forecast metric.** Predict Y(t + 200 ms) from Z at lags (−320, −160, 0 ms): causal history only.

**Methods (fixed).**
| Name | Dims @ 25 Hz | Role |
|---|---|---|
| waveform_pca_16 | 16 | CURRENT method (reimplemented from slides) |
| waveform_pca_20 | 20 | same budget as candidate C1 |
| spatial_avg_16 | 16 | Alon's simplest baseline |
| wave16+amp4 (C1) | 20 | candidate: current 16 PCs unchanged + 4 amplitude PCs |
| wave12+amp4 (C2) | 16 | candidate at the current budget |
| amplitude_pca_cv16 | 16 | run-1 specification (inner-CV split), reported as-is |
| wave14+amp2, wave8+amp8, wave4+amp12 | 16 | trade-off curve only; no claims |
| reference_1152 | 1152 | ceiling |

**Pre-declared comparisons.** C2 vs waveform_pca_16 (equal budget). C1 vs waveform_pca_20
(equal budget) and vs waveform_pca_16 (current).

**Pre-declared meaning of "better than current".** On held-out folds, at equal budget:
beta and gamma envelope R² higher; waveform R² lower by no more than 0.02;
behaviour r not lower. If amplitude improves but behaviour does not, the conclusion is
"preserves more frequency content at little waveform cost; no evidence of more
behavioural information", not "better".

**Replication (only if run 2 does not contradict run 1's basic trade-off).** Same
protocol, raw source, every other cached probe (6 sessions); per-probe paired
differences; counts of probes improved; number of distinct mice. Six sessions are a
preliminary replication, not an estimate of cross-animal reproducibility.

## Run 3 — numerical fix only, no protocol change (2026-09-14 07:36)

Run 2 finished with **lfpack without Cadzow** giving readout R² near −1400. Cause: that
source keeps SVD rank 1, so 15 of the 16 fitted waveform PCs are numerical noise, and
the readout's z-scoring divided by `sd + 1e-6`, amplifying that noise on held-out data.
Fix in `pilot_metrics._standardize`: features whose SD is below 1e-3 of the largest
feature SD are zeroed. Methods, splits, targets, criteria and comparisons are unchanged.
Run 2 outputs are kept as `SUPERSEDED_run2_sd_floor_*`; pilot and replication were
rerun in full with the fix, and only run 3 outputs are reported.

For the record, run-2 values for the pre-declared comparisons (to be checked against run 3):
replication, C2 vs PCA 16, median differences — waveform R² −0.004, gamma amplitude R²
+0.391, behaviour r +0.070 (10/11 probes); C1 vs PCA 20 — waveform −0.002, gamma
+0.389, behaviour +0.068 (11/11).
