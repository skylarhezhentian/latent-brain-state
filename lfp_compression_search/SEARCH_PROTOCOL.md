# Search for a representation better than PCA — protocol

Written 2026-09-14 08:02 EDT, before any search result exists.

## Question

At the current budget (16 dimensions at 25 Hz), is there a representation that beats
spatial PCA of the waveform (the current method, `waveform_pca_16`)?

"Better", following Alon's stated goals (31 Aug slide 7: fidelity to the data, correlation
with behaviour, easy to run, easy to interpret), means on held-out data:

1. **Primary:** behaviour decoding r higher than PCA 16.
2. **Constraint:** waveform R² no more than 0.02 below PCA 16 (median over probes).
3. **Reported, not selected on:** beta/gamma amplitude R², alpha coherence, costs.

## Data

- **Search set:** the 12 probes already cached (6 sessions, `fast_lfp_cache`). These were
  used to develop the amplitude-augmented candidate, so they cannot confirm anything.
- **Confirmation set:** 6 new two-probe sessions (`fast_lfp_cache_confirm`, seed 1,
  excluding the search sessions). No analysis looks at them until the winner is fixed.

Same protocol as the pilot for both: 100 s, 250 Hz shared signal, targets from raw,
4 contiguous 25 s folds with 1 s gaps, ridge readout with inner blocked CV, behaviour with
±320 ms context, 30 s-shifted control. Every basis, scaling and network is fit on training
bins only. Behaviour is never used inside a probe to fit or tune anything.

## Methods (stage 1, fixed)

All 16 dimensions at 25 Hz.

| Name | Idea | Tests |
|---|---|---|
| `waveform_pca_16` | current method | baseline |
| `wave12+amp4` | previous candidate: 12 waveform PCs + 4 PCs of beta/gamma log-RMS (40 ms) | reference |
| `wave12+amp4_multiband` | amplitude PCs over theta, alpha, beta, gamma | do slow-band amplitudes add? |
| `wave12+amp4_smooth` | beta/gamma RMS over 200 ms windows (still sampled at 25 Hz) | is behaviour-relevant amplitude slower than 40 ms? |
| `joint_pca_16` | one PCA over block-standardised [waveform, 4 band amplitudes], blocks equally weighted | let the data choose the split |
| `delay_pca_16` | PCA of the waveform delay-embedded at lags 0, 40, 80 ms | does temporal structure help? |
| `spatiotemporal_pca_16` | PCA of 40 ms × 384-channel windows at 250 Hz | keep within-bin fast waveform |
| `bandpower_pca_16` | PCA over log-RMS of 5 bands only, no waveform | amplitude alone (Alon's band-power idea) |
| `autoencoder_16` | small MLP autoencoder, [waveform, beta/gamma amplitude] → 16 → same | nonlinear, complexity check |

## Selection rule (fixed)

On the search set, per method: median over the 12 probes of (method − PCA 16) for
behaviour r and waveform R². **Winner** = highest median behaviour-r gain among methods
whose median waveform-R² loss is ≤ 0.02. If no method gains behaviour r, the answer is
"no method found".

## Confirmation (fixed)

On the 12 confirmation probes, evaluate PCA 16, the winner, and `wave12+amp4`. The winner
is **better than PCA** only if, at the session level (6 sessions): behaviour r higher in at
least 5 of 6 sessions, and median waveform-R² loss ≤ 0.02. Otherwise report it as not
confirmed. Exploratory stage-2 combinations, if any, are labelled as such and not confirmed.
