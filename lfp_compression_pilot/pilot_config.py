"""
Every setting of the pilot, in one place, fixed before any result was seen.

The operating point mirrors the current method in Alon's slides (31 Aug, 8 Sep):
signals compared at 250 Hz, representations of 8 or 16 dimensions at 25 Hz.
"""
import os

# ---- data -------------------------------------------------------------------
# Probes come from the cache written by fast_lfp_benchmark.extract: destriped at
# 2500 Hz, lfpack round-tripped at 2500 Hz, both decimated to 500 Hz.
CACHE_DIR = os.path.expanduser("~/Downloads/lfp-brain-state/fast_lfp_cache")
PILOT_CACHE_DIR = os.path.expanduser("~/Downloads/lfp-brain-state/lfp_compression_pilot_cache")
RESULTS_DIR = os.path.expanduser("~/Downloads/lfp-brain-state/lfp_compression_pilot_results")

PILOT_PID = "c4b5a9fa-10cb-4195-9c17-15b6a1f77f9a"   # first cached probe; CNU + Isocortex
DURATION_S = 100.0          # Alon's new dataset uses 100 s per probe
FS_CACHE = 500.0
FS = 250.0                  # comparison rate in Alon's slides ("comparing at 250 Hz, because of LFpack")
BIN = 10                    # 10 samples at 250 Hz = 40 ms
FS_REP = FS / BIN           # 25 Hz: the representation rate of the current method
N_CHANNELS = 384

# ---- targets ------------------------------------------------------------------
# The 25 Hz waveform can only carry < 12.5 Hz. Faster bands enter as amplitude.
WAVE_BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0)}
ENVELOPE_BANDS = {"beta": (15.0, 30.0), "gamma": (30.0, 90.0)}
LOG_EPS_UV = 1e-3

# ---- methods --------------------------------------------------------------------
AMP_SHARE_GRID = (0.25, 0.5, 0.75)       # run-1 inner-CV grid, kept for the amplitude_pca_cv16 variant
LFPACK_SOURCES = ("lfpack_default", "lfpack_cadzow_default", "lfpack_cadzow_aggressive")
SOURCES = ("raw",) + LFPACK_SOURCES

# Methods fixed in PREREGISTRATION.md (run 2). (k_wave, m_amp); m_amp = 0 is plain waveform PCA.
CURRENT = (16, 0)
PRIMARY_METHODS = {
    "waveform_pca_16": (16, 0),     # current method
    "waveform_pca_20": (20, 0),     # same budget as C1
    "wave16+amp4": (16, 4),         # C1
    "wave12+amp4": (12, 4),         # C2
}
TRADEOFF_METHODS = {"wave14+amp2": (14, 2), "wave8+amp8": (8, 8), "wave4+amp12": (4, 12)}

# ---- evaluation -------------------------------------------------------------------
N_FOLDS = 4                  # contiguous blocks of 25 s
GAP_BINS = 25                # 1 s excluded from training on each side of a test block
INNER_FOLDS = 3
RIDGE_ALPHAS = tuple(10.0 ** k for k in range(-3, 5))
FAST_WINDOW_BINS = 25        # "fast" = what remains after removing a 1 s moving average
FORECAST_BINS = 5            # 200 ms ahead
FORECAST_LAGS = (-8, -4, 0)  # from causal history: -320, -160, 0 ms
BEHAVIOR_LAGS = (-8, -4, 0, 4, 8)          # ±320 ms of context for behaviour decoding
SHUFFLE_SHIFT_S = 30.0                     # control: behaviour circularly shifted by 30 s
COHERENCE_NPERSEG = 50                     # 2 s at 25 Hz
CHANNEL_STRIDE = 8                         # coherence on every 8th channel
SEED = 0

# Raw storage references for compression ratios.
RAW_FLOAT32_BYTES_PER_S_2500HZ = N_CHANNELS * 2500 * 4
RAW_FLOAT32_BYTES_PER_S_250HZ = N_CHANNELS * 250 * 4
