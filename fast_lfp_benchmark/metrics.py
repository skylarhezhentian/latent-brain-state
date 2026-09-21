"""
Metrics for the five axes. Every function compares a candidate's reconstruction
x_hat against the destriped reference x, both [channels, time] in microvolts.

Why not MSE alone: LFP power falls roughly as 1/f, so broadband MSE is
dominated by delta. A representation can win on MSE while discarding gamma.
The frequency and temporal axes are resolved by band for exactly that reason.
"""
import numpy as np
import scipy.signal
from scipy.ndimage import uniform_filter1d
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score

# ephysatlas BANDS, with delta starting at 1 Hz: destriping high-passes at
# 0.5 Hz and the 1 s spectral windows resolve 1 Hz.
BANDS = {"delta": (1, 4), "theta": (4, 10), "alpha": (8, 12), "beta": (15, 30), "gamma": (30, 90)}
CHANNEL_STRIDE = 8          # spectral/temporal metrics on every 8th channel (160 µm apart)
ENV_FS = 50.0               # band envelopes averaged down to 20 ms bins
BEHAVIOR_WINDOW_S = 0.5


def subset(x):
    return x[::CHANNEL_STRIDE]


# ----------------------------------------------------------------------------- reconstruction

def reconstruction(x, x_hat):
    err = x - x_hat
    total = x - x.mean(axis=1, keepdims=True)
    return {
        "r2": float(1.0 - (err ** 2).sum() / max((total ** 2).sum(), 1e-12)),
        "rmse_uV": float(np.sqrt((err ** 2).mean())),
    }


# ----------------------------------------------------------------------------- frequency content

def frequency(xs, xs_hat, fs):
    """
    Per band:
      coh_*    magnitude-squared coherence between x and x_hat, 0..1. Is the
               band's signal present with the right phase?
      pow_db_* 10·log10(power(x_hat) / power(x)). 0 dB is exact; negative
               means the band was attenuated.
    """
    nper = int(fs)
    f, cxy = scipy.signal.coherence(xs, xs_hat, fs=fs, nperseg=nper, axis=-1)
    _, pxx = scipy.signal.welch(xs, fs=fs, nperseg=nper, axis=-1)
    _, phh = scipy.signal.welch(xs_hat, fs=fs, nperseg=nper, axis=-1)
    cxy = np.nan_to_num(cxy, nan=0.0)

    out = {}
    for band, (lo, hi) in BANDS.items():
        m = (f >= lo) & (f < hi)
        out[f"coh_{band}"] = float(cxy[:, m].mean())
        ratio = phh[:, m].mean(axis=1) / np.maximum(pxx[:, m].mean(axis=1), 1e-20)
        out[f"pow_db_{band}"] = float(np.mean(10 * np.log10(np.maximum(ratio, 1e-6))))
    return out


# ----------------------------------------------------------------------------- temporal information

def band_envelopes(xs, fs):
    """{band: [channels, T_env]} Hilbert amplitude, averaged into 1/ENV_FS bins."""
    k = int(round(fs / ENV_FS))
    out = {}
    for band, (lo, hi) in BANDS.items():
        sos = scipy.signal.butter(4, (lo, hi), btype="bandpass", fs=fs, output="sos")
        env = np.abs(scipy.signal.hilbert(scipy.signal.sosfiltfilt(sos, xs, axis=-1), axis=-1))
        n = env.shape[1] // k
        out[band] = env[:, :n * k].reshape(env.shape[0], n, k).mean(axis=2).astype(np.float32)
    return out


def acf_timescale(env, fs=ENV_FS, max_lag_s=5.0):
    """
    Lag (s) where the channel-averaged envelope autocorrelation first drops
    below 1/e, linearly interpolated between bins: gamma envelopes decorrelate
    within one or two 20 ms bins, so whole-bin lags would be too coarse to compare.
    """
    e = env - env.mean(axis=1, keepdims=True)
    n = e.shape[1]
    spec = np.abs(np.fft.rfft(e, n=2 * n, axis=1)) ** 2
    acf = np.fft.irfft(spec, axis=1)[:, :n]
    acf = acf / np.maximum(acf[:, :1], 1e-20)
    mean_acf = np.nanmean(acf[:, : int(max_lag_s * fs)], axis=0)
    below = np.nonzero(mean_acf < 1 / np.e)[0]
    if not below.size:
        return float(max_lag_s)
    k = below[0]
    if k == 0:
        return 0.0
    a, b = mean_acf[k - 1], mean_acf[k]
    return float((k - 1 + (a - 1 / np.e) / max(a - b, 1e-12)) / fs)


def temporal(env, env_hat):
    """
    Per band:
      env_corr_*  correlation over time of the amplitude envelope, averaged over
                  channels: are the band's fast amplitude dynamics preserved?
      tau_ratio_* envelope timescale of x_hat / timescale of x. 1 is exact; >1
                  means the representation smooths the dynamics out.
    """
    out = {}
    for band in BANDS:
        a, b = env[band], env_hat[band]
        a0 = a - a.mean(axis=1, keepdims=True)
        b0 = b - b.mean(axis=1, keepdims=True)
        denom = np.sqrt((a0 ** 2).sum(axis=1) * (b0 ** 2).sum(axis=1))
        corr = np.where(denom > 0, (a0 * b0).sum(axis=1) / np.maximum(denom, 1e-20), 0.0)
        out[f"env_corr_{band}"] = float(corr.mean())
        out[f"tau_ratio_{band}"] = acf_timescale(b) / max(acf_timescale(a), 1.0 / ENV_FS)
    return out


# ----------------------------------------------------------------------------- behavioural information

def behavior_features(env, n_bins, step_s):
    """
    log band power per group of channels, in BEHAVIOR_WINDOW_S windows centred
    on each behaviour bin -> [n_bins, groups × bands]. With CHANNEL_STRIDE = 8,
    groups of 2 subset channels span the repo's 16-channel samples.
    """
    feats = []
    idx = np.clip(np.round((np.arange(n_bins) * step_s + step_s / 2) * ENV_FS).astype(int), 0, None)
    for band in BANDS:
        p = env[band] ** 2
        g = p.shape[0] // 2
        p = p[: 2 * g].reshape(g, 2, -1).mean(axis=1)
        p = uniform_filter1d(p, size=int(BEHAVIOR_WINDOW_S * ENV_FS), axis=1, mode="nearest")
        feats.append(np.log10(np.maximum(p[:, np.clip(idx, 0, p.shape[1] - 1)], 1e-6)).T)
    return np.concatenate(feats, axis=1)


def behavior_r2(features, B, names, n_folds=5):
    """
    Cross-validated R² of ridge regression from features to each behaviour.
    Folds are contiguous blocks of time, so neighbouring autocorrelated bins
    never sit on both sides of a split.
    """
    out = {}
    n = features.shape[0]
    folds = np.array_split(np.arange(n), n_folds)
    for j, name in enumerate(names):
        y = B[:n, j].astype(float)
        ok = np.isfinite(y) & np.isfinite(features).all(axis=1)
        if ok.sum() < 0.5 * n or np.nanstd(y[ok]) == 0:
            out[f"beh_r2_{name}"] = np.nan
            continue
        pred = np.full(n, np.nan)
        for test in folds:
            train = np.setdiff1d(np.arange(n), test)
            tr, te = train[ok[train]], test[ok[test]]
            if len(tr) < 20 or len(te) == 0:
                continue
            mu, sd = features[tr].mean(0), features[tr].std(0) + 1e-6
            model = RidgeCV(alphas=np.logspace(-2, 4, 13)).fit((features[tr] - mu) / sd, y[tr])
            pred[te] = model.predict((features[te] - mu) / sd)
        good = ok & np.isfinite(pred)
        out[f"beh_r2_{name}"] = float(r2_score(y[good], pred[good])) if good.sum() > 10 else np.nan
    return out


# ----------------------------------------------------------------------------- reproducibility

def subspace_similarity(U_a, U_b):
    """Mean squared cosine of the principal angles between two r-dim subspaces. 1 = identical."""
    r = U_a.shape[1]
    return float(np.linalg.norm(U_a.T @ U_b, "fro") ** 2 / r)
