"""
Splits, the shared linear readout, and the metrics for every axis.

One readout for everything: ridge regression from a representation Z(t) to a
target Y(t), fit on training bins, scored on held-out bins. Ridge strength is
chosen by blocked cross-validation inside the training bins, never on test data.
Because every method is read out the same way to the same raw-derived targets,
differences reflect what each representation contains, not how it is decoded.
"""
import numpy as np
import scipy.signal

from . import pilot_config as C


# ----------------------------------------------------------------------------- splits

def outer_folds(n_bins, n_folds=C.N_FOLDS, gap=C.GAP_BINS):
    """Contiguous test blocks; training excludes `gap` bins on both sides of the test block."""
    idx = np.arange(n_bins)
    for test in np.array_split(idx, n_folds):
        train = idx[(idx < test[0] - gap) | (idx > test[-1] + gap)]
        yield train, test


def inner_folds(train, n_folds=C.INNER_FOLDS, gap=C.GAP_BINS):
    """Contiguous validation chunks of the (sorted) training bins, with the same gap rule."""
    for val in np.array_split(np.sort(train), n_folds):
        tr = train[(train < val[0] - gap) | (train > val[-1] + gap)]
        yield tr, val


# ----------------------------------------------------------------------------- ridge readout

SD_FLOOR_RELATIVE = 1e-3


def _standardize(Z, rows):
    """
    Z-scoring statistics from training rows. Features whose SD is below 1e-3 of the
    largest feature SD get SD = inf, i.e. they are zeroed. This matters when a source
    is nearly degenerate (lfpack without Cadzow keeps SVD rank 1): PCs beyond its
    rank are numerical noise, and dividing them by a tiny SD makes the readout
    explode on held-out data. Healthy features are far above the floor.
    """
    mu = Z[:, rows].mean(axis=1, keepdims=True)
    sd = Z[:, rows].std(axis=1, keepdims=True)
    floor = SD_FLOOR_RELATIVE * max(float(sd.max()), 1e-12)
    return mu, np.where(sd > floor, sd, np.inf)


def _ridge_path(Xtr, Ytr, alphas):
    """Closed-form ridge weights for every alpha from one SVD. X [n, p], Y [n, q], both centred."""
    U, s, Vt = np.linalg.svd(Xtr, full_matrices=False)
    UtY = U.T @ Ytr
    return [Vt.T @ ((s / (s ** 2 + a))[:, None] * UtY) for a in alphas]


def r2_weighted(Y, P):
    """1 - SSE / SST pooled over outputs (variance-weighted R²). Y, P [n, q]."""
    sst = ((Y - Y.mean(axis=0)) ** 2).sum()
    return float(1.0 - ((Y - P) ** 2).sum() / max(sst, 1e-12))


def fit_readout(Z, Y, train, alphas=C.RIDGE_ALPHAS):
    """
    Z [p, T], Y [q, T]. Returns predict(bins) -> [len(bins), q].
    Alpha maximises inner blocked-CV R² on the training bins.
    """
    X = Z.T.astype(np.float64)
    T = Y.T.astype(np.float64)

    def solve(rows):
        mu, sd = _standardize(Z, rows)
        xm = ((X[rows] - mu.T) / sd.T)
        ym = T[rows].mean(axis=0)
        return mu, sd, ym, _ridge_path(xm, T[rows] - ym, alphas)

    scores = np.zeros(len(alphas))
    for tr, val in inner_folds(train):
        mu, sd, ym, Ws = solve(tr)
        xv = (X[val] - mu.T) / sd.T
        scores += [r2_weighted(T[val], xv @ W + ym) for W in Ws]
    best = int(np.argmax(scores))

    mu, sd, ym, Ws = solve(train)
    W = Ws[best]

    def predict(rows):
        return ((X[rows] - mu.T) / sd.T) @ W + ym

    predict.alpha = alphas[best]
    return predict


# ----------------------------------------------------------------------------- reconstruction / frequency / temporal

def moving_average(Y, w=C.FAST_WINDOW_BINS):
    """Centred moving average along rows of Y [n, q] (edges use a shrinking window)."""
    k = np.ones(w) / w
    num = scipy.signal.convolve(Y, k[:, None], mode="same")
    den = scipy.signal.convolve(np.ones((Y.shape[0], 1)), k[:, None], mode="same")
    return num / den


def readout_metrics(Z, Y, train, test, label):
    """Held-out R² of the target, and of its fast (< 1 s) component."""
    predict = fit_readout(Z, Y, train)
    P, Yt = predict(test), Y.T[test].astype(np.float64)
    out = {
        f"r2_{label}": r2_weighted(Yt, P),
        f"fast_r2_{label}": r2_weighted(Yt - moving_average(Yt), P - moving_average(P)),
    }
    return out, P, Yt


def forecast_r2(Z, Y, train, test, label, h=C.FORECAST_BINS):
    """
    R² of predicting Y(t+h) from Z's causal history (lags FORECAST_LAGS, <= t).
    Only pairs whose both ends fall in the same split are used.
    """
    Zl = lagged(Z, C.FORECAST_LAGS)
    tr = train[np.isin(train + h, train)]
    te = test[np.isin(test + h, test)]
    Ysh = np.zeros_like(Y)
    Ysh[:, :-h] = Y[:, h:]
    predict = fit_readout(Zl, Ysh, tr)
    return {f"forecast200ms_r2_{label}": r2_weighted(Ysh.T[te].astype(np.float64), predict(te))}


def waveform_band_coherence(Yt, P, fs=C.FS_REP):
    """Mean magnitude-squared coherence between held-out waveform and its readout, per slow band."""
    cols = np.arange(0, Yt.shape[1], C.CHANNEL_STRIDE)
    f, coh = scipy.signal.coherence(Yt[:, cols], P[:, cols], fs=fs, nperseg=C.COHERENCE_NPERSEG, axis=0)
    coh = np.nan_to_num(coh)
    return {f"coh_{b}": float(coh[(f >= lo) & (f < hi)].mean()) for b, (lo, hi) in C.WAVE_BANDS.items()}


# ----------------------------------------------------------------------------- behaviour

def lagged(Z, lags=C.BEHAVIOR_LAGS):
    """Stack Z at several lags (edge-padded): [p * len(lags), T]."""
    T = Z.shape[1]
    out = []
    for lag in lags:
        idx = np.clip(np.arange(T) + lag, 0, T - 1)
        out.append(Z[:, idx])
    return np.concatenate(out, axis=0)


def behavior_metrics(Z, B, names, train, test, prefix="beh", lags=C.BEHAVIOR_LAGS):
    """Held-out R² and Pearson r per behaviour, from Z with context at `lags` (default ±320 ms)."""
    Zl = lagged(Z, lags)
    out = {}
    for j, name in enumerate(names):
        y = B[:, j].astype(np.float64)
        ok = np.isfinite(y)
        tr, te = train[ok[train]], test[ok[test]]
        if len(tr) < 200 or len(te) < 50 or np.std(y[tr]) == 0:
            out[f"{prefix}_r2_{name}"] = out[f"{prefix}_r_{name}"] = np.nan
            continue
        predict = fit_readout(Zl, y[None, :], tr)
        p = predict(te)[:, 0]
        out[f"{prefix}_r2_{name}"] = r2_weighted(y[te, None], p[:, None])
        out[f"{prefix}_r_{name}"] = float(np.corrcoef(y[te], p)[0, 1]) if np.std(p) > 0 else np.nan
    return out


# ----------------------------------------------------------------------------- codec fidelity and reproducibility

def codec_fidelity(raw250, rec250, fs=C.FS):
    """Direct comparison of a decoded signal with the raw one at 250 Hz: R² and per-band coherence."""
    cols = np.arange(0, raw250.shape[0], C.CHANNEL_STRIDE)
    a, b = raw250[cols].astype(np.float64), rec250[cols].astype(np.float64)
    f, coh = scipy.signal.coherence(a, b, fs=fs, nperseg=int(fs), axis=1)
    coh = np.nan_to_num(coh)
    bands = {**C.WAVE_BANDS, **C.ENVELOPE_BANDS}
    out = {"r2_250hz": r2_weighted(raw250.T.astype(np.float64), rec250.T.astype(np.float64))}
    out.update({f"coh250_{bnd}": float(coh[:, (f >= lo) & (f < hi)].mean()) for bnd, (lo, hi) in bands.items()})
    return out


def subspace_similarity(Ua, Ub):
    """Mean squared cosine of principal angles between two r-dim subspaces (1 = identical, r/n = random)."""
    return float(np.linalg.norm(Ua.T @ Ub, "fro") ** 2 / Ua.shape[1])
