"""
Stage-1 candidate representations (SEARCH_PROTOCOL.md). All map the shared 250 Hz,
384-channel signal to 16 dimensions at 25 Hz, and learn everything on training bins only.

They plug into lfp_compression_pilot's evaluation unchanged: fit(F, train_bins) and
transform(F) on a SearchFeatures object, which extends the pilot's Features with
amplitude envelopes for more bands and longer RMS windows, computed once and cached.
"""
import numpy as np
import scipy.signal
from scipy.ndimage import uniform_filter1d
from sklearn.utils.extmath import randomized_svd

from lfp_compression_pilot import pilot_config as C
from lfp_compression_pilot.pilot_data import bin_mean
from lfp_compression_pilot.pilot_representations import (AmplitudePCA, Representation, WaveformPCA,
                                                         bins_to_samples, top_pcs)

AMP_BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "alpha": (8.0, 12.0),
             "beta": (15.0, 30.0), "gamma": (30.0, 90.0)}
D = 16


class SearchFeatures:
    """Source-derived signals, computed lazily and cached per (band, RMS window)."""

    def __init__(self, x250):
        self.x250 = x250
        self.wave25 = bin_mean(x250)
        self._power = {}
        self._env = {}

    @property
    def env25(self):
        """Beta/gamma 40 ms log-RMS, what the pilot's AmplitudePCA reads. Lazy, so methods
        that never use amplitude are not charged for computing it."""
        return self.env(("beta", "gamma"))

    def _bin_power(self, band):
        """Mean squared band-passed signal per 40 ms bin: [384, T25]."""
        if band not in self._power:
            sos = scipy.signal.butter(4, AMP_BANDS[band], btype="bandpass", fs=C.FS, output="sos")
            xf = scipy.signal.sosfiltfilt(sos, self.x250.astype(np.float64), axis=1)
            c, t = xf.shape
            self._power[band] = (xf[:, : t - t % C.BIN].reshape(c, -1, C.BIN) ** 2).mean(axis=2)
        return self._power[band]

    def env(self, bands, rms_bins=1):
        """log10 RMS per band over rms_bins × 40 ms (centred), sampled every 40 ms: [384 × len(bands), T25]."""
        key = (tuple(bands), rms_bins)
        if key not in self._env:
            blocks = []
            for b in bands:
                p = self._bin_power(b)
                if rms_bins > 1:
                    p = uniform_filter1d(p, size=rms_bins, axis=1, mode="nearest")
                blocks.append(np.log10(np.sqrt(p) + C.LOG_EPS_UV))
            self._env[key] = np.concatenate(blocks, axis=0).astype(np.float32)
        return self._env[key]


def _zscore_fit(X, train):
    mu = X[:, train].mean(axis=1, keepdims=True)
    sd = X[:, train].std(axis=1, keepdims=True) + 1e-6
    return mu, sd


class WaveAmp(Representation):
    """k waveform PCs + m PCs of z-scored log-RMS amplitude over `bands`, RMS window rms_bins × 40 ms."""

    def __init__(self, name, k, m, bands, rms_bins=1):
        self.name, self.k, self.m, self.bands, self.rms_bins = name, k, m, tuple(bands), rms_bins
        self.dims = k + m
        self.wave = WaveformPCA(k)
        self.n_params = self.wave.n_params + C.N_CHANNELS * len(bands) * (m + 2)

    def fit(self, F, train):
        self.wave.fit(F, train)
        E = F.env(self.bands, self.rms_bins)
        self.mu, self.sd = _zscore_fit(E, train)
        self.We = top_pcs((E[:, train] - self.mu) / self.sd, self.m)
        return self

    def transform(self, F):
        E = (F.env(self.bands, self.rms_bins) - self.mu) / self.sd
        return np.concatenate([self.wave.transform(F), self.We.T @ E], axis=0)


class JointPCA(Representation):
    """One PCA over [waveform, amplitude of 4 bands], each channel z-scored, each block scaled to equal total variance."""
    name = "joint_pca_16"
    bands = ("theta", "alpha", "beta", "gamma")

    def __init__(self, d=D):
        self.dims = d
        self.n_params = C.N_CHANNELS * (1 + len(self.bands)) * (d + 2)

    def _stack(self, F):
        return np.concatenate([F.wave25, F.env(self.bands)], axis=0)

    def fit(self, F, train):
        X = self._stack(F)
        self.mu, self.sd = _zscore_fit(X, train)
        n_blocks = 1 + len(self.bands)
        # After per-channel z-scoring every block has total variance 384; scaling by
        # 1/sqrt(384) gives each of the 5 blocks total variance 1.
        self.block_scale = 1.0 / np.sqrt(C.N_CHANNELS)
        self.W = top_pcs((X[:, train] - self.mu) / self.sd * self.block_scale, self.dims)
        return self

    def transform(self, F):
        return self.W.T @ ((self._stack(F) - self.mu) / self.sd * self.block_scale)


class DelayPCA(Representation):
    """PCA of the waveform delay-embedded at 0, 40 and 80 ms (causal lags)."""
    name = "delay_pca_16"
    lags = (0, 1, 2)

    def __init__(self, d=D):
        self.dims = d
        self.n_params = C.N_CHANNELS * len(self.lags) * (d + 1)

    def _embed(self, F):
        w = F.wave25
        T = w.shape[1]
        return np.concatenate([w[:, np.clip(np.arange(T) - lag, 0, T - 1)] for lag in self.lags], axis=0)

    def fit(self, F, train):
        X = self._embed(F)
        self.mu = X[:, train].mean(axis=1, keepdims=True)
        self.W = top_pcs(X[:, train] - self.mu, self.dims)
        return self

    def transform(self, F):
        return self.W.T @ (self._embed(F) - self.mu)


class SpatiotemporalPCA(Representation):
    """PCA of each 40 ms window as one 384 × 10 vector at 250 Hz: keeps within-bin fast waveform."""
    name = "spatiotemporal_pca_16"

    def __init__(self, d=D):
        self.dims = d
        self.n_params = C.N_CHANNELS * C.BIN * (d + 1)

    @staticmethod
    def _windows(F):
        c, t = F.x250.shape
        n = t // C.BIN
        return F.x250[:, : n * C.BIN].reshape(c, n, C.BIN).transpose(1, 0, 2).reshape(n, c * C.BIN)

    def fit(self, F, train):
        X = self._windows(F)[train].astype(np.float64)
        self.mu = X.mean(axis=0)
        _, _, Vt = randomized_svd(X - self.mu, n_components=self.dims, n_iter=5, random_state=C.SEED)
        self.W = Vt.astype(np.float32)          # [d, 3840]
        return self

    def transform(self, F):
        return (self.W @ (self._windows(F) - self.mu).T).astype(np.float32)


class BandPowerPCA(Representation):
    """PCA over z-scored log-RMS of 5 bands, no waveform."""
    name = "bandpower_pca_16"
    bands = ("delta", "theta", "alpha", "beta", "gamma")

    def __init__(self, d=D):
        self.dims = d
        self.n_params = C.N_CHANNELS * len(self.bands) * (d + 2)

    def fit(self, F, train):
        E = F.env(self.bands)
        self.mu, self.sd = _zscore_fit(E, train)
        self.W = top_pcs((E[:, train] - self.mu) / self.sd, self.dims)
        return self

    def transform(self, F):
        return self.W.T @ ((F.env(self.bands) - self.mu) / self.sd)


class Autoencoder(Representation):
    """
    Small MLP autoencoder on z-scored [waveform, beta/gamma amplitude] (1152 features):
    1152 → 128 → 16 → 128 → 1152, trained on training bins with the last contiguous
    15% of them held out for early stopping.
    """
    name = "autoencoder_16"
    hidden, max_epochs, patience, lr, weight_decay, batch = 128, 300, 25, 1e-3, 1e-4, 256

    def __init__(self, d=D):
        self.dims = d
        n_in = C.N_CHANNELS * 3
        self.n_params = 2 * (n_in * self.hidden + self.hidden) + 2 * (self.hidden * d + d)

    def _stack(self, F):
        return np.concatenate([F.wave25, F.env(("beta", "gamma"))], axis=0)

    def fit(self, F, train):
        import torch
        from torch import nn
        torch.manual_seed(C.SEED)
        torch.set_num_threads(1)
        X = self._stack(F)
        self.mu, self.sd = _zscore_fit(X, train)
        Xs = ((X - self.mu) / self.sd).T.astype(np.float32)
        tr = np.sort(train)
        n_val = max(int(0.15 * len(tr)), 1)
        fit_rows, val_rows = tr[:-n_val - C.GAP_BINS], tr[-n_val:]
        n_in = Xs.shape[1]
        self.enc = nn.Sequential(nn.Linear(n_in, self.hidden), nn.GELU(), nn.Linear(self.hidden, self.dims))
        dec = nn.Sequential(nn.Linear(self.dims, self.hidden), nn.GELU(), nn.Linear(self.hidden, n_in))
        params = list(self.enc.parameters()) + list(dec.parameters())
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        xt, xv = torch.from_numpy(Xs[fit_rows]), torch.from_numpy(Xs[val_rows])
        gen = torch.Generator().manual_seed(C.SEED)
        best, best_state, wait = np.inf, None, 0
        for _ in range(self.max_epochs):
            for idx in torch.randperm(len(xt), generator=gen).split(self.batch):
                opt.zero_grad()
                loss = ((dec(self.enc(xt[idx])) - xt[idx]) ** 2).mean()
                loss.backward()
                opt.step()
            with torch.no_grad():
                v = float(((dec(self.enc(xv)) - xv) ** 2).mean())
            if v < best - 1e-5:
                best, wait = v, 0
                best_state = {k: t.clone() for k, t in self.enc.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    break
        self.enc.load_state_dict(best_state)
        return self

    def transform(self, F):
        import torch
        Xs = ((self._stack(F) - self.mu) / self.sd).T.astype(np.float32)
        with torch.no_grad():
            return self.enc(torch.from_numpy(Xs)).numpy().T


def stage1_methods():
    return [
        WaveformPCA(16),
        AmplitudePCA(16, 12),                                              # wave12+amp4, previous candidate
        WaveAmp("wave12+amp4_multiband", 12, 4, ("theta", "alpha", "beta", "gamma")),
        WaveAmp("wave12+amp4_smooth", 12, 4, ("beta", "gamma"), rms_bins=5),
        JointPCA(),
        DelayPCA(),
        SpatiotemporalPCA(),
        BandPowerPCA(),
        Autoencoder(),
    ]
