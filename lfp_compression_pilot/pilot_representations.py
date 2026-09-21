"""
The representations compared in the pilot. Each maps a 250 Hz, 384-channel
source signal to a d-dimensional signal at 25 Hz.

  SpatialAverage       average of d contiguous channel groups, then 40 ms averaging
                       (Alon, 31 Aug: "spatial averaging + temporal downsampling")
  WaveformPCA          d spatial PCs of the signed waveform, then 40 ms averaging
                       (the CURRENT method, Alon 8 Sep: "16 spatial PCA components at
                       25 Hz (temporally averaged)"). Reimplemented from the slides:
                       Alon's code is not in the repository.
  AmplitudePCA         the CANDIDATE family: k waveform PCs plus m PCs of beta and
                       gamma log-RMS envelopes, named wave{k}+amp{m}. C1 = wave16+amp4
                       (current 16 PCs kept, 4 added), C2 = wave12+amp4 (current budget).
  Reference            all 384 channels' waveform plus both envelopes (1152 features),
                       used only as a ceiling.

Why the candidate: a signed waveform averaged into 40 ms bins keeps nothing above
12.5 Hz, and Alon's 8 Sep result is that behaviour is carried mainly by LFP
amplitude (RMS envelope r = 0.46 vs signed PCA r = 0.28). The candidate spends
part of the same budget on exactly that amplitude.

Every learned quantity (PCA bases, envelope z-scoring) is fit on training bins
only. A Features object holds the source-derived signals so they are computed
once per source.
"""
from dataclasses import dataclass

import numpy as np

from . import pilot_config as C
from .pilot_data import bin_mean, log_rms_envelopes


@dataclass
class Features:
    x250: np.ndarray        # [384, T250] µV
    wave25: np.ndarray      # [384, T25]
    env25: np.ndarray       # [768, T25] beta then gamma log-RMS

    @classmethod
    def from_source(cls, x250):
        env = log_rms_envelopes(x250)
        return cls(x250, bin_mean(x250), np.concatenate([env[b] for b in C.ENVELOPE_BANDS], axis=0))


def bins_to_samples(bins):
    return (np.asarray(bins)[:, None] * C.BIN + np.arange(C.BIN)[None, :]).ravel()


def top_pcs(X, rank):
    """X [features, n] already centred. Returns the top-rank eigenvectors of X Xᵀ."""
    evals, evecs = np.linalg.eigh((X @ X.T).astype(np.float64))
    return evecs[:, ::-1][:, :rank].astype(np.float32)


class Representation:
    name = "base"
    dims = 0
    n_params = 0

    def fit(self, F: Features, train_bins):
        return self

    def transform(self, F: Features):
        raise NotImplementedError

    @property
    def floats_per_s(self):
        return self.dims * C.FS_REP


class SpatialAverage(Representation):
    def __init__(self, d):
        self.dims, self.name = d, f"spatial_avg_{d}"
        self.groups = np.array_split(np.arange(C.N_CHANNELS), d)

    def transform(self, F):
        return np.stack([F.wave25[g].mean(axis=0) for g in self.groups])


class WaveformPCA(Representation):
    def __init__(self, d, name=None):
        self.dims, self.name = d, name or f"waveform_pca_{d}"
        self.n_params = C.N_CHANNELS * (d + 1)

    def fit(self, F, train_bins):
        x = F.x250[:, bins_to_samples(train_bins)]
        self.mu = x.mean(axis=1, keepdims=True)
        self.W = top_pcs(x - self.mu, self.dims)      # PCA at 250 Hz, as "spatial PCA + temporal downsampling"
        return self

    def transform(self, F):
        # Projection and 40 ms averaging are both linear, so projecting the
        # averaged signal equals averaging the projected one.
        return self.W.T @ (F.wave25 - self.mu)


class AmplitudePCA(Representation):
    def __init__(self, d, k_wave, name=None):
        assert 0 < k_wave < d
        self.dims, self.k = d, k_wave
        self.name = name or f"wave{k_wave}+amp{d - k_wave}"
        self.wave = WaveformPCA(k_wave)
        n_env = C.N_CHANNELS * len(C.ENVELOPE_BANDS)
        self.n_params = self.wave.n_params + n_env * (d - k_wave) + 2 * n_env

    def fit(self, F, train_bins):
        self.wave.fit(F, train_bins)
        e = F.env25[:, train_bins]
        self.e_mu = e.mean(axis=1, keepdims=True)
        self.e_sd = e.std(axis=1, keepdims=True) + 1e-6
        self.We = top_pcs((e - self.e_mu) / self.e_sd, self.dims - self.k)
        return self

    def transform(self, F):
        z_env = self.We.T @ ((F.env25 - self.e_mu) / self.e_sd)
        return np.concatenate([self.wave.transform(F), z_env], axis=0)


class Reference(Representation):
    """Everything the targets are made of: a ceiling, not a compression."""
    name = "reference_1152"
    dims = C.N_CHANNELS * (1 + len(C.ENVELOPE_BANDS))

    def transform(self, F):
        return np.concatenate([F.wave25, F.env25], axis=0)


def build(name, k_wave, m_amp):
    """Method factory used by the runners: m_amp = 0 means plain waveform PCA."""
    return WaveformPCA(k_wave, name=name) if m_amp == 0 else AmplitudePCA(k_wave + m_amp, k_wave, name=name)
