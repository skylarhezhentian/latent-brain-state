"""
Candidate LFP representations.

Every candidate implements the same small interface, so a new one (for
example the convolutional autoencoder) drops into the benchmark by
subclassing Representation and filling in reconstruct / floats_per_second.

Rate is counted as stored floats per second of recording. The reference
signal is 384 channels at 500 Hz = 192,000 floats/s; the native recording is
384 channels at 2500 Hz = 960,000 floats/s. lfpack's rate is measured on the
native signal it compresses, which is conservative against lfpack.
"""
import numpy as np

FS = 500.0
N_CHANNELS = 384


class Representation:
    name = "base"
    shared_latent = False   # are latent coordinates comparable across probes and sessions?
    n_params = 0            # learned parameters, the complexity axis

    def fit(self, probes):
        """probes: list of probe dicts from the cache (training sessions only)."""

    def reconstruct(self, probe):
        raise NotImplementedError

    def floats_per_second(self, probe):
        raise NotImplementedError


class Raw(Representation):
    """The reference itself. Scores perfectly by construction: a metrics sanity check."""
    name = "raw"

    def reconstruct(self, probe):
        return probe["x_uV"]

    def floats_per_second(self, probe):
        return N_CHANNELS * FS


class ChannelBin(Representation):
    """Average adjacent channels, as the repo's compressed pipeline does (16 -> 1)."""
    shared_latent = True

    def __init__(self, width=16):
        self.width = width
        self.name = f"bin{width}"

    def reconstruct(self, probe):
        x = probe["x_uV"]
        g = x.shape[0] // self.width
        means = x[: g * self.width].reshape(g, self.width, -1).mean(axis=1)
        return np.repeat(means, self.width, axis=0)

    def floats_per_second(self, probe):
        return (N_CHANNELS // self.width) * FS


class SpatialPCA(Representation):
    """
    Rank-r PCA across the 384 channels, fit on training sessions and applied to
    held-out ones. Caveat: channel index is probe depth, not anatomy, so the
    basis learns depth profiles that mix different regions across probes.
    """
    shared_latent = True

    def __init__(self, rank):
        self.rank = rank
        self.name = f"pca{rank}"
        self.n_params = N_CHANNELS * rank

    @staticmethod
    def covariance(x):
        xc = (x - x.mean(axis=1, keepdims=True)).astype(np.float64)
        return xc @ xc.T, xc.shape[1]

    def fit_from_covariance(self, cov):
        evals, evecs = np.linalg.eigh(cov)
        self.components = evecs[:, ::-1][:, : self.rank]
        self.explained = evals[::-1][: self.rank] / evals.sum()
        return self

    def reconstruct(self, probe):
        x = probe["x_uV"]
        mu = x.mean(axis=1, keepdims=True)
        W = self.components.astype(np.float32)
        return (W @ (W.T @ (x - mu)) + mu).astype(np.float32)

    def floats_per_second(self, probe):
        return self.rank * FS


class Lfpack(Representation):
    """
    IBL's lossy codec: adaptive SVD per 2048-sample chunk plus wavelet-packet
    thresholding. Its basis changes every chunk, so the coordinates carry no
    shared meaning across time, probes or sessions: compression for storage.
    """
    shared_latent = False

    def __init__(self, level):
        self.level = level
        self.name = f"lfpack_{level}"

    def reconstruct(self, probe):
        return probe[f"lfpack_{self.level}_uV"]

    def floats_per_second(self, probe):
        return float(probe[f"lfpack_{self.level}_floats_per_s"])
