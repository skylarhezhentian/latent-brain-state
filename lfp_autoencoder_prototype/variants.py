"""
Options C, D and E of the autoencoder design, each a change to option A (DepthLatentAE):
same trunk, same 2 latents per depth (48 numbers per 40 ms), same decoder target
(the 19-feature bank at 24 depths, z-scored per insertion), same view-weighted loss.

MaskedDepthAE      C  whole depth groups are hidden in the input during training and every
                      depth is reconstructed; a mask-indicator channel tells the model which
                      depths are missing. At test time nothing is hidden unless we ask.
AnatomyAE          D  every depth also receives a learned 4-number embedding of its Cosmos
                      region, in the encoder and the decoder, so the latent need not spend
                      capacity on region identity. 'root' (fiber tracts, ventricles: white
                      matter inside the brain) and 'void' (outside the brain) get their own tokens.
FilterbankAE       E  the encoder never sees the hand-made features. It reads 2 raw channels
                      per depth at 250 Hz (160 um apart; the bank averages power over all 16),
                      passes them through 8 learned Gaussian band-pass filters (centre and
                      width trained), takes the mean squared band-passed signal per 40 ms bin
                      (the bank's RMS definition) averaged over the 2 channels, smooths it at
                      40 ms, 200 ms and 1 s, logs it, and adds the 25 Hz group waveform. The
                      decoder still reconstructs the 19-feature bank, so the comparison with A
                      isolates the front end.

WaveAmpAE          F  capacity allocated by physics: the sub-12.5 Hz waveform is spatially smooth and
                      one number per depth already keeps it, so it is stored as is (24 numbers), and
                      the network spends its 24 latents (1 per depth) on the 18 amplitude features
                      only (9 RMS + 9 PSD, equal weight). Linear twin: pooled group PCA with 1
                      amplitude component per depth + the stored waveform.

All models take a batch dict: x [B, 19, 24, T] (always), reg [B, 24] (D), raw [B, 48, L] (E).
"""
import math

import torch
from torch import nn

from .models import N_DEPTH, N_FEAT, conv, trunk, untrunk
from .cache_raw import OFFSETS, PAD

COSMOS = ["Isocortex", "OLF", "HPF", "CTXsp", "CNU", "TH", "HY", "MB", "HB", "CB", "root"]   # + 1 token for void/unknown
FS_RAW, BIN = 250.0, 10
PAD_BINS = PAD // BIN              # context stored in the raw cache (1 s)
CTX_BINS = 13                      # context E actually reads each side (0.52 s; the 1 s smoother needs 12)
RAW_CH = [g * len(OFFSETS) + i for g in range(N_DEPTH) for i in (0, 2)]   # channel offsets 2 and 10 of each group
N_RAW_PER_DEPTH = 2


def raw_window(raw, t0, T):
    """Cached padded raw [96, 25000 + 2*PAD] -> E's input for bins t0..t0+T with CTX_BINS of context."""
    return raw[RAW_CH, (PAD_BINS - CTX_BINS + t0) * BIN:(PAD_BINS + t0 + T + CTX_BINS) * BIN]


def region_ids(regions):
    return [COSMOS.index(r) if r in COSMOS else len(COSMOS) for r in regions]


class DepthLatentBase(nn.Module):
    """A's encoder/decoder with extra input channels to the encoder and the decoder."""

    def __init__(self, enc_in, dec_extra=0, hidden=24, latent_per_depth=2):
        super().__init__()
        self.enc = nn.Sequential(trunk(enc_in, hidden), conv(hidden, latent_per_depth))
        self.dec = nn.Sequential(conv(latent_per_depth + dec_extra, hidden), nn.GELU(), untrunk(hidden, N_FEAT))

    @staticmethod
    def flatten(z):
        return z.reshape(z.shape[0], -1, z.shape[-1])


class MaskedDepthAE(DepthLatentBase):
    name = "ae_masked_48"

    def __init__(self, mask=0.25):
        super().__init__(N_FEAT + 1)
        self.mask = mask

    def _input(self, x, keep):
        return torch.cat([x * keep, keep.expand(-1, 1, -1, x.shape[-1])], dim=1)

    def encode(self, b, keep=None):
        x = b["x"]
        if keep is None:
            keep = torch.ones(x.shape[0], 1, x.shape[2], 1)
        return self.enc(self._input(x, keep))

    def forward(self, b, keep=None):
        x = b["x"]
        if keep is None and self.training and self.mask > 0:
            keep = (torch.rand(x.shape[0], 1, x.shape[2], 1) > self.mask).float()
        return self.dec(self.encode(b, keep))


class AnatomyAE(DepthLatentBase):
    name = "ae_anatomy_48"

    def __init__(self, emb=4):
        super().__init__(N_FEAT + emb, dec_extra=emb)
        self.emb = nn.Embedding(len(COSMOS) + 1, emb)                    # last token: void / unknown

    def _e(self, b, T):
        return self.emb(b["reg"]).permute(0, 2, 1)[..., None].expand(-1, -1, -1, T)     # [B, emb, 24, T]

    def encode(self, b):
        x = b["x"]
        return self.enc(torch.cat([x, self._e(b, x.shape[-1])], dim=1))

    def forward(self, b):
        z = self.encode(b)
        return self.dec(torch.cat([z, self._e(b, z.shape[-1])], dim=1))


class WaveAmpAE(nn.Module):
    name = "ae_waveamp_48"

    def __init__(self, hidden=24):
        super().__init__()
        self.enc = nn.Sequential(trunk(N_FEAT - 1, hidden), conv(hidden, 1))
        self.dec = nn.Sequential(conv(1, hidden), nn.GELU(), untrunk(hidden, N_FEAT - 1))

    def encode(self, b):                                   # -> [B, 1, 24, T]: the learned amplitude latent
        return self.enc(b["x"][:, 1:])

    def forward(self, b):                                  # reconstructs the 18 amplitude features only
        return self.dec(self.encode(b))

    @staticmethod
    def flatten(z, x=None):                                # [waveform (24), amplitude latent (24)] channel-major
        z = z.reshape(z.shape[0], -1, z.shape[-1])
        return z if x is None else torch.cat([x[:, 0], z], dim=1)


class GaussianFilterbank(nn.Module):
    """K band-pass filters with Gaussian frequency responses; centres init log-spaced 2-80 Hz, width 0.35 x centre."""

    def __init__(self, k=8, f_lo=2.0, f_hi=80.0):
        super().__init__()
        c = torch.logspace(math.log10(f_lo), math.log10(f_hi), k)
        self.log_c = nn.Parameter(c.log())
        self.log_s = nn.Parameter((0.35 * c).log())

    def centres(self):
        return self.log_c.exp().clamp(0.5, 120.0)

    def widths(self):
        return self.log_s.exp().clamp(0.3, 60.0)

    def forward(self, raw):
        """raw [B, C, L] -> squared band-passed signal [B, C, K, L] (zero-phase filtering in the frequency domain)."""
        L = raw.shape[-1]
        X = torch.fft.rfft(raw, n=L)                                        # [B, C, F]
        f = torch.fft.rfftfreq(L, d=1.0 / FS_RAW)
        H = torch.exp(-0.5 * ((f[None] - self.centres()[:, None]) / self.widths()[:, None]) ** 2)   # [K, F]
        H = H / (H.pow(2).mean(dim=1, keepdim=True).sqrt() + 1e-8)          # white noise -> equal power per filter
        return torch.fft.irfft(X[:, :, None, :] * H[None, None], n=L) ** 2


class FilterbankAE(nn.Module):
    name = "ae_filterbank_48"
    SCALES = (1, 5, 25)                                                     # 40 ms, 200 ms, 1 s

    def __init__(self, k=8, hidden=24, latent_per_depth=2):
        super().__init__()
        self.fb = GaussianFilterbank(k)
        n_in = k * len(self.SCALES) + 1
        self.norm = nn.BatchNorm2d(n_in)
        self.enc = nn.Sequential(trunk(n_in, hidden), conv(hidden, latent_per_depth))
        self.dec = nn.Sequential(conv(latent_per_depth, hidden), nn.GELU(), untrunk(hidden, N_FEAT))

    @staticmethod
    def flatten(z):
        return z.reshape(z.shape[0], -1, z.shape[-1])

    def features(self, b):
        x, raw = b["x"], b["raw"]                                           # raw has CTX_BINS of context each side
        B, T = x.shape[0], x.shape[-1]
        p = self.fb(raw)                                                    # [B, 48, K, L]
        nb = p.shape[-1] // BIN
        p = p[..., : nb * BIN].reshape(B, N_DEPTH, N_RAW_PER_DEPTH, p.shape[2], nb, BIN).mean(dim=(2, 5))   # [B, 24, K, nb]
        p = p.permute(0, 2, 1, 3)                                           # [B, K, 24, nb]
        feats = []
        for s in self.SCALES:
            q = p if s == 1 else nn.functional.avg_pool2d(p, (1, s), stride=1, padding=(0, s // 2), count_include_pad=False)
            feats.append(torch.log10(q[..., CTX_BINS: CTX_BINS + T] + 1e-6))
        return self.norm(torch.cat(feats + [x[:, :1]], dim=1))              # + the 25 Hz group waveform

    def encode(self, b):
        return self.enc(self.features(b))

    def forward(self, b):
        return self.dec(self.encode(b))
