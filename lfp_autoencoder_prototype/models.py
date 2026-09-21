"""
Two prototype autoencoders on the feature bank X [batch, 19 features, 24 depths, T bins].

Both share the same front end, with priors that suit a small dataset:
  - a 1×1 convolution mixes the 19 features at each depth, with the same weights at
    every depth and time (the local computation is the same everywhere on the probe);
  - two temporal convolutions (5 bins, dilation 1 and 2: ±240 ms of context);
  - one depth convolution over neighbouring groups (±320 µm), replicate-padded at the probe ends.

DepthLatentAE    bottleneck = 2 channels per depth -> 48 numbers per 40 ms, each tied to a depth
GlobalLatentAE   bottleneck = 48 numbers per 40 ms shared by all depths (flattened over depth)

The decoders mirror the encoders and reconstruct all 19 features at all 24 depths.
"""
import torch
from torch import nn

N_FEAT, N_DEPTH = 19, 24


def conv(cin, cout, k=(1, 1), d=(1, 1)):
    pad = (d[0] * (k[0] // 2), d[1] * (k[1] // 2))
    return nn.Conv2d(cin, cout, k, padding=pad, dilation=d, padding_mode="replicate")


def trunk(cin, hidden):
    return nn.Sequential(
        conv(cin, hidden), nn.GELU(),
        conv(hidden, hidden, (1, 5)), nn.GELU(),
        conv(hidden, hidden, (1, 5), (1, 2)), nn.GELU(),
        conv(hidden, hidden, (3, 1)), nn.GELU(),
    )


def untrunk(hidden, cout):
    return nn.Sequential(
        conv(hidden, hidden, (3, 1)), nn.GELU(),
        conv(hidden, hidden, (1, 5), (1, 2)), nn.GELU(),
        conv(hidden, hidden, (1, 5)), nn.GELU(),
        conv(hidden, cout),
    )


class DepthLatentAE(nn.Module):
    name = "ae_depth_48"

    def __init__(self, hidden=24, latent_per_depth=2):
        super().__init__()
        self.enc = nn.Sequential(trunk(N_FEAT, hidden), conv(hidden, latent_per_depth))
        self.dec = nn.Sequential(conv(latent_per_depth, hidden), nn.GELU(), untrunk(hidden, N_FEAT))

    def encode(self, x):                                   # -> [B, 2, 24, T]
        return self.enc(x)

    def forward(self, x):
        return self.dec(self.encode(x))

    @staticmethod
    def flatten(z):                                        # [B, 2, 24, T] -> [B, 48, T]
        return z.reshape(z.shape[0], -1, z.shape[-1])


class GlobalLatentAE(nn.Module):
    name = "ae_global_48"

    def __init__(self, hidden=24, latent=48):
        super().__init__()
        self.hidden = hidden
        self.enc_trunk = trunk(N_FEAT, hidden)
        self.enc_head = nn.Conv1d(hidden * N_DEPTH, latent, 1)          # flatten depth, per time bin
        self.dec_head = nn.Conv1d(latent, hidden * N_DEPTH, 1)
        self.dec_trunk = untrunk(hidden, N_FEAT)

    def encode(self, x):                                   # -> [B, 48, T]
        h = self.enc_trunk(x)
        return self.enc_head(h.reshape(h.shape[0], -1, h.shape[-1]))

    def forward(self, x):
        h = self.dec_head(self.encode(x))
        return self.dec_trunk(torch.nn.functional.gelu(h.reshape(h.shape[0], self.hidden, N_DEPTH, -1)))

    @staticmethod
    def flatten(z):
        return z


def view_weights():
    """Equal weight per view: the waveform feature carries as much loss as all 9 RMS or all 9 PSD features."""
    w = torch.ones(N_FEAT)
    w[1:10] = 1 / 9
    w[10:] = 1 / 9
    return (w / w.sum() * N_FEAT).view(1, N_FEAT, 1, 1)


def n_params(model):
    return sum(p.numel() for p in model.parameters())
