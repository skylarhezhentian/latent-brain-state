"""
Load one probe and derive every target from the raw (destriped) signal.

Identical preprocessing for every source: the cached 500 Hz signal (raw or the
lfpack reconstruction) is decimated to 250 Hz with the same zero-phase FIR
filter, cropped to the same DURATION_S, and passed through the same target code.
"""
import json
from pathlib import Path

import numpy as np
import scipy.signal

from . import pilot_config as C


def find_probe(pid, cache_dir=C.CACHE_DIR):
    for sj in Path(cache_dir).glob("*/session.json"):
        meta = json.load(open(sj))
        if pid in meta["pids"]:
            return sj.parent / f"{pid}.npz", meta
    raise FileNotFoundError(f"{pid} is not in {cache_dir}; run fast_lfp_benchmark.extract first")


def list_cached_probes(cache_dir=C.CACHE_DIR):
    out = []
    for sj in sorted(Path(cache_dir).glob("*/session.json")):
        meta = json.load(open(sj))
        out += [(pid, meta) for pid in meta["pids"] if (sj.parent / f"{pid}.npz").exists()]
    return out


def to_250hz(x_500):
    """[channels, T] at 500 Hz -> 250 Hz, cropped to DURATION_S and to whole 40 ms bins."""
    y = scipy.signal.decimate(x_500.astype(np.float64), 2, ftype="fir", axis=1, zero_phase=True)
    n = int(round(C.DURATION_S * C.FS))
    n -= n % C.BIN
    return y[:, :n].astype(np.float32)


def load_raw(pid, cache_dir=C.CACHE_DIR):
    path, meta = find_probe(pid, cache_dir)
    with np.load(path) as z:
        return to_250hz(z["x_uV"]), meta, z["cosmos"].astype(str)


def load_sources(pid, with_lfpack=True):
    """
    ({source: [384, 25000] µV at 250 Hz}, lfpack cost info, session metadata, Cosmos labels).
    lfpack variants are computed from the raw 250 Hz signal (see pilot_lfpack.py).
    """
    raw, meta, cosmos = load_raw(pid)
    sources, info = {"raw": raw}, {}
    if with_lfpack:
        from .pilot_lfpack import lfpack_sources
        decoded, info = lfpack_sources(pid, raw)
        sources.update(decoded)
    return sources, info, meta, cosmos


def bin_mean(x):
    """[channels, T] at 250 Hz -> [channels, T/10] at 25 Hz by 40 ms averaging (the current method's downsampling)."""
    c, t = x.shape
    return x[:, : t - t % C.BIN].reshape(c, -1, C.BIN).mean(axis=2)


def log_rms_envelopes(x):
    """{band: [channels, T/10]} log10 RMS of the band-passed signal in 40 ms bins."""
    out = {}
    for band, (lo, hi) in C.ENVELOPE_BANDS.items():
        sos = scipy.signal.butter(4, (lo, hi), btype="bandpass", fs=C.FS, output="sos")
        xf = scipy.signal.sosfiltfilt(sos, x.astype(np.float64), axis=1)
        c, t = xf.shape
        rms = np.sqrt((xf[:, : t - t % C.BIN].reshape(c, -1, C.BIN) ** 2).mean(axis=2))
        out[band] = np.log10(rms + C.LOG_EPS_UV).astype(np.float32)
    return out


def targets_from_raw(raw_250):
    """What every representation is asked to preserve. Always computed from the raw source."""
    env = log_rms_envelopes(raw_250)
    return {"wave": bin_mean(raw_250), **{f"env_{b}": v for b, v in env.items()}}


def behavior_25hz(meta, n_bins):
    """
    Wheel speed and motion energy in 40 ms bins over the probe's interval,
    cached so later runs need no network. Loaded with the extractor's own code.
    """
    C_dir = Path(C.PILOT_CACHE_DIR)
    C_dir.mkdir(parents=True, exist_ok=True)
    f = C_dir / f"behavior_25hz_{meta['eid']}.npz"
    if not f.exists():
        from one.api import ONE
        from fast_lfp_benchmark.extract import behavior_matrix
        one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international", silent=True)
        B, names = behavior_matrix(one, meta["eid"], meta["t0_main_s"], C.DURATION_S, step_s=1.0 / C.FS_REP)
        subject = one.get_details(meta["eid"]).get("subject", "unknown")
        np.savez(f, B=B, names=np.array(names), subject=subject)
    with np.load(f) as z:
        return z["B"][:n_bins], [str(n) for n in z["names"]], str(z["subject"])
