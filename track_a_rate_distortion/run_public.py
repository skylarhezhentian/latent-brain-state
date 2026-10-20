"""Run the Track A pilot against the public IBL OpenAlyx database.

The analysis module contains the scientific protocol. This entry point adds the
public OpenAlyx username required by recent ONE-api releases and tolerates the
small probe-clock sampling-rate deviations present in some recordings.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import scipy.signal
from one.api import ONE as RealONE

import track_a_rate_distortion.run_track_a as analysis


def public_one(*args, **kwargs):
    """Instantiate ONE using the documented public OpenAlyx account name."""
    kwargs.setdefault("username", "intbrainlab")
    return RealONE(*args, **kwargs)


def load_destriped_250_tolerant(
    one,
    pid: str,
    t0_main_s: float,
    duration_s: float,
) -> Tuple[np.ndarray, float]:
    """Stream and destripe LFP, accepting realistic probe-clock deviations.

    Neuropixels clocks can report rates such as 2500.017 Hz instead of exactly
    2500 Hz.  The integer decimation factor remains ten; the returned sampling
    rate records the actual post-decimation rate so timing and rate accounting
    remain explicit rather than silently forcing an exact value.
    """
    analysis.CACHE.mkdir(parents=True, exist_ok=True)
    cache_file = analysis.CACHE / f"{pid}_destriped250.npz"
    if cache_file.exists():
        with np.load(cache_file) as z:
            return z["x_uv"].astype(np.float32), float(z["fs"])

    ssl = analysis.SpikeSortingLoader(pid=pid, one=one)
    sr = ssl.raw_electrophysiology(band="lf", stream=True)
    fs_native = float(sr.fs)
    ratio = fs_native / analysis.FS_OUT
    down = int(round(ratio))
    if down < 1 or abs(ratio - down) > 0.05:
        raise ValueError(
            f"Native fs={fs_native} is not close to an integer multiple of "
            f"target fs={analysis.FS_OUT}"
        )
    fs_out = fs_native / down

    try:
        t0_probe = float(
            ssl.timesprobe2times(np.array([t0_main_s]), direction="reverse")[0]
        )
    except Exception:
        t0_probe = float(t0_main_s)

    s0 = int(round(t0_probe * fs_native))
    n_total = int(round(duration_s * fs_native))
    chunk = int(round(10.0 * fs_native))
    pad = int(round(1.0 * fs_native))
    pieces: List[np.ndarray] = []

    try:
        for rel0 in range(0, n_total, chunk):
            rel1 = min(rel0 + chunk, n_total)
            desired0 = s0 + rel0
            desired1 = s0 + rel1
            a = max(0, desired0 - pad)
            b = desired1 + pad
            raw = analysis._read_stream(sr, a, b)
            x = analysis.destripe_lfp(
                raw,
                fs=fs_native,
                channel_labels=np.zeros(raw.shape[0]),
            ).astype(np.float32)
            y = scipy.signal.resample_poly(x, up=1, down=down, axis=1).astype(
                np.float32
            )
            keep0 = int(round((desired0 - a) / down))
            keep_n = int(round((desired1 - desired0) / down))
            pieces.append(y[:, keep0 : keep0 + keep_n])
    finally:
        try:
            sr.close()
        except Exception:
            pass

    x_uv = np.concatenate(pieces, axis=1) * 1e6
    target_n = int(round(duration_s * fs_out))
    x_uv = x_uv[:, :target_n].astype(np.float32)
    if x_uv.shape[0] != 384 or x_uv.shape[1] < target_n - 2:
        raise RuntimeError(
            f"Unexpected LFP shape {x_uv.shape}; expected approximately "
            f"(384, {target_n})"
        )
    np.savez_compressed(
        cache_file,
        x_uv=x_uv,
        fs=fs_out,
        native_fs=fs_native,
        decimation_factor=down,
    )
    return x_uv, fs_out


analysis.ONE = public_one
analysis.load_destriped_250 = load_destriped_250_tolerant

if __name__ == "__main__":
    analysis.main()
