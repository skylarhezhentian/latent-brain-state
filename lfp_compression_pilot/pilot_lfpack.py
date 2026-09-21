"""
lfpack as IBL runs it in production, applied to the same shared 250 Hz signal
every other method receives.

Production chain (lfpack.compress_bin_to_h5): 0.5 Hz high-pass + CAR at 2500 Hz,
FIR decimation to 250 Hz, optional Cadzow denoising, then SVD + wavelet-packet
compression in 2048-sample chunks with 128-sample guard bands. Our shared
preprocessing (ibldsp destripe_lfp + FIR decimation) covers the steps before
compression, so lfpack starts from exactly the pilot's raw reference.

Variants:
  lfpack_default           no Cadzow (compress_bin_to_h5's default), ε150 α28
  lfpack_cadzow_default    Cadzow (640-sample chunks, 64 halo, rank 5, as
                           run_cadzow_checkpoint) then ε150 α28, the setting
                           lfpack's own benchmark figures refer to
  lfpack_cadzow_aggressive Cadzow then ε450 α96

Costs are measured here, on the same 100 s and the same machine as the
representations: encode/decode seconds, peak traced memory, and measured storage
(U_scaled and wavelet coefficients written with numpy.savez_compressed).
Results are cached per probe so later runs are offline and fast.
"""
import io
import json
import time
import tracemalloc
from pathlib import Path

import numpy as np

from . import pilot_config as C

COMPRESS_CHUNK, COMPRESS_OVERLAP = 2048, 128
CADZOW_CHUNK, CADZOW_HALO = 640, 64
LEVELS = {"default": (150.0, 28.0), "aggressive": (450.0, 96.0)}
VARIANTS = {
    "lfpack_default": (False, "default"),
    "lfpack_cadzow_default": (True, "default"),
    "lfpack_cadzow_aggressive": (True, "aggressive"),
}


def _traced(fn):
    tracemalloc.start()
    t0 = time.perf_counter()
    out = fn()
    dt = time.perf_counter() - t0
    peak = tracemalloc.get_traced_memory()[1] / 1e6
    tracemalloc.stop()
    return out, dt, peak


def cadzow_250hz(x):
    """Chunked Cadzow denoising exactly as lfpack.run_cadzow_checkpoint does it."""
    import neuropixel
    from ibldsp.cadzow import cadzow_denoiser
    h = {k: v[: x.shape[0]] for k, v in neuropixel.trace_header(version=1).items()}
    ns = x.shape[1]
    out = np.zeros_like(x)
    for i0 in range(0, ns, CADZOW_CHUNK):
        i1 = min(i0 + CADZOW_CHUNK, ns)
        a, b = max(0, i0 - CADZOW_HALO), min(ns, i1 + CADZOW_HALO)
        y = cadzow_denoiser(x[:, a:b], h=h, fs=C.FS, rank=5, niter=1, fmax=None,
                            nswx=64, ovx=32, gap_threshold=2.0, ppca_k=2.0, n_jobs=1)
        out[:, i0:i1] = y[:, i0 - a:i0 - a + (i1 - i0)]
    return out


def compress_250hz(x, epsilon, alpha):
    import lfpack
    ns = x.shape[1]
    chunks = []
    for c0 in range(0, ns, COMPRESS_CHUNK):
        c1 = min(c0 + COMPRESS_CHUNK, ns)
        a, b = max(0, c0 - COMPRESS_OVERLAP), min(ns, c1 + COMPRESS_OVERLAP)
        chunks.append((c0, c1, a, lfpack.compress(x[:, a:b], epsilon=epsilon, alpha=alpha)))
    return chunks


def decompress_250hz(chunks, shape):
    import lfpack
    out = np.zeros(shape, dtype=np.float32)
    for c0, c1, a, comp in chunks:
        out[:, c0:c1] = lfpack.decompress(comp)[:, c0 - a:c0 - a + (c1 - c0)]
    return out


def stored_bytes(chunks):
    total = 0
    for *_, comp in chunks:
        buf = io.BytesIO()
        np.savez_compressed(buf, U=comp.U_scaled.astype(np.float32), V=comp.Vh_hat.astype(np.float32))
        total += buf.getbuffer().nbytes
    return total


def lfpack_sources(pid, raw250, verbose=True):
    """{variant: decoded [384, T] at 250 Hz}, {variant: cost dict}; cached per probe."""
    cache = Path(C.PILOT_CACHE_DIR)
    cache.mkdir(parents=True, exist_ok=True)
    f_arr, f_info = cache / f"lfpack250_{pid}.npz", cache / f"lfpack250_{pid}.json"
    if f_arr.exists() and f_info.exists():
        with np.load(f_arr) as z:
            return {k: z[k] for k in z.files}, json.load(open(f_info))

    x = raw250.astype(np.float32)
    duration = x.shape[1] / C.FS
    decoded, info = {}, {}

    denoised, t_cz, m_cz = _traced(lambda: cadzow_250hz(x))
    if verbose:
        print(f"[{pid[:8]}] Cadzow on {duration:.0f} s: {t_cz:.0f} s")

    for name, (use_cadzow, level) in VARIANTS.items():
        inp = denoised if use_cadzow else x
        eps, alpha = LEVELS[level]
        chunks, t_enc, m_enc = _traced(lambda: compress_250hz(inp, eps, alpha))
        rec, t_dec, m_dec = _traced(lambda: decompress_250hz(chunks, inp.shape))
        nbytes = stored_bytes(chunks)
        ranks = [comp.U_scaled.shape[1] for *_, comp in chunks]
        floats = sum(comp.U_scaled.size + int(np.count_nonzero(comp.Vh_hat)) for *_, comp in chunks)
        decoded[name] = rec
        info[name] = {
            "cadzow": use_cadzow, "epsilon": eps, "alpha": alpha,
            "cadzow_s_per_s": t_cz / duration if use_cadzow else 0.0,
            "cadzow_peak_traced_mb": m_cz if use_cadzow else 0.0,
            "encode_s_per_s": t_enc / duration, "decode_s_per_s": t_dec / duration,
            "encode_peak_traced_mb": m_enc, "decode_peak_traced_mb": m_dec,
            "stored_floats_per_s": floats / duration, "storage_bytes_per_s": nbytes / duration,
            "svd_rank_median": float(np.median(ranks)), "svd_rank_max": int(np.max(ranks)),
            # Codec-only fidelity: decoded vs the codec's own input (differs from raw only with Cadzow).
            "r2_vs_codec_input_250hz": float(1 - ((inp - rec) ** 2).sum() / ((inp - inp.mean(1, keepdims=True)) ** 2).sum()),
        }
        if verbose:
            print(f"[{pid[:8]}] {name}: rank median {info[name]['svd_rank_median']:.0f}, "
                  f"{info[name]['storage_bytes_per_s']:.0f} B/s, encode {t_enc:.1f} s")

    np.savez(f_arr, **decoded)
    json.dump(info, open(f_info, "w"), indent=1)
    return decoded, info
