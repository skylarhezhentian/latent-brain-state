"""
Extract simultaneous raw-LFP windows for the benchmark.

For each session with two probes, the same stretch of session time is streamed
from both probes (aligned through each probe's sync to the main clock),
destriped at the native 2500 Hz, compressed with lfpack at both IBL levels,
decimated to 500 Hz, and cached per probe together with behaviour.

lfpack is applied to the destriped signal, not the Cadzow-denoised signal IBL
uses in production, so every candidate compresses exactly the same input.
Its published error figures are therefore not directly comparable to ours.

    python -m fast_lfp_benchmark.extract --sessions 6 --duration 120
"""
import argparse
import glob
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import scipy.signal
import lfpack
from lfpack._core import _COMPRESS_CHUNK, _COMPRESS_OVERLAP
from brainbox.io.one import SessionLoader, SpikeSortingLoader
from ibldsp.voltage import destripe_lfp
from iblatlas.regions import BrainRegions
from one.api import ONE

FS_OUT = 500.0
DECIMATE = 5                 # 2500 Hz -> 500 Hz
BEHAVIOR_STEP_S = 0.1
READ_PIECE_S = 10.0          # streamed in pieces; large single reads are fragile behind the proxy
PROCESS_CHUNK_S = 60.0
PAD_S = 2.0                  # destripe / decimation edge guard, trimmed after processing
FIRST_TRIAL = 90             # start inside the task, matching the repo's trial_ind = 90..189
LFPACK_LEVELS = {"default": (150.0, 28.0), "aggressive": (450.0, 96.0)}

DATA_DIR = os.path.expanduser("~/Downloads/lfp-brain-state/behaviroal-ephys-atlas")
CACHE_DIR = os.path.expanduser("~/Downloads/lfp-brain-state/fast_lfp_cache")


def two_probe_sessions(data_dir=DATA_DIR):
    """{eid: [pid, pid]} for sessions where both probes are 384-channel NP1."""
    by_eid = {}
    for f in glob.glob(os.path.join(data_dir, "*", "manifest.json")):
        m = json.load(open(f))
        if m.get("n_channels") == 384:
            by_eid.setdefault(m["eid"], []).append(m["pid"])
    return {eid: sorted(p) for eid, p in sorted(by_eid.items()) if len(p) == 2}


def read_stream(sr, first, last):
    """[nc, ns] float32 volts, read in READ_PIECE_S pieces, sync channel dropped."""
    step = int(READ_PIECE_S * sr.fs)
    pieces = [
        np.asarray(sr[a:min(a + step, last), :-sr.nsync], dtype=np.float32)
        for a in range(first, last, step)
    ]
    return np.concatenate(pieces, axis=0).T


def lfpack_roundtrip(x_v, epsilon, alpha):
    """
    Compress and decompress [nc, ns] exactly as lfpack's writer does: chunks of
    2048 samples, each extended by 128 samples of context on both sides, keeping
    only the centre. Returns (reconstruction, stored floats per sample of signal).
    """
    nc, ns = x_v.shape
    out = np.zeros_like(x_v)
    stored = 0.0
    for c0 in range(0, ns, _COMPRESS_CHUNK):
        c1 = min(c0 + _COMPRESS_CHUNK, ns)
        a, b = max(0, c0 - _COMPRESS_OVERLAP), min(ns, c1 + _COMPRESS_OVERLAP)
        comp = lfpack.compress(x_v[:, a:b], epsilon=epsilon, alpha=alpha)
        out[:, c0:c1] = lfpack.decompress(comp)[:, c0 - a:c0 - a + (c1 - c0)]
        n_kept_floats = nc * (b - a) / comp.cr_total
        stored += n_kept_floats * (c1 - c0) / (b - a)   # charge only the kept centre
    return out, stored / ns


def extract_probe(one, pid, t0_main, duration_s, with_lfpack=True):
    ssl = SpikeSortingLoader(pid=pid, one=one)
    sr = ssl.raw_electrophysiology(band="lf", stream=True)
    fs = float(sr.fs)

    try:
        t0_probe = float(ssl.timesprobe2times(np.array([t0_main]), direction="reverse")[0])
        synced = True
    except Exception as e:
        print(f"  [{pid[:8]}] sync unavailable ({type(e).__name__}); using main-clock time")
        t0_probe, synced = t0_main, False

    s0 = int(round(t0_probe * fs))
    n_total = int(round(duration_s * fs))
    pad = int(PAD_S * fs)
    step = int(PROCESS_CHUNK_S * fs)
    pad_out = pad // DECIMATE

    x_out, rec_out = [], {lvl: [] for lvl in LFPACK_LEVELS}
    rates = {lvl: [] for lvl in LFPACK_LEVELS}

    for c0 in range(0, n_total, step):
        c1 = min(c0 + step, n_total)
        raw = read_stream(sr, s0 + c0 - pad, s0 + c1 + pad)
        x = destripe_lfp(raw, fs=fs, channel_labels=np.zeros(raw.shape[0])).astype(np.float32)

        def to_out(v):
            y = scipy.signal.decimate(v, DECIMATE, ftype="fir", axis=1, zero_phase=True)
            return (y[:, pad_out:y.shape[1] - pad_out] * 1e6).astype(np.float32)

        x_out.append(to_out(x))
        for lvl, (eps, alpha) in (LFPACK_LEVELS.items() if with_lfpack else ()):
            rec, per_sample = lfpack_roundtrip(x, eps, alpha)
            rec_out[lvl].append(to_out(rec))
            rates[lvl].append((per_sample * fs, c1 - c0))
    sr.close()

    channels = ssl.load_channels()
    atlas_id = np.asarray(channels.get("atlas_id", np.zeros(384)), dtype=int)
    cosmos = BrainRegions().id2acronym(atlas_id, mapping="Cosmos")

    def weighted(pairs):
        return float(sum(r * n for r, n in pairs) / sum(n for _, n in pairs))

    lfpack_arrays = {}
    if with_lfpack:   # legacy 2500 Hz lfpack arrays; not used by lfp_compression_pilot
        lfpack_arrays.update({f"lfpack_{lvl}_uV": np.concatenate(rec_out[lvl], axis=1) for lvl in LFPACK_LEVELS})
        lfpack_arrays.update({f"lfpack_{lvl}_floats_per_s": weighted(rates[lvl]) for lvl in LFPACK_LEVELS})
    return {
        "x_uV": np.concatenate(x_out, axis=1),
        **lfpack_arrays,
        "axial_um": np.asarray(channels.get("axial_um", np.arange(384) * 20.0), dtype=float),
        "xyz_m": np.stack([np.asarray(channels.get(k, np.full(384, np.nan)), dtype=float) for k in "xyz"], 1),
        "atlas_id": atlas_id,
        "cosmos": np.asarray(cosmos).astype(str),
        "synced": synced,
    }


def behavior_matrix(one, eid, t0_main, duration_s, step_s=BEHAVIOR_STEP_S):
    """Wheel speed and motion energy, averaged into step_s bins starting at t0."""
    sl = SessionLoader(one=one, eid=eid)
    sl.load_session_data(trials=False, wheel=True, pose=False, motion_energy=True, pupil=False)
    centers = t0_main + step_s / 2 + np.arange(int(round(duration_s / step_s))) * step_s

    series = {}
    wheel = getattr(sl, "wheel", None)
    if wheel is not None and len(wheel):
        series["wheel_speed"] = (wheel["times"].to_numpy(), np.abs(wheel["velocity"].to_numpy()))
    me = getattr(sl, "motion_energy", None) or {}
    for cam, col, name in (
        ("leftCamera", "whiskerMotionEnergy", "whisker_me_left"),
        ("rightCamera", "whiskerMotionEnergy", "whisker_me_right"),
        ("bodyCamera", "bodyMotionEnergy", "body_me"),
    ):
        df = me.get(cam)
        if df is not None and len(df) and col in df:
            series[name] = (df["times"].to_numpy(), df[col].to_numpy())

    names = sorted(series)
    B = np.full((len(centers), len(names)), np.nan, dtype=np.float32)
    edges = np.append(centers - step_s / 2, centers[-1] + step_s / 2)
    for j, name in enumerate(names):
        t, v = series[name]
        ok = np.isfinite(t) & np.isfinite(v)
        idx = np.digitize(t[ok], edges) - 1
        inside = (idx >= 0) & (idx < len(centers))
        sums = np.bincount(idx[inside], weights=v[ok][inside], minlength=len(centers))
        counts = np.bincount(idx[inside], minlength=len(centers))
        B[:, j] = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    return B, names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=6)
    ap.add_argument("--duration", type=float, default=120.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default=CACHE_DIR)
    ap.add_argument("--no-lfpack", action="store_true",
                    help="skip the legacy 2500 Hz lfpack arrays (about halves extraction time)")
    ap.add_argument("--exclude-cache-dir", default=None,
                    help="skip sessions already cached here (to build an untouched confirmation set)")
    args = ap.parse_args()

    sessions = two_probe_sessions()
    rng = np.random.default_rng(args.seed)
    order = [sorted(sessions)[i] for i in rng.permutation(len(sessions))]
    if args.exclude_cache_dir:
        used = {p.parent.name for p in Path(args.exclude_cache_dir).glob("*/session.json")}
        order = [eid for eid in order if eid not in used]
        print(f"excluding {len(used)} sessions already in {args.exclude_cache_dir}")
    print(f"{len(sessions)} two-probe NP1 sessions; extracting {args.sessions} public ones (seed {args.seed})")

    one = ONE(base_url="https://openalyx.internationalbrainlab.org", password="international", silent=True)
    root = Path(args.cache_dir)
    done = sum((root / eid / "session.json").exists() for eid in sessions)

    for eid in order:
        if done >= args.sessions:
            break
        out_dir = root / eid
        if (out_dir / "session.json").exists():
            continue
        tic = time.time()
        try:
            trials = one.load_object(eid, "trials", collection="alf")
        except Exception as e:
            # The atlas includes sessions that are not released on the public server.
            print(f"[{eid[:8]}] not available publicly ({type(e).__name__}), skipping")
            continue

        stim = np.asarray(trials["stimOn_times"], dtype=float)
        start = next((i for i in range(FIRST_TRIAL, len(stim)) if np.isfinite(stim[i])), None)
        if start is None:
            print(f"[{eid[:8]}] fewer than {FIRST_TRIAL} trials, skipping")
            continue
        t0 = float(stim[start] - 1.0)

        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            B, beh_names = behavior_matrix(one, eid, t0, args.duration)
            for pid in sessions[eid]:
                d = extract_probe(one, pid, t0, args.duration, with_lfpack=not args.no_lfpack)
                np.savez_compressed(out_dir / f"{pid}.npz", behavior=B, **d)
                print(f"[{eid[:8]}] {pid[:8]} | {d['x_uV'].shape} | synced {d['synced']} | "
                      f"regions {sorted(set(d['cosmos']) - {'void', 'root'})}")
        except Exception as e:
            print(f"[{eid[:8]}] failed ({type(e).__name__}: {e}), skipping")
            for f in out_dir.glob("*"):
                f.unlink()
            out_dir.rmdir()
            continue

        json.dump(
            {"eid": eid, "pids": sessions[eid], "t0_main_s": t0, "first_trial": start,
             "duration_s": args.duration, "fs": FS_OUT, "behavior_step_s": BEHAVIOR_STEP_S,
             "behavior_names": beh_names},
            open(out_dir / "session.json", "w"), indent=1,
        )
        done += 1
        print(f"[{eid[:8]}] done in {time.time() - tic:.0f} s ({done}/{args.sessions})")


if __name__ == "__main__":
    main()
