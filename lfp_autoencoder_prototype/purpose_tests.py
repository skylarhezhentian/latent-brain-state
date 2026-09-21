"""
Tests of what the representation is FOR, beyond the per-insertion benchmark.

The benchmark (lfp_selected_benchmark.evaluate) asks, within one insertion, how much waveform,
amplitude and behaviour a readout can recover. The later project tasks need more than that:

  transfer    Task 5 (brain-wide atlas across animals) needs coordinates that mean the same thing
              in every insertion. Zero-shot decoding: a ridge readout (±320 ms lags) is trained on
              all OTHER sessions and applied unchanged to the held-out session. A representation
              whose axes are re-fit per insertion (per-insertion PCA) has no reason to transfer.
              Two readouts: 'full' (every dimension) and, for depth-indexed methods, 'pooled'
              (mean and SD over the 24 depths of each latent channel: independent of where on
              the probe a signal sits).
  cca         Task 2 (global vs local dynamics) needs the component shared between simultaneously
              recorded probes. Cross-validated regularised CCA between the two probes of each of
              the 24 pairs (4 contiguous folds, 1 s gaps): held-out canonical correlations, and the
              number of components above a circular-shift null (5 shifts, 15-75 s).
  timescale   Task 3 (timescales of brain state) needs dynamics not smeared by the representation:
              median 1/e autocorrelation time of the representation's dimensions.
  interpret   What each depth-indexed latent channel tracks: correlation, over depths and time of
              held-out insertions, between each latent channel and each of the 19 bank features.

Representations: the PCA family computed here without behaviour (per-insertion fits use the whole
100 s of that insertion's signal, never behaviour; 'pooled' fits use other sessions only, with the
autoencoders' session folds), the autoencoders' held-out latents (A, B from train.py; C, D, E from
train_variants.py) and the full z-scored bank (456) as a reference.

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2 /opt/miniconda3/envs/lfp-brain-state/bin/python \
        -m lfp_autoencoder_prototype.purpose_tests --baselines      # PCA-family latents (loads raw)
    ... -m lfp_autoencoder_prototype.purpose_tests --weighted       # view-weighted pooled group PCA
    ... -m lfp_autoencoder_prototype.purpose_tests --tests          # transfer, cca, timescale, interpret

Writes ~/Downloads/lfp-brain-state/ae_results/purpose/.
"""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd

from lfp_compression_pilot.pilot_config import BEHAVIOR_LAGS, GAP_BINS
from lfp_compression_pilot.pilot_metrics import lagged, outer_folds
from .cache_bank import AE_CACHE, FEATURE_NAMES
from .train import AE_RESULTS, SEED, load_all, pooled_group_pca, session_folds

OUT = AE_RESULTS / "purpose"
BASE = OUT / "latents_baselines"
BASE2 = OUT / "latents_baselines2"
N_G = 24
# layout of each representation's rows: "cm" = channel-major (row = c*24 + g), "dm" = depth-major
# (row = g*k + c), None = not tied to depths
LAYOUT = {"spatial_avg_24": ("cm", 1), "group_pca_2x24": ("dm", 2), "group_pca_pooled_48": ("dm", 2),
          "group_pca_weighted_48": ("dm", 2), "group_wave+amp1_pooled": ("cm", 2), "ae_waveamp_48": ("cm", 2),
          "group_wave+amp1": ("cm", 2), "ae_depth_48": ("cm", 2), "ae_masked_48": ("cm", 2),
          "ae_anatomy_48": ("cm", 2), "ae_filterbank_48": ("cm", 2), "bank_456": ("dm", 19),
          "waveform_pca_48": (None, 0), "waveform_pca_pooled_48": (None, 0), "ae_global_48": (None, 0)}
ORDER = ["waveform_pca_48", "waveform_pca_pooled_48", "spatial_avg_24", "group_pca_2x24", "group_pca_pooled_48",
         "group_pca_weighted_48", "group_wave+amp1", "group_wave+amp1_pooled", "ae_waveamp_48", "ae_depth_48", "ae_global_48", "ae_masked_48", "ae_anatomy_48", "ae_filterbank_48", "bank_456"]
LAMBDAS = np.logspace(-5, 1, 13)          # ridge penalty relative to the number of training rows
CCA_SHRINK = 0.1
SHIFTS_S = (15, 30, 45, 60, 75)


def per_depth(Z, name):
    lay, k = LAYOUT[name]
    if lay == "cm":
        return Z.reshape(k, N_G, -1).transpose(1, 0, 2)
    if lay == "dm":
        return Z.reshape(N_G, k, -1)
    return None


# ----------------------------------------------------------------------------- baselines

def build_baselines():
    """PCA-family latents for all 50 insertions (no behaviour anywhere)."""
    from lfp_compression_pilot.pilot_data import load_raw
    from lfp_compression_pilot.pilot_representations import top_pcs
    from lfp_compression_search.search_methods import bin_mean
    from lfp_selected_benchmark.extract_selected import CACHE_DIR

    BASE.mkdir(parents=True, exist_ok=True)
    data = load_all()                                          # z-scored bank per insertion, as the AEs see it
    folds = session_folds(data, 5)
    covs, wave25, pcs = {}, {}, {}
    for pid in data:
        raw, _, _ = load_raw(pid, CACHE_DIR)
        x = raw.astype(np.float64)
        x -= x.mean(axis=1, keepdims=True)
        wave25[pid] = bin_mean(x).astype(np.float32)
        pcs[pid] = top_pcs(x, 48)                              # waveform PCA 48 fit on this insertion's 250 Hz signal
        covs[pid] = (x @ x.T) / (x.shape[1] * x.var())         # scale-free covariance for pooling across insertions
        print(f"[{pid[:8]}] raw loaded", flush=True)
    for pid, d in data.items():
        fold = folds[d["eid"]]
        train = [p for p, o in data.items() if folds[o["eid"]] != fold]
        evals, evecs = np.linalg.eigh(sum(covs[p] for p in train))
        Wp = evecs[:, ::-1][:, :48]
        mu, Wg = pooled_group_pca([data[p]["X"] for p in train])
        with np.load(AE_CACHE / f"{pid}.npz") as z:
            Xraw = z["X"].astype(np.float64)                   # [24, 19, T] unnormalised
        Xz = d["X"]                                            # [24, 19, T] z-scored per feature
        flat = Xz.transpose(1, 0, 2).reshape(19, -1)
        Wg2 = top_pcs(flat - flat.mean(axis=1, keepdims=True), 2)
        amp = Xz[:, 1:, :]
        fa = amp.transpose(1, 0, 2).reshape(18, -1)
        Wa = top_pcs(fa - fa.mean(axis=1, keepdims=True), 1)
        Z = {
            "waveform_pca_48": pcs[pid].T @ wave25[pid],
            "waveform_pca_pooled_48": Wp.T.astype(np.float32) @ wave25[pid],
            "spatial_avg_24": Xraw[:, 0, :].astype(np.float32),
            "group_pca_2x24": np.einsum("gft,fk->gkt", Xz, Wg2).reshape(48, -1),
            "group_pca_pooled_48": np.einsum("gft,fk->gkt", Xz - mu[None], Wg).reshape(48, -1),
            "group_wave+amp1": np.concatenate([Xraw[:, 0, :], np.einsum("gft,fk->gkt", amp, Wa).reshape(24, -1)]),
            "bank_456": Xz.reshape(24 * 19, -1),
        }
        np.savez(BASE / f"{pid}.npz", fold=fold, **{k: v.astype(np.float32) for k, v in Z.items()})
    print("baselines written:", len(data))


def build_weighted_group_pca():
    """
    The linear twin of the per-depth autoencoder's OBJECTIVE: pooled group PCA (2 per depth, loadings
    shared over depths, fit on the other sessions' folds) on features scaled by sqrt(view weight), i.e.
    the linear, context-free minimiser of the same view-weighted MSE (Baldi & Hornik 1989). Plain group
    PCA gives the waveform 1/19 of the variance; the autoencoders' loss gives it 1/3.
    """
    from .models import view_weights
    BASE2.mkdir(parents=True, exist_ok=True)
    data = load_all()
    folds = session_folds(data, 5)
    sw = np.sqrt(view_weights().numpy().reshape(1, 19, 1))
    for pid, d in data.items():
        train = [data[p]["X"] * sw for p, o in data.items() if folds[o["eid"]] != folds[d["eid"]]]
        mu, W = pooled_group_pca(train)
        Z = np.einsum("gft,fk->gkt", d["X"] * sw - mu[None], W).reshape(48, -1)
        # linear twin of option F: stored group waveform + 1 pooled amplitude component per depth
        tr_amp = [data[p]["X"][:, 1:] for p, o in data.items() if folds[o["eid"]] != folds[d["eid"]]]
        mu_a, Wa = pooled_group_pca(tr_amp, k=1)
        amp = np.einsum("gft,fk->gkt", d["X"][:, 1:] - mu_a[None], Wa).reshape(24, -1)
        Zf = np.concatenate([d["X"][:, 0], amp], axis=0)
        np.savez(BASE2 / f"{pid}.npz", fold=folds[d["eid"]], group_pca_weighted_48=Z.astype(np.float32),
                 **{"group_wave+amp1_pooled": Zf.astype(np.float32)})
    print("weighted group PCA written:", len(data))


def load_reps(pid, ae_dir=AE_RESULTS / "latents"):
    """{name: [d, T]} for every representation available for this insertion."""
    reps = {}
    sources = [BASE / f"{pid}.npz", BASE2 / f"{pid}.npz", Path(ae_dir) / f"{pid}.npz"]
    sources += [AE_RESULTS / "variants" / f"latents_{m}" / f"{pid}.npz" for m in ("C", "D", "E", "F")]
    for f in sources:
        if f.exists():
            with np.load(f) as z:
                for k in z.files:
                    if k != "fold" and k not in reps:
                        reps[k] = z[k].astype(np.float64)
    return reps


def zs(Z):
    mu = Z.mean(axis=1, keepdims=True)
    sd = Z.std(axis=1, keepdims=True)
    return (Z - mu) / np.where(sd > 1e-8, sd, 1.0)


def pooled_readout(Z, name):
    D = per_depth(Z, name)
    return None if D is None else np.concatenate([D.mean(axis=0), D.std(axis=0)], axis=0)


# ----------------------------------------------------------------------------- transfer

def _stats(X, Y):
    """Per behaviour: X'X and X'y over rows where that behaviour is finite."""
    out = []
    for j in range(Y.shape[1]):
        ok = np.isfinite(Y[:, j])
        Xo = X[ok]
        out.append((Xo.T @ Xo, Xo.T @ Y[ok, j], int(ok.sum())))
    return out


def _fit(stats_list, j, lam_rel):
    S = sum(s[j][0] for s in stats_list)
    b = sum(s[j][1] for s in stats_list)
    n = sum(s[j][2] for s in stats_list)
    ev, V = np.linalg.eigh(S)
    Vb = V.T @ b
    return [V @ (Vb / (ev + lam * n)) for lam in lam_rel]


def _r(y, p):
    ok = np.isfinite(y)
    if ok.sum() < 50 or np.std(y[ok]) == 0 or np.std(p[ok]) == 0:
        return np.nan
    return float(np.corrcoef(y[ok], p[ok])[0, 1])


def transfer_test(pids, meta, reps_by_pid, beh_names, shard=(0, 1)):
    rows = []
    names = sorted({k for r in reps_by_pid.values() for k in r} - {"bank_456"})
    names = [n for i, n in enumerate(names) if i % shard[1] == shard[0]]
    eids = sorted({meta[p]["eid"] for p in pids})
    for name in names:
        for readout in ("full", "pooled"):
            X, Y, St = {}, {}, {}
            for p in pids:
                if name not in reps_by_pid[p]:
                    continue
                Z = reps_by_pid[p][name]
                if readout == "pooled":
                    Z = pooled_readout(Z, name)
                    if Z is None:
                        break
                X[p] = lagged(zs(Z), BEHAVIOR_LAGS).T
                B = meta[p]["B"][: X[p].shape[0]].astype(np.float64)
                Y[p] = (B - np.nanmean(B, axis=0)) / np.where(np.nanstd(B, axis=0) > 0, np.nanstd(B, axis=0), 1.0)
                St[p] = _stats(X[p], Y[p])
            if len(X) < len(pids):
                continue
            for s in eids:
                test = [p for p in pids if meta[p]["eid"] == s]
                train = [p for p in pids if meta[p]["eid"] != s]
                tr_eids = sorted({meta[p]["eid"] for p in train})
                groups = {e: i % 5 for i, e in enumerate(np.random.default_rng(SEED).permutation(tr_eids))}
                for j, bn in enumerate(beh_names):
                    score = np.zeros(len(LAMBDAS))
                    for g in range(5):
                        fit_on = [St[p] for p in train if groups[meta[p]["eid"]] != g]
                        val = [p for p in train if groups[meta[p]["eid"]] == g]
                        ws = _fit(fit_on, j, LAMBDAS)
                        for i, w in enumerate(ws):
                            score[i] += np.nanmean([_r(Y[p][:, j], X[p] @ w) for p in val])
                    lam = LAMBDAS[int(np.nanargmax(score))]
                    w = _fit([St[p] for p in train], j, [lam])[0]
                    for p in test:
                        rows.append({"method": name, "readout": readout, "pid": p, "eid": s, "behaviour": bn,
                                     "r": _r(Y[p][:, j], X[p] @ w), "lambda_rel": lam})
            print(f"transfer {name} {readout} done", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- CCA between simultaneous probes

def _cov_shrunk(A):
    C = A.T @ A / len(A)
    return (1 - CCA_SHRINK) * C + CCA_SHRINK * np.trace(C) / C.shape[0] * np.eye(C.shape[0])


def _inv_sqrt(C):
    ev, V = np.linalg.eigh(C)
    return V @ np.diag(1 / np.sqrt(np.maximum(ev, 1e-10))) @ V.T


def highpass(Z, hz=1.0):
    """Zero-phase 1 Hz high-pass at 25 Hz: removes slow drift and slow state changes shared by both probes."""
    import scipy.signal
    sos = scipy.signal.butter(2, hz, btype="highpass", fs=25.0, output="sos")
    return scipy.signal.sosfiltfilt(sos, Z, axis=1)


def cca_heldout(Z1, Z2, k=10):
    """Held-out canonical correlations (mean over 4 contiguous folds) of the top k training pairs."""
    T = min(Z1.shape[1], Z2.shape[1])
    A, B = Z1[:, :T].T, Z2[:, :T].T
    out = []
    for train, test in outer_folds(T):
        ma, sa = A[train].mean(0), A[train].std(0) + 1e-8
        mb, sb = B[train].mean(0), B[train].std(0) + 1e-8
        a, b = (A[train] - ma) / sa, (B[train] - mb) / sb
        Wa, Wb = _inv_sqrt(_cov_shrunk(a)), _inv_sqrt(_cov_shrunk(b))
        U, s, Vt = np.linalg.svd(Wa @ (a.T @ b / len(a)) @ Wb)
        kk = min(k, len(s))
        pa, pb = ((A[test] - ma) / sa) @ Wa @ U[:, :kk], ((B[test] - mb) / sb) @ Wb @ Vt.T[:, :kk]
        out.append([np.corrcoef(pa[:, i], pb[:, i])[0, 1] for i in range(kk)])
    return np.nanmean(np.array(out), axis=0)


def cca_test(pairs, reps_by_pid):
    rows = []
    for p1, p2, eid in pairs:
        r1, r2 = reps_by_pid[p1], reps_by_pid[p2]
        for name in sorted(set(r1) & set(r2)):
            obs = cca_heldout(r1[name], r2[name])
            null = np.max([cca_heldout(r1[name], np.roll(r2[name], int(s * 25), axis=1)) for s in SHIFTS_S], axis=0)
            h1, h2 = highpass(r1[name]), highpass(r2[name])
            hp = cca_heldout(h1, h2)
            hp_null = cca_heldout(h1, np.roll(h2, 30 * 25, axis=1))
            rows.append({"eid": eid, "p1": p1, "p2": p2, "method": name, "cc1": obs[0], "cc_top5": float(np.mean(obs[:5])),
                         "null_cc1": null[0], "null_top5": float(np.mean(null[:5])),
                         "n_shared_dims": int(np.sum(np.cumprod(obs > null))),
                         "hp1hz_top5": float(np.mean(hp[:5])), "hp1hz_null_top5": float(np.mean(hp_null[:5])),
                         **{f"cc{i + 1}": v for i, v in enumerate(obs)}})
        print(f"cca {eid[:8]} done", flush=True)
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- timescale and interpretation

def acf_time(Z, max_lag=250):
    """Median over dimensions of the first lag (s) where the autocorrelation drops below 1/e."""
    Z = zs(Z)
    T = Z.shape[1]
    f = np.fft.rfft(Z, n=2 * T, axis=1)
    acf = np.fft.irfft(np.abs(f) ** 2, axis=1)[:, :max_lag]
    acf /= np.maximum(acf[:, :1], 1e-12)
    below = acf < 1 / np.e
    lag = np.where(below.any(axis=1), below.argmax(axis=1), max_lag)
    return float(np.median(lag) / 25.0)


def interpret(pid, reps, Xz):
    rows = []
    feats = Xz.transpose(1, 0, 2).reshape(19, -1)                  # [19, 24*T] depth-major within feature
    for name in reps:
        if LAYOUT.get(name, (None, 0))[0] is None or name in ("bank_456", "spatial_avg_24", "group_wave+amp1"):
            continue
        D = per_depth(reps[name], name)                             # [24, k, T]
        for c in range(D.shape[1]):
            lat = D[:, c, :].reshape(-1)
            r = [np.corrcoef(lat, feats[f])[0, 1] for f in range(19)]
            rows.append({"pid": pid, "method": name, "channel": c, **dict(zip(FEATURE_NAMES, r))})
    return rows


def run_tests(ae_dir, out, only=None, shard=(0, 1)):
    data = load_all()
    pids = sorted(data)
    meta = {}
    for p in pids:
        with np.load(AE_CACHE / f"{p}.npz") as z:
            meta[p] = {"eid": str(z["eid"]), "B": z["behavior"].astype(np.float64)}
            beh_names = [str(b) for b in z["behavior_names"]]
    reps = {p: load_reps(p, ae_dir) for p in pids}
    have = pd.Series({n: sum(n in reps[p] for p in pids) for n in ORDER})
    print("insertions per representation:\n" + have.to_string(), flush=True)
    out.mkdir(parents=True, exist_ok=True)

    ts, it = [], []
    for p in (pids if only in (None, "basic") else []):
        for n, Z in reps[p].items():
            ts.append({"pid": p, "method": n, "acf_1e_s": acf_time(Z)})
        it += interpret(p, reps[p], data[p]["X"])
    if only in (None, "basic"):
        pd.DataFrame(ts).to_csv(out / "timescale.csv", index=False)
        pd.DataFrame(it).to_csv(out / "interpret.csv", index=False)

    by_eid = {}
    for p in pids:
        by_eid.setdefault(meta[p]["eid"], []).append(p)
    pairs = [(v[0], v[1], e) for e, v in sorted(by_eid.items()) if len(v) == 2]
    if only in (None, "cca"):
        cca_test(pairs, reps).to_csv(out / "cca.csv", index=False)
    if only in (None, "transfer"):
        transfer_test(pids, meta, reps, beh_names, shard).to_csv(
            out / ("transfer.csv" if shard == (0, 1) else f"transfer_{shard[0]}.csv"), index=False)
    json.dump({"n_insertions": len(pids), "n_pairs": len(pairs), "per_method": have.to_dict()},
              open(out / "purpose_meta.json", "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baselines", action="store_true")
    ap.add_argument("--weighted", action="store_true")
    ap.add_argument("--tests", action="store_true")
    ap.add_argument("--ae-dir", default=str(AE_RESULTS / "latents"), help="where A/B latents are read from")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--only", choices=["basic", "cca", "transfer"], default=None)
    ap.add_argument("--shard", default="0/1", help="transfer only: i/n splits the methods across processes")
    a = ap.parse_args()
    if a.baselines:
        build_baselines()
    if a.weighted:
        build_weighted_group_pca()
    if a.tests:
        run_tests(Path(a.ae_dir), Path(a.out), a.only, tuple(map(int, a.shard.split("/"))))


if __name__ == "__main__":
    main()
