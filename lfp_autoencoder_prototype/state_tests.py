"""
Do the representations support the brain-state analyses the project is building toward?

Reference: the Akella et al. (2025) state variable, rebuilt from the full-resolution features:
theta (rms 1 s), beta (rms 200 ms), low gamma (PSD 30-50 Hz) and high gamma (PSD 50-100 Hz) log
power, averaged over the top, middle and bottom thirds of the probe (12 numbers per 40 ms), with a
3-state Gaussian HMM (full covariance). A second, data-driven reference uses the whole 456-number
bank (PCA to 12, same HMM).

Per insertion ('within'):  each representation -> z-score -> PCA to 12 -> the same 3-state HMM.
  agreement   normalised mutual information and adjusted Rand index with the reference states
  behaviour   eta^2: fraction of each behaviour's variance explained by the state labels
  dwell       mean state duration (s)
Across animals ('pooled', leave-session-out with the autoencoders' folds): PCA and HMM fit on the
training sessions' concatenated sequences, states decoded on the held-out session; agreement with the
pooled reference HMM fit the same way. States only mean the same thing across animals if the
representation's coordinates do.

Caveat, stated in the report: the reference is built from bank features, so bank-derived
representations have a home advantage over waveform-only ones.

Runs in the `lfp` env (has hmmlearn; lfp-brain-state does not). Only reads .npz files:
    /opt/miniconda3/envs/lfp/bin/python ~/Downloads/lfp-brain-state/lfp_based_brain_state/lfp_autoencoder_prototype/state_tests.py
"""
import os
import sys
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

ROOT = Path(os.path.expanduser("~/Downloads/lfp-brain-state"))
AE_CACHE, AE_RESULTS = ROOT / "ae_cache", ROOT / "ae_results"
OUT = AE_RESULTS / "purpose"
AKELLA = [2, 5, 16, 17]            # rms_theta_1000ms, rms_beta_200ms, psd_30-50Hz, psd_50-100Hz
K, D_HMM, SEEDS = 3, 12, (0, 1, 2)


def zs(A, axis=0):
    sd = A.std(axis=axis, keepdims=True)
    return (A - A.mean(axis=axis, keepdims=True)) / np.where(sd > 1e-8, sd, 1.0)


def load(pid):
    with np.load(AE_CACHE / f"{pid}.npz") as z:
        X = z["X"].astype(np.float64)                                   # [24, 19, T]
        B = z["behavior"].astype(np.float64)
        eid = str(z["eid"])
    Xz = (X - X.mean(axis=(0, 2), keepdims=True)) / X.std(axis=(0, 2), keepdims=True)
    thirds = np.array_split(np.arange(24), 3)
    ak = np.concatenate([Xz[t][:, AKELLA].mean(axis=0) for t in thirds], axis=0).T      # [T, 12]
    reps, fold = {}, None
    files = [AE_RESULTS / "purpose" / "latents_baselines" / f"{pid}.npz", AE_RESULTS / "purpose" / "latents_baselines2" / f"{pid}.npz",
             AE_RESULTS / "latents" / f"{pid}.npz"]
    files += [AE_RESULTS / "variants" / f"latents_{m}" / f"{pid}.npz" for m in ("C", "D", "E", "F")]
    for f in files:
        if f.exists():
            with np.load(f) as z:
                if fold is None and "fold" in z.files:
                    fold = int(z["fold"])
                for k in z.files:
                    if k != "fold" and k not in reps:
                        reps[k] = z[k].astype(np.float64).T                # [T, d]
    return {"eid": eid, "B": B, "akella": ak, "reps": reps, "fold": fold}


def pca(A, d, fit_rows=None):
    F = A if fit_rows is None else fit_rows
    mu = F.mean(axis=0)
    ev, V = np.linalg.eigh(np.cov((F - mu).T))
    W = V[:, ::-1][:, :d]
    return lambda M: (M - mu) @ W


def fit_hmm(seqs):
    X = np.concatenate(seqs)
    best, best_ll = None, -np.inf
    for s in SEEDS:
        m = GaussianHMM(n_components=K, covariance_type="full", n_iter=100, tol=1e-3, random_state=s, min_covar=1e-3)
        try:
            m.fit(X, lengths=[len(q) for q in seqs])
            ll = m.score(X, lengths=[len(q) for q in seqs])
        except Exception:
            continue
        if np.isfinite(ll) and ll > best_ll:
            best, best_ll = m, ll
    return best


def eta2(labels, B):
    out = []
    for j in range(B.shape[1]):
        y, ok = B[:, j], np.isfinite(B[:, j])
        if ok.sum() < 100 or np.std(y[ok]) == 0:
            continue
        y, l = y[ok], labels[ok]
        ss_b = sum((l == k).sum() * (y[l == k].mean() - y.mean()) ** 2 for k in np.unique(l))
        out.append(ss_b / ((y - y.mean()) ** 2).sum())
    return float(np.mean(out)) if out else np.nan


def dwell(labels):
    change = np.flatnonzero(np.diff(labels)) + 1
    runs = np.diff(np.concatenate([[0], change, [len(labels)]]))
    return float(runs.mean() / 25.0)


def features(d, name):
    return d["akella"] if name == "akella_ref" else d["reps"][name]


def pooled_consistent(k):
    """Pooled HMM with fold-consistent latents: fold k's model encodes every insertion (consistent_latents.py)."""
    pids = sorted(p.stem for p in AE_CACHE.glob("*.npz"))
    data = {p: load(p) for p in pids}
    lat = {p: dict(np.load(OUT / "latents_by_fold" / f"fold{k}" / f"{p}.npz")) for p in pids}
    tr = [p for p in pids if not bool(lat[p]["held_out"])]
    te = [p for p in pids if bool(lat[p]["held_out"])]
    names = ["akella_ref"] + [n for n in lat[pids[0]] if n not in ("fold", "held_out")]
    decoded = {}
    for name in names:
        seqs = {p: zs(data[p]["akella"] if name == "akella_ref" else lat[p][name].astype(np.float64).T) for p in pids}
        proj = pca(None, min(D_HMM, seqs[pids[0]].shape[1]), fit_rows=np.concatenate([seqs[p] for p in tr]))
        seqs = {p: proj(a) for p, a in seqs.items()}
        m = fit_hmm([seqs[p] for p in tr])
        if m is not None:
            decoded[name] = {p: m.predict(seqs[p]) for p in te}
    rows = []
    for name, st in decoded.items():
        for p in te:
            rows.append({"pid": p, "eid": data[p]["eid"], "fold": k, "method": name, "eta2": eta2(st[p], data[p]["B"]),
                         "dwell_s": dwell(st[p]), "nmi_akella_ref": normalized_mutual_info_score(decoded["akella_ref"][p], st[p])})
    pd.DataFrame(rows).to_csv(OUT / f"states_pooled_consistent_{k}.csv", index=False)
    print(f"pooled consistent fold {k} done", flush=True)


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "all"               # all | within:<i>/<n> | pooled:<fold> | pooledc:<fold>
    if part.startswith("pooledc:"):
        return pooled_consistent(int(part.split(":")[1]))
    pids = sorted(p.stem for p in AE_CACHE.glob("*.npz"))
    data = {p: load(p) for p in pids}
    names = sorted({k for d in data.values() for k in d["reps"]})
    counts = {n: sum(n in d["reps"] for d in data.values()) for n in names}
    print("insertions per representation:", counts, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []                                                           # within insertion
    shard = (0, 1) if not part.startswith("within:") else tuple(map(int, part.split(":")[1].split("/")))
    for j, (p, d) in enumerate(data.items()):
        if part.startswith("pooled") or j % shard[1] != shard[0]:
            continue
        ref_states = {}
        for ref in ("akella_ref", "bank_456"):
            A = zs(features(d, ref))
            A = zs(pca(A, D_HMM)(A)) if A.shape[1] > D_HMM else A
            m = fit_hmm([A])
            ref_states[ref] = m.predict(A) if m is not None else None
        for name in ["akella_ref"] + names:
            if name != "akella_ref" and name not in d["reps"]:
                continue
            A = zs(features(d, name))
            A = zs(pca(A, min(D_HMM, A.shape[1]))(A))
            m = fit_hmm([A])
            if m is None:
                continue
            s = m.predict(A)
            row = {"pid": p, "eid": d["eid"], "method": name, "eta2": eta2(s, d["B"]), "dwell_s": dwell(s)}
            for ref, rs in ref_states.items():
                if rs is not None:
                    row[f"nmi_{ref}"] = normalized_mutual_info_score(rs, s)
                    row[f"ari_{ref}"] = adjusted_rand_score(rs, s)
            rows.append(row)
        print(f"within {p[:8]} done", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(OUT / ("states_within.csv" if part == "all" else f"states_within_{shard[0]}.csv"), index=False)
    if part.startswith("within"):
        return

    rows = []                                                           # pooled across animals
    folds = sorted({d["fold"] for d in data.values()})
    for fold in (folds if part == "all" else [int(part.split(":")[1])]):
        tr = [p for p in pids if data[p]["fold"] != fold]
        te = [p for p in pids if data[p]["fold"] == fold]
        decoded = {}
        for name in ["akella_ref"] + names:
            if any(name != "akella_ref" and name not in data[p]["reps"] for p in pids):
                continue
            seqs = {p: zs(features(data[p], name)) for p in pids}
            proj = pca(None, min(D_HMM, seqs[pids[0]].shape[1]), fit_rows=np.concatenate([seqs[p] for p in tr]))
            seqs = {p: proj(a) for p, a in seqs.items()}
            m = fit_hmm([seqs[p] for p in tr])
            if m is None:
                continue
            decoded[name] = {p: m.predict(seqs[p]) for p in te}
        for name, st in decoded.items():
            for p in te:
                row = {"pid": p, "eid": data[p]["eid"], "fold": fold, "method": name,
                       "eta2": eta2(st[p], data[p]["B"]), "dwell_s": dwell(st[p])}
                if "akella_ref" in decoded:
                    row["nmi_akella_ref"] = normalized_mutual_info_score(decoded["akella_ref"][p], st[p])
                rows.append(row)
        print(f"pooled fold {fold} done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / ("states_pooled.csv" if part == "all" else f"states_pooled_{part.split(':')[1]}.csv"), index=False)


if __name__ == "__main__":
    sys.exit(main())
