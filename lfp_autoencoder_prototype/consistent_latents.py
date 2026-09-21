"""
Fold-consistent latents, so tests that pool insertions compare like with like.

purpose_tests.transfer and state_tests 'pooled' train a readout or HMM on many insertions and test
on a held-out one. With the saved held-out latents, every insertion was encoded by its OWN fold's
model, so training and test latents came from different encoders. That asks whether separately
trained encoders agree (an identifiability question), not whether one encoder gives shared
coordinates (R4). Here fold k's model, or linear fit, encodes ALL 50 insertions. The pooled tests then
use only fold k's held-out sessions as test data, and the readout/HMM never sees a test session.

Available for the models with saved weights (C, D, E, F) and the linear fits (pooled group PCA, W,
WF). A and B cannot be re-encoded because train.py saves no weights.

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2 /opt/miniconda3/envs/lfp-brain-state/bin/python \
        -m lfp_autoencoder_prototype.consistent_latents [--encode] [--transfer]

Writes ae_results/purpose/latents_by_fold/fold<k>/<pid>.npz and ae_results/purpose/transfer_consistent.csv.
"""
import argparse
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

from lfp_compression_pilot.pilot_config import BEHAVIOR_LAGS
from lfp_compression_pilot.pilot_metrics import lagged
from .cache_bank import AE_CACHE
from .models import view_weights
from .purpose_tests import LAMBDAS, OUT, _fit, _r, _stats, pooled_readout, zs
from .train import SEED, load_all, pooled_group_pca, session_folds
from .train_variants import OUT as VOUT, full_batch, load_extra
from .variants import AnatomyAE, FilterbankAE, MaskedDepthAE, WaveAmpAE

BYFOLD = OUT / "latents_by_fold"
NAMES = ["group_pca_pooled_48", "group_pca_weighted_48", "group_wave+amp1_pooled",
         "ae_masked_48", "ae_anatomy_48", "ae_filterbank_48", "ae_waveamp_48"]


def encode():
    data = load_all()
    load_extra(data, True)
    folds = session_folds(data, 5)
    sw = np.sqrt(view_weights().numpy().reshape(1, 19, 1))
    for k in range(5):
        (BYFOLD / f"fold{k}").mkdir(parents=True, exist_ok=True)
        tr = [d["X"] for d in data.values() if folds[d["eid"]] != k]
        mu, Wg = pooled_group_pca(tr)
        muw, Ww = pooled_group_pca([X * sw for X in tr])
        mua, Wa = pooled_group_pca([X[:, 1:] for X in tr], k=1)
        models = []
        for cls in (MaskedDepthAE, AnatomyAE, FilterbankAE, WaveAmpAE):
            m = cls()
            m.load_state_dict(torch.load(VOUT / "models" / f"{cls.name}_fold{k}.pt", map_location="cpu"))
            models.append(m.eval())
        for pid, d in data.items():
            X = d["X"]
            Z = {"group_pca_pooled_48": np.einsum("gft,fk->gkt", X - mu[None], Wg).reshape(48, -1),
                 "group_pca_weighted_48": np.einsum("gft,fk->gkt", X * sw - muw[None], Ww).reshape(48, -1),
                 "group_wave+amp1_pooled": np.concatenate([X[:, 0], np.einsum("gft,fk->gkt", X[:, 1:] - mua[None], Wa).reshape(24, -1)])}
            for m in models:
                b = full_batch(d, isinstance(m, FilterbankAE))
                with torch.no_grad():
                    z = m.encode(b)
                    Z[m.name] = (m.flatten(z, b["x"]) if isinstance(m, WaveAmpAE) else m.flatten(z))[0].numpy()
            np.savez(BYFOLD / f"fold{k}" / f"{pid}.npz", fold=folds[d["eid"]], held_out=folds[d["eid"]] == k,
                     **{n: v.astype(np.float32) for n, v in Z.items()})
        print(f"fold {k} encoded", flush=True)


def transfer(only_fold=None):
    data = load_all()
    folds = session_folds(data, 5)
    pids = sorted(data)
    meta = {}
    for p in pids:
        with np.load(AE_CACHE / f"{p}.npz") as z:
            B = z["behavior"].astype(np.float64)
            beh_names = [str(b) for b in z["behavior_names"]]
        sd = np.nanstd(B, axis=0)
        meta[p] = {"eid": data[p]["eid"], "Y": (B - np.nanmean(B, axis=0)) / np.where(sd > 0, sd, 1.0)}
    rows = []
    for k in (range(5) if only_fold is None else [only_fold]):
        lat = {p: dict(np.load(BYFOLD / f"fold{k}" / f"{p}.npz")) for p in pids}
        test_eids = sorted({meta[p]["eid"] for p in pids if folds[meta[p]["eid"]] == k})
        for name in NAMES:
            for readout in ("full", "pooled"):
                X, St = {}, {}
                for p in pids:
                    Z = lat[p][name].astype(np.float64)
                    if readout == "pooled":
                        Z = pooled_readout(Z, name)
                    X[p] = lagged(zs(Z), BEHAVIOR_LAGS).T
                    St[p] = _stats(X[p], meta[p]["Y"][: X[p].shape[0]])
                for s in test_eids:
                    test = [p for p in pids if meta[p]["eid"] == s]
                    train = [p for p in pids if meta[p]["eid"] != s]
                    tr_eids = sorted({meta[p]["eid"] for p in train})
                    groups = {e: i % 5 for i, e in enumerate(np.random.default_rng(SEED).permutation(tr_eids))}
                    for j, bn in enumerate(beh_names):
                        score = np.zeros(len(LAMBDAS))
                        for g_ in range(5):
                            fit_on = [St[p] for p in train if groups[meta[p]["eid"]] != g_]
                            val = [p for p in train if groups[meta[p]["eid"]] == g_]
                            for i, w in enumerate(_fit(fit_on, j, LAMBDAS)):
                                score[i] += np.nanmean([_r(meta[p]["Y"][: len(X[p]), j], X[p] @ w) for p in val])
                        w = _fit([St[p] for p in train], j, [LAMBDAS[int(np.nanargmax(score))]])[0]
                        for p in test:
                            rows.append({"method": name, "readout": readout, "pid": p, "eid": s, "fold": k,
                                         "behaviour": bn, "r": _r(meta[p]["Y"][: len(X[p]), j], X[p] @ w)})
            print(f"fold {k} {name} done", flush=True)
    pd.DataFrame(rows).to_csv(OUT / ("transfer_consistent.csv" if only_fold is None else f"transfer_consistent_{only_fold}.csv"), index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--encode", action="store_true")
    ap.add_argument("--transfer", action="store_true")
    ap.add_argument("--fold", type=int, default=None)
    a = ap.parse_args()
    if a.encode:
        encode()
    if a.transfer:
        transfer(a.fold)


if __name__ == "__main__":
    main()
