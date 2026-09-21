"""
Leave-session-out training of the prototype autoencoders and a pooled group-PCA baseline.

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    KMP_DUPLICATE_LIB_OK=TRUE /opt/miniconda3/envs/lfp-brain-state/bin/python \
        -m lfp_autoencoder_prototype.train [--folds 5] [--epochs 40] [--mask 0.0]

Protocol
  - Sessions (not insertions) are split into K folds, so simultaneous probes never sit
    on both sides. For each fold, models train on the other sessions (2 of them held out
    for early stopping) and encode every insertion of the fold's sessions, which they
    have never seen.
  - Each insertion's features are z-scored with its own mean and SD per feature (pooled
    over depths and time, so relative depth amplitude is kept). This uses the whole
    100 s of that insertion's features but never behaviour.
  - Training: random 64-bin (2.56 s) windows, AdamW, view-weighted MSE; behaviour is never used.
  - Pooled group PCA: 2 components of the 19 features with loadings shared over depths,
    fit on the same training sessions, so it is the linear, context-free counterpart of
    DepthLatentAE at the same 48 numbers.

Writes ~/Downloads/lfp-brain-state/ae_results/latents/<pid>.npz ({method: [48, T]}) and training_log.csv.
"""
import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import numpy as np
import pandas as pd
import torch

from .cache_bank import AE_CACHE
from .models import DepthLatentAE, GlobalLatentAE, n_params, view_weights

AE_RESULTS = Path(os.path.expanduser("~/Downloads/lfp-brain-state/ae_results"))
SEED = 0
WINDOW, BATCH = 64, 32


def load_all():
    data = {}
    for f in sorted(AE_CACHE.glob("*.npz")):
        with np.load(f) as z:
            X = z["X"].astype(np.float32)
            mu = X.mean(axis=(0, 2), keepdims=True)
            sd = X.std(axis=(0, 2), keepdims=True)
            data[f.stem] = {"X": (X - mu) / np.where(sd > 1e-6, sd, 1.0), "eid": str(z["eid"])}
    return data


def session_folds(data, k):
    eids = sorted({d["eid"] for d in data.values()})
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(eids))
    return {eids[i]: int(j % k) for j, i in enumerate(order)}


def batches(Xs, n_steps, rng):
    for _ in range(n_steps):
        xb = []
        for _ in range(BATCH):
            X = Xs[rng.integers(len(Xs))]
            t0 = rng.integers(0, X.shape[2] - WINDOW)
            xb.append(X[..., t0:t0 + WINDOW])
        xb = np.stack(xb).transpose(0, 2, 1, 3)                     # [B, 19, 24, T]
        yield torch.from_numpy(np.ascontiguousarray(xb))


def val_loss(model, Xs, w):
    model.eval()
    with torch.no_grad():
        tot = 0.0
        for X in Xs:
            x = torch.from_numpy(np.ascontiguousarray(X.transpose(1, 0, 2)))[None]
            tot += float((w * (model(x) - x) ** 2).mean())
    model.train()
    return tot / len(Xs)


def train_model(cls, train_X, val_X, epochs, mask, rng, log, fold, patience=4):
    torch.manual_seed(SEED + fold)
    model = cls()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    w = view_weights()
    steps = max(1, sum(X.shape[2] for X in train_X) // (WINDOW * BATCH))
    best, best_state, wait = np.inf, None, 0
    for ep in range(epochs):
        tic = time.time()
        tr = 0.0
        for xb in batches(train_X, steps, rng):
            inp = xb
            if mask > 0:                                             # option C: hide whole depth groups
                keep = (torch.rand(xb.shape[0], 1, xb.shape[2], 1) > mask).float()
                inp = xb * keep
            loss = (w * (model(inp) - xb) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            tr += float(loss)
        v = val_loss(model, val_X, w)
        log.append({"fold": fold, "model": cls.name, "epoch": ep, "train_loss": tr / steps, "val_loss": v, "epoch_s": time.time() - tic})
        if v < best - 1e-4:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model


def pooled_group_pca(train_X, k=2):
    flat = np.concatenate([X.transpose(1, 0, 2).reshape(X.shape[1], -1) for X in train_X], axis=1)   # [19, n]
    mu = flat.mean(axis=1, keepdims=True)
    evals, evecs = np.linalg.eigh(np.cov(flat - mu))
    return mu, evecs[:, ::-1][:, :k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--mask", type=float, default=0.0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--patience", type=int, default=4, help="early-stop patience; raise it for long runs")
    ap.add_argument("--only-fold", type=int, default=None, help="train one fold (run folds in parallel processes)")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    data = load_all()
    folds = session_folds(data, a.folds)
    out = AE_RESULTS / "latents"
    out.mkdir(parents=True, exist_ok=True)
    log, meta = [], {"folds": a.folds, "epochs": a.epochs, "mask": a.mask, "n_insertions": len(data),
                     "n_sessions": len(folds), "params": {}}
    for fold in (range(a.folds) if a.only_fold is None else [a.only_fold]):
        test = [p for p, d in data.items() if folds[d["eid"]] == fold]
        train_eids = sorted({d["eid"] for d in data.values() if folds[d["eid"]] != fold})
        val_eids = set(np.random.default_rng(SEED + fold).choice(train_eids, size=2, replace=False))
        tr_X = [d["X"] for d in data.values() if folds[d["eid"]] != fold and d["eid"] not in val_eids]
        va_X = [d["X"] for d in data.values() if d["eid"] in val_eids]
        tic = time.time()
        rng = np.random.default_rng(SEED + fold)
        models = [train_model(cls, tr_X, va_X, a.epochs, a.mask, rng, log, fold, a.patience)
                  for cls in (DepthLatentAE, GlobalLatentAE)]
        mu, W = pooled_group_pca(tr_X + va_X)
        for m in models:
            meta["params"][m.name] = n_params(m)
        for pid in test:
            X = data[pid]["X"]
            Z = {}
            with torch.no_grad():
                x = torch.from_numpy(np.ascontiguousarray(X.transpose(1, 0, 2)))[None]
                for m in models:
                    Z[m.name] = m.flatten(m.encode(x))[0].numpy().astype(np.float32)
            Z["group_pca_pooled_48"] = np.einsum("gft,fk->gkt", X - mu[None], W).reshape(-1, X.shape[2]).astype(np.float32)
            np.savez(out / f"{pid}.npz", fold=fold, **Z)
        pd.DataFrame(log).to_csv(AE_RESULTS / f"training_log_fold{fold}.csv", index=False)
        print(f"fold {fold}: {len(test)} held-out insertions, {len(tr_X)} train, {len(va_X)} val, {time.time() - tic:.0f} s", flush=True)
    meta["folds_by_session"] = folds
    json.dump(meta, open(AE_RESULTS / f"train_meta{'' if a.only_fold is None else f'_fold{a.only_fold}'}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
