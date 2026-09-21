"""
Leave-session-out training of design options C, D and E with exactly the protocol of train.py
(same session folds, same 2 validation sessions per fold, same windows, batch, optimiser, loss,
epochs and early stopping), plus held-out tests that the benchmark cannot run:

  reconstruction   fraction of the z-scored feature bank each model's decoder reconstructs on
                   insertions from sessions it never saw, per view (waveform / RMS / PSD); pooled
                   group PCA (2 per depth, fit on the same sessions) is the linear reference
  imputation (C)   hide depth groups of a held-out insertion (a 3-group block = 0.96 mm, or a
                   random 25 % of groups), reconstruct them, and compare with linear interpolation
                   in depth from the nearest visible groups
  cost             parameters, training wall time, time to encode one 100 s insertion

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1 /opt/miniconda3/envs/lfp-brain-state/bin/python \
        -m lfp_autoencoder_prototype.train_variants --only-fold 0 [--models C D E] [--epochs 50 --patience 6]

Writes ~/Downloads/lfp-brain-state/ae_results/variants/: latents_<models>/<pid>.npz, models/<name>_fold<k>.pt,
training_log_fold<k>.csv, heldout_fold<k>.csv, imputation_fold<k>.csv, cost_fold<k>.csv.
"""
import argparse
import os
import time

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import torch

from .cache_bank import AE_CACHE
from .cache_raw import AE_CACHE_RAW
from .models import n_params, view_weights
from .train import AE_RESULTS, BATCH, SEED, WINDOW, load_all, pooled_group_pca, session_folds
from .variants import AnatomyAE, FilterbankAE, MaskedDepthAE, WaveAmpAE, raw_window, region_ids

OUT = AE_RESULTS / "variants"
MODELS = {"C": MaskedDepthAE, "D": AnatomyAE, "E": FilterbankAE, "F": WaveAmpAE}
VIEWS = {"waveform": [0], "rms": list(range(1, 10)), "psd": list(range(10, 19))}


def load_extra(data, need_raw):
    for pid, d in data.items():
        with np.load(AE_CACHE / f"{pid}.npz") as z:
            d["reg"] = np.array(region_ids(z["regions"]), dtype=np.int64)
        if need_raw:
            with np.load(AE_CACHE_RAW / f"{pid}.npz") as z:
                d["raw"] = z["raw"]


def full_batch(d, need_raw):
    b = {"x": torch.from_numpy(np.ascontiguousarray(d["X"].transpose(1, 0, 2)))[None],
         "reg": torch.from_numpy(d["reg"])[None]}
    if need_raw:
        b["raw"] = torch.from_numpy(np.ascontiguousarray(raw_window(d["raw"], 0, d["X"].shape[2])))[None]
    return b


def batches(ds, n_steps, rng, need_raw):
    """Same window sampling as train.batches, plus region ids and the raw crop with 0.52 s context."""
    for _ in range(n_steps):
        xb, rb, wb = [], [], []
        for _ in range(BATCH):
            d = ds[rng.integers(len(ds))]
            t0 = rng.integers(0, d["X"].shape[2] - WINDOW)
            xb.append(d["X"][..., t0:t0 + WINDOW])
            rb.append(d["reg"])
            if need_raw:
                wb.append(raw_window(d["raw"], t0, WINDOW))
        b = {"x": torch.from_numpy(np.ascontiguousarray(np.stack(xb).transpose(0, 2, 1, 3))),
             "reg": torch.from_numpy(np.stack(rb))}
        if need_raw:
            b["raw"] = torch.from_numpy(np.stack(wb))
        yield b


def loss_fn(model, b, w):
    """View-weighted MSE on all 19 features; F reconstructs only the 18 amplitude features (RMS and PSD equal)."""
    if isinstance(model, WaveAmpAE):
        return (w[:, 1:] * (model(b) - b["x"][:, 1:]) ** 2).mean()
    return (w * (model(b) - b["x"]) ** 2).mean()


def val_loss(model, ds, w, need_raw):
    model.eval()
    with torch.no_grad():
        tot = 0.0
        for d in ds:
            tot += float(loss_fn(model, full_batch(d, need_raw), w))
    model.train()
    return tot / len(ds)


def train_model(cls, tr, va, a, fold, log):
    need_raw = cls is FilterbankAE
    torch.manual_seed(SEED + fold)
    rng = np.random.default_rng(SEED + fold)               # same window order as A in train.py
    model = cls()
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    w = view_weights()
    steps = max(1, sum(d["X"].shape[2] for d in tr) // (WINDOW * BATCH))
    best, best_state, wait, t_train = np.inf, None, 0, 0.0
    for ep in range(a.epochs):
        tic = time.time()
        trl = 0.0
        for b in batches(tr, steps, rng, need_raw):
            loss = loss_fn(model, b, w)
            opt.zero_grad()
            loss.backward()
            opt.step()
            trl += float(loss)
        v = val_loss(model, va, w, need_raw)
        t_train += time.time() - tic
        log.append({"fold": fold, "model": cls.name, "epoch": ep, "train_loss": trl / steps, "val_loss": v,
                    "epoch_s": time.time() - tic})
        print(f"fold {fold} {cls.name} epoch {ep} train {trl / steps:.4f} val {v:.4f} ({time.time() - tic:.0f} s)", flush=True)
        if v < best - 1e-4:
            best, wait = v, 0
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
        else:
            wait += 1
            if wait >= a.patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, {"model": cls.name, "fold": fold, "params": n_params(model), "epochs_run": ep + 1,
                   "best_val_loss": best, "train_s": t_train}


def view_r2(x, xh, depths=None):
    """Fraction of variance reconstructed per view; x, xh [19, 24, T]; optionally only some depths."""
    if depths is not None:
        x, xh = x[:, depths], xh[:, depths]
    out = {}
    for v, f in VIEWS.items():
        a, b = x[f], xh[f]
        sst = ((a - a.mean(axis=(1, 2), keepdims=True)) ** 2).sum()
        out[v] = float(1 - ((a - b) ** 2).sum() / max(sst, 1e-12))
    return out


def interp_depth(x, hidden):
    """Linear interpolation over depth from the nearest visible groups (nearest group at the probe ends)."""
    vis = np.setdiff1d(np.arange(x.shape[1]), hidden)
    out = x.copy()
    for h in hidden:
        lo, hi = vis[vis < h], vis[vis > h]
        if len(lo) and len(hi):
            a, b = lo[-1], hi[0]
            out[:, h] = x[:, a] + (x[:, b] - x[:, a]) * (h - a) / (b - a)
        else:
            out[:, h] = x[:, lo[-1] if len(lo) else hi[0]]
    return out


def imputation(model, d, fold, pid, rng):
    x = d["X"].transpose(1, 0, 2)                           # [19, 24, T]
    b = full_batch(d, False)
    masks = [("block3", np.arange(s, s + 3)) for s in (2, 8, 14, 19)]
    masks += [("random25", np.sort(rng.choice(24, size=6, replace=False))) for _ in range(4)]
    rows = []
    for kind, hidden in masks:
        keep = torch.ones(1, 1, 24, 1)
        keep[0, 0, hidden] = 0.0
        with torch.no_grad():
            xh = model(b, keep=keep)[0].numpy()
        for method, rec in (("masked_ae", xh), ("depth_interp", interp_depth(x, hidden))):
            r = view_r2(x, rec, hidden)
            rows.append({"pid": pid, "fold": fold, "mask": kind, "hidden": " ".join(map(str, hidden)),
                         "method": method, **{f"r2_{k}": v for k, v in r.items()}})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--only-fold", type=int, required=True)
    ap.add_argument("--models", nargs="+", default=["C", "D", "E"])
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    fold = a.only_fold
    classes = [MODELS[m] for m in a.models]
    need_raw = FilterbankAE in classes
    data = load_all()
    load_extra(data, need_raw)
    folds = session_folds(data, a.folds)
    tag = "_".join(a.models)
    for sub in (f"latents_{tag}", "models"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)

    test = [p for p, d in data.items() if folds[d["eid"]] == fold]
    train_eids = sorted({d["eid"] for d in data.values() if folds[d["eid"]] != fold})
    val_eids = set(np.random.default_rng(SEED + fold).choice(train_eids, size=2, replace=False))
    tr = [d for d in data.values() if folds[d["eid"]] != fold and d["eid"] not in val_eids]
    va = [d for d in data.values() if d["eid"] in val_eids]

    log, cost, held, imp = [], [], [], []
    trained = []
    for cls in classes:
        m, c = train_model(cls, tr, va, a, fold, log)
        torch.save(m.state_dict(), OUT / "models" / f"{cls.name}_fold{fold}.pt")
        trained.append(m)
        cost.append(c)
        pd.DataFrame(log).to_csv(OUT / f"training_log_{tag}_fold{fold}.csv", index=False)

    mu, W = pooled_group_pca([d["X"] for d in tr + va])
    rng = np.random.default_rng(SEED + 100 + fold)
    for pid in test:
        d = data[pid]
        x = d["X"].transpose(1, 0, 2)
        lat_path = OUT / f"latents_{tag}" / f"{pid}.npz"
        Z = {"fold": np.array(fold)}
        xg = d["X"] - mu[None]
        rec_pca = (mu[None] + np.einsum("gft,fk,hk->ght", xg, W, W)).transpose(1, 0, 2)
        held.append({"pid": pid, "fold": fold, "method": "group_pca_pooled_48", **view_r2(x, rec_pca)})
        for m, c in zip(trained, cost):
            b = full_batch(d, isinstance(m, FilterbankAE))
            with torch.no_grad():
                tic = time.perf_counter()
                z = m.encode(b)
                c.setdefault("encode_s", []).append(time.perf_counter() - tic)
                xh = m.dec(torch.cat([z, m._e(b, z.shape[-1])], dim=1)) if isinstance(m, AnatomyAE) else m.dec(z)
            if isinstance(m, WaveAmpAE):                    # waveform is stored, not reconstructed
                Z[m.name] = m.flatten(z, b["x"])[0].numpy().astype(np.float32)
                xh = torch.cat([b["x"][:, :1], xh], dim=1)
            else:
                Z[m.name] = m.flatten(z)[0].numpy().astype(np.float32)
            held.append({"pid": pid, "fold": fold, "method": m.name, **view_r2(x, xh[0].numpy())})
            if isinstance(m, MaskedDepthAE):
                imp += imputation(m, d, fold, pid, rng)
        np.savez(lat_path, **Z)
    for c in cost:
        c["encode_s_per_100s"] = float(np.median(c.pop("encode_s")))
    pd.DataFrame(cost).to_csv(OUT / f"cost_{tag}_fold{fold}.csv", index=False)
    pd.DataFrame(held).to_csv(OUT / f"heldout_{tag}_fold{fold}.csv", index=False)
    if imp:
        pd.DataFrame(imp).to_csv(OUT / f"imputation_fold{fold}.csv", index=False)
    print(f"fold {fold}: {len(test)} held-out insertions, {len(tr)} train, {len(va)} val -- done", flush=True)


if __name__ == "__main__":
    main()
