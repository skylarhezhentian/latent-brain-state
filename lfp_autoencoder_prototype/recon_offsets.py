"""
Held-out reconstruction split into a static part and a dynamic part, for the saved models C-F.

E reads 2 of the 16 channels per depth, so its depth profile of mean power differs from the bank's
16-channel average. That is a constant offset per (feature, depth), invisible to the benchmark's ridge
readouts (they fit intercepts) but it dominates plain reconstruction R². This scores each model both
ways: raw R² and R² after removing each (feature, depth) mean residual on the held-out insertion.

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=2 /opt/miniconda3/envs/lfp-brain-state/bin/python -m lfp_autoencoder_prototype.recon_offsets
"""
import os

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch

from .train import load_all, pooled_group_pca, session_folds
from .train_variants import OUT, full_batch, load_extra, view_r2
from .variants import AnatomyAE, FilterbankAE, MaskedDepthAE, WaveAmpAE


def main():
    data = load_all()
    load_extra(data, True)
    folds = session_folds(data, 5)
    rows = []
    for fold in range(5):
        test = [p for p, d in data.items() if folds[d["eid"]] == fold]
        tr = [d["X"] for d in data.values() if folds[d["eid"]] != fold]
        mu, W = pooled_group_pca(tr)
        models = []
        for cls in (MaskedDepthAE, AnatomyAE, FilterbankAE, WaveAmpAE):
            m = cls()
            m.load_state_dict(torch.load(OUT / "models" / f"{cls.name}_fold{fold}.pt", map_location="cpu"))
            models.append(m.eval())
        for p in test:
            d = data[p]
            x = d["X"].transpose(1, 0, 2)
            recs = {"group_pca_pooled_48": (mu[None] + np.einsum("gft,fk,hk->ght", d["X"] - mu[None], W, W)).transpose(1, 0, 2)}
            for m in models:
                b = full_batch(d, isinstance(m, FilterbankAE))
                with torch.no_grad():
                    z = m.encode(b)
                    xh = m.dec(torch.cat([z, m._e(b, z.shape[-1])], dim=1)) if isinstance(m, AnatomyAE) else m.dec(z)
                    if isinstance(m, WaveAmpAE):
                        xh = torch.cat([b["x"][:, :1], xh], dim=1)
                recs[m.name] = xh[0].numpy()
            for name, xh in recs.items():
                off = (xh - x).mean(axis=2, keepdims=True)
                raw, dyn = view_r2(x, xh), view_r2(x, xh - off)
                rows.append({"pid": p, "fold": fold, "method": name, **{f"{k}_raw": v for k, v in raw.items()},
                             **{f"{k}_dynamic": v for k, v in dyn.items()}})
        print(f"fold {fold} done", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "heldout_static_vs_dynamic.csv", index=False)
    print(df.groupby("method")[[c for c in df.columns if c.endswith(("_raw", "_dynamic"))]].median().round(3).to_string())


if __name__ == "__main__":
    main()
