"""
A fair linear baseline for option C's imputation test.

Plain linear interpolation in depth copies neighbours' fast, locally independent fluctuations, so its
R² on hidden groups can go well below zero; any regression-optimal predictor shrinks instead. This
fits, on the TRAINING sessions of each fold, one slope and intercept per feature mapping the
interpolated value to the true value (random 25 % masks), then applies it to exactly the masks of
imputation_fold<k>.csv. Written next to C's results as method 'depth_interp_shrunk'.

    cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
    /opt/miniconda3/envs/lfp-brain-state/bin/python -m lfp_autoencoder_prototype.imputation_linear
"""
import numpy as np
import pandas as pd

from .train import SEED, load_all, session_folds
from .train_variants import OUT, interp_depth, view_r2


def main():
    data = load_all()
    folds = session_folds(data, 5)
    rows = []
    for fold in range(5):
        rng = np.random.default_rng(SEED + 200 + fold)
        tr = [d for d in data.values() if folds[d["eid"]] != fold]
        num, den, sx, sy, n = np.zeros(19), np.zeros(19), np.zeros(19), np.zeros(19), np.zeros(19)
        for d in tr:
            x = d["X"].transpose(1, 0, 2)
            for _ in range(2):
                hidden = np.sort(rng.choice(24, size=6, replace=False))
                xi = interp_depth(x, hidden)[:, hidden].reshape(19, -1)
                xt = x[:, hidden].reshape(19, -1)
                sx += xi.sum(1); sy += xt.sum(1); num += (xi * xt).sum(1); den += (xi * xi).sum(1); n += xi.shape[1]
        slope = (num - sx * sy / n) / (den - sx * sx / n)
        icpt = (sy - slope * sx) / n
        masks = pd.read_csv(OUT / f"imputation_fold{fold}.csv")
        for (pid, kind, hid), _ in masks[masks.method == "masked_ae"].groupby(["pid", "mask", "hidden"]):
            hidden = np.array(list(map(int, hid.split())))
            x = data[pid]["X"].transpose(1, 0, 2)
            rec = interp_depth(x, hidden)
            rec[:, hidden] = slope[:, None, None] * rec[:, hidden] + icpt[:, None, None]
            r = view_r2(x, rec, hidden)
            rows.append({"pid": pid, "fold": fold, "mask": kind, "hidden": hid, "method": "depth_interp_shrunk",
                         **{f"r2_{k}": v for k, v in r.items()}})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "imputation_linear_baseline.csv", index=False)
    allr = pd.concat([pd.concat([pd.read_csv(OUT / f"imputation_fold{k}.csv") for k in range(5)]), df])
    print(allr.groupby(["mask", "method"])[["r2_waveform", "r2_rms", "r2_psd"]].median().round(3).to_string())


if __name__ == "__main__":
    main()
