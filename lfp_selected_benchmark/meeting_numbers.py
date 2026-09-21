"""
Every number quoted in the 21 Sep meeting script, recomputed from the per-insertion results, and a check
that every decimal in the script matches one of them.

    $PY -m lfp_selected_benchmark.meeting_numbers --out NUMBERS.json --check SCRIPT.md

A decimal in the script passes if some number here, rounded to the same number of places, equals it
(percentages are compared as fractions). Integers, dates, frequencies and sizes quoted as design
constants are listed in CONSTANTS.
"""
import argparse
import glob
import json
import re

import numpy as np
import pandas as pd

from .run_selected import RESULTS
from lfp_autoencoder_prototype.train import AE_RESULTS

# Design constants and physical quantities quoted in text (not results)
CONSTANTS = {0.5, 2.5, 12.5, 3.11}   # 3.11 = Python version


def per_insertion_mean(files, keys):
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    return df.groupby(["pid"] + keys).r.mean().reset_index()


def build():
    N = {}
    s = pd.read_csv(RESULTS / "summary_by_method.csv").set_index("method")
    for m in s.index:
        for c in ("beh_r_mean", "r2_wave", "r2_env_gamma", "r2_amp_global", "r2_amp_spatial", "shuf_r_mean"):
            if c in s.columns:
                N[f"bench.{m}.{c}"] = float(s.loc[m, c])
    b = pd.read_csv(RESULTS / "behaviour_r_by_method.csv")
    for _, r in b.iterrows():
        N[f"bench_beh.{r.method}.{r.behaviour}"] = float(r.median_r)
    for f, key in (("local_global_by_behaviour.csv", "signal"), ("regions_by_behaviour.csv", "region")):
        t = pd.read_csv(RESULTS / f)
        for _, r in t.iterrows():
            for c in t.columns:
                if c.startswith("beh_r_"):
                    N[f"{f.split('_')[0]}.{r[key]}.{c}"] = float(r[c])
    for f in ("saturation_summary.csv", "tradeoff_summary.csv"):
        t = pd.read_csv(RESULTS / f)
        for _, r in t.iterrows():
            for c in t.columns:
                if c not in ("method", "n"):
                    N[f"{f.split('_')[0]}.{r.method}.{c}"] = float(r[c])
    today = json.load(open(RESULTS / "today_numbers.json"))
    for grp in ("split", "view"):
        for k, d in today[grp].items():
            for c, v in d.items():
                N[f"today.{grp}.{k}.{c}"] = float(v)
    p = pd.read_csv(RESULTS / "paired_vs_baselines.csv")
    for _, r in p.iterrows():
        N[f"paired.{r.baseline}.{r.method}.{r.metric}.median_diff"] = float(r.median_diff)

    # verification (benchmark convention: mean over behaviours per insertion, median over insertions)
    ctl = per_insertion_mean(sorted((RESULTS / "probes_controls").glob("*.csv")), ["check", "rep", "variant", "readout"])
    ctl = ctl[ctl.check != "align"]
    for k, v in ctl.groupby(["check", "rep", "variant", "readout"]).r.median().items():
        N["controls." + ".".join(k)] = float(v)
    al = pd.read_csv(RESULTS / "controls_align.csv")
    for beh, g in al.groupby("behaviour"):
        N[f"align.{beh}.median_lag_ms"] = float(g.peak_lag_ms.median())
        N[f"align.{beh}.within_200ms"] = float((g.peak_lag_ms.abs() <= 200).mean())

    # neural validity
    v = pd.concat([pd.read_csv(f) for f in sorted((RESULTS / "probes_validity").glob("*.csv"))], ignore_index=True)
    vp = v.groupby(["pid", "check", "set", "band"]).r.mean().reset_index()
    for k, val in vp.groupby(["check", "set", "band"]).r.median().items():
        N["validity." + ".".join(k)] = float(val)
    w = vp.pivot_table(index="pid", columns=["check", "set", "band"], values="r")
    d = (w[("csd", "grey_all", "all")] - w[("potential", "grey_all", "all")]).dropna()
    N["validity.grey.csd_minus_potential.median"] = float(d.median())
    N["validity.grey.csd_minus_potential.n_higher"] = int((d > 0).sum())
    for sig in ("potential", "csd"):
        dd = (w[(sig, "grey_matched_void", "all")] - w[(sig, "void", "all")]).dropna()
        N[f"validity.void.{sig}.grey_minus_void.median"] = float(dd.median())
        N[f"validity.void.{sig}.n_grey_higher"] = int((dd > 0).sum())

    # aliasing
    a = pd.read_csv(RESULTS / "aliasing_by_insertion.csv")
    for c in a.columns:
        if c != "pid":
            N[f"aliasing.{c.split(chr(10))[0]}.median"] = float(a[c].median())

    # models and purpose tests, when present
    cmp_dir = AE_RESULTS / "compare"
    if (cmp_dir / "summary_models.csv").exists():
        t = pd.read_csv(cmp_dir / "summary_models.csv").set_index("method")
        for m in t.index:
            for c in t.columns:
                if c != "n":
                    N[f"models.{m}.{c}"] = float(t.loc[m, c])
        pr = pd.read_csv(cmp_dir / "paired_models.csv")
        for _, r in pr.iterrows():
            N[f"models_paired.{r.baseline}.{r.method}.{r.metric}.median_diff"] = float(r.median_diff)
    if (cmp_dir / "paired_sessions_models.csv").exists():
        for _, r in pd.read_csv(cmp_dir / "paired_sessions_models.csv").iterrows():
            N[f"models_sessions.{r.baseline}.{r.method}.median_session_diff"] = float(r.median_session_diff)
    ho = sorted((AE_RESULTS / "variants").glob("heldout_*_fold*.csv"))
    if ho:
        h = pd.concat([pd.read_csv(f) for f in ho]).groupby("method")[["waveform", "rms", "psd"]].median()
        for m in h.index:
            for c in h.columns:
                N[f"heldout_recon.{m}.{c}"] = float(h.loc[m, c])
    sd = AE_RESULTS.parent / "model_purpose_2026-09-21" / "tables" / "heldout_static_vs_dynamic.csv"
    if sd.exists():
        t = pd.read_csv(sd).set_index("method")
        for m in t.index:
            for c in t.columns:
                N[f"static_dynamic.{m}.{c}"] = float(t.loc[m, c])
    # parameter counts, in thousands (A/B from train.py's metadata, C-F from the fork's cost files)
    for f in sorted(AE_RESULTS.glob("train_meta_fold*.json"))[:1]:
        for m, n in json.load(open(f))["params"].items():
            N[f"params_k.{m}"] = n / 1000
    for f in sorted((AE_RESULTS / "variants").glob("cost_*_fold0.csv")):
        c = pd.read_csv(f).iloc[0]
        N[f"params_k.{c['model']}"] = float(c["params"]) / 1000
    e15 = AE_RESULTS / "run_e15" / "summary_ae_compare.csv"
    if e15.exists():
        t = pd.read_csv(e15).set_index("method")
        for m in t.index:
            for c in ("beh_r_mean", "r2_wave", "r2_amp_spatial"):
                N[f"models_e15.{m}.{c}"] = float(t.loc[m, c])
    cca = AE_RESULTS / "purpose" / "cca.csv"
    if cca.exists():
        t = pd.read_csv(cca).groupby("method")[["cc_top5", "hp1hz_top5", "hp1hz_null_top5"]].median()
        for m in t.index:
            for c in t.columns:
                N[f"cca.{m}.{c}"] = float(t.loc[m, c])
    ps = AE_RESULTS / "purpose" / "purpose_summary.csv"
    if ps.exists():
        t = pd.read_csv(ps)
        key = t.columns[0]
        for _, r in t.iterrows():
            for c in t.columns[1:]:
                if pd.api.types.is_number(r[c]) and np.isfinite(r[c]):
                    N[f"purpose.{r[key]}.{c}"] = float(r[c])
    return N


def check(N, path):
    text = open(path).read()
    values = [x for x in N.values() if isinstance(x, (int, float)) and np.isfinite(x)]
    bad = []
    for m in re.finditer(r"(?<![\w.])([−-]?\d+\.\d+)(%?)", text):
        tok, pct = m.group(1).replace("−", "-"), m.group(2)
        x = float(tok) / (100 if pct else 1)
        nd = len(tok.split(".")[1]) + (2 if pct else 0)
        if abs(float(tok)) in CONSTANTS:
            continue
        if not any(round(abs(v), nd) == round(abs(x), nd) for v in values):
            bad.append(m.group(0))
    return sorted(set(bad))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", nargs="*", default=[])
    a = ap.parse_args()
    N = build()
    json.dump(N, open(a.out, "w"), indent=1, sort_keys=True)
    print(f"{len(N)} numbers -> {a.out}")
    for f in a.check:
        bad = check(N, f)
        print(f"{f}: {'all decimals traceable' if not bad else 'UNTRACEABLE: ' + ', '.join(bad)}")
