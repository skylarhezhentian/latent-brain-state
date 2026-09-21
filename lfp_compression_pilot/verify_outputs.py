"""
Pre-delivery checks: the code compiles, the reported numbers are the generated
numbers, and every decimal quoted in the deck exists in the outputs.

    /opt/miniconda3/envs/lfp-brain-state/bin/python -m lfp_compression_pilot.verify_outputs --deck DECK.pptx

Checks
  1. every .py in lfp_compression_pilot/ and fast_lfp_benchmark/ compiles
  2. deck_numbers.json equals a fresh recomputation from the result CSVs
  3. replication paired_summary.csv equals a fresh aggregation of the per-probe fold CSVs
  4. every number with 2+ decimals in the deck text matches a value in deck_numbers.json
     (same rounding as displayed, sign-insensitive); unmatched numbers are listed
Exit code 1 if any check fails.
"""
import argparse
import json
import py_compile
import re
import sys
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from . import pilot_config as C
from .export_deck_numbers import collect
from .run_replication import aggregate

HERE = Path(__file__).resolve().parent


def flatten(obj):
    if isinstance(obj, dict):
        for v in obj.values():
            yield from flatten(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from flatten(v)
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield float(obj)


def deck_text(path):
    z = zipfile.ZipFile(path)
    slides = sorted((n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
                    key=lambda n: int(re.findall(r"\d+", n)[-1]))
    return {int(re.findall(r"\d+", n)[-1]): " ".join(re.findall(r"<a:t>(.*?)</a:t>", z.read(n).decode("utf8")))
            for n in slides}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deck", default=None)
    ap.add_argument("--search", action="store_true",
                    help="also check search_numbers.json (lfp_compression_search) and allow its values in the deck")
    args = ap.parse_args()
    ok = True

    files = (sorted(HERE.glob("*.py")) + sorted((HERE.parent / "fast_lfp_benchmark").glob("*.py"))
             + sorted((HERE.parent / "lfp_compression_search").glob("*.py")))
    for f in files:
        try:
            py_compile.compile(str(f), doraise=True)
        except py_compile.PyCompileError as e:
            ok = False
            print("COMPILE FAIL", f, e)
    print(f"[1] {len(files)} python files compile: {'OK' if ok else 'FAIL'}")

    saved = json.load(open(Path(C.RESULTS_DIR) / "deck_numbers.json"))
    fresh = json.loads(json.dumps(collect()))
    same = saved == fresh
    ok &= same
    print(f"[2] deck_numbers.json matches CSVs: {'OK' if same else 'FAIL'}")

    rep = Path(C.RESULTS_DIR) / "replication"
    if (rep / "paired_summary.csv").exists():
        csvs = sorted(Path(C.RESULTS_DIR).glob("replication_*/fold_metrics.csv"))
        with tempfile.TemporaryDirectory() as tmp:
            _, _, s_new = aggregate(csvs, Path(tmp))
        s_old = pd.read_csv(rep / "paired_summary.csv")
        num = s_old.select_dtypes("number").columns
        same = np.allclose(s_old[num].to_numpy(float), s_new[num].to_numpy(float), equal_nan=True)
        ok &= same
        print(f"[3] replication summary reproduces from {len(csvs)} probe files: {'OK' if same else 'FAIL'}")

    values = list(flatten(saved))
    if args.search:
        from lfp_compression_search.export_search_numbers import ROOT as SROOT, collect as collect_search
        s_saved = json.load(open(SROOT / "search_numbers.json"))
        same = s_saved == json.loads(json.dumps(collect_search()))
        ok &= same
        print(f"[3b] search_numbers.json matches search CSVs: {'OK' if same else 'FAIL'}")
        values += list(flatten(s_saved))

    if args.deck:
        unmatched = []
        for slide, text in deck_text(args.deck).items():
            text = text.replace("−", "-").replace("&#8722;", "-")
            for tok in re.findall(r"[-+]?\d*\.\d{2,}", text):
                nd = len(tok.split(".")[1])
                x = abs(float(tok))
                if not any(abs(round(abs(v), nd) - x) < 10 ** (-nd) / 2 + 1e-12 for v in values):
                    unmatched.append((slide, tok))
        ok &= not unmatched
        print(f"[4] deck decimals traceable to outputs: {'OK' if not unmatched else 'FAIL'}")
        for s, t in unmatched:
            print(f"    slide {s}: {t} not found in deck_numbers.json")

    print("ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
