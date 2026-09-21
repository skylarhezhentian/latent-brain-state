#!/bin/bash
# Copy the deck, code, figures, tables and guides into one folder for review.
# Copies only; nothing in the repository or the caches is moved or modified.
#   bash lfp_compression_pilot/assemble_delivery.sh <deck.pptx> <deck.pdf>
set -euo pipefail
ROOT=~/Downloads/lfp-brain-state
REPO=$ROOT/lfp_based_brain_state
RES=$ROOT/lfp_compression_pilot_results
OUT=$ROOT/morning_update_2026-09-14
mkdir -p "$OUT"/{deck,figures,tables,code/lfp_compression_pilot,code/fast_lfp_benchmark,code/dual_state_model_diagnostics}

cp "$1" "$OUT/deck/"
[ -n "${2:-}" ] && cp "$2" "$OUT/deck/"
cp "$RES"/figures/*.png "$OUT/figures/"
cp "$RES/deck_numbers.json" "$OUT/tables/"
for f in fold_metrics summary codec_fidelity reproducibility amplitude_split_selection; do
  cp "$RES/pilot_c4b5a9fa/$f.csv" "$OUT/tables/pilot_$f.csv"
done
cp "$RES/pilot_c4b5a9fa/manifest.json" "$OUT/tables/pilot_manifest.json"
for f in per_probe paired paired_summary session_summary; do
  cp "$RES/replication/$f.csv" "$OUT/tables/replication_$f.csv"
done

cp "$REPO"/lfp_compression_pilot/*.py "$REPO"/lfp_compression_pilot/build_deck.js \
   "$REPO"/lfp_compression_pilot/README.md "$REPO"/lfp_compression_pilot/PREREGISTRATION.md \
   "$REPO"/lfp_compression_pilot/assemble_delivery.sh "$OUT/code/lfp_compression_pilot/"
cp "$REPO"/fast_lfp_benchmark/*.py "$REPO"/fast_lfp_benchmark/README.md "$OUT/code/fast_lfp_benchmark/"
cp "$ROOT"/GitRepo/run_local.py "$ROOT"/GitRepo/eval_diagnostics.py "$OUT/code/dual_state_model_diagnostics/"
echo "assembled $OUT"
find "$OUT" -type f | sort
