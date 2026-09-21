# LFP compression search

Looks for a 16-dimensional, 25 Hz LFP representation better than spatial PCA
(`waveform_pca_16`, the current method), with a search set and an untouched
confirmation set. Protocol, candidate list, selection rule and confirmation rule are in
`SEARCH_PROTOCOL.md`, written before any result.

| File | What it does |
|---|---|
| `SEARCH_PROTOCOL.md` | question, data split, 9 methods, selection and confirmation rules |
| `search_methods.py` | the 9 candidate representations and `SearchFeatures` (amplitude envelopes for any band and RMS window) |
| `run_search.py` | evaluate a set (resumable, shardable), aggregate, pick the winner / issue the verdict |
| `search_figures.py` | ranking figure (search set) and confirmation figure |

Reuses `lfp_compression_pilot` for data loading, targets, folds, the ridge readout and all
metrics, so every method is scored by exactly the same code as the pilot.

## Reproduce

```bash
cd ~/Downloads/lfp-brain-state/lfp_based_brain_state
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
PY=/opt/miniconda3/envs/lfp-brain-state/bin/python

# search set = the 12 probes in ~/Downloads/lfp-brain-state/fast_lfp_cache (see lfp_compression_pilot/README.md)
for k in 0 1 2; do $PY -m lfp_compression_search.run_search --set search --shard $k/3 & done; wait
$PY -m lfp_compression_search.run_search --set search --aggregate-only      # writes winner.json

# confirmation set = 6 new sessions, never used before the winner was fixed
$PY -m fast_lfp_benchmark.extract --sessions 6 --duration 120 --seed 1 --no-lfpack \
    --cache-dir ~/Downloads/lfp-brain-state/fast_lfp_cache_confirm \
    --exclude-cache-dir ~/Downloads/lfp-brain-state/fast_lfp_cache
$PY -m lfp_compression_search.run_search --set confirm                      # writes verdict.json
$PY -m lfp_compression_search.search_figures
```

Timings on an Intel i5 laptop, single-threaded processes: search ~2 min per probe
(9 methods × 4 folds), extraction ~4 min per session, confirmation ~1 min per probe.
Outputs: `~/Downloads/lfp-brain-state/lfp_compression_pilot_results/search/`.
