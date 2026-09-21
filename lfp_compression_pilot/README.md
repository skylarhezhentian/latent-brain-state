# LFP compression pilot

**Question.** The current LFP representation keeps 16 spatial PCs of the signed
waveform at 25 Hz. Averaging to 25 Hz removes everything above 12.5 Hz, and
Alon's 8 Sep result is that behaviour is carried mainly by LFP *amplitude*. At the
same budget, can a representation keep beta/gamma amplitude without losing what
PCA keeps?

**Candidates** (pre-registered, `PREREGISTRATION.md`):
C1 = the current 16 waveform PCs + 4 PCs of beta/gamma log-RMS envelopes (20 dims);
C2 = 12 waveform PCs + 4 amplitude PCs (16 dims, the current budget).
Compared with PCA 16 (current, reimplemented from the slides), PCA 20 (C1's budget),
16-group spatial averaging, and lfpack at IBL's production settings.

## Files

| File | What it does |
|---|---|
| `pilot_config.py` | every setting, fixed before results |
| `pilot_data.py` | load a cached probe, decimate to 250 Hz, build targets, load behaviour at 25 Hz |
| `pilot_lfpack.py` | lfpack as in production (250 Hz, 2048/128 chunks, optional Cadzow) + measured costs |
| `pilot_representations.py` | spatial averaging, waveform PCA, amplitude-augmented PCA, 1152-feature reference |
| `pilot_metrics.py` | folds, blocked-CV ridge readout, every metric |
| `run_pilot.py` | one probe, all methods and sources |
| `run_replication.py` | the same on every other cached probe, paired statistics |
| `make_figures.py` | figures from the CSVs |
| `export_deck_numbers.py` | every number the deck quotes, from the CSVs |
| `verify_outputs.py` | compile check, numbers-match-outputs check, deck-numbers check |
| `PREREGISTRATION.md` | protocol, decisions, and what was superseded and why |

Depends on `../fast_lfp_benchmark/extract.py` for the raw data cache.

## Reproduce

All commands from `~/Downloads/lfp-brain-state/lfp_based_brain_state`, with

```bash
export KMP_DUPLICATE_LIB_OK=TRUE OMP_NUM_THREADS=1
PY=/opt/miniconda3/envs/lfp-brain-state/bin/python
```

1. **Data** (~8 min per session; streams from the public IBL server; skips non-public sessions)
   ```bash
   $PY -m fast_lfp_benchmark.extract --sessions 6 --duration 120
   ```
   Cache: `~/Downloads/lfp-brain-state/fast_lfp_cache/<eid>/<pid>.npz`. The pilot uses
   only `x_uV` (destriped at 2500 Hz, FIR-decimated to 500 Hz). The extractor's
   2500 Hz lfpack arrays are **not** used: lfpack is re-run at 250 Hz in `pilot_lfpack.py`.

2. **Pilot** (one probe, 100 s; ~10 min, of which Cadzow ~4 min)
   ```bash
   $PY -m lfp_compression_pilot.run_pilot
   ```
3. **Replication** (every other cached probe; ~1 min each)
   ```bash
   $PY -m lfp_compression_pilot.run_replication
   ```
4. **Figures, deck numbers, checks**
   ```bash
   $PY -m lfp_compression_pilot.make_figures
   $PY -m lfp_compression_pilot.export_deck_numbers
   $PY -m lfp_compression_pilot.verify_outputs --deck <deck.pptx>
   ```

Outputs: `~/Downloads/lfp-brain-state/lfp_compression_pilot_results/`
(`pilot_c4b5a9fa/`, `replication_*/`, `replication/`, `figures/`, `deck_numbers.json`).
Behaviour at 25 Hz and lfpack reconstructions are cached in
`~/Downloads/lfp-brain-state/lfp_compression_pilot_cache/`.

Environment: conda env `lfp-brain-state` (Python 3.11, numpy 1.26, torch not needed),
lfpack 0.4.0, ibllib 4.0.1, single-threaded on an Intel i5 laptop. The OpenMP
variables avoid a crash from duplicate OpenMP runtimes in this environment.
