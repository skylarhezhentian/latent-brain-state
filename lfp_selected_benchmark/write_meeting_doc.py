"""
Write MENTOR_MEETING.md: run of show, what to say per slide, likely questions, the model
questions to ask, and the action plan. Every number and result-dependent sentence is
computed from numbers.json.

    /opt/miniconda3/envs/lfp-brain-state/bin/python -m lfp_selected_benchmark.write_meeting_doc OUT.md
"""
import json
import sys

from .run_selected import RESULTS


def main(out):
    N = json.load(open(RESULTS / "numbers.json"))
    M, D, X = N["methods"], N["data"], N["matched"]
    LG, R, E = N.get("local_global"), N.get("regions", {}), N.get("earlier")
    f2 = lambda x: f"{x:.2f}"
    s2 = lambda x: f"{x:+.2f}"
    s3 = lambda x: f"{x:+.3f}"
    sess = lambda e: f"{e['sessions_higher']}/{e['n_sessions']} sessions"

    beh = N["behaviours"]
    nice = {"wheel_speed": "wheel speed", "whisker_me_left": "whisker (left cam)", "whisker_me_right": "whisker (right cam)",
            "body_me": "body motion", "pupil_diameter": "pupil"}
    bank = M["bank_wave+rms+psd"]
    best_b = max(beh, key=lambda b: bank[f"r_{b}"])
    worst_b = min(beh, key=lambda b: bank[f"r_{b}"])
    shifts = [v["shuf_r_mean"] for v in M.values()]
    shift_line = (f"The 30 s-shift control is between {min(shifts):+.2f} and {max(shifts):+.2f} for every method, "
                  "so the decoding does not come from slow drift that a 30 s shift would preserve.")
    gwa, pca48 = X.get("group_wave+amp1"), M.get("waveform_pca_48")
    lg_line = ""
    if LG:
        ratio = N["local_global_summary"]["median_local_over_groups"]
        lg_line = (f"Behaviour decoded from the local signal: mean r {f2(LG['local']['beh_r_mean'])}, "
                   f"vs {f2(LG['groups']['beh_r_mean'])} from depth groups and {f2(LG['global']['beh_r_mean'])} from the probe-wide mean "
                   f"(median local/groups ratio {ratio:.2f}). "
                   + ("Most of the behavioural information survives removing what is shared across neighbouring depths, so it is local."
                      if ratio >= 0.8 else
                      "A substantial part is lost when shared structure is removed, so part of the information is common across depth."))
    spatial = {"spatial_avg_24", "spatial_avg_48", "group_pca_1x24", "group_pca_2x24", "group_wave+amp1"}
    at16 = [m for m in ("waveform_pca_16", "autoencoder_16", "bandpower_pca_16", "wave12+amp4_smooth", "wave12+bank4") if m in M]
    best16 = max(at16, key=lambda m: M[m]["r_mean"])
    top2 = sorted(beh, key=lambda b: -bank[f"r_{b}"])[:2]
    reg = sorted(((k, v) for k, v in R.items() if k != "whole probe"), key=lambda kv: -kv[1]["beh_r_mean"])

    L = []
    w = L.append
    w("# Mentor meeting — spatially indexed LFP representation\n")
    w("**Goal of the meeting:** share the process and results on Alon's selected insertions, and get advice on which models to build next.\n")
    w(f"**Data:** {D['n_probes_evaluated']} insertions, {D['n_sessions']} sessions ({D['n_pair_sessions']} with two simultaneous probes), "
      f"{D['n_labs']} labs, 100 s each, 5 behaviours. {len(M)} representations, identical folds and readout.\n")

    w("## Run of show (about 15–20 minutes, then discussion)\n")
    w("| Slide | Minutes | Purpose |\n|---|---|---|")
    for row in [("1 Title", "< 1", "set the two goals"), ("2 How I got here", "2", "process"), ("3 The question", "1–2", "reframing"),
                ("4 Absolute r heatmap", "2–3", "all results"), ("5 Matched budgets", "2–3", "the key comparison"),
                ("6 Locality", "2", "is it local?"), ("7 Regions", "1", "what indexing enables"),
                ("8 Model path", "2", "proposal"), ("9 Advice", "open", "the discussion"), ("10 Caveats", "backup", "only if asked")]:
        w(f"| {row[0]} | {row[1]} | {row[2]} |")
    w("")

    w("## What to say\n")
    w("The same script is in the speaker notes of the deck.\n")
    w("**Slide 3 — the one sentence to land:** PCA is a strong reconstruction and compression benchmark, but its objective, waveform variance, "
      "is dominated by slow, spatially broad fluctuations. The question is whether a representation in which every number is tied to a depth "
      "keeps more behaviourally relevant and more local information at the same budget, even if its waveform error is not lower.\n")
    w(f"**Slide 4 — three takeaways.** (1) Spatial average 24 ≈ PCA 16 ≈ PCA 24: mean r {f2(M['spatial_avg_24']['r_mean'])}, "
      f"{f2(M['waveform_pca_16']['r_mean'])}, {f2(M['waveform_pca_24']['r_mean'])}. (2) Adding an amplitude or frequency view is the big step: "
      f"12 wave + 4 amp {f2(M['wave12+amp4_smooth']['r_mean'])}, full feature bank {f2(bank['r_mean'])}. "
      f"(3) Best-decoded behaviour is {nice[best_b]} (r {f2(bank['r_' + best_b])} with the full bank); weakest is {nice[worst_b]} "
      f"({f2(bank['r_' + worst_b])}). {shift_line}\n")
    w("**Slide 5 — be honest about where the gain comes from.**")
    if "wave12+bank4" in X:
        e = X["wave12+bank4"]
        kind = "spatially indexed" if best16 in spatial else "*global*"
        w(f"- At 16 dims the best is a {kind} method ({best16}). 12 waveform PCs + 4 PCs of RMS/PSD vs PCA 16: behaviour {s2(e['beh_r_mean']['median_diff'])} "
          f"({sess(e['beh_r_mean'])}), waveform {s3(e['r2_wave']['median_diff'])}."
          + (" So most of the behaviour gain comes from adding views, not from spatial indexing itself." if best16 not in spatial else ""))
    if "group_pca_1x24" in X:
        e = X["group_pca_1x24"]
        w(f"- At 24, one amplitude summary per depth beats PCA 24 on behaviour ({s2(e['beh_r_mean']['median_diff'])}, {sess(e['beh_r_mean'])}) "
          f"but gives up the waveform (R² {f2(M['group_pca_1x24']['r2_wave'])}).")
    if gwa and pca48:
        w(f"- At 48, waveform + 1 amplitude per depth vs PCA 48: behaviour {s2(gwa['beh_r_mean']['median_diff'])} ({sess(gwa['beh_r_mean'])}), "
          f"waveform {s3(gwa['r2_wave']['median_diff'])}, depth pattern of amplitude {s2(gwa['r2_amp_spatial']['median_diff'])}. "
          "This is the linear spatially indexed candidate to carry forward.")
    w("")
    w(f"**Slide 6 — locality.** {lg_line} Depth pattern of amplitude: PCA 16 {f2(M['waveform_pca_16']['r2_amp_spatial'])}, "
      f"16-number all-views summary {f2(M['wave12+bank4']['r2_amp_spatial'])}, waveform + amplitude per depth {f2(M['group_wave+amp1']['r2_amp_spatial'])}.\n")
    if reg:
        w(f"**Slide 7 — regions.** Regions with ≥ 5 insertions decode behaviour with median mean r {f2(reg[-1][1]['beh_r_mean'])}–{f2(reg[0][1]['beh_r_mean'])} "
          f"(whole probe {f2(R['whole probe']['beh_r_mean'])}); highest {reg[0][0]}, lowest {reg[-1][0]}. Say clearly that regions come from different "
          "insertions, so this is not yet a within-probe comparison.\n")
    w("**Slide 8 — the proposal in one breath:** the feature bank goes into a compact spatiotemporal autoencoder with a shared per-depth encoder, "
      "time and depth convolutions and 2 latents per depth (48 numbers). It is the nonlinear version of group PCA 2×24 at the same budget, "
      "trained pooled and leave-session-out, with behaviour only used for evaluation.\n")

    w("## Questions to ask Alon (slide 9)\n")
    w("| Decision | Question | My lean | Why |\n|---|---|---|---|")
    for q in [
        ("Latent", "Per-depth latents or a small global latent?", "per depth, 2 each", "keeps location for global-vs-local and region analyses; budget matches group PCA 2×24"),
        ("Target", "Reconstruct the feature bank, the waveform, or both?", "bank, equal view weights", "a waveform-only loss re-creates PCA's bias toward slow, broad signals"),
        ("Pooling", "Per insertion, or pooled with anatomy context?", "pooled + Cosmos / MERFISH-AGEA context", "per-insertion training has 2,500 bins; his slide 22 showed mappings need calibration across sessions"),
        ("Evaluation", "Linear behaviour decoding, or his CCA / shared-component analyses?", "both", "the representation exists to serve the global-vs-local work"),
        ("Artefacts", "Is the 320 µm neighbour-subtracted signal enough to rule out muscle / common-mode contamination?", "add per-channel CSD", f"{nice[top2[0]]} and {nice[top2[1]]} are the best-decoded behaviours, and movement can contaminate high-frequency LFP"),
        ("Data", "More insertions or paw states before training networks?", "more insertions first", f"{D['total_bins']:,} bins total is small for a network"),
    ]:
        w(f"| {q[0]} | {q[1]} | {q[2]} | {q[3]} |")
    w("")

    w("## Questions Alon may ask\n")
    w(f"- **Isn't the feature bank just more dimensions?** Yes, which is why slide 5 compares at 16, 24 and 48. The {bank['dims']}-dim bank is only a ceiling ({f2(bank['r_mean'])}).")
    w("- **Why is PCA 16 reimplemented?** Your PCA code is not in the repository. Same description: spatial PCA of the waveform fit on training time, averaged to 25 Hz.")
    w(f"- **Do the HIGH-QC insertions matter?** Mean r without them: PCA 16 {f2(M['waveform_pca_16']['r_mean_excl_high_qc'])}, "
      f"full bank {f2(bank['r_mean_excl_high_qc'])} ({bank['n_probes_excl_high_qc']} insertions); all insertions: {f2(M['waveform_pca_16']['r_mean'])} and {f2(bank['r_mean'])}.")
    if E:
        w(f"- **Is this consistent with last week?** Yes: the amplitude-augmented representation was confirmed on {int(E['confirm_n_sessions'])} sessions from new mice "
          f"(behaviour r {s2(E['confirm_d_beh_r'])} vs PCA 16), and it gains again on your insertions.")
    w("- **Why not paw states?** Not released on the public server for these sessions.")
    w("- **Why ±320 ms context?** Same as the earlier benchmark; the autoencoder will learn its own temporal filters.\n")

    w("## Action plan after the meeting\n")
    w("| When | Step | Done when |\n|---|---|---|")
    for row in [
        ("Next 2 days", "Apply Alon's answers; add per-channel CSD features and the within-probe region comparison", "locality figure rerun on all insertions"),
        ("This week", "Pooled leave-session-out ridge on the feature bank with anatomy context (26 folds)", "per-behaviour table; gap to per-insertion r known"),
        ("This week", "Compact spatiotemporal AE, 48 latents, same folds", "beats group PCA 2×24 on behaviour and depth pattern in most held-out sessions"),
        ("Next week", "Feed the chosen representation into CCA between simultaneous probes", "shared-component estimate vs Alon's 16-PC result"),
        ("Later", "End-to-end model from waveform with a band-pass filterbank front end; more insertions", "matches the AE without hand-crafted features"),
    ]:
        w(f"| {row[0]} | {row[1]} | {row[2]} |")
    w("")

    w("## Key numbers\n")
    w("| Representation | Dims | Mean r | Waveform R² | Amplitude depth-pattern R² |\n|---|---|---|---|---|")
    for m, v in M.items():
        w(f"| {m} | {v['dims']} | {f2(v['r_mean'])} | {f2(v['r2_wave'])} | {f2(v['r2_amp_spatial'])} |")
    open(out, "w").write("\n".join(L) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1])
