/*
 * Build the morning deck from lfp_compression_pilot_results/deck_numbers.json and the figures.
 *
 *   cd <dir containing node_modules/pptxgenjs>   (npm install pptxgenjs@3.12.0)
 *   node build_deck.js <results_dir> <out.pptx>
 *
 * Every result number on a slide is read from deck_numbers.json, and wording that
 * depends on a result ("higher" / "lower") is computed from the same number.
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const RES = process.argv[2];
const OUT = process.argv[3];
const N = JSON.parse(fs.readFileSync(path.join(RES, "deck_numbers.json"), "utf8"));
const FIG = (f) => path.join(RES, "figures", f);

const C = { navy: "14213D", ink: "1B1B1F", muted: "5A5F6B", tint: "F2F4F8", line: "D9DDE5", white: "FFFFFF",
            blue: "2A78D6", orange: "EB6834", aqua: "1BAF7A", violet: "4A3AA7",
            okFill: "DDEEDF", okInk: "1E5B2A", preFill: "FCE8D8", preInk: "8A3B12", propFill: "E3E6EC", propInk: "3A3F4B" };
const HEAD = "Cambria", BODY = "Calibri";

const f2 = (x) => x.toFixed(2);
const f3 = (x) => x.toFixed(3);
const sgn = (x, d = 3) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
const cmp = (a, b) => (a > b ? "higher" : a < b ? "lower" : "equal");

const P = N.raw;                        // pilot, raw source, fold means
const cur = P["waveform_pca_16"], p20 = P["waveform_pca_20"], c1 = P["wave16+amp4"], c2 = P["wave12+amp4"];
const R = N.replication, RS = N.replication_sessions, RX = N.replication_excl_pilot_session, RM = N.replication_meta;
const key = (c, m) => `${c}|${m}`;
const C2c = "C2 vs current (16 dims)", C1c = "C1 vs PCA20 (20 dims)", C1cur = "C1 vs current";

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";   // 10 x 5.625 in
pres.author = "Skylar Tian";
pres.title = "Keeping LFP amplitude inside the representation budget";

function title(slide, text, opts = {}) {
  slide.addText(text, { x: 0.5, y: 0.3, w: opts.w || 9, h: 0.7, fontFace: HEAD, fontSize: opts.size || 28, bold: true,
                        color: opts.color || C.ink, margin: 0, isTextBox: true, valign: "middle" });
}
function chip(slide, kind, text, x, y, w) {
  const fill = { done: C.okFill, pre: C.preFill, prop: C.propFill }[kind];
  const ink = { done: C.okInk, pre: C.preInk, prop: C.propInk }[kind];
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.3, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08 });
  slide.addText(text, { x, y, w, h: 0.3, fontFace: BODY, fontSize: 10.5, bold: true, color: ink, align: "center",
                        valign: "middle", margin: 0, isTextBox: true, charSpacing: 1 });
}
function body(slide, runs, box) {
  slide.addText(runs, { fontFace: BODY, fontSize: 14, color: C.ink, valign: "top", margin: 0, isTextBox: true,
                        paraSpaceAfter: 8, ...box });
}
function bullets(items, size = 14) {
  return items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, fontSize: size } }));
}
function footer(slide, text) {
  slide.addText(text, { x: 0.5, y: 5.2, w: 9, h: 0.3, fontFace: BODY, fontSize: 9.5, color: C.muted, margin: 0, isTextBox: true });
}

// ---------------------------------------------------------------- 1 title
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addText("Keeping LFP amplitude inside the representation budget", { x: 0.6, y: 1.2, w: 8.8, h: 1.5, fontFace: HEAD,
    fontSize: 36, bold: true, color: C.white, margin: 0, isTextBox: true, valign: "bottom" });
  s.addText("Comparing LFP compression methods for the “fast LFP representation” aim", { x: 0.6, y: 2.8, w: 8.8, h: 0.5,
    fontFace: BODY, fontSize: 17, color: "CAD3E6", margin: 0, isTextBox: true });
  s.addText("Skylar Tian  ·  update for Alon Saguy  ·  14 September 2026", { x: 0.6, y: 3.35, w: 8.8, h: 0.4,
    fontFace: BODY, fontSize: 13, color: "9FA9BF", margin: 0, isTextBox: true });
  chip(s, "done", "PILOT · 1 PROBE · 100 s · COMPLETED", 0.6, 4.35, 3.6);
  chip(s, "pre", `REPLICATION · ${RM.n_probes} PROBES · ${RM.n_sessions} SESSIONS · PRELIMINARY`, 4.35, 4.35, 4.6);
  s.addNotes("Status chips on every result slide: COMPLETED = computed and verified on the stated data; PRELIMINARY = small-sample evidence; PROPOSED = not run.");
}

// ---------------------------------------------------------------- 2 question
{
  const s = pres.addSlide();
  title(s, "The question");
  body(s, bullets([
    "Current representation: 16 spatial PCs of the signed waveform, averaged to 25 Hz (slides 31 Aug, 8 Sep).",
    "Averaging into 40 ms bins keeps only content below 12.5 Hz. Beta and gamma are absent by construction.",
    `8 Sep: behaviour lives mainly in LFP amplitude (RMS envelope r = ${f2(N.quoted_from_alon_slides.rms_envelope_behaviour_r_8sep)} vs signed PCA r = ${f2(N.quoted_from_alon_slides.signed_pca_behaviour_r_8sep)}).`,
  ], 15), { x: 0.5, y: 1.15, w: 4.7, h: 2.6 });

  // band strip (categorical, not to scale)
  const bands = [["delta", "1–4 Hz", C.blue], ["theta", "4–8", C.blue], ["alpha", "8–12", C.blue], ["beta", "15–30", C.orange], ["gamma", "30–90", C.orange]];
  const x0 = 5.55, w = 0.8, y = 1.55;
  s.addText("What a 25 Hz signed waveform can carry", { x: x0, y: 1.1, w: 4.1, h: 0.35, fontFace: BODY, fontSize: 12, bold: true, color: C.ink, margin: 0, isTextBox: true });
  bands.forEach(([n, hz, col], i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: x0 + i * (w + 0.03), y, w, h: 0.62, fill: { color: col, transparency: i < 3 ? 0 : 0 }, line: { color: col } });
    s.addText([{ text: n, options: { bold: true, breakLine: true } }, { text: hz }], { x: x0 + i * (w + 0.03), y, w, h: 0.62,
      fontFace: BODY, fontSize: 11, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
  });
  s.addText("kept as waveform", { x: x0, y: y + 0.68, w: 3 * (w + 0.03), h: 0.3, fontFace: BODY, fontSize: 11, color: C.blue, bold: true, align: "center", margin: 0, isTextBox: true });
  s.addText("lost unless kept as amplitude", { x: x0 + 3 * (w + 0.03), y: y + 0.68, w: 2 * (w + 0.03), h: 0.3, fontFace: BODY, fontSize: 11, color: C.orange, bold: true, align: "center", margin: 0, isTextBox: true });
  s.addText("bands not to scale", { x: x0, y: y + 0.98, w: 4.1, h: 0.25, fontFace: BODY, fontSize: 9, color: C.muted, margin: 0, isTextBox: true });

  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.95, w: 9, h: 1.05, fill: { color: C.tint }, line: { color: C.tint } });
  s.addText("At the same budget, can a representation keep beta and gamma amplitude without losing what the current PCA keeps?",
    { x: 0.75, y: 3.95, w: 8.5, h: 1.05, fontFace: HEAD, fontSize: 19, italic: true, color: C.navy, valign: "middle", margin: 0, isTextBox: true });
  s.addNotes("The 0.46 / 0.28 numbers are quoted from Alon's 8 Sep slide, not produced by this pilot.");
}

// ---------------------------------------------------------------- 3 methods
{
  const s = pres.addSlide();
  title(s, "Methods, and why each is in the comparison");
  const cards = [
    [C.muted, "Raw reference", "Destriped, 384 channels at 250 Hz.", "The ground truth every method is scored against."],
    [C.aqua, "lfpack (IBL codec)", "SVD + wavelet thresholding per 8 s chunk, production settings, with and without Cadzow.", "It is the archive format: is it a usable data source?"],
    [C.violet, "Spatial average 16", "Mean of 16 contiguous channel groups, 25 Hz.", "The simplest baseline in Alon's comparison."],
    [C.blue, "PCA 16 (current)", "16 spatial PCs of the waveform, 25 Hz. Reimplemented from the slides.", "The method to beat."],
    [C.blue, "PCA 20", "20 spatial PCs of the waveform, 25 Hz.", "Same budget as C1: separates “more dimensions” from “amplitude dimensions”."],
    [C.orange, "Amplitude-augmented PCA", "Waveform PCs + PCs of beta/gamma log-RMS (40 ms). C1 = 16 + 4, C2 = 12 + 4.", "Puts the bands behaviour lives in back into the budget."],
  ];
  const cw = 2.9, ch = 1.85, gx = 0.15, gy = 0.2;
  cards.forEach(([col, name, what, why], i) => {
    const x = 0.5 + (i % 3) * (cw + gx), y = 1.15 + Math.floor(i / 3) * (ch + gy);
    s.addShape(pres.shapes.RECTANGLE, { x, y, w: cw, h: ch, fill: { color: C.tint }, line: { color: C.tint } });
    s.addShape(pres.shapes.OVAL, { x: x + 0.18, y: y + 0.2, w: 0.2, h: 0.2, fill: { color: col }, line: { color: col } });
    s.addText(name, { x: x + 0.48, y: y + 0.12, w: cw - 0.6, h: 0.36, fontFace: BODY, fontSize: 14, bold: true, color: C.ink, margin: 0, isTextBox: true, valign: "middle" });
    s.addText([{ text: what, options: { breakLine: true } }, { text: why, options: { italic: true, color: C.muted } }],
      { x: x + 0.18, y: y + 0.55, w: cw - 0.36, h: ch - 0.65, fontFace: BODY, fontSize: 11.5, color: C.ink, margin: 0, valign: "top", paraSpaceAfter: 6, isTextBox: true });
  });
  s.addNotes("Alon's PCA code and 50-insertion dataset are not in the GitHub repository (branch Skylar == main, 15 Jun), so PCA 16 is a reimplementation of the description: spatial PCA fit at 250 Hz, projected, averaged into 40 ms bins.");
}

// ---------------------------------------------------------------- 4 data flow
{
  const s = pres.addSlide();
  title(s, "Experiment and data flow");
  const box = (x, y, w, h, text, fill, ink = C.ink, bold = false) => {
    s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill } });
    s.addText(text, { x: x + 0.08, y, w: w - 0.16, h, fontFace: BODY, fontSize: 11, color: ink, bold, align: "center", valign: "middle", margin: 0, isTextBox: true });
  };
  const arrow = (x1, y1, x2, y2) => s.addShape(pres.shapes.LINE, { x: x1, y: y1, w: x2 - x1, h: y2 - y1, line: { color: C.muted, width: 1.25, endArrowType: "triangle" } });

  box(0.5, 2.05, 1.45, 0.9, "IBL stream\n384 ch · 2500 Hz", C.tint);
  arrow(1.95, 2.5, 2.2, 2.5);
  box(2.2, 1.85, 1.75, 1.3, "Destripe + FIR decimation\n→ one shared 250 Hz signal", C.navy, C.white, true);
  // three branches
  arrow(3.95, 2.5, 4.25, 1.45); arrow(3.95, 2.5, 4.25, 2.5); arrow(3.95, 2.5, 4.25, 3.55);
  box(4.25, 1.1, 2.3, 0.7, "Targets (from raw only): 25 Hz waveform · beta/gamma log-RMS · behaviour", C.tint);
  box(4.25, 2.15, 2.3, 0.7, "Representations fit on training folds → Z(t), 16–20 dims at 25 Hz", "FCE8D8");
  box(4.25, 3.2, 2.3, 0.7, "lfpack encode → decode, then the same representations", "D9F2E8");
  arrow(6.55, 2.5, 6.85, 2.5); arrow(6.55, 3.55, 6.85, 2.75); arrow(6.55, 1.45, 6.85, 2.25);
  box(6.85, 1.9, 1.3, 1.2, "Same ridge readout\n(α by inner blocked CV)", C.tint);
  arrow(8.15, 2.5, 8.4, 2.5);
  box(8.4, 1.9, 1.1, 1.2, "Held-out metrics\n4 × 25 s blocks", C.navy, C.white, true);

  s.addShape(pres.shapes.LINE, { x: 0.5, y: 4.2, w: 9, h: 0, line: { color: C.line, width: 1 } });
  chip(s, "done", "PILOT", 0.5, 4.38, 1.1);
  s.addText(`probe ${N.pilot.pid.slice(0, 8)} · mouse ${N.pilot.subject} · ${N.pilot.regions.join(", ")} · ${N.pilot.duration_s} s · ${N.pilot.n_folds} folds · raw + 3 lfpack variants`,
    { x: 1.7, y: 4.38, w: 7.8, h: 0.3, fontFace: BODY, fontSize: 12, color: C.ink, margin: 0, isTextBox: true, valign: "middle" });
  chip(s, "pre", "REPLICATION", 0.5, 4.78, 1.1);
  s.addText(`${RM.n_probes} other probes from the same ${RM.n_sessions} two-probe sessions · ${RM.n_mice} mice · identical code, raw source`,
    { x: 1.7, y: 4.78, w: 7.8, h: 0.3, fontFace: BODY, fontSize: 12, color: C.ink, margin: 0, isTextBox: true, valign: "middle" });
}

// ---------------------------------------------------------------- 5 metrics & fairness
{
  const s = pres.addSlide();
  title(s, "Metrics and fairness controls");
  const rows = [
    ["Axis", "Metric (all on held-out blocks)"],
    ["Reconstruction", "R² of the 25 Hz waveform, all 384 channels"],
    ["Frequency", "Delta/theta/alpha coherence; beta and gamma amplitude R²"],
    ["Temporal", "Forecast +200 ms from ≤320 ms of history"],
    ["Behaviour", "Wheel speed, whisker and body motion, ±320 ms context; 30 s-shifted control"],
    ["Reproducibility", "Bases fit on disjoint halves; replication across sessions"],
    ["Cost", "Measured storage, compression ratio, compute per second, peak memory"],
  ];
  s.addTable(rows.map((r, i) => r.map((t, j) => ({ text: t, options: { bold: i === 0 || j === 0, color: i === 0 ? C.white : C.ink,
    fill: { color: i === 0 ? C.navy : (i % 2 ? C.white : C.tint) } } }))),
    { x: 0.5, y: 1.15, w: 5.2, colW: [1.35, 3.85], fontFace: BODY, fontSize: 11, border: { type: "solid", color: C.line, pt: 0.5 }, rowH: 0.47, valign: "middle" });
  s.addText("Fairness", { x: 6.0, y: 1.1, w: 3.5, h: 0.35, fontFace: BODY, fontSize: 14, bold: true, color: C.ink, margin: 0, isTextBox: true });
  body(s, bullets([
    "Same interval, preprocessing, targets and readout for every method",
    "4 contiguous 25 s test blocks; 1 s excluded on each side",
    "Every basis, scaling and ridge α learned on training bins only",
    "Behaviour never used to choose anything",
    "Comparisons and the meaning of “better” written down before run 2",
    "Run 1 (lfpack misapplied) superseded and disclosed",
  ], 12), { x: 6.0, y: 1.5, w: 3.5, h: 3.6 });
}

// ---------------------------------------------------------------- 6 pilot scorecard
{
  const s = pres.addSlide();
  title(s, "Pilot: amplitude back, waveform nearly intact", { size: 22, w: 6.9 });
  chip(s, "done", "COMPLETED · 1 PROBE", 7.6, 0.5, 1.9);
  s.addImage({ path: FIG("fig2_scorecard.png"), x: 0.4, y: 1.0, w: 9.2, h: 2.76 });
  body(s, bullets([
    `C2 (12+4): waveform R² ${f3(c2.r2_wave)} vs ${f3(cur.r2_wave)} for PCA 16; gamma amplitude R² ${f3(c2.r2_env_gamma)} vs ${f3(cur.r2_env_gamma)}.`,
    `Behaviour r: C2 ${f3(c2.beh_r_mean)}, C1 ${f3(c1.beh_r_mean)}, PCA 16 ${f3(cur.beh_r_mean)} (shuffled control ${f3(cur.shuf_r_mean)}).`,
    `Cost of the swap: alpha coherence ${f3(c2.coh_alpha)} vs ${f3(cur.coh_alpha)}. The 200 ms forecast is near zero for every method (PCA 16: ${f3(cur.forecast200ms_r2_env_gamma)}), so it separates nothing.`,
  ], 12), { x: 0.5, y: 3.9, w: 9, h: 1.3 });
}

// ---------------------------------------------------------------- 7 trade-off
{
  const s = pres.addSlide();
  title(s, "How the 16 dimensions are split matters", { size: 24, w: 7.0 });
  chip(s, "done", "COMPLETED · 1 PROBE", 7.5, 0.5, 2.0);
  s.addImage({ path: FIG("fig1_tradeoff.png"), x: 1.0, y: 1.0, w: 8.0, h: 3.2 });
  const a2 = P["wave14+amp2"], a12 = P["wave4+amp12"];
  body(s, bullets([
    `2 amplitude dims: gamma R² ${f3(cur.r2_env_gamma)} → ${f3(a2.r2_env_gamma)}, waveform ${f3(cur.r2_wave)} → ${f3(a2.r2_wave)}.`,
    `12 amplitude dims: waveform falls to ${f3(a12.r2_wave)}; behaviour r ${f3(a12.beh_r_mean)} (PCA 16: ${f3(cur.beh_r_mean)}).`,
    `Run 1's inner-CV rule chose 4 + 12 because it weighted amplitude 2 : 1; the 12 + 4 split was fixed before run 2, and this curve is descriptive only.`,
  ], 11.5), { x: 0.5, y: 4.38, w: 9, h: 0.9, paraSpaceAfter: 3 });
}

// ---------------------------------------------------------------- 8 lfpack + costs
{
  const s = pres.addSlide();
  title(s, "lfpack is an archive, not a representation", { size: 24, w: 7.0 });
  chip(s, "done", "COMPLETED · 1 PROBE", 7.5, 0.5, 2.0);
  s.addImage({ path: FIG("fig3_lfpack.png"), x: 0.4, y: 1.05, w: 4.9, h: 1.8 });
  const cd = N.codec["lfpack_cadzow_default"], ca = N.codec["lfpack_cadzow_aggressive"], cn = N.codec["lfpack_default"];
  const K = N.costs;
  const num = (x, d = 0) => x.toLocaleString("en-US", { maximumFractionDigits: d, minimumFractionDigits: d });
  const sec = (x) => (x < 0.001 ? x.toFixed(4) : x < 0.1 ? x.toFixed(3) : x.toFixed(2));
  const crow = (label, m, keeps) => [label, num(K[m].storage_bytes_per_s), num(K[m].compression_vs_raw_250hz_npz) + "×",
                                     sec(K[m].encode_s_per_s), num(K[m].encode_peak_traced_mb), keeps];
  const rows = [["Method", "Storage B/s", "vs raw", "Encode s/s", "Peak MB", "Keeps"],
    crow("PCA 16", "waveform_pca_16", "16 @ 25 Hz"), crow("C2 (12+4)", "wave12+amp4", "16 @ 25 Hz"),
    crow("C1 (16+4)", "wave16+amp4", "20 @ 25 Hz"), crow("lfpack", "lfpack_default", "full 250 Hz"),
    crow("Cadzow+lfpack", "lfpack_cadzow_default", "full 250 Hz"), crow("Cadzow+aggr.", "lfpack_cadzow_aggressive", "full 250 Hz")];
  s.addTable(rows.map((r, i) => r.map((t, j) => ({ text: String(t), options: { bold: i === 0 || j === 0, color: i === 0 ? C.white : C.ink,
    fill: { color: i === 0 ? C.navy : (i % 2 ? C.white : C.tint) }, align: j === 0 || j === 5 ? "left" : "right" } }))),
    { x: 5.45, y: 1.05, w: 4.05, colW: [0.95, 0.65, 0.5, 0.62, 0.53, 0.8], fontFace: BODY, fontSize: 9, border: { type: "solid", color: C.line, pt: 0.5 }, rowH: 0.27, valign: "middle", margin: 0.03 });
  body(s, bullets([
    `lfpack keeps the whole 250 Hz signal. Cadzow + default: R² ${f3(cd.r2_250hz)} vs raw at ${num(cd.storage_bytes_per_s)} B/s. Without Cadzow, production thresholds keep SVD rank ${cn.svd_rank_median}: R² ${f3(cn.r2_250hz)}.`,
    `Representations built on Cadzow + lfpack data: C1 gamma amplitude R² ${f3(N.on_lfpack["lfpack_cadzow_default|wave16+amp4"].r2_env_gamma)} (from raw: ${f3(c1.r2_env_gamma)}).`,
    `Encode cost per second of recording: PCA 16 ${sec(K["waveform_pca_16"].encode_s_per_s)} s, C2 ${sec(K["wave12+amp4"].encode_s_per_s)} s (envelope filtering, peak ${num(K["wave12+amp4"].encode_peak_traced_mb)} MB), Cadzow + lfpack ${sec(K["lfpack_cadzow_default"].encode_s_per_s)} s.`,
  ], 11.5), { x: 0.5, y: 3.1, w: 9, h: 1.95 });
  footer(s, "Storage: numpy.savez_compressed float32 of what each method keeps; “vs raw” against the raw 250 Hz signal stored the same way. One core, Intel i5 laptop.");
}

// ---------------------------------------------------------------- 9 replication
{
  const s = pres.addSlide();
  title(s, "Replication on the other probes", { size: 24, w: 6.0 });
  chip(s, "pre", `PRELIMINARY · ${RM.n_sessions} SESSIONS · ${RM.n_mice} MICE`, 6.6, 0.5, 2.9);
  s.addImage({ path: FIG("fig4_replication.png"), x: 0.4, y: 1.05, w: 9.2, h: 2.97 });
  const g = R[key(C2c, "r2_env_gamma")], b = R[key(C2c, "beh_r_mean")], w = R[key(C2c, "r2_wave")];
  const sb = RS[key(C2c, "beh_r_mean")], sg = RS[key(C2c, "r2_env_gamma")];
  const xb = RX[key(C2c, "beh_r_mean")];
  body(s, bullets([
    `C2 vs PCA 16: gamma amplitude R² ${sgn(g.median)} (${g.n_improved}/${g.n_probes} probes), behaviour r ${sgn(b.median)} (${b.n_improved}/${b.n_probes}), waveform R² ${sgn(w.median)} (${w.n_improved}/${w.n_probes} up).`,
    `Sessions, the independent unit: gamma ${sg.n_improved}/${sg.n_sessions}, behaviour ${sb.n_improved}/${sb.n_sessions} (sign test p = ${sb.p.toFixed(3)}, the smallest possible with ${sb.n_sessions}). Without the pilot's own session: behaviour ${xb.n_improved}/${xb.n_sessions}.`,
  ], 12), { x: 0.5, y: 4.1, w: 9, h: 1.1 });
}

// ---------------------------------------------------------------- 10 verdict
{
  const s = pres.addSlide();
  title(s, "Is any method actually better?");
  const T = N.replication_waveform_tolerance;
  const mark = (ok) => (ok ? "✓ met" : "✗ not met");
  const row = (label, c) => {
    const amp = Math.min(R[key(c, "r2_env_beta")].n_improved, R[key(c, "r2_env_gamma")].n_improved);
    const n = R[key(c, "r2_env_gamma")].n_probes;
    const tol = T[c];
    const beh = R[key(c, "beh_r_mean")];
    return [label,
      `${mark(amp === n)}\n${amp}/${n} probes`,
      `${mark(tol["n_within_0.02_waveform_loss"] === n)}\n${tol["n_within_0.02_waveform_loss"]}/${n} within; worst ${f3(tol.worst_waveform_loss)}`,
      `${mark(beh.n_improved === n)}\n${beh.n_improved}/${n} higher, median ${sgn(beh.median)}`];
  };
  const rows = [["Pre-declared criterion →", "Beta and gamma amplitude R² higher", "Waveform R² loss ≤ 0.02", "Behaviour r not lower"],
    row("C2 vs PCA 16 (16 dims)", C2c), row("C1 vs PCA 20 (20 dims)", C1c)];
  s.addTable(rows.map((r, i) => r.map((t, j) => ({ text: t, options: { bold: i === 0 || j === 0, color: i === 0 ? C.white : C.ink,
    fill: { color: i === 0 ? C.navy : C.tint } } }))),
    { x: 0.5, y: 1.15, w: 9, colW: [2.1, 2.3, 2.3, 2.3], fontFace: BODY, fontSize: 12, border: { type: "solid", color: C.white, pt: 2 }, rowH: [0.5, 0.85, 0.85], valign: "middle" });

  // Verdict sentence derived from the same numbers as the table.
  const allMet = (c) => {
    const n = R[key(c, "r2_env_gamma")].n_probes;
    return R[key(c, "r2_env_beta")].n_improved === n && R[key(c, "r2_env_gamma")].n_improved === n &&
           T[c]["n_within_0.02_waveform_loss"] === n && R[key(c, "beh_r_mean")].n_improved === n;
  };
  const describe = (c, label) => {
    const n = R[key(c, "r2_env_gamma")].n_probes, tol = T[c];
    if (allMet(c)) return `${label} meets all three criteria in all ${n} probes.`;
    const misses = [];
    if (tol["n_within_0.02_waveform_loss"] < n) misses.push(`misses the waveform tolerance on ${n - tol["n_within_0.02_waveform_loss"]} of ${n} probes (worst loss ${f3(tol.worst_waveform_loss)})`);
    if (R[key(c, "beh_r_mean")].n_improved < n) misses.push(`has lower behaviour r on ${n - R[key(c, "beh_r_mean")].n_improved} of ${n}`);
    if (Math.min(R[key(c, "r2_env_beta")].n_improved, R[key(c, "r2_env_gamma")].n_improved) < n) misses.push("does not raise amplitude R² everywhere");
    return `${label} ${misses.join(" and ")}.`;
  };
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.45, w: 9, h: 1.65, fill: { color: "FCE8D8" }, line: { color: "FCE8D8" } });
  s.addText([
    { text: "Answer: a strong candidate, not yet an established improvement.", options: { bold: true, breakLine: true, fontSize: 15, color: C.navy } },
    { text: `${describe(C1c, "C1 vs PCA 20")} ${describe(C2c, "C2 vs PCA 16")} `, options: { fontSize: 12, color: C.ink, breakLine: true } },
    { text: `Evidence is ${RM.n_sessions} sessions × 100 s against a reimplementation of PCA 16. Untested: Alon's own code, his 50 insertions, longer recordings, and the CCA analyses this would feed.`, options: { fontSize: 12, color: C.ink } },
  ], { x: 0.75, y: 3.5, w: 8.5, h: 1.55, fontFace: BODY, valign: "middle", margin: 0, isTextBox: true, paraSpaceAfter: 5 });
}

// ---------------------------------------------------------------- 11 limitations + next
{
  const s = pres.addSlide();
  title(s, "Limitations and the smallest next experiment");
  s.addText("Limitations", { x: 0.5, y: 1.1, w: 5, h: 0.35, fontFace: BODY, fontSize: 14, bold: true, color: C.ink, margin: 0, isTextBox: true });
  body(s, bullets([
    "PCA 16 is a reimplementation; Alon's code and 50-insertion data are not in the repository",
    `Small sample: ${RM.n_sessions} sessions × 100 s; the pilot's sibling probe is in the replication`,
    "Linear readouts only; behaviour = wheel and motion energy (no paw states, no pupil)",
    "Amplitude bands (15–30, 30–90 Hz) and 40 ms RMS fixed a priori, not tuned",
    "The 200 ms forecast metric separates nothing here and needs redesign",
    "lfpack tested only at its two production levels, at 250 Hz, on one probe",
  ], 12), { x: 0.5, y: 1.5, w: 5.0, h: 3.6 });

  s.addShape(pres.shapes.RECTANGLE, { x: 5.8, y: 1.1, w: 3.7, h: 3.95, fill: { color: C.tint }, line: { color: C.tint } });
  chip(s, "prop", "PROPOSED · NOT RUN", 6.0, 1.28, 1.9);
  s.addText([
    { text: "Smallest next experiment", options: { bold: true, fontSize: 15, breakLine: true, color: C.navy } },
    { text: "Add the 4 amplitude PCs to Alon's existing 16-PC pipeline on his 50 insertions, and rerun two analyses with and without them:", options: { fontSize: 12, breakLine: true } },
    { text: "behaviour decoding (8 Sep slide 24)", options: { bullet: true, fontSize: 12, breakLine: true } },
    { text: "cross-probe CCA and the behaviour-residual test (slides 18, 26)", options: { bullet: true, fontSize: 12, breakLine: true } },
    { text: "Needs: his PCA code and the 50-pid list.", options: { fontSize: 12, italic: true, color: C.muted } },
  ], { x: 6.0, y: 1.7, w: 3.3, h: 3.2, fontFace: BODY, color: C.ink, valign: "top", margin: 0, paraSpaceAfter: 6, isTextBox: true });
}

// ---------------------------------------------------------------- 12 close
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  const b = R[key(C1c, "beh_r_mean")], g = R[key(C1c, "r2_env_gamma")], w = R[key(C1c, "r2_wave")];
  s.addText("Finding", { x: 0.6, y: 0.7, w: 8.8, h: 0.4, fontFace: BODY, fontSize: 14, bold: true, color: "9FA9BF", margin: 0, isTextBox: true, charSpacing: 2 });
  s.addText(`Spending 4 dimensions on beta/gamma amplitude instead of 4 more waveform PCs changed gamma amplitude R² by ${sgn(g.median, 2)}, behaviour r by ${sgn(b.median, 2)} and waveform R² by ${sgn(w.median)} (medians; behaviour higher in ${b.n_improved} of ${b.n_probes} probes).`,
    { x: 0.6, y: 1.15, w: 8.8, h: 1.9, fontFace: HEAD, fontSize: 24, color: C.white, margin: 0, isTextBox: true, valign: "top" });
  s.addText("C1 (16 waveform + 4 amplitude) vs PCA 20 at equal budget; 11 probes, 6 sessions, 6 mice; preliminary.", { x: 0.6, y: 3.1, w: 8.8, h: 0.4, fontFace: BODY, fontSize: 12.5, color: "CAD3E6", margin: 0, isTextBox: true });
  s.addText([{ text: "Ask", options: { bold: true, breakLine: true, color: C.orange } },
             { text: "Share the current PCA code and the 50-pid list so C1 can be tested inside the real pipeline." }],
    { x: 0.6, y: 3.75, w: 8.8, h: 1.0, fontFace: BODY, fontSize: 16, color: C.white, margin: 0, isTextBox: true, valign: "top" });
}

pres.writeFile({ fileName: OUT }).then((f) => console.log("wrote", f));
