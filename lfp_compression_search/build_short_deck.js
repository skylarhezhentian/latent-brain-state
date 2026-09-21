/*
 * Short deck (8 slides): search for a representation better than PCA, and its confirmation.
 *
 *   node build_short_deck.js <results_dir> <out.pptx>      (needs pptxgenjs@3.12.0)
 *
 * Reads <results_dir>/search/search_numbers.json (search + confirmation + costs) and
 * <results_dir>/deck_numbers.json (pilot: trade-off curve, lfpack, numbers quoted from
 * Alon's slides). Every result number and every result-dependent word comes from them.
 */
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const RES = process.argv[2], OUT = process.argv[3];
const S = JSON.parse(fs.readFileSync(path.join(RES, "search", "search_numbers.json"), "utf8"));
const D = JSON.parse(fs.readFileSync(path.join(RES, "deck_numbers.json"), "utf8"));
const FIG = (p) => path.join(RES, p);

const C = { navy: "14213D", ink: "1B1B1F", muted: "5A5F6B", tint: "F2F4F8", line: "D9DDE5", white: "FFFFFF",
            blue: "2A78D6", orange: "EB6834", aqua: "1BAF7A",
            okFill: "DDEEDF", okInk: "1E5B2A", preFill: "FCE8D8", preInk: "8A3B12", propFill: "E3E6EC", propInk: "3A3F4B" };
const HEAD = "Cambria", BODY = "Calibri";
const f2 = (x) => x.toFixed(2), f3 = (x) => x.toFixed(3);
const sgn = (x, d = 3) => (x >= 0 ? "+" : "−") + Math.abs(x).toFixed(d);
const num = (x) => Math.round(x).toLocaleString("en-US");

const W = S.winner.winner;                       // pre-selected on the search set
const V = S.verdict;
const sp = S.search.probe_summary, cp = S.confirm.probe_summary, cs = S.confirm.session_summary;
const NAME = { "wave12+amp4_smooth": "12 wave + 4 amp (200 ms)", "wave12+amp4": "12 wave + 4 amp (40 ms)" };

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.author = "Skylar Tian";
pres.title = "Amplitude-augmented PCA: a better 16-number LFP representation";

function title(s, text, opts = {}) {
  s.addText(text, { x: 0.5, y: 0.3, w: opts.w || 9, h: 0.7, fontFace: HEAD, fontSize: opts.size || 26, bold: true,
                    color: C.ink, margin: 0, isTextBox: true, valign: "middle" });
}
function chip(s, kind, text, x, y, w) {
  const fill = { done: C.okFill, pre: C.preFill, prop: C.propFill }[kind], ink = { done: C.okInk, pre: C.preInk, prop: C.propInk }[kind];
  s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.3, fill: { color: fill }, line: { color: fill }, rectRadius: 0.08 });
  s.addText(text, { x, y, w, h: 0.3, fontFace: BODY, fontSize: 10, bold: true, color: ink, align: "center", valign: "middle", margin: 0, isTextBox: true, charSpacing: 1 });
}
function bullets(items, size) {
  return items.map((t, i) => ({ text: t, options: { bullet: true, breakLine: i < items.length - 1, fontSize: size } }));
}
function body(s, runs, box) {
  s.addText(runs, { fontFace: BODY, fontSize: 13, color: C.ink, valign: "top", margin: 0, isTextBox: true, paraSpaceAfter: 7, ...box });
}
function footer(s, text) {
  s.addText(text, { x: 0.5, y: 5.2, w: 9, h: 0.3, fontFace: BODY, fontSize: 9.5, color: C.muted, margin: 0, isTextBox: true });
}

// ------------------------------------------------------------ 1 title
{
  const s = pres.addSlide();
  s.background = { color: C.navy };
  s.addText("Amplitude-augmented PCA", { x: 0.6, y: 1.0, w: 8.8, h: 0.9, fontFace: HEAD, fontSize: 40, bold: true, color: C.white, margin: 0, isTextBox: true, valign: "bottom" });
  s.addText("A 16-number LFP representation that keeps more behavioural information than PCA", { x: 0.6, y: 1.95, w: 8.8, h: 0.9, fontFace: HEAD, fontSize: 22, color: "CAD3E6", margin: 0, isTextBox: true, valign: "top" });
  s.addText("Skylar Tian  ·  update for Alon Saguy  ·  14 September 2026", { x: 0.6, y: 3.0, w: 8.8, h: 0.4, fontFace: BODY, fontSize: 13, color: "9FA9BF", margin: 0, isTextBox: true });
  chip(s, "done", `SEARCH · ${S.n_methods} METHODS · ${S.search.n_probes} PROBES · ${S.search.n_mice} MICE`, 0.6, 4.2, 4.1);
  chip(s, V.confirmed ? "done" : "pre", `CONFIRMATION · ${S.confirm.n_mice} NEW MICE · ${V.confirmed ? "CONFIRMED" : "NOT CONFIRMED"}`, 4.85, 4.2, 4.1);
  s.addNotes("Two chips: the search picked a winner by a rule fixed in advance; the confirmation tested only that winner on six mice never used before.");
}

// ------------------------------------------------------------ 2 question, and why 16
{
  const s = pres.addSlide();
  title(s, "The question, and why 16 dimensions");
  body(s, bullets([
    "Current representation: 16 spatial PCs of the signed waveform at 25 Hz (Alon, 8 Sep). Averaging to 25 Hz keeps only content below 12.5 Hz.",
    `Alon's 8 Sep result: behaviour lives mainly in LFP amplitude (RMS envelope r = ${f2(D.quoted_from_alon_slides.rms_envelope_behaviour_r_8sep)} vs signed PCA r = ${f2(D.quoted_from_alon_slides.signed_pca_behaviour_r_8sep)}).`,
    "16 is the budget of the current method, not an optimum. Every method gets the same 16 numbers per 40 ms, so a win reflects how the numbers are spent, not having more of them.",
  ], 13), { x: 0.5, y: 1.15, w: 4.85, h: 2.9 });

  const bands = [["delta", "1–4 Hz", C.blue], ["theta", "4–8", C.blue], ["alpha", "8–12", C.blue], ["beta", "15–30", C.orange], ["gamma", "30–90", C.orange]];
  const x0 = 5.6, w = 0.76, y = 1.55;
  s.addText("What a 25 Hz signed waveform can carry", { x: x0, y: 1.12, w: 4, h: 0.33, fontFace: BODY, fontSize: 12, bold: true, color: C.ink, margin: 0, isTextBox: true });
  bands.forEach(([n, hz, col], i) => {
    s.addShape(pres.shapes.RECTANGLE, { x: x0 + i * (w + 0.03), y, w, h: 0.6, fill: { color: col }, line: { color: col } });
    s.addText([{ text: n, options: { bold: true, breakLine: true } }, { text: hz }], { x: x0 + i * (w + 0.03), y, w, h: 0.6, fontFace: BODY, fontSize: 10.5, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
  });
  s.addText("kept as waveform", { x: x0, y: y + 0.66, w: 3 * (w + 0.03), h: 0.28, fontFace: BODY, fontSize: 10.5, color: C.blue, bold: true, align: "center", margin: 0, isTextBox: true });
  s.addText("lost unless kept as amplitude", { x: x0 + 3 * (w + 0.03), y: y + 0.66, w: 2 * (w + 0.03), h: 0.5, fontFace: BODY, fontSize: 10.5, color: C.orange, bold: true, align: "center", valign: "top", margin: 0, isTextBox: true });

  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 4.15, w: 9, h: 0.9, fill: { color: C.tint }, line: { color: C.tint } });
  s.addText("With the same 16 numbers, can we keep more behavioural information than PCA without losing waveform fidelity?",
    { x: 0.75, y: 4.15, w: 8.5, h: 0.9, fontFace: HEAD, fontSize: 17, italic: true, color: C.navy, valign: "middle", margin: 0, isTextBox: true });
  s.addNotes("PCA 16 is reimplemented from Alon's slide description: his code and 50-insertion data are not in the repository.");
}

// ------------------------------------------------------------ 3 how it was tested
{
  const s = pres.addSlide();
  title(s, "How it was tested: search, then confirm");
  const box = (x, y, w, h, head, sub, fill, ink = C.ink) => {
    s.addShape(pres.shapes.RECTANGLE, { x, y, w, h, fill: { color: fill }, line: { color: fill } });
    s.addText([{ text: head, options: { bold: true, breakLine: true, fontSize: 12.5 } }, { text: sub, options: { fontSize: 10.5 } }],
      { x: x + 0.1, y, w: w - 0.2, h, fontFace: BODY, color: ink, align: "center", valign: "middle", margin: 0, isTextBox: true });
  };
  const arrow = (x1, x2, y) => s.addShape(pres.shapes.LINE, { x: x1, y, w: x2 - x1, h: 0, line: { color: C.muted, width: 1.5, endArrowType: "triangle" } });
  const y = 1.3, h = 1.15;
  box(0.5, y, 1.95, h, "Search set", `${S.search.n_probes} probes · ${S.search.n_sessions} sessions · ${S.search.n_mice} mice`, C.tint);
  arrow(2.45, 2.7, y + h / 2);
  box(2.7, y, 1.95, h, `${S.n_methods} methods`, "all 16 dims at 25 Hz, PCA 16 as baseline", C.tint);
  arrow(4.65, 4.9, y + h / 2);
  box(4.9, y, 1.95, h, "Fixed rule picks one", "best behaviour gain, waveform R² loss ≤ 0.02", "FCE8D8");
  arrow(6.85, 7.1, y + h / 2);
  box(7.1, y, 2.4, h, "Confirmation set", `${S.confirm.n_probes} probes · ${S.confirm.n_mice} new mice (overlap: ${S.mice_overlap})`, C.navy, C.white);

  s.addText("Fairness controls", { x: 0.5, y: 2.75, w: 4.3, h: 0.35, fontFace: BODY, fontSize: 14, bold: true, color: C.ink, margin: 0, isTextBox: true });
  body(s, bullets([
    "Same 100 s, preprocessing, targets and ridge readout for every method",
    "4 contiguous 25 s test blocks, 1 s gaps; everything fit on training bins only",
    "Behaviour never used to fit or tune anything within a probe",
  ], 12), { x: 0.5, y: 3.12, w: 4.3, h: 2.0 });
  s.addText("Written down before any result", { x: 5.1, y: 2.75, w: 4.4, h: 0.35, fontFace: BODY, fontSize: 14, bold: true, color: C.ink, margin: 0, isTextBox: true });
  body(s, bullets([
    "Methods, selection rule and confirmation rule (SEARCH_PROTOCOL.md)",
    "Confirmed only if behaviour r is higher in ≥ 5 of 6 sessions and median waveform-R² loss ≤ 0.02",
    "30 s-shifted behaviour as a chance control",
  ], 12), { x: 5.1, y: 3.12, w: 4.4, h: 2.0 });
}

// ------------------------------------------------------------ 4 search result
{
  const s = pres.addSlide();
  title(s, "Search: amplitude-augmented PCA wins", { size: 24, w: 6.9 });
  chip(s, "done", `SEARCH SET · ${S.search.n_mice} MICE`, 7.6, 0.5, 1.9);
  s.addImage({ path: FIG("search/figures/fig_search_ranking.png"), x: 0.5, y: 1.0, w: 8.4, h: 3.5 });
  const w = sp[W], ae = sp["autoencoder_16"], jp = sp["joint_pca_16"], st = sp["spatiotemporal_pca_16"];
  body(s, bullets([
    `Winner by the fixed rule: 12 waveform PCs + 4 amplitude PCs over 200 ms, behaviour r ${sgn(w.median_d_beh_r)} and waveform R² ${sgn(w.median_d_wave_r2)} vs PCA 16.`,
    `Autoencoder and joint PCA also gain behaviour but lose too much waveform (R² ${sgn(ae.median_d_wave_r2, 2)}, ${sgn(jp.median_d_wave_r2, 2)}); spatiotemporal PCA is worse than PCA (${sgn(st.median_d_beh_r)}).`,
  ], 11.5), { x: 0.5, y: 4.58, w: 9, h: 0.8, paraSpaceAfter: 3 });
}

// ------------------------------------------------------------ 5 confirmation
{
  const s = pres.addSlide();
  title(s, "Confirmation on six new mice", { size: 24, w: 6.5 });
  chip(s, V.confirmed ? "done" : "pre", V.confirmed ? "CONFIRMED · PRE-DECLARED RULE" : "NOT CONFIRMED", 6.9, 0.5, 2.6);
  s.addImage({ path: FIG("search/figures/fig_confirmation.png"), x: 0.5, y: 1.05, w: 9.0, h: 2.74 });
  const c = cp[W], se = cs[W], c40 = cp["wave12+amp4"];
  const over = S.confirm["probes_wave_loss_over_0.02"][W];
  body(s, bullets([
    `Winner vs PCA 16: behaviour r ${sgn(c.median_d_beh_r)} (${c.probes_beh_up}/${c.n_probes} probes, ${se.sessions_beh_up}/${se.n_sessions} sessions); waveform R² ${sgn(c.median_d_wave_r2)}; gamma amplitude R² ${sgn(c.median_d_gamma_r2)}.`,
    `Chance control is not quite zero for amplitude methods (${f3(S.confirm.absolute_medians[W].shuf_r_mean)} vs ${f3(S.confirm.absolute_medians["waveform_pca_16"].shuf_r_mean)}); subtracting it leaves a gain of ${sgn(S.confirm_conservative_gain, 2)}.`,
    `Waveform cost is real: ${over} of ${c.n_probes} probes lose more than 0.02. The 40 ms version keeps fast amplitude better (gamma ${sgn(c40.median_d_gamma_r2, 2)}) but gains less behaviour (${sgn(c40.median_d_beh_r)}).`,
  ], 11.5), { x: 0.5, y: 3.9, w: 9, h: 1.4, paraSpaceAfter: 4 });
}

// ------------------------------------------------------------ 6 why 12 + 4, and 16
{
  const s = pres.addSlide();
  title(s, "Why 12 + 4, and what 16 does not tell us", { size: 24, w: 7.0 });
  chip(s, "done", "PILOT PROBE · 40 ms", 7.6, 0.5, 1.9);
  s.addImage({ path: FIG("figures/fig1_tradeoff_plain.png"), x: 0.5, y: 1.05, w: 5.6, h: 2.24 });
  const P = D.raw, a2 = P["wave14+amp2"], a12 = P["wave4+amp12"], cur = P["waveform_pca_16"];
  body(s, bullets([
    `First amplitude dims are cheap: 2 of them take gamma R² from ${f3(cur.r2_env_gamma)} to ${f3(a2.r2_env_gamma)} while waveform drops ${f3(cur.r2_wave)} → ${f3(a2.r2_wave)}.`,
    `Too many are not: with 12, waveform falls to ${f3(a12.r2_wave)} and behaviour r to ${f3(a12.beh_r_mean)}.`,
    "12 + 4 was fixed before testing, not tuned to this curve.",
  ], 11.5), { x: 6.3, y: 1.1, w: 3.2, h: 2.3 });
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 3.55, w: 9, h: 1.45, fill: { color: C.tint }, line: { color: C.tint } });
  s.addText([
    { text: "About the 16", options: { bold: true, breakLine: true, fontSize: 13, color: C.navy } },
    { text: `16 = Alon's current budget (8 Sep). On 31 Aug, 8–12 PCA dims already gave R² ≈ ${D.quoted_from_alon_slides.pca_r2_8_to_12_dims_31aug}, so 16 is a practical choice, not a derived optimum. The winner was tested only at 16; whether it still beats PCA at 8, 24 or 32 is untested.`, options: { fontSize: 12 } },
  ], { x: 0.75, y: 3.6, w: 8.5, h: 1.35, fontFace: BODY, color: C.ink, valign: "middle", margin: 0, isTextBox: true, paraSpaceAfter: 4 });
}

// ------------------------------------------------------------ 7 costs + lfpack
{
  const s = pres.addSlide();
  title(s, "Costs: same storage as PCA, a little more compute");
  const K = S.costs, L = D.costs, cod = D.codec;
  const sec = (x) => (x < 0.001 ? x.toFixed(4) : x < 0.1 ? x.toFixed(4) : x.toFixed(2));
  const rep = (label, k, keeps, fid) => [label, num(k.storage_bytes_per_s), num(k.compression_vs_raw_250hz_npz) + "×", sec(k.encode_s_per_s), num(k.encode_peak_traced_mb), keeps, fid];
  const rows = [["Method", "Storage B/s", "vs raw", "Encode s/s", "Peak MB", "Keeps", "Note"],
    rep("PCA 16 (current)", K["waveform_pca_16"], "16 dims, 25 Hz", "baseline"),
    rep("Winner: 12 + 4 (200 ms)", K[W], "16 dims, 25 Hz", `behaviour r ${sgn(cp[W].median_d_beh_r, 2)}`),
    rep("12 + 4 (40 ms)", K["wave12+amp4"], "16 dims, 25 Hz", `behaviour r ${sgn(cp["wave12+amp4"].median_d_beh_r, 2)}`),
    rep("Cadzow + lfpack", L["lfpack_cadzow_default"], "full 250 Hz", `R² ${f3(cod.lfpack_cadzow_default.r2_250hz)} vs raw`),
    rep("lfpack, no Cadzow", L["lfpack_default"], "full 250 Hz", `R² ${f3(cod.lfpack_default.r2_250hz)} vs raw`)];
  s.addTable(rows.map((r, i) => r.map((t, j) => ({ text: String(t), options: { bold: i === 0 || j === 0, color: i === 0 ? C.white : C.ink,
    fill: { color: i === 0 ? C.navy : (i === 2 ? "FCE8D8" : (i % 2 ? C.white : C.tint)) }, align: j === 0 || j >= 5 ? "left" : "right" } }))),
    { x: 0.5, y: 1.15, w: 9, colW: [1.95, 1.05, 0.75, 1.0, 0.85, 1.35, 2.05], fontFace: BODY, fontSize: 11, border: { type: "solid", color: C.line, pt: 0.5 }, rowH: 0.4, valign: "middle" });
  body(s, bullets([
    `The winner stores exactly what PCA 16 stores: 16 floats at 25 Hz (${num(K[W].storage_bytes_per_s)} B/s, ${num(K[W].compression_vs_raw_250hz_npz)}× smaller than raw).`,
    `The extra cost is band-pass filtering: ${sec(K[W].encode_s_per_s)} s of compute per second of recording and ${num(K[W].encode_peak_traced_mb)} MB peak, vs ${sec(K["waveform_pca_16"].encode_s_per_s)} s for PCA. No training.`,
    "lfpack is an archive of the full signal, not a compact representation: a different job.",
  ], 12), { x: 0.5, y: 3.75, w: 9, h: 1.4, paraSpaceAfter: 4 });
  footer(s, "Pilot probe, 100 s, one core of an Intel i5 laptop. Storage = numpy.savez_compressed float32.");
}

// ------------------------------------------------------------ 8 verdict + next
{
  const s = pres.addSlide();
  title(s, "Verdict, caveats and next steps");
  s.addShape(pres.shapes.RECTANGLE, { x: 0.5, y: 1.1, w: 9, h: 1.05, fill: { color: "FCE8D8" }, line: { color: "FCE8D8" } });
  const c = cp[W], se = cs[W];
  s.addText([
    { text: V.confirmed ? "Better than PCA on behaviour, at the same budget — confirmed on this data." : "Not confirmed on the new mice.", options: { bold: true, fontSize: 15, color: C.navy, breakLine: true } },
    { text: `Behaviour r ${sgn(c.median_d_beh_r)} in ${se.sessions_beh_up}/${se.n_sessions} new sessions, for waveform R² ${sgn(c.median_d_wave_r2)}. PCA still reconstructs the raw waveform slightly better.`, options: { fontSize: 12.5, color: C.ink } },
  ], { x: 0.75, y: 1.1, w: 8.5, h: 1.05, fontFace: BODY, valign: "middle", margin: 0, isTextBox: true, paraSpaceAfter: 4 });

  s.addText("Caveats", { x: 0.5, y: 2.35, w: 4.3, h: 0.35, fontFace: BODY, fontSize: 14, bold: true, color: C.ink, margin: 0, isTextBox: true });
  body(s, bullets([
    "PCA 16 reimplemented from slides; Alon's code and 50 insertions not tested",
    `${S.search.n_mice + S.confirm.n_mice} mice, 100 s each; linear readouts only`,
    `Waveform loss near the 0.02 limit (${S.confirm["probes_wave_loss_over_0.02"][W]}/${c.n_probes} probes over)`,
    "Only the 16-dim budget tested",
    `Chance control slightly above zero; conservative behaviour gain ${sgn(S.confirm_conservative_gain, 2)}`,
  ], 11.5), { x: 0.5, y: 2.72, w: 4.3, h: 2.4 });

  s.addShape(pres.shapes.RECTANGLE, { x: 5.1, y: 2.35, w: 4.4, h: 2.75, fill: { color: C.tint }, line: { color: C.tint } });
  chip(s, "prop", "PROPOSED · NOT RUN", 5.3, 2.5, 1.9);
  s.addText([
    { text: "Next", options: { bold: true, fontSize: 14, color: C.navy, breakLine: true } },
    { text: "Budget sweep: 8, 12, 16, 24, 32 dims, PCA vs 3:1 waveform + amplitude (~15 min)", options: { bullet: true, fontSize: 11.5, breakLine: true } },
    { text: "Add the 4 amplitude PCs to Alon's pipeline; rerun behaviour decoding and cross-probe CCA", options: { bullet: true, fontSize: 11.5, breakLine: true } },
    { text: "Ask: the current PCA code and the 50-pid list", options: { bold: true, fontSize: 11.5, color: C.orange } },
  ], { x: 5.3, y: 2.9, w: 4.0, h: 2.1, fontFace: BODY, color: C.ink, valign: "top", margin: 0, isTextBox: true, paraSpaceAfter: 5 });
}

pres.writeFile({ fileName: OUT }).then((f) => console.log("wrote", f));
