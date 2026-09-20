// Builds the IE 643 prep presentation.  node build_deck.js   (reads numbers.json + ../results/figures)
const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");
const React = require("react");
const ReactDOMServer = require("react-dom/server");
const sharp = require("sharp");
const fa = require("react-icons/fa");

// ------------------------------------------------------------------ team details (EDIT THESE)
const TEAM = "[Team Name]";
const MEMBERS = [
  ["[Member 1 Name]", "[Roll No.]"],
  ["[Member 2 Name]", "[Roll No.]"],
  ["[Member 3 Name]", "[Roll No.]"],
  ["[Member 4 Name]", "[Roll No.]"],
];

const HERE = __dirname;
const FIG = path.join(HERE, "..", "results", "figures");
const A = (f) => path.join(HERE, "assets", f);
const N = JSON.parse(fs.readFileSync(path.join(HERE, "numbers.json"), "utf8"));

const C = {
  basalt: "16232A", ink: "1B2B31", slate: "4F5F65", muted: "7C8B90", glacier: "E6F0EE", panel: "F1F6F5",
  teal: "2F7F73", tealLt: "6FB3A6", saffron: "D98A1E", saffronLt: "FBEBD3", white: "FFFFFF", line: "CFDAD8",
  rose: "B5473A",
};
const H = "Cambria", B = "Calibri";
const DS = ["climate", "energy", "finance"];
const DSN = { climate: "Climate", energy: "Energy", finance: "Finance" };

const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const R = (ds, m, col) => (N.results[ds] && N.results[ds][m] ? N.results[ds][m][col] : null);
const pct = (v, d = 0) => (v === null || v === undefined ? "—" : `${(100 * v).toFixed(d)}%`);

function pngSize(p) {
  const b = fs.readFileSync(p);
  return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) };
}
function fitImage(slide, p, x, y, w, h, align = "center") {
  if (!fs.existsSync(p)) {
    slide.addShape("rect", { x, y, w, h, fill: { color: C.panel }, line: { color: C.line, dashType: "dash" } });
    slide.addText(`figure pending: ${path.basename(p)}`, { x, y, w, h, align: "center", fontFace: B, fontSize: 12, color: C.muted, isTextBox: true });
    return;
  }
  const s = pngSize(p), r = s.w / s.h;
  let W = w, Hh = w / r;
  if (Hh > h) { Hh = h; W = h * r; }
  const dx = align === "left" ? 0 : (w - W) / 2;
  slide.addImage({ path: p, x: x + dx, y: y + (h - Hh) / 2, w: W, h: Hh });
}

async function icon(name, color, size = 256) {
  const svg = ReactDOMServer.renderToStaticMarkup(React.createElement(fa[name], { color: "#" + color, size: String(size) }));
  const buf = await sharp(Buffer.from(svg)).png().toBuffer();
  return "image/png;base64," + buf.toString("base64");
}

async function main() {
  const pres = new pptxgen();
  pres.layout = "LAYOUT_WIDE"; // 13.333 x 7.5
  pres.title = "Unsupervised Domain Adaptation for Time-Series Forecasting - IE 643";
  pres.author = TEAM;

  const IC = {};
  for (const [k, n, c] of [
    ["mountain", "FaMountain", C.white], ["city", "FaCity", C.white], ["chart", "FaChartLine", C.white],
    ["bolt", "FaBolt", C.white], ["coins", "FaCoins", C.white], ["temp", "FaThermometerHalf", C.white],
    ["layers", "FaLayerGroup", C.white], ["compress", "FaCompressArrowsAlt", C.white], ["anchor", "FaAnchor", C.white],
    ["puzzle", "FaPuzzlePiece", C.white], ["desktop", "FaDesktop", C.white], ["flask", "FaFlask", C.white],
    ["db", "FaDatabase", C.white], ["broom", "FaBroom", C.white], ["cogs", "FaCogs", C.white], ["cut", "FaCut", C.white],
    ["ruler", "FaRulerHorizontal", C.white], ["window", "FaWindowRestore", C.white], ["check", "FaCheck", C.white],
    ["hour", "FaHourglassHalf", C.white], ["book", "FaBookOpen", C.white], ["wave", "FaWaveSquare", C.white],
    ["mask", "FaEyeSlash", C.white], ["balance", "FaBalanceScale", C.white], ["upload", "FaFileUpload", C.white],
    ["lock", "FaLock", C.white],
  ]) IC[k] = await icon(n, c);

  const iconCircle = (s, key, x, y, d = 0.62, fill = C.teal) => {
    s.addShape(pres.shapes.OVAL, { x, y, w: d, h: d, fill: { color: fill }, line: { color: fill } });
    const p = d * 0.26;
    s.addImage({ data: IC[key], x: x + p, y: y + p, w: d - 2 * p, h: d - 2 * p });
  };
  const T = (s, text, o) => s.addText(text, { fontFace: B, fontSize: 14, color: C.ink, margin: 0, isTextBox: true, valign: "top", ...o });
  const card = (s, x, y, w, h, fill = C.panel) =>
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill } });
  const arrow = (s, x1, y1, x2, y2, color = C.slate, width = 1.75) =>
    s.addShape(pres.shapes.LINE, { x: Math.min(x1, x2), y: Math.min(y1, y2), w: Math.abs(x2 - x1) || 0.001, h: Math.abs(y2 - y1) || 0.001,
      flipH: x2 < x1, flipV: y2 < y1, line: { color, width, endArrowType: "triangle" } });
  const chip = (s, text, x, y, w, fill, color = C.white, fs_ = 11) => {
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h: 0.36, rectRadius: 0.18, fill: { color: fill }, line: { color: fill } });
    T(s, text, { x, y, w, h: 0.36, align: "center", valign: "middle", fontSize: fs_, bold: true, color });
  };

  function content(kicker, title) {
    const s = pres.addSlide();
    s.background = { path: A("bg_light_corner.png") };
    T(s, kicker.toUpperCase(), { x: 0.6, y: 0.34, w: 9, h: 0.3, fontSize: 11, bold: true, color: C.teal, charSpacing: 3 });
    s.addText(title, { x: 0.6, y: 0.62, w: 12.1, h: 0.72, fontFace: H, fontSize: 30, bold: true, color: C.ink, margin: 0, isTextBox: true, valign: "top" });
    T(s, `IE 643 Course Project  ·  ${TEAM}`, { x: 0.6, y: 7.02, w: 6, h: 0.28, fontSize: 9, color: C.muted });
    s.slideNumber = { x: 12.23, y: 7.0, w: 0.5, h: 0.3, fontFace: B, fontSize: 9, color: C.muted, align: "right" };
    return s;
  }

  // ================================================================ 1 TITLE
  {
    const s = pres.addSlide();
    s.background = { path: A("bg_title.png") };
    s.addShape(pres.shapes.RECTANGLE, { x: 0, y: 0, w: 8.4, h: 7.5, fill: { color: C.basalt, transparency: 12 }, line: { color: C.basalt, transparency: 100 } });
    T(s, "IE 643 COURSE PROJECT  ·  PREP PRESENTATION  ·  MID-PROJECT STATUS", { x: 0.7, y: 0.75, w: 7.5, h: 0.3, fontSize: 12, bold: true, color: C.tealLt, charSpacing: 2 });
    s.addText("Unsupervised Domain Adaptation for Forecasting Multivariate Time-Series Data", {
      x: 0.7, y: 1.2, w: 7.4, h: 2.3, fontFace: H, fontSize: 38, bold: true, color: C.white, margin: 0, isTextBox: true, valign: "top" });
    T(s, "Progress so far: problem, datasets, a first LiteTCN + CARE-DA campaign, and an interface prototype. The second half locks the method, the report and the final demo.", {
      x: 0.7, y: 3.55, w: 7.2, h: 0.95, fontSize: 15, color: C.glacier });
    T(s, `Team: ${TEAM}`, { x: 0.7, y: 4.75, w: 7, h: 0.35, fontSize: 16, bold: true, color: C.white });
    T(s, MEMBERS.map((m, i) => ({ text: `${m[0]}  —  ${m[1]}`, options: { breakLine: i < MEMBERS.length - 1 } })), {
      x: 0.7, y: 5.15, w: 7, h: 1.25, fontSize: 13, color: C.glacier, paraSpaceAfter: 3 });
    T(s, "September 2026", { x: 0.7, y: 6.65, w: 4, h: 0.3, fontSize: 11, color: C.tealLt });
    // domain motif
    chip(s, "DOMAIN 1  ·  source", 9.15, 6.55, 1.75, C.teal);
    arrow(s, 10.98, 6.73, 11.28, 6.73, C.white, 1.5);
    chip(s, "DOMAIN 2  ·  target", 11.35, 6.55, 1.75, C.saffron, C.basalt);
    s.addNotes("This is a prep / mid-project talk, not the final. First cycle: data, LiteTCN, first CARE-DA table. Second half is still open. Then the hook: a model trained in the hills does not know the plains.");
  }

  // ================================================================ 2 OUTLINE
  {
    const s = content("Roadmap of this talk", "Outline");
    const items = [
      ["Problem understanding", "What changes between domains and why forecasts break"],
      ["Background reading", "Forecasting models, domain-adaptation theory and methods"],
      ["Datasets & preparation", "Three source → target pairs; extraction and curation steps"],
      ["Lightweight forecaster", "LiteTCN: RevIN + dilated convolutions + linear path"],
      ["CARE-DA strategy & loss", "v1 design; v2 is a first follow-up, not locked"],
      ["Metrics & protocol", "Forecast accuracy and adaptation-efficiency metrics"],
      ["First experiments", "3-seed v1 campaign + seed-0 v2 probe"],
      ["Prototype & remaining work", "Interface v1, open questions, second-half plan"],
    ];
    items.forEach((it, i) => {
      const col = i < 4 ? 0 : 1, row = i % 4;
      const x = 0.6 + col * 6.2, y = 1.65 + row * 1.28;
      card(s, x, y, 5.85, 1.08);
      T(s, String(i + 1).padStart(2, "0"), { x: x + 0.25, y: y + 0.14, w: 0.9, h: 0.8, fontFace: H, fontSize: 34, bold: true, color: i % 2 ? C.saffron : C.teal, valign: "middle" });
      T(s, it[0], { x: x + 1.2, y: y + 0.17, w: 4.5, h: 0.4, fontSize: 17, bold: true });
      T(s, it[1], { x: x + 1.2, y: y + 0.55, w: 4.5, h: 0.4, fontSize: 12.5, color: C.slate });
    });
    s.addNotes("One breath per item. Frame as a first cycle plus a plan: we have data, a model and a first table, but the method is not locked and the report/demo are still mid-way.");
  }

  // ================================================================ 3 PROBLEM
  {
    const s = content("01 · Problem understanding", "Train on one domain, forecast well in another");
    // domain 1 panel
    card(s, 0.6, 1.6, 4.1, 2.9, C.glacier);
    iconCircle(s, "mountain", 0.85, 1.82, 0.62, C.teal);
    T(s, "Domain 1 — source", { x: 1.62, y: 1.85, w: 3, h: 0.3, fontSize: 12, bold: true, color: C.teal });
    T(s, "Hilly region (Shimla)", { x: 1.62, y: 2.13, w: 3, h: 0.35, fontSize: 16, bold: true });
    T(s, [
      { text: "Past window x and future y both available", options: { bullet: true, breakLine: true } },
      { text: "Train forecaster f_θ : x → ŷ", options: { bullet: true, breakLine: true } },
      { text: "Years of labeled history", options: { bullet: true } },
    ], { x: 0.85, y: 2.75, w: 3.7, h: 1.5, fontSize: 13, color: C.slate, paraSpaceAfter: 4 });
    // arrow + label
    arrow(s, 4.85, 3.05, 5.85, 3.05, C.ink, 2);
    T(s, "adapt\n(no target labels)", { x: 4.7, y: 3.2, w: 1.3, h: 0.6, fontSize: 11, color: C.slate, align: "center" });
    // domain 2 panel
    card(s, 6.0, 1.6, 4.1, 2.9, C.saffronLt);
    iconCircle(s, "city", 6.25, 1.82, 0.62, C.saffron);
    T(s, "Domain 2 — target", { x: 7.02, y: 1.85, w: 3, h: 0.3, fontSize: 12, bold: true, color: "A5650F" });
    T(s, "Plain region (Delhi)", { x: 7.02, y: 2.13, w: 3, h: 0.35, fontSize: 16, bold: true });
    T(s, [
      { text: "Only input windows x available", options: { bullet: true, breakLine: true } },
      { text: "Different level, cycles, correlations", options: { bullet: true, breakLine: true } },
      { text: "Goal: adapted f_θ' with low target error", options: { bullet: true } },
    ], { x: 6.25, y: 2.75, w: 3.7, h: 1.5, fontSize: 13, color: C.slate, paraSpaceAfter: 4 });
    // analogies
    card(s, 10.4, 1.6, 2.33, 2.9);
    T(s, "Same pattern elsewhere", { x: 10.58, y: 1.78, w: 2.0, h: 0.3, fontSize: 12, bold: true, color: C.teal });
    iconCircle(s, "coins", 10.6, 2.25, 0.46, C.slate);
    T(s, "IT stock → Pharma stock", { x: 11.15, y: 2.25, w: 1.5, h: 0.5, fontSize: 11.5 });
    iconCircle(s, "bolt", 10.6, 3.05, 0.46, C.slate);
    T(s, "Transformer A → Transformer B", { x: 11.15, y: 3.05, w: 1.5, h: 0.5, fontSize: 11.5 });
    // formal setup
    card(s, 0.6, 4.75, 12.13, 2.0, C.white);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 4.75, w: 12.13, h: 2.0, rectRadius: 0.08, fill: { color: C.white }, line: { color: C.line, width: 1 } });
    T(s, "Formal setting — unsupervised domain adaptation (UDA) for forecasting", { x: 0.85, y: 4.9, w: 11, h: 0.35, fontSize: 14, bold: true, color: C.teal });
    T(s, [
      { text: "Source ", options: { bold: true } }, { text: "D_S = {(x_i, y_i)}: look-back x ∈ ℝ^(L×C) and horizon y ∈ ℝ^(H×C) (multivariate, C variables).", options: { breakLine: true } },
      { text: "Target ", options: { bold: true } }, { text: "D_T = {x_j}: input windows only, and P_T(x, y) ≠ P_S(x, y).", options: { breakLine: true } },
      { text: "Goal ", options: { bold: true } }, { text: "find f minimising the target risk  E_(x,y)~P_T ‖f(x) − y‖², starting from the source model, with no target future values used for training or model selection." },
    ], { x: 0.85, y: 5.3, w: 11.7, h: 1.35, fontSize: 13.5, color: C.ink, paraSpaceAfter: 5 });
    s.addNotes("Explain with the temperature story from the project brief. Stress the constraint: in the target domain we observe inputs but do not use future values for training — that is what makes it UDA. Mention the finance and energy analogues briefly.");
  }

  // ================================================================ 4 WHAT SHIFTS
  {
    const s = content("01 · Problem understanding", "What actually shifts between hills and plains?");
    fitImage(s, path.join(FIG, "climate_domain_shift.png"), 0.6, 1.5, 12.1, 3.3);
    const m = N.datasets.climate;
    const tS = m.source.mean.temperature_2m, tT = m.target.mean.temperature_2m;
    const pS = m.source.mean.surface_pressure, pT = m.target.mean.surface_pressure;
    const cards = [
      ["Marginal / level shift", `Mean temperature ${tS.toFixed(1)} °C vs ${tT.toFixed(1)} °C; surface pressure ${pS.toFixed(0)} vs ${pT.toFixed(0)} hPa. Handled by per-domain scaling + RevIN.`, "ruler"],
      ["Dynamics (covariate) shift", "Different diurnal amplitude, humidity regimes and cross-variable correlations: the encoder sees unfamiliar patterns.", "wave"],
      ["Conditional shift", "The same recent history can lead to a different future (terrain, urban heat, monsoon). Needs adaptation, not rescaling.", "puzzle"],
    ];
    cards.forEach((c, i) => {
      const x = 0.6 + i * 4.1;
      card(s, x, 5.0, 3.9, 1.85);
      iconCircle(s, c[2], x + 0.2, 5.18, 0.5, i === 0 ? C.teal : C.saffron);
      T(s, c[0], { x: x + 0.85, y: 5.25, w: 2.9, h: 0.4, fontSize: 14.5, bold: true });
      T(s, c[1], { x: x + 0.2, y: 5.8, w: 3.55, h: 1.0, fontSize: 11.5, color: C.slate });
    });
    s.addNotes("Point at the histogram: the two domains barely overlap. Then separate three kinds of shift. The first one is easy (normalisation); the second and third are why we need an adaptation strategy.");
  }

  // ================================================================ 5 BACKGROUND: FORECASTERS
  {
    const s = content("02 · Background reading", "Lightweight forecasting models we studied");
    const cards = [
      ["DLinear", "Zeng et al., AAAI 2023", "A single linear map from look-back to horizon (after trend/seasonal split) rivals Transformers on long-horizon benchmarks.", "Linear temporal path in our model", "chart"],
      ["TCN", "Bai et al., 2018", "Stacked dilated 1-D convolutions: large receptive field, parallel training, few parameters.", "Strided, dilated conv encoder", "layers"],
      ["PatchTST", "Nie et al., ICLR 2023", "Patches of the series as tokens + channel independence; state of the art but heavier.", "Candidate encoder for later stages", "window"],
      ["RevIN", "Kim et al., ICLR 2022", "Normalise each input window by its own mean/std, then de-normalise the output: removes per-window level/scale shift.", "Built in: handles marginal shift", "balance"],
    ];
    cards.forEach((c, i) => {
      const x = 0.6 + i * 3.07;
      card(s, x, 1.6, 2.87, 4.35);
      iconCircle(s, c[4], x + 0.25, 1.85, 0.62, i % 2 ? C.saffron : C.teal);
      T(s, c[0], { x: x + 0.25, y: 2.65, w: 2.4, h: 0.45, fontFace: H, fontSize: 21, bold: true });
      T(s, c[1], { x: x + 0.25, y: 3.1, w: 2.4, h: 0.3, fontSize: 11, italic: true, color: C.muted });
      T(s, c[2], { x: x + 0.25, y: 3.5, w: 2.45, h: 1.6, fontSize: 12.5, color: C.slate });
      T(s, "→ " + c[3], { x: x + 0.25, y: 5.2, w: 2.45, h: 0.6, fontSize: 12, bold: true, color: C.teal });
    });
    card(s, 0.6, 6.15, 12.13, 0.66, C.basalt);
    T(s, `Design decision: combine a TCN encoder, a DLinear-style linear path and RevIN in one small model (${N.params_climate || "≈0.38M"} parameters), so adaptation stays cheap enough to run on a laptop CPU.`, {
      x: 0.85, y: 6.15, w: 11.7, h: 0.66, fontSize: 13, color: C.white, valign: "middle" });
    s.addNotes("Why lightweight: the brief asks for it, and adaptation is repeated per domain, so it must be cheap. DLinear shows simple models are strong; TCN gives a compact nonlinear encoder whose features we can align; RevIN deals with level shift for free.");
  }

  // ================================================================ 6 BACKGROUND: THEORY
  {
    const s = content("02 · Background reading", "Why aligning features can work: the domain-adaptation bound");
    card(s, 0.6, 1.6, 12.13, 1.45, C.basalt);
    s.addText([
      { text: "ε", options: { fontFace: H } }, { text: "T", options: { subscript: true } }, { text: "(h)  ≤  " },
      { text: "ε", options: { color: C.tealLt } }, { text: "S", options: { subscript: true, color: C.tealLt } }, { text: "(h)", options: { color: C.tealLt } },
      { text: "  +  " }, { text: "½ d", options: { color: "F2B45C" } }, { text: "HΔH", options: { subscript: true, color: "F2B45C" } },
      { text: "(D", options: { color: "F2B45C" } }, { text: "S", options: { subscript: true, color: "F2B45C" } }, { text: ", D", options: { color: "F2B45C" } },
      { text: "T", options: { subscript: true, color: "F2B45C" } }, { text: ")", options: { color: "F2B45C" } }, { text: "  +  " },
      { text: "λ*", options: { color: "E8A7A0" } },
    ], { x: 0.6, y: 1.6, w: 12.13, h: 1.1, fontFace: H, fontSize: 34, color: C.white, align: "center", valign: "middle", margin: 0, isTextBox: true });
    T(s, "Ben-David et al., Machine Learning 2010: target error is bounded by source error + domain divergence + the error of the best joint hypothesis.", {
      x: 0.6, y: 2.62, w: 12.13, h: 0.35, fontSize: 11.5, color: C.glacier, align: "center" });
    const terms = [
      ["Source error", "Keep minimising the supervised forecasting loss on Domain 1 during adaptation.", C.teal, "→ L_fcst"],
      ["Domain divergence", "Make source and target feature distributions look alike: MMD, CORAL, adversarial (DANN).", C.saffron, "→ L_MMD + L_CORAL"],
      ["Joint error λ*", "Cannot be measured without target labels. Alignment alone can destroy forecast-relevant information; keep features meaningful with self-supervision and stay close to the source weights.", C.rose, "→ L_SSL + L_anchor"],
    ];
    terms.forEach((t, i) => {
      const x = 0.6 + i * 4.1;
      card(s, x, 3.35, 3.9, 3.45);
      s.addShape(pres.shapes.OVAL, { x: x + 0.25, y: 3.58, w: 0.34, h: 0.34, fill: { color: t[2] }, line: { color: t[2] } });
      T(s, t[0], { x: x + 0.72, y: 3.56, w: 3.0, h: 0.4, fontSize: 17, bold: true });
      T(s, t[1], { x: x + 0.25, y: 4.15, w: 3.45, h: 1.9, fontSize: 13, color: C.slate });
      T(s, t[3], { x: x + 0.25, y: 6.2, w: 3.45, h: 0.4, fontSize: 14, bold: true, color: C.ink, fontFace: "Consolas" });
    });
    s.addNotes("This one equation motivates every term of our loss. Walk left to right: source error, divergence, joint error. The third term is the trap: aligning at any cost can make forecasting impossible, so we add self-supervision and anchoring.");
  }

  // ================================================================ 7 BACKGROUND: UDA TOOLBOX
  {
    const s = content("02 · Background reading", "The domain-adaptation toolbox and what we take from it");
    const hdr = (t) => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.basalt } } });
    const Y = { text: "✓ used", options: { bold: true, color: C.teal } };
    const rows = [
      [hdr("Family"), hdr("Core idea"), hdr("Representative methods"), hdr("Role in our project")],
      ["Discrepancy-based", "Minimise a statistical distance between source and target features", "MMD / DAN (Long et al. 2015), Deep CORAL (Sun & Saenko 2016)", "Alignment terms of CARE-DA; also stand-alone baselines"],
      ["Adversarial", "A domain classifier is fooled through a gradient-reversal layer", "DANN (Ganin et al. 2016), CoDATS (Wilson et al. 2020)", "Baseline"],
      ["Self-supervised / test-time training", "Auxiliary label-free task on target data keeps features informative", "TTT (Sun et al. 2020), masked reconstruction", "SSL term: masked reconstruction of target windows"],
      ["Parameter regularisation", "Penalise drifting away from the pre-trained (source) solution", "L2-SP (Li et al. 2018)", "Elastic anchor: prevents forgetting and over-alignment"],
      ["Normalisation-based", "Remove per-instance or per-domain statistics", "RevIN (Kim et al. 2022), AdaBN", "RevIN inside the model + per-domain z-scoring"],
    ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: { bold: j === 0, color: j === 3 ? C.teal : C.ink, fill: { color: i % 2 ? C.white : C.panel } } })));
    s.addTable(rows, { x: 0.6, y: 1.6, w: 12.13, colW: [2.45, 3.6, 3.33, 2.75], fontFace: B, fontSize: 12.5, valign: "middle",
      border: { type: "solid", pt: 0.75, color: C.line }, rowH: [0.45, 0.85, 0.75, 0.85, 0.75, 0.75], margin: 0.08 });
    T(s, "Insight from reading: no single family is sufficient for regression on time series. Our strategy combines the four complementary ideas in a single loss.", {
      x: 0.6, y: 6.25, w: 12.1, h: 0.5, fontSize: 13, italic: true, color: C.slate });
    s.addNotes("This is our map of the literature. Emphasise we did not invent a family; we composed complementary pieces and justify each one with the bound on the previous slide.");
  }

  // ================================================================ 8 BACKGROUND: TS-DA LITERATURE
  {
    const s = content("02 · Background reading", "Domain adaptation for time series: prior work");
    const papers = [
      ["CoDATS", "KDD 2020", "Adversarial 1-D CNN adaptation for multi-source sensor classification."],
      ["SASA", "AAAI 2021", "Aligns sparse associative structure between variables (inter-channel dependencies)."],
      ["DAF", "ICML 2022", "Attention-sharing Transformer for forecasting; needs some labeled target data."],
      ["CLUDA", "ICLR 2023", "Contrastive learning + nearest-neighbour alignment for time-series UDA."],
      ["RAINCOAT", "ICML 2023", "Aligns time- and frequency-domain features under feature and label shift."],
    ];
    papers.forEach((p, i) => {
      const y = 1.6 + i * 1.03;
      card(s, 0.6, y, 7.6, 0.88);
      T(s, p[0], { x: 0.85, y: y + 0.1, w: 1.6, h: 0.4, fontFace: H, fontSize: 18, bold: true, color: C.teal });
      T(s, p[1], { x: 0.85, y: y + 0.5, w: 1.6, h: 0.3, fontSize: 10.5, color: C.muted });
      T(s, p[2], { x: 2.55, y: y + 0.1, w: 5.5, h: 0.7, fontSize: 13, color: C.slate, valign: "middle" });
    });
    card(s, 8.5, 1.6, 4.23, 5.03, C.basalt);
    T(s, "The gap we target", { x: 8.8, y: 1.85, w: 3.7, h: 0.4, fontFace: H, fontSize: 20, bold: true, color: C.white });
    T(s, [
      { text: "Most time-series UDA work is for classification, not forecasting.", options: { bullet: true, breakLine: true } },
      { text: "Forecasting DA (e.g. DAF) usually assumes some labeled target data.", options: { bullet: true, breakLine: true } },
      { text: "Heavy backbones are costly to adapt for every new domain.", options: { bullet: true, breakLine: true } },
      { text: "Our aim: label-free adaptation of a lightweight multivariate forecaster, measured by how much of the supervised gap it closes.", options: { bullet: true, bold: true, color: "F2B45C" } },
    ], { x: 8.8, y: 2.4, w: 3.7, h: 4.1, fontSize: 13.5, color: C.glacier, paraSpaceAfter: 8 });
    s.addNotes("Name one takeaway per paper: CoDATS = adversarial; SASA = variable relationships; DAF = forecasting but with labels; CLUDA = contrastive; RAINCOAT = frequency. Then state our niche.");
  }

  // ================================================================ 9 DATASETS
  {
    const s = content("03 · Datasets", "Three benchmarks, each with two distinct domains");
    const d = N.datasets;
    const hdr = (t) => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.basalt } } });
    const row = (ds, name, src, tgt, why, freq) => {
      const x = d[ds];
      return [name, src, tgt, `${x.columns.length} vars: ${x.columns.join(", ")}`, `${freq}\n${x.source.start} → ${x.source.end}`, `${x.L} → ${x.H}`, why]
        .map((c, j) => ({ text: c, options: { bold: j === 0, fill: { color: C.white }, fontSize: j === 3 ? 10.5 : 12 } }));
    };
    const rows = [
      [hdr("Benchmark"), hdr("Domain 1 (source)"), hdr("Domain 2 (target)"), hdr("Variables"), hdr("Frequency & span"), hdr("L → H"), hdr("Why it is a real domain shift")],
      row("climate", "Climate\n(Open-Meteo ERA5)", "Shimla, 2195 m\n(hilly)", "New Delhi, 214 m\n(plain)", "Elevation, pressure, diurnal cycle and monsoon response differ", "Hourly"),
      row("finance", "Finance\n(Yahoo Finance, NSE)", "IT basket: TCS, INFY, WIPRO, HCLTECH, TECHM", "Pharma basket: SUNPHARMA, DRREDDY, CIPLA, DIVISLAB, LUPIN", "Sector-specific volatility, news and regulation", "Daily (trading)"),
      row("energy", "Energy\n(ETT benchmark)", "ETTh1: transformer, county 1", "ETTh2: transformer, county 2", "Different load profiles and oil-temperature dynamics", "Hourly"),
    ];
    s.addTable(rows, { x: 0.6, y: 1.6, w: 12.13, colW: [1.6, 1.75, 1.85, 2.35, 1.55, 0.78, 2.25], fontFace: B, fontSize: 12, valign: "middle",
      border: { type: "solid", pt: 0.75, color: C.line }, rowH: [0.45, 1.2, 1.2, 1.2], margin: 0.07 });
    const tiles = [
      [`${(d.climate.source.rows / 1000).toFixed(1)}k`, "hourly records per climate site (4 years)"],
      [`${d.finance.source.rows / d.finance.source.series}`, "trading days per stock (5 stocks per sector)"],
      [`${(d.energy.source.rows / 1000).toFixed(1)}k`, "hourly records per transformer (2 years)"],
    ];
    tiles.forEach((t, i) => {
      const x = 0.6 + i * 4.1;
      T(s, t[0], { x, y: 5.9, w: 1.75, h: 0.8, fontFace: H, fontSize: 36, bold: true, color: i === 1 ? C.saffron : C.teal, valign: "middle" });
      T(s, t[1], { x: x + 1.8, y: 5.95, w: 2.2, h: 0.7, fontSize: 12, color: C.slate, valign: "middle" });
    });
    s.addNotes("All three are public and reproducible with one script. The climate pair maps directly to the brief (hilly -> plain). Finance maps to the IT -> pharma example. ETT is a standard forecasting benchmark with two stations. Datasets to be confirmed with TAs.");
  }

  // ================================================================ 10 DATA PIPELINE
  {
    const s = content("03 · Data extraction & curation", "From raw APIs to leakage-free training windows");
    const steps = [
      ["db", "Extract", "Open-Meteo archive API (7 hourly vars, IST); yfinance OHLCV; ETT CSVs from GitHub"],
      ["broom", "Clean", "Linear interpolation of gaps; drop zero-volume / invalid trading days"],
      ["cogs", "Engineer", "Finance: log-return, high-low range, overnight gap, log-volume (stationary)"],
      ["cut", "Split", "Chronological 70/10/20 per series (ETT 60/20/20); test windows keep L steps of context"],
      ["ruler", "Scale", "z-score per domain; target statistics from unlabeled target inputs only"],
      ["window", "Window", "Sliding windows: L look-back → H horizon, all variables in and out"],
    ];
    steps.forEach((st, i) => {
      const x = 0.6 + i * 2.05;
      card(s, x, 1.65, 1.85, 3.1);
      iconCircle(s, st[0], x + 0.6, 1.85, 0.62, i < 3 ? C.teal : C.saffron);
      T(s, `${i + 1}. ${st[1]}`, { x: x + 0.12, y: 2.6, w: 1.62, h: 0.4, fontSize: 15, bold: true, align: "center" });
      T(s, st[2], { x: x + 0.15, y: 3.05, w: 1.58, h: 1.65, fontSize: 12, color: C.slate, align: "center" });
      if (i < steps.length - 1) arrow(s, x + 1.87, 2.16, x + 2.03, 2.16, C.slate, 1.5);
    });
    card(s, 0.6, 5.1, 7.4, 1.7, C.white);
    s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 5.1, w: 7.4, h: 1.7, rectRadius: 0.08, fill: { color: C.white }, line: { color: C.line } });
    iconCircle(s, "lock", 0.8, 5.28, 0.46, C.rose);
    T(s, "Leakage guards", { x: 1.4, y: 5.3, w: 5, h: 0.4, fontSize: 15, bold: true });
    T(s, [
      { text: "Strictly chronological splits; no shuffling across time", options: { bullet: true, breakLine: true } },
      { text: "Target future values never used for UDA training or for choosing checkpoints", options: { bullet: true, breakLine: true } },
      { text: "Adaptation runs for a fixed number of epochs (no peeking at target validation)", options: { bullet: true } },
    ], { x: 0.8, y: 5.75, w: 7.0, h: 1.0, fontSize: 12, color: C.slate, paraSpaceAfter: 2 });
    card(s, 8.25, 5.1, 4.48, 1.7, C.basalt);
    T(s, "Reproduce in one command", { x: 8.5, y: 5.28, w: 4, h: 0.35, fontSize: 13, bold: true, color: C.tealLt });
    T(s, "python -m src.data_download", { x: 8.5, y: 5.7, w: 4, h: 0.4, fontSize: 14, fontFace: "Consolas", color: C.white });
    T(s, "Raw pulls saved in data/raw, curated CSVs in data/processed/<benchmark>/<domain>.csv", { x: 8.5, y: 6.12, w: 4.0, h: 0.6, fontSize: 11, color: C.glacier });
    s.addNotes("Walk the pipeline left to right. The important design choice is step 5: each domain gets its own scaler, and for the target that only needs unlabeled inputs — legitimate under UDA. Then the leakage guards.");
  }

  // ================================================================ 11 SHIFT IN OTHER DATASETS
  {
    const s = content("03 · Datasets", "Domain shift in the energy and finance benchmarks");
    fitImage(s, path.join(FIG, "energy_domain_shift.png"), 0.6, 1.5, 8.6, 2.6, "left");
    fitImage(s, path.join(FIG, "finance_domain_shift.png"), 0.6, 4.25, 8.6, 2.6, "left");
    const e = N.datasets.energy, f = N.datasets.finance;
    card(s, 9.4, 1.55, 3.33, 2.5);
    T(s, "ETTh1 → ETTh2", { x: 9.6, y: 1.7, w: 3, h: 0.35, fontSize: 15, bold: true, color: C.teal });
    T(s, `Oil temperature mean ${e.source.mean.OT.toFixed(1)} vs ${e.target.mean.OT.toFixed(1)} °C; load channels have different scales and daily/weekly structure.`, { x: 9.6, y: 2.1, w: 2.95, h: 1.8, fontSize: 12, color: C.slate });
    card(s, 9.4, 4.3, 3.33, 2.5);
    T(s, "IT → Pharma", { x: 9.6, y: 4.45, w: 3, h: 0.35, fontSize: 15, bold: true, color: "A5650F" });
    T(s, `Daily high-low range ${f.source.mean.hl_range.toFixed(2)}% vs ${f.target.mean.hl_range.toFixed(2)}%. Returns are near-unpredictable, while volatility and volume are persistent and forecastable.`, { x: 9.6, y: 4.85, w: 2.95, h: 1.9, fontSize: 12, color: C.slate });
    s.addNotes("Energy has a strong shift; finance has a subtler one. That contrast becomes important when we interpret results later.");
  }

  // ================================================================ 12 MODEL
  {
    const s = content("04 · Lightweight forecaster", "LiteTCN-Forecaster: small, and built to be adapted");
    const box = (x, y, w, h, title, sub, fill, tc = C.ink) => {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: fill }, line: { color: fill } });
      T(s, title, { x, y: y + 0.1, w, h: 0.35, fontSize: 13.5, bold: true, align: "center", color: tc });
      if (sub) T(s, sub, { x: x + 0.08, y: y + 0.45, w: w - 0.16, h: h - 0.5, fontSize: 10.5, align: "center", color: tc === C.ink ? C.slate : C.glacier });
    };
    const y0 = 2.0;
    box(0.6, y0, 1.55, 1.2, "Input x", "L × C window\n(all variables)", C.glacier);
    box(2.45, y0, 1.55, 1.2, "RevIN", "normalise per window\n(stats kept)", C.panel);
    box(4.3, y0, 2.05, 1.2, "Conv encoder", "3 × Conv1d, stride 2\ndilation 1-2-2, GELU", C.teal, C.white);
    box(6.65, y0, 1.6, 1.2, "Latent z", "Flatten → Linear\n128-d features", C.basalt, C.white);
    box(8.6, y0 + 0.1, 1.95, 1.0, "Forecast head", "Linear 128 → H·C", C.panel);
    box(8.6, 3.55, 1.95, 1.0, "Recon head", "Linear 128 → L·C\n(self-supervision)", C.saffronLt);
    box(2.45, 4.75, 8.1, 0.85, "Linear temporal path", "DLinear-style L → H, shared across channels", C.panel);
    s.addShape(pres.shapes.OVAL, { x: 10.85, y: y0 + 0.35, w: 0.5, h: 0.5, fill: { color: C.white }, line: { color: C.ink, width: 1.25 } });
    T(s, "+", { x: 10.85, y: y0 + 0.35, w: 0.5, h: 0.5, fontSize: 20, bold: true, align: "center", valign: "middle" });
    box(11.55, y0 + 0.1, 1.18, 1.0, "ŷ", "RevIN⁻¹\nH × C", C.glacier);
    arrow(s, 2.15, y0 + 0.6, 2.45, y0 + 0.6);
    arrow(s, 4.0, y0 + 0.6, 4.3, y0 + 0.6);
    arrow(s, 6.35, y0 + 0.6, 6.65, y0 + 0.6);
    arrow(s, 8.25, y0 + 0.6, 8.6, y0 + 0.6);
    arrow(s, 8.25, y0 + 1.0, 8.6, 4.05);
    arrow(s, 10.55, y0 + 0.6, 10.85, y0 + 0.6);
    arrow(s, 11.35, y0 + 0.6, 11.55, y0 + 0.6);
    arrow(s, 3.22, y0 + 1.2, 3.22, 4.75);
    arrow(s, 10.55, 5.17, 11.1, 5.17, C.slate);
    s.addShape(pres.shapes.LINE, { x: 11.1, y: y0 + 0.85, w: 0, h: 5.17 - (y0 + 0.85), flipV: true, line: { color: C.slate, width: 1.75, endArrowType: "triangle" } });
    T(s, "↑ z is aligned across domains", { x: 6.2, y: 3.3, w: 2.1, h: 0.35, fontSize: 11, italic: true, color: "A5650F", align: "right" });
    const stats = [[N.params_climate || "381,606", "parameters (L=168, C=7)"], ["< 1 min", "per epoch on a laptop CPU"], ["3", "outputs: forecast, latent, reconstruction"]];
    stats.forEach((t, i) => {
      const x = 0.6 + i * 4.1;
      card(s, x, 5.95, 3.9, 0.85);
      T(s, t[0], { x: x + 0.2, y: 5.95, w: 1.7, h: 0.85, fontFace: H, fontSize: 24, bold: true, color: C.teal, valign: "middle" });
      T(s, t[1], { x: x + 1.95, y: 5.95, w: 1.85, h: 0.85, fontSize: 12, color: C.slate, valign: "middle" });
    });
    s.addNotes("Trace one window through the model. Two ideas: (1) the latent z is where we align domains; (2) the reconstruction head gives us a label-free task in the target domain. The linear path keeps a strong DLinear-like baseline inside the model.");
  }

  // ================================================================ 13 CARE-DA PIPELINE
  {
    const s = content("05 · Adaptation strategy", "CARE-DA: our label-free adaptation strategy");
    T(s, "Correlation-Aligned, Reconstruction-regularised, Elastic-anchored Domain Adaptation", { x: 0.6, y: 1.32, w: 12, h: 0.35, fontSize: 14, italic: true, color: C.slate });
    const pill = (x, y, w, h, title, sub, fill, tc = C.white) => {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.1, fill: { color: fill }, line: { color: fill } });
      T(s, title, { x, y: y + 0.08, w, h: 0.35, fontSize: 13.5, bold: true, align: "center", color: tc });
      if (sub) T(s, sub, { x: x + 0.08, y: y + 0.42, w: w - 0.16, h: h - 0.45, fontSize: 10.5, align: "center", color: tc });
    };
    pill(0.6, 2.2, 2.2, 1.1, "Source batch", "(x_s, y_s) labeled", C.teal);
    pill(0.6, 4.4, 2.2, 1.1, "Target batch", "x_t only — no labels", C.saffron, C.basalt);
    pill(3.35, 3.05, 2.3, 1.6, "Shared LiteTCN", "weights θ, initialised\nfrom source model θ_S", C.basalt);
    arrow(s, 2.8, 2.75, 3.35, 3.5, C.teal, 2);
    arrow(s, 2.8, 4.95, 3.35, 4.2, C.saffron, 2);
    const L = [
      ["L_fcst", "MSE(ŷ_s, y_s)", "keeps source skill", C.teal, 1.95],
      ["L_align", "MMD² + CORAL on (z_s, z_t)", "closes domain gap", C.saffron, 3.05],
      ["L_SSL / L_inner", "v1 mask-recon · v2 inner-forecast", "forecast-consistent pretext", C.saffron, 4.15],
      ["L_anchor", "‖θ − θ_S‖²", "no forgetting / collapse", C.rose, 5.25],
    ];
    L.forEach(([n, f, why, col, y]) => {
      arrow(s, 5.65, 3.85, 6.25, y + 0.42, C.muted, 1.25);
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 6.25, y, w: 3.6, h: 0.85, rectRadius: 0.08, fill: { color: C.white }, line: { color: col, width: 1.5 } });
      T(s, n, { x: 6.4, y: y + 0.07, w: 1.3, h: 0.35, fontSize: 14, bold: true, color: col === C.saffron ? "A5650F" : col, fontFace: "Consolas" });
      T(s, f, { x: 7.7, y: y + 0.09, w: 2.1, h: 0.35, fontSize: 11, color: C.ink });
      T(s, why, { x: 6.4, y: y + 0.45, w: 3.4, h: 0.3, fontSize: 10.5, italic: true, color: C.slate });
    });
    card(s, 10.15, 1.95, 2.58, 4.15, C.basalt);
    T(s, "One training step", { x: 10.35, y: 2.12, w: 2.2, h: 0.35, fontSize: 13, bold: true, color: C.tealLt });
    T(s, [
      { text: "Sample a source and a target batch", options: { bullet: { type: "number" }, breakLine: true } },
      { text: "Forward both through the shared model", options: { bullet: { type: "number" }, breakLine: true } },
      { text: "Compute the loss terms (v1 or v2)", options: { bullet: { type: "number" }, breakLine: true } },
      { text: "One AdamW update of θ", options: { bullet: { type: "number" }, breakLine: true } },
      { text: "Log each term, feature MMD and source error per epoch", options: { bullet: { type: "number" } } },
    ], { x: 10.35, y: 2.55, w: 2.25, h: 3.4, fontSize: 11.5, color: C.glacier, paraSpaceAfter: 5 });
    T(s, "v1 is the designed objective we ran first. Ablations already suggest replacing L_SSL; a seed-0 v2 probe is in progress (second half will lock this).", {
      x: 0.6, y: 6.4, w: 12.1, h: 0.4, fontSize: 13, italic: true, color: C.slate });
    s.addNotes("Explain the name. Be honest that v1 SSL hurt. Do not say the project is finished: v2 is a follow-up experiment we will confirm over 3 seeds.");
  }

  // ================================================================ 14 LOSS FUNCTION
  {
    const s = content("05 · Loss function design", "The adaptation objective, term by term");
    card(s, 0.6, 1.55, 12.13, 1.05, C.basalt);
    T(s, "L_CARE  =  L_fcst(S)  +  λ_M · MMD²_k(z_s, z_t)  +  λ_C · CORAL(z_s, z_t)  +  λ_R · L_mask(x_t)  +  λ_A · ‖θ − θ_S‖²", {
      x: 0.6, y: 1.55, w: 12.13, h: 1.05, fontFace: "Cambria Math", fontSize: 18, color: C.white, align: "center", valign: "middle" });
    const hdr = (t) => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.teal } } });
    const rows = [
      [hdr("Term"), hdr("Definition"), hdr("Why it is needed"), hdr("λ")],
      ["L_fcst", "(1/HC)‖f_θ(x_s) − y_s‖² on the source batch", "Bound term ε_S; prevents the model from forgetting how to forecast", "1"],
      ["MMD²_k", "Multi-kernel RBF MMD between latent batches, bandwidth from median heuristic × {¼ … 4}", "Matches the whole feature distribution (all moments)", "1"],
      ["CORAL", "‖Cov(z_s) − Cov(z_t)‖²_F / 4d²", "Matches cross-feature correlation, i.e. inter-variable structure", "1"],
      ["L_mask", "Mask 30% of target time steps; MSE of reconstructing them from z", "Label-free target signal; stops alignment from collapsing features", "1"],
      ["Anchor", "L2-SP: squared distance to source weights θ_S", "Elastic memory of Domain 1; limits negative transfer", "0.01"],
    ].map((r, i) => i === 0 ? r : r.map((c, j) => ({ text: c, options: { bold: j === 0, fontFace: j === 0 ? "Consolas" : B, fill: { color: i % 2 ? C.white : C.panel } } })));
    s.addTable(rows, { x: 0.6, y: 2.85, w: 12.13, colW: [1.35, 5.0, 5.03, 0.75], fontFace: B, fontSize: 12.5, valign: "middle",
      border: { type: "solid", pt: 0.75, color: C.line }, rowH: [0.42, 0.62, 0.62, 0.62, 0.62, 0.62], margin: 0.08 });
    T(s, "v1 weights were fixed a priori (fair vs baselines). Ablations say λ_R = 0 is better. Second-half plan: replace L_mask with a forecast-consistent pretext and confirm over 3 seeds.", {
      x: 0.6, y: 6.55, w: 12.1, h: 0.35, fontSize: 12.5, italic: true, color: C.slate });
    s.addNotes("This is deliverable 5, the designed loss. Connect each row to the bound. Mention the weights were fixed, not tuned per dataset, which makes the comparison fair.");
  }

  // ================================================================ 15 METRICS
  {
    const s = content("06 · Performance metrics", "Measuring forecast quality and adaptation efficiency");
    card(s, 0.6, 1.6, 5.9, 5.2);
    T(s, "Forecast accuracy (per domain)", { x: 0.85, y: 1.8, w: 5.4, h: 0.4, fontSize: 17, bold: true, color: C.teal });
    const fm = [
      ["MSE / MAE", "on z-scored values, averaged over all H steps and C variables (comparable across variables)"],
      ["MASE", "MAE ÷ in-sample seasonal-naive MAE: < 1 beats the naive forecast; scale-free across domains"],
      ["Key-variable RMSE", "in physical units: °C temperature, % range, °C oil temperature (interpretable for users)"],
    ];
    fm.forEach((m, i) => {
      T(s, m[0], { x: 0.85, y: 2.4 + i * 1.3, w: 5.4, h: 0.35, fontSize: 14.5, bold: true });
      T(s, m[1], { x: 0.85, y: 2.75 + i * 1.3, w: 5.4, h: 0.8, fontSize: 12.5, color: C.slate });
    });
    card(s, 6.83, 1.6, 5.9, 5.2, C.basalt);
    T(s, "Adaptation efficiency (our additions)", { x: 7.08, y: 1.8, w: 5.4, h: 0.4, fontSize: 17, bold: true, color: "F2B45C" });
    T(s, "Gap-Closure Ratio", { x: 7.08, y: 2.35, w: 5.4, h: 0.35, fontSize: 14.5, bold: true, color: C.white });
    T(s, "GCR = (E_source-only − E_method) / (E_source-only − E_oracle)", { x: 7.08, y: 2.72, w: 5.5, h: 0.4, fontSize: 13, fontFace: "Cambria Math", color: C.white });
    T(s, "Share of the gap to a fully supervised fine-tuned model that is closed without labels. 1 = oracle level; < 0 = negative transfer.", { x: 7.08, y: 3.12, w: 5.4, h: 0.75, fontSize: 12, color: C.glacier });
    const am = [
      ["Feature discrepancy", "MMD² between source and target latents: did alignment happen?"],
      ["Forgetting %", "change in Domain 1 test MSE after adapting to Domain 2"],
      ["Stability", "mean ± std over 3 random seeds"],
    ];
    am.forEach((m, i) => {
      T(s, m[0], { x: 7.08, y: 4.05 + i * 0.9, w: 5.4, h: 0.33, fontSize: 14, bold: true, color: C.white });
      T(s, m[1], { x: 7.08, y: 4.38 + i * 0.9, w: 5.4, h: 0.45, fontSize: 12, color: C.glacier });
    });
    s.addNotes("Deliverable 6. Accuracy metrics are standard; GCR is what tells us whether adaptation is worth it relative to simply collecting labels. Forgetting tells us whether Domain 1 still works.");
  }

  // ================================================================ 16 PROTOCOL
  {
    const s = content("07 · Experiments", "Experimental protocol");
    card(s, 0.6, 1.6, 5.9, 5.2);
    T(s, "Methods compared (same source checkpoint)", { x: 0.85, y: 1.8, w: 5.4, h: 0.4, fontSize: 16, bold: true, color: C.teal });
    const ms = [
      ["Seasonal naive", "repeat last season (no learning)"],
      ["Source-only", "Domain 1 model applied as-is (lower bound)"],
      ["DANN / Deep CORAL / MMD", "classic single-principle UDA baselines"],
      ["CARE-DA v1 / v2 (ours)", "v1 + ablations (3 seeds); v2 is a seed-0 probe only"],
      ["Target-only*", "trained from scratch on labeled Domain 2"],
      ["Fine-tune oracle*", "source model fine-tuned with Domain 2 labels (upper bound)"],
    ];
    ms.forEach((m, i) => {
      T(s, m[0], { x: 0.85, y: 2.35 + i * 0.7, w: 2.45, h: 0.6, fontSize: 13, bold: true, color: m[0].startsWith("CARE") ? C.teal : C.ink });
      T(s, m[1], { x: 3.3, y: 2.35 + i * 0.7, w: 3.1, h: 0.6, fontSize: 12, color: C.slate });
    });
    T(s, "* uses target labels: reference only, not UDA", { x: 0.85, y: 6.4, w: 5.4, h: 0.3, fontSize: 10.5, italic: true, color: C.muted });
    const hp = [
      ["Source training", "20 epochs, AdamW, lr 1e-3, cosine schedule, best source-validation checkpoint"],
      ["Adaptation", "10 epochs, lr 3e-4; v1 = last epoch; v2 picks min(source-val MSE + feature MMD) — still no target labels"],
      ["Self-supervision", "mask ratio 0.3; also used during source training so the head is ready"],
      ["Seeds / hardware", "3 seeds (0, 1, 2); PyTorch on a laptop CPU"],
    ];
    hp.forEach((h, i) => {
      const y = 1.6 + i * 1.33;
      card(s, 6.83, y, 5.9, 1.15);
      T(s, h[0], { x: 7.08, y: y + 0.12, w: 5.4, h: 0.35, fontSize: 14, bold: true });
      T(s, h[1], { x: 7.08, y: y + 0.48, w: 5.4, h: 0.6, fontSize: 12, color: C.slate });
    });
    s.addNotes("Emphasise fairness: every method starts from the same source checkpoint and gets the same budget; checkpoints for UDA are never chosen using target labels.");
  }

  // ================================================================ 17 RESULTS TABLE
  {
    const s = content("07 · First experimental campaign", "Preliminary Domain-2 results — not the final table");
    const methods = [["seasonal_naive", "Seasonal naive"], ["source_only", "Source-only"], ["dann", "DANN"], ["coral", "Deep CORAL"], ["mmd", "MMD"],
      ["care", "CARE-DA v1 (3 seeds)"], ["care_nossl", "CARE-DA − SSL (ablation)"], ["care_v2", "CARE-DA v2 (seed 0 probe)"],
      ["target_only", "Target-only*"], ["oracle_finetune", "Fine-tune oracle*"]];
    const uda = ["dann", "coral", "mmd", "care", "care_nossl", "care_noalign", "care_noanchor"];
    const best = {};
    DS.forEach((d) => { let b = null; uda.forEach((m) => { const v = R(d, m, "MSE_mean"); if (v !== null && (b === null || v < R(d, b, "MSE_mean"))) b = m; }); best[d] = b; });
    const hdr = (t) => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.basalt }, align: "center" } });
    const rows = [[hdr("Method"), ...DS.map((d) => hdr(`${DSN[d]} MSE`)), ...DS.map((d) => hdr(`${DSN[d]} Δ vs src-only`))]];
    methods.forEach(([m, name], i) => {
      const isC = m.startsWith("care");
      const fill = { color: isC ? C.glacier : (i % 2 ? C.white : C.panel) };
      const r = [{ text: name, options: { bold: isC, fill, color: isC ? C.teal : C.ink } }];
      DS.forEach((d) => {
        const mu = R(d, m, "MSE_mean"), sd = R(d, m, "MSE_std");
        r.push({ text: sd === null ? fmt(mu) : `${fmt(mu)} ± ${fmt(sd, 3)}`, options: { align: "center", fill, bold: best[d] === m, color: best[d] === m ? C.teal : C.ink } });
      });
      DS.forEach((d) => {
        const v = R(d, m, "rel_change");
        r.push({ text: m === "source_only" ? "—" : `${v >= 0 ? "+" : ""}${(100 * v).toFixed(1)}%`, options: { align: "center", fill, color: v > 0.005 ? C.rose : C.ink } });
      });
      rows.push(r);
    });
    s.addTable(rows, { x: 0.6, y: 1.48, w: 12.13, colW: [2.85, 1.72, 1.72, 1.72, 1.37, 1.38, 1.37], fontFace: B, fontSize: 11.5, valign: "middle",
      border: { type: "solid", pt: 0.75, color: C.line }, rowH: 0.40, margin: 0.04 });
    const nS = (N.results.climate && N.results.climate._n_seeds) || "?";
    T(s, `First campaign: v1 is mean ± std over ${nS} seeds. v2 is a seed-0 probe only — we will re-run it over 3 seeds before locking the method. Bold teal = best 3-seed label-free so far. * uses Domain 2 labels.`, {
      x: 0.6, y: 6.12, w: 12.1, h: 0.55, fontSize: 11, italic: true, color: C.slate });
    s.addNotes("Call this a first campaign. Be honest: full v1 is not the best; −SSL is stronger. v2 looks promising on seed 0 but is not confirmed. That is the work of the second half.");
  }

  // ================================================================ 18 RELATIVE CHANGE (native chart)
  {
    const s = content("07 · First experimental campaign", "Where the first campaign helped — and where it did not");
    const ms = [["dann", "DANN"], ["coral", "CORAL"], ["mmd", "MMD"], ["care_noalign", "CARE −align"], ["care_nossl", "CARE −SSL"], ["care", "CARE v1"], ["care_v2", "CARE v2"], ["oracle_finetune", "Oracle*"]];
    const data = DS.map((d) => ({ name: DSN[d], labels: ms.map((m) => m[1]), values: ms.map(([m]) => { const v = R(d, m, "rel_change"); return v === null ? 0 : Math.max(-12, Math.round(-v * 1000) / 10); }) }));
    s.addChart(pres.charts.BAR, data, {
      x: 0.6, y: 1.45, w: 8.4, h: 5.1, barDir: "col", barGrouping: "clustered", barGapWidthPct: 55,
      chartColors: [C.teal, C.saffron, "8A97A0"], showLegend: true, legendPos: "t", legendFontSize: 11, legendFontFace: B,
      catAxisLabelFontSize: 10.5, catAxisLabelFontFace: B, catAxisLabelColor: C.slate, valAxisLabelFontSize: 10, valAxisLabelColor: C.slate,
      valAxisTitle: "MSE reduction vs source-only (%), higher is better", showValAxisTitle: true, valAxisTitleFontSize: 11, valAxisTitleColor: C.slate,
      valGridLine: { color: "E3E8E7", size: 0.75 }, catGridLine: { style: "none" }, valAxisLabelFormatCode: "0",
      valAxisMinVal: -12, valAxisMaxVal: 12, valAxisMajorUnit: 4, catAxisLabelPos: "low",
    });
    T(s, `Bars clipped at −12%: DANN on energy is −${fmt(100 * R("energy", "dann", "rel_change"), 0)}%.`, { x: 0.8, y: 6.55, w: 7, h: 0.3, fontSize: 10, italic: true, color: C.muted });
    card(s, 9.25, 1.5, 3.48, 5.35, C.basalt);
    T(s, "Best so far (still provisional)", { x: 9.5, y: 1.68, w: 3.1, h: 0.35, fontSize: 14, bold: true, color: C.tealLt });
    const uda = ["dann", "coral", "mmd", "care", "care_nossl", "care_noalign", "care_noanchor", "care_v2"];
    const lab = { mmd: "MMD", care: "CARE v1", care_nossl: "CARE −SSL", care_noalign: "CARE −align", care_noanchor: "CARE −anchor", dann: "DANN", coral: "CORAL", care_v2: "CARE v2*" };
    DS.forEach((d, i) => {
      let b = uda[0]; uda.forEach((m) => { if (R(d, m, "MSE_mean") < R(d, b, "MSE_mean")) b = m; });
      const v = R(d, b, "rel_change"), o = R(d, "oracle_finetune", "rel_change");
      const y = 2.15 + i * 1.55;
      T(s, `${(100 * v).toFixed(1)}%`, { x: 9.5, y, w: 3.0, h: 0.72, fontFace: H, fontSize: 36, bold: true, color: C.white });
      T(s, `${DSN[d]} · ${lab[b]}  (oracle* ${(100 * o).toFixed(1)}%)`, { x: 9.5, y: y + 0.72, w: 3.15, h: 0.35, fontSize: 11.5, color: C.glacier });
    });
    s.addNotes("First-campaign reading: alignment helps, DANN/CORAL hurt. Climate's best 3-seed method is still MMD. Energy/finance v2 numbers are seed 0 only — say we will confirm them. Finance has little headroom even for the oracle.");
  }

  // ================================================================ 19 ADAPTATION CURVES
  {
    const s = content("07 · Adaptation behaviour", "Logged adaptation dynamics (climate, seed 0)");
    fitImage(s, path.join(FIG, "climate_adaptation_curves.png"), 0.6, 1.45, 12.13, 3.45);
    const obs = [
      ["Alignment happens", `CARE-DA cuts latent MMD² between domains from ${fmt(R("climate", "source_only", "feat_mmd_mean"), 3)} to ${fmt(R("climate", "care", "feat_mmd_mean"), 3)} (climate, mean of 3 seeds).`],
      ["Lower gap ≠ lower error", `CARE-DA reaches the smallest discrepancy, yet MMD-only forecasts better; DANN is unstable (MMD² ${fmt(R("climate", "dann", "feat_mmd_mean"), 1)}) and its target error drifts upward.`],
      ["No forgetting", `Label-free variants keep Domain 1 error flat (CARE-DA ${fmt(R("climate", "care", "forgetting_%_mean"), 1)}%), while supervised fine-tuning degrades Domain 1 by +${fmt(R("climate", "oracle_finetune", "forgetting_%_mean"), 0)}%.`],
    ];
    obs.forEach((o, i) => {
      const x = 0.6 + i * 4.1;
      card(s, x, 5.1, 3.9, 1.72);
      T(s, o[0], { x: x + 0.2, y: 5.22, w: 3.5, h: 0.35, fontSize: 14, bold: true, color: i === 1 ? "A5650F" : C.teal });
      T(s, o[1], { x: x + 0.2, y: 5.6, w: 3.5, h: 1.15, fontSize: 12, color: C.slate });
    });
    s.addNotes("These are the adaptation logs (deliverable 4). Target MSE is plotted for monitoring only, never used to pick checkpoints. Three messages: alignment works as intended, a smaller feature distance does not guarantee better forecasts, and label-free adaptation does not destroy Domain 1 performance.");
  }

  // ================================================================ 20 LATENT + FORECAST
  {
    const s = content("07 · Qualitative results", "What adaptation does to features and forecasts");
    fitImage(s, path.join(FIG, "energy_latent_pca.png"), 0.6, 1.45, 6.2, 2.85, "left");
    fitImage(s, path.join(FIG, "climate_forecast_example.png"), 6.95, 1.45, 5.8, 2.85);
    fitImage(s, path.join(FIG, "energy_forecast_example.png"), 6.95, 4.35, 5.8, 2.5);
    card(s, 0.6, 4.5, 6.1, 2.3);
    T(s, "Reading the plots", { x: 0.85, y: 4.65, w: 5.6, h: 0.35, fontSize: 14, bold: true, color: C.teal });
    T(s, [
      { text: `Left: PCA of encoder features on ETT (blue = Domain 1, orange = Domain 2). After CARE-DA the two clouds mix more evenly; latent MMD² falls from ${fmt(R("energy", "source_only", "feat_mmd_mean"), 2)} to ${fmt(R("energy", "care", "feat_mmd_mean"), 2)}.`, options: { bullet: true, breakLine: true } },
      { text: "Right: Domain 2 test windows. Grey = input, black = actual, orange dashed = source-only, blue = CARE-DA, green dotted = supervised oracle.", options: { bullet: true } },
    ], { x: 0.85, y: 5.05, w: 5.7, h: 1.7, fontSize: 12, color: C.slate, paraSpaceAfter: 6 });
    s.addNotes("Qualitative evidence. Pick one forecast window and explain where the source-only model is biased and how the adapted one moves towards the actual values.");
  }

  // ================================================================ 21 INSIGHTS / ABLATIONS
  {
    const s = content("07 · Insights so far", "What the first campaign taught us — open for the second half");
    const abl = [["care", "CARE-DA v1"], ["care_noalign", "− alignment"], ["care_nossl", "− SSL"], ["care_noanchor", "− anchor"], ["care_v2", "CARE-DA v2 (probe)"]];
    const hdr = (t) => ({ text: t, options: { bold: true, color: C.white, fill: { color: C.basalt }, align: "center" } });
    const rows = [[hdr("Variant (Δ MSE vs src-only)"), ...DS.map((d) => hdr(DSN[d]))]];
    abl.forEach(([m, n], i) => rows.push([{ text: n, options: { bold: i === 0, color: i === 0 ? C.teal : C.ink, fill: { color: i % 2 ? C.panel : C.white } } },
      ...DS.map((d) => { const v = R(d, m, "rel_change"); return { text: `${v >= 0 ? "+" : ""}${(100 * v).toFixed(1)}%`, options: { align: "center", color: v > 0.005 ? C.rose : C.ink, fill: { color: i % 2 ? C.panel : C.white } } }; })]));
    s.addTable(rows, { x: 0.6, y: 1.5, w: 6.0, colW: [2.55, 1.15, 1.15, 1.15], fontFace: B, fontSize: 12.5, valign: "middle",
      border: { type: "solid", pt: 0.75, color: C.line }, rowH: 0.42, margin: 0.06 });
    fitImage(s, path.join(FIG, "sensitivity.png"), 0.6, 3.75, 6.0, 3.1);
    const ins = [
      ["Alignment is the working core", "Removing the MMD + CORAL terms turns every gain into negative transfer. Matching feature distributions is what adapts the model."],
      ["SSL hurt; v2 is the next experiment", "Mask-recon raised error as λ_R grew. A seed-0 v2 probe looks better (climate −6.2%, energy −7.8%, finance −1.8%) but is not locked until we run 3 seeds."],
      ["Anchor helps where the shift is strong", "On energy the anchor turns −0.1% into −3.3%; DANN or CORAL alone cause negative transfer. Finance has little to gain (oracle only −2.4%)."],
    ];
    ins.forEach((t, i) => {
      const y = 1.5 + i * 1.8;
      card(s, 6.95, y, 5.78, 1.62);
      iconCircle(s, ["layers", "flask", "anchor"][i], 7.15, y + 0.2, 0.52, i === 1 ? C.saffron : C.teal);
      T(s, t[0], { x: 7.85, y: y + 0.16, w: 4.75, h: 0.35, fontSize: 14, bold: true });
      T(s, t[1], { x: 7.85, y: y + 0.52, w: 4.75, h: 1.05, fontSize: 11.5, color: C.slate });
    });
    s.addNotes("This is the pivot slide. First campaign taught us SSL hurts and alignment is the core. v2 is a probe, not the locked method. End with: that is why the second half exists.");
  }

  // ================================================================ 22 INTERFACE
  {
    const s = content("08 · Interface prototype", "Working demo — still to polish before the intensive assessment");
    const shot = path.join(HERE, "assets", "app_screenshot.png");
    if (fs.existsSync(shot)) {
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x: 0.6, y: 1.55, w: 7.5, h: 5.25, rectRadius: 0.06, fill: { color: C.panel }, line: { color: C.line } });
      fitImage(s, shot, 0.7, 1.65, 7.3, 5.05);
    } else fitImage(s, shot, 0.6, 1.55, 7.5, 5.25);
    const feats = [
      ["upload", "Pick benchmark and domain, then upload a multivariate CSV (or use the bundled samples). Raw OHLCV is converted to features automatically."],
      ["chart", "Per-variable plots of input, actual and forecast. Domain 2 can overlay the un-adapted source model (prototype comparison)."],
      ["ruler", "Window-level MSE, MAE, MASE and key-variable RMSE, plus a whole-file rolling evaluation against baselines."],
      ["desktop", "Still to add: batch upload, per-variable metric cards, uncertainty bands, and a cleaner Domain-2 comparison."],
    ];
    feats.forEach((f, i) => {
      const y = 1.6 + i * 1.18;
      iconCircle(s, f[0], 8.4, y + 0.05, 0.5, i % 2 ? C.saffron : C.teal);
      T(s, f[1], { x: 9.05, y, w: 3.68, h: 1.05, fontSize: 12, color: C.slate });
    });
    card(s, 8.4, 6.35, 4.33, 0.45, C.basalt);
    T(s, "streamlit run app/app.py", { x: 8.4, y: 6.35, w: 4.33, h: 0.45, fontFace: "Consolas", fontSize: 13, color: C.white, align: "center", valign: "middle" });
    s.addNotes("Show the prototype, then say what is still missing. Do not present this as the final interface.");
  }

  // ================================================================ 23 ROADMAP
  {
    const s = content("08 · Plan ahead", "Roadmap: sub-tasks, status and next steps");
    const phases = [
      ["First half (done)", C.teal, [
        "Problem study and literature review",
        "3 domain-pair datasets curated (pending TA)",
        "LiteTCN + CARE-DA v1 first campaign (3 seeds)",
        "Metrics, logs, plots defined",
        "Interface prototype (both domains)",
      ]],
      ["Second half (next)", C.saffron, [
        "Get the 3 datasets signed off by the TAs",
        "Lock CARE-DA v2 over 3 seeds + ablations",
        "Unsupervised λ / checkpoint selection",
        "Polish the interface and write the full report",
      ]],
      ["Before intensive assessment", C.slate, [
        "Stage-wise report (final version)",
        "Few-shot / extra-domain extras if time",
        "Recorded demo + live walkthrough",
        "Final presentation of locked results",
      ]],
    ];
    phases.forEach(([name, col, items], i) => {
      const x = 0.6 + i * 4.1;
      s.addShape(pres.shapes.OVAL, { x: x + 0.05, y: 1.7, w: 0.42, h: 0.42, fill: { color: col }, line: { color: col } });
      if (i === 0) s.addImage({ data: IC.check, x: x + 0.15, y: 1.8, w: 0.22, h: 0.22 });
      else if (i === 1) s.addImage({ data: IC.hour, x: x + 0.15, y: 1.8, w: 0.22, h: 0.22 });
      else s.addImage({ data: IC.book, x: x + 0.15, y: 1.8, w: 0.22, h: 0.22 });
      if (i < 2) s.addShape(pres.shapes.LINE, { x: x + 0.55, y: 1.91, w: 3.45, h: 0, line: { color: C.line, width: 2, dashType: "dash" } });
      T(s, name, { x: x + 0.6, y: 1.72, w: 3.3, h: 0.4, fontSize: 16, bold: true, color: col === C.saffron ? "A5650F" : col });
      card(s, x, 2.35, 3.9, 3.05, i === 0 ? C.glacier : C.panel);
      T(s, items.map((t, j) => ({ text: t, options: { bullet: true, breakLine: j < items.length - 1 } })), {
        x: x + 0.2, y: 2.5, w: 3.5, h: 2.85, fontSize: 12.5, color: C.ink, paraSpaceAfter: 5 });
    });
    T(s, "Course deliverables status", { x: 0.6, y: 5.6, w: 6, h: 0.35, fontSize: 13, bold: true, color: C.teal });
    const dl = [["Domain 1 + adapted models", 0.5], ["Interface: Domain 1", 0.5], ["Interface: Domain 2", 0.5], ["Performance metrics", 0.5], ["Stage-wise report", 0.5], ["Intensive assessment", 0]];
    dl.forEach(([t, st], i) => {
      const x = 0.6 + i * 2.04;
      const fill = st === 1 ? C.teal : st > 0 ? C.saffron : "B9C4C2";
      s.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y: 6.0, w: 1.92, h: 0.72, rectRadius: 0.08, fill: { color: fill }, line: { color: fill } });
      T(s, [{ text: st === 1 ? "✓ locked" : st > 0 ? "◐ first version" : "○ not started", options: { bold: true, breakLine: true, fontSize: 10.5 } }, { text: t, options: { fontSize: 11 } }],
        { x: x + 0.1, y: 6.03, w: 1.75, h: 0.66, color: st > 0 && st < 1 ? C.basalt : C.white, valign: "middle" });
    });
    s.addNotes("Close as mid-project: first cycle is real, nothing is locked. Ask the TAs to verify the datasets. Second half = lock v2, write the report, polish the demo.");
  }

  // ================================================================ 24-25 REFERENCES
  const refs1 = [
    "Shai Ben-David, John Blitzer, Koby Crammer, Alex Kulesza, Fernando Pereira and Jennifer Wortman Vaughan. A theory of learning from different domains, Machine Learning, 2010. https://doi.org/10.1007/s10994-009-5152-4",
    "Yaroslav Ganin, Evgeniya Ustinova, Hana Ajakan, Pascal Germain, Hugo Larochelle, François Laviolette, Mario Marchand and Victor Lempitsky. Domain-Adversarial Training of Neural Networks, Journal of Machine Learning Research, 2016. https://arxiv.org/abs/1505.07818",
    "Baochen Sun and Kate Saenko. Deep CORAL: Correlation Alignment for Deep Domain Adaptation, ECCV Workshops, 2016. https://arxiv.org/abs/1607.01719",
    "Mingsheng Long, Yue Cao, Jianmin Wang and Michael I. Jordan. Learning Transferable Features with Deep Adaptation Networks, ICML, 2015. https://arxiv.org/abs/1502.02791",
    "Arthur Gretton, Karsten M. Borgwardt, Malte J. Rasch, Bernhard Schölkopf and Alexander Smola. A Kernel Two-Sample Test, Journal of Machine Learning Research, 2012. https://jmlr.org/papers/v13/gretton12a.html",
    "Xuhong Li, Yves Grandvalet and Franck Davoine. Explicit Inductive Bias for Transfer Learning with Convolutional Networks, ICML, 2018. https://arxiv.org/abs/1802.01483",
    "Yu Sun, Xiaolong Wang, Zhuang Liu, John Miller, Alexei A. Efros and Moritz Hardt. Test-Time Training with Self-Supervision for Generalization under Distribution Shifts, ICML, 2020. https://arxiv.org/abs/1909.13231",
    "Xiaoyong Jin, Youngsuk Park, Danielle C. Maddix, Hao Wang and Yuyang Wang. Domain Adaptation for Time Series Forecasting via Attention Sharing, ICML, 2022. https://arxiv.org/abs/2102.06828",
    "Garrett Wilson, Janardhan Rao Doppa and Diane J. Cook. Multi-Source Deep Domain Adaptation with Weak Supervision for Time-Series Sensor Data, KDD, 2020. https://arxiv.org/abs/2005.10996",
    "Ruichu Cai, Jiawei Chen, Zijian Li, Wei Chen, Keli Zhang, Junjian Ye, Zhuozhang Li, Xiaoyan Yang and Zhenjie Zhang. Time Series Domain Adaptation via Sparse Associative Structure Alignment, AAAI, 2021. https://arxiv.org/abs/2012.11797",
  ];
  const refs2 = [
    "Yilmazcan Ozyurt, Stefan Feuerriegel and Ce Zhang. Contrastive Learning for Unsupervised Domain Adaptation of Time Series, ICLR, 2023. https://arxiv.org/abs/2206.06243",
    "Huan He, Owen Queen, Teddy Koker, Consuelo Cuevas, Theodoros Tsiligkaridis and Marinka Zitnik. Domain Adaptation for Time Series Under Feature and Label Shifts, ICML, 2023. https://arxiv.org/abs/2302.03133",
    "Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi and Jaegul Choo. Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift, ICLR, 2022. https://openreview.net/forum?id=cGDAkQo1C0p",
    "Ailing Zeng, Muxi Chen, Lei Zhang and Qiang Xu. Are Transformers Effective for Time Series Forecasting?, AAAI, 2023. https://arxiv.org/abs/2205.13504",
    "Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong and Jayant Kalagnanam. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers, ICLR, 2023. https://arxiv.org/abs/2211.14730",
    "Shaojie Bai, J. Zico Kolter and Vladlen Koltun. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling, arXiv preprint, 2018. https://arxiv.org/abs/1803.01271",
    "Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong and Wancai Zhang. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting (ETT dataset), AAAI, 2021. https://arxiv.org/abs/2012.07436",
    "Rob J. Hyndman and Anne B. Koehler. Another look at measures of forecast accuracy, International Journal of Forecasting, 2006. https://doi.org/10.1016/j.ijforecast.2006.03.001",
  ];
  const webs = [
    "Open-Meteo. https://open-meteo.com/en/docs/historical-weather-api. Accessed on: 18th September, 2026.",
    "Ran Aroussi. https://github.com/ranaroussi/yfinance. Accessed on: 18th September, 2026.",
    "Haoyi Zhou. https://github.com/zhouhaoyi/ETDataset. Accessed on: 18th September, 2026.",
  ];
  const refSlide = (title, list, start, extra) => {
    const s = content("References", title);
    const runs = list.map((r, i) => ({ text: `[${start + i}]  ${r}`, options: { breakLine: i < list.length - 1 } }));
    T(s, runs, { x: 0.6, y: 1.5, w: 12.13, h: extra ? 3.9 : 5.35, fontSize: 11, color: C.ink, paraSpaceAfter: 5.5 });
    return s;
  };
  refSlide("Research papers (1/2)", refs1, 1).addNotes("References follow the course-prescribed format.");
  {
    const s = refSlide("Research papers (2/2) and web resources", refs2, refs1.length + 1, true);
    T(s, "Websites", { x: 0.6, y: 5.45, w: 5, h: 0.3, fontSize: 12, bold: true, color: C.teal });
    T(s, webs.map((w, i) => ({ text: `[${refs1.length + refs2.length + 1 + i}]  ${w}`, options: { breakLine: i < webs.length - 1 } })), {
      x: 0.6, y: 5.75, w: 9.2, h: 1.1, fontSize: 11, color: C.ink, paraSpaceAfter: 3 });
    card(s, 10.05, 5.5, 2.68, 1.3, C.basalt);
    T(s, "Thank you", { x: 10.05, y: 5.6, w: 2.68, h: 0.6, fontFace: H, fontSize: 24, bold: true, color: C.white, align: "center" });
    T(s, "Questions welcome", { x: 10.05, y: 6.2, w: 2.68, h: 0.4, fontSize: 12, color: C.tealLt, align: "center" });
    s.addNotes("Wrap up in one sentence: lightweight forecaster + principled label-free adaptation + honest metrics. Thank the audience.");
  }

  let out = path.join(HERE, "IE643_Prep_Presentation.pptx");
  try {
    await pres.writeFile({ fileName: out });
  } catch (e) {
    if (e && e.code === "EBUSY") {
      out = path.join(HERE, "IE643_Prep_Presentation_updated.pptx");
      await pres.writeFile({ fileName: out });
    } else throw e;
  }
  console.log("wrote", out);
}

main().catch((e) => { console.error(e); process.exit(1); });
