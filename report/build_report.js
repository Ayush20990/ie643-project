// Builds the stage-wise project report (report/IE643_Project_Report.docx).
// node build_report.js     (needs ../presentation/numbers.json and ../results/figures)
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell, WidthType,
  ShadingType, BorderStyle, ImageRun, PageBreak, TableOfContents, LevelFormat, Footer, PageNumber, Header,
} = require("docx");

const TEAM = "[Team Name]";
const MEMBERS = [["[Member 1 Name]", "[Roll No.]"], ["[Member 2 Name]", "[Roll No.]"], ["[Member 3 Name]", "[Roll No.]"], ["[Member 4 Name]", "[Roll No.]"]];

const ROOT = path.join(__dirname, "..");
const FIG = path.join(ROOT, "results", "figures");
const N = JSON.parse(fs.readFileSync(path.join(ROOT, "presentation", "numbers.json"), "utf8"));
const NARR = JSON.parse(fs.readFileSync(path.join(__dirname, "narrative.json"), "utf8"));
const DS = ["climate", "energy", "finance"];
const DSN = { climate: "Climate", energy: "Energy", finance: "Finance" };
const MNAME = { seasonal_naive: "Seasonal naive", source_only: "Source-only", dann: "DANN", coral: "Deep CORAL", mmd: "MMD",
  care_noalign: "CARE-DA − align", care_nossl: "CARE-DA − SSL", care_noanchor: "CARE-DA − anchor", care: "CARE-DA (ours)",
  target_only: "Target-only*", oracle_finetune: "Fine-tune oracle*" };
const fmt = (v, d = 3) => (v === null || v === undefined || Number.isNaN(v) ? "—" : Number(v).toFixed(d));
const R = (ds, m, c) => (N.results[ds] && N.results[ds][m] ? N.results[ds][m][c] : null);

const FONT = "Calibri", HFONT = "Cambria", TEAL = "2F7F73", INK = "1B2B31", GREY = "5A6A70";
const PAGE_W = 11906, MARGIN = 1300, CONTENT_W = PAGE_W - 2 * MARGIN; // A4, DXA

// ------------------------------------------------------------------ helpers
function runs(text, base = {}) {
  // **bold** and _italic_ inline markup
  const out = [];
  const re = /(\*\*[^*]+\*\*)/g;
  let last = 0, m;
  while ((m = re.exec(text))) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), ...base }));
    const t = m[0];
    if (t.startsWith("**")) out.push(new TextRun({ text: t.slice(2, -2), bold: true, ...base }));
    else out.push(new TextRun({ text: t.slice(1, -1), italics: true, ...base }));
    last = m.index + t.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), ...base }));
  return out;
}
const P = (text, o = {}) => new Paragraph({ children: runs(text), spacing: { after: 120, line: 276 }, alignment: AlignmentType.JUSTIFIED, ...o });
const H1 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_1, children: [new TextRun(t)], pageBreakBefore: true });
const H2 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_2, children: [new TextRun(t)], keepNext: true });
const H3 = (t) => new Paragraph({ heading: HeadingLevel.HEADING_3, children: [new TextRun(t)], keepNext: true });
const B = (t, lvl = 0) => new Paragraph({ numbering: { reference: "bul", level: lvl }, children: runs(t), spacing: { after: 60 } });
const NUM = (t, ref = "num") => new Paragraph({ numbering: { reference: ref, level: 0 }, children: runs(t), spacing: { after: 60 } });
const EQ = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 160 },
  children: [new TextRun({ text: t, font: "Cambria Math", size: 23 })] });
const CODE = (t) => new Paragraph({ spacing: { after: 40 }, shading: { type: ShadingType.CLEAR, fill: "F1F6F5", color: "auto" },
  children: [new TextRun({ text: t, font: "Consolas", size: 18 })] });
const CAP = (t) => new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 200 }, children: runs(t, { size: 18, italics: true, color: GREY }) });

function pngSize(p) { const b = fs.readFileSync(p); return { w: b.readUInt32BE(16), h: b.readUInt32BE(20) }; }
function IMG(file, widthIn = 6.2, caption) {
  const p = path.isAbsolute(file) ? file : path.join(FIG, file);
  if (!fs.existsSync(p)) return [P(`[Figure pending: ${path.basename(p)}]`)];
  const s = pngSize(p), w = widthIn * 96, h = w * s.h / s.w;
  const out = [new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 120, after: 60 },
    children: [new ImageRun({ type: "png", data: fs.readFileSync(p), transformation: { width: w, height: h } })] })];
  if (caption) out.push(CAP(caption));
  return out;
}
const border = { style: BorderStyle.SINGLE, size: 4, color: "CFDAD8" };
function TABLE(rows, widthsFrac, { header = true, fontSize = 18, highlightRow = -1 } = {}) {
  const widths = widthsFrac.map((f) => Math.round(f * CONTENT_W));
  const total = widths.reduce((a, b) => a + b, 0);
  return new Table({
    width: { size: total, type: WidthType.DXA }, columnWidths: widths,
    rows: rows.map((r, i) => new TableRow({
      tableHeader: header && i === 0,
      children: r.map((c, j) => new TableCell({
        width: { size: widths[j], type: WidthType.DXA },
        borders: { top: border, bottom: border, left: border, right: border },
        shading: { type: ShadingType.CLEAR, color: "auto", fill: header && i === 0 ? "16232A" : (i === highlightRow ? "E6F0EE" : (i % 2 ? "FFFFFF" : "F7FAF9")) },
        margins: { top: 50, bottom: 50, left: 90, right: 90 },
        children: [new Paragraph({ children: runs(String(c), { size: fontSize, color: header && i === 0 ? "FFFFFF" : INK, bold: (header && i === 0) || i === highlightRow }) })],
      })),
    })),
  });
}
const SP = () => new Paragraph({ children: [], spacing: { after: 60 } });

// ------------------------------------------------------------------ content
const d = N.datasets;
const children = [];

// Title page
children.push(
  new Paragraph({ spacing: { before: 2200, after: 200 }, children: [new TextRun({ text: "IE 643 COURSE PROJECT  ·  STAGE-WISE REPORT", bold: true, color: TEAL, size: 22, characterSpacing: 40 })] }),
  new Paragraph({ spacing: { after: 300 }, children: [new TextRun({ text: "Unsupervised Domain Adaptation for Forecasting Multivariate Time-Series Data", font: HFONT, bold: true, size: 52, color: INK })] }),
  new Paragraph({ spacing: { after: 600 }, children: [new TextRun({ text: "A lightweight forecaster trained on one domain (hilly terrain, IT stocks, transformer 1) and adapted without target labels to another (plain terrain, pharma stocks, transformer 2) with CARE-DA.", size: 24, color: GREY })] }),
  new Paragraph({ spacing: { after: 120 }, children: [new TextRun({ text: `Team: ${TEAM}`, bold: true, size: 26 })] }),
  ...MEMBERS.map((m) => new Paragraph({ spacing: { after: 60 }, children: [new TextRun({ text: `${m[0]}  —  ${m[1]}`, size: 22 })] })),
  new Paragraph({ spacing: { before: 400 }, children: [new TextRun({ text: `Draft version · ${NARR.date}`, size: 20, color: GREY })] }),
  new Paragraph({ children: [new PageBreak()] }),
  new Paragraph({ children: [new TextRun({ text: "Contents", font: HFONT, bold: true, size: 32, color: INK })], spacing: { after: 200 } }),
  new TableOfContents("Contents", { hyperlink: true, headingStyleRange: "1-2" }),
);

// Abstract
children.push(H1("Abstract"));
NARR.abstract.forEach((t) => children.push(P(t)));

// 1 Introduction
children.push(H1("1. Introduction and problem statement"));
children.push(P("A forecasting model is typically trained on historical data from one setting and then deployed in another. When the statistical behaviour of the series changes between the two settings (a **domain shift**), the accuracy of the model degrades. The project brief illustrates this with temperature forecasting: a model built for a hilly region must be reused for a plain region, and similarly a model for an IT-sector stock must be reused for a pharma-sector stock."));
children.push(P("We study the **unsupervised** version of this problem. In Domain 1 (the source) we have full multivariate histories and train a forecaster. In Domain 2 (the target) we may use input windows, but no future values are used for training the adapted model or for choosing its checkpoint. The goal is to reduce the forecasting error on Domain 2 as much as possible relative to (i) simply reusing the source model and (ii) an oracle that is fine-tuned with target labels."));
children.push(H2("1.1 Formal setting"));
children.push(P("Let x ∈ ℝ^(L×C) be a look-back window of L time steps and C variables and y ∈ ℝ^(H×C) the next H steps. The source domain provides pairs D_S = {(x_i, y_i)} ~ P_S; the target provides inputs D_T = {x_j} ~ P_T(x) with P_T(x, y) ≠ P_S(x, y). Starting from a source model f_θS we seek parameters θ' that minimise the target risk:"));
children.push(EQ("θ' = argmin_θ  E_(x,y)~P_T [ (1/HC) ‖ f_θ(x) − y ‖² ]   using only D_S and D_T"));
children.push(H2("1.2 Mapping of project tasks to this report"));
children.push(TABLE([
  ["Project requirement", "Where addressed", "Status"],
  ["Collect ≥ 3 time-series datasets with distinct domains", "Section 3 (climate, finance, energy)", "Done (to be verified with TAs)"],
  ["Lightweight forecasting model on Domain 1", "Section 4 (LiteTCN, ~0.13–0.38 M parameters)", "Done"],
  ["Adaptation strategy", "Section 5 (CARE-DA)", "Done (v1)"],
  ["Implement and record adaptation behaviour (logs, plots)", "Section 8.4, results/<ds>/histories.json, train_log.txt", "Done"],
  ["Design a loss function for adaptation", "Section 6", "Done"],
  ["Explore performance metrics", "Section 7", "Done"],
  ["Interfaces for Domain 1 and Domain 2 CSV input + metrics", "Section 9 (Streamlit app)", "Done (v1)"],
], [0.42, 0.38, 0.2]));

// 2 Background
children.push(H1("2. Stage 1 — Background study"));
children.push(P("We first studied (a) lightweight forecasting architectures, (b) the theory of domain adaptation, (c) families of adaptation methods and (d) prior work on domain adaptation for time series. The purpose was to find components that are cheap, well understood, and suited to **regression** (forecasting) rather than classification."));
children.push(H2("2.1 Lightweight forecasting models"));
[
  "**DLinear** [14] shows that a single linear layer mapping the look-back window to the horizon (after a trend/seasonal decomposition) is competitive with Transformer forecasters on long-horizon benchmarks. We keep such a linear path inside our model as a strong, stable baseline component.",
  "**Temporal Convolutional Networks** [16] use stacked dilated 1-D convolutions, which give a large receptive field with few parameters and fully parallel training. We use a strided, dilated convolutional encoder to produce a compact latent representation that can be aligned across domains.",
  "**PatchTST** [15] tokenises the series into patches and uses channel-independent Transformers. It is state of the art but heavier; we list it as a candidate encoder for later stages.",
  "**RevIN** [13] normalises each input window by its own mean and standard deviation and de-normalises the output. It directly removes per-window level and scale shift, the simplest form of domain shift.",
].forEach((t) => children.push(B(t)));
children.push(H2("2.2 Theory of domain adaptation"));
children.push(P("Ben-David et al. [1] bound the target error of a hypothesis h by"));
children.push(EQ("ε_T(h) ≤ ε_S(h) + ½ d_HΔH(D_S, D_T) + λ*"));
children.push(P("where ε_S is the source error, d_HΔH measures how distinguishable the two domains are to the hypothesis class, and λ* is the error of the best joint hypothesis. The bound motivates three design goals that we map directly onto the terms of our loss: keep the source error low; reduce the divergence between source and target representations; and avoid representations for which no good joint predictor exists (i.e. do not align at the cost of losing forecast-relevant information)."));
children.push(H2("2.3 Families of domain-adaptation methods"));
children.push(TABLE([
  ["Family", "Idea", "Representative work", "Use in this project"],
  ["Discrepancy-based", "Minimise a statistical distance between feature distributions", "MMD / DAN [4,5], Deep CORAL [3]", "Alignment terms of CARE-DA; baselines"],
  ["Adversarial", "Fool a domain classifier via gradient reversal", "DANN [2], CoDATS [9]", "Baseline"],
  ["Self-supervised / test-time training", "Auxiliary label-free task on target data", "TTT [7]", "Masked-reconstruction term"],
  ["Parameter regularisation", "Penalise distance to pre-trained weights", "L2-SP [6]", "Elastic anchor term"],
  ["Normalisation", "Remove instance or domain statistics", "RevIN [13], AdaBN", "RevIN + per-domain scaling"],
], [0.2, 0.3, 0.25, 0.25]));
children.push(H2("2.4 Domain adaptation for time series"));
[
  "**CoDATS** [9]: adversarial adaptation of 1-D CNNs for multi-source sensor classification.",
  "**SASA** [10]: aligns the sparse associative structure between variables, i.e. which variables influence which.",
  "**DAF** [8]: an attention-sharing Transformer for forecasting across domains; it assumes some labeled target data.",
  "**CLUDA** [11]: contrastive learning with nearest-neighbour alignment for time-series UDA.",
  "**RAINCOAT** [12]: aligns both time- and frequency-domain features and handles feature and label shift.",
].forEach((t) => children.push(B(t)));
children.push(P("**Take-away.** Most time-series UDA work targets classification; forecasting-oriented DA usually needs labeled target data; and heavy backbones are expensive to adapt repeatedly. We therefore aim for label-free adaptation of a lightweight multivariate forecaster and evaluate how much of the supervised gap it closes."));

// 3 Datasets
children.push(H1("3. Stage 2 — Dataset collection and preparation"));
children.push(P("We selected three public sources so that every step is reproducible with one command (python -m src.data_download). Each benchmark provides two domains with **identical variables** but different underlying behaviour."));
children.push(TABLE([
  ["Benchmark", "Domain 1 (source)", "Domain 2 (target)", "Variables", "Span / frequency", "L → H"],
  ["Climate", "Shimla (31.10°N 77.17°E, 2195 m, hilly)", "New Delhi (28.61°N 77.21°E, 214 m, plain)", d.climate.columns.join(", "), `${d.climate.source.start} to ${d.climate.source.end}, hourly (${d.climate.source.rows} rows each)`, `${d.climate.L} → ${d.climate.H}`],
  ["Finance", "NSE IT: TCS, INFY, WIPRO, HCLTECH, TECHM", "NSE Pharma: SUNPHARMA, DRREDDY, CIPLA, DIVISLAB, LUPIN", d.finance.columns.join(", "), `${d.finance.source.start} to ${d.finance.source.end}, trading days (${d.finance.source.rows / d.finance.source.series} per stock)`, `${d.finance.L} → ${d.finance.H}`],
  ["Energy", "ETTh1 (transformer, county 1)", "ETTh2 (transformer, county 2)", d.energy.columns.join(", "), `${d.energy.source.start} to ${d.energy.source.end}, hourly (${d.energy.source.rows} rows each)`, `${d.energy.L} → ${d.energy.H}`],
], [0.1, 0.19, 0.19, 0.25, 0.19, 0.08], { fontSize: 16 }));
children.push(SP());
children.push(H2("3.1 Extraction steps"));
children.push(H3("Climate (Open-Meteo historical weather API, ERA5 reanalysis)"));
[
  "Endpoint https://archive-api.open-meteo.com/v1/archive with latitude/longitude of each site, start_date=2021-01-01, end_date=2024-12-31, timezone=Asia/Kolkata.",
  "Hourly variables: temperature_2m, relative_humidity_2m, dew_point_2m, surface_pressure, wind_speed_10m, cloud_cover, shortwave_radiation.",
  "Raw JSON responses are stored in data/raw/climate/; the returned elevation (2195 m vs 214 m) confirms the hilly-vs-plain contrast.",
].forEach((t) => children.push(B(t)));
children.push(H3("Finance (Yahoo Finance via yfinance)"));
[
  "Daily split/dividend-adjusted OHLCV for five NSE IT and five NSE pharma stocks, 2012-01-01 to 2025-06-30; raw files in data/raw/finance/.",
  "Prices are non-stationary and on very different scales across stocks, so we derive four stationary features per day: log_return = 100·ln(C_t/C_{t−1}); hl_range = 100·ln(H_t/L_t) (a volatility proxy); gap = 100·ln(O_t/C_{t−1}); log_volume = ln(1+V_t).",
  "Days with zero volume or invalid ranges are removed. Each stock is a separate series (windows never cross stock boundaries); the five series are pooled per sector.",
].forEach((t) => children.push(B(t)));
children.push(H3("Energy (ETT benchmark [17])"));
[
  "ETTh1.csv and ETTh2.csv downloaded from https://github.com/zhouhaoyi/ETDataset: hourly load features (HUFL, HULL, MUFL, MULL, LUFL, LULL) and oil temperature (OT) of two electricity transformers in different counties.",
  "The standard 12/4/4-month (60/20/20) chronological split is used.",
].forEach((t) => children.push(B(t)));
children.push(H2("3.2 Curation and preparation"));
[
  "**Missing values**: linear interpolation along time (the climate API returned complete series; the step is kept for user-uploaded CSVs).",
  "**Chronological split** per series: 70/10/20 (train/validation/test) for climate and finance, 60/20/20 for ETT. Validation and test segments are prefixed with L steps of history so that the first test window has a full look-back; no future information leaks backwards.",
  "**Per-domain z-scoring**: each domain is standardised with the mean and standard deviation of its own training segment. For the target this needs only unlabeled inputs, which is allowed in UDA. Metrics are computed in this standardised space so that variables with different units contribute equally.",
  "**Windowing**: sliding windows of L + H steps (stride 2 for climate training to limit memory, stride 1 otherwise). All C variables are both inputs and forecast targets (multivariate-to-multivariate).",
].forEach((t) => children.push(B(t)));
children.push(H2("3.3 Evidence of domain shift"));
children.push(...IMG("climate_domain_shift.png", 6.3, `Figure 1. Temperature in Shimla vs New Delhi: mean ${d.climate.source.mean.temperature_2m.toFixed(1)} °C vs ${d.climate.target.mean.temperature_2m.toFixed(1)} °C; surface pressure ${d.climate.source.mean.surface_pressure.toFixed(0)} vs ${d.climate.target.mean.surface_pressure.toFixed(0)} hPa.`));
children.push(...IMG("energy_domain_shift.png", 6.3, `Figure 2. Oil temperature of the two ETT transformers (mean ${d.energy.source.mean.OT.toFixed(1)} vs ${d.energy.target.mean.OT.toFixed(1)} °C).`));
children.push(...IMG("finance_domain_shift.png", 6.3, `Figure 3. Daily high-low range (volatility proxy) for an IT and a pharma stock (sector means ${d.finance.source.mean.hl_range.toFixed(2)}% vs ${d.finance.target.mean.hl_range.toFixed(2)}%).`));
children.push(P("We distinguish three kinds of shift: (i) **marginal/level shift** (different means and scales), which per-domain scaling and RevIN remove; (ii) **dynamics or covariate shift** (different daily amplitude, regimes and cross-variable correlations), which changes what the encoder sees; and (iii) **conditional shift**, where the same recent history leads to a different future. Types (ii) and (iii) are what the adaptation strategy must address."));

// 4 Model
children.push(H1("4. Stage 3 — Lightweight forecaster for Domain 1"));
children.push(P(`The **LiteTCN-Forecaster** (src/model.py) has ${N.params_climate || "≈0.38 M"} parameters for L = 168, C = 7 (≈0.13 M for finance). It was designed for adaptation: it exposes a latent vector that can be aligned between domains, and it has an auxiliary head that provides a label-free learning signal in the target domain.`));
[
  "**RevIN** normalises each input window per variable (learnable affine), and the output is de-normalised with the same statistics.",
  "**Encoder**: three Conv1d layers (kernel 5; stride 2; dilation 1, 2, 2; GELU), widths 32-64-64, followed by flatten → dropout (0.1) → linear → GELU, giving a 128-d latent vector z.",
  "**Forecast head**: linear z → H·C, added to a **linear temporal path** (DLinear-style L → H map shared across variables) applied to the normalised input.",
  "**Reconstruction head**: linear z → L·C. During training, 30% of the time steps of the input are masked and must be reconstructed from z (masked-reconstruction pretext task).",
].forEach((t) => children.push(B(t)));
children.push(H2("4.1 Source training"));
children.push(P("The source model is trained with MSE on the forecast plus 0.5 × the masked-reconstruction loss (so that the reconstruction head is already meaningful before adaptation), using AdamW (lr 1e-3, weight decay 1e-4), a cosine schedule, batch size 256, gradient clipping at 1.0 and 20 epochs. The checkpoint with the lowest source-validation MSE is kept. Source-domain test MSE (z-scored): " +
  DS.map((x) => `${DSN[x]} ${fmt(N.results[x] ? N.results[x]._source_test_mse : null)}`).join(", ") + "."));

// 5 Strategy
children.push(H1("5. Stage 4 — Adaptation strategy: CARE-DA"));
children.push(P("**CARE-DA** (Correlation-Aligned, Reconstruction-regularised, Elastic-anchored Domain Adaptation) fine-tunes a copy of the source model on mixed batches: a labeled source batch (x_s, y_s) and an unlabeled target batch x_t. The shared network produces latents z_s and z_t. Four goals, each motivated by the bound in Section 2.2, are optimised jointly:"));
[
  "**Keep source skill** (ε_S): supervised forecasting loss on the source batch.",
  "**Close the domain gap** (d_HΔH): align the distributions of z_s and z_t with multi-kernel MMD (all moments) and CORAL (second-order, cross-feature correlation).",
  "**Keep target features informative** (λ*): masked reconstruction of target windows, which is label-free.",
  "**Do not drift or forget**: an L2-SP penalty pulls the weights towards the source solution θ_S.",
].forEach((t) => children.push(B(t)));
children.push(H2("5.1 Algorithm"));
[
  "Initialise θ ← θ_S (trained source model); store a frozen copy θ_S.",
  "For each of E = 10 epochs, for ⌈|D_T| / 256⌉ steps: sample a source batch (x_s, y_s) and a target batch x_t of 256 windows each.",
  "Forward both batches: ŷ_s, z_s = f_θ(x_s) and z_t = enc_θ(x_t); compute the loss of Section 6; take one AdamW step (lr 3e-4, gradient clipping 1.0).",
  "After every epoch log each loss term, the feature MMD² between source-validation and target-validation latents, and the source-validation MSE. Target test MSE is logged only for drawing curves; it is never used for selection.",
  "Return the final θ' (no early stopping on target data).",
].forEach((t) => children.push(NUM(t)));

// 6 Loss
children.push(H1("6. Stage 5 — Loss function design"));
children.push(EQ("L_CARE = L_fcst + λ_M · MMD²_k(z_s, z_t) + λ_C · CORAL(z_s, z_t) + λ_R · L_mask(x_t) + λ_A · ‖θ − θ_S‖²"));
children.push(P("**Forecasting loss.** L_fcst = (1/(B·H·C)) Σ ‖f_θ(x_s) − y_s‖², the mean squared error on standardised values."));
children.push(P("**Multi-kernel MMD** [4,5]. With Gaussian kernels k_σ(a,b) = exp(−‖a−b‖²/σ) for σ ∈ {¼, ½, 1, 2, 4} × the median pairwise squared distance (median heuristic):"));
children.push(EQ("MMD²_k = E[k(z_s, z_s')] + E[k(z_t, z_t')] − 2 E[k(z_s, z_t)]"));
children.push(P("**CORAL** [3] matches second-order statistics, i.e. how latent features co-vary, which for multivariate series reflects inter-variable structure:"));
children.push(EQ("CORAL = ‖Cov(z_s) − Cov(z_t)‖²_F / (4 d²)"));
children.push(P("**Masked reconstruction.** A random 30% of the time steps of x_t are zeroed (after RevIN); the reconstruction head predicts the full window from z and the squared error is averaged over the masked positions only. This keeps the encoder sensitive to the actual structure of target data, so that alignment cannot collapse the features."));
children.push(P("**Elastic anchor (L2-SP)** [6]: the squared Euclidean distance between current and source weights. It limits forgetting of Domain 1 and makes negative transfer less likely."));
children.push(P("**Weights.** λ_M = λ_C = λ_R = 1 and λ_A = 0.01, identical across all benchmarks; they were fixed a priori and not tuned on target labels. A post-hoc sensitivity study is reported in Section 8.5."));

// 7 Metrics
children.push(H1("7. Stage 6 — Performance metrics"));
children.push(H2("7.1 Forecast accuracy"));
[
  "**MSE and MAE** on per-domain standardised values, averaged over H steps and C variables: comparable across variables with different units, and standard in the forecasting literature.",
  "**MASE** [18]: forecast MAE divided by the in-sample MAE of the seasonal-naive forecast on the same input window (season 24 for hourly data, 1 for daily data). Values below 1 beat the naive forecast; it is scale-free and comparable across domains.",
  "**Key-variable RMSE / MAE in physical units**: temperature in °C (climate), daily range in % (finance) and oil temperature in °C (energy); interpretable for end users and shown in the interface.",
].forEach((t) => children.push(B(t)));
children.push(H2("7.2 Adaptation efficiency"));
children.push(EQ("GCR = (E_source-only − E_method) / (E_source-only − E_oracle)"));
[
  "**Gap-closure ratio (GCR)**: the share of the error gap between the un-adapted model and a model fine-tuned with target labels that a label-free method closes. 1 means oracle-level; 0 means no gain; negative values indicate negative transfer.",
  "**Feature discrepancy**: MMD² between source and target latents on held-out windows, which checks whether alignment actually happened.",
  "**Forgetting %**: relative change of Domain 1 test MSE after adaptation.",
  "**Stability**: mean ± standard deviation over three seeds.",
].forEach((t) => children.push(B(t)));

// 8 Experiments
children.push(H1("8. Stage 7 — Experiments"));
children.push(H2("8.1 Protocol"));
[
  "All adaptation methods start from the **same source checkpoint** per seed and run for the same number of epochs with the same optimiser.",
  "**Baselines**: seasonal naive; source-only; DANN [2] (gradient reversal, weight 0.1 with the standard λ schedule); Deep CORAL [3]; MMD [4]. **Ablations**: CARE-DA without alignment, without SSL, without anchor.",
  "**Label-using references** (not UDA): target-only (trained from scratch on labeled Domain 2) and fine-tune oracle (source model fine-tuned on labeled Domain 2 with target-validation early stopping).",
  "Seeds 0, 1, 2; CPU only (PyTorch).",
].forEach((t) => children.push(B(t)));
children.push(H2("8.2 Main results"));
children.push(P("Table 2 lists target-domain (Domain 2) test metrics. Standard deviations are over seeds."));
const methods = ["seasonal_naive", "source_only", "dann", "coral", "mmd", "care_noalign", "care_nossl", "care_noanchor", "care", "target_only", "oracle_finetune"];
DS.forEach((x, k) => {
  if (!N.results[x]) return;
  children.push(H3(d[x].title.replace("->", "→")));
  const rows = [["Method", "MSE", "Δ vs src-only", "MASE", `${d[x].key} RMSE`, "Feat. MMD²", "GCR", "Forget %"]];
  methods.forEach((m) => {
    if (!N.results[x][m]) return;
    const sd = R(x, m, "MSE_std");
    const rc = R(x, m, "rel_change");
    rows.push([MNAME[m], sd === null ? fmt(R(x, m, "MSE_mean")) : `${fmt(R(x, m, "MSE_mean"))} ± ${fmt(sd)}`,
      m === "source_only" ? "—" : `${rc >= 0 ? "+" : ""}${(100 * rc).toFixed(1)}%`,
      fmt(R(x, m, "MASE_mean")), fmt(R(x, m, "key_RMSE_mean")), fmt(R(x, m, "feat_mmd_mean")),
      m === "seasonal_naive" ? "—" : fmt(R(x, m, "gcr_means"), 2), m === "seasonal_naive" ? "—" : fmt(R(x, m, "forgetting_%_mean"), 1)]);
  });
  children.push(TABLE(rows, [0.22, 0.15, 0.09, 0.09, 0.12, 0.11, 0.1, 0.12], { fontSize: 16, highlightRow: rows.findIndex((r) => r[0] === MNAME.care) }));
  children.push(CAP(`Table 2${"abc"[k]}. ${DSN[x]} target test set. * uses Domain 2 labels. GCR: gap-closure ratio computed from seed-averaged MSEs (source-only = 0, fine-tune oracle = 1). Forget %: change of Domain 1 test MSE.`));
});
children.push(...IMG("gap_closure_all.png", 6.4, "Figure 4. Change of Domain 2 test MSE relative to the un-adapted source model (mean of 3 seeds)."));
children.push(H2("8.3 Findings"));
NARR.findings.forEach((t) => children.push(B(t)));
children.push(H2("8.4 Adaptation behaviour (logs and plots)"));
children.push(P("Per-epoch logs of every loss term, the latent MMD², source-validation MSE and (for monitoring only) target test MSE are stored in results/<benchmark>/histories.json and train_log.txt."));
DS.forEach((x) => children.push(...IMG(`${x}_adaptation_curves.png`, 6.5, `Adaptation curves, ${DSN[x]} (seed 0).`)));
children.push(...IMG("climate_loss_components.png", 4.6, "Weighted CARE-DA loss components during adaptation (climate, seed 0)."));
children.push(...IMG("energy_latent_pca.png", 5.6, "PCA of encoder latents for source (blue) and target (orange) windows before and after CARE-DA (energy)."));
children.push(...IMG("climate_latent_pca.png", 5.6, "Same for the climate benchmark."));
DS.forEach((x) => children.push(...IMG(`${x}_forecast_example.png`, 6.0, `Qualitative Domain 2 forecast, ${DSN[x]} (key variable, z-scored).`)));
children.push(H2("8.5 Post-hoc sensitivity"));
children.push(P("To understand the effect of the two regularisers, we varied λ_R (SSL) and λ_A (anchor) one at a time from the seed-0 source model. These runs were performed **after** fixing the main configuration and are reported for transparency; they are not used to select the model in Table 2."));
children.push(...IMG("sensitivity.png", 6.2, "Relative change of target MSE vs source-only as a function of λ_R and λ_A."));
NARR.sensitivity.forEach((t) => children.push(P(t)));

// 9 Interface
children.push(H1("9. Stage 8 — Interface methodology"));
children.push(P("The interface (app/app.py) is a Streamlit web application that satisfies deliverables 2–4: a user uploads multivariate time-varying data from Domain 1 or Domain 2 as CSV and sees the input, the actual future and the forecast, together with goodness-of-fit metrics."));
children.push(H2("9.1 Design"));
[
  "**Model registry**: for each benchmark, models/<ds>/ holds source_model.pt (Domain 1), adapted_model.pt (Domain 2, CARE-DA) and meta.json with variable names, L, H, seasonal period and both domain scalers. Models are loaded once and cached.",
  "**Input handling**: the CSV must contain a date column and the benchmark's variables; for finance, raw Yahoo-style OHLCV is accepted and converted with the same feature code used in training (ohlcv_to_features), avoiding train/serve skew. Missing values are interpolated; missing columns or too-short files give clear error messages.",
  "**Domain routing**: Domain 1 data are scaled with the source scaler and forecast by the source model; Domain 2 data use the target scaler and the adapted model, and the source-only forecast can be overlaid to make the effect of adaptation visible.",
  "**Visualisation**: interactive Plotly panels, one per selected variable, show the look-back input (grey), actual future (black), adapted forecast (blue) and source-only forecast (orange, dashed), with a unified hover tooltip. A slider selects the forecast origin.",
  "**Metrics**: for the selected window: MSE, MAE, MASE and key-variable RMSE; for the whole file: a rolling evaluation over non-overlapping windows comparing the deployed model with source-only and seasonal-naive forecasts, plus the relative change in MSE due to adaptation.",
  "**Export and benchmark view**: forecasts can be downloaded as CSV; a second tab shows the benchmark tables and figures from Section 8.",
].forEach((t) => children.push(B(t)));
children.push(...IMG(path.join(ROOT, "presentation", "assets", "app_screenshot.png"), 6.2, "Figure. The interface with Domain 2 (New Delhi) data."));
children.push(H2("9.2 How to run"));
["pip install -r requirements.txt", "streamlit run app/app.py"].forEach((t) => children.push(CODE(t)));
children.push(P("Sample inputs from the held-out test period of every domain are provided in sample_inputs/ and are used automatically when no file is uploaded."));

// 10 Discussion
children.push(H1("10. Discussion and limitations"));
NARR.discussion.forEach((t) => children.push(B(t)));

// 11 Future work
children.push(H1("11. Future work and roadmap"));
NARR.future.forEach((t) => children.push(B(t)));

// 12 Reproducibility
children.push(H1("12. Code, documentation and reproducibility"));
children.push(TABLE([
  ["Path", "Purpose"],
  ["src/config.py", "Benchmark definitions (domains, variables, L, H, splits), paths"],
  ["src/data_download.py", "Downloads and curates all datasets"],
  ["src/data.py", "Splitting, per-domain scaling, windowing, batching"],
  ["src/model.py", "LiteTCN-Forecaster and RevIN"],
  ["src/losses.py", "MK-MMD, CORAL, DANN discriminator with gradient reversal, L2-SP"],
  ["src/metrics.py", "MSE, MAE, MASE, key-variable errors, gap-closure ratio"],
  ["src/train.py", "Source training, CARE-DA and baseline adaptation"],
  ["src/run_experiments.py", "Full benchmark for one dataset (all methods, seeds, logs, checkpoints)"],
  ["src/sensitivity.py, src/plots.py", "Sensitivity study; all figures"],
  ["app/app.py", "Streamlit interface"],
  ["models/, results/, sample_inputs/", "Trained models + scalers; logs, metrics, figures; demo CSVs"],
], [0.33, 0.67]));
children.push(SP());
children.push(P("**Dependencies**: Python ≥ 3.10, torch, numpy, pandas, matplotlib, requests, yfinance, streamlit, plotly (requirements.txt). **Commands**:"));
["python -m src.data_download", "python -m src.run_experiments --dataset climate --seeds 0 1 2   (and energy, finance)", "python -m src.sensitivity --dataset climate   (and energy, finance)", "python -m src.plots", "streamlit run app/app.py"].forEach((t) => children.push(CODE(t)));

// References
children.push(H1("References"));
const refs = [
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
  "Yilmazcan Ozyurt, Stefan Feuerriegel and Ce Zhang. Contrastive Learning for Unsupervised Domain Adaptation of Time Series, ICLR, 2023. https://arxiv.org/abs/2206.06243",
  "Huan He, Owen Queen, Teddy Koker, Consuelo Cuevas, Theodoros Tsiligkaridis and Marinka Zitnik. Domain Adaptation for Time Series Under Feature and Label Shifts, ICML, 2023. https://arxiv.org/abs/2302.03133",
  "Taesung Kim, Jinhee Kim, Yunwon Tae, Cheonbok Park, Jang-Ho Choi and Jaegul Choo. Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift, ICLR, 2022. https://openreview.net/forum?id=cGDAkQo1C0p",
  "Ailing Zeng, Muxi Chen, Lei Zhang and Qiang Xu. Are Transformers Effective for Time Series Forecasting?, AAAI, 2023. https://arxiv.org/abs/2205.13504",
  "Yuqi Nie, Nam H. Nguyen, Phanwadee Sinthong and Jayant Kalagnanam. A Time Series is Worth 64 Words: Long-term Forecasting with Transformers, ICLR, 2023. https://arxiv.org/abs/2211.14730",
  "Shaojie Bai, J. Zico Kolter and Vladlen Koltun. An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling, arXiv preprint, 2018. https://arxiv.org/abs/1803.01271",
  "Haoyi Zhou, Shanghang Zhang, Jieqi Peng, Shuai Zhang, Jianxin Li, Hui Xiong and Wancai Zhang. Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting, AAAI, 2021. https://arxiv.org/abs/2012.07436",
  "Rob J. Hyndman and Anne B. Koehler. Another look at measures of forecast accuracy, International Journal of Forecasting, 2006. https://doi.org/10.1016/j.ijforecast.2006.03.001",
  "Open-Meteo. https://open-meteo.com/en/docs/historical-weather-api. Accessed on: 18th September, 2026.",
  "Ran Aroussi. https://github.com/ranaroussi/yfinance. Accessed on: 18th September, 2026.",
  "Haoyi Zhou. https://github.com/zhouhaoyi/ETDataset. Accessed on: 18th September, 2026.",
];
refs.forEach((r, i) => children.push(new Paragraph({ spacing: { after: 90 }, indent: { left: 440, hanging: 440 },
  children: [new TextRun({ text: `[${i + 1}]\t${r}`, size: 19 })] })));

const doc = new Document({
  creator: TEAM, title: "IE 643 Project Report - Unsupervised Domain Adaptation for Time-Series Forecasting",
  styles: {
    default: { document: { run: { font: FONT, size: 22, color: INK } } },
    paragraphStyles: [
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: HFONT, size: 34, bold: true, color: INK }, paragraph: { spacing: { before: 120, after: 220 }, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: HFONT, size: 27, bold: true, color: TEAL }, paragraph: { spacing: { before: 260, after: 120 }, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true,
        run: { font: FONT, size: 23, bold: true, color: INK }, paragraph: { spacing: { before: 200, after: 100 }, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bul", levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1080, hanging: 270 } } } }] },
    { reference: "num", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 300 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: 1300, bottom: 1300, left: MARGIN, right: MARGIN } } },
    headers: { default: new Header({ children: [new Paragraph({ alignment: AlignmentType.RIGHT, children: [new TextRun({ text: `IE 643 Course Project · ${TEAM}`, size: 16, color: GREY })] })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], size: 18, color: GREY })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then((buf) => {
  const out = path.join(__dirname, "IE643_Project_Report.docx");
  fs.writeFileSync(out, buf);
  console.log("wrote", out);
});
