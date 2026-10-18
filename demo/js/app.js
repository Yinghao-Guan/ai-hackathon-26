// RoleKV demo UI. All numbers come from metrics.js; this file only renders them.
// Loaded text is always inserted with textContent / createTextNode, never as HTML.
import {
  METHODS, METHOD_META, TIMING_MACHINE, parseJSONL, isolateDataset, analyze, formatCtxK,
  replayTaskKeys, turnTotals, ttftEligible, turnSegments, compareTaskKeys, mean,
} from "./metrics.js";

const REAL_DEFAULT = "data/events.jsonl";
const MOCK_DEFAULT = "mock/mock_events.jsonl";
const STEP_MS = 650;

const $ = (id) => document.getElementById(id);
const state = {
  source: null,     // {kind: "default-real"|"fallback-mock"|"file"|"url", label}
  parsed: null,
  dataset: null,
  analysis: null,
  tab: "replay",
  replay: { a2: null, a5: null, task: null },
  playTimer: null,
};

// ---------- DOM helpers ----------
function el(tag, props = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") node.className = v;
    else if (k === "text") node.textContent = v;
    else if (k === "style") Object.assign(node.style, v);
    else if (k.startsWith("on")) node.addEventListener(k.slice(2), v);
    else node.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat()) {
    if (c === null || c === undefined || c === false) continue;
    node.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return node;
}
const SVG_NS = "http://www.w3.org/2000/svg";
function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v === undefined || v === null) continue;
    if (k === "text") node.textContent = v; else node.setAttribute(k, v);
  }
  for (const c of children.flat()) if (c) node.append(c);
  return node;
}

const fmtInt = (v) => (v === null || v === undefined || !Number.isFinite(v) ? "—" : Math.round(v).toLocaleString("en-US"));
const fmtPct = (r, d = 1) => (r === null || r === undefined ? "—" : `${(r * 100).toFixed(d)}%`);
const isMock = () => state.dataset?.mode === "mock";

// ---------- Loading ----------
async function fetchText(path) {
  const res = await fetch(path, { cache: "no-store" });
  if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
  return res.text();
}

async function loadDefault() {
  const param = new URLSearchParams(location.search).get("log");
  if (param) {
    try { return ingest(await fetchText(param), { kind: "url", label: param }); }
    catch (e) { return ingestFailure(`Could not load ?log=${param} (${e.message}).`); }
  }
  try {
    const text = await fetchText(REAL_DEFAULT);
    if (text.trim()) return ingest(text, { kind: "default-real", label: REAL_DEFAULT });
  } catch { /* no real default log yet: fall back to mock below */ }
  try {
    ingest(await fetchText(MOCK_DEFAULT), {
      kind: "fallback-mock", label: MOCK_DEFAULT,
      reason: `No real log at demo/${REAL_DEFAULT}; showing mock fixture.`,
    });
  } catch (e) {
    ingestFailure(`Could not load the mock fixture (${e.message}). Serve this folder over HTTP (python -m http.server) or load a file.`);
  }
}

function readFile(file) {
  if (!file) return;
  const reader = new FileReader();
  reader.onload = () => ingest(String(reader.result), { kind: "file", label: file.name });
  reader.onerror = () => ingestFailure(`Could not read ${file.name}.`);
  reader.readAsText(file);
}

/** Replace the active dataset completely. Nothing from the previous file survives. */
function ingest(text, source) {
  stopReplay();
  const parsed = parseJSONL(text);
  const dataset = isolateDataset(parsed);
  state.source = source;
  state.parsed = parsed;
  state.dataset = dataset;
  state.analysis = dataset.mode === "empty" ? null : analyze(dataset);
  state.replay = { a2: null, a5: null, task: null };
  chooseReplayDefaults();
  renderAll();
}

function ingestFailure(message) {
  stopReplay();
  state.source = { kind: "error", label: message };
  state.parsed = { turns: [], tasks: [], errors: [], accepted: 0, rejected: 0, nonEmpty: 0 };
  state.dataset = { mode: "empty", turns: [], tasks: [], notes: [] };
  state.analysis = null;
  state.replay = { a2: null, a5: null, task: null };
  renderAll();
}

// ---------- Rendering ----------
function renderAll() {
  renderStatus();
  renderBanner();
  renderMetrics();
  renderLegend();
  renderReplay();
  renderQuality();
  renderSweep();
  renderValidation();
}

function auditIssueCount() {
  if (!state.analysis) return 0;
  let n = 0;
  for (const a of state.analysis.audits.values()) n += a.issues.length;
  return n;
}

function renderStatus() {
  const p = state.parsed;
  const btn = $("status");
  const issues = auditIssueCount();
  btn.dataset.state = state.source?.kind === "error" || (p.accepted === 0) ? "error"
    : (p.rejected > 0 || issues > 0 || state.dataset.notes.length) ? "warn" : "ok";
  $("status-text").textContent = `${p.accepted.toLocaleString("en-US")} records · ${p.rejected} rejected`
    + (issues ? ` · ${issues} audit issues` : "");
  const badge = $("mode-badge");
  badge.hidden = !(state.dataset.mode === "mock" || state.dataset.mode === "real");
  badge.dataset.mode = state.dataset.mode;
  badge.textContent = state.dataset.mode === "mock" ? "MOCK DATA" : "REAL DATA";
}

function renderBanner() {
  const b = $("banner");
  b.replaceChildren();
  b.hidden = false;
  const src = state.source;
  if (src?.kind === "error") {
    b.dataset.kind = "error";
    b.append(el("strong", { text: "NO DATA" }), el("span", { text: src.label }));
  } else if (state.dataset.mode === "empty") {
    b.dataset.kind = "error";
    b.append(el("strong", { text: "NO VALID RECORDS" }),
      el("span", { text: `${src.label}: ${state.parsed.nonEmpty} non-empty lines, none matched the CONTRACT.md §3 schema. Open "Data validation" below for line-level reasons.` }));
  } else if (state.dataset.mode === "mock") {
    b.dataset.kind = "mock";
    b.append(el("strong", { text: "MOCK DATA — NOT EXPERIMENTAL RESULTS" }),
      el("span", { text: `${src.reason ? src.reason + " " : ""}Synthetic logs from demo/mock/make_mock.py. Load a real event log to replace them.` }));
  } else {
    b.dataset.kind = "real";
    b.append(el("strong", { text: "REAL DATA" }),
      el("span", { text: `Source: ${src.label} · ${state.analysis.runs.size} runs.` }),
      ...state.dataset.notes.map((n) => el("span", { text: n })));
  }
  if (state.parsed.rejected > 0 && state.dataset.mode !== "empty") {
    const n = state.parsed.rejected;
    b.append(el("span", { text: `${n} ${n === 1 ? "line" : "lines"} rejected — see Data validation.` }));
  }
}

function metric({ label, value, sub, subClass, lead, unavailable, tags = [] }) {
  return el("div", { class: `metric${lead ? " lead" : ""}` },
    el("div", { class: "metric-label" }, label, ...tags),
    el("div", { class: `metric-value${unavailable ? " unavailable" : ""}`, text: value }),
    el("div", { class: `metric-sub${subClass ? " " + subClass : ""}`, text: sub }));
}

function renderMetrics() {
  const box = $("metrics");
  const warn = $("metric-warnings");
  box.replaceChildren();
  warn.replaceChildren();
  const a = state.analysis;
  if (!a) {
    for (const label of ["Prefill ratio · A2 / A5-H1", "Saved / task", "A2 pass", "A5-H1 pass"]) {
      box.append(metric({ label, value: "—", sub: "no data loaded", unavailable: true, lead: label.startsWith("Prefill") }));
    }
    return;
  }
  const h = a.headline;
  const ctx = formatCtxK(a.selection.ctxK);
  const tags = [];
  if (isMock()) tags.push(el("span", { class: "tag mock", text: "MOCK" }));
  if (h.ok && !a.headlineTrusted) tags.push(el("span", { class: "tag warn", text: "CHECK" }));
  box.append(
    metric({
      lead: true, label: "Prefill ratio · A2 / A5-H1", tags,
      value: h.ok ? `${h.ratio.toFixed(2)}×` : "—", unavailable: !h.ok,
      sub: h.ok ? `${h.pctSaved.toFixed(1)}% fewer · ${h.n} paired · ${ctx} ctx` : h.reason,
      subClass: h.ok ? "" : "warn",
    }),
    metric({
      label: "Saved / task", value: h.ok ? fmtInt(h.savedPerTask) : "—", unavailable: !h.ok,
      sub: h.ok ? "prefill tokens (mean)" : "needs A2 + A5-H1",
    }),
    passMetric("A2 pass", a.a2Pass, null),
    passMetric("A5-H1 pass", a.a5Pass, a.a2Pass),
  );

  if (isMock()) {
    warn.append(el("p", { class: "note warn", text: "Every figure above is computed from the mock fixture. It demonstrates the dashboard, not RoleKV." }));
  }
  if (h.ok && !a.headlineTrusted) {
    const parts = a.headAudit.filter((x) => !x.audit.clean).map((x) => {
      const c = x.audit.counts;
      const bits = Object.entries(c).filter(([k, v]) => k !== "ok" && v > 0).map(([k, v]) => `${v} ${k.replace("_", " ")}`);
      return `${x.run.run_id}: ${bits.join(", ")}`;
    });
    warn.append(el("p", { class: "note warn" },
      el("b", { text: "Accounting check failed. " }),
      `Headline uses the task-level prefill_tokens_total field as logged; it could not be verified against Σ(prefill_tokens + canon_tokens) per turn. ${parts.join(" · ")}.`));
  }
  if (a.derivedHeadline?.ok) {
    const d = a.derivedHeadline;
    warn.append(el("p", { class: "note warn" },
      el("b", { text: "Derived (turn-level, includes canonicalization): " }),
      `${d.ratio.toFixed(2)}× · ${d.pctSaved.toFixed(1)}% fewer · n=${d.n}. Rule: per task, Σ over turns of prefill_tokens + canon_tokens. Shown because some task totals omit canon_tokens.`));
  }
  if (a.selection.ambiguous) {
    warn.append(el("p", { class: "note warn", text: a.selection.notes[0] }));
  }
}

function passMetric(label, pr, baseline) {
  if (!pr || pr.n === 0) return metric({ label, value: "—", sub: "method not in log", unavailable: true });
  let sub = `${pr.passed} / ${pr.n} tasks`;
  let subClass = "";
  if (baseline && baseline.n > 0) {
    const diff = (pr.rate - baseline.rate) * 100;
    sub = `${pr.passed} / ${pr.n} · ${diff >= 0 ? "+" : "−"}${Math.abs(diff).toFixed(1)} pts vs A2`;
    subClass = diff < 0 ? "warn" : "";
  }
  return metric({ label, value: fmtPct(pr.rate), sub, subClass });
}

function renderLegend() {
  const ul = $("legend");
  ul.replaceChildren();
  const present = new Set(state.analysis ? [...state.analysis.runs.values()].map((r) => r.method) : []);
  for (const m of METHODS) {
    const meta = METHOD_META[m];
    const dot = el("span", { class: `dot${m === "A5-H2" ? " ring" : ""}`, style: m === "A5-H2" ? { borderColor: meta.color } : { background: meta.color } });
    const cls = [present.has(m) ? "" : "absent", m === "A5-H1" ? "focus" : ""].join(" ").trim();
    ul.append(el("li", { class: cls || null, title: `${meta.desc}${present.has(m) ? "" : " — not in this log"}` },
      dot, `${m} ${meta.label}`));
  }
}

function emptyState(title, body) {
  return el("div", { class: "empty" }, el("h3", { text: title }), el("p", { text: body }));
}

// ---------- Session replay ----------
function runsOf(method) {
  if (!state.analysis) return [];
  return [...state.analysis.runs.values()].filter((r) => r.method === method && r.turns.size > 0)
    .sort((a, b) => (a.ctxK ?? 0) - (b.ctxK ?? 0) || a.run_id.localeCompare(b.run_id));
}

function chooseReplayDefaults() {
  const a = state.analysis;
  if (!a) return;
  const sel = a.selection.byMethod;
  const pick = (method) => {
    const preferred = sel[method]?.run;
    if (preferred && preferred.turns.size) return preferred.run_id;
    return runsOf(method)[0]?.run_id ?? null;
  };
  state.replay.a2 = pick("A2");
  state.replay.a5 = pick("A5-H1");
  pickReplayTask();
}

function pickReplayTask() {
  const { runs } = state.analysis;
  const keys = replayTaskKeys(runs.get(state.replay.a2), runs.get(state.replay.a5));
  if (!keys.includes(state.replay.task)) {
    // Default to the task with the most role switches, which shows the contrast best.
    const a2 = runs.get(state.replay.a2);
    state.replay.task = keys.slice().sort((x, y) =>
      (a2.turns.get(y).length - a2.turns.get(x).length) || compareTaskKeys(x, y))[0] ?? null;
  }
  return keys;
}

function runSelect(label, method, current, onChange) {
  const runs = runsOf(method);
  const select = el("select", { onchange: (e) => onChange(e.target.value), "aria-label": `${method} run` },
    runs.map((r) => {
      const o = el("option", { value: r.run_id, text: `${r.run_id} · ${r.machine}` });
      o.selected = r.run_id === current;
      return o;
    }));
  return el("label", {}, label, select);
}

function renderReplay() {
  const view = $("view-replay");
  view.replaceChildren();
  const a = state.analysis;
  if (!a) { view.append(emptyState("No data", "Load a JSONL event log to replay a session.")); return; }
  if (!state.replay.a2 || !state.replay.a5) {
    const missing = [!state.replay.a2 && "A2", !state.replay.a5 && "A5-H1"].filter(Boolean).join(" and ");
    view.append(emptyState("Replay unavailable", `This log has no turn records for ${missing}. Session Replay compares A2 against A5-H1 on the same task.`));
    return;
  }
  const keys = pickReplayTask();
  const taskSelect = el("select", { "aria-label": "Task", onchange: (e) => { state.replay.task = e.target.value; stopReplay(); renderReplay(); } },
    keys.map((k) => { const o = el("option", { value: k, text: `#${k}` }); o.selected = k === state.replay.task; return o; }));

  const playing = state.playTimer !== null;
  const controls = el("div", { class: "controls" },
    el("label", {}, "Task", taskSelect),
    runsOf("A2").length > 1 ? runSelect("A2 run", "A2", state.replay.a2, (v) => { state.replay.a2 = v; stopReplay(); renderReplay(); }) : null,
    runsOf("A5-H1").length > 1 ? runSelect("A5-H1 run", "A5-H1", state.replay.a5, (v) => { state.replay.a5 = v; stopReplay(); renderReplay(); }) : null,
    el("button", { type: "button", class: "btn primary small", id: "play-btn", onclick: togglePlay, disabled: !keys.length || null },
      playing ? "❚❚ Pause" : "▶ Replay"),
    el("button", { type: "button", class: "btn small", onclick: resetReplay, disabled: !keys.length || null }, "Reset"));
  const key = el("div", { class: "seg-key" },
    el("span", {}, el("i", { class: "swatch sw-recomputed" }), "Recomputed"),
    el("span", {}, el("i", { class: "swatch sw-reused" }), "Reused from cache"),
    el("span", {}, el("i", { class: "swatch sw-canon" }), "Canonicalized at handoff"));
  view.append(el("div", { class: "view-head" }, controls, key));

  if (!keys.length) {
    view.append(emptyState("No common tasks", `${state.replay.a2} and ${state.replay.a5} share no task_id with turn records.`));
    return;
  }
  const runA2 = a.runs.get(state.replay.a2), runA5 = a.runs.get(state.replay.a5);
  const turnsA2 = runA2.turns.get(state.replay.task), turnsA5 = runA5.turns.get(state.replay.task);
  // One scale for both panels so bar lengths are comparable side by side.
  const maxTokens = Math.max(1, ...[...turnsA2, ...turnsA5].map((t) => t.prefill_tokens + t.reused_tokens + t.canon_tokens));
  view.append(el("div", { class: "panels" },
    replayPanel(runA2, turnsA2, maxTokens, "A2 · Per-role LoRA", ""),
    replayPanel(runA5, turnsA5, maxTokens, "A5-H1 · aLoRA + canonicalization", "a5")));
  view.append(el("div", { class: "replay-note" },
    el("p", { class: "note", text: "Bar length = tokens the turn touched, on one shared scale. Coral is prefill computed before decoding, cyan is KV served from the existing cache, violet is the previous role's header + output recomputed with base weights after the turn (H1). Decoded tokens are listed separately and never counted as prefill." }),
    el("p", { class: "note mono", text: `TTFT shown for machine=${TIMING_MACHINE} rows only` })));
}

function replayPanel(run, turns, maxTokens, title, extraClass) {
  const task = run.tasks.get(state.replay.task);
  const totals = turnTotals(turns);
  const timed = run.machine === TIMING_MACHINE;
  const outcome = task ? el("span", { class: task.passed ? "outcome-pass" : "outcome-fail", text: task.passed ? "PASS" : "FAIL" })
    : el("span", { class: "outcome-fail", text: "no task record" });
  const meta = METHOD_META[run.method];
  return el("div", { class: `card panel ${extraClass}` },
    el("div", { class: "card-head" },
      el("div", { class: "panel-title" }, el("span", { class: "dot", style: { background: meta.color } }), el("h3", { class: "card-title", text: title })),
      el("div", { class: "card-meta" }, `${run.machine} · task #${state.replay.task} · `, outcome)),
    el("div", { class: "panel-stats" },
      el("span", {}, el("b", { text: fmtInt(totals.prefill) }), " prefill"),
      el("span", {}, el("b", { text: fmtInt(totals.reused) }), " reused"),
      el("span", {}, el("b", { text: fmtInt(totals.canon) }), " canon"),
      el("span", {}, el("b", { text: fmtInt(totals.compute) }), " computed"),
      el("span", {}, el("b", { text: fmtInt(totals.decode) }), " decoded")),
    el("div", { class: "turns" }, turns.map((t) => turnRow(t, maxTokens))),
    el("div", { class: "panel-foot", text: `${run.run_id}${timed ? "" : ` · timing hidden: ${run.machine} is not comparable with ${TIMING_MACHINE}`}` }));
}

function turnRow(t, maxTokens) {
  const seg = turnSegments(t);
  const pct = (v) => `${(100 * v / maxTokens).toFixed(2)}%`;
  const parts = [];
  if (seg.reused) parts.push(el("span", { class: "seg-reused", style: { width: pct(seg.reused) } }));
  if (seg.recomputed) parts.push(el("span", { class: "seg-recomputed", style: { width: pct(seg.recomputed), minWidth: "2px" } }));
  if (seg.canon) parts.push(el("span", { class: "seg-canon", style: { width: pct(seg.canon), minWidth: "2px" } }));
  const firstLine = (t.text ?? "").split("\n")[0];
  let verdict = null;
  if (t.role === "reviewer" && t.text) {
    if (firstLine.startsWith("APPROVE")) verdict = el("span", { class: "verdict approve", text: "APPROVE" });
    else if (firstLine.startsWith("REVISE")) verdict = el("span", { class: "verdict revise", text: "REVISE" });
  }
  const timing = ttftEligible(t)
    ? el("div", { class: "turn-time", text: `${t.ttft_ms.toFixed(0)} ms`, title: "Time to first token" })
    : el("div", { class: "turn-time na", text: "—", title: t.machine === TIMING_MACHINE ? "ttft_ms missing" : `Timing unavailable for cross-machine comparison (${t.machine})` });
  const row = el("div", { class: "turn", "data-turn": t.turn },
    el("div", { class: "turn-label" }, `T${t.turn} ${t.role === "reviewer" ? "review" : "coder"}`, verdict),
    el("div", { class: "bar", role: "img", "aria-label": `Turn ${t.turn}: ${seg.recomputed} recomputed, ${seg.reused} reused, ${seg.canon} canonicalized` }, parts),
    el("div", { class: "turn-dec", text: `+${t.decode_tokens} dec`, title: "Decoded tokens" }),
    timing);
  attachTooltip(row, () => [
    `T${t.turn} · ${t.role}`,
    `prefill ${fmtInt(t.prefill_tokens)} · reused ${fmtInt(t.reused_tokens)} · canon ${fmtInt(t.canon_tokens)}`,
    `decoded ${fmtInt(t.decode_tokens)}${t.kv_mb !== null ? ` · KV ${t.kv_mb} MB` : ""}`,
    firstLine ? `“${firstLine.slice(0, 90)}${firstLine.length > 90 ? "…" : ""}”` : null,
  ]);
  return row;
}

function reducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function togglePlay() {
  if (state.playTimer !== null) { stopReplay(); setPlayLabel(); return; }
  const rows = [...document.querySelectorAll("#view-replay .turn")];
  if (!rows.length) return;
  if (reducedMotion()) { rows.forEach((r) => r.classList.remove("pending")); return; }
  const maxTurn = Math.max(...rows.map((r) => Number(r.dataset.turn)));
  rows.forEach((r) => r.classList.add("pending"));
  let step = 0;
  const reveal = () => {
    rows.filter((r) => Number(r.dataset.turn) === step).forEach((r) => r.classList.remove("pending"));
    step++;
    if (step > maxTurn) { stopReplay(); setPlayLabel(); }
  };
  state.playTimer = setInterval(reveal, STEP_MS);
  setTimeout(reveal, 80);
  setPlayLabel();
}

function setPlayLabel() {
  const b = $("play-btn");
  if (b) b.textContent = state.playTimer !== null ? "❚❚ Pause" : "▶ Replay";
}

function stopReplay() {
  if (state.playTimer !== null) clearInterval(state.playTimer);
  state.playTimer = null;
}

function resetReplay() {
  stopReplay();
  document.querySelectorAll("#view-replay .turn").forEach((r) => r.classList.remove("pending"));
  setPlayLabel();
}

// ---------- Charts ----------
/** Round-step axis from 0 to at least maxVal, about `count` intervals. */
function niceTicks(maxVal, count = 4) {
  const raw = (maxVal > 0 ? maxVal : 1) / count;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * p).find((s) => s >= raw);
  const max = Math.ceil((maxVal > 0 ? maxVal : 1) / step) * step;
  const ticks = [];
  for (let v = 0; v <= max + step / 2; v += step) ticks.push(+v.toFixed(6));
  return { max, ticks };
}
const kfmt = (v) => (v >= 1000 ? `${+(v / 1000).toFixed(1)}k` : `${Math.round(v)}`);

/** Chart width in CSS pixels, so SVG text renders at its design size instead of scaling with the card. */
function chartWidth() {
  const main = document.querySelector("main");
  const gutter = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--gutter")) || 40;
  return Math.max(300, Math.min(1320, main.clientWidth - 2 * gutter - 42));
}

/** Merge labels of points closer than `gap` px so coincident methods read "A0 · A3" instead of overprinting. */
function clusterLabels(items, gap = 16) {
  const groups = [];
  for (const it of items) {
    const g = groups.find((gr) => gr.some((o) => Math.hypot(o.x - it.x, o.y - it.y) < gap));
    if (g) g.push(it); else groups.push([it]);
  }
  return groups.map((g) => ({
    x: Math.max(...g.map((o) => o.x)), y: mean(g.map((o) => o.y)), text: g.map((o) => o.label).join(" · "),
  }));
}

function chartFrame({ w, h, m, xMax, yMin, yMax, xTicks, yTicks, xFmt, yFmt, xTitle, yTitle }) {
  const X = (v) => m.l + (v / xMax) * (w - m.l - m.r);
  const Y = (v) => h - m.b - ((v - yMin) / (yMax - yMin)) * (h - m.t - m.b);
  const root = svg("svg", { class: "chart", viewBox: `0 0 ${w} ${h}`, width: w, height: h, role: "group" });
  for (const t of yTicks) {
    root.append(svg("line", { class: t === yMin ? "axis" : "grid", x1: m.l, x2: w - m.r, y1: Y(t), y2: Y(t) }));
    root.append(svg("text", { class: "tick", x: m.l - 8, y: Y(t) + 3, "text-anchor": "end", text: yFmt(t) }));
  }
  for (const t of xTicks) {
    root.append(svg("text", { class: "tick", x: X(t), y: h - m.b + 16, "text-anchor": "middle", text: xFmt(t) }));
  }
  root.append(svg("text", { class: "axis-title", x: (m.l + w - m.r) / 2, y: h - 6, "text-anchor": "middle", text: xTitle }));
  root.append(svg("text", { class: "axis-title", x: 14, y: (m.t + h - m.b) / 2, "text-anchor": "middle", transform: `rotate(-90 14 ${(m.t + h - m.b) / 2})`, text: yTitle }));
  return { root, X, Y };
}

function pointGroup(x, y, r, meta, label, tipLines, ring) {
  const g = svg("g", { class: "pt", tabindex: "0", role: "img", "aria-label": tipLines.filter(Boolean).join(", ") },
    svg("circle", ring
      ? { cx: x, cy: y, r, fill: "none", stroke: meta.color, "stroke-width": 2 }
      : { cx: x, cy: y, r, fill: meta.color }));
  attachTooltip(g, () => tipLines);
  return g;
}

function renderQuality() {
  const view = $("view-quality");
  view.replaceChildren();
  const a = state.analysis;
  if (!a) { view.append(emptyState("No data", "Load a JSONL event log to compare methods.")); return; }
  const rows = a.rows.filter((r) => r.available);
  const ctx = formatCtxK(a.selection.ctxK);
  const card = el("div", { class: "card" },
    el("div", { class: "card-head" }, el("h3", { class: "card-title", text: "Cost vs Quality" }),
      el("div", { class: "card-meta", text: `${ctx} context runs only · sweep runs excluded · pass rate = pass@1 (greedy)` })));
  if (!rows.length) {
    card.append(emptyState("No eligible runs", `No task records at ${ctx} context.`));
    view.append(card);
    return;
  }
  const w = chartWidth(), h = Math.round(Math.min(400, Math.max(260, w * 0.36))), m = { l: 64, r: 90, t: 16, b: 46 };
  const { max: xMax, ticks: xTicks } = niceTicks(Math.max(...rows.map((r) => r.meanPrefill)) * 1.08);
  const rates = rows.map((r) => r.rate);
  const yMin = Math.max(0, Math.floor((Math.min(...rates) - 0.05) * 10) / 10);
  const yMax = Math.min(1, Math.ceil((Math.max(...rates) + 0.05) * 10) / 10);
  const yTicks = []; for (let v = yMin; v <= yMax + 1e-9; v += 0.1) yTicks.push(+v.toFixed(2));
  const { root, X, Y } = chartFrame({ w, h, m, xMax, yMin, yMax, xTicks, yTicks, xFmt: kfmt, yFmt: (v) => `${Math.round(v * 100)}%`, xTitle: "Mean prefill tokens per task (incl. canonicalization) →", yTitle: "Pass rate" });
  const items = [];
  // Draw A5-H1 last so it stays on top where methods coincide.
  for (const r of [...rows].sort((p, q) => (p.method === "A5-H1") - (q.method === "A5-H1"))) {
    const meta = METHOD_META[r.method];
    const x = X(r.meanPrefill), y = Y(r.rate);
    const tip = [`${r.method} · ${meta.label}`, `pass ${fmtPct(r.rate)} (${r.passed}/${r.n})`, `mean prefill ${fmtInt(r.meanPrefill)} tokens/task`, `${r.run.run_id} · ${r.run.machine}`];
    root.append(pointGroup(x, y, r.method === "A5-H1" || r.method === "A2" ? 7 : 6, meta, r.method, tip, r.method === "A5-H2"));
    items.push({ x, y, label: r.method });
  }
  for (const l of clusterLabels(items)) {
    root.append(svg("text", { class: "pt-label", x: l.x + 12, y: l.y + 4, text: l.text }));
  }
  card.append(root);

  const gpuNote = rows.some((r) => r.gpu.value === null);
  const table = el("table", { class: "data" },
    el("thead", {}, el("tr", {}, ["Method", "Pass rate", "Mean prefill tokens/task", `Mean GPU-s/task (${TIMING_MACHINE})`, "Tasks", "Run"].map((t) => el("th", { text: t })))),
    el("tbody", {}, a.rows.map((r) => {
      const meta = METHOD_META[r.method];
      const name = el("td", { style: { color: meta.color }, text: `${r.method} ${meta.label}` });
      if (!r.available) {
        return el("tr", {}, name, el("td", { class: "na", colspan: "5", text: `not in this log at ${ctx} context` }));
      }
      return el("tr", { class: r.method === "A5-H1" ? "focus" : null }, name,
        el("td", { text: `${fmtPct(r.rate)}` }),
        el("td", { text: fmtInt(r.meanPrefill) }),
        r.gpu.value === null ? el("td", { class: "na", text: `— ${r.gpu.reason}` }) : el("td", { text: r.gpu.value.toFixed(2) }),
        el("td", { text: String(r.n) }),
        el("td", { class: "run", text: `${r.run.run_id} · ${r.run.machine}` }));
    })));
  card.append(el("div", { class: "table-wrap" }, table));
  const notes = [...a.selection.notes];
  if (gpuNote) notes.push(`GPU-seconds are shown only for ${TIMING_MACHINE} runs; other machines are not comparable.`);
  const tasksDiffer = new Set(rows.map((r) => r.n)).size > 1;
  if (tasksDiffer) notes.push("Methods were evaluated on different numbers of tasks; compare pass rates with that in mind.");
  for (const n of notes) card.append(el("p", { class: "note", text: n }));
  view.append(card);
}

function renderSweep() {
  const view = $("view-sweep");
  view.replaceChildren();
  const a = state.analysis;
  if (!a) { view.append(emptyState("No data", "Load a JSONL event log to see the context sweep.")); return; }
  const { points, notes } = a.sweep;
  const card = el("div", { class: "card" },
    el("div", { class: "card-head" }, el("h3", { class: "card-title", text: "Context Sweep" }),
      el("div", { class: "card-meta", text: "A2 − A5-H1 prefill tokens per task · paired by task_id" })));
  if (!points.length) {
    card.append(emptyState("No sweep pairs", "Needs at least one context size with both an A2 and an A5-H1 run sharing task_ids."));
    for (const n of notes) card.append(el("p", { class: "note", text: n }));
    view.append(card);
    return;
  }
  const w = chartWidth(), h = Math.round(Math.min(380, Math.max(240, w * 0.34))), m = { l: 64, r: 30, t: 20, b: 46 };
  const { max: xMax, ticks: xTicks } = niceTicks(Math.max(...points.map((p) => p.ctxMedian)) * 1.05);
  const { max: yMax, ticks: yTicks } = niceTicks(Math.max(...points.map((p) => p.meanSaved ?? 0)) * 1.12);
  const { root, X, Y } = chartFrame({ w, h, m, xMax, yMin: 0, yMax, xTicks, yTicks, xFmt: kfmt, yFmt: kfmt, xTitle: "Median ctx_tokens (shared context) →", yTitle: "Prefill tokens saved / task" });
  const valid = points.filter((p) => p.meanSaved !== null);
  if (valid.length > 1) root.append(svg("polyline", { class: "line", points: valid.map((p) => `${X(p.ctxMedian)},${Y(p.meanSaved)}`).join(" ") }));
  valid.forEach((p, i) => {
    const x = X(p.ctxMedian), y = Y(p.meanSaved);
    const tip = [`${formatCtxK(p.k)} context (median ${fmtInt(p.ctxMedian)} tokens)`, `saved ${fmtInt(p.meanSaved)} tokens/task`, `${fmtPct(p.fractionSaved)} of A2 prefill · n=${p.n} pairs`, `${p.a2Run.run_id} vs ${p.a5Run.run_id}`];
    root.append(pointGroup(x, y, 5, { color: "#3FC1C9" }, "", tip));
    // Alternate below/above the line so neighbouring context sizes don't collide.
    const left = x > w - 160;
    root.append(svg("text", { class: "pt-label", x: left ? x - 10 : x + 10, y: i % 2 ? y - 10 : y + 18, "text-anchor": left ? "end" : "start", text: `${fmtPct(p.fractionSaved)} · n=${p.n}` }));
  });
  card.append(root);
  card.append(el("p", { class: "note", text: "More shared context creates more opportunities to avoid repeated prefill, but the observed savings depend on role outputs and handoff canonicalization." }));
  const table = el("table", { class: "data" },
    el("thead", {}, el("tr", {}, ["Context", "Median ctx_tokens", "Pairs", "A2 prefill/task", "A5-H1 prefill/task", "Saved/task", "Fraction saved", "Runs"].map((t) => el("th", { text: t })))),
    el("tbody", {}, points.map((p) => {
      const pairs = [...p.a2Run.tasks.keys()].filter((k) => p.a5Run.tasks.has(k));
      const mA = mean(pairs.map((k) => p.a2Run.tasks.get(k).prefill_tokens_total));
      const mB = mean(pairs.map((k) => p.a5Run.tasks.get(k).prefill_tokens_total));
      return el("tr", {},
        el("td", { text: `${formatCtxK(p.k)} (${p.a2Run.ctxSource})` }),
        el("td", { text: fmtInt(p.ctxMedian) }),
        el("td", { text: String(p.n) }),
        el("td", { text: fmtInt(mA) }),
        el("td", { text: fmtInt(mB) }),
        el("td", { text: fmtInt(p.meanSaved) }),
        el("td", { text: fmtPct(p.fractionSaved) }),
        el("td", { class: "run", text: `${p.a2Run.run_id} · ${p.a5Run.run_id}` }));
    })));
  card.append(el("div", { class: "table-wrap" }, table));
  for (const n of notes) card.append(el("p", { class: "note warn", text: n }));
  view.append(card);
}

// ---------- Validation drawer ----------
function renderValidation() {
  const body = $("validation-body");
  body.replaceChildren();
  const p = state.parsed;
  body.append(el("p", { class: "note" },
    `Source: ${state.source?.label ?? "—"} · mode: ${state.dataset.mode} · ${p.nonEmpty} non-empty lines · `,
    el("b", { text: `${p.accepted} accepted` }), ` · ${p.rejected} rejected`));
  for (const n of state.dataset.notes) body.append(el("p", { class: "note warn", text: n }));
  if (p.errors.length) {
    const shown = p.errors.slice(0, 200);
    body.append(el("h4", { text: `Rejected lines (${p.errors.length})` }),
      el("ul", {}, shown.map((e) => el("li", { class: "error", text: `line ${e.line}: ${e.reason}` }))),
      p.errors.length > shown.length ? el("p", { class: "note", text: `…and ${p.errors.length - shown.length} more.` }) : null);
  }
  const a = state.analysis;
  if (!a) return;
  const runItems = [...a.runs.values()].sort((x, y) => x.run_id.localeCompare(y.run_id)).map((r) => {
    const au = a.audits.get(r.run_id);
    const c = au.counts;
    const bad = Object.entries(c).filter(([k, v]) => k !== "ok" && v > 0).map(([k, v]) => `${v} ${k.replace("_", " ")}`);
    const text = `${r.run_id} · ${r.method} · ${r.machine} · ${formatCtxK(r.ctxK)} (${r.ctxSource}) · ${r.tasks.size} tasks · accounting: ${c.ok} ok${bad.length ? ", " + bad.join(", ") : ""}${au.orphanTurns ? `, ${au.orphanTurns} tasks with turns but no task record` : ""}`;
    return el("li", { class: au.clean && !au.orphanTurns ? null : "warn", text });
  });
  body.append(el("h4", { text: "Runs and accounting audit" }),
    el("p", { class: "note", text: "Audit rule: task prefill_tokens_total must equal Σ over its turns of (prefill_tokens + canon_tokens), and n_turns must equal the number of turn records." }),
    el("ul", {}, runItems));
  const issues = [...a.audits.entries()].flatMap(([id, au]) => au.issues.map((i) => `${id} task #${i.task_key}: ${i.status}${i.detail ? ` (${i.detail})` : ""}`));
  if (issues.length) {
    body.append(el("h4", { text: `Accounting issues (${issues.length})` }),
      el("ul", {}, issues.slice(0, 200).map((t) => el("li", { class: "warn", text: t }))));
  }
}

// ---------- Tooltip ----------
function attachTooltip(node, linesFn) {
  const tip = $("tooltip");
  const show = (x, y) => {
    tip.replaceChildren(...linesFn().filter(Boolean).map((l) => el("div", { text: l })));
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    tip.style.left = `${Math.min(window.innerWidth - r.width - 8, x + 14)}px`;
    tip.style.top = `${Math.max(8, y - r.height - 10)}px`;
  };
  const hide = () => { tip.hidden = true; };
  node.addEventListener("mousemove", (e) => show(e.clientX, e.clientY));
  node.addEventListener("mouseleave", hide);
  node.addEventListener("focus", () => { const b = node.getBoundingClientRect(); show(b.left + b.width / 2, b.top); });
  node.addEventListener("blur", hide);
}

// ---------- Tabs ----------
function selectTab(name, focus = false) {
  state.tab = name;
  for (const t of ["replay", "quality", "sweep"]) {
    const tab = $(`tab-${t}`), panel = $(`view-${t}`);
    const on = t === name;
    tab.setAttribute("aria-selected", String(on));
    tab.tabIndex = on ? 0 : -1;
    panel.hidden = !on;
    if (on && focus) tab.focus();
  }
  $("tooltip").hidden = true;
}

function initTabs() {
  const names = ["replay", "quality", "sweep"];
  for (const n of names) {
    const tab = $(`tab-${n}`);
    tab.addEventListener("click", () => selectTab(n));
    tab.addEventListener("keydown", (e) => {
      const i = names.indexOf(state.tab);
      if (e.key === "ArrowRight") selectTab(names[(i + 1) % names.length], true);
      else if (e.key === "ArrowLeft") selectTab(names[(i + names.length - 1) % names.length], true);
      else return;
      e.preventDefault();
    });
  }
}

// ---------- Boot ----------
function initInputs() {
  $("file-input").addEventListener("change", (e) => { readFile(e.target.files[0]); e.target.value = ""; });
  $("status").addEventListener("click", () => {
    const d = $("validation");
    d.open = true;
    d.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
  });
  const dz = $("dropzone");
  let depth = 0;
  window.addEventListener("dragenter", (e) => { if (e.dataTransfer?.types?.includes("Files")) { depth++; dz.hidden = false; } });
  window.addEventListener("dragleave", () => { depth = Math.max(0, depth - 1); if (!depth) dz.hidden = true; });
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => {
    e.preventDefault();
    depth = 0;
    dz.hidden = true;
    readFile(e.dataTransfer?.files?.[0]);
  });
}

let resizeTimer = null;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (state.analysis) { renderQuality(); renderSweep(); } }, 150);
});

initTabs();
initInputs();
loadDefault();
