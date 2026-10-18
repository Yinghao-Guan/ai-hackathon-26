// RoleKV demo UI. All numbers come from metrics.js; this file only renders them.
// Loaded text is always inserted with textContent / createTextNode, never as HTML.
import {
  METHOD_META, TIMING_MACHINE, parseJSONL, isolateDataset, analyze,
  replayTaskKeys, turnTotals, ttftEligible, turnSegments, compareTaskKeys, mean,
} from "./metrics.js";

const REAL_DEFAULT = "data/events.jsonl";
const MOCK_DEFAULT = "mock/mock_events.jsonl";
const STEP_MS = 700;

const $ = (id) => document.getElementById(id);
const state = {
  source: null,     // {kind: "default-real"|"fallback-mock"|"file"|"url"|"error", label, reason?}
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
const ctxWords = (k) => (k === null || k === undefined ? "an unknown amount" : `about ${k}k`);
const plural = (n, one, many = `${one}s`) => `${n.toLocaleString("en-US")} ${n === 1 ? one : many}`;
const isMock = () => state.dataset?.mode === "mock";

// Plain-language names for metrics.js audit statuses.
const AUDIT_NAMES = {
  excludes_canon: ["task total leaves out canon tokens", "task totals leave out canon tokens"],
  mismatch: ["task total does not match its turns", "task totals do not match their turns"],
  no_turns: ["task has no turn records", "tasks have no turn records"],
  turn_count: ["task has the wrong turn count", "tasks have the wrong turn count"],
};
const auditPhrase = (status, n) => `${n} ${AUDIT_NAMES[status][n === 1 ? 0 : 1]}`;
const auditSummary = (counts) => Object.entries(counts).filter(([k, v]) => k !== "ok" && v > 0).map(([k, v]) => auditPhrase(k, v));

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
      reason: `No real log at demo/${REAL_DEFAULT}, so the mock fixture is shown.`,
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
  renderFinding();
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
  const issues = auditIssueCount();
  const btn = $("status");
  btn.dataset.state = state.source?.kind === "error" || p.accepted === 0 ? "error"
    : (p.rejected > 0 || issues > 0 || state.dataset.notes.length) ? "warn" : "ok";
  $("status-text").textContent = `${plural(p.accepted, "record")}, ${p.rejected} rejected`
    + (issues ? `, ${plural(issues, "accounting issue")}` : "");
  const badge = $("mode-badge");
  badge.hidden = !(state.dataset.mode === "mock" || state.dataset.mode === "real");
  badge.dataset.mode = state.dataset.mode;
  badge.textContent = state.dataset.mode === "mock" ? "Mock data" : "Real data";
}

function renderBanner() {
  const b = $("banner");
  b.replaceChildren();
  b.hidden = false;
  const src = state.source;
  if (src?.kind === "error") {
    b.dataset.kind = "error";
    b.append(el("strong", { text: "No data loaded." }), el("span", { text: src.label }));
  } else if (state.dataset.mode === "empty") {
    b.dataset.kind = "error";
    b.append(el("strong", { text: "No valid records." }),
      el("span", { text: `${src.label} has ${plural(state.parsed.nonEmpty, "non-empty line")}, and none match the event-log schema in CONTRACT.md. Open Data validation below to see why each line was rejected.` }));
  } else if (state.dataset.mode === "mock") {
    b.dataset.kind = "mock";
    b.append(el("strong", { text: "MOCK DATA — NOT EXPERIMENTAL RESULTS" }),
      el("span", { text: `${src.reason ? src.reason + " " : ""}These synthetic logs come from demo/mock/make_mock.py. Load a real event log to replace them.` }));
  } else {
    b.dataset.kind = "real";
    b.append(el("strong", { text: "Real data" }),
      el("span", { text: `from ${src.label}, ${plural(state.analysis.runs.size, "run")}.` }),
      ...state.dataset.notes.map((n) => el("span", { text: n })));
  }
  if (state.parsed.rejected > 0 && state.dataset.mode !== "empty") {
    b.append(el("span", { text: `${plural(state.parsed.rejected, "line")} rejected; see Data validation.` }));
  }
}

/** The opening: the result written as a sentence, so the ratio never appears without n and pass rates. */
function renderFinding() {
  const p = $("finding");
  const notes = $("finding-notes");
  notes.replaceChildren();
  const a = state.analysis;
  if (!a) {
    p.className = "finding pending";
    p.textContent = "Load an event log to compare what each role switch costs.";
    return;
  }
  const h = a.headline;
  if (!h.ok) {
    p.className = "finding pending";
    p.textContent = `The cost comparison needs A2 and A5-H1 runs on the same tasks: ${h.reason}.`;
  } else {
    p.className = "finding";
    const ctx = ctxWords(a.selection.ctxK);
    let text = `A5-H1 needed ${h.ratio.toFixed(2)}× less prefill than A2: ${fmtInt(h.meanB)} tokens per task instead of ${fmtInt(h.meanA)}, `
      + `across ${plural(h.n, "paired task")} with ${ctx} tokens of shared context.`;
    if (a.a5Pass.n && a.a2Pass.n) {
      text += ` It passed ${a.a5Pass.passed} of ${a.a5Pass.n} tasks; A2 passed ${a.a2Pass.passed} of ${a.a2Pass.n}.`;
    }
    p.textContent = text;
  }

  if (isMock()) {
    notes.append(el("p", { class: "note warn", text: "Computed from the mock fixture. It demonstrates the dashboard, not RoleKV." }));
  }
  if (h.ok && !a.headlineTrusted) {
    const parts = a.headAudit.filter((x) => !x.audit.clean).map((x) => {
      return `in ${x.run.run_id}, ${auditSummary(x.audit.counts).join(", ")}`;
    });
    notes.append(el("p", { class: "note warn" },
      el("b", { text: "Accounting check failed. " }),
      `These figures use prefill_tokens_total as logged, and it does not match the sum of prefill_tokens and canon_tokens over each task's turns: ${parts.join("; ")}.`));
  }
  if (a.derivedHeadline?.ok) {
    const d = a.derivedHeadline;
    notes.append(el("p", { class: "note warn" },
      el("b", { text: "Recomputed from turns, including canonicalization: " }),
      `${d.ratio.toFixed(2)}× (${d.pctSaved.toFixed(1)}% fewer, n=${d.n}). Shown because some task totals leave out canon_tokens.`));
  }
  if (a.selection.ambiguous) notes.append(el("p", { class: "note warn", text: a.selection.notes[0] }));
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
  const pick = (method) => {
    const preferred = a.selection.byMethod[method]?.run;
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
  const select = el("select", { onchange: (e) => onChange(e.target.value) },
    runsOf(method).map((r) => {
      const o = el("option", { value: r.run_id, text: `${r.run_id} (${r.machine})` });
      o.selected = r.run_id === current;
      return o;
    }));
  return el("label", {}, label, select);
}

function renderReplay() {
  const view = $("view-replay");
  view.replaceChildren(
    el("p", { class: "lede", text: "One coding task, turn by turn. A2 re-reads the whole session at every role switch. A5-H1 keeps one shared cache and computes only the new tokens, plus the previous role's output at each handoff." }));
  const a = state.analysis;
  if (!a) { view.append(emptyState("Nothing to replay yet", "Load a JSONL event log with A2 and A5-H1 runs.")); return; }
  if (!state.replay.a2 || !state.replay.a5) {
    const missing = [!state.replay.a2 && "A2", !state.replay.a5 && "A5-H1"].filter(Boolean).join(" and ");
    view.append(emptyState("Replay needs both methods", `This log has no turn records for ${missing}.`));
    return;
  }
  const keys = pickReplayTask();
  const rerender = () => { stopReplay(); renderReplay(); };
  const taskSelect = el("select", { onchange: (e) => { state.replay.task = e.target.value; rerender(); } },
    keys.map((k) => { const o = el("option", { value: k, text: `Task ${k}` }); o.selected = k === state.replay.task; return o; }));
  const controls = el("div", { class: "controls" },
    el("label", {}, el("span", { class: "visually-hidden", text: "Task" }), taskSelect),
    runsOf("A2").length > 1 ? runSelect("A2 run", "A2", state.replay.a2, (v) => { state.replay.a2 = v; rerender(); }) : null,
    runsOf("A5-H1").length > 1 ? runSelect("A5-H1 run", "A5-H1", state.replay.a5, (v) => { state.replay.a5 = v; rerender(); }) : null,
    el("button", { type: "button", class: "btn primary", id: "play-btn", onclick: togglePlay, disabled: !keys.length || null }, "Play replay"),
    el("button", { type: "button", class: "btn", onclick: resetReplay, disabled: !keys.length || null }, "Show all turns"));
  const key = el("div", { class: "key", "aria-label": "Bar colors" },
    el("span", {}, el("i", { class: "swatch sw-recompute" }), "Computed"),
    el("span", {}, el("i", { class: "swatch sw-reuse" }), "Reused from cache"),
    el("span", {}, el("i", { class: "swatch sw-canon" }), "Recomputed at handoff"));
  const surface = el("div", { class: "replay" }, el("div", { class: "replay-head" }, controls, key));
  view.append(surface);

  if (!keys.length) {
    surface.append(emptyState("No shared tasks", `${state.replay.a2} and ${state.replay.a5} have no task_id in common.`));
    return;
  }
  const runA2 = a.runs.get(state.replay.a2), runA5 = a.runs.get(state.replay.a5);
  const turnsA2 = runA2.turns.get(state.replay.task), turnsA5 = runA5.turns.get(state.replay.task);
  surface.append(turnGrid(turnsA2, turnsA5), el("div", { class: "table-wrap" }, totalsTable([[runA2, turnsA2], [runA5, turnsA5]])));
  surface.append(el("div", { class: "replay-foot" },
    el("p", { class: "note", text: "Both methods share one scale. Coral is prefill computed before decoding, cyan is served from the existing cache, violet is the previous role's header and output recomputed with base weights after the turn (H1). Decoded tokens are counted separately, never as prefill." }),
    el("p", { class: "note", text: `First-token times appear for ${TIMING_MACHINE} runs only.` })));
}

function turnGrid(turnsA2, turnsA5) {
  const maxTokens = Math.max(1, ...[...turnsA2, ...turnsA5].map((t) => t.prefill_tokens + t.reused_tokens + t.canon_tokens));
  const grid = el("div", { class: "turns", id: "turn-grid" },
    el("div", { class: "colhead" }), el("div", { class: "colhead" }),
    el("div", { class: "colhead", text: "Tokens the turn processed" }),
    el("div", { class: "colhead num dec", text: "Decoded" }),
    el("div", { class: "colhead num", text: "First token" }));
  const count = Math.max(turnsA2.length, turnsA5.length);
  for (let i = 0; i < count; i++) {
    const t2 = turnsA2[i], t5 = turnsA5[i];
    const ref = t2 ?? t5;
    grid.append(el("div", { class: "turn-label" }, `Turn ${i}`, el("small", { text: ref.role })));
    grid.append(...turnCells("A2", t2, i, maxTokens, "row-start"));
    grid.append(...turnCells("A5-H1", t5, i, maxTokens, "row-end"));
  }
  return grid;
}

function turnCells(method, t, i, maxTokens, rowClass) {
  const cls = (extra) => `cell ${rowClass} ${extra}`;
  // Each run's reviewer can decide differently on the same task, so the verdict sits on the method's own row.
  const line = (t?.text ?? "").split("\n")[0];
  const verdict = t?.role !== "reviewer" ? null
    : line.startsWith("APPROVE") ? el("span", { class: "verdict approve", text: "approve" })
    : line.startsWith("REVISE") ? el("span", { class: "verdict revise", text: "revise" }) : null;
  const methodCell = el("div", { class: cls(`method${method === "A5-H1" ? " a5" : ""}`) }, method, verdict);
  if (!t) {
    return [methodCell, el("div", { class: cls("") }, el("div", { class: "bar absent", text: "no turn in this run" })),
      el("div", { class: cls("num dim dec") }), el("div", { class: cls("num dim") })];
  }
  const seg = turnSegments(t);
  const pct = (v) => `${(100 * v / maxTokens).toFixed(2)}%`;
  const parts = [];
  if (seg.reused) parts.push(el("span", { class: "seg-reuse", style: { width: pct(seg.reused) } }));
  if (seg.recomputed) parts.push(el("span", { class: "seg-recompute", style: { width: pct(seg.recomputed), minWidth: "3px" } }));
  if (seg.canon) parts.push(el("span", { class: "seg-canon", style: { width: pct(seg.canon), minWidth: "3px" } }));
  const barCell = el("div", { class: cls(""), "data-turn": i },
    el("div", { class: "bar", role: "img", "aria-label": `${method} turn ${i}: ${seg.recomputed} computed, ${seg.reused} reused, ${seg.canon} recomputed at handoff` }, parts));
  const firstLine = (t.text ?? "").split("\n")[0];
  attachTooltip(barCell, () => [
    `${method}, turn ${i} (${t.role})`,
    `Computed before decoding: ${fmtInt(t.prefill_tokens)}`,
    `Reused from cache: ${fmtInt(t.reused_tokens)}`,
    `Recomputed at handoff: ${fmtInt(t.canon_tokens)}`,
    `Decoded: ${fmtInt(t.decode_tokens)}${t.kv_mb !== null ? `, KV cache ${t.kv_mb} MB` : ""}`,
    firstLine ? `“${firstLine.slice(0, 90)}${firstLine.length > 90 ? "…" : ""}”` : null,
  ]);
  const timing = ttftEligible(t)
    ? el("div", { class: cls("num"), "data-turn": i, text: `${t.ttft_ms.toFixed(0)} ms` })
    : el("div", { class: cls("num dim"), "data-turn": i, text: "—", title: t.machine === TIMING_MACHINE ? "ttft_ms missing" : `Not comparable: this run is on the ${t.machine}` });
  return [methodCell, barCell, el("div", { class: cls("num dim dec"), "data-turn": i, text: fmtInt(t.decode_tokens) }), timing];
}

function totalsTable(pairs) {
  return el("table", { class: "totals" },
    el("thead", {}, el("tr", {}, ["Whole session", "Computed", "Reused", "Recomputed at handoff", "Decoded", "Result", "Run"]
      .map((t, i) => el("th", { text: t, class: i === 6 ? "run" : null })))),
    el("tbody", {}, pairs.map(([run, turns]) => {
      const tt = turnTotals(turns);
      const task = run.tasks.get(state.replay.task);
      const result = task ? (task.passed ? "passed" : "failed") : "no task record";
      return el("tr", {},
        el("td", { text: `${run.method} ${METHOD_META[run.method].label}` }),
        el("td", { text: fmtInt(tt.compute) }),
        el("td", { text: fmtInt(tt.reused) }),
        el("td", { text: fmtInt(tt.canon) }),
        el("td", { text: fmtInt(tt.decode) }),
        el("td", { class: task?.passed ? "pass" : "fail", text: result }),
        el("td", { class: "run", text: `${run.run_id} on the ${run.machine}` }));
    })));
}

function reducedMotion() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches;
}

function togglePlay() {
  if (state.playTimer !== null) { stopReplay(); setPlayLabel(); return; }
  const grid = $("turn-grid");
  if (!grid) return;
  const cells = [...grid.querySelectorAll("[data-turn]")];
  if (reducedMotion() || !cells.length) return;
  const maxTurn = Math.max(...cells.map((c) => Number(c.dataset.turn)));
  // Collapse instantly, then let only the reveal animate.
  grid.classList.add("instant", "playing");
  cells.forEach((c) => c.classList.add("pending"));
  void grid.offsetWidth;
  grid.classList.remove("instant");
  let step = 0;
  const reveal = () => {
    cells.filter((c) => Number(c.dataset.turn) === step).forEach((c) => c.classList.remove("pending"));
    step++;
    if (step > maxTurn) { stopReplay(); setPlayLabel(); }
  };
  state.playTimer = setInterval(reveal, STEP_MS);
  setTimeout(reveal, 80);
  setPlayLabel();
}

function setPlayLabel() {
  const b = $("play-btn");
  if (b) b.textContent = state.playTimer !== null ? "Pause" : "Play replay";
}

function stopReplay() {
  if (state.playTimer !== null) clearInterval(state.playTimer);
  state.playTimer = null;
}

function resetReplay() {
  stopReplay();
  const grid = $("turn-grid");
  if (grid) {
    grid.querySelectorAll(".pending").forEach((c) => c.classList.remove("pending"));
    grid.classList.remove("playing");
  }
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

/** Chart width in CSS pixels, so SVG text renders at its design size instead of scaling with the page. */
function chartWidth() {
  const main = document.querySelector("main");
  const gutter = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--gutter")) || 48;
  return Math.max(300, Math.min(1200, main.clientWidth - 2 * gutter));
}

/** Merge labels of points closer than `gap` px so coincident methods read "A0, A3" instead of overprinting. */
function clusterLabels(items, gap = 30) {
  const groups = [];
  for (const it of items) {
    const g = groups.find((gr) => gr.some((o) => Math.hypot(o.x - it.x, o.y - it.y) < gap));
    if (g) g.push(it); else groups.push([it]);
  }
  return groups.map((g) => ({
    x: Math.max(...g.map((o) => o.x)), y: mean(g.map((o) => o.y)),
    text: g.map((o) => o.label).join(", "), strong: g.some((o) => o.strong),
  }));
}

function chartFrame({ w, h, m, xMax, yMin, yMax, xTicks, yTicks, xFmt, yFmt, xTitle, yTitle }) {
  const X = (v) => m.l + (v / xMax) * (w - m.l - m.r);
  const Y = (v) => h - m.b - ((v - yMin) / (yMax - yMin)) * (h - m.t - m.b);
  const root = svg("svg", { class: "chart", viewBox: `0 0 ${w} ${h}`, width: w, height: h, role: "group" });
  for (const t of yTicks) {
    root.append(svg("line", { class: t === yMin ? "axis" : "grid", x1: m.l, x2: w - m.r, y1: Y(t), y2: Y(t) }));
    root.append(svg("text", { class: "tick", x: m.l - 10, y: Y(t) + 4, "text-anchor": "end", text: yFmt(t) }));
  }
  for (const t of xTicks) {
    root.append(svg("text", { class: "tick", x: X(t), y: h - m.b + 20, "text-anchor": "middle", text: xFmt(t) }));
  }
  root.append(svg("text", { class: "axis-title", x: (m.l + w - m.r) / 2, y: h - 4, "text-anchor": "middle", text: xTitle }));
  root.append(svg("text", { class: "axis-title", x: 14, y: (m.t + h - m.b) / 2, "text-anchor": "middle", transform: `rotate(-90 14 ${(m.t + h - m.b) / 2})`, text: yTitle }));
  return { root, X, Y };
}

function pointGroup(x, y, r, { fill, stroke }, tipLines) {
  const g = svg("g", { class: "pt", tabindex: "0", role: "img", "aria-label": tipLines.filter(Boolean).join(". ") },
    svg("circle", { cx: x, cy: y, r, fill: fill ?? "none", stroke: stroke ?? "none", "stroke-width": stroke ? 2 : 0 }));
  attachTooltip(g, () => tipLines);
  return g;
}

function renderQuality() {
  const view = $("view-quality");
  view.replaceChildren();
  const a = state.analysis;
  if (!a) { view.append(emptyState("No methods to compare yet", "Load a JSONL event log.")); return; }
  const ctx = ctxWords(a.selection.ctxK);
  view.append(el("p", { class: "lede", text: `One point per method, from its main run with ${ctx} tokens of shared context; context-sweep runs are left out. Further right means more prefill per task, higher means more tasks passed.` }));
  const rows = a.rows.filter((r) => r.available);
  if (!rows.length) { view.append(emptyState("No eligible runs", `No task records at ${ctx} tokens of context.`)); return; }

  const w = chartWidth(), h = Math.round(Math.min(420, Math.max(260, w * 0.36))), m = { l: 64, r: 110, t: 16, b: 50 };
  const { max: xMax, ticks: xTicks } = niceTicks(Math.max(...rows.map((r) => r.meanPrefill)) * 1.08);
  const rates = rows.map((r) => r.rate);
  const yMin = Math.max(0, Math.floor((Math.min(...rates) - 0.05) * 10) / 10);
  const yMax = Math.min(1, Math.ceil((Math.max(...rates) + 0.05) * 10) / 10);
  const yTicks = []; for (let v = yMin; v <= yMax + 1e-9; v += 0.1) yTicks.push(+v.toFixed(2));
  const { root, X, Y } = chartFrame({ w, h, m, xMax, yMin, yMax, xTicks, yTicks, xFmt: kfmt, yFmt: (v) => `${Math.round(v * 100)}%`, xTitle: "Mean prefill tokens per task, including handoff recomputation", yTitle: "Tasks passed" });
  const items = [];
  // The two methods the project compares are filled; the rest are outlined. Draw them last so they sit on top.
  const strong = (mth) => mth === "A2" || mth === "A5-H1";
  for (const r of [...rows].sort((p, q) => strong(p.method) - strong(q.method))) {
    const x = X(r.meanPrefill), y = Y(r.rate);
    const tip = [`${r.method} ${METHOD_META[r.method].label}`, `Passed ${r.passed} of ${r.n} (${fmtPct(r.rate)})`, `Mean prefill ${fmtInt(r.meanPrefill)} tokens per task`, `${r.run.run_id} on the ${r.run.machine}`];
    root.append(pointGroup(x, y, strong(r.method) ? 8 : 6, strong(r.method) ? { fill: "#E8EEF5" } : { stroke: "#93A1B3" }, tip));
    items.push({ x, y, label: r.method, strong: strong(r.method) });
  }
  for (const l of clusterLabels(items)) {
    root.append(svg("text", { class: `pt-label${l.strong ? "" : " quiet"}`, x: l.x + 14, y: l.y + 5, text: l.text }));
  }
  view.append(root);

  const table = el("table", { class: "data" },
    el("thead", {}, el("tr", {}, ["Method", "What it is", "Passed", "Prefill tokens per task", `GPU seconds per task (${TIMING_MACHINE})`, "Tasks", "Run", "Machine"].map((t, i) => el("th", { text: t, class: i === 1 ? "desc" : null })))),
    el("tbody", {}, a.rows.map((r) => {
      const meta = METHOD_META[r.method];
      const head = [el("td", { text: r.method }), el("td", { class: "desc", text: meta.desc })];
      if (!r.available) return el("tr", {}, ...head, el("td", { class: "na", colspan: "6", text: `Not in this log at ${ctx} tokens of context` }));
      return el("tr", { class: strong(r.method) ? "focus" : null }, ...head,
        el("td", { text: fmtPct(r.rate) }),
        el("td", { text: fmtInt(r.meanPrefill) }),
        r.gpu.value === null ? el("td", { class: "na", text: r.gpu.reason === "gpu_s missing" ? "missing" : "not comparable" }) : el("td", { text: r.gpu.value.toFixed(2) }),
        el("td", { text: String(r.n) }),
        el("td", { class: "run", text: r.run.run_id }),
        el("td", { class: "run", text: r.run.machine }));
    })));
  view.append(el("div", { class: "table-wrap" }, table));
  const notes = [...a.selection.notes];
  if (rows.some((r) => r.gpu.value === null)) notes.push(`GPU seconds are compared only between ${TIMING_MACHINE} runs.`);
  if (new Set(rows.map((r) => r.n)).size > 1) notes.push("Methods ran on different numbers of tasks, so compare pass rates with care.");
  view.append(el("div", { class: "after-chart" }, notes.map((n) => el("p", { class: "note", text: n }))));
}

function renderSweep() {
  const view = $("view-sweep");
  view.replaceChildren();
  const a = state.analysis;
  if (!a) { view.append(emptyState("No sweep yet", "Load a JSONL event log.")); return; }
  view.append(el("p", { class: "lede", text: "Prefill tokens A5-H1 saves per task compared with A2, at each shared-context size. Tasks are paired by task_id within each size." }));
  const { points, notes } = a.sweep;
  if (!points.length) {
    view.append(emptyState("No sizes to compare", "The sweep needs at least one context size with A2 and A5-H1 runs that share task_ids."));
    view.append(el("div", { class: "after-chart" }, notes.map((n) => el("p", { class: "note warn", text: n }))));
    return;
  }
  const w = chartWidth(), h = Math.round(Math.min(400, Math.max(240, w * 0.34))), m = { l: 64, r: 30, t: 20, b: 50 };
  const { max: xMax, ticks: xTicks } = niceTicks(Math.max(...points.map((p) => p.ctxMedian)) * 1.05);
  const { max: yMax, ticks: yTicks } = niceTicks(Math.max(...points.map((p) => p.meanSaved ?? 0)) * 1.12);
  const { root, X, Y } = chartFrame({ w, h, m, xMax, yMin: 0, yMax, xTicks, yTicks, xFmt: kfmt, yFmt: kfmt, xTitle: "Shared context per task (median tokens)", yTitle: "Prefill tokens saved per task" });
  const valid = points.filter((p) => p.meanSaved !== null);
  if (valid.length > 1) root.append(svg("polyline", { class: "line", points: valid.map((p) => `${X(p.ctxMedian)},${Y(p.meanSaved)}`).join(" ") }));
  valid.forEach((p, i) => {
    const x = X(p.ctxMedian), y = Y(p.meanSaved);
    const tip = [`${ctxWords(p.k)} tokens of context (median ${fmtInt(p.ctxMedian)})`, `Saved ${fmtInt(p.meanSaved)} tokens per task`, `${fmtPct(p.fractionSaved)} of A2's prefill, ${plural(p.n, "pair")}`, `${p.a2Run.run_id} vs ${p.a5Run.run_id}`];
    root.append(pointGroup(x, y, 6, { fill: "#4CC3CB" }, tip));
    // Alternate below/above the line so neighbouring context sizes don't collide.
    const left = x > w - 170;
    root.append(svg("text", { class: "pt-label", x: left ? x - 12 : x + 12, y: i % 2 ? y - 18 : y + 22, "text-anchor": left ? "end" : "start", text: `${fmtPct(p.fractionSaved, 0)} saved, n=${p.n}` }));
  });
  view.append(root);
  view.append(el("div", { class: "after-chart" },
    el("p", { class: "note", text: "More shared context creates more opportunities to avoid repeated prefill, but the observed savings depend on role outputs and handoff canonicalization." }),
    notes.map((n) => el("p", { class: "note warn", text: n }))));
  const table = el("table", { class: "data" },
    el("thead", {}, el("tr", {}, ["Context size", "Median context tokens", "Pairs", "A2 prefill per task", "A5-H1 prefill per task", "Saved per task", "Share of A2 saved", "Runs"].map((t) => el("th", { text: t })))),
    el("tbody", {}, points.map((p) => {
      const keys = [...p.a2Run.tasks.keys()].filter((k) => p.a5Run.tasks.has(k));
      const mA = mean(keys.map((k) => p.a2Run.tasks.get(k).prefill_tokens_total));
      const mB = mean(keys.map((k) => p.a5Run.tasks.get(k).prefill_tokens_total));
      return el("tr", {},
        el("td", { text: ctxWords(p.k), title: `Size taken from ${p.a2Run.ctxSource}` }),
        el("td", { text: fmtInt(p.ctxMedian) }),
        el("td", { text: String(p.n) }),
        el("td", { text: fmtInt(mA) }),
        el("td", { text: fmtInt(mB) }),
        el("td", { text: fmtInt(p.meanSaved) }),
        el("td", { text: fmtPct(p.fractionSaved) }),
        el("td", { class: "run", text: `${p.a2Run.run_id}, ${p.a5Run.run_id}` }));
    })));
  view.append(el("div", { class: "table-wrap" }, table));
}

// ---------- Validation drawer ----------
function renderValidation() {
  const body = $("validation-body");
  body.replaceChildren();
  const p = state.parsed;
  body.append(el("p", { class: "note" },
    `Source: ${state.source?.label ?? "none"}. Mode: ${state.dataset.mode}. ${plural(p.nonEmpty, "non-empty line")}: `,
    el("b", { text: `${p.accepted} accepted` }), `, ${p.rejected} rejected.`));
  for (const n of state.dataset.notes) body.append(el("p", { class: "note warn", text: n }));
  if (p.errors.length) {
    const shown = p.errors.slice(0, 200);
    body.append(el("h4", { text: `Rejected lines (${p.errors.length})` }),
      el("ul", {}, shown.map((e) => el("li", { class: "error", text: `Line ${e.line}: ${e.reason}` }))),
      p.errors.length > shown.length ? el("p", { class: "note", text: `${p.errors.length - shown.length} more not shown.` }) : null);
  }
  const a = state.analysis;
  if (!a) return;
  const runItems = [...a.runs.values()].sort((x, y) => x.run_id.localeCompare(y.run_id)).map((r) => {
    const au = a.audits.get(r.run_id);
    const bad = auditSummary(au.counts);
    const text = `${r.run_id}: ${r.method} on the ${r.machine}, ${ctxWords(r.ctxK)} tokens of context (from ${r.ctxSource}), ${plural(r.tasks.size, "task")}. `
      + `Accounting: ${au.counts.ok} ok${bad.length ? ", " + bad.join(", ") : ""}${au.orphanTurns ? `, ${au.orphanTurns} tasks with turns but no task record` : ""}.`;
    return el("li", { class: au.clean && !au.orphanTurns ? null : "warn", text });
  });
  body.append(el("h4", { text: "Runs and accounting audit" }),
    el("p", { class: "note", text: "Each task's prefill_tokens_total must equal the sum of prefill_tokens and canon_tokens over its turns, and n_turns must equal its number of turn records." }),
    el("ul", {}, runItems));
  const issues = [...a.audits.entries()].flatMap(([id, au]) => au.issues.map((i) => `${id}, task ${i.task_key}: ${AUDIT_NAMES[i.status][0]}${i.detail ? ` (${i.detail})` : ""}`));
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
