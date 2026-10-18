// RoleKV demo entry. ?present opens the presentation stage; otherwise this renders the team's dashboard.
// All numbers come from metrics.js; this file only renders them.
import {
  METHOD_META, TIMING_MACHINE, replayTaskKeys, turnTotals, ttftEligible, featuredTask,
  replayContextSizes, runFor, mean,
} from "./metrics.js";
import { loadDefault, readFile } from "./data.js";
import {
  $, el, fmtInt, fmtPct, ctxWords, plural, methodName, reducedMotion, announce, attachTooltip,
} from "./ui.js";
import { tapeEl, sweep } from "./tape.js";
import { qualityChart, sweepChart, sweepHeadline } from "./charts.js";

const params = new URLSearchParams(location.search);
const TABS = ["replay", "quality", "sweep"];
const state = {
  source: null, parsed: null, dataset: null, analysis: null,
  tab: TABS.includes(params.get("tab")) ? params.get("tab") : "replay",
  replay: { ctx: params.has("ctx") ? Number(params.get("ctx")) : null, task: params.get("task"), shown: Infinity },
  playId: 0, playing: false,
};

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

/** Keep tab, task and context size in the URL so a view can be linked and survives a reload. */
function syncURL() {
  const p = new URLSearchParams(location.search);
  p.set("tab", state.tab);
  if (state.replay.task !== null) p.set("task", state.replay.task); else p.delete("task");
  if (state.replay.ctx !== null) p.set("ctx", String(state.replay.ctx)); else p.delete("ctx");
  history.replaceState(null, "", `?${p}`);
}

function apply(data) {
  stopReplay();
  Object.assign(state, data);
  chooseReplayDefaults();
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
  selectTab(state.tab);
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
  $("status-text").textContent = p.rejected || issues ? `${plural(p.rejected + issues, "data issue")}` : "Data validation";
  btn.setAttribute("aria-label", `Data validation: ${plural(p.accepted, "record")}, ${p.rejected} rejected${issues ? `, ${plural(issues, "accounting issue")}` : ""}`);
  const tag = $("mode-tag");
  tag.hidden = !(state.dataset.mode === "mock" || state.dataset.mode === "real");
  tag.dataset.mode = state.dataset.mode;
  tag.textContent = state.dataset.mode === "mock" ? "Mock data — not results" : "Real data";
  const log = new URLSearchParams(location.search).get("log");
  $("present-link").href = `?present${log ? `&log=${encodeURIComponent(log)}` : ""}${state.replay.task ? `&task=${encodeURIComponent(state.replay.task)}` : ""}`;
}

/** The banner only speaks when something needs attention; mock mode is carried by the tag and the finding note. */
function renderBanner() {
  const b = $("banner");
  b.replaceChildren();
  const src = state.source;
  b.hidden = true;
  if (src?.kind === "error") {
    b.dataset.kind = "error";
    b.append(el("strong", { text: "No data loaded." }), el("span", { text: src.label }));
  } else if (state.dataset.mode === "empty") {
    b.dataset.kind = "error";
    b.append(el("strong", { text: "No valid records." }),
      el("span", { text: `${src.label} has ${plural(state.parsed.nonEmpty, "non-empty line")}, and none match the event-log schema in CONTRACT.md. Open Data validation below to see why each line was rejected.` }));
  } else if (state.dataset.mode === "real") {
    b.dataset.kind = "real";
    b.append(el("strong", { text: "Real data" }),
      el("span", { text: `from ${src.label}, ${plural(state.analysis.runs.size, "run")}.` }),
      ...state.dataset.notes.map((n) => el("span", { text: n })));
  } else if (state.parsed.rejected === 0) {
    return;
  }
  if (state.parsed.rejected > 0 && state.dataset.mode !== "empty") {
    b.dataset.kind ||= "warn";
    b.append(el("span", { text: `${plural(state.parsed.rejected, "line")} rejected; see Data validation.` }));
  }
  b.hidden = false;
}

/** The opening: the result as one plain sentence, then n and both pass rates, so the saving never travels alone. */
function renderFinding() {
  const h1 = $("finding"), sub = $("finding-sub"), notes = $("finding-notes");
  notes.replaceChildren();
  sub.textContent = "";
  const a = state.analysis;
  if (!a) {
    h1.className = "finding pending";
    h1.textContent = "Load an event log to see what each role switch costs.";
    return;
  }
  const h = a.headline;
  if (!h.ok && a.sweep.points.length) {
    h1.className = "finding pending";
    h1.textContent = "No evaluation runs loaded yet. The Context sweep tab shows what each method computes in a replayed conversation (costs only, no quality).";
  } else if (!h.ok) {
    h1.className = "finding pending";
    h1.textContent = `The comparison needs per-role LoRA (A2) and RoleKV (A5-H1) runs on the same tasks: ${h.reason}.`;
  } else {
    h1.className = "finding";
    const pct = Math.abs(h.pctSaved);
    h1.replaceChildren("RoleKV ", el("span", { class: "nowrap", text: "re-read" }), " ", el("em", { text: `${pct.toFixed(pct < 10 ? 1 : 0)}% ${h.pctSaved >= 0 ? "fewer" : "more"} tokens` }), " than per-role LoRAs.");
    const parts = [
      `${fmtInt(h.meanA)} → ${fmtInt(h.meanB)} tokens per task (${h.ratio.toFixed(2)}× ${h.ratio >= 1 ? "fewer" : "more"})`,
      `${plural(h.n, "paired task")}, ${ctxWords(a.selection.ctxK)} tokens of shared context`,
    ];
    if (a.a5Pass.n && a.a2Pass.n) {
      parts.push(el("span", {}, "tasks passed: ", el("span", { class: "nowrap", text: `RoleKV ${a.a5Pass.passed} of ${a.a5Pass.n}` }),
        ", ", el("span", { class: "nowrap", text: `per-role LoRAs ${a.a2Pass.passed} of ${a.a2Pass.n}` })));
    }
    sub.replaceChildren(...parts.flatMap((t, i) => (i ? [" ", el("span", { class: "sep", "aria-hidden": "true", text: "·\u00A0" }), t] : [t])));
  }

  if (isMock()) {
    notes.append(el("p", { class: "note warn", text: "Computed from the mock fixture. It demonstrates the dashboard, not RoleKV's results." }));
  }
  if (h.ok && !a.headlineTrusted) {
    const parts = a.headAudit.filter((x) => !x.audit.clean).map((x) => `in ${x.run.run_id}, ${auditSummary(x.audit.counts).join(", ")}`);
    notes.append(el("p", { class: "note warn" },
      el("b", { text: "Accounting check failed. " }),
      `These figures use prefill_tokens_total as logged, and it does not match the sum of prefill_tokens and canon_tokens over each task's turns: ${parts.join("; ")}.`));
  }
  if (a.derivedHeadline?.ok) {
    const d = a.derivedHeadline;
    notes.append(el("p", { class: "note warn" },
      el("b", { text: "Recomputed from turns, including handoff re-reads: " }),
      `${d.ratio.toFixed(2)}× (${d.pctSaved.toFixed(1)}% fewer, n=${d.n}). Shown because some task totals leave out canon_tokens.`));
  }
  if (a.selection.ambiguous) notes.append(el("p", { class: "note warn", text: a.selection.notes[0] }));
}

function emptyState(title, body) {
  return el("div", { class: "empty" }, el("h3", { text: title }), el("p", { text: body }));
}

function viewHead(view, title, ...ledes) {
  view.append(el("h2", { text: title }), ...ledes.filter(Boolean).map((t) => el("p", { class: "lede", text: t })));
}

// ---------- Session replay ----------
function replayRuns() {
  const a = state.analysis;
  return { a2: runFor(a.runs, "A2", state.replay.ctx), a5: runFor(a.runs, "A5-H1", state.replay.ctx) };
}

function chooseReplayDefaults() {
  const a = state.analysis;
  if (!a) return;
  const sizes = replayContextSizes(a.runs);
  if (!sizes.includes(state.replay.ctx)) {
    state.replay.ctx = sizes.includes(a.selection.ctxK) ? a.selection.ctxK : (sizes[0] ?? null);
  }
  pickReplayTask();
}

function pickReplayTask() {
  const { a2, a5 } = replayRuns();
  const keys = replayTaskKeys(a2, a5);
  if (!keys.includes(state.replay.task)) state.replay.task = featuredTask(a2, a5);
  return keys;
}

const resultWords = (a2, a5, k) => {
  const p2 = a2.tasks.get(k)?.passed, p5 = a5.tasks.get(k)?.passed;
  if (p2 === undefined || p5 === undefined) return "no result";
  return p2 && p5 ? "both passed" : p5 ? "only RoleKV passed" : p2 ? "only per-role passed" : "both failed";
};

function renderReplay() {
  const view = $("view-replay");
  view.replaceChildren();
  viewHead(view, "One task, turn by turn",
    "Each row is the session so far, drawn to scale. Ink marks the tokens a turn had to read again; the cyan-ruled stretch was reused from the cache for free.");
  const a = state.analysis;
  if (!a) { view.append(emptyState("Nothing to replay yet", "Load a JSONL event log with A2 and A5-H1 runs.")); return; }
  const { a2, a5 } = replayRuns();
  if (!a2 || !a5) {
    const missing = [!runFor(a.runs, "A2", a.selection.ctxK) && "A2", !runFor(a.runs, "A5-H1", a.selection.ctxK) && "A5-H1"].filter(Boolean).join(" and ") || "both methods at one context size";
    view.append(emptyState("Replay needs both methods", `This log has no turn records for ${missing}.`));
    return;
  }
  const keys = pickReplayTask();
  const featured = featuredTask(a2, a5);
  const rerender = () => { stopReplay(); state.replay.shown = Infinity; syncURL(); renderStatus(); renderReplay(); };
  const taskSelect = el("select", { id: "task-select", onchange: (e) => { state.replay.task = e.target.value; rerender(); } },
    keys.map((k) => {
      const o = el("option", { value: k, text: `Task ${k} · ${plural(Math.max(a2.turns.get(k).length, a5.turns.get(k).length), "turn")} · ${resultWords(a2, a5, k)}${k === featured ? " · featured" : ""}` });
      o.selected = k === state.replay.task;
      return o;
    }));
  const sizes = replayContextSizes(a.runs);
  const ctxSelect = sizes.length > 1 ? el("label", { class: "field" }, "Shared context",
    el("select", { onchange: (e) => { state.replay.ctx = Number(e.target.value); pickReplayTask(); rerender(); } },
      sizes.map((k) => { const o = el("option", { value: k, text: `${ctxWords(k)} tokens` }); o.selected = k === state.replay.ctx; return o; }))) : null;
  const scope = el("div", { class: "controls scope" },
    el("label", { class: "field" }, "Task", taskSelect), ctxSelect);

  const turnsA2 = a2.turns.get(state.replay.task) ?? [], turnsA5 = a5.turns.get(state.replay.task) ?? [];
  const count = Math.max(turnsA2.length, turnsA5.length);
  const playback = el("div", { class: "controls playback", role: "group", "aria-label": "Replay" },
    el("button", { type: "button", class: "btn primary", id: "play-btn", onclick: togglePlay, disabled: !count || null }, "Play replay"),
    el("button", { type: "button", class: "btn", id: "prev-btn", onclick: () => step(-1), disabled: !count || null }, "Previous turn"),
    el("button", { type: "button", class: "btn", id: "next-btn", onclick: () => step(1), disabled: !count || null }, "Next turn"),
    el("button", { type: "button", class: "btn quiet", id: "all-btn", onclick: showAll, disabled: !count || null }, "Show all"),
    el("span", { class: "turn-counter", id: "turn-counter" }));
  const key = el("ul", { class: "key", "aria-label": "How to read the tapes" },
    el("li", {}, el("i", { class: "sw ink" }), "Read again this turn"),
    el("li", {}, el("i", { class: "sw cache" }), "Reused from the cache"),
    el("li", {}, el("i", { class: "sw canon" }), "Re-read at handoff"),
    el("li", {}, el("i", { class: "sw written" }), "Written this turn"));
  view.append(el("div", { class: "replay-head" }, scope, playback), key);

  if (!keys.length) { view.append(emptyState("No shared tasks", `${a2.run_id} and ${a5.run_id} have no task_id in common.`)); return; }

  const totA2 = turnTotals(turnsA2), totA5 = turnTotals(turnsA5);
  view.append(el("p", { class: "session-line" },
    "This session: per-role LoRAs read ", el("strong", { class: "n-base", text: `${fmtInt(totA2.compute)} tokens` }),
    "; RoleKV read ", el("strong", { class: "n-hero", text: `${fmtInt(totA5.compute)} tokens` }), "."));
  view.append(turnGrid(turnsA2, turnsA5, count));
  view.append(el("div", { class: "table-wrap" }, totalsTable([[a2, turnsA2], [a5, turnsA5]])));
  view.append(el("p", { class: "note foot", text: `Both methods share one scale. Written tokens are decoded, not prefilled, and never count as re-read. First-token times appear for ${TIMING_MACHINE} runs only.` }));
  applyShown();
}

function turnGrid(turnsA2, turnsA5, count) {
  const scale = Math.max(1, ...[...turnsA2, ...turnsA5].map((t) => t.reused_tokens + t.prefill_tokens + t.decode_tokens));
  const grid = el("div", { class: "turns", id: "turn-grid" },
    el("div", { class: "colhead" }, el("span", { class: "visually-hidden", text: "Turn" })),
    el("div", { class: "colhead" }, el("span", { class: "visually-hidden", text: "Method" })),
    el("div", { class: "colhead", text: "Session so far" }),
    el("div", { class: "colhead num", text: "Read" }),
    el("div", { class: "colhead num ttft", text: "First token" }));
  for (let i = 0; i < count; i++) {
    const ref = turnsA2[i] ?? turnsA5[i];
    grid.append(el("div", { class: "turn-label", "data-turn": i }, `Turn ${i + 1}`, el("small", { text: ref.role })));
    grid.append(...turnCells("A2", turnsA2[i], i, scale, "row-start"));
    grid.append(...turnCells("A5-H1", turnsA5[i], i, scale, "row-end"));
  }
  return grid;
}

function turnCells(method, t, i, scale, rowClass) {
  const cls = (extra) => `cell ${rowClass} ${extra}`;
  // Each run's reviewer can decide differently on the same task, so the verdict sits on the method's own row.
  const line = (t?.text ?? "").split("\n")[0];
  const verdict = t?.role !== "reviewer" ? null
    : line.startsWith("APPROVE") ? el("span", { class: "verdict approve", text: "approve" })
    : line.startsWith("REVISE") ? el("span", { class: "verdict revise", text: "revise" }) : null;
  const methodCell = el("div", { class: cls(`method ${method === "A5-H1" ? "hero" : "base"}`), "data-turn": i },
    el("span", { class: "m-name", text: methodName(method) }), verdict);
  if (!t) {
    return [methodCell, el("div", { class: cls("tape-cell"), "data-turn": i }, el("span", { class: "absent", text: "no turn in this run" })),
      el("div", { class: cls("num dim"), "data-turn": i, text: "—" }), el("div", { class: cls("num dim ttft"), "data-turn": i })];
  }
  const tape = tapeEl(t, scale, { method: methodName(method), index: i });
  const cell = el("div", { class: cls("tape-cell"), "data-turn": i, tabindex: "0" }, tape);
  attachTooltip(cell, () => [
    `${methodName(method)} (${method}), turn ${i + 1}, ${t.role}`,
    `Read again this turn: ${fmtInt(t.prefill_tokens)}`,
    `Reused from the cache: ${fmtInt(t.reused_tokens)}`,
    `Re-read at handoff: ${fmtInt(t.canon_tokens)}`,
    `Written: ${fmtInt(t.decode_tokens)}${t.kv_mb !== null ? ` · KV cache ${t.kv_mb} MB` : ""}`,
    line ? `“${line.slice(0, 90)}${line.length > 90 ? "…" : ""}”` : null,
  ], rowClass === "row-start" ? "above" : "below");
  const timing = ttftEligible(t)
    ? el("div", { class: cls("num ttft"), "data-turn": i, text: `${t.ttft_ms.toFixed(0)} ms` })
    : el("div", { class: cls("num dim ttft"), "data-turn": i, text: "—", title: t.machine === TIMING_MACHINE ? "ttft_ms missing" : `Not comparable: this run is on the ${t.machine}` });
  return [methodCell, cell, el("div", { class: cls(`num read ${method === "A5-H1" ? "hero" : "base"}`), "data-turn": i, text: fmtInt(t.prefill_tokens + t.canon_tokens) }), timing];
}

function totalsTable(pairs) {
  return el("table", { class: "totals" },
    el("caption", { class: "visually-hidden", text: "Whole-session totals" }),
    el("thead", {}, el("tr", {}, ["Whole session", "Read", "Reused", "Re-read at handoff", "Written", "Result", "Run"]
      .map((t, i) => el("th", { scope: "col", text: t, class: i === 6 ? "run" : null })))),
    el("tbody", {}, pairs.map(([run, turns]) => {
      const tt = turnTotals(turns);
      const task = run.tasks.get(state.replay.task);
      const result = task ? (task.passed ? "passed" : "failed") : "no task record";
      return el("tr", {},
        el("th", { scope: "row" }, methodName(run.method), el("small", { text: ` ${run.method}` })),
        el("td", { text: fmtInt(tt.compute) }),
        el("td", { text: fmtInt(tt.reused) }),
        el("td", { text: fmtInt(tt.canon) }),
        el("td", { text: fmtInt(tt.decode) }),
        el("td", { class: task?.passed ? "pass" : "fail", text: result }),
        el("td", { class: "run mono", text: `${run.run_id} · ${run.machine}` }));
    })));
}

// Replay state: the first `shown` turns are drawn; the rest wait as faint paper.
function turnCount() {
  return Math.max(0, ...[...document.querySelectorAll("#turn-grid .turn-label")].map((n) => Number(n.dataset.turn) + 1));
}

function applyShown() {
  const grid = $("turn-grid");
  if (!grid) return;
  const n = turnCount();
  const shown = Math.min(state.replay.shown, n);
  grid.querySelectorAll("[data-turn]").forEach((c) => {
    const pending = Number(c.dataset.turn) >= shown;
    c.classList.toggle("pending", pending);
    c.querySelector(".tape")?.classList.toggle("pending", pending);
  });
  const counter = $("turn-counter");
  if (counter) counter.textContent = shown >= n ? `All ${n} turns` : `Turn ${shown} of ${n}`;
  const prev = $("prev-btn"), next = $("next-btn");
  if (prev) prev.disabled = shown <= 0 || state.playing;
  if (next) next.disabled = shown >= n || state.playing;
  const play = $("play-btn");
  if (play) play.textContent = state.playing ? "Pause" : shown > 0 && shown < n ? "Resume" : "Play replay";
}

async function revealTurn(i, id) {
  const grid = $("turn-grid");
  const cells = [...grid.querySelectorAll(`[data-turn="${i}"]`)];
  cells.forEach((c) => c.classList.remove("pending"));
  state.replay.shown = i + 1;
  const n = turnCount();
  const ref = grid.querySelector(`.turn-label[data-turn="${i}"] small`)?.textContent ?? "";
  announce(`Turn ${i + 1} of ${n}, ${ref}.`);
  const counter = $("turn-counter");
  if (counter) counter.textContent = `Turn ${i + 1} of ${n}`;
  await Promise.all(cells.map((c) => c.querySelector(".tape")).filter(Boolean).map(sweep));
  return id === state.playId;
}

async function togglePlay() {
  if (state.playing) { stopReplay(); applyShown(); return; }
  const n = turnCount();
  if (!n) return;
  if (state.replay.shown >= n) state.replay.shown = 0;
  const id = ++state.playId;
  state.playing = true;
  applyShown();
  for (let i = state.replay.shown; i < n; i++) {
    if (!(await revealTurn(i, id))) return;
    if (!reducedMotion()) await new Promise((r) => setTimeout(r, 420));
    if (id !== state.playId) return;
  }
  state.playing = false;
  applyShown();
  announce(`Replay finished: all ${n} turns shown.`);
}

function step(dir) {
  stopReplay();
  const n = turnCount();
  const from = Math.min(state.replay.shown, n);
  const to = Math.max(0, Math.min(n, from + dir));
  if (dir > 0 && to > from) {
    state.replay.shown = from;
    applyShown();
    revealTurn(from, state.playId).then(applyShown);
  } else {
    state.replay.shown = to;
    applyShown();
    announce(to ? `Turn ${to} of ${n}.` : "No turns shown.");
  }
}

function showAll() {
  stopReplay();
  state.replay.shown = Infinity;
  applyShown();
  announce("All turns shown.");
}

function stopReplay() {
  state.playId++;
  state.playing = false;
}

// ---------- Cost vs quality ----------
function chartWidth() {
  const gutter = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--gutter")) || 48;
  return Math.max(300, Math.min(1100, document.querySelector("main").clientWidth - 2 * gutter));
}

function renderQuality() {
  const view = $("view-quality");
  view.replaceChildren();
  const a = state.analysis;
  viewHead(view, "Cost against quality");
  if (!a) { view.append(emptyState("No methods to compare yet", "Load a JSONL event log.")); return; }
  const ctx = ctxWords(a.selection.ctxK);
  const rows = a.rows.filter((r) => r.available);
  view.append(el("p", { class: "lede", text: `Each point is one method's main run with ${ctx} tokens of shared context; context-sweep runs are left out. Up and to the left is better.` }));
  if (rows.some((r) => r.method === "A1")) {
    view.append(el("p", { class: "lede", text: "One shared LoRA (A1) also keeps a single cache, so it costs about the same as RoleKV. What it gives up is separate role adapters: with RoleKV each role can be retrained or added on its own." }));
  }
  if (!rows.length) { view.append(emptyState("No eligible runs", `No task records at ${ctx} tokens of context.`)); return; }
  const w = chartWidth();
  view.append(qualityChart(a, { w, h: Math.round(Math.min(400, Math.max(260, w * 0.34))) }));

  const table = el("table", { class: "data" },
    el("thead", {}, el("tr", {}, ["Method", "What it is", "Passed", "Re-read per task", `GPU s per task (${TIMING_MACHINE})`, "Tasks", "Run"].map((t, i) => el("th", { scope: "col", text: t, class: i === 1 ? "desc" : i === 6 ? "run" : null })))),
    el("tbody", {}, a.rows.map((r) => {
      const meta = METHOD_META[r.method];
      const kind = r.method === "A5-H1" ? "hero" : r.method === "A2" ? "base" : null;
      const head = [el("th", { scope: "row", class: kind }, methodName(r.method), el("small", { text: ` ${r.method}` })), el("td", { class: "desc", text: meta.desc })];
      if (!r.available) return el("tr", {}, ...head, el("td", { class: "na", colspan: "5", text: `Not in this log at ${ctx} tokens of context` }));
      return el("tr", { class: kind ? "focus" : null }, ...head,
        el("td", { text: fmtPct(r.rate) }),
        el("td", { text: fmtInt(r.meanPrefill) }),
        r.gpu.value === null ? el("td", { class: "na", text: r.gpu.reason === "gpu_s missing" ? "missing" : "not comparable" }) : el("td", { text: r.gpu.value.toFixed(2) }),
        el("td", { text: String(r.n) }),
        el("td", { class: "run mono", text: `${r.run.run_id} · ${r.run.machine}` }));
    })));
  view.append(el("div", { class: "table-wrap" }, table));
  const notes = [...a.selection.notes];
  if (rows.some((r) => r.gpu.value === null)) notes.push(`GPU seconds are compared only between ${TIMING_MACHINE} runs.`);
  if (new Set(rows.map((r) => r.n)).size > 1) notes.push("Methods ran on different numbers of tasks, so compare pass rates with care.");
  view.append(el("div", { class: "after-chart" }, notes.map((n) => el("p", { class: "note", text: n }))));
}

// ---------- Context sweep ----------
function renderSweep() {
  const view = $("view-sweep");
  view.replaceChildren();
  const a = state.analysis;
  if (!a) { viewHead(view, "Saving by context size"); view.append(emptyState("No sweep yet", "Load a JSONL event log.")); return; }
  const { points, notes } = a.sweep;
  viewHead(view, sweepHeadline(points), "Tokens RoleKV avoids re-reading per task, compared with per-role LoRAs, at each shared-context size. Tasks are paired by task_id within each size.");
  if (!points.length) {
    view.append(emptyState("No sizes to compare", "The sweep needs at least one context size with A2 and A5-H1 runs that share task_ids."));
    view.append(el("div", { class: "after-chart" }, notes.map((n) => el("p", { class: "note warn", text: n }))));
    return;
  }
  const w = chartWidth();
  view.append(sweepChart(a, { w, h: Math.round(Math.min(380, Math.max(240, w * 0.32))) }));
  view.append(el("div", { class: "after-chart" }, notes.map((n) => el("p", { class: "note warn", text: n }))));
  const table = el("table", { class: "data" },
    el("thead", {}, el("tr", {}, ["Context size", "Median context tokens", "Pairs", "Per-role re-read per task", "RoleKV re-read per task", "Saved per task", "Share saved", "Runs"].map((t, i) => el("th", { scope: "col", text: t, class: i === 7 ? "run" : null })))),
    el("tbody", {}, points.map((p) => {
      const keys = [...p.a2Run.tasks.keys()].filter((k) => p.a5Run.tasks.has(k));
      const mA = mean(keys.map((k) => p.a2Run.tasks.get(k).prefill_tokens_total));
      const mB = mean(keys.map((k) => p.a5Run.tasks.get(k).prefill_tokens_total));
      return el("tr", {},
        el("th", { scope: "row", text: ctxWords(p.k), title: `Size taken from ${p.a2Run.ctxSource}` }),
        el("td", { text: fmtInt(p.ctxMedian) }),
        el("td", { text: String(p.n) }),
        el("td", { text: fmtInt(mA) }),
        el("td", { text: fmtInt(mB) }),
        el("td", { text: fmtInt(p.meanSaved) }),
        el("td", { text: fmtPct(p.fractionSaved) }),
        el("td", { class: "run mono", text: `${p.a2Run.run_id}, ${p.a5Run.run_id}` }));
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
  if (isMock()) body.append(el("p", { class: "note warn", text: `${state.source?.reason ? state.source.reason + " " : ""}These synthetic logs come from demo/mock/make_mock.py. Load a real event log to replace them.` }));
  for (const n of state.dataset.notes) body.append(el("p", { class: "note warn", text: n }));
  if (p.errors.length) {
    const shown = p.errors.slice(0, 200);
    body.append(el("h3", { text: `Rejected lines (${p.errors.length})` }),
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
  body.append(el("h3", { text: "Runs and accounting audit" }),
    el("p", { class: "note", text: "Each task's prefill_tokens_total must equal the sum of prefill_tokens and canon_tokens over its turns, and n_turns must equal its number of turn records." }),
    el("ul", {}, runItems));
  const issues = [...a.audits.entries()].flatMap(([id, au]) => au.issues.map((i) => `${id}, task ${i.task_key}: ${AUDIT_NAMES[i.status][0]}${i.detail ? ` (${i.detail})` : ""}`));
  if (issues.length) {
    body.append(el("h3", { text: `Accounting issues (${issues.length})` }),
      el("ul", {}, issues.slice(0, 200).map((t) => el("li", { class: "warn", text: t }))));
  }
}

// ---------- Tabs ----------
function selectTab(name, focus = false) {
  state.tab = name;
  for (const t of TABS) {
    const tab = $(`tab-${t}`), panel = $(`view-${t}`);
    const on = t === name;
    tab.setAttribute("aria-selected", String(on));
    tab.tabIndex = on ? 0 : -1;
    panel.hidden = !on;
    if (on && focus) tab.focus();
  }
  $("tooltip").hidden = true;
  if (state.analysis) syncURL();
}

function initTabs() {
  for (const n of TABS) {
    const tab = $(`tab-${n}`);
    tab.addEventListener("click", () => selectTab(n));
    tab.addEventListener("keydown", (e) => {
      const i = TABS.indexOf(state.tab);
      if (e.key === "ArrowRight") selectTab(TABS[(i + 1) % TABS.length], true);
      else if (e.key === "ArrowLeft") selectTab(TABS[(i + TABS.length - 1) % TABS.length], true);
      else if (e.key === "Home") selectTab(TABS[0], true);
      else if (e.key === "End") selectTab(TABS[TABS.length - 1], true);
      else return;
      e.preventDefault();
    });
  }
}

// ---------- Boot ----------
function initInputs() {
  const load = async (file) => { if (file) apply(await readFile(file)); };
  $("file-input").addEventListener("change", (e) => { load(e.target.files[0]); e.target.value = ""; });
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
    load(e.dataTransfer?.files?.[0]);
  });
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => { if (state.analysis) { renderQuality(); renderSweep(); selectTab(state.tab); } }, 150);
  });
}

if (params.has("present")) {
  import("./present.js").then((m) => m.start());
} else {
  initTabs();
  initInputs();
  loadDefault().then(apply);
}
