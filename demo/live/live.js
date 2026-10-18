// RoleKV live demo: one task on one model, Per-role LoRA (A2) and RoleKV (A5-H1) side by side.
// demo/live/server.py streams one Server-Sent Event per finished turn; this file only renders them.
// Model output is always inserted with textContent, never as HTML.
import { $, el, fmtInt, reducedMotion, announce, attachTooltip } from "../js/ui.js";
import { tapeEl, sweep } from "../js/tape.js";
import { tapeSpans } from "../js/metrics.js";

const METHODS = ["A2", "A5-H1"];
const COL = { "A2": 1, "A5-H1": 2 };
const CLS = { "A2": "a2", "A5-H1": "hero" };
const NAME = { "A2": "Per-role LoRA", "A5-H1": "RoleKV" };
const ROLE = { coder: "Coder", reviewer: "Reviewer" };

const state = {
  info: null,
  tasks: [],
  ctx: 30,
  running: false,
  run: null, // { start, turns: {method: [event]}, finals: {method: event}, cards: Map, pending: Map, scale, t0, wall }
  timer: null,
  log: null, // where the server records finished runs for the dashboard (live_log.py summary)
};

// ---------- small helpers ----------

const sum = (xs, f) => xs.reduce((a, x) => a + f(x), 0);
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);
const fmtMs = (ms) => (ms === null ? "—" : ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`);
const fmtS = (s) => (s === null ? "—" : `${s.toFixed(s < 10 ? 2 : 1)} s`);
const fmtUsd = (v) => (v === null ? "—" : v >= 100 ? `$${fmtInt(v)}` : v >= 1 ? `$${v.toFixed(2)}` : `$${v.toPrecision(2)}`);
const shortModel = (m) => String(m).split("/").pop();
const rate = () => {
  const v = parseFloat($("rate").value);
  return Number.isFinite(v) && v >= 0 ? v : state.info?.default_usd_per_gpu_hour ?? 0;
};

async function getJSON(url) {
  const res = await fetch(url, { cache: "no-store" });
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return res.json();
}

function banner(kind, title, detail, action) {
  const b = $("banner");
  if (!kind) { b.hidden = true; b.replaceChildren(); return; }
  b.dataset.kind = kind;
  b.replaceChildren(...[el("strong", { text: title }), detail ? el("span", { text: detail }) : null, action].filter(Boolean));
  b.hidden = false;
}

/** Number that counts up to its new value, so a change is seen from the back of the room. */
function setNum(node, value, fmt) {
  const from = Number(node.dataset.v);
  node.dataset.v = value === null ? "" : String(value);
  if (value === null) { node.textContent = "—"; return; }
  if (!Number.isFinite(from) || node.dataset.v === "" || reducedMotion() || from === value) { node.textContent = fmt(value); return; }
  const t0 = performance.now(), dur = 550;
  const tick = (t) => {
    if (node.dataset.v !== String(value)) return; // a newer value took over
    const k = Math.min(1, (t - t0) / dur), e = 1 - (1 - k) ** 3;
    node.textContent = fmt(from + (value - from) * e);
    if (k < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// ---------- setup ----------

function selectedTask() {
  const v = $("task").value;
  if (v === "custom") return { custom: true, text: $("custom-text").value.trim(), test_list: $("custom-asserts").value.split("\n").map((s) => s.trim()).filter(Boolean) };
  return state.tasks.find((t) => String(t.task_id) === v) ?? null;
}

function syncUrl() {
  const q = new URLSearchParams();
  q.set("task", $("task").value);
  q.set("ctx", String(state.ctx));
  if ($("mirror").checked) q.set("mirror", "1");
  if ($("rate").value) q.set("rate", $("rate").value);
  history.replaceState(null, "", `?${q}`);
}

function renderTaskStrip() {
  const strip = $("task-strip");
  const task = selectedTask();
  $("custom").hidden = !task?.custom;
  if (!task) { strip.replaceChildren(); return; }
  const start = state.run?.start;
  const ctxLine = start
    ? `Shared context: ${fmtInt(start.context_tokens)} tokens (system prompt, task and a utils.py with ${start.ctx_funcs} functions)`
    : `Context: a utils.py with ${state.ctx} functions, as in the evals at 30`;
  const visible = task.test_list[0];
  strip.replaceChildren(...[
    el("p", { class: "task-text", text: task.text || "Describe the function to write, then add one assert per line." }),
    visible ? el("p", { class: "task-meta" }, "Visible test ", el("code", { text: visible }),
      ` · ${task.test_list.length} assert${task.test_list.length === 1 ? "" : "s"} run after every Coder turn`) : null,
    el("p", { class: "task-meta", text: ctxLine })].filter(Boolean));
}

function renderCtx() {
  const seg = $("ctx");
  seg.replaceChildren(...state.info.ctx_options.map((n) => el("button", {
    type: "button", "aria-pressed": String(n === state.ctx), disabled: state.running || undefined,
    onclick: () => { state.ctx = n; renderCtx(); renderTaskStrip(); syncUrl(); },
    text: `${n}`,
  })));
}

function renderRig(info) {
  $("rig").textContent = `${info.device_name} · ${shortModel(info.model)} · ${info.dtype}`;
  $("smoke-tag").hidden = !info.smoke_adapters;
  const updates = info.adapter_runs.map((a) => a.updates).filter((u) => u !== null);
  const notes = [`Model ${info.model} on ${info.device_name} (${info.dtype}), adapters from ${info.adapters}`
    + (updates.length ? ` (${Math.min(...updates) === Math.max(...updates) ? Math.min(...updates) : `${Math.min(...updates)}–${Math.max(...updates)}`} training updates each)` : "") + "."];
  if (info.smoke_adapters) notes.push("These are smoke-test adapters: the code and reviews they write are not meaningful; the token counts are.");
  if (info.device !== "cuda") notes.push("Timings on this machine are indicative only. The reported numbers come from the RTX 4090.");
  notes.push(`Greedy decoding, at most ${info.max_new_tokens.coder} tokens per Coder turn and ${info.max_new_tokens.reviewer} per Reviewer turn; after ${info.max_revise} REVISE verdicts the next Coder turn is final.`);
  $("rig-note").textContent = notes.join(" ");
}

async function init() {
  $("controls").addEventListener("submit", (e) => { e.preventDefault(); run(); });
  $("stop").addEventListener("click", stop);
  $("task").addEventListener("change", () => { renderTaskStrip(); syncUrl(); });
  $("mirror").addEventListener("change", syncUrl);
  $("rate").addEventListener("input", () => { renderBoard(); renderFinals(); syncUrl(); });
  $("custom-text").addEventListener("input", renderTaskStrip);
  $("custom-asserts").addEventListener("input", renderTaskStrip);

  let info, tasks;
  try {
    [info, tasks] = await Promise.all([getJSON("/api/info"), getJSON("/api/tasks")]);
  } catch (err) {
    $("rig").textContent = "Model server not reachable";
    banner("error", "Cannot reach the model server.", "Start it with: uv run python demo/live/server.py (see demo/README.md, Live demo), then reload.");
    return;
  }
  state.info = info;
  state.tasks = tasks;
  const q = new URLSearchParams(location.search);
  state.ctx = Number(q.get("ctx")) || info.default_ctx_funcs;
  if (!info.ctx_options.includes(state.ctx)) state.ctx = info.default_ctx_funcs;
  $("rate").value = q.get("rate") ?? String(info.default_usd_per_gpu_hour);
  $("mirror").checked = q.get("mirror") === "1";

  const sel = $("task");
  sel.replaceChildren(
    ...tasks.map((t) => el("option", { value: String(t.task_id), text: `#${t.task_id} · ${t.text.length > 96 ? `${t.text.slice(0, 95)}…` : t.text}` })),
    el("option", { value: "custom", text: "Custom task…" }));
  const want = q.get("task");
  if (want && [...sel.options].some((o) => o.value === want)) sel.value = want;
  sel.disabled = false;
  $("run").disabled = false;

  renderRig(info);
  renderCtx();
  renderTaskStrip();
  renderBoard();
  state.log = info.live_log ?? null;
  renderRecord();
  if (info.busy) busyBanner();
}

/** After a run: how many tasks this server session has recorded, and the way to their stats. */
function renderRecord() {
  const box = $("live-record");
  const log = state.log;
  const link = $("dash-link");
  const recorded = log?.enabled && log.sessions > 0;
  link.href = recorded ? log.dashboard : "../dashboard.html";
  link.textContent = recorded ? "Dashboard: this session" : "Dashboard";
  if (!log) { box.hidden = true; return; }
  if (!log.enabled) {
    box.replaceChildren(el("p", { class: "note", text: `Live runs are not being recorded: ${log.reason}.` }));
    box.hidden = false;
    return;
  }
  if (!recorded) { box.hidden = true; return; }
  box.replaceChildren(
    el("p", {},
      el("b", { text: `${log.tasks} ${log.tasks === 1 ? "task" : "tasks"} recorded this session. ` }),
      "Every finished run is saved for the dashboard: tokens computed, first-token times and pass rates over all of them."),
    el("a", { class: "btn primary", href: log.dashboard, text: "See the stats on the dashboard" }),
    el("p", { class: "note", text: `Saved to demo/${log.path}. Runs at another context size, and same-transcript runs, are kept as separate runs there.` }));
  box.hidden = false;
}

function busyBanner() {
  banner("warn", "The model server is busy with another run.", "It may have been started from another tab.",
    el("button", { type: "button", class: "btn", text: "Stop that run", onclick: async () => { await fetch("/api/cancel", { method: "POST" }); banner(null); } }));
}

// ---------- running ----------

function setRunning(on) {
  state.running = on;
  $("run").disabled = on;
  $("run").textContent = on ? "Running…" : "Run both";
  $("stop").hidden = !on;
  $("task").disabled = on;
  $("mirror").disabled = on;
  for (const id of ["custom-text", "custom-asserts"]) $(id).disabled = on;
  renderCtx();
  clearInterval(state.timer);
  if (on) state.timer = setInterval(renderFoot, 200);
  renderFoot();
}

function resetRun() {
  state.run = { start: null, turns: { "A2": [], "A5-H1": [] }, finals: {}, cards: new Map(), pending: new Map(), scale: 1, t0: performance.now(), wall: null, next: null, ended: null };
  const lanes = $("lanes");
  for (const node of [...lanes.querySelectorAll(".tcard")]) node.remove();
  $("finals").replaceChildren();
  renderBoard();
}

async function run() {
  if (state.running || !state.info) return;
  const task = selectedTask();
  if (!task) return;
  const body = { ctx_funcs: state.ctx, usd_per_gpu_hour: rate(), mirror: $("mirror").checked };
  if (task.custom) {
    if (!task.text || !task.test_list.length) { banner("warn", "A custom task needs a description and at least one assert line."); return; }
    body.custom = { text: task.text, asserts: task.test_list.join("\n") };
  } else {
    body.task_id = task.task_id;
  }
  banner(null);
  resetRun();
  setRunning(true);
  try {
    const res = await fetch("/api/run", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (res.status === 409) { busyBanner(); return; }
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    await readEvents(res.body, onEvent);
    if (!state.run.ended) throw new Error("the stream ended early; check the server terminal");
  } catch (err) {
    banner("error", "The run failed.", err.message);
  } finally {
    for (const node of state.run.pending.values()) node.remove();
    state.run.pending.clear();
    setRunning(false);
  }
}

async function stop() {
  $("stop").disabled = true;
  try { await fetch("/api/cancel", { method: "POST" }); } finally { $("stop").disabled = false; }
}

/** Parse a text/event-stream body (fetch, since EventSource cannot POST and would retry a finished run). */
async function readEvents(stream, handle) {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let cut;
    while ((cut = buf.indexOf("\n\n")) >= 0) {
      const chunk = buf.slice(0, cut);
      buf = buf.slice(cut + 2);
      let type = "message", data = "";
      for (const line of chunk.split("\n")) {
        if (line.startsWith("event:")) type = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trimStart();
      }
      if (data) handle(type, JSON.parse(data));
    }
  }
}

function onEvent(type, ev) {
  const r = state.run;
  if (type === "start") {
    r.start = ev;
    renderTaskStrip();
  } else if (type === "turn_start") {
    r.next = ev;
    showPending(ev);
  } else if (type === "turn") {
    r.next = null;
    r.turns[ev.method].push(ev);
    addTurnCard(ev);
    renderBoard();
    announce(`${NAME[ev.method]}, turn ${ev.turn + 1}, ${ROLE[ev.role]}: ${ev.computed_tokens} tokens computed, first token after ${Math.round(ev.ttft_ms)} ms.`);
  } else if (type === "final") {
    r.finals[ev.method] = ev;
    renderFinals();
    renderBoard();
  } else if (type === "done" || type === "cancelled") {
    r.ended = type;
    r.wall = ev.wall_s;
    if (ev.log) { state.log = ev.log; renderRecord(); }
    if (type === "cancelled") banner("warn", "Stopped.", "The run was stopped after the turn in progress.");
  } else if (type === "error") {
    r.ended = "error";
    banner("error", "The model server reported an error.", ev.message);
  }
  renderFoot();
}

// ---------- lanes ----------

function place(node, method, turn) {
  node.style.gridColumn = String(COL[method]);
  node.style.gridRow = String(turn + 2);
}

function showPending(ev) {
  const role = ev.role;
  const final = role === "coder" && ev.turn === 2 * state.info.max_revise;
  const node = el("article", { class: `tcard pending ${CLS[ev.method]}`, "aria-busy": "true" },
    el("div", { class: "tcard-head" },
      el("span", { class: `role-pill ${role}`, text: ROLE[role] }),
      el("span", { class: "turn-no", text: `turn ${ev.turn + 1}${final ? " · final revision" : ""}` })),
    el("div", { class: "working", text: role === "coder" ? "Writing code…" : "Reviewing…" }),
    el("div", { class: "shimmer" }));
  place(node, ev.method, ev.turn);
  $("lanes").append(node);
  state.run.pending.set(`${ev.method}:${ev.turn}`, node);
  if (!reducedMotion()) node.scrollIntoView({ block: "nearest", behavior: "smooth" });
}

function tapeFor(ev, pending) {
  const node = tapeEl(ev, state.run.scale, { method: ev.method, index: ev.turn, pending });
  node.tabIndex = 0;
  attachTooltip(node, () => [
    `${NAME[ev.method]}, turn ${ev.turn + 1} (${ROLE[ev.role]})`,
    `${fmtInt(ev.computed_tokens)} tokens computed${ev.canon_tokens ? `, ${fmtInt(ev.canon_tokens)} of them the handoff re-read` : ""}`,
    `${fmtInt(ev.reused_tokens)} reused from the KV cache`,
    `${fmtInt(ev.decode_tokens)} written`,
  ]);
  return node;
}

function rescaleTapes() {
  const r = state.run;
  const all = METHODS.flatMap((m) => r.turns[m]);
  const scale = Math.max(1, ...all.map((t) => tapeSpans(t).end));
  if (scale === r.scale) return false;
  r.scale = scale;
  for (const [key, card] of r.cards) {
    const ev = card.ev;
    const fresh = tapeFor(ev, false);
    card.tape.replaceWith(fresh);
    card.tape = fresh;
  }
  return true;
}

function addTurnCard(ev) {
  const r = state.run;
  const key = `${ev.method}:${ev.turn}`;
  r.pending.get(key)?.remove();
  r.pending.delete(key);
  rescaleTapes();

  const verdict = ev.role === "reviewer"
    ? el("span", { class: `vchip ${ev.verdict}`, text: ev.verdict })
    : ev.last ? el("span", { class: "vchip final", text: "final code" }) : null;
  const tape = tapeFor(ev, true);
  let tool = null;
  if (ev.tool) {
    const ok = ev.tool.passed;
    tool = el("details", { class: "tool" },
      el("summary", {}, "Tests ", el("span", { class: ok ? "ok" : "bad", text: ok ? "pass" : ev.tool.timed_out ? "timed out" : "fail" }),
        ev.tool.output ? " · output" : ""),
      ev.tool.output ? el("pre", { text: ev.tool.output }) : null);
    if (!ev.tool.output) tool.querySelector("summary").append(" (no output)");
  }
  const node = el("article", { class: `tcard ${CLS[ev.method]}` },
    el("div", { class: "tcard-head" },
      el("span", { class: `role-pill ${ev.role}`, text: ROLE[ev.role] }),
      el("span", { class: "turn-no", text: `turn ${ev.turn + 1}` }),
      verdict),
    el("div", { class: "tmetrics" },
      el("span", { class: "tm computed" }, el("b", { text: fmtInt(ev.computed_tokens) }), "computed"),
      el("span", { class: "tm reused" }, el("b", { text: fmtInt(ev.reused_tokens) }), "reused"),
      el("span", { class: "tm ttft" }, el("b", { text: fmtMs(ev.ttft_ms) }), "to first token")),
    tape,
    el("pre", { class: "ttext", text: ev.text || "(empty reply)" }),
    tool,
    el("div", { class: "tfoot", text: `${fmtInt(ev.decode_tokens)} tokens written${ev.truncated ? " (hit the token limit)" : ""} · turn ${fmtS(ev.turn_ms / 1000)}`
      + `${ev.canon_tokens ? ` · ${fmtInt(ev.canon_tokens)} of the computed tokens re-read the last reply` : ""}` }));
  place(node, ev.method, ev.turn);
  $("lanes").append(node);
  r.cards.set(key, { ev, node, tape });
  sweep(tape);
}

// ---------- totals ----------

function totals(method, n) {
  const ts = state.run ? state.run.turns[method].slice(0, n) : [];
  const switches = ts.filter((t) => t.turn >= 1).map((t) => t.ttft_ms);
  const gpu = sum(ts, (t) => t.turn_ms) / 1000;
  return { n: ts.length, computed: ts.length ? sum(ts, (t) => t.computed_tokens) : null, ttft: mean(switches), gpu: ts.length ? gpu : null };
}

function cmp(id, text, sub, kind) {
  const node = $(id);
  node.className = `bcmp${kind ? ` ${kind}` : ""}`;
  node.replaceChildren(...[text ? document.createTextNode(text) : null, sub ? el("small", { text: sub }) : null].filter(Boolean));
}

function renderBoard() {
  const r = state.run;
  const perM = (gpu) => (gpu === null ? null : gpu * rate() / 3600 * 1e6);
  for (const m of METHODS) {
    const t = totals(m, Infinity);
    setNum($(`b-tok-${m}`), t.computed, fmtInt);
    setNum($(`b-ttft-${m}`), t.ttft, fmtMs);
    setNum($(`b-gpu-${m}`), t.gpu, fmtS);
    setNum($(`b-usd-${m}`), perM(t.gpu), fmtUsd);
  }
  const n = r ? Math.min(r.turns["A2"].length, r.turns["A5-H1"].length) : 0;
  if (!n) {
    for (const id of ["b-tok-cmp", "b-ttft-cmp", "b-gpu-cmp", "b-usd-cmp"]) cmp(id, "");
    return;
  }
  const a = totals("A2", n), b = totals("A5-H1", n);
  const over = `over the ${n} turn${n === 1 ? "" : "s"} both finished`;
  const tokRatio = a.computed / Math.max(1, b.computed);
  cmp("b-tok-cmp", tokRatio >= 1 ? `${tokRatio.toFixed(1)}× fewer` : `${(1 / tokRatio).toFixed(1)}× more`, over, tokRatio >= 1 ? "win" : "loss");
  if (a.ttft !== null && b.ttft !== null) {
    const x = a.ttft / b.ttft;
    cmp("b-ttft-cmp", x >= 1 ? `${x.toFixed(1)}× faster` : `${(1 / x).toFixed(1)}× slower`, "after each role switch", x >= 1 ? "win" : "loss");
  } else {
    cmp("b-ttft-cmp", "", "from the first role switch on");
  }
  const pct = 1 - b.gpu / a.gpu;
  const word = Math.abs(pct) < 0.01 ? ["about the same", ""] : pct > 0 ? [`${Math.round(pct * 100)}% less`, "win"] : [`${Math.round(-pct * 100)}% more`, "loss"];
  cmp("b-gpu-cmp", word[0], "prefill and decoding", word[1]);
  cmp("b-usd-cmp", word[0], `at $${rate().toFixed(2)}/GPU-hour, assumed`, word[1]);
}

function renderFoot() {
  const foot = $("board-foot");
  const r = state.run;
  if (!r) return;
  const elapsed = r.wall ?? (performance.now() - r.t0) / 1000;
  const parts = [];
  if (state.running && r.next) parts.push(el("b", { text: `${NAME[r.next.method]} · ${ROLE[r.next.role]} is generating` }), " · ");
  else if (state.running) parts.push(el("b", { text: "Preparing the context" }), " · ");
  else if (r.ended === "done") parts.push(el("b", { text: "Done" }), " · ");
  else if (r.ended) parts.push(el("b", { text: r.ended === "cancelled" ? "Stopped" : "Failed" }), " · ");
  parts.push(el("span", { class: "elapsed", text: `${elapsed.toFixed(1)} s wall clock` }));
  if (r.start) parts.push(` · ${fmtInt(r.start.context_tokens)}-token shared context`, r.start.mirror ? " · same transcript: A2 replays RoleKV's tokens" : " · both columns generate their own replies");
  foot.replaceChildren(...parts);
}

function endWords(f) {
  if (f.end === "APPROVE") return `The Reviewer approved after ${f.n_turns} turns.`;
  if (f.end === "INVALID") return `The Reviewer's reply was neither APPROVE nor REVISE, which ends the session (${f.n_turns} turns).`;
  return `Final revision after ${f.verdicts.length} REVISE verdicts (${f.n_turns} turns).`;
}

function renderFinals() {
  const host = $("finals");
  const r = state.run;
  if (!r) return;
  host.replaceChildren(...METHODS.filter((m) => r.finals[m]).map((m) => {
    const f = r.finals[m];
    const usd = f.gpu_s * rate() / 3600;
    const node = el("article", { class: `fcard ${CLS[m]}` },
      el("p", { class: `fresult ${f.passed ? "pass" : "fail"}`, text: f.passed ? `${NAME[m]}: passes all ${f.n_asserts} asserts` : `${NAME[m]}: fails the asserts` }),
      el("p", { class: "fline", text: endWords(f) }),
      el("p", { class: "fline", text: `${fmtInt(f.prefill_tokens_total)} tokens computed · ${fmtS(f.gpu_s)} on GPU · ${fmtUsd(usd)} per session at $${rate().toFixed(2)}/GPU-hour (assumed)` }),
      el("details", {}, el("summary", { text: "Final code" }), el("pre", { class: "ttext", text: f.final_code || "(no code)" })),
      f.passed ? null : el("details", {}, el("summary", { text: "Test output" }), el("pre", { class: "ttext", text: f.test_output || "(no output)" })));
    node.style.gridColumn = String(COL[m]);
    return node;
  }));
}

init();
