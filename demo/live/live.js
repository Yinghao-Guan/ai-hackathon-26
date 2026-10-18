// RoleKV live demo: one task on one model, Per-role LoRA (A2) and RoleKV (A5-H1) side by side, with the KV
// cache of both sessions in 3D (../js/cache3d.js) and as tapes (../js/tapecanvas.js), as on the homepage.
// demo/live/server.py streams one Server-Sent Event per finished turn. Each turn's card appears as soon as it
// arrives; the boards play a turn once both methods have sent it (livemodel.nextRound), so the two caches
// fill side by side. Model output is always inserted with textContent, never as HTML.
import { $, el, fmtInt, reducedMotion, announce } from "../js/ui.js";
import { EMPTY_SPANS } from "../js/metrics.js";
import { tapePainter } from "../js/tapecanvas.js";
import {
  METHODS, NAME, ROLE, newRun, applyEvent, totals, compare, perMillion, capacityFor, binFor, sessionEnd,
  nextRound, backlog, playMs, fmtMs, fmtS, fmtUsd, endWords, readCaption, displaySpans,
} from "./livemodel.js";

const CELL_WORDS = {
  reused: "held in the cache, reused for free",
  recomputed: "computed this turn",
  canon: "the last reply, re-read with base weights at the handoff",
  written: "written this turn",
};
// The 3D blocks get the panel to themselves: centred, a little room above them for the labels.
const VIEW = { fraction: 0.8, shift: 0, lift: -0.03, height: 1.7 };
const IDLE_CONTEXT = 2000; // tokens, for the empty blocks before the first run (30 functions ≈ 1.6–2k)

const state = {
  info: null,
  tasks: [],
  ctx: 30,
  running: false,
  run: newRun(),
  runId: 0,
  shown: -1,          // the last round both boards have played
  playingId: null,    // the run whose rounds are being played
  finals: [],         // final events held back until the boards have played every round
  boardsDone: false,  // the boards have caught up with an ended run: show "Done" and the final cards
  frames: {},         // per method: the board's current { i, read, write }
  capacity: capacityFor(IDLE_CONTEXT + 300),
  scene: null,
  painters: {},
  timer: null,
  t0: 0,
};

const api = (path) => new URL(`../api/${path}`, location.href);
const sumComputed = (turns) => turns.reduce((a, t) => a + t.prefill_tokens + t.canon_tokens, 0);
const rate = () => {
  const v = parseFloat($("rate").value);
  return Number.isFinite(v) && v >= 0 ? v : state.info?.default_usd_per_gpu_hour ?? 0;
};

function banner(kind, title, detail, action) {
  const b = $("banner");
  if (!kind) { b.hidden = true; b.replaceChildren(); return; }
  b.dataset.kind = kind;
  b.replaceChildren(...[el("strong", { text: title }), detail ? el("span", { text: detail }) : null, action].filter(Boolean));
  b.hidden = false;
}

/** Drive k from 0 to 1 over `ms`; resolves false if `alive()` turns false first. */
function tween(ms, alive, frame) {
  return new Promise((resolve) => {
    if (ms <= 0 || reducedMotion()) { frame(1); resolve(alive()); return; }
    const t0 = performance.now();
    const tick = (t) => {
      if (!alive()) { resolve(false); return; }
      const k = Math.min(1, (t - t0) / ms);
      frame(k);
      if (k < 1) requestAnimationFrame(tick); else resolve(true);
    };
    requestAnimationFrame(tick);
  });
}
const wait = (ms, alive) => tween(ms, alive, () => {});

// ---------- Setup ----------

function selectedTask() {
  const v = $("task").value;
  if (v === "custom") {
    return { custom: true, text: $("custom-text").value.trim(), test_list: $("custom-asserts").value.split("\n").map((s) => s.trim()).filter(Boolean) };
  }
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

function renderTaskLine() {
  const line = $("task-line");
  const task = selectedTask();
  $("custom").hidden = !task?.custom;
  if (!task) { line.replaceChildren(); return; }
  const start = state.run.start;
  const visible = task.test_list[0];
  line.replaceChildren(...[
    el("p", { class: "task-text", text: task.text || "Describe the function to write, then add one assert per line." }),
    visible ? el("p", { class: "task-meta" }, "Visible test ", el("code", { text: visible }),
      ` · ${task.test_list.length} assert${task.test_list.length === 1 ? "" : "s"} run after every Coder turn`) : null,
    el("p", { class: "task-meta", text: start
      ? `Shared context ${fmtInt(start.context_tokens)} tokens (system prompt, task, utils.py with ${start.ctx_funcs} functions)`
      : `utils.py with ${state.ctx} functions (the evals use 30)` }),
  ].filter(Boolean));
}

function renderCtx() {
  $("ctx").replaceChildren(...state.info.ctx_options.map((n) => el("button", {
    type: "button", "aria-pressed": String(n === state.ctx), disabled: state.running || undefined, text: `${n}`,
    onclick: () => { state.ctx = n; renderCtx(); renderTaskLine(); syncUrl(); },
  })));
}

function renderRig(info) {
  $("rig").textContent = `${info.device_name} · ${String(info.model).split("/").pop()} · ${info.dtype}`;
  $("smoke-tag").hidden = !info.smoke_adapters;
  const updates = info.adapter_runs.map((a) => a.updates).filter((u) => u !== null);
  const lo = Math.min(...updates), hi = Math.max(...updates);
  const notes = [`Model ${info.model} on ${info.device_name} (${info.dtype}), adapters from ${info.adapters}`
    + (updates.length ? ` (${lo === hi ? lo : `${lo}–${hi}`} training updates each)` : "") + "."];
  if (info.smoke_adapters) notes.push("These are smoke-test adapters: the code and reviews they write are not meaningful; the token counts are.");
  if (info.device !== "cuda") notes.push("Timings on this machine are indicative only; the reported numbers come from the RTX 4090.");
  notes.push(`Greedy decoding, at most ${info.max_new_tokens.coder} tokens per Coder turn and ${info.max_new_tokens.reviewer} per Reviewer turn; after ${info.max_revise} REVISE verdicts the next Coder turn is final.`);
  $("rig-note").textContent = notes.join(" ");
}

function hoverTooltip() {
  const tip = $("tooltip");
  return (info) => {
    if (!info) { tip.hidden = true; return; }
    tip.replaceChildren(
      el("div", { text: `${NAME[info.key]} · layer ${info.layer} of ${info.layers}` }),
      el("div", { text: `tokens ${fmtInt(info.from)}–${fmtInt(info.to)} · ${CELL_WORDS[info.category]}` }));
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    tip.style.left = `${Math.max(8, Math.min(window.innerWidth - r.width - 8, info.x + 16))}px`;
    tip.style.top = `${Math.max(8, Math.min(window.innerHeight - r.height - 8, info.y - r.height - 14))}px`;
  };
}

/** The 3D cache is optional: without WebGL the tapes and the cards still tell the story. */
async function makeScene(layers) {
  try {
    const { createCacheScene } = await import("../js/cache3d.js");
    state.scene = createCacheScene($("scene"), {
      methods: METHODS.map((m) => ({ key: m, turns: state.run.turns[m] })),
      scale: state.capacity, bin: binFor(state.capacity), layers, reducedMotion: reducedMotion(),
      labels: { "A2": $("label-A2"), "A5-H1": $("label-A5-H1") },
      onHover: hoverTooltip(),
      view: { wide: VIEW, narrow: VIEW },
      spans: (turn) => displaySpans(turn, state.capacity),
    });
    // ?debug: inspect the page, or replay recorded events through it (feed(type, event) after reset()).
    if (new URLSearchParams(location.search).has("debug")) window.__rolekv = { scene: state.scene, state, reset: resetRun, feed: onEvent };
  } catch (err) {
    console.warn("3D scene unavailable; showing the tapes only.", err);
    document.body.classList.add("no-3d");
  }
}

function offline() {
  $("offline").classList.remove("connecting");
  $("rig").textContent = "No model server";
  $("caption").textContent = "Waiting for the model server (demo/live/server.py).";
  $("task").replaceChildren(el("option", { text: "Start demo/live/server.py to pick a task" }));
}

async function init() {
  $("offline").classList.add("connecting");
  for (const m of METHODS) state.painters[m] = tapePainter($(`tape-${m}`));
  drawBoards();
  $("controls").addEventListener("submit", (e) => { e.preventDefault(); run(); });
  $("stop").addEventListener("click", stop);
  $("task").addEventListener("change", () => { renderTaskLine(); syncUrl(); });
  $("mirror").addEventListener("change", syncUrl);
  $("rate").addEventListener("input", () => { renderStats(); syncUrl(); });
  $("custom-text").addEventListener("input", renderTaskLine);
  $("custom-asserts").addEventListener("input", renderTaskLine);

  let info, tasks;
  try {
    const get = async (path) => {
      const res = await fetch(api(path), { cache: "no-store" });
      if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
      return res.json();
    };
    [info, tasks] = await Promise.all([get("info"), get("tasks")]);
  } catch (err) {
    offline();
    return;
  }
  state.info = info;
  state.tasks = tasks;
  $("offline").hidden = true;

  const q = new URLSearchParams(location.search);
  state.ctx = info.ctx_options.includes(Number(q.get("ctx"))) ? Number(q.get("ctx")) : info.default_ctx_funcs;
  $("rate").value = q.get("rate") ?? String(info.default_usd_per_gpu_hour);
  $("mirror").checked = q.get("mirror") === "1";
  const sel = $("task");
  sel.replaceChildren(
    ...tasks.map((t) => el("option", { value: String(t.task_id), text: `#${t.task_id} · ${t.text.length > 96 ? `${t.text.slice(0, 95)}…` : t.text}` })),
    el("option", { value: "custom", text: "Custom task…" }));
  if (q.get("task") && [...sel.options].some((o) => o.value === q.get("task"))) sel.value = q.get("task");
  for (const id of ["task", "rate", "mirror", "run"]) $(id).disabled = false;

  renderRig(info);
  renderCtx();
  renderTaskLine();
  renderStats();
  $("caption").textContent = "Pick a task and press Run both.";
  if (info.busy) busyBanner();
  await makeScene(info.layers ?? 28);
}

function busyBanner() {
  banner("warn", "The model server is busy with another run.", "It may have been started from another tab.",
    el("button", { type: "button", class: "btn", text: "Stop that run", onclick: async () => { await fetch(api("cancel"), { method: "POST" }); banner(null); } }));
}

// ---------- Running ----------

function setRunning(on) {
  state.running = on;
  $("run").disabled = on;
  $("run").textContent = on ? "Running…" : "Run both";
  $("stop").hidden = !on;
  for (const id of ["task", "mirror", "custom-text", "custom-asserts"]) $(id).disabled = on;
  renderCtx();
  clearInterval(state.timer);
  if (on) state.timer = setInterval(renderStatus, 200);
  renderStatus();
}

function resetRun() {
  state.runId++;
  state.run = newRun();
  state.shown = -1;
  state.finals = [];
  state.boardsDone = false;
  state.t0 = performance.now();
  for (const m of METHODS) {
    $(`cards-${m}`).replaceChildren();
    state.frames[m] = null;
    state.scene?.setTurns(m, state.run.turns[m]);
    state.scene?.setFrame(m, { turn: 0, read: 0, write: 0 });
    $(`count-${m}`).textContent = "0";
  }
  $("ratio").hidden = true;
  drawBoards();
  renderStats();
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
  $("caption").textContent = "Preparing the context…";
  setRunning(true);
  try {
    const res = await fetch(api("run"), { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    if (res.status === 409) { busyBanner(); $("caption").textContent = "The model server is busy."; return; }
    if (!res.ok) {
      const err = await res.json().catch(() => ({}));
      throw new Error(err.error || `HTTP ${res.status}`);
    }
    await readEvents(res.body, onEvent);
    if (!state.run.ended) throw new Error("the stream ended early; check the server's terminal");
  } catch (err) {
    applyEvent(state.run, "error", { message: err.message });
    banner("error", "The run failed.", err.message);
    pump();
  } finally {
    for (const m of METHODS) $(`cards-${m}`).querySelector(".tcard.pending")?.remove();
    setRunning(false);
  }
}

async function stop() {
  $("stop").disabled = true;
  try { await fetch(api("cancel"), { method: "POST" }); } finally { $("stop").disabled = false; }
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
  applyEvent(state.run, type, ev);
  if (type === "start") {
    state.capacity = capacityFor(ev.context_tokens + 300);
    state.scene?.relayout({ scale: state.capacity, bin: binFor(state.capacity) });
    drawBoards();
    renderTaskLine();
  } else if (type === "turn_start") {
    showPending(ev);
  } else if (type === "turn") {
    addTurnCard(ev);
    renderStats();
    announce(`${NAME[ev.method]}, turn ${ev.turn + 1}, ${ROLE[ev.role]}: ${ev.computed_tokens} tokens computed, first token after ${Math.round(ev.ttft_ms)} ms.`);
  } else if (type === "final") {
    // The result card waits for the boards, so "passes" never shows while the 3D is still a round behind.
    state.finals.push(ev);
    renderStats();
  } else if (type === "cancelled") {
    // No banner: the status line says "Stopped" at once and the caption once the boards finish. A banner
    // here pushed the whole stage down mid-demo.
  } else if (type === "error") {
    banner("error", "The model server reported an error.", ev.message);
  }
  if (type !== "turn_start") pump();
  renderStatus();
}

// ---------- The boards: 3D blocks and tapes, one round at a time ----------

function frame(m, i, read, write) {
  state.frames[m] = { i, read, write };
  state.painters[m].draw({ spans: displaySpans(state.run.turns[m][i], state.capacity), scale: state.capacity, read, write });
  state.scene?.setFrame(m, { turn: i, read, write });
}

/** Redraw both tapes at their current frame (after a resize of the scale, or a reset). */
function drawBoards() {
  for (const m of METHODS) {
    const f = state.frames[m];
    if (f) frame(m, f.i, f.read, f.write);
    else state.painters[m].draw({ spans: EMPTY_SPANS, scale: state.capacity, read: 0, write: 0 });
  }
}

/** Play every round that is ready, in order; a new run takes over from an old one mid-round. */
async function pump() {
  if (state.playingId === state.runId) return;
  const id = state.playingId = state.runId;
  const alive = () => id === state.runId;
  try {
    for (let r = nextRound(state.run, state.shown); r !== null && alive(); r = nextRound(state.run, state.shown)) {
      await playRound(r, alive, backlog(state.run, state.shown) > 1 ? 2 : 1);
      if (!alive()) return;
      state.shown = r;
      const c = compare(state.run, r + 1);
      // Only after a role switch: on the first turn both read the same context.
      if (c.tokens && c.n >= 2) { $("ratio").textContent = `${c.tokens.text} tokens computed`; $("ratio").hidden = false; }
    }
    if (alive() && state.run.ended && nextRound(state.run, state.shown) === null) finishBoards();
  } finally {
    if (state.playingId === id) state.playingId = null;
  }
  // Events that arrived during the last await may have made another round ready.
  if (alive() && nextRound(state.run, state.shown) !== null) pump();
}

async function playRound(r, alive, speed) {
  const run = state.run;
  const ms = METHODS.filter((m) => run.turns[m][r]);
  const need = sessionEnd(run, r + 1);
  if (need > state.capacity) {
    state.capacity = capacityFor(need, state.capacity);
    state.scene?.relayout({ scale: state.capacity });
    drawBoards();
  }
  $("caption").textContent = readCaption(run, r);
  await Promise.all(ms.map((m) => {
    const t = run.turns[m][r], before = sumComputed(run.turns[m].slice(0, r)), add = t.prefill_tokens + t.canon_tokens;
    return tween(playMs(t, speed).read, alive, (k) => { frame(m, r, k, 0); $(`count-${m}`).textContent = fmtInt(before + add * k); });
  }));
  if (!alive()) return;
  const verdicts = [...new Set(ms.map((m) => run.turns[m][r].verdict).filter(Boolean))];
  const role = run.turns[ms[0]][r].role;
  $("caption").textContent = `Turn ${r + 1} · the ${ROLE[role]} writes`
    + (verdicts.length === 1 ? ` · ${verdicts[0]}` : verdicts.length ? ` · ${ms.map((m) => `${NAME[m]}: ${run.turns[m][r].verdict}`).join(" · ")}` : "");
  await Promise.all(ms.map((m) => tween(playMs(run.turns[m][r], speed).write, alive, (k) => frame(m, r, 1, k))));
  await wait(280 / speed, alive);
}

/** The boards have played every round of an ended run: the final cards and "Done" can show now. */
function finishBoards() {
  endCaption();
  for (const f of state.finals) addFinalCard(f);
  state.finals = [];
  state.boardsDone = true;
  renderStatus();
}

function endCaption() {
  const a = totals(state.run.turns["A2"]), b = totals(state.run.turns["A5-H1"]);
  const caption = $("caption");
  if (state.run.ended === "done") caption.textContent = `All turns played · ${fmtInt(a.computed)} vs ${fmtInt(b.computed)} tokens computed`;
  else if (state.run.ended === "cancelled") caption.textContent = "Stopped after the turn in progress.";
  else caption.textContent = "The run failed; see the message above.";
}

// ---------- Totals ----------

function setCmp(id, word, note) {
  const node = $(id);
  node.className = `stat-cmp${word?.kind ? ` ${word.kind}` : ""}`;
  node.replaceChildren(...[word ? document.createTextNode(word.text) : null, note ? el("small", { text: note }) : null].filter(Boolean));
}

function renderStats() {
  const run = state.run;
  for (const m of METHODS) {
    const t = totals(run.turns[m]);
    $(`ttft-${m}`).textContent = fmtMs(t.ttft);
    $(`gpu-${m}`).textContent = fmtS(t.gpu);
    $(`usd-${m}`).textContent = fmtUsd(perMillion(t.gpu, rate()));
  }
  const c = compare(run);
  // Like the TTFT, the GPU and cost comparisons start at the first role switch: on turn 1 both sides do the
  // same work, so any gap there is noise, not the cache.
  const gpu = c.n >= 2 ? c.gpu : null;
  setCmp("ttft-cmp", c.ttft, c.ttft ? "after each switch" : c.n ? "from the first switch on" : "");
  setCmp("gpu-cmp", gpu, gpu ? "prefill and decoding" : c.n ? "from the first switch on" : "");
  setCmp("usd-cmp", gpu, gpu ? `at $${rate().toFixed(2)}/GPU-hour, assumed` : "");
}

function renderStatus() {
  const r = state.run, parts = [];
  if (!state.info) return;
  const elapsed = r.wall ?? (state.t0 ? (performance.now() - state.t0) / 1000 : null);
  if (state.running && r.next) parts.push(el("b", { text: `${NAME[r.next.method]} · ${ROLE[r.next.role]} is generating` }));
  else if (state.running) parts.push(el("b", { text: "Preparing the context" }));
  else if (r.ended === "done") parts.push(el("b", { text: state.boardsDone ? "Done" : "Replaying the last turns" }));
  else if (r.ended) parts.push(el("b", { text: r.ended === "cancelled" ? "Stopped" : "Failed" }));
  if (elapsed !== null) parts.push(` · ${elapsed.toFixed(1)} s`);
  if (r.start) parts.push(r.start.mirror ? " · same transcript: A2 replays RoleKV's tokens" : " · each side writes its own replies");
  $("status").replaceChildren(...parts);
}

// ---------- Cards ----------

function lane(m) { return $(`cards-${m}`); }
function follow(node) {
  const box = node.parentElement;
  if (box.scrollHeight > box.clientHeight) box.scrollTo({ top: box.scrollHeight, behavior: reducedMotion() ? "auto" : "smooth" });
}

function showPending(ev) {
  const final = ev.role === "coder" && ev.turn === 2 * state.info.max_revise;
  const node = el("article", { class: "tcard pending", "aria-busy": "true" },
    el("div", { class: "tcard-head" },
      el("span", { class: `role-pill ${ev.role}`, text: ROLE[ev.role] }),
      el("span", { class: "turn-no", text: `turn ${ev.turn + 1}${final ? " · final revision" : ""}` })),
    el("div", { class: "working", text: ev.role === "coder" ? "Writing code…" : "Reviewing…" }),
    el("div", { class: "shimmer" }));
  lane(ev.method).append(node);
  follow(node);
}

/** Share of the turn's input that was reused (cyan), re-read at the handoff (violet) and computed (coral). */
function mini(ev) {
  const seg = (cls, n) => (n > 0 ? el("span", { class: cls, style: { flex: `${n} 0 0`, minWidth: "3px" } }) : null);
  return el("div", { class: "mini", "aria-hidden": "true" }, seg("r", ev.reused_tokens), seg("k", ev.canon_tokens), seg("c", ev.prefill_tokens));
}

function addTurnCard(ev) {
  lane(ev.method).querySelector(".tcard.pending")?.remove();
  const verdict = ev.role === "reviewer"
    ? el("span", { class: `vchip ${ev.verdict}`, text: ev.verdict })
    : ev.last ? el("span", { class: "vchip final", text: "final code" }) : null;
  let tool = null;
  if (ev.tool) {
    tool = el("details", { class: "tool" },
      el("summary", {}, "Tests ", el("span", { class: ev.tool.passed ? "ok" : "bad", text: ev.tool.passed ? "pass" : ev.tool.timed_out ? "timed out" : "fail" }),
        ev.tool.output ? " · output" : " (no output)"),
      ev.tool.output ? el("pre", { text: ev.tool.output }) : null);
  }
  const node = el("article", { class: "tcard" },
    el("div", { class: "tcard-head" },
      el("span", { class: `role-pill ${ev.role}`, text: ROLE[ev.role] }),
      el("span", { class: "turn-no", text: `turn ${ev.turn + 1}` }),
      verdict),
    el("div", { class: "tmetrics" },
      el("span", { class: "tm computed" }, el("b", { text: fmtInt(ev.computed_tokens) }), "computed"),
      el("span", { class: "tm reused" }, el("b", { text: fmtInt(ev.reused_tokens) }), "reused"),
      el("span", { class: "tm ttft" }, el("b", { text: fmtMs(ev.ttft_ms) }), "first token")),
    mini(ev),
    el("pre", { class: "ttext", text: ev.text || "(empty reply)" }),
    tool,
    el("div", { class: "tfoot", text: `${fmtInt(ev.decode_tokens)} tokens written${ev.truncated ? " (hit the token limit)" : ""} · turn ${fmtS(ev.turn_ms / 1000)}`
      + (ev.canon_tokens ? ` · ${fmtInt(ev.canon_tokens)} of the computed tokens re-read the last reply` : "") }));
  lane(ev.method).append(node);
  follow(node);
}

function addFinalCard(f) {
  const usd = f.gpu_s * rate() / 3600;
  const node = el("article", { class: "fcard" },
    el("p", { class: `fresult ${f.passed ? "pass" : "fail"}`, text: f.passed ? `Passes all ${f.n_asserts} asserts` : "Fails the asserts" }),
    el("p", { class: "fline", text: endWords(f) }),
    el("p", { class: "fline", text: `${fmtInt(f.prefill_tokens_total)} tokens computed · ${fmtS(f.gpu_s)} on GPU · ${fmtUsd(usd)} per session at $${rate().toFixed(2)}/GPU-hour (assumed)` }),
    el("details", {}, el("summary", { text: "Final code" }), el("pre", { class: "ttext", text: f.final_code || "(no code)" })),
    f.passed ? null : el("details", {}, el("summary", { text: "Test output" }), el("pre", { class: "ttext", text: f.test_output || "(no output)" })));
  lane(f.method).append(node);
  follow(node);
}

init();
