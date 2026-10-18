// The presentation stage (?present): one full-viewport beat per part of the 2-minute video.
// Arrow keys, space and Page Up/Down move between beats; R replays the current beat. Every number is
// computed from the loaded log, exactly as on the dashboard.
import { featuredTask, replayTaskKeys, runFor, turnTotals } from "./metrics.js";
import { loadDefault } from "./data.js";
import { $, el, svg, fmtInt, ctxWords, plural, methodName, reducedMotion, announce } from "./ui.js";
import { tapeEl, sweep, sessionMap } from "./tape.js";
import { qualityChart, sweepChart, sweepHeadline } from "./charts.js";

const REPO = "github.com/Yinghao-Guan/ai-hackathon-26";
const params = new URLSearchParams(location.search);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
let playId = 0;

export async function start() {
  document.title = "RoleKV — Presentation";
  // Every load opens on the first beat, so a retake never starts mid-deck from a restored scroll position.
  history.scrollRestoration = "manual";
  document.body.classList.add("presenting");
  $("dashboard").hidden = true;
  const stage = $("stage");
  stage.hidden = false;
  stage.tabIndex = -1;
  stage.append(el("p", { class: "stage-loading", text: "Loading the event log." }));
  render(stage, await loadDefault());
}

function dashboardHref() {
  const p = new URLSearchParams(location.search);
  p.delete("present");
  return `dashboard.html${p.size ? `?${p}` : ""}`;
}

function render(stage, data) {
  stage.replaceChildren();
  const a = data.analysis;
  const ctxK = a?.selection.ctxK ?? null;
  const a2 = a && runFor(a.runs, "A2", ctxK), a5 = a && runFor(a.runs, "A5-H1", ctxK);
  const keys = replayTaskKeys(a2, a5);
  if (!a || !a.headline.ok || !keys.length) {
    stage.append(el("section", { class: "beat" }, el("div", { class: "beat-body" },
      el("h1", { class: "display", text: "Nothing to present yet." }),
      el("p", { class: "beat-sub", text: data.source?.kind === "error" ? data.source.label : "The stage needs per-role LoRA (A2) and RoleKV (A5-H1) runs on the same tasks." }),
      el("a", { class: "btn", href: dashboardHref() }, "Open the dashboard"))));
    return;
  }
  const task = keys.includes(params.get("task")) ? params.get("task") : featuredTask(a2, a5);
  const t2 = a2.turns.get(task), t5 = a5.turns.get(task);
  const mock = data.dataset.mode === "mock";

  const beats = [
    beatCost(t2),
    beatIdea(t5, a5.tasks.get(task)),
    beatReplay(t2, t5),
    beatQuality(a),
    a.sweep.points.length ? beatSweep(a) : null,
    beatResult(a, mock),
    beatEnd(),
  ].filter(Boolean);
  beats.forEach((b, i) => { b.node.dataset.beat = String(i); b.node.setAttribute("aria-label", `Beat ${i + 1} of ${beats.length}`); });
  stage.append(...beats.map((b) => b.node));
  stage.append(chrome(beats.length, mock));
  navigate(stage, beats);
}

// ---------- Shared pieces ----------
function tapeBlock(method, turns, { tapeScale, ledgerScale }) {
  const host = el("div", { class: "stage-tape-host" });
  const ledger = el("div", { class: "ledger", "aria-hidden": "true" });
  const count = el("strong", { class: "count", text: "0" });
  const node = el("div", { class: `stage-block ${method === "A5-H1" ? "hero" : "base"}` },
    el("div", { class: "block-head" },
      el("span", { class: "block-name" }, methodName(method), el("small", { text: ` ${method}` })),
      el("span", { class: "block-count" }, count, " tokens read")),
    host,
    el("div", { class: "ledger-row" }, el("span", { class: "ledger-label", text: "Read so far" }), ledger));
  let total = 0;
  const reset = () => {
    total = 0;
    count.textContent = "0";
    ledger.replaceChildren();
    host.replaceChildren(tapeEl(turns[0], tapeScale, { method: methodName(method), index: 0, pending: true }));
  };
  /** Draw turn i: its tape sweeps, its cost joins the ledger and the counter climbs with the ink. */
  const turn = (i) => {
    const t = turns[i];
    if (!t) return Promise.resolve();
    const tape = tapeEl(t, tapeScale, { method: methodName(method), index: i, pending: true, perToken: 0.5 });
    host.replaceChildren(tape);
    const cost = t.prefill_tokens + t.canon_tokens;
    const from = total;
    total += cost;
    const chip = el("span", { class: "chip", style: { "--w": `${(100 * cost / ledgerScale).toFixed(3)}%`, "--ink-ms": `${tape.duration}ms` } });
    ledger.append(chip);
    countUp(count, from, total, tape.duration);
    requestAnimationFrame(() => requestAnimationFrame(() => chip.classList.add("on")));
    return sweep(tape);
  };
  const finish = () => {
    const last = turns[turns.length - 1];
    host.replaceChildren(tapeEl(last, tapeScale, { method: methodName(method), index: turns.length - 1 }));
    ledger.replaceChildren(...turns.map((t) => el("span", { class: "chip on", style: { "--w": `${(100 * (t.prefill_tokens + t.canon_tokens) / ledgerScale).toFixed(3)}%` } })));
    total = turnTotals(turns).compute;
    count.textContent = fmtInt(total);
  };
  reset();
  return { node, reset, turn, finish };
}

function countUp(node, from, to, ms) {
  if (!ms) { node.textContent = fmtInt(to); return; }
  const t0 = performance.now();
  const tick = (now) => {
    const k = Math.min(1, (now - t0) / ms);
    node.textContent = fmtInt(from + (to - from) * (1 - (1 - k) ** 3));
    if (k < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function swatch(kind) {
  return el("i", { class: `sw ${kind}`, "aria-hidden": "true" });
}

function beat(id, ...children) {
  return el("section", { class: `beat beat-${id}`, id: `beat-${id}` }, el("div", { class: "beat-body" }, ...children));
}

/** Plays tape blocks turn by turn in lockstep; the caption names the turn. */
function sequencer(blocks, turnsList, caption) {
  const n = Math.max(...turnsList.map((t) => t.length));
  return async () => {
    const id = ++playId;
    blocks.forEach((b) => b.reset());
    if (reducedMotion()) { blocks.forEach((b) => b.finish()); caption.textContent = `All ${n} turns`; return; }
    await sleep(500);
    for (let i = 0; i < n; i++) {
      if (id !== playId) return;
      const role = turnsList[0][i]?.role ?? turnsList[1]?.[i]?.role;
      caption.textContent = `Turn ${i + 1} of ${n} · ${role === "coder" ? "the Coder" : "the Reviewer"}`;
      announce(caption.textContent);
      await Promise.all(blocks.map((b) => b.turn(i)));
      await sleep(520);
    }
    if (id === playId) caption.textContent = `All ${n} turns`;
  };
}

// ---------- Beats ----------
function beatCost(t2) {
  const tapeScale = Math.max(...t2.map((t) => t.reused_tokens + t.prefill_tokens + t.decode_tokens));
  const block = tapeBlock("A2", t2, { tapeScale, ledgerScale: turnTotals(t2).compute });
  const caption = el("p", { class: "turn-caption", "aria-hidden": "true" });
  const node = beat("cost",
    el("h1", { class: "display" }, "Every role switch ", el("span", { class: "nowrap", text: "re-reads" }), " the whole session."),
    el("p", { class: "beat-sub" }, "A Coder and a Reviewer take turns on one task, each with its own LoRA. Each turn starts by reading the task, the code and the tests again: ", swatch("ink"), " ink is every token read."),
    caption, block.node);
  return { node, enter: sequencer([block], [t2], caption), leave: () => playId++ };
}

function beatIdea(t5, task) {
  const node = beat("idea",
    el("h2", { class: "display", text: "RoleKV keeps one shared cache." }),
    sessionMap(t5, task),
    el("ul", { class: "idea-points" },
      el("li", {}, swatch("cache"), el("span", {}, "Each role is an ", el("b", { text: "Activated LoRA" }), ": it switches on only after its role header. Everything before is read once by the base model, and both roles reuse it.")),
      el("li", {}, swatch("canon"), el("span", {}, "At each handoff RoleKV re-reads only the last answer, with base weights, so the cache stays exactly what each role was trained on."))));
  return {
    node,
    enter: () => { node.classList.remove("drawn"); void node.offsetWidth; node.classList.add("drawn"); },
    leave: () => {},
  };
}

function beatReplay(t2, t5) {
  const all = [...t2, ...t5];
  const tapeScale = Math.max(...all.map((t) => t.reused_tokens + t.prefill_tokens + t.decode_tokens));
  const ledgerScale = Math.max(turnTotals(t2).compute, turnTotals(t5).compute);
  const b2 = tapeBlock("A2", t2, { tapeScale, ledgerScale });
  const b5 = tapeBlock("A5-H1", t5, { tapeScale, ledgerScale });
  const caption = el("p", { class: "turn-caption", "aria-hidden": "true" });
  const node = beat("replay",
    el("h2", { class: "display", text: "Same task. RoleKV reads only what's new." }),
    el("p", { class: "beat-sub" }, swatch("ink"), " read again  ", swatch("cache"), " reused from the cache  ", swatch("canon"), " re-read at handoff  ", swatch("written"), " written this turn"),
    caption, b2.node, b5.node);
  return { node, enter: sequencer([b2, b5], [t2, t5], caption), leave: () => playId++ };
}

function stageChartWidth() {
  return Math.max(320, Math.min(1180, window.innerWidth - 2 * Math.min(120, Math.max(24, window.innerWidth * 0.07))));
}

function beatQuality(a) {
  const w = stageChartWidth();
  const hasA1 = a.rows.some((r) => r.method === "A1" && r.available);
  const node = beat("quality",
    el("h2", { class: "display", text: "Cheaper only counts if the code still works." }),
    qualityChart(a, { w, h: Math.round(Math.min(520, Math.max(300, window.innerHeight * 0.52))), large: true }),
    hasA1 ? el("p", { class: "beat-note", text: "One shared LoRA (A1) is about as cheap, because it also keeps one cache. It gives up separate role adapters, which RoleKV keeps." }) : null);
  return { node, enter: () => {}, leave: () => {} };
}

function beatSweep(a) {
  const w = stageChartWidth();
  const node = beat("sweep",
    el("h2", { class: "display", text: "The longer the shared context, the more there is to skip." }),
    el("p", { class: "beat-sub", text: sweepHeadline(a.sweep.points) }),
    sweepChart(a, { w, h: Math.round(Math.min(480, Math.max(280, window.innerHeight * 0.46))), large: true }));
  return { node, enter: () => {}, leave: () => {} };
}

function beatResult(a, mock) {
  const h = a.headline;
  const pct = Math.abs(h.pctSaved);
  const node = beat("result",
    el("h2", { class: "display result" }, "RoleKV ", el("span", { class: "nowrap", text: "re-read" }), " ", el("em", { text: `${pct.toFixed(pct < 10 ? 1 : 0)}% ${h.pctSaved >= 0 ? "fewer" : "more"} tokens` }), " than per-role LoRAs."),
    el("p", { class: "result-line" },
      `That is ${fmtInt(h.meanA)} → ${fmtInt(h.meanB)} tokens per task across ${plural(h.n, "paired task")} with ${ctxWords(a.selection.ctxK)} tokens of shared context, `,
      `and RoleKV passed ${a.a5Pass.passed} of ${a.a5Pass.n} tasks where per-role LoRAs passed ${a.a2Pass.passed} of ${a.a2Pass.n}.`),
    mock ? el("p", { class: "beat-note warn", text: "Computed from the labelled mock fixture: it shows the pipeline, not RoleKV's results." }) : null);
  return { node, enter: () => {}, leave: () => {} };
}

function beatEnd() {
  const node = beat("end",
    svg("svg", { class: "mark big", viewBox: "0 0 40 16", "aria-hidden": "true" },
      svg("rect", { x: 0.75, y: 2.75, width: 38.5, height: 10.5, rx: 1.5, class: "mark-paper" }),
      svg("rect", { x: 1.5, y: 3.5, width: 27, height: 9, class: "mark-cache" }),
      svg("rect", { x: 28.5, y: 3.5, width: 10, height: 9, class: "mark-ink" })),
    el("h2", { class: "display end-word", text: "RoleKV" }),
    el("p", { class: "beat-sub", text: "Shared-KV role adapters for coding agents. LA Hacks 2026." }),
    el("p", { class: "repo mono", text: REPO }));
  return { node, enter: () => {}, leave: () => {} };
}

// ---------- Chrome and navigation ----------
function chrome(n, mock) {
  return el("div", { class: "stage-chrome" },
    el("a", { class: "chrome-item exit", href: dashboardHref() }, "Dashboard"),
    mock ? el("span", { class: "mode-tag", "data-mode": "mock", text: "Mock data — not results" }) : null,
    el("p", { class: "chrome-item pager", id: "pager", "aria-hidden": "true" },
      el("span", { id: "pager-n", text: `1 / ${n}` }), el("span", { class: "hint", text: "Arrow keys to move · R to replay" })));
}

function navigate(stage, beats) {
  let current = -1;
  const go = (i) => {
    const k = Math.max(0, Math.min(beats.length - 1, i));
    beats[k].node.scrollIntoView({ behavior: reducedMotion() ? "auto" : "smooth", block: "start" });
  };
  const arrive = (i) => {
    if (i === current) return;
    if (current >= 0) beats[current].leave();
    current = i;
    $("pager-n").textContent = `${i + 1} / ${beats.length}`;
    beats[i].enter();
  };
  const io = new IntersectionObserver((entries) => {
    for (const e of entries) if (e.isIntersecting) arrive(Number(e.target.dataset.beat));
  }, { root: stage, threshold: 0.6 });
  beats.forEach((b) => io.observe(b.node));
  arrive(0);

  window.addEventListener("keydown", (e) => {
    if (e.altKey || e.ctrlKey || e.metaKey) return;
    if ((e.key === " " || e.key === "Enter") && e.target.closest?.("a, button, select, input")) return;
    const next = ["ArrowRight", "ArrowDown", "PageDown", " ", "Enter"];
    const prev = ["ArrowLeft", "ArrowUp", "PageUp"];
    if (next.includes(e.key)) go(current + 1);
    else if (prev.includes(e.key)) go(current - 1);
    else if (e.key === "Home") go(0);
    else if (e.key === "End") go(beats.length - 1);
    else if (e.key === "r" || e.key === "R") { beats[current]?.leave(); beats[current]?.enter(); }
    else return;
    e.preventDefault();
  });

  // The chrome steps back once the pointer rests, so recordings stay clean; the mock tag never hides.
  let idle = null;
  const wake = () => {
    stage.classList.remove("idle");
    clearTimeout(idle);
    idle = setTimeout(() => stage.classList.add("idle"), 1800);
  };
  ["mousemove", "keydown", "focusin"].forEach((ev) => window.addEventListener(ev, wake));
  wake();
  stage.focus({ preventScroll: true });
}
