// RoleKV homepage: one screen. The KV cache is drawn in 3D behind the words (cache3d.js) and as two 2D
// tapes in the HUD; both are driven here, from the same per-turn numbers, so they always agree. Every
// number on the page is a token count copied from a logged run, with its source next to it. Timings and
// pass rates stay off the page until their runs land (docs/STATUS.md).
import { tapeSpans } from "./metrics.js";
import { tapePainter } from "./tapecanvas.js";
import { $, fmtInt, kfmt, reducedMotion } from "./ui.js";

// ---------- Measured numbers ----------
// One real 4-turn session (buggy code → REVISE → fixed code → APPROVE) with ~1.1k tokens of shared context:
// bench/results/cost_sweep_4090.jsonl, runs sweep-A2-ctx1k and sweep-A5-H1-ctx1k, task ctx1000-r0. Token
// counts depend only on the tokens; the Mac run (cost_sweep_mac.jsonl) has the identical figures.
const T = (role, prefill_tokens, reused_tokens, canon_tokens, decode_tokens, verdict = null) =>
  ({ role, prefill_tokens, reused_tokens, canon_tokens, decode_tokens, verdict });
const SESSION = {
  "A2": [T("coder", 1227, 0, 0, 80), T("reviewer", 1393, 0, 0, 30, "REVISE"), T("coder", 1434, 0, 0, 83), T("reviewer", 1545, 0, 0, 4, "APPROVE")],
  "A5-H1": [T("coder", 1227, 0, 0, 80), T("reviewer", 87, 1217, 89, 30, "REVISE"), T("coder", 12, 1382, 40, 83), T("reviewer", 29, 1424, 92, 4, "APPROVE")],
};
// Tokens computed per session (prefill + handoff re-read) by shared-context size: bench/results/cost_sweep_4090.md.
const SWEEP = [
  { ctx: 1101, A2: 5599, A5: 1576 },
  { ctx: 2100, A2: 9595, A5: 2575 },
  { ctx: 4095, A2: 17575, A5: 4570 },
  { ctx: 8114, A2: 33651, A5: 8589 },
];
const CANON_PER_SESSION = 221;   // "of which canon" in the same table, the same at every context size
const LAYERS = 28;               // decoder layers in Qwen2.5-Coder-1.5B-Instruct: the depth of the cache slabs

const METHODS = ["A2", "A5-H1"];
const NAMES = { "A2": "Per-role LoRAs (A2)", "A5-H1": "RoleKV (A5-H1)" };
const ROLE = { coder: "the Coder", reviewer: "the Reviewer" };
const CELL_WORDS = {
  reused: "held in the cache, reused for free",
  recomputed: "read this turn",
  canon: "re-read at the handoff, with base weights",
  written: "written this turn",
};
const cost = (t) => t.prefill_tokens + t.canon_tokens;

// ---------- A clock that can be paused ----------
// Every wait and tween reads this clock, so Pause freezes the whole replay mid-sweep and Play resumes it.
const clock = { paused: false, t: 0, last: null };
function now() {
  const real = performance.now();
  if (clock.last === null) clock.last = real;
  if (!clock.paused) clock.t += real - clock.last;
  clock.last = real;
  return clock.t;
}
/** Drive k from 0 to 1 over ms of clock time; resolves false if alive() turns false first. */
function tween(ms, alive, frame) {
  return new Promise((resolve) => {
    if (ms <= 0) { frame(1); resolve(true); return; }
    let t0 = null;
    const tick = () => {
      if (!alive()) { resolve(false); return; }
      const t = now();
      if (t0 === null) t0 = t;
      const k = Math.min(1, (t - t0) / ms);
      frame(k);
      if (k < 1) requestAnimationFrame(tick); else resolve(true);
    };
    requestAnimationFrame(tick);
  });
}
const wait = (ms, alive) => tween(ms, alive, () => {});

// ---------- Hover on the scene ----------
function hoverTooltip() {
  const tip = $("tooltip");
  return (info) => {
    if (!info) { tip.hidden = true; return; }
    tip.replaceChildren(
      Object.assign(document.createElement("div"), { textContent: `${NAMES[info.key]} · layer ${info.layer} of ${info.layers}` }),
      Object.assign(document.createElement("div"), { textContent: `tokens ${fmtInt(info.from)}–${fmtInt(info.to)} · ${CELL_WORDS[info.category]}` }));
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    tip.style.left = `${Math.max(8, Math.min(window.innerWidth - r.width - 8, info.x + 16))}px`;
    tip.style.top = `${Math.max(8, Math.min(window.innerHeight - r.height - 8, info.y - r.height - 14))}px`;
  };
}

// ---------- The replay: HUD tapes and the 3D slabs in lockstep ----------
async function liveSession() {
  const n = SESSION["A2"].length;
  const scale = Math.max(...METHODS.map((m) => tapeSpans(SESSION[m][n - 1]).end));
  const totals = Object.fromEntries(METHODS.map((m) => [m, SESSION[m].reduce((s, t) => s + cost(t), 0)]));
  const rows = Object.fromEntries(METHODS.map((m) => {
    const id = m === "A2" ? "a2" : "a5";
    return [m, { painter: tapePainter($(`tape-${id}`)), count: $(`count-${id}`), ready: $(`ready-${id}`), total: 0 }];
  }));
  const caption = $("live-turn"), ratio = $("live-ratio");
  const still = reducedMotion();

  // The 3D scene is optional: without WebGL the HUD still tells the story.
  let scene = null;
  try {
    const { createCacheScene } = await import("./cache3d.js");
    scene = createCacheScene($("scene"), {
      methods: METHODS.map((m) => ({ key: m, turns: SESSION[m] })),
      scale, layers: LAYERS, reducedMotion: still,
      labels: { "A2": $("label-a2"), "A5-H1": $("label-a5") },
      onHover: hoverTooltip(),
    });
    $("hud-hint").textContent = `Each block is the KV cache of one real 4-turn session: a ${scene.side} × ${scene.side} map of ${scene.bin}-token cells, ${scene.layers} layers deep · drag to look around, hover a cell`;
    if (new URLSearchParams(location.search).has("debug")) window.__rolekv = { scene };
  } catch (err) {
    console.warn("3D scene unavailable; showing the tapes only.", err);
    $("scene").hidden = true;
    document.body.classList.add("no-3d");
  }

  let playId = 0;
  const alive = (id) => () => id === playId;
  const frame = (m, i, read, write) => {
    rows[m].painter.draw({ spans: tapeSpans(SESSION[m][i]), scale, read, write });
    scene?.setFrame(m, { turn: i, read, write });
  };
  const setCount = (m, v) => { rows[m].count.textContent = fmtInt(v); };
  // Reading runs at a fixed speed per token, so a long re-read is felt as a long wait; writing too.
  const readMs = (tokens) => Math.round(Math.min(1700, Math.max(260, tokens * 0.95)));
  const writeMs = (tokens) => Math.round(Math.min(800, Math.max(200, tokens * 9)));

  const readTurn = async (m, i, id) => {
    const t = SESSION[m][i], from = rows[m].total, add = cost(t);
    const ok = await tween(readMs(add), alive(id), (k) => { frame(m, i, k, 0); setCount(m, from + add * k); });
    if (!ok) return;
    rows[m].total = from + add;
    setCount(m, rows[m].total);
    rows[m].ready.classList.add("on");
  };
  const writeTurn = (m, i, id) => tween(writeMs(SESSION[m][i].decode_tokens), alive(id), (k) => frame(m, i, 1, k));

  const finish = () => {
    for (const m of METHODS) {
      rows[m].total = totals[m];
      setCount(m, totals[m]);
      rows[m].ready.classList.remove("on");
      frame(m, n - 1, 1, 1);
    }
    caption.textContent = `All ${n} turns · ${fmtInt(totals["A2"])} vs ${fmtInt(totals["A5-H1"])} tokens computed`;
    ratio.textContent = `${(totals["A2"] / totals["A5-H1"]).toFixed(1)}× fewer tokens`;
    ratio.hidden = false;
  };

  const play = async () => {
    const id = ++playId;
    ratio.hidden = true;
    for (const m of METHODS) { rows[m].total = 0; setCount(m, 0); rows[m].ready.classList.remove("on"); frame(m, 0, 0, 0); }
    if (still) { finish(); return; }
    caption.textContent = `Turn 1 of ${n}`;
    await wait(900, alive(id));
    for (let i = 0; i < n; i++) {
      if (id !== playId) return;
      const t = SESSION["A2"][i];
      caption.textContent = `Turn ${i + 1} of ${n} · ${ROLE[t.role]} reads before answering`;
      await Promise.all(METHODS.map((m) => readTurn(m, i, id)));
      if (id !== playId) return;
      await wait(520, alive(id));
      for (const m of METHODS) rows[m].ready.classList.remove("on");
      caption.textContent = `Turn ${i + 1} of ${n} · ${ROLE[t.role]} writes${t.verdict ? ` · ${t.verdict}` : ""}`;
      await Promise.all(METHODS.map((m) => writeTurn(m, i, id)));
      if (id !== playId) return;
      await wait(620, alive(id));
    }
    finish();
    await wait(4500, alive(id));
    if (id === playId) play();
  };

  $("live-replay").addEventListener("click", () => { clock.paused = false; syncPause(); play(); });
  const pauseBtn = $("live-pause");
  const syncPause = () => { pauseBtn.textContent = clock.paused ? "Play" : "Pause"; pauseBtn.setAttribute("aria-pressed", String(clock.paused)); };
  pauseBtn.addEventListener("click", () => { clock.paused = !clock.paused; syncPause(); });
  play();
}

// ---------- The chips ----------
function chips() {
  const last = SWEEP[SWEEP.length - 1];
  $("chip-ratio").textContent = `${(last.A2 / last.A5).toFixed(1)}×`;
  $("chip-ratio-text").textContent = `fewer tokens computed at ${kfmt(last.ctx)} context`;
  $("chip-canon").textContent = fmtInt(CANON_PER_SESSION);
  const end = tapeSpans(SESSION["A2"][SESSION["A2"].length - 1]).end;
  $("chip-cache").textContent = `${LAYERS} × ${fmtInt(end)}`;
}

chips();
liveSession();
