// RoleKV homepage: the pitch as one page. The hero replays one real session with light; the rest tells
// what was measured. Every number here is a token count copied from a logged run, with its source next to
// it. Timings and pass rates stay off this page until their runs land (docs/STATUS.md).
import { tapeSpans } from "./metrics.js";
import { $, el, fmtInt, kfmt, reducedMotion, attachTooltip } from "./ui.js";
import { sessionMap } from "./tape.js";

// ---------- Measured numbers ----------
// One real 4-turn session (buggy code → REVISE → fixed code → APPROVE) with ~1.1k tokens of shared context:
// bench/results/cost_sweep_4090.jsonl, runs sweep-A2-ctx1k and sweep-A5-H1-ctx1k, task ctx1000-r0. Token
// counts depend only on the tokens; the Mac run (cost_sweep_mac.jsonl) has the identical figures.
const T = (role, prefill_tokens, reused_tokens, canon_tokens, decode_tokens, verdict = null) =>
  ({ role, prefill_tokens, reused_tokens, canon_tokens, decode_tokens, verdict });
const SESSION = {
  task: { ctx_tokens: 1101 },
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
const CANON_PER_SESSION = 221; // "of which canon" in the same table, the same at every context size

const METHODS = ["A2", "A5-H1"];
const NAMES = { "A2": "Per-role LoRAs (A2)", "A5-H1": "RoleKV (A5-H1)" };
const ROLE = { coder: "the Coder", reviewer: "the Reviewer" };
const cost = (t) => t.prefill_tokens + t.canon_tokens;
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** Drive k from 0 to 1 over ms with requestAnimationFrame; resolves false if alive() turns false first. */
function tween(ms, alive, frame) {
  return new Promise((resolve) => {
    if (ms <= 0) { frame(1); resolve(true); return; }
    const t0 = performance.now();
    const tick = (now) => {
      if (!alive()) { resolve(false); return; }
      const k = Math.min(1, (now - t0) / ms);
      frame(k);
      if (k < 1) requestAnimationFrame(tick); else resolve(true);
    };
    requestAnimationFrame(tick);
  });
}

function roundRect(ctx, x, y, w, h, r) {
  const rr = Math.max(0, Math.min(r, h / 2, w / 2));
  ctx.beginPath();
  ctx.moveTo(x + rr, y);
  ctx.arcTo(x + w, y, x + w, y + h, rr);
  ctx.arcTo(x + w, y + h, x, y + h, rr);
  ctx.arcTo(x, y + h, x, y, rr);
  ctx.arcTo(x, y, x + w, y, rr);
  ctx.closePath();
}

/** Once, when the node scrolls into view (at once under reduced motion or without IntersectionObserver). */
function onVisible(node, fn, threshold = 0.3) {
  if (!node) return;
  if (reducedMotion() || !("IntersectionObserver" in window)) { fn(); return; }
  const io = new IntersectionObserver((es) => {
    if (es.some((e) => e.isIntersecting)) { io.disconnect(); fn(); }
  }, { threshold });
  io.observe(node);
}

function countUp(node, to, { dec = 0, suffix = "", ms = 1300 } = {}) {
  const show = (v) => { node.textContent = `${dec ? v.toFixed(dec) : fmtInt(v)}${suffix}`; };
  if (reducedMotion() || !ms) { show(to); return; }
  tween(ms, () => true, (k) => show(to * (1 - (1 - k) ** 3)));
}

// ---------- The hero: one session, lit ----------
/**
 * Paints one turn of a session on a canvas, the way the dashboard's tape does, with light. Paper is the
 * session so far, the cyan rule is what the cache holds, coral is read this turn, violet is the handoff
 * re-read, and the outline is what the role writes. `read` and `write` are the progress of the two phases.
 */
function tapePainter(canvas) {
  const ctx = canvas.getContext("2d");
  let w = 1, h = 1, c = {}, last = null;
  const resize = () => {
    const r = canvas.getBoundingClientRect();
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    w = Math.max(1, r.width); h = Math.max(1, r.height);
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    c = {
      paper: cssVar("--paper"), rule: cssVar("--paper-rule"),
      cache: cssVar("--cache"), cacheSoft: cssVar("--cache-soft"), cacheGlow: cssVar("--cache-glow"),
      ink: cssVar("--recompute"), inkGlow: cssVar("--recompute-glow"),
      canon: cssVar("--canon"), canonGlow: cssVar("--canon-glow"), written: cssVar("--written"),
    };
    if (last) draw(last);
  };
  const draw = (f) => {
    last = f;
    const { spans: s, scale, read, write } = f;
    const X = (tok) => (tok / scale) * w;
    // A segment never vanishes: 3px minimum, like the dashboard's tapes.
    const span = (a, b) => { const x0 = X(a); return [x0, Math.max(3, X(b) - x0)]; };
    const readTo = s.reused[1] + (s.recomputed[1] - s.reused[1]) * read;
    const paperEnd = s.recomputed[1] + (s.end - s.recomputed[1]) * write;
    const pad = 7, y = pad, hh = h - 2 * pad;
    ctx.clearRect(0, 0, w, h);
    ctx.shadowBlur = 0;

    let [x, ww] = span(0, paperEnd);
    roundRect(ctx, x, y, ww, hh, 3);
    ctx.fillStyle = c.paper; ctx.fill();
    ctx.lineWidth = 1; ctx.strokeStyle = c.rule; ctx.stroke();

    if (s.reused[1] > 0) {
      [x, ww] = span(0, s.reused[1]);
      ctx.fillStyle = c.cacheSoft; ctx.fillRect(x, y, ww, hh);
      ctx.shadowColor = c.cacheGlow; ctx.shadowBlur = 14;
      ctx.fillStyle = c.cache; ctx.fillRect(x, y + hh - 4, ww, 4);
      ctx.shadowBlur = 0;
    }
    if (s.canon && readTo > s.canon[0]) {
      [x, ww] = span(s.canon[0], Math.min(readTo, s.canon[1]));
      ctx.shadowColor = c.canonGlow; ctx.shadowBlur = 14;
      ctx.fillStyle = c.canon; ctx.fillRect(x, y + hh * 0.52, ww, hh * 0.48);
      ctx.shadowBlur = 0;
    }
    if (readTo > s.recomputed[0]) {
      [x, ww] = span(s.recomputed[0], Math.min(readTo, s.recomputed[1]));
      ctx.shadowColor = c.inkGlow; ctx.shadowBlur = 16;
      ctx.fillStyle = c.ink; ctx.fillRect(x, y, ww, hh);
      ctx.shadowBlur = 0;
    }
    if (read > 0 && read < 1) {
      // The read head: a bright bar with a bloom, so a long re-read is seen travelling.
      const hx = X(readTo);
      const g = ctx.createRadialGradient(hx, h / 2, 0, hx, h / 2, 28);
      g.addColorStop(0, "rgba(255,255,255,0.55)"); g.addColorStop(1, "rgba(255,255,255,0)");
      ctx.fillStyle = g; ctx.fillRect(hx - 28, 0, 56, h);
      ctx.fillStyle = "#fff"; ctx.fillRect(hx - 1, y - 2, 2, hh + 4);
    }
    if (write > 0) {
      [x, ww] = span(s.recomputed[1], paperEnd);
      ctx.lineWidth = 1.5; ctx.strokeStyle = c.written;
      ctx.strokeRect(x + 0.75, y + 0.75, Math.max(1, ww - 1.5), hh - 1.5);
      if (write < 1 && Math.floor(performance.now() / 240) % 2 === 0) {
        ctx.fillStyle = c.written; ctx.fillRect(x + ww - 1, y + 3, 2, hh - 6);
      }
    }
  };
  new ResizeObserver(resize).observe(canvas);
  resize();
  return { draw };
}

function liveSession() {
  const fig = $("live");
  if (!fig) return;
  const turns = { "A2": SESSION["A2"], "A5-H1": SESSION["A5-H1"] };
  const n = turns["A2"].length;
  const scale = Math.max(...METHODS.map((m) => tapeSpans(turns[m][n - 1]).end));
  const totals = Object.fromEntries(METHODS.map((m) => [m, turns[m].reduce((s, t) => s + cost(t), 0)]));
  const rows = Object.fromEntries(METHODS.map((m) => {
    const id = m === "A2" ? "a2" : "a5";
    return [m, { painter: tapePainter($(`tape-${id}`)), count: $(`count-${id}`), ready: $(`ready-${id}`), total: 0 }];
  }));
  const caption = $("live-turn"), ratio = $("live-ratio");
  let playId = 0, visible = false;
  const alive = (id) => () => id === playId;

  const frame = (m, i, read, write) => rows[m].painter.draw({ spans: tapeSpans(turns[m][i]), scale, read, write });
  const setCount = (m, v) => { rows[m].count.textContent = fmtInt(v); };
  // Reading runs at a fixed speed per token, so a long re-read is felt as a long wait; writing too.
  const readMs = (tokens) => Math.round(Math.min(1700, Math.max(260, tokens * 0.95)));
  const writeMs = (tokens) => Math.round(Math.min(800, Math.max(200, tokens * 9)));

  const readTurn = async (m, i, id) => {
    const t = turns[m][i], from = rows[m].total, add = cost(t);
    const ok = await tween(readMs(add), alive(id), (k) => { frame(m, i, k, 0); setCount(m, from + add * k); });
    if (!ok) return;
    rows[m].total = from + add;
    setCount(m, rows[m].total);
    rows[m].ready.classList.add("on");
  };
  const writeTurn = (m, i, id) => tween(writeMs(turns[m][i].decode_tokens), alive(id), (k) => frame(m, i, 1, k));

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
    if (reducedMotion()) { finish(); return; }
    caption.textContent = `Turn 1 of ${n}`;
    await sleep(700);
    for (let i = 0; i < n; i++) {
      if (id !== playId) return;
      const t = turns["A2"][i];
      caption.textContent = `Turn ${i + 1} of ${n} · ${ROLE[t.role]} reads before answering`;
      await Promise.all(METHODS.map((m) => readTurn(m, i, id)));
      if (id !== playId) return;
      await sleep(480);
      for (const m of METHODS) rows[m].ready.classList.remove("on");
      caption.textContent = `Turn ${i + 1} of ${n} · ${ROLE[t.role]} writes${t.verdict ? ` · ${t.verdict}` : ""}`;
      await Promise.all(METHODS.map((m) => writeTurn(m, i, id)));
      if (id !== playId) return;
      await sleep(560);
    }
    finish();
    await sleep(4000);
    if (id === playId && visible) play();
  };

  $("live-replay").addEventListener("click", () => play());
  // Plays while on screen; off screen it stops, and comes back from the top.
  if (!("IntersectionObserver" in window)) { visible = true; play(); return; }
  new IntersectionObserver((es) => {
    for (const e of es) {
      visible = e.isIntersecting;
      if (visible) play(); else playId++;
    }
  }, { threshold: 0.3 }).observe(fig);
}

// ---------- Backdrop: tokens drifting past ----------
function tokenField() {
  const canvas = $("field");
  if (!canvas) return;
  if (reducedMotion()) { canvas.remove(); return; }
  const ctx = canvas.getContext("2d");
  let w = 1, h = 1, parts = [], raf = 0;
  const resize = () => {
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    w = window.innerWidth; h = window.innerHeight;
    canvas.width = Math.round(w * dpr); canvas.height = Math.round(h * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const cols = [cssVar("--cache"), cssVar("--cache"), cssVar("--cache"), cssVar("--recompute"), cssVar("--canon")];
    const count = Math.round(Math.min(110, (w * h) / 16000));
    parts = Array.from({ length: count }, () => ({
      x: Math.random() * w, y: Math.random() * h, len: 8 + Math.random() * 22, v: 0.12 + Math.random() * 0.45,
      a: 0.05 + Math.random() * 0.16, color: cols[Math.floor(Math.random() * cols.length)], ph: Math.random() * Math.PI * 2,
    }));
  };
  const tick = (now) => {
    ctx.clearRect(0, 0, w, h);
    for (const p of parts) {
      p.x += p.v;
      if (p.x > w + 30) { p.x = -30; p.y = Math.random() * h; }
      ctx.globalAlpha = p.a; ctx.fillStyle = p.color;
      roundRect(ctx, p.x, p.y + Math.sin(now / 2200 + p.ph) * 5, p.len, 3, 1.5); ctx.fill();
    }
    ctx.globalAlpha = 1;
    raf = requestAnimationFrame(tick);
  };
  const start = () => { if (!raf) raf = requestAnimationFrame(tick); };
  const stop = () => { cancelAnimationFrame(raf); raf = 0; };
  window.addEventListener("resize", resize);
  document.addEventListener("visibilitychange", () => (document.hidden ? stop() : start()));
  resize();
  start();
}

// ---------- The mechanism ----------
function mechanism() {
  const host = $("map-host");
  if (host) host.append(sessionMap(SESSION["A5-H1"], SESSION.task));
  const note = $("canon-note");
  if (note) note.textContent = `${fmtInt(CANON_PER_SESSION)} tokens per session`;
}

// ---------- Numbers ----------
function numbers() {
  const ratios = SWEEP.map((r) => r.A2 / r.A5);
  const lo = Math.min(...ratios), hi = Math.max(...ratios), last = SWEEP[SWEEP.length - 1];
  $("ratio-lo").textContent = lo.toFixed(1);
  $("ratio-hi").textContent = hi.toFixed(1);
  $("tile-ratio-ctx").textContent = kfmt(last.ctx);
  onVisible(document.querySelector(".tiles"), () => {
    countUp($("tile-ratio"), last.A2 / last.A5, { dec: 1, suffix: "×" });
    countUp($("tile-canon"), CANON_PER_SESSION);
    countUp($("tile-one"), 1, { ms: 600 });
  });

  const chart = $("sweep-chart");
  const max = Math.max(...SWEEP.map((r) => r.A2));
  const bar = (m, v, ctxTok, delay) => {
    const node = el("div", {
      class: `bar ${m === "A2" ? "base" : "hero"}`, tabindex: "0",
      style: { "--f": (v / max).toFixed(4), "--d": `${delay}ms` },
    }, el("span", { class: "fill" }), el("span", { class: "val" }, fmtInt(v), el("small", { text: " tokens" })));
    attachTooltip(node, () => [NAMES[m], `${fmtInt(v)} tokens computed per session`, `${fmtInt(ctxTok)} tokens of shared context`]);
    return node;
  };
  chart.append(el("ul", { class: "legend" },
    el("li", {}, el("i", { class: "base", "aria-hidden": "true" }), NAMES["A2"]),
    el("li", {}, el("i", { class: "hero", "aria-hidden": "true" }), NAMES["A5-H1"])));
  SWEEP.forEach((r, i) => {
    chart.append(el("div", { class: "grp" },
      el("div", { class: "grp-head" }, el("b", { text: `${kfmt(r.ctx)} tokens` }), " of shared context",
        el("span", { class: "ratio", text: `${(r.A2 / r.A5).toFixed(1)}× fewer` })),
      bar("A2", r.A2, r.ctx, i * 140),
      bar("A5-H1", r.A5, r.ctx, i * 140 + 90)));
  });

  $("sweep-table").append(
    el("thead", {}, el("tr", {}, el("th", { text: "Shared context" }), el("th", { text: NAMES["A2"] }), el("th", { text: NAMES["A5-H1"] }), el("th", { text: "Ratio" }))),
    el("tbody", {}, ...SWEEP.map((r) => el("tr", {},
      el("th", { scope: "row", text: `${fmtInt(r.ctx)} tokens` }), el("td", { text: fmtInt(r.A2) }),
      el("td", { class: "hero", text: fmtInt(r.A5) }), el("td", { text: `${(r.A2 / r.A5).toFixed(2)}×` })))));
}

// ---------- Page chrome ----------
function chrome() {
  const nav = $("nav");
  const onScroll = () => nav.classList.toggle("scrolled", window.scrollY > 8);
  window.addEventListener("scroll", onScroll, { passive: true });
  onScroll();

  const nodes = [...document.querySelectorAll(".section > *, .foot-inner")];
  if (reducedMotion() || !("IntersectionObserver" in window)) return;
  const io = new IntersectionObserver((es) => {
    for (const e of es) if (e.isIntersecting) { e.target.classList.add("in"); io.unobserve(e.target); }
  }, { threshold: 0.12, rootMargin: "0px 0px -6% 0px" });
  nodes.forEach((node) => {
    node.style.setProperty("--i", String([...node.parentNode.children].indexOf(node)));
    node.classList.add("reveal");
    io.observe(node);
  });
}

chrome();
mechanism();
numbers();
liveSession();
tokenField();
