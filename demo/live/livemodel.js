// Pure state for the live demo (demo/live/live.js): the run as the server's events build it, the totals and
// comparisons on the page, the size of the 3D cache, and which turn the boards may play next. No DOM and no
// three.js, so Node tests can import it (demo/tests/live.test.mjs).
import { tapeSpans } from "../js/metrics.js";
import { fmtInt } from "../js/ui.js";

export const METHODS = ["A2", "A5-H1"];
export const NAME = { "A2": "Per-role LoRA", "A5-H1": "RoleKV" };
export const ROLE = { coder: "Coder", reviewer: "Reviewer" };

/** A run before its first event. */
export function newRun() {
  return { start: null, turns: { "A2": [], "A5-H1": [] }, finals: {}, next: null, ended: null, wall: null, error: null };
}

/**
 * Fold one server event (demo/live/server.py: start, turn_start, turn, final, done, cancelled, error) into
 * the run. Turns arrive in order per method; a turn that repeats or skips an index is a protocol error.
 */
export function applyEvent(run, type, ev) {
  if (type === "start") {
    run.start = ev;
  } else if (type === "turn_start") {
    run.next = ev;
  } else if (type === "turn") {
    const list = run.turns[ev.method];
    if (!list) throw new Error(`unknown method ${ev.method}`);
    if (ev.turn !== list.length) throw new Error(`${ev.method} turn ${ev.turn} arrived after ${list.length} turns`);
    list.push(ev);
    run.next = null;
  } else if (type === "final") {
    run.finals[ev.method] = ev;
  } else if (type === "done" || type === "cancelled") {
    run.ended = type;
    run.wall = ev.wall_s ?? null;
    run.next = null;
  } else if (type === "error") {
    run.ended = "error";
    run.error = ev.message ?? "unknown error";
    run.next = null;
  }
  return run;
}

/** A method's session is over: its final arrived or the whole run ended. */
export const methodOver = (run, m) => Boolean(run.finals[m] || run.ended);

const sum = (xs, f) => xs.reduce((a, x) => a + f(x), 0);
const mean = (xs) => (xs.length ? xs.reduce((a, b) => a + b, 0) / xs.length : null);

/**
 * Totals over a method's first `n` turns: tokens computed (prefill + handoff re-read), the mean time to first
 * token after a role switch (turns >= 1), and time on GPU (sum of turn times, seconds).
 */
export function totals(turns, n = Infinity) {
  const ts = turns.slice(0, n);
  if (!ts.length) return { n: 0, computed: null, ttft: null, gpu: null };
  return {
    n: ts.length,
    computed: sum(ts, (t) => t.prefill_tokens + t.canon_tokens),
    ttft: mean(ts.filter((t) => t.turn >= 1).map((t) => t.ttft_ms)),
    gpu: sum(ts, (t) => t.turn_ms) / 1000,
  };
}

/** "3.8× fewer" or "1.2× more": how many times less of something RoleKV needed than A2. */
export function timesWord(a2, a5, less, more) {
  if (a2 === null || a5 === null || !(a2 > 0) || !(a5 > 0)) return null;
  const x = a2 / a5;
  return x >= 1 ? { text: `${x.toFixed(1)}× ${less}`, kind: "win", x } : { text: `${(1 / x).toFixed(1)}× ${more}`, kind: "loss", x };
}

/** "42% less", "4% more" or "about the same". */
export function percentWord(a2, a5) {
  if (a2 === null || a5 === null || !(a2 > 0)) return null;
  const pct = 1 - a5 / a2;
  if (Math.abs(pct) < 0.01) return { text: "about the same", kind: "", pct };
  return pct > 0 ? { text: `${Math.round(pct * 100)}% less`, kind: "win", pct } : { text: `${Math.round(-pct * 100)}% more`, kind: "loss", pct };
}

/**
 * The comparisons on the page, over complete rounds: a round counts once each method has sent that turn or
 * its session is over. Mid-round totals are not lopsided (A2 runs first in each round), and a method whose
 * session ended early keeps its whole session, so at the end the comparisons cover both whole sessions, like
 * the totals next to them. `limit` caps the rounds considered, e.g. to the rounds already played.
 */
export function compare(run, limit = Infinity) {
  const has = (m, r) => run.turns[m].length > r || methodOver(run, m);
  let n = 0;
  while (n < limit && METHODS.some((m) => run.turns[m].length > n) && METHODS.every((m) => has(m, n))) n++;
  const a = totals(run.turns["A2"], n), b = totals(run.turns["A5-H1"], n);
  return {
    n, a2: a, a5: b,
    tokens: n ? timesWord(a.computed, b.computed, "fewer", "more") : null,
    ttft: n ? timesWord(a.ttft, b.ttft, "faster", "slower") : null,
    gpu: n ? percentWord(a.gpu, b.gpu) : null,
  };
}

/** Estimated dollars per million sessions: GPU seconds × an assumed $/GPU-hour. */
export const perMillion = (gpuS, usdPerHour) => (gpuS === null ? null : gpuS * usdPerHour / 3600 * 1e6);

// ---------- The 3D cache's size ----------

/**
 * Tokens a block holds, so a live session has room to grow without reflowing at every turn: 20% over what is
 * needed, rounded up to 256. Never shrinks during a run.
 */
export function capacityFor(needed, current = 0) {
  if (needed <= current) return current;
  return Math.max(256, Math.ceil((needed * 1.2) / 256) * 256);
}

/** The violet handoff band is drawn at least this share of the block wide (on the boards only). */
export const MIN_CANON_FRACTION = 0.04;

/**
 * A turn's spans as the boards draw them: the handoff re-read (canon) widened to at least `minFraction` of
 * `scale`, growing left over cache cells so the computed (coral) tokens keep their true size. At 30 functions
 * the real band is about 2% of the block, too thin to see from the back of a room. The cards keep the true
 * numbers.
 */
export function displaySpans(turn, scale, minFraction = MIN_CANON_FRACTION) {
  const s = tapeSpans(turn);
  if (!s.canon) return s;
  const min = Math.ceil(scale * minFraction);
  if (s.canon[1] - s.canon[0] >= min) return s;
  const from = Math.max(0, s.canon[1] - min);
  return { ...s, reused: [0, from], canon: [from, s.canon[1]] };
}

/** Tokens per cell: 8 like the homepage, more for long contexts so a layer stays at most `maxCells` cells. */
export const binFor = (capacity, maxCells = 400) => Math.max(8, Math.ceil(capacity / maxCells));

/** The longest session either method has reached in its first `n` turns (the tape's end, in tokens). */
export function sessionEnd(run, n = Infinity) {
  let end = 0;
  for (const m of METHODS) for (const t of run.turns[m].slice(0, n)) end = Math.max(end, tapeSpans(t).end);
  return end;
}

// ---------- Playing turns on the boards ----------

/**
 * The next round (turn index) the boards may play after `shown`, or null. Both boards play a round together,
 * so a round waits until each method has sent that turn or is over (sessions may differ in length).
 */
export function nextRound(run, shown) {
  const r = shown + 1;
  const has = (m) => run.turns[m].length > r;
  if (!METHODS.some(has)) return null;
  return METHODS.every((m) => has(m) || methodOver(run, m)) ? r : null;
}

/** Rounds that could play right after `shown` without waiting, for speeding up a backlog. */
export function backlog(run, shown) {
  let n = 0;
  for (let r = nextRound(run, shown); r !== null; r = nextRound(run, r)) n++;
  return n;
}

/** Read and write durations of one turn on the boards (ms): time grows with the tokens, like the homepage. */
export function playMs(turn, speed = 1) {
  const read = Math.round(Math.min(1700, Math.max(260, (turn.prefill_tokens + turn.canon_tokens) * 0.95)) / speed);
  const write = Math.round(Math.min(800, Math.max(200, turn.decode_tokens * 9)) / speed);
  return { read, write };
}

// ---------- Words ----------

export const fmtMs = (ms) => (ms === null || ms === undefined ? "—" : ms >= 1000 ? `${(ms / 1000).toFixed(2)} s` : `${Math.round(ms)} ms`);
export const fmtS = (s) => (s === null || s === undefined ? "—" : `${s.toFixed(s < 10 ? 2 : 1)} s`);
export const fmtUsd = (v) => (v === null || v === undefined ? "—" : v >= 100 ? `$${fmtInt(v)}` : v >= 1 ? `$${v.toFixed(2)}` : `$${v.toPrecision(2)}`);

/** How a method's session ended, in words. */
export function endWords(f) {
  if (f.end === "APPROVE") return `The Reviewer approved after ${f.n_turns} turns.`;
  if (f.end === "INVALID") return `The Reviewer's reply was neither APPROVE nor REVISE, which ends the session (${f.n_turns} turns).`;
  return `Final revision after ${f.verdicts.length} REVISE verdicts (${f.n_turns} turns).`;
}

/** The caption while a round plays: what each method reads before answering. */
export function readCaption(run, r) {
  const parts = METHODS.filter((m) => run.turns[m][r])
    .map((m) => `${NAME[m]} ${fmtInt(run.turns[m][r].prefill_tokens + run.turns[m][r].canon_tokens)}`);
  const role = (run.turns["A5-H1"][r] ?? run.turns["A2"][r]).role;
  return `Turn ${r + 1} · the ${ROLE[role]} reads · computed: ${parts.join(" · ")}`;
}
