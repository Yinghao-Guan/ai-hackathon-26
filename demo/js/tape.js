// The session tape: one turn of a session drawn to scale as a strip of tokens.
// Paper = tokens in the session so far. Cyan rule = held in the KV cache, reused for free.
// Coral ink = prefilled this turn. Outline = written by the role this turn. Violet underline = the H1
// handoff: the previous turn's header and output, re-read at the start of this turn. Ink is cost;
// everything uninked was free.
import { tapeSpans } from "./metrics.js";
import { el, svg, fmtInt, reducedMotion } from "./ui.js";

/** Sweep time grows with the tokens recomputed, so a long re-read is felt as a long sweep. */
export const sweepMs = (tokens, { min = 160, perToken = 0.42, max = 1700 } = {}) =>
  Math.round(Math.max(min, Math.min(max, tokens * perToken)));

export function tapeLabel(method, i, turn) {
  const s = tapeSpans(turn);
  return `${method}, turn ${i + 1} (${turn.role}): ${fmtInt(turn.prefill_tokens)} tokens computed, `
    + `${fmtInt(turn.reused_tokens)} reused from the cache, ${fmtInt(turn.canon_tokens)} re-read at handoff, `
    + `${fmtInt(turn.decode_tokens)} written. Session length ${fmtInt(s.end)} tokens.`;
}

// --l/--r are the span's edges; CSS snaps both to whole pixels so neighbouring segments share an edge exactly.
const pos = (span, scale) => ({
  "--l": `${(100 * span[0] / scale).toFixed(3)}%`,
  "--r": `${(100 * span[1] / scale).toFixed(3)}%`,
  "--w": `${(100 * (span[1] - span[0]) / scale).toFixed(3)}%`,
});

// An outline narrower than a few pixels draws as a dark sliver, not a box. Mark such segments once the
// tape's real width is known so CSS can drop the outline (the tooltip still reports the tokens written).
const NARROW_PX = 8;
const narrowWatch = typeof ResizeObserver === "undefined" ? null : new ResizeObserver((entries) => {
  for (const { target, contentRect } of entries) {
    // Tapes are rebuilt on every task switch; a removed tape gets one last callback, so stop watching it.
    if (!target.isConnected) { narrowWatch.unobserve(target); continue; }
    const scale = Number(target.dataset.scale);
    for (const seg of target.querySelectorAll(".seg.written")) {
      seg.classList.toggle("narrow", (Number(seg.dataset.tokens) / scale) * contentRect.width < NARROW_PX);
    }
  }
});
const written = (span, scale, style) =>
  el("span", { class: "seg written", "data-tokens": span[1] - span[0], style: style ?? pos(span, scale) });

/**
 * One turn as a tape. `scale` is the token count that fills the full width, shared by every tape that is
 * compared. With `pending` the ink starts hidden; call sweep() to draw it.
 */
export function tapeEl(turn, scale, { method, index, pending = false, perToken } = {}) {
  const s = tapeSpans(turn);
  const inkTokens = s.recomputed[1] - s.recomputed[0];
  const inkMs = sweepMs(inkTokens, perToken ? { perToken } : undefined);
  const canonMs = s.canon ? sweepMs(s.canon[1] - s.canon[0], perToken ? { perToken } : undefined) : 0;
  const node = el("div", {
    class: `tape${pending ? " pending" : ""}`, role: "img",
    "aria-label": tapeLabel(method, index, turn), "data-scale": scale,
    style: { "--ink-ms": `${inkMs}ms`, "--canon-ms": `${canonMs}ms` },
  },
  el("span", { class: "seg paper", style: pos([0, s.end], scale) }),
  s.reused[1] > 0 ? el("span", { class: "seg cache", style: pos(s.reused, scale) }) : null,
  inkTokens > 0 ? el("span", { class: "seg ink", style: pos(s.recomputed, scale) }) : null,
  s.written[1] > s.written[0] ? written(s.written, scale) : null,
  s.canon ? el("span", { class: "seg canon", style: pos(s.canon, scale) }) : null);
  narrowWatch?.observe(node);
  node.duration = reducedMotion() ? 0 : inkMs + canonMs;
  return node;
}

/** Draw a pending tape's ink. Resolves when the sweep has finished. */
export function sweep(node) {
  return new Promise((resolve) => {
    if (reducedMotion()) { node.classList.remove("pending"); resolve(); return; }
    requestAnimationFrame(() => requestAnimationFrame(() => {
      node.classList.remove("pending");
      setTimeout(resolve, node.duration);
    }));
  });
}

/**
 * The whole session as a map, for explaining the mechanism: shared context under one cache rule, a flag
 * where each role's adapter switches on, the role outputs outlined, and every handoff re-read in violet.
 */
export function sessionMap(turns, task) {
  const spans = turns.map(tapeSpans);
  const end = spans[spans.length - 1].end;
  // Everything before the first role output is task, repository file and the first header.
  const ctx = Math.min(task?.ctx_tokens ?? spans[0].written[0], spans[0].written[0]);
  const node = el("div", { class: "session-map", role: "img",
    "aria-label": `The session as one tape of ${fmtInt(end)} tokens: ${fmtInt(ctx)} tokens of task and repository file, then ${turns.length} role turns. Each role's adapter switches on at its header; every handoff re-reads the previous output with base weights.` });

  // Overview at true scale: the shared context dominates; the turns are the short tail.
  const overview = el("div", { class: "tape map" },
    el("span", { class: "seg paper", style: pos([0, end], end) }),
    el("span", { class: "seg cache", style: pos([0, end], end) }),
    el("span", { class: "seg zoom", style: pos([ctx, end], end) }));
  const overviewLabels = el("div", { class: "map-labels", "aria-hidden": "true" },
    el("span", { class: "map-label", style: pos([0, ctx], end), text: `Task + repository file · ${fmtInt(ctx)} tokens · read once, shared by both roles` }),
    el("span", { class: "map-label end", style: pos([ctx, end], end), text: `${turns.length} turns · ${fmtInt(end - ctx)} tokens` }));

  // The turns, enlarged ×(end / (end − ctx)) so each role's output and handoff are visible.
  const zoomScale = end - ctx;
  const zpos = (span) => pos([span[0] - ctx, span[1] - ctx], zoomScale);
  const detail = el("div", { class: "tape map detail", "data-scale": zoomScale },
    el("span", { class: "seg paper", style: zpos([ctx, end]) }),
    el("span", { class: "seg cache", style: zpos([ctx, end]) }));
  narrowWatch?.observe(detail);
  const flags = el("div", { class: "map-flags", "aria-hidden": "true" });
  // A flag closer than this (in % of the width) to the previous one gets a taller stem so labels don't collide.
  const FLAG_GAP = 8;
  let prevAt = -Infinity, prevEnd = null;
  // Turn i's handoff re-read happens at the start of turn i+1 (tapeSpans: canon sits after the cache), and it
  // re-reads exactly turn i's header + output. So turn i starts where turn i+1's re-read starts, and the
  // header is that re-read minus turn i's output. The last turn is never re-read, so it borrows the others'
  // header length.
  const hdrs = turns.slice(0, -1).map((t, i) => (turns[i + 1].canon_tokens || 0) - t.decode_tokens).filter((v) => v > 0);
  const hdr = hdrs.length ? Math.min(...hdrs) : 0;
  spans.forEach((sp, i) => {
    const next = spans[i + 1];
    let turnStart = next?.canon ? next.canon[0] : Math.max(ctx, sp.written[0] - hdr);
    // Only the end-of-turn markers (not re-read) separate back-to-back turns; close that gap so they share
    // one divider instead of drawing two outlines a hair apart.
    if (prevEnd !== null && turnStart > prevEnd && turnStart - prevEnd <= 3) turnStart = prevEnd;
    const box = written([turnStart, sp.end], zoomScale, zpos([turnStart, sp.end]));
    // Back-to-back turns (no tool output between them) share one divider instead of two touching outlines.
    if (prevEnd !== null && turnStart <= prevEnd) box.classList.add("joined");
    detail.append(box);
    if (sp.canon) detail.append(el("span", { class: "seg canon", style: zpos(sp.canon) }));
    const at = 100 * (turnStart - ctx) / zoomScale;
    const stacked = at - prevAt < FLAG_GAP;
    flags.append(el("span", { class: `flag ${turns[i].role}${at > 88 ? " flip" : ""}${stacked ? " stack" : ""}`, style: { "--l": `${at.toFixed(3)}%` } },
      el("b", { text: turns[i].role === "coder" ? "Coder" : "Reviewer" })));
    // A stacked label sits above the row, so later flags only need room from the last unstacked one.
    if (!stacked) prevAt = at;
    if (stacked) flags.classList.add("has-stack");
    prevEnd = sp.end;
  });
  node.append(overviewLabels, overview, svg("svg", { class: "map-zoom", viewBox: "0 0 100 10", preserveAspectRatio: "none", "aria-hidden": "true" },
    svg("polygon", { points: `${(100 * ctx / end).toFixed(3)},0 100,0 100,10 0,10` })), flags, detail,
    el("p", { class: "map-scale", "aria-hidden": "true", text: `Lower tape: the turns enlarged ${+(end / zoomScale).toFixed(1)}×` }));
  return node;
}
