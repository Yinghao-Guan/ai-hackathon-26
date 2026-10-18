// Hand-written SVG charts shared by the dashboard and the presentation stage.
import { METHOD_META, mean, sweepSummary } from "./metrics.js";
import { svg, attachTooltip, niceTicks, kfmt, fmtInt, fmtPct, ctxWords, plural, methodName } from "./ui.js";

export const SHORT_NAMES = { "A0": "Base", "A1": "Shared LoRA", "A2": "Per-role LoRAs", "A3": "Naive reuse", "A5-H1": "RoleKV", "A5-H2": "RoleKV, keep KV" };

function frame({ w, h, m, xMax, yMin, yMax, xTicks, yTicks, xFmt, yFmt, xTitle, yTitle, label }) {
  const X = (v) => m.l + (v / xMax) * (w - m.l - m.r);
  const Y = (v) => h - m.b - ((v - yMin) / (yMax - yMin)) * (h - m.t - m.b);
  const root = svg("svg", { class: "chart", viewBox: `0 0 ${w} ${h}`, width: w, height: h, role: "group", "aria-label": label });
  for (const t of yTicks) {
    root.append(svg("line", { class: t === yMin ? "axis" : "grid", x1: m.l, x2: w - m.r, y1: Y(t), y2: Y(t) }));
    root.append(svg("text", { class: "tick", x: m.l - 10, y: Y(t) + 4, "text-anchor": "end", text: yFmt(t) }));
  }
  for (const t of xTicks) root.append(svg("text", { class: "tick", x: X(t), y: h - m.b + 22, "text-anchor": "middle", text: xFmt(t) }));
  root.append(svg("text", { class: "axis-title", x: m.l, y: h - 6, text: xTitle }));
  root.append(svg("text", { class: "axis-title", x: 0, y: m.t - 18, text: yTitle }));
  return { root, X, Y };
}

function point(x, y, r, cls, tipLines, place) {
  const g = svg("g", { class: `pt ${cls}`, tabindex: "0", role: "img", "aria-label": tipLines.filter(Boolean).join(". ") },
    svg("circle", { cx: x, cy: y, r }));
  attachTooltip(g, () => tipLines, place);
  return g;
}

/** Merge labels of points closer than `gap` px so coincident methods read "Base, Naive reuse". */
function cluster(items, gap) {
  const groups = [];
  for (const it of items) {
    const g = groups.find((gr) => gr.some((o) => Math.hypot(o.x - it.x, o.y - it.y) < gap));
    if (g) g.push(it); else groups.push([it]);
  }
  return groups.map((g) => ({ x: Math.max(...g.map((o) => o.x)), y: mean(g.map((o) => o.y)), text: g.map((o) => o.label).join(", ") }));
}

/** Cost against pass rate, one point per method. RoleKV and the per-role baseline carry their own colours. */
export function qualityChart(a, { w, h, large = false }) {
  const rows = a.rows.filter((r) => r.available);
  const m = large ? { l: 72, r: 160, t: 56, b: 64 } : { l: 56, r: 130, t: 44, b: 54 };
  const { max: xMax, ticks: xTicks } = niceTicks(Math.max(...rows.map((r) => r.meanPrefill)) * 1.08);
  const rates = rows.map((r) => r.rate);
  const yMin = Math.max(0, Math.floor((Math.min(...rates) - 0.05) * 10) / 10);
  const yMax = Math.min(1, Math.ceil((Math.max(...rates) + 0.05) * 10) / 10);
  const yTicks = []; for (let v = yMin; v <= yMax + 1e-9; v += 0.1) yTicks.push(+v.toFixed(2));
  const { root, X, Y } = frame({
    w, h, m, xMax, yMin, yMax, xTicks, yTicks, xFmt: kfmt, yFmt: (v) => `${Math.round(v * 100)}%`,
    xTitle: "Tokens re-read per task (prefill, including handoff re-reads)", yTitle: "Tasks passed",
    label: "Cost against quality: tokens re-read per task against share of tasks passed, one point per method",
  });
  // Direction cue: up and to the left is better.
  const cx = m.l + 18, cy = m.t + 8;
  root.append(svg("g", { class: "better", "aria-hidden": "true" },
    svg("path", { d: `M ${cx + 58} ${cy + 30} L ${cx + 4} ${cy + 4}` }),
    svg("path", { d: `M ${cx + 4} ${cy + 15} L ${cx + 4} ${cy + 4} L ${cx + 15} ${cy + 4}` }),
    svg("text", { x: cx + 66, y: cy + 36, text: "Better: fewer tokens, more tasks passed" })));

  const role = (mth) => (mth === "A5-H1" ? "hero" : mth === "A2" ? "base" : "quiet");
  const quiet = [], strongPts = [];
  const order = [...rows].sort((p, q) => (role(p.method) !== "quiet") - (role(q.method) !== "quiet"));
  for (const r of order) {
    const x = X(r.meanPrefill), y = Y(r.rate), kind = role(r.method);
    const tip = [`${methodName(r.method)} (${r.method})`, METHOD_META[r.method].desc, `Passed ${r.passed} of ${r.n} (${fmtPct(r.rate)})`, `${fmtInt(r.meanPrefill)} tokens re-read per task`, `${r.run.run_id} on the ${r.run.machine}`];
    root.append(point(x, y, kind === "quiet" ? (large ? 7 : 6) : (large ? 11 : 8), kind, tip));
    if (kind === "quiet") quiet.push({ x, y, label: SHORT_NAMES[r.method] });
    else {
      strongPts.push({ x, y, r: large ? 11 : 8 });
      // The two compared methods are labelled hard against their own filled points.
      root.append(svg("text", { class: `pt-label ${kind}`, x: x + (large ? 19 : 14), y: y + (large ? 6 : 5), text: methodName(r.method) }));
    }
  }
  const fs = large ? 16 : 14, gap = large ? 16 : 12;
  for (const l of cluster(quiet, large ? 34 : 26)) {
    // A quiet label that would land beside a compared method's point goes on the far side of its own circle,
    // so a filled point is only ever labelled with its own name. When that side has no room inside the plot
    // (text width estimated as in sweepChart), the label goes under its circles instead of over the y axis.
    const crowded = strongPts.some((p) => Math.abs(p.y - l.y) < p.r + 10 && p.x > l.x - 4 && p.x - l.x < 160);
    const lx = Math.min(...quiet.filter((q) => Math.hypot(q.x - l.x, q.y - l.y) < 40).map((q) => q.x));
    const fitsLeft = lx - gap - l.text.length * fs * 0.56 >= m.l;
    const below = l.y + gap + fs <= h - m.b - 4 ? l.y + gap + fs : l.y - gap;
    const spot = !crowded ? { x: l.x + gap, y: l.y + 5, anchor: "start" }
      : fitsLeft ? { x: lx - gap, y: l.y + 5, anchor: "end" }
      : { x: lx - (large ? 7 : 6), y: below, anchor: "start" };
    root.append(svg("text", { class: "pt-label quiet", x: spot.x, y: spot.y, "text-anchor": spot.anchor, text: l.text }));
  }
  return root;
}

/** Tokens saved per task against shared-context size. */
export function sweepChart(a, { w, h, large = false }) {
  const { points } = a.sweep;
  const m = large ? { l: 72, r: 40, t: 56, b: 64 } : { l: 56, r: 30, t: 44, b: 54 };
  const { max: xMax, ticks: xTicks } = niceTicks(Math.max(...points.map((p) => p.ctxMedian)) * 1.05);
  const { max: yMax, ticks: yTicks } = niceTicks(Math.max(...points.map((p) => p.meanSaved ?? 0)) * 1.04);
  const { root, X, Y } = frame({
    w, h, m, xMax, yMin: 0, yMax, xTicks, yTicks, xFmt: kfmt, yFmt: kfmt,
    xTitle: "Shared context per task (median tokens)", yTitle: "Tokens saved per task",
    label: "Context sweep: tokens RoleKV saves per task against shared-context size",
  });
  const valid = points.filter((p) => p.meanSaved !== null);
  if (valid.length > 1) root.append(svg("polyline", { class: "line", points: valid.map((p) => `${X(p.ctxMedian)},${Y(p.meanSaved)}`).join(" ") }));
  // Each label tries four spots around its point and takes the first that crosses no line segment and no
  // earlier label, judged on an estimated text box.
  const fs = large ? 19 : 15;
  const pts = valid.map((p) => [X(p.ctxMedian), Y(p.meanSaved)]);
  const segs = pts.slice(1).map((q, i) => [pts[i], q]);
  const placed = [];
  const hitsSeg = (b) => segs.some(([[x1, y1], [x2, y2]]) => {
    for (let k = 0; k <= 24; k++) {
      const x = x1 + (x2 - x1) * k / 24, y = y1 + (y2 - y1) * k / 24;
      if (x > b.x0 - 3 && x < b.x1 + 3 && y > b.y0 - 3 && y < b.y1 + 3) return true;
    }
    return false;
  });
  const overlaps = (b) => placed.some((o) => b.x0 < o.x1 && b.x1 > o.x0 && b.y0 < o.y1 && b.y1 > o.y0);
  valid.forEach((p, i) => {
    const [x, y] = pts[i];
    const tip = [`${ctxWords(p.k)} tokens of context (median ${fmtInt(p.ctxMedian)})`, `Saved ${fmtInt(p.meanSaved)} tokens per task`, `${fmtPct(p.fractionSaved)} of what per-role LoRAs re-read, ${plural(p.n, "pair")}`, `${p.a2Run.run_id} vs ${p.a5Run.run_id}`];
    root.append(point(x, y, large ? 8 : 6, "hero", tip));
    const text = `${fmtPct(p.fractionSaved, 0)} saved, n=${p.n}`;
    const tw = text.length * fs * 0.56;
    const spots = [[1, -1], [1, 1], [-1, -1], [-1, 1]].map(([sx, sy]) => {
      const tx = x + sx * 12, ty = sy < 0 ? y - 12 : y + fs + 8;
      const b = sx > 0 ? { x0: tx, x1: tx + tw } : { x0: tx - tw, x1: tx };
      return { tx, ty, anchor: sx > 0 ? "start" : "end", b: { ...b, y0: ty - fs, y1: ty + 3 } };
    }).filter((sp) => sp.b.x0 >= m.l && sp.b.x1 <= w - 4);
    const others = pts.filter((_, j) => j !== i);
    const ownsSpot = (b) => !others.some(([ox]) => ox > b.x0 - 10 && ox < b.x1 + 10);
    const pick = spots.find((sp) => ownsSpot(sp.b) && !hitsSeg(sp.b) && !overlaps(sp.b))
      ?? spots.find((sp) => ownsSpot(sp.b) && !overlaps(sp.b))
      ?? spots.find((sp) => !overlaps(sp.b)) ?? spots[0];
    if (!pick) return;
    placed.push(pick.b);
    root.append(svg("text", { class: "pt-label halo", x: pick.tx, y: pick.ty, "text-anchor": pick.anchor, text }));
  });
  return root;
}

/** The sweep's point in one sentence: the share saved holds while the absolute saving grows. */
export function sweepHeadline(points) {
  const s = sweepSummary(points);
  if (!s) return "Saving by context size";
  const share = Math.round(s.minShare * 100) === Math.round(s.maxShare * 100) ? fmtPct(s.minShare, 0)
    : `${Math.round(s.minShare * 100)}–${fmtPct(s.maxShare, 0)}`;
  return `RoleKV saves ${share} at every size; per task, the saving grows from ${fmtInt(s.firstSaved)} to ${fmtInt(s.lastSaved)} tokens.`;
}
