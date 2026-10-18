// Run: node --test demo/tests/
import { test } from "node:test";
import assert from "node:assert/strict";
import { qualityChart } from "../js/charts.js";

// The charts only create SVG nodes and attach tooltips, so a minimal stand-in for document is enough to
// read back where each label lands.
class FakeNode {
  constructor(tag) { this.tag = tag; this.attrs = {}; this.children = []; this.textContent = ""; }
  setAttribute(k, v) { this.attrs[k] = String(v); }
  removeAttribute(k) { delete this.attrs[k]; }
  append(...nodes) { this.children.push(...nodes); }
  addEventListener() {}
}
globalThis.document = { createElementNS: (_ns, tag) => new FakeNode(tag), getElementById: () => null };
const walk = (n) => [n, ...n.children.flatMap(walk)];

/** Cost-vs-quality rows as analyze() returns them: [method, mean prefill per task, tasks passed of 100]. */
const rows = (spec) => ({
  rows: spec.map(([method, meanPrefill, passed]) => ({
    method, available: true, meanPrefill, passed, n: 100, rate: passed / 100,
    run: { run_id: `eval-${method}`, machine: "4090" },
  })),
});

test("a crowded quiet label never crosses the y axis; it moves under its points", () => {
  // The 1.5B eval on the 4090 (eval-1.5b-4090-202610180231): Shared LoRA, Naive reuse and RoleKV-keep-KV
  // sit next to RoleKV at about 2k tokens, so their merged label went left, over the 60% tick and off the chart.
  const real = rows([["A0", 2038, 51], ["A1", 2060, 61], ["A2", 7013, 59], ["A3", 2061, 62], ["A5-H1", 2197, 61], ["A5-H2", 2059, 60]]);
  for (const [w, h, large] of [[928, 316, false], [1100, 374, false], [1400, 520, true]]) {
    const fs = large ? 16 : 14, ml = large ? 72 : 56;
    const labels = walk(qualityChart(real, { w, h, large })).filter((n) => n.attrs.class === "pt-label quiet");
    const merged = labels.find((n) => n.textContent === "Shared LoRA, Naive reuse, RoleKV, keep KV");
    assert.ok(merged, `merged label missing at w=${w}`);
    for (const n of labels) {
      const x = Number(n.attrs.x), width = n.textContent.length * fs * 0.56;
      const left = n.attrs["text-anchor"] === "end" ? x - width : x;
      assert.ok(left >= ml, `"${n.textContent}" starts at x=${left.toFixed(0)}, left of the plot (${ml}) at w=${w}`);
      assert.ok(Number(n.attrs.y) <= h - (large ? 64 : 54), `"${n.textContent}" is below the x axis at w=${w}`);
    }
  }
});

test("a crowded quiet label with room on the left still goes left of its points", () => {
  // Quiet points far enough right that the label fits between the y axis and them.
  const roomy = rows([["A2", 7000, 59], ["A5-H1", 6000, 61], ["A1", 5900, 61]]);
  const label = walk(qualityChart(roomy, { w: 928, h: 316 })).find((n) => n.attrs.class === "pt-label quiet");
  assert.equal(label.attrs["text-anchor"], "end");
});
