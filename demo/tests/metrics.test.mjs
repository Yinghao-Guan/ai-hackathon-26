// Run: node --test demo/tests/
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import {
  parseJSONL, isolateDataset, buildRuns, analyze, headline, pairTasks, passRate, sweepSeries,
  selectDefaultContextRuns, auditTask, ttftEligible, turnSegments, turnTotals, meanGpuSeconds,
  tapeSpans, tapeScale, featuredTask, replayContextSizes, runFor, sweepSummary, isSweepRun,
} from "../js/metrics.js";

// ---------- toy fixture helpers ----------
function turn(run_id, method, task_id, i, o = {}) {
  return {
    type: "turn", run_id, machine: o.machine ?? "4090", method, task_id, turn: i,
    role: i % 2 ? "reviewer" : "coder", prefill_tokens: o.prefill ?? 100, reused_tokens: o.reused ?? 0,
    canon_tokens: o.canon ?? 0, decode_tokens: o.decode ?? 10, ttft_ms: "ttft" in o ? o.ttft : 20,
    turn_ms: 100, kv_mb: 1.5, text: o.text ?? "APPROVE",
  };
}
function task(run_id, method, task_id, o = {}) {
  return {
    type: "task", run_id, machine: o.machine ?? "4090", method, task_id, ctx_tokens: o.ctx ?? 3000,
    passed: o.passed ?? true, n_turns: o.n_turns ?? 2, prefill_tokens_total: o.total,
    decode_tokens_total: 20, gpu_s: "gpu" in o ? o.gpu : 1.0, peak_kv_mb: 2.0,
  };
}
const jsonl = (recs) => recs.map((r) => JSON.stringify(r)).join("\n");
const load = (recs) => analyze(isolateDataset(parseJSONL(jsonl(recs))));

/** Two-turn session whose task total is consistent with its turns. */
function session(run_id, method, task_id, [p0, p1], { canon0 = 0, ...o } = {}) {
  return [
    turn(run_id, method, task_id, 0, { prefill: p0, canon: canon0, machine: o.machine }),
    turn(run_id, method, task_id, 1, { prefill: p1, machine: o.machine }),
    task(run_id, method, task_id, { total: p0 + canon0 + p1, ...o }),
  ];
}

// 1. JSONL parsing
test("parser rejects malformed lines and unknown types without dropping valid ones", () => {
  const text = [
    JSON.stringify(turn("r-ctx3k", "A2", 1, 0)),
    "{not json",
    JSON.stringify({ type: "banana", run_id: "x" }),
    "",
    JSON.stringify({ ...turn("r-ctx3k", "A2", 1, 1), prefill_tokens: -5 }),
    JSON.stringify({ ...turn("r-ctx3k", "A2", 1, 1), method: "A9" }),
    JSON.stringify(task("r-ctx3k", "A2", 1, { total: 200 })),
    "[1,2]",
  ].join("\n");
  const p = parseJSONL(text);
  assert.equal(p.accepted, 2);
  assert.equal(p.rejected, 5);
  assert.deepEqual(p.errors.map((e) => e.line), [2, 3, 5, 6, 8]);
  assert.match(p.errors[0].reason, /malformed JSON/);
  assert.match(p.errors[1].reason, /unknown record type/);
  assert.match(p.errors[2].reason, /prefill_tokens/);
  assert.match(p.errors[3].reason, /method/);
});

test("parser rejects missing required fields, duplicates and run_id identity conflicts", () => {
  const t = turn("r-ctx3k", "A2", 1, 0);
  const { canon_tokens, ...noCanon } = t;
  const p = parseJSONL(jsonl([t, t, noCanon, { ...turn("r-ctx3k", "A5-H1", 2, 0) }]));
  assert.equal(p.accepted, 1);
  assert.match(p.errors[0].reason, /duplicate/);
  assert.match(p.errors[1].reason, /missing field "canon_tokens"/);
  assert.match(p.errors[2].reason, /run_id "r-ctx3k" was A2\/4090/);
});

test("empty input yields empty mode, not a crash", () => {
  const d = isolateDataset(parseJSONL("\n\n"));
  assert.equal(d.mode, "empty");
});

// 2. Pass rate
test("pass rate counts passed task records over valid task records", () => {
  const recs = [
    ...session("a2-ctx3k", "A2", 1, [100, 100], { passed: true }),
    ...session("a2-ctx3k", "A2", 2, [100, 100], { passed: false }),
    ...session("a2-ctx3k", "A2", 3, [100, 100], { passed: true }),
    ...session("a2-ctx3k", "A2", 4, [100, 100], { passed: true }),
  ];
  const a = load(recs);
  assert.deepEqual(passRate(a.runs.get("a2-ctx3k")), { n: 4, passed: 3, rate: 0.75 });
  assert.deepEqual(passRate(null), { n: 0, passed: 0, rate: null });
});

// 3 + 4. Headline ratio on paired task ids
test("headline ratio and percent saved use only paired task_ids", () => {
  const recs = [
    ...session("a2-ctx3k", "A2", 1, [3000, 3200]),          // total 6200
    ...session("a2-ctx3k", "A2", 2, [3000, 3800]),          // total 6800
    ...session("a2-ctx3k", "A2", 99, [50000, 50000]),       // unpaired outlier, must be ignored
    ...session("a5-ctx3k", "A5-H1", 1, [3000, 200], { canon0: 100 }), // total 3300
    ...session("a5-ctx3k", "A5-H1", 2, [3000, 300], { canon0: 0 }),   // total 3300
    ...session("a5-ctx3k", "A5-H1", 7, [1, 1]),             // unpaired
  ];
  const a = load(recs);
  const h = a.headline;
  assert.equal(h.ok, true);
  assert.equal(h.n, 2);
  assert.equal(h.meanA, 6500);
  assert.equal(h.meanB, 3300);
  assert.ok(Math.abs(h.ratio - 6500 / 3300) < 1e-12);
  assert.ok(Math.abs(h.pctSaved - 100 * (1 - 3300 / 6500)) < 1e-12);
  assert.equal(h.savedPerTask, 3200);
  assert.deepEqual(pairTasks(a.runs.get("a2-ctx3k"), a.runs.get("a5-ctx3k")).map((p) => p.task_key), ["1", "2"]);
  assert.equal(a.headlineTrusted, true);
});

test("headline refuses unmatched tasks, missing runs and zero denominators", () => {
  const unmatched = load([...session("a2-ctx3k", "A2", 1, [10, 10]), ...session("a5-ctx3k", "A5-H1", 2, [5, 5])]);
  assert.equal(unmatched.headline.ok, false);
  assert.match(unmatched.headline.reason, /no overlapping/);

  const noA5 = load(session("a2-ctx3k", "A2", 1, [10, 10]));
  assert.equal(noA5.headline.ok, false);
  assert.match(noA5.headline.reason, /no A5-H1 task records/);
  assert.equal(noA5.rows.find((r) => r.method === "A5-H1").available, false);

  const zero = load([...session("a2-ctx3k", "A2", 1, [10, 10]), ...session("a5-ctx3k", "A5-H1", 1, [0, 0])]);
  assert.equal(zero.headline.ok, false);
  assert.match(zero.headline.reason, /zero/);
});

// 5. Sweep aggregation
test("sweep pairs by task within each context class and reports mean saved and fraction", () => {
  const recs = [
    ...session("a2-ctx1k", "A2", 1, [1000, 1100], { ctx: 1000 }),   // 2100
    ...session("a2-ctx1k", "A2", 2, [1000, 1300], { ctx: 1040 }),   // 2300
    ...session("a5-ctx1k", "A5-H1", 1, [1000, 100], { ctx: 1000 }), // 1100
    ...session("a5-ctx1k", "A5-H1", 2, [1000, 300], { ctx: 1040 }), // 1300
    ...session("a2-ctx8k", "A2", 1, [8000, 8100], { ctx: 8000 }),   // 16100
    ...session("a5-ctx8k", "A5-H1", 3, [8000, 100], { ctx: 8000 }), // different task: no pairs
    ...session("a2-ctx4k", "A2", 1, [4000, 4000], { ctx: 4000 }),   // A2 only
  ];
  const { points, notes } = sweepSeries(buildRuns(isolateDataset(parseJSONL(jsonl(recs)))));
  assert.equal(points.length, 1);
  const p = points[0];
  assert.equal(p.k, 1);
  assert.equal(p.n, 2);
  assert.equal(p.ctxMedian, 1020);
  assert.equal(p.meanSaved, 1000);         // (2100+2300)/2 − (1100+1300)/2
  assert.ok(Math.abs(p.fractionSaved - 1000 / 2200) < 1e-12);
  assert.ok(notes.some((n) => /~4k: only A2/.test(n)));
  assert.ok(notes.some((n) => /~8k: no shared task_ids/.test(n)));
});

// 6. Canonicalization counted exactly once
test("canonicalization is counted once in turn totals and segments, and the audit enforces it", () => {
  const t0 = turn("a5-ctx3k", "A5-H1", 1, 0, { prefill: 3000, canon: 150 });
  const t1 = turn("a5-ctx3k", "A5-H1", 1, 1, { prefill: 200, reused: 3150 });
  assert.deepEqual(turnSegments(t0), { recomputed: 3000, reused: 0, canon: 150 });
  assert.equal(turnTotals([t0, t1]).compute, 3350);

  const ok = auditTask(task("a5-ctx3k", "A5-H1", 1, { total: 3350 }), [t0, t1]);
  assert.equal(ok.status, "ok");
  const excl = auditTask(task("a5-ctx3k", "A5-H1", 1, { total: 3200 }), [t0, t1]);
  assert.equal(excl.status, "excludes_canon");
  const dbl = auditTask(task("a5-ctx3k", "A5-H1", 1, { total: 3500 }), [t0, t1]); // canon added twice
  assert.equal(dbl.status, "mismatch");
  assert.equal(auditTask(task("x", "A2", 1, { total: 1, n_turns: 3 }), [t0, t1]).status, "turn_count");
});

test("a task total that omits canon makes the headline untrusted and offers a labelled derived value", () => {
  const recs = [
    ...session("a2-ctx3k", "A2", 1, [3000, 3200]),
    turn("a5-ctx3k", "A5-H1", 1, 0, { prefill: 3000, canon: 100 }),
    turn("a5-ctx3k", "A5-H1", 1, 1, { prefill: 200 }),
    task("a5-ctx3k", "A5-H1", 1, { total: 3200 }), // omits the 100 canon tokens
  ];
  const a = load(recs);
  assert.equal(a.headline.ok, true);
  assert.equal(a.headline.meanB, 3200);         // authoritative field is not silently changed
  assert.equal(a.headlineTrusted, false);
  assert.equal(a.derivedHeadline.source, "turns");
  assert.equal(a.derivedHeadline.meanB, 3300);
});

// 7. Missing timings
test("timing is never fabricated: TTFT only for 4090 rows with a value, GPU-s only for complete 4090 runs", () => {
  assert.equal(ttftEligible(turn("r", "A2", 1, 0, { ttft: 12.5 })), true);
  assert.equal(ttftEligible(turn("r", "A2", 1, 0, { ttft: null })), false);
  assert.equal(ttftEligible(turn("r", "A2", 1, 0, { machine: "4070", ttft: 12.5 })), false);

  const p = parseJSONL(jsonl([turn("r-ctx3k", "A2", 1, 0, { ttft: null })]));
  assert.equal(p.accepted, 1);
  assert.equal(p.turns[0].ttft_ms, null);

  const a = load([
    ...session("a2-ctx3k", "A2", 1, [10, 10], { gpu: null }),
    ...session("a0-ctx3k", "A0", 1, [10, 10], { machine: "4070" }),
  ]);
  assert.deepEqual(meanGpuSeconds(a.runs.get("a2-ctx3k")), { value: null, reason: "gpu_s missing" });
  assert.deepEqual(meanGpuSeconds(a.runs.get("a0-ctx3k")), { value: null, reason: "4070 run" });
});

// 8. Mock / real isolation
test("mock records never mix into a real dataset", () => {
  const recs = [
    ...session("mock-a2-ctx3k", "A2", 1, [99999, 99999]),
    ...session("a2-ctx3k", "A2", 1, [10, 10]),
  ];
  const d = isolateDataset(parseJSONL(jsonl(recs)));
  assert.equal(d.mode, "real");
  assert.equal(d.droppedMock, 3);
  assert.ok(d.tasks.every((t) => !t.run_id.startsWith("mock-")));
  assert.equal(isolateDataset(parseJSONL(jsonl(session("mock-a2-ctx3k", "A2", 1, [1, 1])))).mode, "mock");
});

// 9. Default-context selection excludes sweep runs
test("default-context selection uses ~3k runs and excludes sweep runs, by run_id or by median ctx", () => {
  const recs = [
    ...session("a2-1.5b-ctx3k", "A2", 1, [3000, 3000], { passed: true }),
    ...session("a2-1.5b-ctx8k", "A2", 1, [8000, 8000], { passed: false }),
    // No ctx in the run_id: classified by median ctx_tokens (3,1xx → 3k; 7,9xx → 8k).
    ...session("a5-main", "A5-H1", 1, [3000, 100], { ctx: 3100 }),
    ...session("a5-main", "A5-H1", 2, [3000, 100], { ctx: 3150 }),
    ...session("a5-long", "A5-H1", 1, [8000, 100], { ctx: 7900 }),
  ];
  const a = load(recs);
  assert.equal(a.selection.ctxK, 3);
  assert.equal(a.selection.byMethod.A2.run.run_id, "a2-1.5b-ctx3k");
  assert.equal(a.selection.byMethod["A5-H1"].run.run_id, "a5-main");
  assert.equal(a.runs.get("a5-long").ctxK, 8);
  assert.equal(a.rows.find((r) => r.method === "A2").rate, 1);
});

test("without any ~3k run the largest context class is used and flagged as ambiguous", () => {
  const runs = buildRuns(isolateDataset(parseJSONL(jsonl([
    ...session("a2-ctx1k", "A2", 1, [1, 1]),
    ...session("a2-ctx8k", "A2", 1, [1, 1]),
    ...session("a2-ctx8k", "A2", 2, [1, 1]),
  ]))));
  const s = selectDefaultContextRuns(runs);
  assert.equal(s.ctxK, 8);
  assert.equal(s.ambiguous, true);
});

// 10. Loading a second file resets everything
test("analysis of a second log shares no state with the first", () => {
  const first = load([...session("a2-ctx3k", "A2", 1, [3000, 3000]), ...session("a5-ctx3k", "A5-H1", 1, [3000, 100])]);
  const second = load(session("other-ctx3k", "A0", 5, [10, 10]));
  assert.equal(first.headline.ok, true);
  assert.equal(second.headline.ok, false);
  assert.deepEqual([...second.runs.keys()], ["other-ctx3k"]);
  assert.equal(second.selection.byMethod.A2.run, null);
});

// Fixture integrity: the committed mock log satisfies the contract and the accounting invariant.
test("mock fixture parses cleanly, is all mock, and every task passes the accounting audit", () => {
  const path = fileURLToPath(new URL("../mock/mock_events.jsonl", import.meta.url));
  const p = parseJSONL(readFileSync(path, "utf8"));
  assert.equal(p.rejected, 0);
  const d = isolateDataset(p);
  assert.equal(d.mode, "mock");
  const a = analyze(d);
  for (const [id, audit] of a.audits) assert.equal(audit.clean, true, `${id} audit`);
  assert.deepEqual(new Set([...a.runs.values()].map((r) => r.method)), new Set(["A0", "A1", "A2", "A3", "A5-H1", "A5-H2"]));
  assert.deepEqual(a.sweep.points.map((pt) => pt.k), [1, 3, 4, 8]);
  assert.equal(a.headlineTrusted, true);
});

// Session tape geometry, replay defaults and the sweep summary.
test("tape spans place cache, the H1 re-read, prefill and written output end to end", () => {
  // CONTRACT.md section 3: canon_tokens are the previous turn's tokens, re-read at the start of this turn.
  const t = turn("a5-ctx3k", "A5-H1", 1, 1, { prefill: 98, reused: 3305, decode: 19, canon: 28 });
  const s = tapeSpans(t);
  assert.deepEqual(s.reused, [0, 3305]);
  assert.deepEqual(s.canon, [3305, 3333]);
  assert.deepEqual(s.recomputed, [3333, 3431]);
  assert.deepEqual(s.written, [3431, 3450]);
  assert.equal(s.end, 3450);
  // Same session length as a method that re-reads everything: reused + canon + prefill is the input.
  assert.equal(tapeSpans(turn("a2-ctx3k", "A2", 1, 1, { prefill: 3431, decode: 19 })).end, s.end);
  assert.equal(tapeSpans(turn("a2-ctx3k", "A2", 1, 0, { prefill: 3212, decode: 93 })).canon, null);
});

test("the tape scale covers the H1 re-read, so no tape runs past its column", () => {
  // Eval task 466 on the 4090 (eval-1.5b-4090-202610180231): RoleKV takes four turns and its last ends at
  // 2,290 tokens. A scale without canon_tokens was 2,164, which drew that tape at 106% of its column.
  const a2 = [turn("a2", "A2", 466, 0, { prefill: 1759, decode: 152 }), turn("a2", "A2", 466, 1, { prefill: 1936, decode: 4 })];
  const a5 = [
    turn("a5", "A5-H1", 466, 0, { prefill: 1759, decode: 104 }),
    turn("a5", "A5-H1", 466, 1, { reused: 1749, canon: 113, prefill: 108, decode: 79 }),
    turn("a5", "A5-H1", 466, 2, { reused: 1959, canon: 89, prefill: 12, decode: 117 }),
    turn("a5", "A5-H1", 466, 3, { reused: 2050, canon: 126, prefill: 110, decode: 4 }),
  ];
  const scale = tapeScale([...a2, ...a5]);
  assert.equal(scale, 2290);
  for (const t of [...a2, ...a5]) assert.ok(tapeSpans(t).end <= scale);
  assert.equal(tapeScale([]), 1);
});

test("the featured replay task prefers both methods passing, then the most turns", () => {
  const recs = [];
  const add = (method, run, id, n, passed) => {
    for (let i = 0; i < n; i++) recs.push(turn(run, method, id, i));
    recs.push(task(run, method, id, { passed, n_turns: n, total: 100 * n }));
  };
  add("A2", "a2-ctx3k", 1, 6, false); add("A5-H1", "a5-ctx3k", 1, 6, false);
  add("A2", "a2-ctx3k", 2, 4, true); add("A5-H1", "a5-ctx3k", 2, 4, true);
  add("A2", "a2-ctx3k", 3, 2, true); add("A5-H1", "a5-ctx3k", 3, 2, true);
  const a = load(recs);
  assert.equal(featuredTask(a.runs.get("a2-ctx3k"), a.runs.get("a5-ctx3k")), "2");
});

test("replay context sizes need both methods, and runFor picks within one size", () => {
  const a = load([
    ...session("a2-ctx3k", "A2", 1, [3000, 3000]), ...session("a5-ctx3k", "A5-H1", 1, [3000, 100]),
    ...session("a2-ctx8k", "A2", 1, [8000, 8000]),
  ]);
  assert.deepEqual(replayContextSizes(a.runs), [3]);
  assert.equal(runFor(a.runs, "A2", 8).run_id, "a2-ctx8k");
  assert.equal(runFor(a.runs, "A5-H1", 8), null);
});

test("sweep summary reports the share and absolute saving ranges, or null with one point", () => {
  const pts = [
    { k: 1, meanSaved: 1000, fractionSaved: 0.6 }, { k: 3, meanSaved: 4000, fractionSaved: 0.7 },
    { k: 8, meanSaved: 9000, fractionSaved: 0.65 },
  ];
  assert.deepEqual(sweepSummary(pts), { minShare: 0.6, maxShare: 0.7, firstSaved: 1000, lastSaved: 9000, firstK: 1, lastK: 8 });
  assert.equal(sweepSummary(pts.slice(0, 1)), null);
});

test("sweep runs never enter the main comparison, even when they are the only runs", () => {
  const a = load([
    ...session("sweep-A2-ctx2k", "A2", "ctx2000-r0", [2000, 2000], { ctx: 2000 }),
    ...session("sweep-A5-H1-ctx2k", "A5-H1", "ctx2000-r0", [2000, 100], { ctx: 2000 }),
  ]);
  assert.ok(isSweepRun(a.runs.get("sweep-A2-ctx2k")));
  assert.equal(a.selection.byMethod.A2.run, null);
  assert.equal(a.headline.ok, false);  // no quality claim from forced replays
  assert.equal(a.sweep.points.length, 1);  // ...but the sweep itself is plotted
});

test("with eval and sweep runs at one size, quality uses the eval runs and the sweep uses the sweep runs", () => {
  const a = load([
    ...session("eval-A2", "A2", 1, [2000, 2000], { ctx: 1800, passed: false }),
    ...session("eval-A5-H1", "A5-H1", 1, [2000, 100], { ctx: 1800 }),
    ...session("sweep-A2-ctx2k", "A2", "ctx2000-r0", [2000, 2000], { ctx: 2000 }),
    ...session("sweep-A5-H1-ctx2k", "A5-H1", "ctx2000-r0", [2000, 100], { ctx: 2000 }),
    ...session("sweep-A2-ctx8k", "A2", "ctx8000-r0", [8000, 8000], { ctx: 8000 }),
    ...session("sweep-A5-H1-ctx8k", "A5-H1", "ctx8000-r0", [8000, 100], { ctx: 8000 }),
  ]);
  assert.equal(a.selection.ctxK, 2);
  assert.equal(a.selection.byMethod.A2.run.run_id, "eval-A2");
  assert.equal(a.a2Pass.passed, 0);
  assert.deepEqual(a.sweep.points.map((p) => [p.k, p.a2Run.run_id]), [[2, "sweep-A2-ctx2k"], [8, "sweep-A2-ctx8k"]]);
});
