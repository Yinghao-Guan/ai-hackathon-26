// Run: node --test demo/tests/
import { test } from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import {
  METHODS, newRun, applyEvent, methodOver, totals, compare, timesWord, percentWord, perMillion, capacityFor, binFor,
  sessionEnd, nextRound, backlog, playMs, fmtMs, fmtS, fmtUsd, endWords, readCaption,
} from "../live/livemodel.js";
import { tapeSpans } from "../js/metrics.js";

// One turn event as demo/live/server.py sends it.
function turn(method, i, o = {}) {
  const prefill = o.prefill ?? 100, canon = o.canon ?? 0;
  return {
    type: "turn", method, turn: i, role: i % 2 ? "reviewer" : "coder", text: o.text ?? "x",
    prefill_tokens: prefill, reused_tokens: o.reused ?? 0, canon_tokens: canon, computed_tokens: prefill + canon,
    decode_tokens: o.decode ?? 10, ttft_ms: o.ttft ?? 50, turn_ms: o.ms ?? 1000, verdict: o.verdict ?? null,
  };
}
// The 4-turn sweep session from the homepage (bench/results/cost_sweep_4090.jsonl, ctx1k), as live events.
const A2 = [turn("A2", 0, { prefill: 1227, decode: 80, ttft: 90 }), turn("A2", 1, { prefill: 1393, decode: 30, ttft: 94, verdict: "REVISE" }),
  turn("A2", 2, { prefill: 1434, decode: 83, ttft: 96 }), turn("A2", 3, { prefill: 1545, decode: 4, ttft: 98, verdict: "APPROVE" })];
const A5 = [turn("A5-H1", 0, { prefill: 1227, decode: 80, ttft: 88 }), turn("A5-H1", 1, { prefill: 87, reused: 1217, canon: 89, decode: 30, ttft: 50, verdict: "REVISE" }),
  turn("A5-H1", 2, { prefill: 12, reused: 1382, canon: 40, decode: 83, ttft: 49 }), turn("A5-H1", 3, { prefill: 29, reused: 1424, canon: 92, decode: 4, ttft: 51, verdict: "APPROVE" })];

function runWith(a2, a5, extra = (run) => run) {
  const run = newRun();
  applyEvent(run, "start", { type: "start", context_tokens: 1217 });
  for (let i = 0; i < Math.max(a2.length, a5.length); i++) {
    if (a2[i]) applyEvent(run, "turn", a2[i]);
    if (a5[i]) applyEvent(run, "turn", a5[i]);
  }
  return extra(run);
}

test("applyEvent folds the stream and rejects turns out of order", () => {
  const run = newRun();
  applyEvent(run, "start", { context_tokens: 1217, mirror: false });
  applyEvent(run, "turn_start", { method: "A2", turn: 0, role: "coder" });
  assert.equal(run.next.method, "A2");
  applyEvent(run, "turn", A2[0]);
  assert.equal(run.next, null);
  assert.equal(run.turns["A2"].length, 1);
  assert.throws(() => applyEvent(run, "turn", A2[0]), /arrived after 1 turns/);
  assert.throws(() => applyEvent(run, "turn", A5[1]), /arrived after 0 turns/);
  assert.throws(() => applyEvent(run, "turn", turn("A9", 0)), /unknown method/);
  applyEvent(run, "final", { method: "A2" });
  assert.ok(methodOver(run, "A2") && !methodOver(run, "A5-H1"));
  applyEvent(run, "done", { wall_s: 4.2 });
  assert.equal(run.ended, "done");
  assert.equal(run.wall, 4.2);
  assert.ok(methodOver(run, "A5-H1"));
  const failed = applyEvent(newRun(), "error", { message: "boom" });
  assert.equal(failed.ended, "error");
  assert.equal(failed.error, "boom");
});

test("totals count prefill + canon, average TTFT over role switches only, and sum turn times", () => {
  const t = totals(A5);
  assert.equal(t.computed, 1227 + 87 + 89 + 12 + 40 + 29 + 92);
  assert.equal(t.computed, 1576); // the homepage's figure for this session
  assert.equal(t.ttft, 50); // mean of 50, 49, 51; turn 0 (88 ms) is not a switch
  assert.equal(t.gpu, 4);
  assert.equal(totals(A2).computed, 5599);
  assert.deepEqual(totals([]), { n: 0, computed: null, ttft: null, gpu: null });
  assert.equal(totals(A5, 1).ttft, null); // no switch yet
});

test("compare counts complete rounds and, once a session is over, all of it", () => {
  const all = compare(runWith(A2, A5));
  assert.equal(all.n, 4);
  assert.equal(all.tokens.text, "3.6× fewer");
  assert.equal(all.tokens.kind, "win");
  assert.equal(all.ttft.text, "1.9× faster");
  assert.equal(all.gpu.text, "about the same");
  // A2 runs first in each round: while RoleKV's turn 1 is computing, only round 0 is compared.
  const mid = compare(runWith(A2.slice(0, 2), A5.slice(0, 1)));
  assert.equal(mid.n, 1);
  assert.equal(mid.tokens.text, "1.0× fewer");
  assert.equal(mid.ttft, null);
  assert.equal(compare(runWith(A2, A5), 2).n, 2);
  assert.equal(compare(newRun()).tokens, null);
  // RoleKV approved after 2 turns, A2 went on to 4 (the 120-function run on the 4090): before RoleKV's final
  // arrives the rounds stay aligned; after it, A2's later turns count too, so the ratio matches the totals.
  const shortA5 = runWith(A2.slice(0, 3), A5.slice(0, 2));
  assert.equal(compare(shortA5).n, 2);
  applyEvent(shortA5, "final", { method: "A5-H1" });
  applyEvent(shortA5, "turn", A2[3]);
  const whole = compare(shortA5);
  assert.equal(whole.n, 4);
  assert.equal(whole.a2.computed, 5599);
  assert.equal(whole.a5.computed, 1227 + 87 + 89);
  assert.equal(whole.tokens.text, "4.0× fewer");
  assert.equal(compare(shortA5, 2).tokens.text, `${((1227 + 1393) / (1227 + 87 + 89)).toFixed(1)}× fewer`); // 2 rounds played
  // The other way round: A2 approved after 2 turns, RoleKV took 4.
  const shortA2 = runWith(A2.slice(0, 2), A5, (run) => applyEvent(run, "final", { method: "A2" }));
  assert.equal(compare(shortA2).n, 4);
  assert.equal(compare(shortA2).tokens.text, `${((1227 + 1393) / 1576).toFixed(1)}× fewer`);
});

test("words for ratios and percentages say which way they go", () => {
  assert.deepEqual(timesWord(583, 50, "faster", "slower").text, "11.7× faster");
  assert.equal(timesWord(50, 100, "faster", "slower").text, "2.0× slower");
  assert.equal(timesWord(50, 100, "faster", "slower").kind, "loss");
  assert.equal(timesWord(null, 1, "a", "b"), null);
  assert.equal(timesWord(0, 1, "a", "b"), null);
  assert.equal(percentWord(8.4, 7.32).text, "13% less");
  assert.equal(percentWord(6.56, 6.84).text, "4% more");
  assert.equal(percentWord(6.56, 6.56).text, "about the same");
  assert.equal(perMillion(7.2, 0.5), 1000);
  assert.equal(perMillion(null, 0.5), null);
});

test("the 3D cache grows in 256-token steps with 20% headroom and keeps layers at most 400 cells", () => {
  assert.equal(capacityFor(2297), 2816);
  assert.equal(capacityFor(2297, 2816), 2816); // never shrinks
  assert.equal(capacityFor(3000, 2816), 3840);
  assert.equal(capacityFor(10), 256);
  assert.equal(binFor(2816), 8);
  assert.equal(binFor(9216), 24);
  assert.ok(Math.ceil(9216 / binFor(9216)) <= 400);
  const run = runWith(A2, A5);
  assert.equal(sessionEnd(run), tapeSpans(A5[3]).end);
  assert.equal(sessionEnd(run, 1), 1227 + 80);
  assert.equal(sessionEnd(newRun()), 0);
});

test("nextRound plays both boards together and waits for the slower method", () => {
  const run = newRun();
  assert.equal(nextRound(run, -1), null);
  applyEvent(run, "turn", A2[0]);
  assert.equal(nextRound(run, -1), null); // RoleKV's turn 0 is still computing
  applyEvent(run, "turn", A5[0]);
  assert.equal(nextRound(run, -1), 0);
  assert.equal(nextRound(run, 0), null);
  applyEvent(run, "turn", A2[1]);
  applyEvent(run, "turn", A5[1]);
  applyEvent(run, "turn", A2[2]);
  assert.equal(backlog(run, -1), 2);
  assert.equal(backlog(run, 0), 1);
  // Sessions of different lengths: A2 approved after 2 turns, RoleKV goes on alone once A2 is over.
  const diverged = runWith(A2.slice(0, 2), A5);
  assert.equal(nextRound(diverged, 1), null);
  applyEvent(diverged, "final", { method: "A2" });
  assert.equal(nextRound(diverged, 1), 2);
  assert.equal(nextRound(diverged, 3), null);
  // A stopped run plays what it has.
  const stopped = runWith(A2.slice(0, 1), []);
  assert.equal(nextRound(stopped, -1), null);
  applyEvent(stopped, "cancelled", { wall_s: 1 });
  assert.equal(nextRound(stopped, -1), 0);
});

test("playing a turn takes longer the more it computes, within bounds", () => {
  assert.deepEqual(playMs(A2[1]), { read: 1323, write: 270 });
  assert.deepEqual(playMs(A5[1]), { read: 260, write: 270 });
  assert.equal(playMs(turn("A2", 0, { prefill: 9000, decode: 500 })).read, 1700);
  assert.equal(playMs(turn("A2", 0, { prefill: 9000, decode: 500 })).write, 800);
  assert.deepEqual(playMs(A2[1], 2), { read: 662, write: 135 });
});

test("captions and formats", () => {
  const run = runWith(A2, A5);
  assert.equal(readCaption(run, 1), "Turn 2 · the Reviewer reads · computed: Per-role LoRA 1,393 · RoleKV 176");
  assert.equal(readCaption(runWith(A2.slice(0, 2), A5), 2), "Turn 3 · the Coder reads · computed: RoleKV 52");
  assert.equal(endWords({ end: "APPROVE", n_turns: 4, verdicts: ["REVISE", "APPROVE"] }), "The Reviewer approved after 4 turns.");
  assert.match(endWords({ end: "INVALID", n_turns: 2, verdicts: ["INVALID"] }), /neither APPROVE nor REVISE/);
  assert.equal(endWords({ end: "LAST_REVISION", n_turns: 5, verdicts: ["REVISE", "REVISE"] }), "Final revision after 2 REVISE verdicts (5 turns).");
  assert.equal(fmtMs(583.4), "583 ms");
  assert.equal(fmtMs(1390), "1.39 s");
  assert.equal(fmtMs(null), "—");
  assert.equal(fmtS(4.031), "4.03 s");
  assert.equal(fmtS(12.34), "12.3 s");
  assert.equal(fmtUsd(559.2), "$559");
  assert.equal(fmtUsd(1234.5), "$1,235");
  assert.equal(fmtUsd(3.456), "$3.46");
  assert.equal(fmtUsd(0.00094), "$0.00094");
  assert.deepEqual(METHODS, ["A2", "A5-H1"]);
});

test("the live page stands alone in the dark theme and explains itself without a server", () => {
  const page = readFileSync(new URL("../live/index.html", import.meta.url), "utf8");
  const css = readFileSync(new URL("../live/live.css", import.meta.url), "utf8");
  assert.doesNotMatch(page, /styles\.css/);
  assert.match(page, /href="live\.css"/);
  assert.match(css, /--bg: #0B0E12/);
  // The offline panel is in the HTML, so it shows even when the scripts cannot run (file://).
  assert.match(page, /id="offline"/);
  assert.match(page, /The live demo needs its model server/);
  assert.match(page, /uv run python demo\/live\/server\.py/);
  assert.match(page, /"three": "\.\.\/vendor\/three\/three\.module\.js"/);
  const home = readFileSync(new URL("../index.html", import.meta.url), "utf8");
  assert.match(home, /<a href="live\/">Live<\/a>/);
});
