// Pure data functions for the RoleKV demo. No DOM access, so Node tests can import this file.
// Schema source of truth: docs/CONTRACT.md §3 (event log).

export const METHODS = ["A0", "A1", "A2", "A3", "A5-H1", "A5-H2"];
export const MACHINES = ["4090", "4070", "mac"];
export const ROLES = ["coder", "reviewer"];
export const TIMING_MACHINE = "4090";
export const DEFAULT_CTX_K = 2; // the main eval: ctx_funcs=30 gives ≈1.5–1.9k tokens of shared context, as in training
export const MOCK_PREFIX = "mock-";

export const METHOD_META = {
  "A0": { label: "Base", desc: "Base model, no adapter", color: "#6B7685" },
  "A1": { label: "Shared LoRA", desc: "One standard LoRA shared between roles", color: "#7AA2F7" },
  "A2": { label: "Per-role LoRA", desc: "Separate adapters; full re-prefill", color: "#F47C6B" },
  "A3": { label: "Naive KV reuse", desc: "Reuses adapter-mismatched cache", color: "#C98A5E" },
  "A5-H1": { label: "aLoRA + canonicalize", desc: "Shared base cache; canonicalized handoff", color: "#A78BFA" },
  "A5-H2": { label: "aLoRA + keep KV", desc: "Shared cache with approximate handoff", color: "#D6B4FC" },
};

// Field specs. "count" = non-negative integer, "num" = finite non-negative number.
// Timing / memory fields may be null or absent (e.g. a run without CUDA timers); everything else is required.
const COMMON_FIELDS = {
  run_id: "string",
  machine: "machine",
  method: "method",
  task_id: "task_id",
};
const TURN_FIELDS = {
  ...COMMON_FIELDS,
  turn: "count",
  role: "role",
  prefill_tokens: "count",
  reused_tokens: "count",
  canon_tokens: "count",
  decode_tokens: "count",
};
const TURN_OPTIONAL = { ttft_ms: "num", turn_ms: "num", kv_mb: "num", text: "string" };
const TASK_FIELDS = {
  ...COMMON_FIELDS,
  ctx_tokens: "count",
  passed: "boolean",
  n_turns: "count",
  prefill_tokens_total: "count",
  decode_tokens_total: "count",
};
const TASK_OPTIONAL = { gpu_s: "num", peak_kv_mb: "num" };

function checkType(kind, v) {
  switch (kind) {
    case "string": return typeof v === "string" && v.length > 0;
    case "count": return Number.isInteger(v) && v >= 0;
    case "num": return typeof v === "number" && Number.isFinite(v) && v >= 0;
    case "boolean": return typeof v === "boolean";
    case "machine": return MACHINES.includes(v);
    case "method": return METHODS.includes(v);
    case "role": return ROLES.includes(v);
    case "task_id": return (Number.isInteger(v)) || (typeof v === "string" && v.length > 0);
    default: return false;
  }
}

function describe(kind) {
  return {
    count: "a non-negative integer",
    num: "a non-negative number",
    machine: `one of ${MACHINES.join(", ")}`,
    method: `one of ${METHODS.join(", ")}`,
    role: `one of ${ROLES.join(", ")}`,
    task_id: "an integer or non-empty string",
  }[kind] || `a ${kind}`;
}

/** Validate one parsed JSON value. Returns {ok:true, record} or {ok:false, reason}. */
export function validateRecord(obj) {
  if (obj === null || typeof obj !== "object" || Array.isArray(obj)) {
    return { ok: false, reason: "line is not a JSON object" };
  }
  let required, optional;
  if (obj.type === "turn") { required = TURN_FIELDS; optional = TURN_OPTIONAL; }
  else if (obj.type === "task") { required = TASK_FIELDS; optional = TASK_OPTIONAL; }
  else return { ok: false, reason: `unknown record type ${JSON.stringify(obj.type)}` };

  for (const [field, kind] of Object.entries(required)) {
    if (!(field in obj) || obj[field] === null) return { ok: false, reason: `missing field "${field}"` };
    if (!checkType(kind, obj[field])) {
      return { ok: false, reason: `field "${field}" must be ${describe(kind)} (got ${JSON.stringify(obj[field])})` };
    }
  }
  const record = { type: obj.type };
  for (const field of Object.keys(required)) record[field] = obj[field];
  for (const [field, kind] of Object.entries(optional)) {
    const v = obj[field];
    if (v === undefined || v === null) { record[field] = null; continue; }
    if (!checkType(kind, v)) {
      return { ok: false, reason: `field "${field}" must be ${describe(kind)} or null (got ${JSON.stringify(v)})` };
    }
    record[field] = v;
  }
  record.task_key = String(record.task_id);
  return { ok: true, record };
}

/**
 * Parse a JSONL event log. Malformed or invalid lines are rejected individually.
 * Also rejects duplicates and records whose method/machine contradict earlier records of the same run_id.
 */
export function parseJSONL(text) {
  const turns = [], tasks = [], errors = [];
  const seen = new Set();
  const runIdentity = new Map(); // run_id -> {method, machine, line}
  const lines = String(text ?? "").split(/\r?\n/);
  let nonEmpty = 0;
  lines.forEach((raw, i) => {
    const line = i + 1;
    if (raw.trim() === "") return;
    nonEmpty++;
    let obj;
    try { obj = JSON.parse(raw); }
    catch (e) { errors.push({ line, reason: `malformed JSON (${e.message})` }); return; }
    const v = validateRecord(obj);
    if (!v.ok) { errors.push({ line, reason: v.reason }); return; }
    const r = v.record;
    const id = runIdentity.get(r.run_id);
    if (id && (id.method !== r.method || id.machine !== r.machine)) {
      errors.push({ line, reason: `run_id "${r.run_id}" was ${id.method}/${id.machine} on line ${id.line}, here ${r.method}/${r.machine}` });
      return;
    }
    if (!id) runIdentity.set(r.run_id, { method: r.method, machine: r.machine, line });
    const key = r.type === "turn" ? `turn|${r.run_id}|${r.task_key}|${r.turn}` : `task|${r.run_id}|${r.task_key}`;
    if (seen.has(key)) { errors.push({ line, reason: `duplicate ${r.type} record (${key.slice(r.type.length + 1)})` }); return; }
    seen.add(key);
    r.line = line;
    (r.type === "turn" ? turns : tasks).push(r);
  });
  return { turns, tasks, errors, accepted: turns.length + tasks.length, rejected: errors.length, nonEmpty };
}

export const isMockRunId = (runId) => runId.startsWith(MOCK_PREFIX);

/**
 * Decide whether a parsed log is mock or real. A file with any real run is real, and its mock-prefixed
 * records are dropped so mock numbers can never enter real calculations.
 */
export function isolateDataset(parsed) {
  const real = (r) => !isMockRunId(r.run_id);
  const realCount = parsed.turns.filter(real).length + parsed.tasks.filter(real).length;
  const mockCount = parsed.accepted - realCount;
  const notes = [];
  if (parsed.accepted === 0) return { mode: "empty", turns: [], tasks: [], notes, droppedMock: 0 };
  if (realCount > 0) {
    if (mockCount > 0) notes.push(`${mockCount} mock-prefixed ${mockCount === 1 ? "record was" : "records were"} ignored because this log also contains real runs.`);
    return { mode: "real", turns: parsed.turns.filter(real), tasks: parsed.tasks.filter(real), notes, droppedMock: mockCount };
  }
  return { mode: "mock", turns: parsed.turns, tasks: parsed.tasks, notes, droppedMock: 0 };
}

export function median(values) {
  const xs = values.filter((v) => Number.isFinite(v)).sort((a, b) => a - b);
  if (xs.length === 0) return null;
  const m = Math.floor(xs.length / 2);
  return xs.length % 2 ? xs[m] : (xs[m - 1] + xs[m]) / 2;
}

export function mean(values) {
  if (values.length === 0) return null;
  return values.reduce((s, v) => s + v, 0) / values.length;
}

/**
 * Context-sweep runs (bench/cost_sweep.py, run_id "sweep-<method>-ctx<N>k") replay one forced conversation:
 * they measure cost only, so their `passed` says nothing about quality and they never enter the main comparison.
 */
export function isSweepRun(run) {
  return /^sweep-/i.test(run.run_id);
}

/** Nominal context size in k tokens from a run_id such as "a5h1-1.5b-ctx3k", or null. */
export function nominalCtxK(runId) {
  const m = /ctx(\d+(?:\.\d+)?)k/i.exec(runId);
  return m ? Number(m[1]) : null;
}

/** Group records into runs keyed by run_id. */
export function buildRuns(dataset) {
  const runs = new Map();
  const get = (r) => {
    if (!runs.has(r.run_id)) {
      runs.set(r.run_id, { run_id: r.run_id, method: r.method, machine: r.machine, tasks: new Map(), turns: new Map() });
    }
    return runs.get(r.run_id);
  };
  for (const t of dataset.tasks) get(t).tasks.set(t.task_key, t);
  for (const t of dataset.turns) {
    const run = get(t);
    if (!run.turns.has(t.task_key)) run.turns.set(t.task_key, []);
    run.turns.get(t.task_key).push(t);
  }
  for (const run of runs.values()) {
    for (const list of run.turns.values()) list.sort((a, b) => a.turn - b.turn);
    run.ctxMedian = median([...run.tasks.values()].map((t) => t.ctx_tokens));
    run.nominalK = nominalCtxK(run.run_id);
    // Prefer the documented run_id convention; otherwise round the median context to the nearest 0.5k.
    run.ctxK = run.nominalK ?? (run.ctxMedian === null ? null : Math.round(run.ctxMedian / 500) / 2);
    run.ctxSource = run.nominalK !== null ? "run_id" : "median ctx_tokens";
  }
  return runs;
}

export function formatCtxK(k) {
  return k === null || k === undefined ? "unknown ctx" : `~${k}k`;
}

/** Pick one run among candidates: most tasks, then the timing machine, then run_id order. */
function pickRun(candidates) {
  const sorted = [...candidates].sort((a, b) =>
    (b.tasks.size - a.tasks.size)
    || ((b.machine === TIMING_MACHINE) - (a.machine === TIMING_MACHINE))
    || a.run_id.localeCompare(b.run_id));
  return sorted[0] ?? null;
}

/**
 * Choose the main-evaluation run per method, excluding context-sweep runs.
 * Returns {ctxK, ambiguous, notes, byMethod: {method: {run, alternatives}}}.
 */
export function selectDefaultContextRuns(runs) {
  const all = [...runs.values()].filter((r) => r.tasks.size > 0 && r.ctxK !== null && !isSweepRun(r));
  const notes = [];
  let ctxK = DEFAULT_CTX_K;
  let ambiguous = false;
  if (!all.some((r) => r.ctxK === DEFAULT_CTX_K)) {
    // No run at the documented default: fall back to the context class holding the most task records.
    const totals = new Map();
    for (const r of all) totals.set(r.ctxK, (totals.get(r.ctxK) || 0) + r.tasks.size);
    const best = [...totals.entries()].sort((a, b) => b[1] - a[1] || a[0] - b[0])[0];
    if (best) {
      ctxK = best[0];
      ambiguous = true;
      notes.push(`No ~${DEFAULT_CTX_K}k runs found; using ${formatCtxK(ctxK)} runs (largest context class) as the main comparison.`);
    }
  }
  const byMethod = {};
  for (const method of METHODS) {
    const candidates = all.filter((r) => r.method === method && r.ctxK === ctxK);
    const run = pickRun(candidates);
    byMethod[method] = { run, alternatives: candidates.filter((r) => r !== run) };
    if (candidates.length > 1) {
      notes.push(`${method}: ${candidates.length} runs at ${formatCtxK(ctxK)}; using ${run.run_id} (most tasks).`);
    }
  }
  return { ctxK, ambiguous, notes, byMethod };
}

/** Pair the task records of two runs on task_id. Only tasks present in both runs are returned. */
export function pairTasks(runA, runB) {
  if (!runA || !runB) return [];
  const pairs = [];
  for (const [key, a] of runA.tasks) {
    const b = runB.tasks.get(key);
    if (b) pairs.push({ task_key: key, a, b });
  }
  return pairs.sort((x, y) => compareTaskKeys(x.task_key, y.task_key));
}

export function compareTaskKeys(a, b) {
  const na = Number(a), nb = Number(b);
  if (Number.isFinite(na) && Number.isFinite(nb)) return na - nb;
  return String(a).localeCompare(String(b));
}

/** Per-turn sums for one task: actual compute = prefill + canonicalization (each counted once). */
export function turnTotals(turns) {
  let prefill = 0, canon = 0, reused = 0, decode = 0;
  for (const t of turns) { prefill += t.prefill_tokens; canon += t.canon_tokens; reused += t.reused_tokens; decode += t.decode_tokens; }
  return { prefill, canon, reused, decode, compute: prefill + canon };
}

/**
 * Check a task record's prefill_tokens_total against its turn records.
 * Contract expectation: prefill_tokens_total = Σ(prefill_tokens + canon_tokens).
 * Status: ok | excludes_canon | mismatch | no_turns | turn_count
 */
export function auditTask(task, turns) {
  if (!turns || turns.length === 0) return { status: "no_turns", detail: "no turn records to verify against" };
  const tt = turnTotals(turns);
  if (turns.length !== task.n_turns) {
    return { status: "turn_count", detail: `n_turns=${task.n_turns} but ${turns.length} turn records`, totals: tt };
  }
  if (task.prefill_tokens_total === tt.compute) return { status: "ok", totals: tt };
  if (tt.canon > 0 && task.prefill_tokens_total === tt.prefill) {
    return { status: "excludes_canon", detail: `total ${task.prefill_tokens_total} omits ${tt.canon} canon tokens`, totals: tt };
  }
  return { status: "mismatch", detail: `total ${task.prefill_tokens_total} ≠ Σ(prefill+canon) ${tt.compute}`, totals: tt };
}

export function auditRun(run) {
  const counts = { ok: 0, excludes_canon: 0, mismatch: 0, no_turns: 0, turn_count: 0 };
  const issues = [];
  for (const [key, task] of run.tasks) {
    const a = auditTask(task, run.turns.get(key));
    counts[a.status]++;
    if (a.status !== "ok") issues.push({ task_key: key, ...a });
  }
  const orphanTurns = [...run.turns.keys()].filter((k) => !run.tasks.has(k)).length;
  return { counts, issues, orphanTurns, clean: issues.length === 0 };
}

/** Which prefill figure to use for a task: the authoritative task field, or the turn-derived compute. */
function taskPrefill(run, task, source) {
  if (source === "task") return task.prefill_tokens_total;
  const turns = run.turns.get(task.task_key);
  return turns && turns.length ? turnTotals(turns).compute : null;
}

/**
 * Headline comparison of two runs on their common task set.
 * ratio = mean A prefill / mean B prefill; pctSaved = 100 × (1 − meanB / meanA).
 */
export function headline(runA, runB, { source = "task" } = {}) {
  if (!runA || !runB) return { ok: false, reason: `no ${!runA ? "A2" : "A5-H1"} task records at this context` };
  const pairs = pairTasks(runA, runB);
  const xs = [], ys = [];
  for (const p of pairs) {
    const x = taskPrefill(runA, p.a, source), y = taskPrefill(runB, p.b, source);
    if (x === null || y === null) continue;
    xs.push(x); ys.push(y);
  }
  if (xs.length === 0) return { ok: false, reason: "no overlapping task_ids between the two runs", n: 0 };
  const meanA = mean(xs), meanB = mean(ys);
  if (!(meanB > 0) || !(meanA > 0)) return { ok: false, reason: "zero mean prefill; ratio undefined", n: xs.length, meanA, meanB };
  return {
    ok: true, n: xs.length, meanA, meanB,
    ratio: meanA / meanB,
    pctSaved: 100 * (1 - meanB / meanA),
    savedPerTask: meanA - meanB,
    source,
  };
}

export function passRate(run) {
  if (!run || run.tasks.size === 0) return { n: 0, passed: 0, rate: null };
  let passed = 0;
  for (const t of run.tasks.values()) if (t.passed) passed++;
  return { n: run.tasks.size, passed, rate: passed / run.tasks.size };
}

/** Mean GPU-seconds, only for runs on the timing machine with complete timings. */
export function meanGpuSeconds(run) {
  if (!run || run.machine !== TIMING_MACHINE) return { value: null, reason: run ? `${run.machine} run` : "no run" };
  const vals = [...run.tasks.values()].map((t) => t.gpu_s);
  if (vals.some((v) => v === null)) return { value: null, reason: "gpu_s missing" };
  return { value: mean(vals), reason: null };
}

/** Cost-vs-quality rows, one per method, from the default-context selection. */
export function costQualityRows(selection) {
  return METHODS.map((method) => {
    const run = selection.byMethod[method]?.run ?? null;
    if (!run) return { method, available: false };
    const pr = passRate(run);
    return {
      method, available: true, run,
      n: pr.n, passed: pr.passed, rate: pr.rate,
      meanPrefill: mean([...run.tasks.values()].map((t) => t.prefill_tokens_total)),
      gpu: meanGpuSeconds(run),
    };
  });
}

/**
 * Context sweep: for each context class that has both an A2 and an A5-H1 run, pair tasks and
 * report mean tokens saved and the fraction of A2 prefill saved.
 */
export function sweepSeries(runs) {
  const byK = new Map();
  // Prefer dedicated sweep runs: they replay the same conversation at every size, so points are comparable.
  const all = [...runs.values()];
  const pool = all.some(isSweepRun) ? all.filter(isSweepRun) : all;
  for (const run of pool) {
    if (run.ctxK === null || run.tasks.size === 0) continue;
    if (run.method !== "A2" && run.method !== "A5-H1") continue;
    if (!byK.has(run.ctxK)) byK.set(run.ctxK, { A2: [], "A5-H1": [] });
    byK.get(run.ctxK)[run.method].push(run);
  }
  const points = [], notes = [];
  for (const [k, group] of [...byK.entries()].sort((a, b) => a[0] - b[0])) {
    const a2 = pickRun(group.A2), a5 = pickRun(group["A5-H1"]);
    if (!a2 || !a5) { notes.push(`${formatCtxK(k)}: only ${a2 ? "A2" : "A5-H1"} present; not plotted.`); continue; }
    const pairs = pairTasks(a2, a5);
    if (pairs.length === 0) { notes.push(`${formatCtxK(k)}: no shared task_ids between ${a2.run_id} and ${a5.run_id}.`); continue; }
    const h = headline(a2, a5);
    const ctxMedian = median(pairs.map((p) => p.a.ctx_tokens));
    const ctxMismatch = pairs.filter((p) => p.a.ctx_tokens !== p.b.ctx_tokens).length;
    if (ctxMismatch) notes.push(`${formatCtxK(k)}: ${ctxMismatch} paired tasks have different ctx_tokens in A2 vs A5-H1.`);
    points.push({
      k, ctxMedian, n: h.n ?? 0, a2Run: a2, a5Run: a5,
      meanSaved: h.ok ? h.savedPerTask : null,
      fractionSaved: h.ok ? h.pctSaved / 100 : null,
    });
  }
  return { points, notes };
}

/** TTFT may only be shown for 4090 rows with a valid measurement. */
export function ttftEligible(turn) {
  return turn.machine === TIMING_MACHINE && typeof turn.ttft_ms === "number" && Number.isFinite(turn.ttft_ms);
}

/** Stacked-bar segments for one turn. Canonicalization is separate from prefill, never summed into it. */
export function turnSegments(turn) {
  return { recomputed: turn.prefill_tokens, reused: turn.reused_tokens, canon: turn.canon_tokens };
}

/**
 * Token positions of one turn on the session tape, in tokens from the start of the session.
 * The cache holds [0, reused). Under A5-H1 the turn first re-reads the previous turn's header and output with
 * base weights, [reused, reused + canon) (CONTRACT.md section 3: canon counts on the turn that does it), then
 * prefills [reused + canon, inputEnd); the role then writes [inputEnd, end). So every method's tape for a turn
 * has the same length.
 */
export function tapeSpans(turn) {
  const canonEnd = turn.reused_tokens + turn.canon_tokens;
  const inputEnd = canonEnd + turn.prefill_tokens;
  const end = inputEnd + turn.decode_tokens;
  return {
    reused: [0, turn.reused_tokens],
    canon: turn.canon_tokens ? [turn.reused_tokens, canonEnd] : null,
    recomputed: [canonEnd, inputEnd],
    written: [inputEnd, end],
    end,
  };
}

/** The token count that fills a tape's full width when these turns are drawn on one scale: the longest end. */
export function tapeScale(turns) {
  return Math.max(1, ...turns.map((t) => tapeSpans(t).end));
}

/** Task ids that have turn records in both runs (for the replay task picker). */
export function replayTaskKeys(runA, runB) {
  if (!runA || !runB) return [];
  return [...runA.turns.keys()].filter((k) => runB.turns.has(k)).sort(compareTaskKeys);
}

/**
 * The task the replay opens on: both methods passed if possible (a failing default reads as a broken
 * demo), then the most role switches (the contrast is clearest), then the lowest task id.
 */
export function featuredTask(runA, runB) {
  const keys = replayTaskKeys(runA, runB);
  const score = (k) => (runA.tasks.get(k)?.passed ? 1 : 0) + (runB.tasks.get(k)?.passed ? 1 : 0);
  const turns = (k) => Math.min(runA.turns.get(k).length, runB.turns.get(k).length);
  return keys.slice().sort((x, y) => score(y) - score(x) || turns(y) - turns(x) || compareTaskKeys(x, y))[0] ?? null;
}

/** Context sizes (k tokens) that have both an A2 and an A5-H1 run with turn records. */
export function replayContextSizes(runs) {
  const has = (method, k) => [...runs.values()].some((r) => r.method === method && r.ctxK === k && r.turns.size > 0);
  const sizes = new Set([...runs.values()].map((r) => r.ctxK).filter((k) => k !== null));
  return [...sizes].filter((k) => has("A2", k) && has("A5-H1", k)).sort((a, b) => a - b);
}

/** The pick-one-run rule, restricted to one method and context size, among runs with turn records. */
export function runFor(runs, method, ctxK) {
  return pickRun([...runs.values()].filter((r) => r.method === method && r.ctxK === ctxK && r.turns.size > 0));
}

/** One-line reading of the sweep: range of the share saved and of the absolute saving. */
export function sweepSummary(points) {
  const valid = points.filter((p) => p.meanSaved !== null && p.fractionSaved !== null);
  if (valid.length < 2) return null;
  const shares = valid.map((p) => p.fractionSaved);
  return {
    minShare: Math.min(...shares), maxShare: Math.max(...shares),
    firstSaved: valid[0].meanSaved, lastSaved: valid[valid.length - 1].meanSaved,
    firstK: valid[0].k, lastK: valid[valid.length - 1].k,
  };
}

/** Complete analysis of a dataset. The UI renders only from this object. */
export function analyze(dataset) {
  const runs = buildRuns(dataset);
  const selection = selectDefaultContextRuns(runs);
  const a2 = selection.byMethod.A2.run, a5 = selection.byMethod["A5-H1"].run;
  const head = headline(a2, a5);
  const audits = new Map([...runs.values()].map((r) => [r.run_id, auditRun(r)]));
  const headAudit = [a2, a5].filter(Boolean).map((r) => ({ run: r, audit: audits.get(r.run_id) }));
  const excludesCanon = headAudit.some((x) => x.audit.counts.excludes_canon > 0);
  const headlineTrusted = head.ok && headAudit.every((x) => x.audit.clean);
  return {
    runs, selection, headline: head, audits, headAudit, headlineTrusted,
    // Offered only when task totals demonstrably omit canonicalization; always labelled as derived.
    derivedHeadline: excludesCanon ? headline(a2, a5, { source: "turns" }) : null,
    a2Pass: passRate(a2), a5Pass: passRate(a5),
    rows: costQualityRows(selection),
    sweep: sweepSeries(runs),
  };
}
