# RoleKV demo dashboard

A static page that reads RoleKV event logs (JSONL, [CONTRACT.md §3](../docs/CONTRACT.md)) and shows three views:

1. **Session Replay**: A2 and A5-H1 on the same task, one bar per turn, split into recomputed, reused and
   canonicalized tokens. TTFT appears only for `machine == "4090"` rows.
2. **Cost vs Quality**: mean `prefill_tokens_total` per task against pass rate, one point per method, plus a
   table. Only the main (~3k context) runs are used; sweep runs are excluded.
3. **Context Sweep**: prefill tokens saved per task (A2 − A5-H1, paired by `task_id`) against the median
   `ctx_tokens` of each context size.

The headline numbers (prefill ratio, % fewer tokens, pass rates) are computed from whatever log is loaded.
Nothing is hardcoded.

No build step, no dependencies. Charts are hand-written SVG, so the page works without network access
(only the Google Fonts request needs a connection, and it falls back to system fonts).

## Run it

```bash
cd demo
python -m http.server 8000      # then open http://localhost:8000
```

Serve the folder over HTTP. Opening `index.html` from `file://` will not load the ES modules.

## Load data

The page picks its data in this order:

1. `?log=<path>` in the URL, e.g. `http://localhost:8000/?log=data/a5h1-1.5b-ctx3k.jsonl`
2. `demo/data/events.jsonl`, if it exists (the real default, so the 404 for it in the console is expected
   until real logs arrive)
3. `demo/mock/mock_events.jsonl`, the mock fixture, with a visible "No real log" note

You can also use **Load event log…** or drag a `.jsonl` file onto the page. A loaded file **replaces** the
current data; it is never merged.

To show real results: concatenate the eval logs into `demo/data/events.jsonl` (e.g.
`cat eval/logs/*.jsonl > demo/data/events.jsonl`) or load them with the file picker.

## Mock data

```bash
python demo/mock/make_mock.py    # rewrites demo/mock/mock_events.jsonl (seed 0, deterministic)
```

Every mock `run_id` starts with `mock-`. When a log contains any non-mock run, all `mock-` records are dropped
and the banner says so, so mock numbers cannot leak into real results. In mock mode the page shows
"MOCK DATA — NOT EXPERIMENTAL RESULTS" and tags the headline `MOCK`.

The mock token accounting follows the runtime rules in TRACKS.md (A2 re-prefills everything; one-cache
methods prefill only new tokens; A5-H1 recomputes header + body at each handoff). Pass rates and timings are
invented, and deliberately not all in A5's favour.

## Tests

```bash
cd demo
node --test tests/*.test.mjs     # Node 18+; no packages needed
```

The tests cover parsing and validation, pass rate, the headline ratio on paired tasks, sweep aggregation,
counting canonicalization exactly once, missing timings, mock/real isolation, default-context selection, and
the mock fixture's accounting invariants.

## How the numbers are computed

All in [`js/metrics.js`](js/metrics.js), shared by the headline, table and charts.

| Metric | Definition |
|---|---|
| Prefill ratio | mean A2 `prefill_tokens_total` ÷ mean A5-H1 `prefill_tokens_total`, over `task_id`s present in both runs |
| % fewer prefill tokens | 100 × (1 − mean A5-H1 / mean A2), same task set |
| Saved / task | mean A2 − mean A5-H1, same task set |
| Pass rate | passed task records ÷ task records in the run (pass@1, greedy) |
| Sweep point | the same three numbers for the A2 and A5-H1 runs of one context size; x = median `ctx_tokens` |

**Run selection.** A run is one `run_id`. Its context size comes from `ctxNk` in the `run_id` if present,
otherwise from the median `ctx_tokens` rounded to 0.5k. The main comparison uses the ~3k runs; if several runs
of a method qualify, the one with the most tasks wins (then 4090, then name), and the page says which.

**Accounting audit.** For every task, `prefill_tokens_total` must equal Σ over its turns of
`prefill_tokens + canon_tokens`, and `n_turns` must equal the number of turn records. Failures appear in
**Data validation**, and the headline gets a `CHECK` tag. If task totals demonstrably omit `canon_tokens`,
a separately labelled turn-derived ratio is shown next to (never instead of) the logged one.

## Assumptions about the contract (confirm with Lead)

- `prefill_tokens_total` includes `canon_tokens` (actual compute). The audit flags logs where it doesn't.
- `ttft_ms`, `turn_ms`, `kv_mb`, `text`, `gpu_s`, `peak_kv_mb` may be `null`; every other field is required.
- `task_id` may be an integer or a string; `machine` must be `4090`, `4070` or `mac`.
- A `run_id` has exactly one method and one machine; conflicting lines are rejected.
- `canon_tokens` on turn *i* is work done after turn *i*; the replay draws it on that turn's bar.

## Files

```text
index.html            page shell
styles.css            styles; tokens match the Paper design
js/metrics.js         pure data functions (parse, validate, select, compute); no DOM
js/app.js             rendering and interaction
mock/make_mock.py     deterministic mock generator
mock/mock_events.jsonl
tests/metrics.test.mjs
data/                 put the real events.jsonl here
```
