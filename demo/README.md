# RoleKV demo

A static site with no build step and no dependencies. Three surfaces:

- **Homepage** (`index.html`), for judges and visitors. The pitch on one dark page: a live replay of one measured
  4-turn session (per-role LoRAs vs RoleKV, drawn to scale and lit as it is read), the session map that explains
  the shared cache and the H1 handoff, the measured token counts from the cost sweep as tiles and a bar chart, the
  methods compared, and links to the dashboard, the stage and the repository. It needs no event log: its numbers
  are the token counts from `bench/results/cost_sweep_4090.{md,jsonl}`, copied into [`js/home.js`](js/home.js)
  next to their source. It deliberately shows no timing (the 4090 TTFT sweep is being re-run) and no pass rate
  (the eval is pending).
- **Dashboard** (`dashboard.html`), for the team. It reads RoleKV event logs (JSONL,
  [CONTRACT.md §3](../docs/CONTRACT.md)) and opens with the result written as one sentence ("RoleKV re-read N%
  fewer tokens than per-role LoRAs"), followed by tokens per task, the number of paired tasks and both pass
  rates, so the saving never appears without its sample size and quality. Three tabs:
  1. **Session replay**: A2 (per-role LoRAs) and A5-H1 (RoleKV) on the same task. Each turn is a *session tape*
     drawn to scale: paper is the session so far, coral ink is what the turn had to read again, the cyan rule
     is what it reused from the cache, the violet underline is the H1 handoff re-read, and the outline is what
     the role wrote. Play, step, or show all; the tapes are focusable and their tooltips work from the keyboard.
     The default task is the one where both methods passed with the most turns. First-token time appears only
     for `machine == "4090"` rows.
  2. **Cost vs quality**: mean `prefill_tokens_total` per task against pass rate, one point per method, plus a
     table. Only the main eval runs (~2k context, the eval's real size) are used; sweep runs are excluded.
  3. **Context sweep**: prefill tokens saved per task (A2 − A5-H1, paired by `task_id`) against the median
     `ctx_tokens` of each context size, headed by a sentence stating the range of the share saved.

  Tab, task and context size are kept in the URL (`dashboard.html?tab=replay&task=86&ctx=3`).
- **Presentation stage** (`dashboard.html?present`), for the video. One full-viewport beat per part of the
  pitch: the cost of a role switch, how the shared cache works, the replay, cost vs quality, the sweep, the
  result, the end card. Arrow keys / space / Page Up and Down move between beats, R replays the current beat,
  and the corner controls fade when the pointer rests. The mock tag never hides. Add `&task=<id>` to choose the
  replay task.

Every number on the dashboard and the stage is computed from whatever log is loaded; nothing there is
hardcoded. The homepage is the one place with copied numbers, each next to its source.

Charts are hand-written SVG and canvas, so the pages work without network access (only the Google Fonts
request needs a connection, and it falls back to system fonts).

## Run it

```bash
cd demo
python -m http.server 8000      # then open http://localhost:8000
```

That opens the homepage; the dashboard is at `http://localhost:8000/dashboard.html`. Serve the folder over
HTTP. Opening the pages from `file://` will not load the ES modules.

## Load data

The dashboard picks its data in this order:

1. `?log=<path>` in the URL, e.g. `http://localhost:8000/dashboard.html?log=data/a5h1-1.5b-ctx3k.jsonl`
2. `demo/data/events.jsonl`, if it exists (the real default, so the 404 for it in the console is expected
   until real logs arrive)
3. `demo/mock/mock_events.jsonl`, the mock fixture, labelled "Mock data — not results"

You can also use **Load event log** or drag a `.jsonl` file onto the page. A loaded file **replaces** the
current data; it is never merged.

To show real results: concatenate the eval logs into `demo/data/events.jsonl` (e.g.
`cat eval/logs/*.jsonl > demo/data/events.jsonl`) or load them with the file picker.

## Mock data

```bash
python demo/mock/make_mock.py    # rewrites demo/mock/mock_events.jsonl (seed 0, deterministic)
```

Every mock `run_id` starts with `mock-`. When a log contains any non-mock run, all `mock-` records are dropped
and the banner says so, so mock numbers cannot leak into real results. In mock mode both surfaces show a
"Mock data — not results" tag, and the finding and the stage's result beat carry a note.

The mock token accounting follows the runtime rules in TRACKS.md (A2 re-prefills everything; one-cache
methods prefill only new tokens; A5-H1 recomputes header + body at each handoff). Pass rates and timings are
invented, and deliberately not all in A5's favour.

## Tests

```bash
cd demo
node --test tests/*.test.mjs     # Node 18+; no packages needed
```

The tests cover parsing and validation, pass rate, the headline ratio on paired tasks, sweep aggregation,
counting canonicalization exactly once, missing timings, mock/real isolation, default-context selection,
session-tape geometry, the featured replay task, the sweep summary, and the mock fixture's accounting invariants.

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
otherwise from the median `ctx_tokens` rounded to 0.5k. Runs whose `run_id` starts with `sweep-` are
context-sweep replays (`bench/cost_sweep.py`): they feed the Context sweep tab and never the main comparison or
pass rates. The main comparison uses the ~2k runs; if several runs
of a method qualify, the one with the most tasks wins (then 4090, then name), and the page says which.

**Accounting audit.** For every task, `prefill_tokens_total` must equal Σ over its turns of
`prefill_tokens + canon_tokens`, and `n_turns` must equal the number of turn records. Failures appear in
**Data validation**, and an "Accounting check failed" warning appears under the finding. If task totals demonstrably omit `canon_tokens`,
a separately labelled turn-derived ratio is shown next to (never instead of) the logged one.

**The homepage's numbers.** `js/home.js` holds the per-turn token counts of one sweep session (run
`sweep-A2-ctx1k` / `sweep-A5-H1-ctx1k`, task `ctx1000-r0`) and the per-session totals at the four context
sizes, copied from `bench/results/cost_sweep_4090.{jsonl,md}`. Token counts are deterministic, so the Mac run
has the same figures. When the sweep is re-run, update those constants from the new results.

## Assumptions about the contract (confirm with Lead)

- `prefill_tokens_total` includes `canon_tokens` (actual compute). The audit flags logs where it doesn't.
- `ttft_ms`, `turn_ms`, `kv_mb`, `text`, `gpu_s`, `peak_kv_mb` may be `null`; every other field is required.
- `task_id` may be an integer or a string; `machine` must be `4090`, `4070` or `mac`.
- A `run_id` has exactly one method and one machine; conflicting lines are rejected.
- `canon_tokens` on turn *i* is the previous turn's header and output, re-read at the start of turn *i*
  (CONTRACT §3); the replay draws it right after the cached prefix on turn *i*'s bar.

## Files

```text
index.html            the homepage
home.css              homepage styles: the same tape language on a dark stage
dashboard.html        page shell for the dashboard and the presentation stage
styles.css            the session-tape world: tokens, tape, dashboard and stage styles
js/home.js            homepage: the live session replay, the session map, the sweep chart; copied numbers with sources
js/metrics.js         pure data functions (parse, validate, select, compute, tape geometry); no DOM
js/data.js            event-log loading (?log=, real default, mock fallback, file picker)
js/ui.js              DOM helpers, formatting, plain method names, tooltip
js/tape.js            the session tape and the session map
js/charts.js          cost-vs-quality and context-sweep SVG charts
js/app.js             dashboard entry point and rendering
js/present.js         the ?present stage
mock/make_mock.py     deterministic mock generator
mock/mock_events.jsonl
tests/metrics.test.mjs
data/                 put the real events.jsonl here
```
