# RoleKV demo

A static site with no build step. Three surfaces:

- **Homepage** (`index.html`), for judges and visitors. One dark screen, no scrolling on a laptop. Behind the
  words, the KV cache is drawn in three dimensions ([`js/cache3d.js`](js/cache3d.js), Three.js): two blocks of
  cells side by side, pointing at the viewer. The face in front is a 14 × 14 map of the token positions
  (8 tokens per cell, laid out row by row like a page, newest at the bottom), and behind it the model's 28
  layers line up into the depth. They play one measured 4-turn session: the per-role LoRA block is emptied
  and re-read at every turn by a coral cross-section that sweeps out of the depth towards the viewer; the
  RoleKV block stays cyan and only its handoff cells (violet) and new tokens light up. Cells not yet in the
  session are faint dots, so the block's final size is always visible. Drag to look around, hover a cell for
  its tokens and layer. The HUD along the bottom shows the same session as the dashboard's 2D tapes with the
  running token counts; both are driven from the same per-turn numbers, so they never disagree. Pause freezes
  the replay mid-sweep. It needs no event log: its numbers are the token counts from
  `bench/results/cost_sweep_4090.{md,jsonl}`, copied into [`js/home.js`](js/home.js) next to their source. It
  deliberately shows no timing (the 4090 TTFT sweep is being re-run) and no pass rate (the eval is pending).
  Without WebGL the page falls back to the HUD alone.
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

The dashboard and the stage have no dependencies: charts are hand-written SVG. The homepage's 3D scene uses
Three.js, vendored in [`vendor/three`](vendor/three/README.md) (MIT), so every page works without network
access (only the Google Fonts request needs a connection, and it falls back to system fonts).

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
index.html            the homepage: one screen, the 3D cache behind the words, the HUD along the bottom
home.css              homepage styles: the same tape language on a dark stage
dashboard.html        page shell for the dashboard and the presentation stage
styles.css            the session-tape world: tokens, tape, dashboard and stage styles
js/home.js            homepage: the replay controller (HUD tapes + 3D in lockstep), copied numbers with sources
js/cache3d.js         the KV cache in 3D: two instanced slabs of cells, the read wave, bloom, drag, hover
js/tapecanvas.js      the session tape on a <canvas> for the dark pages (homepage HUD, live demo)
vendor/three/         Three.js r186 (MIT), official build files only; see its README
js/metrics.js         pure data functions (parse, validate, select, compute, tape geometry); no DOM
js/data.js            event-log loading (?log=, real default, mock fallback, file picker)
js/ui.js              DOM helpers, formatting, plain method names, tooltip
js/tape.js            the session tape and the session map
js/charts.js          cost-vs-quality and context-sweep SVG charts
js/app.js             dashboard entry point and rendering
js/present.js         the ?present stage
mock/make_mock.py     deterministic mock generator
mock/mock_events.jsonl
tests/*.test.mjs      Node tests: metrics, charts, the live demo's state (live.test.mjs)
data/                 put the real events.jsonl here
live/                 the live demo (below): server.py, index.html, live.js, livemodel.js, live.css
```

## Live demo

`demo/live/` runs the comparison live instead of replaying logs: the same MBPP task on the same model, Per-role
LoRA (A2) in the left column and RoleKV (A5-H1) in the right. It is the homepage's dark world with live data:
in the middle, the homepage's two 3D cache blocks and HUD tapes, and under them the running totals (tokens
computed and the ratio, time to first token after a switch, time on GPU, an estimated cost). The homepage's nav
has a **Live** link to it.

- **Turns:** each turn's card (text, verdict, code, tokens computed and reused, time to first token, test result)
  appears in its column as soon as the server sends it.
- **The boards** play a turn once both methods have sent it, so the two caches fill side by side: A2's block is
  emptied and re-read at every turn, RoleKV's keeps its cache and computes only the new tokens, with the handoff
  re-read in violet. A block holds the session's context plus headroom and grows (with a reflow) if a session
  outgrows it. If WebGL is missing, the tapes and cards still work.
- **Without the server** (a static copy of `demo/`, or the file opened directly), the page says what to run
  instead of staying blank.

- **Server:** `demo/live/server.py`, stdlib only (`http.server` and Server-Sent Events) on top of the locked
  environment. It loads the model once with both methods' adapters and runs the two sessions interleaved turn
  by turn (A2's turn, then A5-H1's), in one worker thread, never in parallel ([CONTRACT.md §6](../docs/CONTRACT.md)).
  The protocol is `run_session`'s, and the tool output is `eval.testrun.run_mbpp_task` on all asserts, as in the
  evals. One run at a time: a second tab gets "busy", and Stop or closing the page ends the run after the turn
  in progress.
- **Page:** `demo/live/index.html`, `live.js` (DOM, the stream, the boards), `livemodel.js` (pure state, tested in
  Node) and `live.css` (home.css's tokens and fonts, sized for a projector); no build step. It reuses
  `../js/cache3d.js`, `../js/tapecanvas.js` and `metrics.tapeSpans`, so the homepage and the live page draw the
  cache the same way.
- **Tasks:** any of the 100 eval tasks, or a custom task (a description plus one or more `assert` lines; the
  first is shown to the model as the visible test). The context selector sets the utils.py size (30 functions
  is the eval's ~2k tokens; 60 and 120 show the saving grow).
- **Same transcript** (off by default): A2 replays RoleKV's generated tokens and test output, so both columns
  are exactly the same conversation and every difference comes from the cache. Off, each column generates its
  own replies with its own adapters, so the two sessions can differ in length.
- **Cost** is time on GPU × a $/GPU-hour price you type in (default $0.50, an assumption, not a measured bill).

**On the 4090 laptop** (WSL2, from the repo root, with the real adapters):

```bash
uv run python demo/live/server.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --adapters adapters/1.5b --device cuda
```

Then open http://localhost:8001/. The server listens on 127.0.0.1 only, because a custom task runs
model-written code with your asserts on this machine. If Windows cannot reach it from WSL2, use `--host 0.0.0.0`
on a trusted network only.

**On the Mac, for development**, with throwaway 0.5B adapters (about 20 s each on the Mac GPU; they only check
the plumbing, and the page labels them "smoke adapters"):

```bash
for spec in "coder alora sft_coder" "reviewer alora sft_reviewer" "coder lora sft_coder" "reviewer lora sft_reviewer"; do
  set -- $spec
  uv run python train/train_role_adapter.py --role $1 --kind $2 --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
    --data data/$3.jsonl --out tmp/live-adapters --max-steps 2 --grad-accum 1 --device mps
done
uv run python demo/live/server.py --model Qwen/Qwen2.5-Coder-0.5B-Instruct --adapters tmp/live-adapters
```

Other flags: `--device` (default mps if available, else cuda, else cpu), `--dtype bf16|fp16|fp32` (default
fp32 on cpu, else bf16), `--port` (default 8001), `--ctx-funcs` (default 30), `--max-coder-tokens` and
`--max-reviewer-tokens` (default 512 and 128, as in the evals). Startup loads MBPP and warms both methods up at
every context size, so the first run is not slowed by one-time costs. URL parameters `?task=443&ctx=60&mirror=1&rate=0.5`
preselect the controls.

`tests/test_live_demo.py` drives the session runner (`compare_sessions`) without HTTP: event order, counters
equal to the engine's, the same turns and token counts as `run_session`, the same-transcript mode, and stopping.
