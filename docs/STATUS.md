# Status

> Read this first every time you pull. Each owner keeps their own row in the table current. Changes that
> affect other tracks go under "Changes", newest first.

## Tracks

| Track | Owner | Status | Notes |
|---|---|---|---|
| Environment | Lead | Done | `uv sync` installs everything, pinned by `uv.lock` |
| Prompt contract (`rolekv/roles.py`) | Lead | Done, frozen | |
| Runtime core | Lead | Done on Mac CPU | 28 tests pass; CUDA/bf16 check on the 4090 pending |
| Cost sweep (`bench/cost_sweep.py`) | Lead | Done: final 4090 numbers with bf16 adapters (`79e1f6a`) | Results in `bench/results/` on `peter`; see Changes below |
| 4090 machine | Lead | Environment ready (Windows + WSL2); runtime tests and cost sweep pending | All training and timing in WSL2. [TRACKS.md › 4090 machine](TRACKS.md#4090-machine--lead-rtx-4090-laptop) |
| Training script | Member A | Done; merged into `peter` | Reviewed by Lead; smoke-trained on the Mac with 0.5B; all gates pass |
| Data | Member B → Lead | 280 trajectories in `data/` (B done) | Checked by Lead against CONTRACT §2: all rows valid, max 2,829 tokens. Reviewer labels are 79% APPROVE (see Changes) |
| Eval driver | Member B → Lead | Written; connected to the runtime; real-data dry run passes (B done) | `--runner rolekv.eval_runner:make_runner`, see [CONTRACT.md §6](CONTRACT.md#6-runtime-api-runtime--eval) |
| Demo | Member C → Lead | Dashboard merged (C done); homepage added; real-data check pending | Homepage at `demo/index.html`, dashboard at `demo/dashboard.html`. Runs on mock logs; already counts compute as prefill + canon. Real 4090 sample: 4090 machine step 4 |

## Changes

### 2026-10-18: demo homepage in 3D; the dashboard URL moved; `ui` and `report` merged into `main`

- `demo/index.html` is now a one-screen homepage for judges: the KV cache rendered in 3D (`demo/js/cache3d.js`,
  two blocks of cells side by side pointing at the viewer, a 14 × 14 map of the token positions in front and
  28 layers deep) replaying one measured 4-turn session, with the 2D tapes and running token counts in the HUD.
  This adds **Three.js r186 (MIT), vendored** in `demo/vendor/three` (~2 MB, provenance in its README) so the
  demo still works offline. The dashboard and the stage remain dependency-free.
- The dashboard moved to **`demo/dashboard.html`**; `?present`, `?log=`, `?tab=` work as before on that URL,
  and the stage's exit link and the dashboard's brand link were updated. Bookmarks and the recording
  checklist should use `dashboard.html`.
- The homepage shows token counts only, copied from `bench/results/cost_sweep_4090.md` with the source next
  to each constant (`demo/js/home.js`); the re-run sweep (entry below) left them unchanged. No TTFT and no
  pass rate on it yet.
- Merged into `main`: `ui` (which carries `peter` up to `139a373`: the hook bypass, bf16 adapters, the cost
  sweeps and the SFT system-segment fix, so `main` now has the fixed SFT data too) and `report` (the
  two-column research report under `report/`).

### 2026-10-18: eval protocol for the 100-task runs

Member A checked the eval pipeline with a 3-task A0 pilot on the 4070, and Lead confirmed the inputs:
- **Eval task ids:** 100 ids, first `[443, 208, 399]`, sha256 `f53d1618…b79c`. Identical on the 4070 and the Mac.
- **MBPP:** revision `4bb6404f…`. **Qwen2.5-Coder-1.5B-Instruct:** snapshot `2e1fd397…`. The same on the 4070
  and the Mac. The 4090's is recorded as `model_revision` in each adapter's `train_run.json`.
- **Eval commit:** `peter` at `114918e`. It adds `canon_tokens_total` to the task rows and changes nothing that
  the model sees, so the pilot at `89af6b9` is still valid.

**Where each method runs.** Pass rates are compared only between runs on Linux (CONTRACT.md §6). Tool output
contains the OS's temp path, and native Windows changes that text.

| Method | Machine | Why |
|---|---|---|
| A2, A5-H1 | 4090, WSL2 | the headline comparison: quality and timing |
| A1, A3, A5-H2 | 4090, WSL2 | they need the trained adapters, which live on the 4090 |
| A0 | 4070 in WSL2 if that is quick to set up, otherwise the 4090 | it needs no adapters |

**Run ids:** `eval-1.5b-<machine>-<YYYYMMDDHHMM>`, which the driver logs as `…-<method>`. The 4070 runs are
labelled `machine` `4070`, and their timings are never compared.

### 2026-10-18: final 4090 cost numbers (bf16 adapters for every method)

| Context | Tokens computed: A2 → A5-H1 | Switch TTFT: A2 → A5-H1 | GPU s / session: A2 → A5-H1 |
|---|---|---|---|
| 1k | 5,599 → 1,576 (3.6× less) | 94 → 50 ms (1.9×) | 6.56 → 6.84 s |
| 2k | 9,595 → 2,575 (3.7× less) | 155 → 48 ms (3.2×) | 6.82 → 6.89 s |
| 4k | 17,575 → 4,570 (3.8× less) | 279 → 47 ms (5.9×) | 7.19 → 7.01 s |
| 8k | 33,651 → 8,589 (3.9× less) | 583 → 50 ms (11.7×) | 8.40 → 7.32 s |

- **These replace every earlier 4090 timing.** They are now in `README.md`, `docs/PITCH.md` and
  `docs/DEVPOST_DRAFT.md`. Source: `bench/results/cost_sweep_4090.*` on `peter`, commit `79e1f6a`.
- **Decode:** A5-H1 decodes 8% slower per token (34.3 vs 31.7 ms). That is the Python overhead of PEFT's aLoRA
  path, and it counts against us. Whole-session time is therefore 4% worse at 1k and 13% better at 8k.
- **What to claim:** ~4× less prefill compute, and a flat ~50 ms switch TTFT against a linearly growing one.
  Don't claim whole-session speedups at short contexts.

### 2026-10-17: training script merged into `peter`; ready to train on the 4090

**Review.** Member A's `train/train_role_adapter.py` (branch `member-a/training-adapters`) matches the
contract:
- **Data:** it builds rows with `roles.tokenize_sft_example`, rejects any row whose system segment or role
  header differs from `rolekv/roles.py`, and checks per row that PEFT activates the aLoRA exactly at the role
  header.
- **Output:** it saves `{out}/{kind}_{role}/`, the names the runtime expects.
- **Oversampling:** `--oversample-revise N` is implemented as agreed.
- **Gates after training:** the parameters changed; an aLoRA leaves the logits before its header unchanged;
  the adapter reloads exactly.

**Smoke test on the Mac (0.5B, 1–2 updates per adapter).**
- **Training:** all 4 adapters for A2 and A5-H1 passed every gate. All 280 reviewer rows were used and none
  were rejected. aLoRA `pre_header_diff` was 0.0 and `reload_err` was 0.
- **Eval:** `eval/run_eval.py` with `rolekv.eval_runner:make_runner` loaded the adapters and ran 2 MBPP
  tasks per method. Even after 1–2 updates, the trained Reviewer answers `APPROVE`/`REVISE` in format; the
  base model's does not.

**Train only on `peter`.** `main` still has the SFT data with the old system segment, and the script rejects
those rows. Train on `peter`, which has the fixed data.

**4090 training commands** (WSL2; one at a time; `adapters/` is gitignored):

```bash
M=Qwen/Qwen2.5-Coder-1.5B-Instruct; OUT=adapters/1.5b
uv run python train/train_role_adapter.py --role coder    --kind alora --model $M --data data/sft_coder.jsonl --out $OUT
uv run python train/train_role_adapter.py --role reviewer --kind alora --model $M --data data/sft_reviewer.jsonl --out $OUT --oversample-revise 3
uv run python train/train_role_adapter.py --role coder    --kind lora  --model $M --data data/sft_coder.jsonl --out $OUT
uv run python train/train_role_adapter.py --role reviewer --kind lora  --model $M --data data/sft_reviewer.jsonl --out $OUT --oversample-revise 3
uv run python train/train_role_adapter.py --role all      --kind lora  --model $M --data data/sft_coder.jsonl data/sft_reviewer.jsonl --out $OUT --oversample-revise 3
```

### 2026-10-17: adapters run in bf16; the dashboard reads real logs; doc numbers filled

- **Fairness fix.** PEFT upcasts adapter weights to fp32 by default. Plain LoRA adapts every token, so the
  upcast slowed it much more than aLoRA, which adapts the role header only: a 4k-token prefill on the Mac was
  +39% over base for LoRA and +7% for aLoRA. With bf16 adapters it is +16% and +2%.
  - The runtime now keeps adapters in the model dtype for every method (on `peter`).
  - The 4090 TTFT numbers in the entry below inflated A2's TTFT. **The 4090 sweep is being re-run**, and those
    TTFT figures must not be used.
  - Token counts are unaffected.
- **Dashboard + real logs** (on `peter`). Loading the real sweep log showed three bugs, now fixed:
  - Sweep runs were merged into one fake "~3k" run that showed "20 of 20 passed". Sweep runs (`sweep-…` run
    ids) are now excluded from quality numbers.
  - The default context is now ~2k, the real eval size.
  - The replay drew the H1 re-read in the old place, so A5-H1 bars were shorter than A2's.
- **Eval run ids:** `eval/run_eval.py` now logs each method as `<run-id>-<method>`, because the dashboard
  rejects a run id that has two methods.
- **Docs:** `README.md`, `docs/PITCH.md` and `docs/DEVPOST_DRAFT.md` now carry the measured token counts and
  the real challenges. 4090 TTFT/GPU-s and every quality number are still `[[placeholders]]`.

### 2026-10-17: 4090 cost numbers after the decode fix (TTFT superseded: fp32 adapters, see above)

The setup is the same as before: WSL2, 1.5B in bf16, untrained adapters, the forced 4-turn session, and
5 repeats. Token counts are identical to the Mac run.

| Context | Tokens computed: A2 → A5-H1 | Switch TTFT: A2 → A5-H1 | GPU s / session: A2 → A5-H1 |
|---|---|---|---|
| 1k | 5,599 → 1,576 (3.6× less) | 124 → 52 ms (2.4×) | 7.81 → 7.76 s |
| 2k | 9,595 → 2,575 (3.7× less) | 214 → 50 ms (4.3×) | 8.11 → 7.64 s |
| 4k | 17,575 → 4,570 (3.8× less) | 388 → 51 ms (7.6×) | 8.81 → 7.88 s (11% less) |
| 8k | 33,651 → 8,589 (3.9× less) | 798 → 52 ms (15.5×) | 10.51 → 8.22 s (22% less) |

- **Decode speed is now even:** A5-H1 decodes within 1–4% of A2 per token (37.8–39.2 vs 37.4–38.1 ms).
  The fix worked.
- **What to claim, for the pitch, README and Devpost:**
  1. ~4× less prefill compute per 4-turn session. The saving approaches the number of turns.
  2. Switch TTFT stays flat at ~51 ms, while A2's grows linearly with context: 2.4× faster at 1k, 15.5× at 8k.
  3. Whole-session time is secondary. Decoding ~230 tokens dominates a session, so it is the same at 1k and
     up to 22% less at 8k.
- **Still no memory claim against A2.**
- **Sources:** `bench/results/cost_sweep_4090.{jsonl,md}` on `peter` (the re-run commit from the 4090), and
  `bench/results/cost_sweep_mac.*` for the token counts.

### 2026-10-17: SFT data now uses the runtime's system segment; REVISE oversampling is a training flag

**System segment fix.** Member A noticed that every SFT row started with `You are a careful Python coding
assistant.`, while the runtime starts every eval session with `roles.system_segment()`. Trained that way, the
adapters would have seen a different context at eval than in training.
- **Fixed on `peter` (commit `f43e5e0`), with no API spend.**
  - `python -m train.build_dataset --reexport` rebuilt the SFT files from `data/trajectories.jsonl`.
  - Only the system segment changed (checked row by row), and `data/README.md` has the new hashes.
- `build_dataset.py` now always writes `roles.system_segment()`. `--system-prompt` reaches only the teacher.
- `tests/test_data_contract.py` fails if the committed data ever drifts from the contract again.
- **Train only on the data from `peter`.** The data on `main` still has the old system segment until `peter` is
  merged.

**REVISE oversampling (decision).** It is a training-script option: `--oversample-revise N`, default 1. With
N > 1, each reviewer row whose target starts with `REVISE` is used N times. It applies to `--role reviewer` and
to the reviewer rows of `--role all`. The 4090 runs use `--oversample-revise 3`. The data files stay as
generated.

### 2026-10-17: Members B and C are done; Lead takes over what is left

Members B and C have finished their tracks. Lead now owns everything still open in them:

| From | Open item | Plan |
|---|---|---|
| B | Reviewer labels are 79% `APPROVE` | Oversample the `REVISE` rows at training time; no new API spend |
| B | Running the evals | 4090, after training (it was already Lead's) |
| C | Dashboard has only seen mock data | Load the real 4090 sweep and eval logs into `demo/data/events.jsonl` and check every view |
| C | Number placeholders in `README.md`, `docs/PITCH.md`, `docs/DEVPOST_DRAFT.md` | Cost numbers now; pass rates after the evals |
| C | Video and pitch rehearsal | Lead records and presents |

**Eval dry run with real data.** `eval/run_eval.py` with `rolekv.eval_runner:make_runner`, A0, 3 MBPP tasks,
0.5B on the Mac. The whole pipeline works end to end.
- **Context length:** the shared context with `ctx_funcs=30` is 1.5k–1.9k tokens, not ~3k. Training data uses
  the same builder, so evals keep it.
- **A0 Reviewer:** the base model's Reviewer copies the code instead of answering `APPROVE` or `REVISE`, so
  `INVALID` ends A0 sessions after 2 turns. That is the expected quality floor. Watch the `INVALID` rate per
  method in the evals.

### 2026-10-17: first 4090 numbers, and a decode fix (superseded by the re-run above)

The 4090 ran the same cost sweep: WSL2, 1.5B in bf16, untrained adapters, the forced 4-turn session, and
5 repeats. Token counts are identical to the Mac run turn by turn, as expected.

| Context | Switch TTFT: A2 → A5-H1 | GPU s / session: A2 → A5-H1 |
|---|---|---|
| 1k | 124 → 54 ms (2.3×) | 7.4 → 8.1 s |
| 2k | 214 → 53 ms (4.0×) | 7.8 → 8.2 s |
| 4k | 386 → 54 ms (7.2×) | 8.4 → 8.3 s |
| 8k | 796 → 56 ms (14×) | 10.1 → 8.6 s |

- **TTFT:** A5-H1's switch TTFT stays flat at ~54 ms, while A2's grows linearly with context.
- **Session time:** A5-H1 decoded 15% slower per token than A2 in this run (41.0 vs 35.7 ms). So at 1k–2k
  context its whole session was slower than A2's, even though every switch was faster.
- **Cause and fix:** `PeftModel.forward` registers a hook on every LoRA layer at every aLoRA call, which costs
  ~5 ms per token on the 4090.
  - The runtime on `peter` now bypasses it: one persistent hook per layer, installed once.
  - On the Mac, aLoRA and LoRA now decode equally fast (28.7 vs 28.2 ms per token).
- **Next:** the 4090 sweep is being re-run with the fix, and these numbers will be replaced.
  - **Member C:** use the TTFT column now (it will only get a few ms faster), but keep the session-time column
    out of the pitch until the re-run.
- The result files are in `bench/results/` on the `peter` branch until it is merged into `main`.

### 2026-10-17: 4090 is ready; train and time it in WSL2

The 4090 laptop (RTX 4090 Laptop GPU, 16 GB) has the locked environment both on Windows and in WSL2. Measured
with 1.5B in bf16:

| | Windows | WSL2 |
|---|---|---|
| Prefill 8k tokens | 478 ms | 443 ms |
| Decode at 8k context | 47 tok/s | 66 tok/s |
| Training step, 3,072 tokens: LoRA / aLoRA (r=32) | 0.89 / 1.11 s | 0.82 / 1.05 s |
| 4,096-token step | fits | fits (10.6 GiB allocated) |

- **Decision: all training and timed evals on the 4090 run in WSL2.** WSL2 is faster everywhere and has the
  flash attention kernel. Never mix Windows and WSL2 timings. The `machine` label stays `4090`.
- **No GPU rental needed.** The step times above, applied to our data (338 coder and 280 reviewer rows,
  median ~1.9k tokens), put all five adapters at roughly 1 hour of training. This is an estimate; the real
  runs will show.
- **Member A:** aLoRA training steps are ~25% slower than LoRA. The cause is the same boolean-mask issue that
  `rolekv/fast_alora.py` fixes for inference. It is not worth fixing before the first training run.

### 2026-10-17: data checked, eval driver connected to the runtime

**Data check (Lead, for Member A and B).** Lead ran every row of `data/sft_coder.jsonl` (338 rows) and
`data/sft_reviewer.jsonl` (280 rows) through `roles.tokenize_sft_example`.
- **Format:** every row matches CONTRACT §2. The last segment is the right role header, and targets end with
  `<|im_end|>\n`.
- **Length:** the median is ~1.9k tokens and the longest row is 2,829, so `max length 4096` truncates nothing.
- **Coder:** every coder target has a code block.
- **Reviewer labels are imbalanced:** 222 `APPROVE` against 58 `REVISE`. A reviewer trained on this may
  approve almost everything.
  - Member B: if API budget allows, generate more buggy first attempts.
  - Member A: otherwise, consider oversampling the `REVISE` rows about 3×.

**Eval driver connected (for Member B).**
- `eval/run_eval.py` expected a runner factory from the runtime. It now exists as
  `rolekv.eval_runner:make_runner`, and `tests/test_eval_runner.py` runs your `evaluate()` end to end with it
  for A2 and A5-H1. The command is in [CONTRACT.md §6](CONTRACT.md#6-runtime-api-runtime--eval).
- **Tool output:** `build_dataset.py` uses `run_mbpp_task` (all asserts) for the tool output, while PLAN said
  "visible test only". The runner follows the data, so train and eval see the same tool output. PLAN.md now
  says this, with the caveat: absolute pass@1 is optimistic, and comparisons between methods stay fair.
- **Environment:** `uv sync` is the team's pinned environment. You don't need `requirements.txt` or
  `PYTHONPATH`; run modules from the repo root, e.g. `uv run python -m eval.run_eval ...`.

### 2026-10-17: first cost numbers, and a fix that makes aLoRA decoding fast

**First numbers (Mac).** `bench/cost_sweep.py` replays the same forced 4-turn session (buggy code → REVISE →
fixed code → APPROVE) for each method at growing context sizes.
- **Token counts are exact on any machine.** They depend only on the tokens, and the 0.5B and 1.5B models
  share a tokenizer.
- **Timings below are from the Mac GPU with the 0.5B model, so treat them as indicative.** The 4090 run with
  1.5B replaces them ([TRACKS.md › 4090 machine](TRACKS.md#4090-machine--lead-rtx-4090-laptop), step 4).

| Context | Tokens computed: A2 → A5-H1 | Switch TTFT: A2 → A5-H1 | GPU s / session: A2 → A5-H1 |
|---|---|---|---|
| 1k | 5,599 → 1,576 (3.6× less) | 595 → 77 ms (7.8×) | 7.6 → 6.3 s |
| 2k | 9,595 → 2,575 (3.7× less) | 996 → 83 ms (12×) | 9.5 → 6.9 s |
| 4k | 17,575 → 4,570 (3.8× less) | 1,923 → 80 ms (24×) | 13.3 → 7.8 s |
| 8k | 33,651 → 8,589 (3.9× less) | 4,071 → 101 ms (40×) | 22.3 → 9.9 s |

Full table, including A5-H2: `bench/results/cost_sweep_mac.md`. Raw event log: `bench/results/cost_sweep_mac.jsonl`.

How to read it (for the pitch and the Devpost):
- The token saving approaches the number of turns. A2 re-reads the whole context on every turn, while A5
  reads it once. "~4× less recompute" holds for this 4-turn session; a 5-turn session approaches 5×.
- Switch TTFT is the striking number. A2's grows linearly with context, while A5-H1's stays flat at about
  80–100 ms.
- A5-H1 recomputes 221 tokens per session (the previous turns' outputs). That is the price of an exact
  handoff; A5-H2 skips it.
- Decode speed per token is within ~6% between methods.
- Still no memory claim against A2.

**Decode fix.**
- **Problem:** PEFT's aLoRA forward waited for the GPU on every layer of every forward. aLoRA decoding was
  6–15× slower than plain LoRA, which wiped out the TTFT win.
- **Fix:** `rolekv/fast_alora.py` replaces it with an equivalent slice, and the runtime installs it
  automatically ([CONTRACT.md §4](CONTRACT.md#4-peft-alora-behavior)). If you run aLoRA inference outside the
  runtime, call `rolekv.fast_alora.install()` first.

**Member A:** the 7 target modules are now a constant, `rolekv.runtime.TARGET_MODULES`.

**Member C:**
- `bench/results/cost_sweep_mac.jsonl` is a full CONTRACT log you can load now. Its `machine` is `mac`, so
  4090-only timing views will show token counts only.
- `bench/results/cost_sweep_4090.jsonl` (same format) will follow from the 4090.

### 2026-10-17: environment, prompt contract and runtime are in `main`

**Everyone**

- Run `git pull`, then `uv sync`. Run everything through uv, e.g. `uv run pytest -q`. The first test run
  downloads the 0.5B model (about 1 GB).
- Build prompts and tokenize only through `rolekv/roles.py`. Don't reimplement any of it: the role headers
  must be the same token ids in data, training and runtime.
- [CONTRACT.md](CONTRACT.md) changed. §3 (event log) was revised, and §5 (adapters) and §6 (runtime API) are new.

**Member A (training)**

- Turn an SFT row into training tensors with `roles.tokenize_sft_example(tokenizer, row["segments"],
  row["target"])`. It returns `input_ids` and `labels`, with the context already masked to -100.
- aLoRA config: `alora_invocation_tokens=roles.invocation_tokens(tokenizer, role)`, `task_type="CAUSAL_LM"`.
  `tests/conftest.py` builds working configs for both adapter kinds.
- Adapter names and folder layout: [CONTRACT.md §5](CONTRACT.md#5-adapters-training--runtime). If a
  `PeftModel` holds adapters with those names, `save_pretrained(adapter_dir)` writes the right layout by
  itself. The runtime refuses adapters whose names or invocation tokens don't match.
- Target only the 7 Linear projections. Never target `lm_head` or the embeddings.
- If you run inference in the same process after training, call `gradient_checkpointing_disable()` first.

**Member B (data and eval)**

- Segment builders: `roles.task_segment(repo, text, visible_test)`, `roles.role_header(role)`,
  `roles.body_segment(output)` and `roles.tool_segment(output)`. `tool_segment` truncates to 2000 characters
  and keeps the tail, where the traceback is.
- Parsers: `roles.parse_coder_output`, `roles.parse_reviewer_verdict`, and `rolekv.runtime.extract_code`
  (it also handles an unclosed code fence).
- Eval driver: [CONTRACT.md §6](CONTRACT.md#6-runtime-api-runtime--eval) has the whole calling pattern.
  - `run_visible_test(code)` returns raw stdout + stderr.
  - Use the same function for data generation, so training and eval see the same tool output.
- Session protocol: Coder → visible test → Reviewer.
  - `APPROVE` ends the session, and so does a Reviewer reply whose first line is neither `APPROVE` nor
    `REVISE` (`INVALID`).
  - After the 2nd `REVISE`, the next Coder turn is final, so a session has at most 5 turns.
  - Generate training trajectories that follow the same rules.

**Member C (demo)**

- The dashboard from PR #1 already follows the points below; they are listed for reference.
- Event log semantics changed ([CONTRACT.md §3](CONTRACT.md#3-event-log-runtime--eval--demo)):
  - `canon_tokens` is counted in the turn that does the recompute, and that turn's `ttft_ms` includes it.
  - "Recomputed" in charts = `prefill_tokens + canon_tokens`.
  - `prefill_tokens_total` already includes canon tokens. It is the headline cost number.
  - New fields: `truncated`, `verdict`, `canon_tokens_total`. Ignore any key you don't know.
  - "TTFT after a role switch" means turns ≥ 1.
- Don't show KV memory as a win against A2: at the end of each turn it's the same for every method.
- A real log, from a 2-turn session on the Mac with random adapters and forced outputs (timings on a Mac CPU
  mean nothing). In every turn, `reused + canon + prefill` = the session length so far:

```json
{"type": "turn", "run_id": "check-A5-H1", "machine": "mac", "method": "A5-H1", "task_id": 1, "turn": 0, "role": "coder", "prefill_tokens": 174, "reused_tokens": 0, "canon_tokens": 0, "decode_tokens": 15, "ttft_ms": 206.3, "turn_ms": 845.4, "kv_mb": 4.4, "text": "```python\ndef square(x):\n    return x * x\n```", "truncated": false, "verdict": null}
{"type": "turn", "run_id": "check-A5-H1", "machine": "mac", "method": "A5-H1", "task_id": 1, "turn": 1, "role": "reviewer", "prefill_tokens": 29, "reused_tokens": 164, "canon_tokens": 24, "decode_tokens": 4, "ttft_ms": 121.3, "turn_ms": 257.1, "kv_mb": 5.2, "text": "APPROVE", "truncated": false, "verdict": "APPROVE"}
{"type": "task", "run_id": "check-A5-H1", "machine": "mac", "method": "A5-H1", "task_id": 1, "passed": true, "n_turns": 2, "prefill_tokens_total": 227, "canon_tokens_total": 24, "decode_tokens_total": 19, "gpu_s": 1.1, "peak_kv_mb": 5.2, "ctx_tokens": 48}
```

  In the same session, A2's Reviewer turn computed 217 tokens (`reused_tokens` 0). A5-H1 computed 53
  (29 prefill + 24 canon).
- A larger sample with real 4090 timings will follow from the 4090 cost preview
  ([TRACKS.md › 4090 machine](TRACKS.md#4090-machine--lead-rtx-4090-laptop), step 4).
