# Status

> Read this first every time you pull. Each owner keeps their own row in the table current. Changes that
> affect other tracks go under "Changes", newest first.

## Tracks

| Track | Owner | Status | Notes |
|---|---|---|---|
| Environment | Lead | Done | `uv sync` installs everything, pinned by `uv.lock` |
| Prompt contract (`rolekv/roles.py`) | Lead | Done, frozen | |
| Runtime core | Lead | Done on Mac CPU | 28 tests pass; CUDA/bf16 check on the 4090 pending |
| Cost sweep (`bench/cost_sweep.py`) | Lead | Done on Mac; 4090 run pending | Results in `bench/results/`; see Changes below |
| 4090 machine | Lead | In progress | [TRACKS.md › 4090 machine](TRACKS.md#4090-machine--lead-rtx-4090-laptop) |
| Training script | Member A | | |
| Data | Member B | 280 trajectories in `data/` | Checked by Lead against CONTRACT §2: all rows valid, max 2,829 tokens. Reviewer labels are 79% APPROVE (see Changes) |
| Eval driver | Member B | Written; connected to the runtime | `--runner rolekv.eval_runner:make_runner`, see [CONTRACT.md §6](CONTRACT.md#6-runtime-api-runtime--eval) |
| Demo | Member C | Dashboard merged (PR #1) | Runs on mock logs; already counts compute as prefill + canon. Real 4090 sample: 4090 machine step 4 |

## Changes

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
