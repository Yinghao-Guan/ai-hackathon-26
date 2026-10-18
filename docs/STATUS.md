# Status

> Read this first every time you pull. Each owner keeps their own row in the table current. Changes that
> affect other tracks go under "Changes", newest first.

## Tracks

| Track | Owner | Status | Notes |
|---|---|---|---|
| Environment | Lead | Done | `uv sync` installs everything, pinned by `uv.lock` |
| Prompt contract (`rolekv/roles.py`) | Lead | Done, frozen | |
| Runtime core | Lead | Done on Mac CPU | 28 tests pass; CUDA/bf16 check on the 4090 pending |
| 4090 machine | Lead | In progress | [TRACKS.md › 4090 machine](TRACKS.md#4090-machine--lead-rtx-4090-laptop) |
| Training script | Member A | | |
| Data | Member B | | |
| Eval driver | Member B | Not started | Runtime API ready: [CONTRACT.md §6](CONTRACT.md#6-runtime-api-runtime--eval) |
| Demo | Member C | | |

## Changes

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
