# Interface Contract

> Frozen at T+1. Every track depends on it. To change anything here, tell the whole team first.

## 1. Prompt segments

Prompts are hand-built ChatML. Do **not** use `apply_chat_template`. A session is made of these segments:

````text
SYSTEM  <|im_start|>system\n{system prompt}<|im_end|>\n
TASK    <|im_start|>user\n# Repository file: utils.py\n```python\n{repo}\n```\n# Task\n{text}\n# Visible test\n{assert}<|im_end|>\n
ROLE    <|im_start|>assistant\n<|role:coder|>\n          (or <|role:reviewer|>)   <- the role HEADER
BODY    {role output}<|im_end|>\n
TOOL    <|im_start|>user\n<tool_output>\n{stdout + stderr, truncated to 2000 chars}\n</tool_output><|im_end|>\n
````

Order: `SYSTEM`, `TASK`, then repeated rounds of `ROLE(coder)`, `BODY`, `TOOL`, `ROLE(reviewer)`, `BODY`.

Rules:

1. **Tokenize each segment separately and concatenate the ids.** Never tokenize the joined string. BPE can
   merge across a segment boundary and change the header's tokens.
2. **The role HEADER's token ids are that role's aLoRA `alora_invocation_tokens`.** They are defined once in
   `rolekv/roles.py`; data, training and runtime all import them from there.
3. **Coder BODY:** exactly one ` ```python ` block with the new function (plus any imports).
4. **Reviewer BODY:** the first line is `APPROVE` or `REVISE: <short feedback>`.

## 2. Training examples (data → training)

Files: `data/sft_coder.jsonl` and `data/sft_reviewer.jsonl`, one JSON object per line:

```json
{"task_id": 11, "role": "coder",
 "segments": ["<|im_start|>system\n...<|im_end|>\n", "<|im_start|>user\n# Repository file: ...<|im_end|>\n", "<|im_start|>assistant\n<|role:coder|>\n"],
 "target": "```python\ndef f(x):\n    ...\n```<|im_end|>\n"}
```

- `segments`: the exact segment strings before the turn, in order. The last one is always the role HEADER.
- `target`: the full BODY segment, including the trailing `<|im_end|>\n`. Loss is computed on these tokens
  only.

## 3. Event log (runtime → eval → demo)

JSONL with one object per line. There are two line types:

```jsonc
// one line per turn
{"type":"turn","run_id":"a5h1-1.5b-ctx3k","machine":"4090","method":"A5-H1","task_id":11,"turn":1,
 "role":"reviewer","prefill_tokens":812,"reused_tokens":3120,"canon_tokens":96,"decode_tokens":41,
 "ttft_ms":38.2,"turn_ms":610.5,"kv_mb":97.1,"text":"REVISE: ...","truncated":false,"verdict":"REVISE"}
// one line per task
{"type":"task","run_id":"a5h1-1.5b-ctx3k","machine":"4090","method":"A5-H1","task_id":11,"ctx_tokens":3050,
 "passed":true,"n_turns":4,"prefill_tokens_total":5230,"canon_tokens_total":310,"decode_tokens_total":402,
 "gpu_s":3.9,"peak_kv_mb":120.4}
```

The example values are made up and show the shape only. `rolekv/costs.py` writes these records; readers
must ignore keys they don't know.

| Field | Meaning |
|---|---|
| `run_id` | one eval run (method + model + config) |
| `machine` | `4090`, `4070` or `mac`; timing fields are only compared within `4090` |
| `method` | `A0`, `A1`, `A2`, `A3`, `A5-H1`, `A5-H2` |
| `turn` | 0-based turn index within the session |
| `prefill_tokens` | new tokens computed before the first generated token this turn (excludes reused and canon) |
| `reused_tokens` | tokens served from the existing cache this turn |
| `canon_tokens` | the previous turn's adapter-computed tokens, recomputed with base weights **in this turn** (A5-H1 only, else 0). It happens in the same forward as `prefill_tokens`, so this turn's `ttft_ms` includes it. The last turn of a session is never recomputed. |
| `decode_tokens` | tokens generated this turn, including the stop token |
| `ttft_ms` | turn start → first generated token |
| `turn_ms` | whole turn |
| `kv_mb` | logical KV cache size (MiB) at the end of the turn |
| `truncated` | the turn hit its token limit without a stop token |
| `verdict` | Reviewer turns: `APPROVE`, `REVISE` or `INVALID`; `null` for Coder turns |
| `ctx_tokens` | length of the TASK segment, in tokens |
| `passed` | the final code passes all MBPP asserts |
| `prefill_tokens_total` | Σ (`prefill_tokens` + `canon_tokens`): every token computed outside decoding. **This is the headline cost number.** |
| `canon_tokens_total` | Σ `canon_tokens` |
| `gpu_s` | Σ `turn_ms`, in seconds |
| `peak_kv_mb` | largest `kv_mb` during the task |

For each turn, `reused_tokens + canon_tokens + prefill_tokens` equals the session length up to and
including that turn's role header. In charts, "recomputed" = `prefill_tokens + canon_tokens`, and "TTFT after a
role switch" means turns ≥ 1.

**Memory caveat:** with A2 as defined (full re-prefill, one cache at a time), `kv_mb` at the end of a turn is
the same for every method. KV memory savings only exist against a per-role warm cache (one cache per role kept
alive), so never claim memory savings against A2.

## 4. PEFT aLoRA behavior

These facts come from the peft 0.21.2 source (`peft/tuners/lora/variants.py`). Pin `peft==0.21.2`.

- **Where the adapter is active:** from the start of the **last** occurrence of the invocation sequence to
  the end of the input. The invocation tokens themselves are adapted.
  - So earlier turns by the same role are processed with base weights, which matches H1.
- **How the activation offset is computed:** on the `input_ids` passed to *that call*, counted back from the
  end. Two consequences:
  - `generate(input_ids=full_seq, past_key_values=base_cache)` works, **as long as the cache ends exactly
    before the header**. The offset is computed on the full ids and applied to the uncached tail.
  - A manual decode step, `model(input_ids=[[tok]], past_key_values=...)`, does not contain the invocation
    sequence, so **the adapter is silently OFF**. Pass `alora_offsets=[1]` (one entry per batch row)
    explicitly. Test 3 in TRACKS.md covers this.
- **No merging:** aLoRA adapters cannot be merged into the base weights.
- **Gradient checkpointing:** with it enabled, a second aLoRA forward without a backward raises
  "Multiple invocations of PEFT forward hooks". Disable it before any inference.
- **Slow decode:** PEFT's aLoRA forward selects the adapted rows with a boolean mask. That waits for the GPU
  on every LoRA layer of every forward, and on the Mac GPU it made aLoRA decoding 6–15× slower than plain
  LoRA.
  - `rolekv/fast_alora.py` replaces it with an equivalent slice, which `tests/test_fast_alora.py` checks
    against PEFT's version. `KVSession` installs it automatically.
  - Any other aLoRA inference code should call `fast_alora.install()` first.
  - Training does one forward per batch, so it doesn't need the patch.

## 5. Adapters (training → runtime)

- **Names:** `alora_coder`, `alora_reviewer`, `lora_coder`, `lora_reviewer`, `lora_all`.
- **Layout:** `{adapter_dir}/{name}/`, as written by `save_pretrained`, with one `adapter_dir` per base model
  size. A `PeftModel` holding adapters with these names writes exactly this layout when saved.
- **aLoRA config:** `alora_invocation_tokens` must equal `rolekv.roles.invocation_tokens(tokenizer, role)`,
  and `task_type="CAUSAL_LM"`. `load_model` refuses an aLoRA whose tokens differ, or a `lora_*` adapter that
  is actually an aLoRA.
- **Targets:** only the 7 Linear projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`,
  `up_proj`, `down_proj`). Never `lm_head` or the embeddings, which would break KV sharing with the base model.

## 6. Runtime API (runtime → eval)

```python
from rolekv.costs import EventLog
from rolekv.runtime import Engine, load_model, run_session, warmup

model, tok = load_model(["A2", "A5-H1"], adapter_dir="adapters/1.5b", device="cuda", dtype=torch.bfloat16)
engines = {m: Engine(model, tok, m) for m in ["A2", "A5-H1"]}
for engine in engines.values():
    warmup(engine)
with EventLog("runs/eval.jsonl") as log:
    for task in tasks:
        for method, engine in engines.items():  # interleave methods per task on the 4090
            res = run_session(engine, task_segment, run_visible_test)
            passed = ...  # run the hidden asserts on res.final_code
            for record in res.records(run_id=..., machine="4090", task_id=..., passed=passed):
                log.write(record)
```

- `run_visible_test(code) -> str` returns the raw stdout + stderr of the visible test. The runtime truncates
  it through the TOOL segment. It is not called after the final Coder turn.
- **Protocol:** Coder → visible test → Reviewer; `APPROVE` or an `INVALID` verdict ends the session; after the
  2nd `REVISE` the next Coder turn is final. So a session has at most 5 turns.
- **One model, sessions run one after another:** adapter switching is global state, so never run sessions
  in parallel threads on one model.
- If a forward raises, the engine's session is marked broken; log the task as failed and continue. The next
  `run_session` starts a fresh session.
- `run_session(..., forced_outputs=[...])` replays fixed turn texts instead of decoding (same code path),
  which makes token counts identical across methods, e.g. for the context-length sweep.

## 7. Repository layout and ownership

```text
docs/                         PLAN.md, CONTRACT.md, TRACKS.md                 Lead
rolekv/roles.py               segments, role headers, tokenization helpers    Lead (frozen after T+1)
rolekv/cache.py               KVSession: one cache, chunked prefill, crop     Lead
rolekv/runtime.py             role-switch engine for A0–A5                    Lead
rolekv/costs.py               token and timing counters                       Lead
rolekv/fast_alora.py          sync-free aLoRA forward (patches PEFT)          Lead
tests/                        correctness tests                               Lead
bench/cost_sweep.py           cost vs context length, forced conversation     Lead
bench/results/                sweep logs (JSONL) + summary tables (in git)     Lead
rolekv/tasks.py               MBPP loading, TASK segment, utils.py context    Member B
train/build_dataset.py        teacher data generation                         Member B
eval/testrun.py               sandboxed test runner (data and eval share it)  Member B
eval/run_eval.py              eval driver, writes the event log               Member B
data/                         SFT files (in git) + data/README.md             Member B
train/train_role_adapter.py   LoRA / aLoRA training                           Member A
demo/                         web demo                                        Member C
```
