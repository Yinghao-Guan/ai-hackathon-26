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
 "role":"reviewer","prefill_tokens":812,"reused_tokens":3120,"canon_tokens":0,"decode_tokens":41,
 "ttft_ms":38.2,"turn_ms":610.5,"kv_mb":97.1,"text":"REVISE: ..."}
// one line per task
{"type":"task","run_id":"a5h1-1.5b-ctx3k","machine":"4090","method":"A5-H1","task_id":11,"ctx_tokens":3050,
 "passed":true,"n_turns":4,"prefill_tokens_total":5230,"decode_tokens_total":402,"gpu_s":3.9,"peak_kv_mb":120.4}
```

The example values are made up and show the shape only.

| Field | Meaning |
|---|---|
| `run_id` | one eval run (method + model + config) |
| `machine` | `4090`, `4070` or `mac`; timing fields are only compared within `4090` |
| `method` | `A0`, `A1`, `A2`, `A3`, `A5-H1`, `A5-H2` |
| `turn` | 0-based turn index within the session |
| `prefill_tokens` | tokens actually computed before decoding this turn (excludes reused tokens) |
| `reused_tokens` | tokens served from the existing cache this turn |
| `canon_tokens` | tokens recomputed with base weights at the H1 handoff after this turn (A5-H1 only, else 0) |
| `decode_tokens` | tokens generated this turn |
| `ttft_ms` | turn start → first generated token |
| `turn_ms` | whole turn |
| `kv_mb` | KV cache size at the end of the turn |
| `ctx_tokens` | length of the TASK segment, in tokens |
| `passed` | the final code passes all MBPP asserts |
| `gpu_s` | GPU time for the whole task |
| `peak_kv_mb` | largest KV cache size during the task |

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

## 5. Repository layout and ownership

```text
docs/                         PLAN.md, CONTRACT.md, TRACKS.md                 Lead
rolekv/roles.py               segments, role headers, tokenization helpers    Lead (frozen after T+1)
rolekv/cache.py               SharedSession: one cache, crop, canonicalize    Lead
rolekv/runtime.py             role-switch engine for A0–A5                    Lead
rolekv/costs.py               token and timing counters                       Lead
tests/                        correctness tests                               Lead
rolekv/tasks.py               MBPP loading, TASK segment, utils.py context    Member B
train/build_dataset.py        teacher data generation                         Member B
eval/testrun.py               sandboxed test runner (data and eval share it)  Member B
eval/run_eval.py              eval driver, writes the event log               Member B
data/                         SFT files (in git) + data/README.md             Member B
train/train_role_adapter.py   LoRA / aLoRA training                           Member A
demo/                         web demo                                        Member C
```
