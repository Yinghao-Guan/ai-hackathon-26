# RoleKV: shared-KV role adapters for coding agents

LA Hacks 2026 project. A multi-role coding agent (Coder + Reviewer) where every role is its own adapter on one
small base model, and switching roles does not mean re-reading the whole context.

> **Status:** in progress. The demo dashboard runs on mock data; experimental results are not in yet.
> Every number in this repo that is a result will trace back to a logged run.

## The problem

A common way to build a multi-role agent cheaply is one open base model plus one LoRA adapter per role. The
roles share base weights, but a standard LoRA changes the hidden states, and therefore the KV cache, of
**every** token. The Coder's cache is useless to the Reviewer. So at every role switch the engine re-prefills
the whole session: the task, the repository file, test output and all earlier turns. In an agent, that shared
context is long and grows every turn, while each role's own output is short.

## The idea

An **Activated LoRA** (aLoRA, Greenewald et al., 2025) is only switched on from its *invocation sequence*
onward. Here that sequence is the role header (`<|im_start|>assistant\n<|role:coder|>\n`). Everything before
the header is computed with base weights, so all roles can share **one** KV cache, and a role switch only
computes the tokens added since the last turn.

One wrinkle remains: the tokens the previous role generated were computed *with its adapter*. We test two
handoff policies:

- **H1 (canonicalize):** after a turn, recompute that turn's header and output with base weights and overwrite
  their cache entries. Exact with respect to training; costs one parallel prefill of the output.
- **H2 (keep):** leave the adapter-computed entries in place. Cheaper; differs from what the next role saw in
  training.

## Methods compared

| ID | Method | Cache |
|---|---|---|
| A0 | Base model + role prompts, no adapter | one cache (quality floor) |
| A1 | One LoRA shared by both roles | one cache |
| A2 | One LoRA per role, full re-prefill at every switch | no sharing (quality reference, highest cost) |
| A3 | One LoRA per role, reuse the other role's KV as is | shared but mismatched |
| **A5-H1** | One aLoRA per role, canonicalize at handoff | shared, exact (our method) |
| A5-H2 | One aLoRA per role, keep the previous role's KV | shared, approximate |

Setup (from [docs/PLAN.md](docs/PLAN.md)): Qwen2.5-Coder-1.5B-Instruct; MBPP tasks with a ≈3k-token
`utils.py` "repository file" as shared context; Coder writes code → the visible test runs → Reviewer answers
`APPROVE` or `REVISE` (at most 2 rounds) → final code scored on all asserts. Which methods were actually
evaluated will be listed under Results.

## Demo dashboard

```bash
cd demo
python -m http.server 8000     # open http://localhost:8000
```

- **Session replay:** A2 vs A5-H1 on the same task, turn by turn: tokens recomputed, reused from cache, and
  canonicalized at handoff.
- **Cost vs quality:** prefill tokens per task vs pass rate for every method in the log.
- **Context sweep:** tokens saved per task as the shared context grows (≈1k / 3k / 4k / 8k).

With no real log present, it shows a clearly labelled mock fixture. To load real results, put the eval log at
`demo/data/events.jsonl`, or use **Load event log**, or drag a `.jsonl` file onto the page. Details,
metric definitions and the event-log assumptions are in [demo/README.md](demo/README.md).

## Metrics

| Metric | Definition |
|---|---|
| Prefill ratio (X×) | mean A2 `prefill_tokens_total` ÷ mean A5-H1 `prefill_tokens_total`, on task_ids both ran |
| % fewer prefill tokens | 100 × (1 − A5-H1 mean / A2 mean) |
| Pass rate | final code passes all MBPP asserts, over all tasks in the run (greedy, pass@1) |
| Context sweep | the same comparison at each shared-context size, x = median `ctx_tokens` |

`prefill_tokens_total` counts every token actually computed before decoding, **including** H1
canonicalization. Prefill token counts are deterministic and are the primary cost metric; TTFT and
GPU-seconds are compared only within runs on the same RTX 4090.

## Results

> _Pending real runs. Nothing below may be filled from mock data._

| | A2 | A5-H1 |
|---|---|---|
| Pass rate (n tasks) | `TBD` | `TBD` |
| Mean prefill tokens / task | `TBD` | `TBD` |
| Prefill ratio | | `TBD` × (`TBD`% fewer, `TBD` paired tasks) |

Source runs: `TBD` (run_id, machine, log path, commit).

## Reproducibility

- Event-log schema: [docs/CONTRACT.md](docs/CONTRACT.md) §3. Plan and scope: [docs/PLAN.md](docs/PLAN.md).
- Eval set: 100 MBPP test-split problems, sampled with seed 0; training uses the train, validation and prompt splits.
- Hardware: training and timed evals on an RTX 4090 laptop GPU (16 GB); untimed evals on an RTX 4070 laptop GPU.
- Runs, configs and commits: `TBD` when the eval logs land.

## Limitations

- One small model (1.5B) and function-level tasks (MBPP), not repository-level agents.
- Two roles only; a single run per method (greedy decoding), so pass-rate differences of a few points may be noise.
- Laptop GPUs; timings are noisy and only compared within one machine.
- aLoRA reportedly needs a higher rank than LoRA for the same quality; the prior work targets short
  "intrinsic" outputs, while our roles generate whole functions.
- H2 differs from training; whether that matters is part of what we measure.
- Baselines from related work (e.g. LRAgent-style inference-time sharing) are out of scope for the hackathon.

## Team

Lead: runtime core and training runs · Member A: training script · Member B: data and eval · Member C: demo,
README, Devpost, pitch. See [docs/TRACKS.md](docs/TRACKS.md).

## References

- Greenewald et al., *Activated LoRA: Fine-tuned LLMs for Intrinsics*, arXiv:2504.12397.
- Hugging Face PEFT (aLoRA via `alora_invocation_tokens`), pinned at `peft==0.21.2`.
- Further related work is listed in [docs/SPEC.md](docs/SPEC.md) §2.
