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

Setup (from [docs/PLAN.md](docs/PLAN.md)):
- **Model:** Qwen2.5-Coder-1.5B-Instruct.
- **Tasks:** MBPP, each with a `utils.py` "repository file" of other MBPP solutions as shared context
  (≈1.5k–1.9k tokens, the same as in the training data).
- **Session:** the Coder writes code → the task's tests run → the Reviewer answers `APPROVE` or `REVISE` (at
  most 2 rounds) → the final code is scored on all asserts.

The methods actually evaluated are listed under Results.

## Demo

```bash
cd demo
python -m http.server 8000     # open http://localhost:8000
```

- **Homepage** (`index.html`): the pitch on one screen. The KV cache is drawn in 3D (Three.js, vendored in
  `demo/vendor/three`): two blocks of cells pointing at the viewer, a 14 × 14 map of the token positions in
  front and 28 layers deep, replaying one measured 4-turn session for per-role LoRAs and RoleKV side by
  side, with the same session as 2D tapes and running token counts in the HUD. Its numbers are copied from `bench/results/cost_sweep_4090.md` into `demo/js/home.js` next
  to their source; it shows no timing or quality number until those runs land.
- **Dashboard** (`dashboard.html`), which reads event logs:
  - **Session replay:** A2 vs A5-H1 on the same task, turn by turn: tokens recomputed, reused from cache, and
    canonicalized at handoff.
  - **Cost vs quality:** prefill tokens per task vs pass rate for every method in the log.
  - **Context sweep:** tokens saved per task as the shared context grows (≈1k / 2k / 4k / 8k).
- **Presentation stage** (`dashboard.html?present`): one full-screen beat per part of the pitch, for the video.

With no real log present, the dashboard shows a clearly labelled mock fixture. To load real results, put the
eval log at `demo/data/events.jsonl`, or use **Load event log**, or drag a `.jsonl` file onto the page. Details,
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

### Cost (measured)

`bench/cost_sweep.py` replays one fixed 4-turn session (buggy code → `REVISE` → fixed code → `APPROVE`) with
each method at four shared-context sizes.
- **Token counts** depend only on the tokens, so they are exact on any machine. They came out identical on the
  Mac and the 4090.
- **Timings** are from the RTX 4090 laptop: WSL2, 1.5B in bf16, adapters in bf16, 5 repeats.

| Shared context | Tokens computed per session: A2 → A5-H1 | Switch TTFT: A2 → A5-H1 | GPU s per session: A2 → A5-H1 |
|---|---|---|---|
| 1.1k | 5,599 → 1,576 (3.6× fewer) | 94 → 50 ms (1.9×) | 6.56 → 6.84 s |
| 2.1k | 9,595 → 2,575 (3.7× fewer) | 155 → 48 ms (3.2×) | 6.82 → 6.89 s |
| 4.1k | 17,575 → 4,570 (3.8× fewer) | 279 → 47 ms (5.9×) | 7.19 → 7.01 s |
| 8.1k | 33,651 → 8,589 (3.9× fewer) | 583 → 50 ms (11.7×) | 8.40 → 7.32 s (13% less) |

- **Why the saving approaches the number of turns:** A2 re-reads the whole context on every turn, while A5-H1
  reads it once.
- **The H1 handoff cost:** A5-H1's count includes 221 tokens of H1 handoff recompute per session.
- **Switch TTFT:** A5-H1's stays at ~50 ms. A2's grows linearly, because it re-reads the context on every
  switch.
- **Whole-session time is mostly decoding** (~230 tokens per session), so it moves much less:
  - 4% slower than A2 at 1k and 13% faster at 8k.
  - A5-H1 decodes 8% slower per token (34.3 vs 31.7 ms). That is the Python overhead of PEFT's aLoRA code
    path across 196 adapted layers, not extra compute, and it counts against RoleKV.
- **What these numbers measure:** cost only. The turn texts are forced, so they say nothing about quality.
- **Sources:** `bench/results/cost_sweep_4090.{jsonl,md}` and `bench/results/cost_sweep_mac.*`, on the
  `peter` branch.

### Quality and cost on 100 MBPP eval tasks (measured)

Setup: one greedy run per method, all on the RTX 4090 (WSL2, 1.5B in bf16, adapters in bf16), ~1.8k tokens of
shared context, the same 100 tasks for every method.

| Method | Final passes / 100 | Mean prefill tokens / task | Median switch TTFT | Mean model-turn s / task |
|---|---|---|---|---|
| A0 base, no adapters | 51 | 2,038 | 20 ms | 4.29 |
| A1 one LoRA for both roles | 61 | 2,060 | 35 ms | 5.29 |
| **A2 one LoRA per role, full re-prefill** | **59** | **7,013** | **139 ms** | **5.56** |
| A3 one LoRA per role, reuse mismatched KV | 62 | 2,061 | 41 ms | 5.54 |
| **A5-H1 RoleKV (aLoRA, exact handoff)** | **61** | **2,197** | **45 ms** | **4.29** |
| A5-H2 RoleKV, keep the previous role's KV | 60 | 2,059 | 44 ms | 4.40 |

Final passes score the session's last code after up to two revisions, with full-suite tool feedback.
Model-turn time sums logged turn durations; it excludes tools, final scoring, and loading.

**A5-H1 vs A2:**
- **Prefill volume:** 3.2× fewer processed prefill tokens per task (68.7% fewer, including 138 tokens of H1 handoff
  recompute), on the same 100 tasks.
- **Latency:** 3.1× faster switches (median TTFT).
- **Model-turn time:** 23% less per task, with different generated trajectories and stopping decisions;
  this is not an isolated cache speedup or a tool-inclusive wall-time result.
- **Quality:** no detected paired success-rate difference. Both methods pass 51, only A2 passes 8,
  only A5-H1 passes 10 (exact two-sided McNemar p = 0.815). This does not establish equivalence.

How to read the other rows:
- **The five adapter methods have similar observed pass counts** (59–62). One binomial standard error is about
  5 points at n = 100.
- **Mismatched KV reuse (A3) and one shared LoRA (A1) remain competitive on this benchmark.**
  - H1 restores a base-model prefix for the incoming role, but this experiment does not demonstrate a
    quality premium for that invariant. Numerical behavior at the deployed precision still requires validation.
  - And each role keeps its own adapter, so roles can be trained and updated independently.
- **A known weakness of aLoRA here: second reviews.**
  - The aLoRA Reviewers approved 41/45 (H1) and 39/45 (H2) of their second reviews, and 35 and 34 of those
    tasks' final code failed.
  - The LoRA Reviewers (A1, A2, A3) mostly revised and never approved failing code in that round.
  - The training data has no second reviews.
- **A0 on the RTX 4070 (WSL2) passes 54, vs 51 on the 4090.** Greedy decoding in bf16 on another GPU shifts
  results by a few points, which is why every comparison here is on one machine.

**Sources (branch `peter`):**
- `eval/results/eval-1.5b-4090-202610180231.jsonl` and its `.summary.md` (eval commit `00dc40e`, adapters
  from `train/results/1.5b/`);
- `eval/results/eval-1.5b-4070-202610180229.*` (A0 on the 4070, run by Member A).

## Reproducibility

- Event-log schema: [docs/CONTRACT.md](docs/CONTRACT.md) §3. Plan and scope: [docs/PLAN.md](docs/PLAN.md).
- Eval set: 100 MBPP test-split problems, sampled with seed 0; training uses the train, validation and prompt splits.
- Hardware: training and every reported eval on an RTX 4090 laptop GPU (16 GB) under WSL2; an extra A0 run on an RTX 4070 laptop GPU (WSL2).
- Runs, configs and commits:
  - eval: `eval/results/` on `peter` (eval commit `00dc40e`);
  - training records: `train/results/1.5b/*/train_run.json`;
  - cost sweep: `bench/results/`.
- GPU correctness gate: `RT_DEVICE=cuda uv run pytest -q` passed 60/60 in fp32 on the 4090 before the evals.

## Limitations

- One small model (1.5B) and function-level tasks (MBPP), not repository-level agents.
- Two roles only; a single run per method (greedy decoding), so pass-rate differences of a few points may be noise.
- On MBPP with 1.5B, naive KV reuse (A3) and one shared LoRA (A1) were as good as per-role adapters, so the
  quality case for exact sharing is not shown here.
- aLoRA Reviewers mostly approve on a second review, including failing code, because the SFT data has no
  multi-round reviews. Fix: generate multi-round trajectories.
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
