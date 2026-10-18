# Devpost draft: RoleKV

> Draft for the submission page. Everything in `[[double brackets]]` is a placeholder that must be filled
> from validated real logs (run_ids noted next to each number) before submitting. Do not use mock values.

**Tagline:** One model, many roles, one KV cache: coding agents that switch roles without re-reading
everything.

## Inspiration

Multi-role coding agents (a coder, a reviewer, a planner) keep re-processing the same context. A popular cheap
setup is one small open model with a LoRA adapter per role. The roles share almost all their weights, yet every
role switch throws away the KV cache and re-prefills the task, the repository code, the test output and every
earlier turn. As sessions grow, most of the GPU time before each answer goes into re-reading text the model has
already read. We wanted to know whether that waste is avoidable without giving up the benefit of separate
role adapters.

## What it does

RoleKV trains each role as an **Activated LoRA** (aLoRA): an adapter that only switches on after the role's
header. Everything before the header is processed with plain base weights, so the Coder and the Reviewer share
one KV cache. A role switch only computes the new tokens. When a role hands off, we recompute its short output
with base weights ("canonicalization", H1), which keeps the shared cache exactly what each role saw in training.

We compare this against per-role standard LoRAs that re-prefill at every switch (A2), a naive cache-reuse
baseline (A3), a single shared LoRA (A1), the base model (A0), and an aLoRA variant that skips
canonicalization (H2), all on MBPP coding tasks where code is checked by real unit tests.

The dashboard makes the cost visible:

- **Session Replay** plays a Coder ↔ Reviewer session turn by turn, showing how many tokens each method
  recomputes, reuses or canonicalizes.
- **Cost vs Quality** puts every method on one chart: prefill tokens per task against pass rate.
- **Context Sweep** shows how the savings change as the shared context grows from ≈1k to ≈8k tokens.

Every headline number is computed in the browser from the loaded event logs; nothing is hardcoded.

## How we built it

- **Model:** Qwen2.5-Coder-1.5B-Instruct (0.5B for CPU development and tests), trained with Hugging Face PEFT
  (`peft==0.21.2`, aLoRA via `alora_invocation_tokens`) on PyTorch + transformers.
- **Runtime:** a shared-session engine that keeps one KV cache, switches adapters per role, crops and
  recomputes the cache at H1 handoffs, and logs prefill, reuse and canonicalization tokens per turn.
  `[[confirm against rolekv/ once merged]]`
- **Data:** MBPP. Teacher dialogues (Claude) for training, with every tool output coming from actually running
  the tests, never invented. `[[confirm final row counts from data/README.md]]`
- **Correctness tests:** cache equivalence against a full aLoRA forward pass, canonicalization equal to a fresh
  base prefill, adapter-on during decode, and a hand-computed cost-accounting check.
- **Dashboard:** a static HTML/CSS/JS page with hand-drawn SVG charts and no build step or dependencies,
  designed first in Paper. It validates every log line against our frozen event-log contract, audits token
  accounting, and keeps mock data from ever mixing with real results.

## Challenges we ran into

- **Exact cache semantics.** aLoRA's activation point is computed from the input ids of each forward call. In
  a hand-written decode loop, the adapter silently turns off unless you pass the offsets explicitly. We wrote
  a test specifically for "adapter on during decode".
- **Tokenization boundaries.** The role header is the adapter's trigger, so it must tokenize identically in
  training and at inference. We tokenize every prompt segment separately and never let BPE merge across a
  boundary.
- **Handoff policy.** The previous role's output was computed with its adapter. Deciding whether to recompute
  it (H1) or keep it (H2), and accounting for that cost honestly, shaped the whole runtime.
- **Honest cost accounting.** It's easy to make savings look bigger by forgetting the canonicalization tokens.
  Our logs count them, and the dashboard audits every task total against its turns.
- **Comparable timing.** We trained and evaluated across laptops with different GPUs; timing is compared only
  on one RTX 4090, and token counts (deterministic) are the primary metric.

`[[add any challenge that actually happened during training/eval]]`

## Accomplishments that we're proud of

- `[[X.XX]]`× fewer prefill tokens per task for A5-H1 vs A2 (`[[YY.Y]]`% fewer, `[[N]]` paired MBPP tasks,
  ≈3k-token shared context). Source: `[[run_ids]]`.
- Pass rate `[[A5-H1 %]]` vs `[[A2 %]]` for per-role LoRAs. `[[State the difference plainly; do not claim
  parity unless the numbers show it.]]`
- `[[Correctness tests: N/4 passing, on what hardware/precision]]`
- A dashboard that makes a KV-cache idea understandable in seconds, and refuses to show unverified numbers as
  results.

## What we learned

- Cache sharing is only as good as its correctness: a single silently-disabled adapter would have produced
  great-looking savings and wrong outputs. Equivalence tests had to come before any experiment.
- The savings come from the shared context, so they depend on how long it is relative to what each role
  writes. `[[Summarize the context-sweep result once measured.]]`
- `[[Measured finding about H1 vs H2 quality, if run.]]`
- `[[Measured finding about aLoRA vs LoRA quality at our ranks.]]`

Hypotheses we did not test are under What's next.

## What's next

Future work, not done in this hackathon:

- More roles (a planner) and longer agent loops, where switches are more frequent.
- Larger models (7B) and repository-level tasks (SWE-bench-style issues).
- Head-to-head comparison with inference-time cache-sharing methods for standard LoRAs (e.g. LRAgent,
  DroidSpeak-style partial recompute).
- Serving with an aLoRA-aware engine and measuring throughput under concurrent sessions.
- Training with simulated H2 contexts to see whether the cheaper handoff can match H1.

## Built with

`python` · `pytorch` · `transformers` · `peft` · `qwen2.5-coder` · `mbpp` · `claude` (teacher data) ·
`javascript` · `svg` · `paper`

## Links

- Repo: `[[url]]`
- Demo video: `[[url]]`
