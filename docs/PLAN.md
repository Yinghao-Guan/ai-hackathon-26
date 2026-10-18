# RoleKV — LA Hacks Team Plan

> 2026-10-17 · ~20 hours left. Read this first, then [CONTRACT.md](CONTRACT.md) and your section of
> [TRACKS.md](TRACKS.md). [SPEC.md](SPEC.md) is the long-term research spec; this plan overrides its
> scope for the hackathon.

## Goal

Ship a live demo showing that a multi-role coding agent can switch roles with **the same pass rate and far
less recomputation**.

Pitch:

> Multi-agent coding with one small model and a LoRA per role re-reads the whole context on every role switch.
> RoleKV trains each role as an Activated LoRA, so all roles share one KV cache: same pass rate, X× less
> recompute.

X comes from our measurements. Never estimate it.

## Background (read this if you skipped SPEC.md)

- Our agent has two roles, **Coder** and **Reviewer**. Both run on one base model, and each role has its own
  LoRA adapter.
- **Problem:** a standard LoRA changes the KV cache of every token. The Coder's cache is useless to the
  Reviewer, so every role switch re-prefills the whole context: the task, repo code, test output and earlier
  turns.
- **Fix:** an Activated LoRA (aLoRA) only turns on after its role header. Everything before the header is
  computed with base weights, so all roles share one cache, and a switch computes only the new tokens.
- **Handoff:** the tokens the previous role generated were computed with its adapter. **H1** recomputes them
  with base weights at the handoff (exact). **H2** keeps them as they are (cheaper, but differs from
  training). We test both.

What gets computed at each turn (schematic):

| Turn | A2: one standard LoRA per role | A5: one aLoRA per role |
|---|---|---|
| 1 · Coder | the whole context | the whole context |
| 2 · Reviewer | everything again | only turn 1's output + the test output |
| 3 · Coder | everything again | only the tokens added since turn 2 |

## Scope

| Keep | Cut |
|---|---|
| Qwen2.5-Coder-1.5B-Instruct on GPU; Qwen2.5-Coder-0.5B-Instruct for CPU dev and tests (same tokenizer) | 7B (would need 4-bit QLoRA on 16 GB, which changes numerics) |
| Coder + Reviewer | Planner |
| MBPP: train on the train + validation + prompt splits (~470 problems); evaluate on 100 fixed test problems (seed 0) | SWE-bench, agent frameworks |
| Methods A0, A2, A3, A5-H1, A5-H2; A1 if time allows | A4 (LRAgent reimplementation), vLLM serving |
| 4 correctness tests | Long literature review |

| ID | Method | Cache | Why it's here |
|---|---|---|---|
| A0 | Base model + role prompts, no adapter | one cache | quality floor |
| A1 | One LoRA shared by both roles | one cache | answers "why not just one LoRA?" |
| A2 | One LoRA per role, full re-prefill on every switch | no sharing | quality reference, highest cost |
| A3 | One LoRA per role, reuse the other role's KV as is | shared, mismatched | shows how much quality breaks |
| **A5-H1** | One aLoRA per role, recompute the previous role's output at handoff | shared, exact | **our method** |
| A5-H2 | Same, but keep the previous role's KV | shared, approximate | cheaper; how much quality drops |

**Session protocol:**
1. The Coder writes code.
2. The harness runs the visible test (MBPP's first assert) and appends the output.
3. The Reviewer answers `APPROVE` (done) or `REVISE: ...` (back to the Coder). At most 2 REVISE rounds.
4. The final code is scored on all of MBPP's asserts.

**Long shared context:** each task starts with a "repository file", `utils.py`, built from other MBPP
reference solutions (train split). It defaults to ≈3k tokens; the cost sweep uses ≈1k / 4k / 8k. Without a
long shared context there is nothing to save.

## Team and machines

The runtime core is the hardest part and the easiest place for silent bugs. Give it to whoever knows
PyTorch and transformers best (default: Lead). Fill in names below.

| Who | Machine | T+1–5 | After T+5 | Owns |
|---|---|---|---|---|
| Lead: ___ | M4 Mac 24 GB + RTX 4090 laptop 16 GB | Runtime core + 4 correctness tests (Mac CPU, 0.5B) | Trains all 1.5B adapters and runs timed evals on the 4090 | `rolekv/cache.py`, `rolekv/runtime.py`, `rolekv/costs.py`, `rolekv/roles.py`, `tests/` |
| Member A: ___ | RTX 4070 laptop 8 GB (Windows) | Training script, debugged with 0.5B on the 4070 | Untimed quality evals on the 4070 | `train/train_role_adapter.py` |
| Member B: ___ | Mac | Data: Claude-written dialogues → real test runs → SFT files | Eval driver (reuses the test runner from data generation) | `train/build_dataset.py`, `rolekv/tasks.py`, `eval/`, `data/` |
| Member C: ___ | Windows (no GPU) | Demo web page, 3 views, on mock logs | README, Devpost, video, pitch | `demo/` |

## GPUs

The 4070 laptop has 8 GB, which is not enough to train 1.5B at 4k tokens, so **all 1.5B training runs on
the 4090**. The 4070 runs 0.5B debugging and 1.5B inference (about 3 GB of weights in bf16).

- **Timing only on the 4090.** TTFT and GPU-seconds are not comparable across machines, so the headline
  A2 vs A5 comparison runs on the 4090.
- **Token counts are deterministic.** Prefill token counts are the same on any machine, so the other
  methods can run on the 4070.

| Phase | RTX 4090 laptop (16 GB) | RTX 4070 laptop (8 GB) |
|---|---|---|
| Training, T+5–8 | 5 adapters, one after another: aLoRA coder, aLoRA reviewer, LoRA coder, LoRA reviewer, shared LoRA (A1) | A0 eval (needs no adapter); pull the other adapters as they finish |
| Eval, T+8–12 | A2 and A5-H1 with timing; context-length sweep | A1, A3, A5-H2: pass rate and token counts |

If the first adapter takes more than 35 minutes to train, rent a cloud GPU (24 GB or more) and train in
parallel.

## Timeline

T = when the team starts on this plan.

| When | What |
|---|---|
| T+0–1 | GitHub repo; environment on every machine; [CONTRACT.md](CONTRACT.md) frozen; `rolekv/roles.py` pushed |
| T+1–5 | Four tracks in parallel ([TRACKS.md](TRACKS.md)) |
| **T+5** | **Merge point: data, training script and runtime tests must all be ready.** GPU work cannot start without all three. |
| T+5–8 | Training on the 4090; A0 eval on the 4070 |
| T+8–12 | Evals on both GPUs (fine to run overnight) |
| T+12–16 | Real numbers into the demo; README |
| T+16–20 | Devpost, video, pitch rehearsal, buffer |

**Cut order if behind:** A1 → H2 → context sweep → A3. **Never cut:** A2 vs A5, and the correctness tests.

## Collaboration

- **Repo:** a private GitHub repo. Lead creates it and invites everyone. Each track edits only the paths it
  owns (see the team table).
- **Frozen interfaces:** [CONTRACT.md](CONTRACT.md) and `rolekv/roles.py` are frozen after T+1. To change
  either one, tell the whole team first.
- **What goes where:**
  - Data files (a few MB) go in git.
  - Adapters (tens of MB each) go in a private Hugging Face repo or a shared drive, not git.
- **API key:** `.env` is gitignored. Lead puts the Anthropic API key on Member B's machine and sets a spend
  limit in the Anthropic Console.
- **Environment:** install [uv](https://docs.astral.sh/uv/), then run `uv sync` in the repo. `pyproject.toml` and
  `uv.lock` pin every version (Python 3.11, `peft==0.21.2`). Windows and Linux get the CUDA 12.6 build of torch;
  macOS gets the default build. Native Windows is fine, no WSL needed. Run things with `uv run ...`, e.g.
  `uv run pytest`.
- **Coding agents:** before your Claude Code or Codex session starts work, have it read `docs/PLAN.md`,
  `docs/CONTRACT.md` and your section of `docs/TRACKS.md`.

## Risks

| Risk | Response |
|---|---|
| aLoRA quality well below LoRA | Raise the rank, report it honestly; the cost win still stands |
| PEFT `generate` + cache edge cases | Fall back to a hand-written decode loop with explicit `alora_offsets` |
| 4090 training too slow | Rent a cloud GPU (see GPUs) |
| Laptop GPU throttling makes timings noisy | Prefill tokens (deterministic) are the primary metric; TTFT is secondary |
| A track misses T+5 | Start training as soon as data is ready; if the runtime is late, run A2 and A3 first |
| Savings look small | The context-length sweep shows how savings grow with context length |

## Open questions

- [ ] Who owns the runtime core (default: Lead)?
- [ ] Names for Lead and Members A, B, C.
- [ ] Repo name and access.
- [ ] Do Anthropic's usage terms cover fine-tuning a model on Claude outputs for this project?
- [ ] Hugging Face or a shared drive for adapters?
