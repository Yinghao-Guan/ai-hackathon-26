# RoleKV — Shared-KV Role Adapters for Coding Agents

> Working name. Owner: Peter. This file is the project spec for Claude Code.
> Status: proposal / Phase 0. Last updated: 2026-10-17.

---

## 0. TL;DR

Multi-role coding agents (planner → coder → reviewer) are often built as **one open base model + one LoRA adapter per role**. Even though the roles share the base weights, standard LoRA changes the KV cache, so **every role switch forces a full re-prefill of the whole session** (issue text, repo files, tool outputs, earlier turns). That is wasted GPU time and memory.

Two families of fixes now exist:

1. **Training-side, exact sharing** — *Activated LoRA (aLoRA)*: the adapter only activates *after* an invocation sequence, so all earlier tokens use base-model KV, which every role can share exactly.
2. **Inference-side, approximate sharing** — *LRAgent, ForkKV, PReCache, DroidSpeak*: keep standard LoRAs, but share the base part of the cache and recompute / store only the small adapter-dependent part (or a few critical layers).

**This project:** a head-to-head comparison of these families **in a coding-agent setting where the roles generate long outputs (plans, code, reviews)**, plus a study of the *handoff policy* between two aLoRA roles. Main output: cost–quality curves (GPU-seconds / prefill tokens vs. tests passed / issues resolved).

The novelty claim must be verified in Phase 0 before anything else (see §2).

---

## 1. Background and motivation

### 1.1 The problem

- KV cache for token *t* at layer *l* = hidden state × that layer's `W_K`, `W_V`.
- A standard LoRA modifies q/k/v/o and MLP projections for **every** token. Even if layer-1 K/V happened to match, adapted outputs change layer-2 inputs, and the difference compounds with depth.
- So role A's cache is not valid for role B. Serving engines therefore keep prefix caches separate per adapter (verify how vLLM / SGLang key their cache blocks): **no cross-adapter reuse by default**.
- In an agent, the shared context (repo snippets, tool outputs) is long and grows every turn, while each role's own output is comparatively short. Re-prefilling the shared part at every switch dominates prefill cost and inflates time-to-first-token (TTFT).

### 1.2 Simple cost model (use this to sanity-check measurements)

Session with shared context `C` tokens, `R` role switches; turn `i` adds `g_i` generated tokens and `t_i` tool-output tokens.

| Scheme | Prefill tokens computed (approx.) | KV memory held |
|---|---|---|
| Per-role standard LoRA, full recompute on switch | `Σ_k (C + history_k)` ≈ `O(R · C)` | up to `R` full caches if roles kept warm |
| aLoRA roles, shared base cache, canonicalize handoff (§4.3) | `C + Σ t_i + Σ g_i` (+ invocation tokens) | 1 full cache + current role's tail |
| aLoRA roles, keep-adapter-KV handoff | `C + Σ t_i` (+ invocation tokens) | 1 full cache |

Prefill is parallel and cheaper per token than decode, so wall-clock wins show up mostly as **TTFT after a role switch** and as **throughput under concurrency** (memory freed → bigger batches).

---

## 2. Prior work (verify and extend in Phase 0)

| Work | Family | Exact / approx. | Needs training? | Notes |
|---|---|---|---|---|
| Activated LoRA (Greenewald et al., IBM, 2025) | Training-side | Exact for pre-invocation tokens | Yes (adapter trained with activation masking) | Designed for "intrinsics": short specialized calls inside a base-model chain. Reports aLoRA often needs **higher rank** than LoRA for parity. PEFT supports it via `alora_invocation_tokens`. |
| aLoRA serving (Li et al., Dec 2025) | Serving | Exact | — | vLLM fork with base-aligned block hashing + activation-aware masking; large TTFT / latency gains in multi-turn multi-adapter pipelines. |
| LRAgent (Feb 2026) | Inference-side | Approx. | No (BaseShared); shared-A LoRA variant for BaseLRShared | Splits cache into shared base cache + per-agent low-rank cache. Reports key caches are almost identical across agents; value cache carries most of the divergence. Planner / action / reflection roles. |
| ForkKV (Apr 2026) | Serving | Approx. | No | Copy-on-write disaggregated KV (shared + per-agent parts), DualRadixTree. |
| PReCache (Sep 2026) | Inference-side | Approx. | No | Precomputes each agent's low-rank cache; reconstructs base cache from adapter-free hidden states. Notes that naive reuse weakens role-specific behavior. **Found via an aggregator — confirm the arXiv entry.** |
| DroidSpeak (NSDI '26) | Inference-side | Approx. | No | Reuses most layers of another fine-tune's KV, recomputes a few critical layers. |
| Cache-to-Cache (C2C, 2025) | Cross-model fusion | — | Yes (fuser) | Different models; not a direct baseline, cite as related. |

### Working hypothesis about the gap (must be checked)

As far as an initial search shows, nobody has:

1. Compared **aLoRA-style trained sharing vs. inference-time sharing of standard LoRAs** on the same roles, data and budget.
2. Evaluated aLoRA when the **adapter itself generates most tokens** (long code / plans), rather than short intrinsic outputs.
3. Studied the **handoff between two aLoRA roles** (what to do with KV of tokens generated by role A when role B takes over).
4. Done any of the above on **agentic coding** (function-level tasks with tests, then SWE-bench-style issues).

If Phase 0 finds this is already done, pivot to the narrowest remaining open question (likely RQ3 handoff policy, or role-count scaling) and update this spec.

---

## 3. Research questions

- **RQ1 (quality).** On coding-agent roles, how close does aLoRA get to per-role standard LoRA with full recompute? How does the gap depend on rank and on output length?
- **RQ2 (cost).** For the same quality target, how do prefill tokens, TTFT after switches, GPU-seconds per task and peak KV memory compare across A0–A5 (§4.2)?
- **RQ3 (handoff).** For aLoRA roles: canonicalize vs. keep adapter KV for previously generated tokens — quality and cost trade-off.
- **RQ4 (is multi-adapter even needed?).** Does a single multi-role LoRA (which shares cache trivially) match per-role adapters? If yes, what is left of the argument for multiple adapters (modularity, independent updates)?

---

## 4. Method

### 4.1 Roles and protocol

Start with a minimal agent loop, expand later.

- **Planner**: issue + context → short plan (files to touch, steps).
- **Coder**: plan + context → edits / tool calls; runs in a multi-step loop with tool outputs appended.
- **Reviewer**: diff + test output → `APPROVE` or `REVISE: <feedback>`.

Loop: Planner → Coder (≤ k steps) → Reviewer → (back to Coder on REVISE, max m rounds).

MVP may use only Coder + Reviewer. Every turn is delimited with a role header in the chat template, e.g. `<|role:coder|>`. For aLoRA, that header (tokenized exactly as in training) is the role's **invocation sequence**. Watch tokenization boundaries — the invocation tokens must appear identically in training and inference.

### 4.2 Approaches compared

All share the same base model, data, and training-step budget (tune rank per method).

| ID | Approach | Cache behavior |
|---|---|---|
| A0 | Base model + role system prompts, no adapters | Trivially shared (one model). Quality floor. |
| A1 | One multi-role standard LoRA (role in prompt) | Trivially shared. Key baseline for RQ4. |
| A2 | Per-role standard LoRA, full re-prefill on every switch | No sharing. Quality reference, worst cost. |
| A3 | Per-role standard LoRA, naive reuse of whatever KV exists | Shared but mismatched. Shows the degradation. |
| A4 | Per-role standard LoRA + inference-time sharing | Approx. Implement the simplest faithful version of LRAgent BaseShared (or use their code), optionally DroidSpeak-style partial-layer recompute. |
| A5 | Per-role **aLoRA** (ours), handoff H1 or H2 | Exact for base-processed tokens. |

### 4.3 aLoRA role mechanics and handoff policy

- Context tokens (issue, repo text, tool outputs) are always processed with **base weights** → one shared cache per session.
- When role R is invoked, its adapter is active from the invocation sequence onward; its queries attend over the shared base KV plus its own new KV.
- **Problem:** tokens role A generated have **adapter-A KV**. Role B was trained expecting base KV for everything before its invocation.
  - **H1 — canonicalize:** after A's turn ends, re-prefill A's generated tokens with base weights and overwrite those cache entries. Exact match to training distribution; costs one parallel prefill of `g_i` tokens.
  - **H2 — keep:** leave adapter-A KV in place. Cheaper; train/test mismatch unless training data simulates it. Measure.
  - Optional **H2-trained:** generate training contexts where earlier role turns carry the producing adapter's KV (harder; only if H2 looks promising).
- Training: PEFT `LoraConfig(alora_invocation_tokens=..., task_type="CAUSAL_LM")`; loss only on the role's output tokens. Sweep rank (aLoRA reportedly needs higher rank than LoRA). Verify the exact activation semantics (e.g., which occurrence of the invocation sequence counts) in the installed PEFT version before writing training code.

---

## 5. Data

Two tiers. Use tier 1 for all development; tier 2 for final results.

1. **Function-level, test-verified** (fast iteration): multi-role transcripts on small coding tasks with unit tests (MBPP / HumanEval-style, or a held-out custom set). Generate role transcripts with a stronger teacher model, keep only trajectories whose final code passes tests.
2. **Repo-level, agentic**: issue-fixing trajectories. Candidate sources to check (availability, size, license): SWE-Gym, SWE-smith, public SWE-agent trajectory datasets on Hugging Face. Alternatively run a teacher model inside the harness with the §4.1 protocol and segment trajectories by role.

Rules: strict train/eval split by repository / task; never train on evaluation instances; record dataset versions and hashes.

---

## 6. Evaluation

### 6.1 Quality

- Tier 1: pass@1 on unit tests (final Coder output after Reviewer loop).
- Tier 2: % resolved on a fixed SWE-bench Lite or Verified subset (start 50, grow to 100–300 if compute allows), run inside mini-SWE-agent or OpenHands.
- Role fidelity: planner format compliance; reviewer accuracy on a labeled set of good/bad patches (precision/recall of REVISE).

### 6.2 Cost

- Prefill tokens actually computed per task (instrumented, not estimated).
- TTFT after each role switch.
- GPU-seconds per task (end to end).
- Peak KV-cache memory.
- Stretch: throughput at N concurrent sessions in a serving engine.

Report **cost–quality scatter / Pareto plots**: x = GPU-seconds (or prefill tokens) per task, y = pass rate.

### 6.3 Correctness tests (must pass before any experiment is trusted)

1. **Cache equivalence:** for an aLoRA role, logits from (base-prefilled cache + adapter decode) == logits from a single full forward with the activation mask. Test in fp32 with tight tolerance; bf16 with looser tolerance.
2. **Canonicalization:** after H1, the overwritten cache equals a cache built by base-prefilling the full sequence from scratch.
3. **Positions:** position ids / RoPE offsets stay consistent after cache surgery.
4. **Cost accounting:** token counters match a hand-computed toy session.

---

## 7. Phased plan

| Phase | Goal | Deliverable | Exit criterion |
|---|---|---|---|
| P0 | Literature and gap check | `notes/related_work.md`: table of each paper's roles, benchmarks, trained vs. training-free, exact vs. approx., output lengths, code availability | Gap confirmed, or RQs revised in this spec |
| P1 | Measure the problem | Per-layer K and V similarity (cosine, relative L2) between base and two quick standard LoRAs on the same context; A2 vs. A3 quality on a small set | Figure + numbers in `RESULTS.md` |
| P2 | Core runtime | `SharedSession` with one cache, role switching, H1/H2, cost counters (HF transformers, single stream) | All §6.3 tests pass |
| P3 | Train and compare on tier 1 | Role adapters for A1, A2–A4, A5; rank sweep | Cost–quality table for A0–A5 on tier 1 |
| P4 | Agentic tier 2 | Harness integration, SWE subset runs | Cost–quality curves on tier 2 |
| P5 (stretch) | Serving | Run A5 on the aLoRA vLLM fork; concurrency benchmark | Throughput vs. concurrency plot |
| P6 | Write-up | Short paper / report + clean repo | Draft reviewed by Peter |

---

## 8. Suggested repo layout

```
rolekv/
  SPEC.md                 # this file
  README.md
  RESULTS.md              # every number links to config + log + commit
  DECISIONS.md            # dated decision log
  notes/related_work.md
  configs/                # yaml per experiment
  rolekv/
    roles.py              # role defs, invocation tokens, chat template
    adapters.py           # load / activate standard LoRA and aLoRA (PEFT)
    cache.py              # SharedSession: shared cache, canonicalize, slicing
    runtime.py            # role-switch engine implementing A0–A5
    costs.py              # prefill/decode token + timing accounting
  train/
    build_dataset.py
    train_role_adapter.py
  eval/
    func_suite/
    swe_subset/
    run_eval.py
    analysis/
  tests/
    test_cache_equivalence.py
    test_canonicalize.py
    test_positions.py
    test_cost_accounting.py
```

---

## 9. Models and compute

- Base: an open instruction-tuned **coder** model, same family across sizes so tokenizers match (e.g., a current Qwen coder model — check the latest release). Develop on ~1.5B; main results on ~7B.
- Rough needs: 1.5B LoRA/aLoRA training fits on a single 24 GB GPU; 7B needs a 40–80 GB GPU in bf16, or QLoRA on 24 GB (note QLoRA changes numerics — keep equivalence tests in the same precision as experiments).
- **Peter to fill in:** available GPUs / cloud budget: `____`.

---

## 10. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Idea already fully covered by 2026 work | P0 gap check; pivot to handoff (RQ3) or role scaling |
| aLoRA quality drops for long-generation roles | Rank sweep; report as a finding — a negative result here is still a contribution |
| A1 (single multi-role LoRA) matches everything | Report honestly; reframe around modularity (independent role updates, adding roles without retraining) |
| Cache surgery bugs silently corrupt results | §6.3 tests gate all experiments |
| Library APIs differ from docs / papers | Check installed versions; pin versions in `requirements.txt` |
| Tier-2 eval too expensive | Fixed small subset, multiple seeds, report confidence intervals |

---

## 11. Working agreements for Claude Code

- Verify every library API (PEFT aLoRA arguments, transformers cache classes, vLLM fork) against the **installed** version before use; cache APIs change often.
- Never write a number into `RESULTS.md` that was not produced by a logged run. Each entry: config path, log path, git commit, hardware, seed.
- Develop on the smallest model first; scale only after tests pass.
- **Ask Peter before:** any job expected to exceed ~2 GPU-hours, any paid API usage (e.g., teacher-model data generation), downloading datasets larger than ~20 GB, or changing the research questions.
- Log design choices in `DECISIONS.md` with date and reason.
- Prefer reusing official code (IBM activated-lora, LRAgent repo, aLoRA vLLM fork) over reimplementing; note license and commit used.

---

## 12. Open questions for Peter

1. Goal: research paper, portfolio project, or learning/replication? (Changes how much of P4–P6 matters.)
2. Compute available?
3. Teacher model for data generation: open model only, or paid API allowed?
4. MVP roles: Coder + Reviewer only, or all three from the start?

---

## 13. References

- Greenewald et al. *Activated LoRA: Fine-tuned LLMs for Intrinsics.* arXiv:2504.12397. Code: https://github.com/IBM/activated-lora
- Li et al. *Efficient Multi-Adapter LLM Serving via Cross-Model KV-Cache Reuse with Activated LoRA.* arXiv:2512.17910. Code: https://github.com/tdoublep/vllm/tree/alora
- *LRAgent: Efficient KV Cache Sharing for Multi-LoRA LLM Agents.* arXiv:2602.01053. Code: https://github.com/hjeon2k/LRAgent
- *ForkKV: Scaling Multi-LoRA Agent Serving via Copy-on-Write Disaggregated KV Cache.* arXiv:2604.06370
- *PReCache: Efficient KV Cache Sharing for Multi-LoRA Agents via Low-Rank Precomputation and Neutral Reconstruction.* Reported as arXiv:2609.34054 — confirm.
- Liu et al. *DroidSpeak: KV Cache Sharing Across Fine-tuned Model Variants.* NSDI '26; arXiv:2411.02820
- Hugging Face PEFT — aLoRA support (`alora_invocation_tokens`): https://github.com/huggingface/peft
- Cache-to-Cache (C2C), cross-model KV fusion, 2025 — related work, confirm citation.
- Harnesses / benchmarks: mini-SWE-agent, OpenHands, SWE-bench (Lite / Verified).
