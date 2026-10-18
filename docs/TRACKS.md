# Tracks

> One section per person. Each section lists what to build, when it counts as done (by T+5), and what you
> do next. Interfaces are in [CONTRACT.md](CONTRACT.md); machines and timeline are in [PLAN.md](PLAN.md).

## Runtime core — Lead (M4 Mac, then the 4090)

**Build** `rolekv/roles.py` first (push by T+1), then `rolekv/cache.py`, `rolekv/runtime.py`, `rolekv/costs.py`:

- **`SharedSession`:** holds one `DynamicCache`, the token list and the cost counters.
- **A5 turn:**
  1. Base-prefill any pending tokens (task, tool output).
  2. aLoRA-prefill the role header.
  3. Decode.
  4. Handoff. **H1:** `cache.crop(pre_header_len)`, then base-prefill header + body and count them as
     `canon_tokens`. **H2:** keep the adapter KV.
- **A2 turn:** start a new cache, prefill the whole history with that role's LoRA, then decode.
- **A3:** one cache. Tokens are prefilled by whichever adapter is active when they arrive, and nothing is
  ever recomputed.
- **A0 / A1:** one cache, one model.
- **Switching:** use `with model.disable_adapter():` for base passes and `model.set_adapter(role)` to switch
  roles.
- **Decoding:** greedy; `max_new_tokens` is 512 for the Coder and 128 for the Reviewer.
- **Timing:** wrap timed regions in `torch.cuda.synchronize()`. Write the event log as specified in
  CONTRACT.md.
- **APIs:** check the cache API (`DynamicCache`, `crop`) against the installed transformers version before
  relying on it.

**Done when** all four tests pass on the Mac CPU. Run them with 0.5B in fp32, with adapters created using
`init_lora_weights=False` so they are non-zero:

1. **Cache equivalence:** logits from a base-prefilled cache + an aLoRA tail equal one full aLoRA forward
   (atol 1e-4).
2. **Canonicalize:** after H1, the cache equals a base prefill of the full sequence from scratch.
3. **Adapter on during decode:** decode-step logits differ from the base logits and match a full forward.
4. **Cost accounting:** the counters on a hand-written toy session match hand-computed numbers.

**Next:** train all 1.5B adapters on the 4090 with Member A's script. Then run the timed evals: A2, A5-H1 and
the context-length sweep.

## Training script — Member A (RTX 4070 laptop)

**Build** `train/train_role_adapter.py`:

- **Args:** `--role coder|reviewer|all --kind lora|alora --rank N --model ... --data ... --out ...`
- **Input:** reads `data/sft_*.jsonl` (CONTRACT.md §2). Tokenize segment by segment with the helpers in
  `rolekv/roles.py`.
- **Loss:** computed on the `target` tokens only.
- **aLoRA config:** `LoraConfig(alora_invocation_tokens=<header ids from roles.py>, task_type="CAUSAL_LM")`.
- **Defaults:**
  - LoRA r=16, aLoRA r=32 (aLoRA reportedly needs a higher rank), alpha=2r.
  - Target modules: q/k/v/o/gate/up/down projections.
  - lr 2e-4, 2 epochs, bf16, max length 4096, gradient checkpointing.
- **Logging:** record wall-clock time and peak GPU memory for each run.
- **Before real data exists:** use a few hand-made examples in the CONTRACT format.

**Done when** the 0.5B model runs a few dozen steps on the 4070, the loss falls, and the saved adapter loads in
the runtime.

**Next:** hand the script to Lead for the 1.5B runs on the 4090. Then set up the 4070 for untimed evals, starting
with A0, which needs no adapter.

## Data — Member B (Mac)

**Build** `rolekv/tasks.py`, `eval/testrun.py` and `train/build_dataset.py`:

- **Dataset:** MBPP from Hugging Face `datasets` (`mbpp`, full config).
  - Training uses the train + validation + prompt splits.
  - Eval uses 100 test-split problems, sampled with seed 0.
  - Never mix the two.
- **`rolekv/tasks.py`:** builds the TASK segment, including a `utils.py` made of other train-split reference
  solutions. A `ctx_funcs` knob sets its size, and the result is deterministic per task.
- **`eval/testrun.py`:** runs code + asserts in a subprocess with a timeout and returns pass/fail plus
  truncated output. Data generation and eval both use it.
- **Teacher:** Claude API, model `claude-sonnet-5-5`, with the key read from `.env`.
- **Per train problem:**
  1. The teacher writes a correct solution. For about 50% of problems, it also writes a plausible buggy first
     attempt.
  2. The harness runs the tests. The teacher never invents tool output.
  3. The Reviewer label is `APPROVE` if and only if the code passes all asserts. Otherwise the teacher writes
     `REVISE` feedback from the code and the real failure output.
- **Filter:** keep a trajectory only if the buggy attempt really fails and the final code passes all asserts.
- **Output:** `data/trajectories.jsonl` → `data/sft_coder.jsonl` and `data/sft_reviewer.jsonl`. Record row
  counts and sha256 hashes in `data/README.md`.
- **Spend check:** run 20 problems first, check the actual spend, then run the rest.

**Done when** both SFT files exist and 10 random rows have been checked by hand.

**Next:** build `eval/run_eval.py`:
- 100 eval tasks × each method, greedy decoding, writing the event log.
- Cost sweep: 20 tasks at ≈1k / 4k / 8k tokens of context.

## Demo — Member C (Windows)

**Build** `demo/`, a static web page that reads event logs (JSONL). It has three views:

1. **Session replay:** A2 vs A5 side by side. One bar per turn, split into recomputed vs reused tokens, with
   TTFT per turn.
2. **Cost vs quality:** prefill tokens per task vs pass@1, one point per method.
3. **Context-length sweep:** prefill tokens saved vs context length.

- Build against mock logs in the exact CONTRACT.md schema, and swap in real logs at T+12.
- Timing views use only rows where `machine` is `4090`.

**Done when** all three views work on mock logs.

**Next:** README, Devpost page, a 2-minute video and the pitch.
