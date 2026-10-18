# 1.5B eval on the 4090: eval-1.5b-4090-202610180231

Six methods, 100 MBPP test tasks each (the fixed seed-0 set; identical task ids for every method), run
unattended on the RTX 4090 laptop in WSL2 on 2026-10-18 from 02:31 to 03:23.

- **Eval commit:** `00dc40e` (peter). Adapters: `adapters/1.5b`, trained at `79e1f6a` per
  docs/TRAINING_HANDOFF.md; run records in `train/results/1.5b/`.
- **Model:** `Qwen/Qwen2.5-Coder-1.5B-Instruct`, bf16, `--ctx-funcs 30` (TASK segment 1,280–2,710 tokens,
  median 1,810), greedy decoding.
- **Gate:** `RT_DEVICE=cuda uv run pytest -q` (fp32 on the GPU) passed 60/60 before any eval.
- **Log:** `eval/results/eval-1.5b-4090-202610180231.jsonl` (all six methods). The driver appends the method to
  each row's `run_id` (`eval-1.5b-4090-202610180231-A2`, ...).
- **Command** (per step, with `--methods A2 A5-H1`, then `A5-H2 A3 A1`, then `A0` without `--adapters`):
  `uv run --no-sync python -m eval.run_eval --runner rolekv.eval_runner:make_runner --model $M --adapters adapters/1.5b --methods ... --machine 4090 --run-id eval-1.5b-4090-202610180231 --out eval/results/eval-1.5b-4090-202610180231.jsonl --limit 100`

## Results

| Method | pass@1 | Turns mean / max | Reviewer APPROVE / REVISE / INVALID | 2nd-round reviews: APPROVE / REVISE / INVALID | 2nd-round APPROVE of code that finally failed | Truncated turns | Mean prefill_tokens_total | Median TTFT, turns ≥ 1 (ms) | Session time (sum of turn_ms) |
|---|---|---|---|---|---|---|---|---|---|
| A2 | 59/100 | 3.27 / 5 | 59 / 84 / 0 | 2 / 41 / 0 (of 43) | 0 | 21 | 7,013 | 139.4 | 9.3 min |
| A5-H1 | 61/100 | 2.94 / 5 | 96 / 49 / 0 | 41 / 4 / 0 (of 45) | 35 | 15 | 2,197 | 45.4 | 7.2 min |
| A5-H2 | 60/100 | 2.96 / 5 | 94 / 51 / 0 | 39 / 6 / 0 (of 45) | 34 | 24 | 2,059 | 43.7 | 7.3 min |
| A3 | 62/100 | 3.24 / 5 | 62 / 81 / 0 | 5 / 38 / 0 (of 43) | 0 | 46 | 2,061 | 40.9 | 9.2 min |
| A1 | 61/100 | 3.24 / 5 | 60 / 82 / 0 | 2 / 40 / 0 (of 42) | 0 | 41 | 2,060 | 35.3 | 8.8 min |
| A0 | 51/100 | 3.58 / 5 | 0 / 102 / 54 | 0 / 46 / 10 (of 56) | 0 | 93 | 2,038 | 19.9 | 7.1 min |

Definitions:
- *2nd-round reviews* are Reviewer turns after a REVISE, i.e. every Reviewer turn of a task except the first.
  The last column counts those that said APPROVE on a task whose final code failed the asserts.
- *Truncated turns* hit `max_new_tokens`. All are Reviewer turns (128-token cap) except 3 Coder turns in A0.
  The verdict is parsed from the first line, so a truncated Reviewer turn still has a verdict.
- *Session time* sums `turn_ms` over the method's turns; it excludes model loading.

Wall time per step, measured around each `run_eval` call (includes model loading):

| Step | Methods | Wall time |
|---|---|---|
| 2 | A2, A5-H1 | 1,011 s (16.9 min) |
| 3 | A5-H2, A3, A1 | 1,555 s (25.9 min) |
| 4 | A0 | 444 s (7.4 min) |

## Observations

- **Cost:** A5-H1 computes 2,197 prefill tokens per task against 7,013 for A2 (3.2× fewer), and its median
  switch TTFT is 45.4 ms against 139.4 ms (3.1× lower).
- **Quality:** pass@1 is 59–62 for every adapter method (A2, A5-H1, A5-H2, A3, A1) and 51 for the base model
  (A0). With 100 tasks, one binomial standard error is about 5 points, so the adapter methods are not
  distinguishable from each other here.
- **aLoRA Reviewer approves revisions too easily:** in second-round reviews, the aLoRA reviewers (A5-H1, A5-H2)
  said APPROVE 41/45 and 39/45 times, and 35 and 34 of those approvals were for code that finally failed. The
  LoRA reviewers (A2, A3, A1) mostly said REVISE in the second round and never approved failing code there.
  The training data is 79% APPROVE (docs/STATUS.md).
- **A0** (base model with role prompts) never says APPROVE: 54 of its Reviewer replies are INVALID (first line
  is neither APPROVE nor REVISE), and 90 of its Reviewer turns hit the 128-token cap.

## What went wrong

Nothing crashed: all three `run_eval` calls exited 0 and wrote 100 task rows per method; the logs contain no
tracebacks. Conditions worth knowing:

- No sign of GPU memory spill: Windows' shared GPU memory stayed at its ~84 MB baseline in every sample taken
  (02:36–03:00); after that only `nvidia-smi` was sampled, and dedicated memory never exceeded 7.1 GB of 16 GB
  in any sample. `PYTORCH_CUDA_ALLOC_CONF` was not set for the evals (it was set for training only; see the
  training commit).
- A progress monitor ran every 4 minutes during the evals (one `nvidia-smi` call and one JSONL read), which
  adds negligible CPU load. Nothing else ran on the GPU.
- The bf16 test suite was not rerun; per the Lead's decision, the fp32 run on CUDA is the gate.
