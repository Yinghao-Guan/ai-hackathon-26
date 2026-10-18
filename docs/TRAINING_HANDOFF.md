RoleKV · Member A → Lead

# RoleKV Trainer Handoff

> **Lead notes (2026-10-17).**
> - **The trainer is already on `peter`.** The branch is merged into `peter` (commit `c4b1519`), and
>   `train/train_role_adapter.py` there has the sha256 below. Skip the "git checkout … -- train/…" line.
> - **The data hashes on `peter` match the table.**
> - **Already checked on the Mac (0.5B, 1–2 updates per adapter):**
>   - All four A2/A5-H1 adapters passed every gate.
>   - `load_model` loaded them in bf16.
>   - `eval/run_eval.py` ran real MBPP tasks with them.
> - **Step 5 (Lead), after Step 4:** run a 5-task smoke eval for A2 and A5-H1:
>   `uv run --no-sync python -m eval.run_eval --runner rolekv.eval_runner:make_runner --model $M --adapters $OUT --methods A2 A5-H1 --machine 4090 --run-id smoke-1.5b --out tmp/smoke-eval.jsonl --limit 5`


LoRA and aLoRA role-adapter training script, ready for 1.5B training on the RTX 4090 (WSL2). Run the short validation in Steps 1–3 before any full run.

- **Synthetic validation:** passed on the 4070 (0.5B)

- **Real-data training:** pending, run Steps 1–3 on the 4090

- **4070 real-data run:** out of GPU memory on update 1

## What to train with

| Item | Value |
| --- | --- |
| Trainer commit | 801506364703ae9abbb27184015b0b328a4c2975 branch `member-a/training-adapters` on origin, parent `d581cc7` |
| Trainer file | `train/train_role_adapter.py`, sha256 eb90494ab184425761efa8c76c6656849dc35f803a07d289f6838bde573029b4 |
| Data | Fixed SFT data from `peter` commit `f43e5e0` (first segment is `roles.system_segment()`). The data on `main` still has the old prompt and is rejected by the trainer as `system_segment_mismatch`. |
| `data/sft_coder.jsonl` | 338 rows · ba3be511640388415f6eb05acd551bba5af24e8cfe3f581f9aca9eba69470906 |
| `data/sft_reviewer.jsonl` | 280 rows (222 APPROVE, 58 REVISE) · 5bcf59b7e32c481c9695ebd3c690e1d3a7766099d984ad339d84db1f1de8754a |

Hashes are the ones in `data/README.md` on `peter`, computed on LF line endings. A Windows checkout with `core.autocrlf` needs CR stripped first, as in Step 0.

One way to get the trainer onto the `peter` checkout without merging anything: `git fetch origin`, then `git checkout origin/member-a/training-adapters -- train/train_role_adapter.py`. Merging the branch works too; that choice is yours.

## Settings and planned updates

Defaults: LoRA r=16, aLoRA r=32, alpha = 2r, dropout 0.05, the 7 projections from `rolekv.runtime.TARGET_MODULES`, AdamW lr 2e-4, weight decay 0, no clipping, 2 epochs, accumulation 8, 5% linear warm-up then linear decay, batch of one row, no packing, loss on target tokens only, bf16 base with fp32 adapter, non-reentrant gradient checkpointing, max length 4,096 (nothing in the data is longer).

**Reviewer oversampling:** pass `--oversample-revise 3` to both reviewer adapters (LoRA and aLoRA) and to the shared A1 run, so every REVISE row counts three times per pass. The coder uses the default of 1.

| Adapter | Rows per pass | Microsteps | Updates | Warm-up |
| --- | --- | --- | --- | --- |
| `alora_coder`, `lora_coder` | 338 | 676 | 85 (last window 4) | 5 |
| `alora_reviewer`, `lora_reviewer` (×3) | 396 | 792 | 99 | 5 |
| `lora_all`, optional A1 (×3) | 734 | 1,468 | 184 (last window 4) | 10 |

STEP 0 · INPUTS

## Confirm the data and trainer

Run from the repo root in WSL2. All four hashes must match the table above.

```
for f in sft_coder sft_reviewer; do printf '%s  ' "$f"; tr -d '\r' < data/$f.jsonl | sha256sum; done
grep SHA-256 data/README.md
sha256sum train/train_role_adapter.py
git rev-parse HEAD   # record the runtime commit you are on
```

STEP 1 · MEMORY

## One forward/backward on the longest real rows

Uses the trainer's own setup (bf16 base, fp32 adapter, checkpointing, train mode), no optimizer step, nothing saved. The longest rows are task 973: 2,932 tokens (coder) and 2,882 tokens (reviewer). The short runs in Step 2 do not sample the 2,932-token coder row.

```
uv run --no-sync python - <<'EOF'
from pathlib import Path
import torch
from peft import get_peft_model
from transformers import AutoTokenizer
from rolekv import roles
from train import train_role_adapter as tra
tok = AutoTokenizer.from_pretrained(roles.BASE_MODEL)
for role, kind, path in [("coder", "alora", "data/sft_coder.jsonl"), ("coder", "lora", "data/sft_coder.jsonl"),
                         ("reviewer", "alora", "data/sft_reviewer.jsonl"), ("reviewer", "lora", "data/sft_reviewer.jsonl")]:
    rank = tra.DEFAULT_RANK[kind]
    cfg = tra.make_lora_config(kind, role, rank, 2 * rank, tok)
    rows, report = tra.load_examples([Path(path)], tok, role_filter=role, max_length=4096, lora_config=cfg)
    longest = max(rows, key=lambda e: len(e.input_ids))
    torch.manual_seed(0); torch.cuda.reset_peak_memory_stats()
    model = get_peft_model(tra.load_base(roles.BASE_MODEL, "cuda"), cfg, adapter_name=tra.TRAIN_ADAPTER,
                           autocast_adapter_dtype=True)
    params = tra.trainable_adapter_params(model, tra.TRAIN_ADAPTER, rank)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); model.train()
    (loss,) = tra.accumulate_window(model, [longest], "cuda")
    print(f"{kind}_{role}: usable {report['used']}, longest task {longest.task_id} = {len(longest.input_ids)} tokens, "
          f"loss {loss:.4f}, grad norm {tra.grad_norm(params):.4g}, peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MB")
    del model, params; torch.cuda.empty_cache()
EOF
```

**Pass:** four lines, each with usable 338 or 280, a finite loss, a grad norm above 0, and no out-of-memory error. Note the peak memory.

STEP 2 · SHORT RUNS

## Five updates per adapter on real data

Into a throwaway folder, never into `adapters/1.5b`. Each run trains, saves, reloads into a fresh model and checks the reload.

```
M=Qwen/Qwen2.5-Coder-1.5B-Instruct; OUT=adapters/1.5b-validate
uv run --no-sync python train/train_role_adapter.py --role coder    --kind alora --model $M --data data/sft_coder.jsonl    --out $OUT --max-steps 5
uv run --no-sync python train/train_role_adapter.py --role coder    --kind lora  --model $M --data data/sft_coder.jsonl    --out $OUT --max-steps 5
uv run --no-sync python train/train_role_adapter.py --role reviewer --kind alora --model $M --data data/sft_reviewer.jsonl --out $OUT --max-steps 5 --oversample-revise 3
uv run --no-sync python train/train_role_adapter.py --role reviewer --kind lora  --model $M --data data/sft_reviewer.jsonl --out $OUT --max-steps 5 --oversample-revise 3
```

Summarize the four run records:

```
uv run --no-sync python - <<'EOF'
import glob, json
for p in sorted(glob.glob("adapters/1.5b-validate/*/train_run.json")):
    r = json.load(open(p)); g = r.get
    print(p, g("status"), g("gate_failures"), g("error"), "| updates", g("updates"), "| loss", g("loss_first"), "->", g("loss_last"),
          "| dL2", g("param_delta_l2"), "| reload", g("reload_err"), "<=", g("reload_tol"),
          "| prehdr", g("pre_header_diff"), "<=", g("pre_header_tol"), "| peak MB", g("peak_gpu_mb"))
EOF
```

**Pass:** every run exits 0 with status `passed` and empty `gate_failures`; 5 updates each; `param_delta_l2` above 0; `reload_err` within `reload_tol`; aLoRA `pre_header_diff` within `pre_header_tol`. Losses come from a few mixed rows, so they need not fall over 5 updates.

STEP 3 · RUNTIME

## Load and run the adapters on your runtime commit

Run on the runtime commit you will evaluate with (the one recorded in Step 0). `load_model` checks the adapter names and aLoRA invocation ids; this also prints the adapter dtypes at inference and runs a short real decode for A2 and A5-H1.

```
uv run --no-sync python - <<'EOF'
import collections, torch
from rolekv.runtime import Engine, load_model, warmup
model, tok = load_model(["A2", "A5-H1"], adapter_dir="adapters/1.5b-validate", device="cuda", dtype=torch.bfloat16)
dtypes = collections.defaultdict(set)
for name, p in model.named_parameters():
    if ".lora_" in name:
        dtypes[name.split(".lora_")[1].split(".")[1]].add(str(p.dtype))
print("inference adapter dtypes:", dict(dtypes))
for name, cfg in sorted(model.peft_config.items()):
    print(name, "r", cfg.r, "alpha", cfg.lora_alpha, "invocation", cfg.alora_invocation_tokens)
for method in ("A2", "A5-H1"):
    warmup(Engine(model, tok, method)); print("exercised", method)
EOF
```

**Check the dtypes against your runtime.** The trainer saves adapters in fp32. Per STATUS (`2c4de2a`), the runtime on `peter` keeps adapters in the model dtype, so expect `torch.bfloat16` there; a runtime without that change shows `torch.float32`. Use one setting for every method in the evals.

STEP 4 · FULL TRAINING

## Full runs, only after Steps 1–3 pass

```
M=Qwen/Qwen2.5-Coder-1.5B-Instruct; OUT=adapters/1.5b
uv run --no-sync python train/train_role_adapter.py --role coder    --kind alora --model $M --data data/sft_coder.jsonl    --out $OUT
uv run --no-sync python train/train_role_adapter.py --role coder    --kind lora  --model $M --data data/sft_coder.jsonl    --out $OUT
uv run --no-sync python train/train_role_adapter.py --role reviewer --kind alora --model $M --data data/sft_reviewer.jsonl --out $OUT --oversample-revise 3
uv run --no-sync python train/train_role_adapter.py --role reviewer --kind lora  --model $M --data data/sft_reviewer.jsonl --out $OUT --oversample-revise 3
# optional A1
uv run --no-sync python train/train_role_adapter.py --role all --kind lora --model $M --data data/sft_coder.jsonl data/sft_reviewer.jsonl --out $OUT --oversample-revise 3
```

Each run writes `adapters/1.5b/{kind}_{role}/` with `train_run.json` and `train_log.jsonl` and refuses to overwrite an existing folder.

EVIDENCE

## Why real-data training is still pending

The short aLoRA coder run on the RTX 4070 Laptop (8 GB, about 6.9 GiB free) failed on its first update with `AcceleratorError: CUDA error: out of memory`. Planned: 5 updates, 40 microsteps on the fixed data; trainer at `d581cc7` (uncommitted at the time). The reviewer run, the longest-row check and the runtime checks on real-data adapters were not run.

The record is preserved on Member A's laptop at `tmp/smoke/adapters/real-0.5b/alora_coder/train_run.json` (gitignored).

Likely cause, estimated and not measured: the loss uses logits over the whole sequence. At 2.7k–2.9k tokens that path alone is about 3.8–4.1 GiB (bf16 logits, their fp32 upcast and its gradient). Lead measured a 1.5B 4,096-token step at 10.6 GiB on the 4090 with a separate loop, not this script; Step 1 confirms it for the trainer.

ALREADY CHECKED

## Completed on the 4070 with the 0.5B model

- **33 trainer CPU checks:** label masking; rejection reasons, including the stale system prompt tested on the real pre-fix rows; last-header aLoRA activation; the real optimizer and scheduler sequence (one update, partial last window, 40 updates); window gradient normalization; argument checks; gate logic; oversampling; overwrite refusal.
- **Lead's CPU tests:** 38/38 (`test_fast_alora`, `test_runtime`, `test_roles`).
- **Synthetic GPU smoke runs** (mechanics, not quality): all four role adapters plus one-update, partial-window and oversampling ×3 runs passed every gate; reload error 0 and aLoRA pre-header difference 0 everywhere.
- **Runtime:** `load_model(["A2", "A5-H1"])` loaded the smoke adapters; forced A2 and A5-H1 sessions ran with `fast_alora` active, with balanced token accounting.
- **Patch equivalence:** original and patched aLoRA forwards gave identical logits (max difference 0) on trained adapters, including a second-round context.
- **Real data:** all 618 rows pass validation; system segments and hashes match `data/README.md`.
- **Official PEFT docs** ([aLoRA guide](https://huggingface.co/docs/peft/developer_guides/lora), [LoraConfig reference](https://huggingface.co/docs/peft/package_reference/lora), [release notes](https://github.com/huggingface/peft/releases)) agree with the pinned 0.21.2 code. The docs recommend an invocation string that starts and ends with special tokens; our header ends in a newline, which segment-by-segment tokenization and the trainer's per-row `activation_mismatch` check cover.