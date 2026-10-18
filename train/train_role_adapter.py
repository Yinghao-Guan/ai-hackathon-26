"""Train one role adapter (standard LoRA or Activated LoRA) on the SFT files of CONTRACT.md section 2.

    uv run --no-sync python train/train_role_adapter.py --role coder --kind alora \
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct --data data/sft_coder.jsonl --out adapters/1.5b

The adapter is saved as {out}/{kind}_{role}/ (CONTRACT.md section 5), next to train_run.json (the run
record) and train_log.jsonl (one line per optimizer update). An existing {out}/{kind}_{role}/ is never
overwritten.

--oversample-revise N uses every reviewer row whose target starts with REVISE N times per pass over the data
(--role reviewer, and the reviewer rows of --role all); the data files are not changed.

Training: batch of one row, no packing, loss on the target tokens only (labels from
roles.tokenize_sft_example). Each row's loss is the mean over its target tokens; an optimizer update
averages the rows of its accumulation window, and the backward pass follows each row's forward. The base
model runs in bf16 with non-reentrant gradient checkpointing; the adapter stays in fp32.

Every run must have finite losses, finite non-zero gradient norms and adapter weights that actually change.
After training, the script checks the adapter on the first usable row against a conservative acceptance bound
(see BF16_STEP): an aLoRA must leave the positions before its final role header as the base model computes
them, and the adapter reloaded from disk into a fresh model must reproduce the trained logits. The measured
errors (repeat_err, reload_err, pre_header_diff) are recorded separately from the bounds (reload_tol,
pre_header_tol). The adapter's effect on the logits is recorded but is not a gate.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import math
import subprocess
import sys
import time
from collections import Counter
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import peft
import torch
import transformers
from peft import LoraConfig, PeftModel, get_peft_model
from peft.tuners.lora import LoraLayer
from peft.tuners.lora.variants import calculate_alora_offsets
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from rolekv import roles  # noqa: E402
from rolekv.runtime import TARGET_MODULES  # noqa: E402  (CONTRACT.md section 5; importing installs no patch)

log = logging.getLogger("train_role_adapter")

DEFAULT_RANK = {"lora": 16, "alora": 32}
LORA_DROPOUT = 0.05
WEIGHT_DECAY = 0.0
WARMUP_FRACTION = 0.05
# Acceptance bound for two computations of the same bf16 logits: BF16_STEP * the largest |logit| compared.
# bf16 keeps 8 significant bits, so representable values near magnitude m are at most 2**-7 * m apart; the
# bound is one such spacing at the largest logit, applied to every compared position, so it is conservative
# for the smaller logits. It is a pass/fail bound, not a measurement: identical weights and inputs on the same
# kernels are expected to differ far less (the recorded repeat_err and reload_err), while wiring errors
# (adapter active before its header, a module missing after reload) move logits by the adapter's whole effect.
BF16_STEP = 2.0**-7
# In memory the adapter is PEFT's "default", so save_pretrained writes straight into {out}/{name}/ and leaves
# the shared {out}/ alone. Saved files do not depend on this name; the runtime loads them by path as {name}.
TRAIN_ADAPTER = "default"
PRE_HEADER_PROBE = 64  # positions before the final header compared against the base model
BODY_END = roles.IM_END + "\n"
REJECT_REASONS = (
    "malformed_json",
    "missing_or_bad_field",
    "unknown_role",
    "role_filtered",
    "system_segment_mismatch",
    "header_mismatch",
    "bad_target_ending",
    "empty_target",
    "overlength",
    "activation_mismatch",
)


class TrainingError(RuntimeError):
    """A gate or invariant failed; the run stops and its record says why."""


@dataclass(frozen=True)
class Example:
    task_id: object
    role: str
    input_ids: list[int]
    labels: list[int]
    header_len: int
    revise: bool = False  # a reviewer row whose target starts with REVISE

    @property
    def target_len(self) -> int:
        return sum(label != roles.IGNORE_INDEX for label in self.labels)


# --- Data ----------------------------------------------------------------------------------------


def prepare_row(line: str, tokenizer, *, role_filter: str, max_length: int,
                lora_config: LoraConfig) -> tuple[Example | None, str]:
    """Validate and tokenize one JSONL line. Returns (example, "ok") or (None, reject reason)."""
    try:
        row = json.loads(line)
    except json.JSONDecodeError:
        return None, "malformed_json"
    segments, target = (row.get("segments"), row.get("target")) if isinstance(row, dict) else (None, None)
    if (not isinstance(row, dict) or "task_id" not in row or not isinstance(segments, list) or not segments
            or not all(isinstance(s, str) for s in segments) or not isinstance(target, str)):
        return None, "missing_or_bad_field"
    role = row.get("role")
    if role not in roles.ROLES:
        return None, "unknown_role"
    if role_filter != "all" and role != role_filter:
        return None, "role_filtered"
    if segments[0] != roles.system_segment():  # CONTRACT.md section 2: the runtime's system segment
        return None, "system_segment_mismatch"
    if segments[-1] != roles.role_header(role):
        return None, "header_mismatch"
    if not target.endswith(BODY_END):
        return None, "bad_target_ending"
    if not target[: -len(BODY_END)].strip():
        return None, "empty_target"

    tokens = roles.tokenize_sft_example(tokenizer, segments, target)
    if len(tokens["input_ids"]) > max_length:
        return None, "overlength"
    header_len = len(roles.invocation_tokens(tokenizer, role))
    revise = role == "reviewer" and target.startswith("REVISE")
    example = Example(row["task_id"], role, tokens["input_ids"], tokens["labels"], header_len, revise)
    if lora_config.alora_invocation_tokens is not None:
        (offset,) = calculate_alora_offsets({TRAIN_ADAPTER: lora_config}, TRAIN_ADAPTER,
                                            torch.tensor([example.input_ids]))
        if offset != header_len + example.target_len:
            return None, "activation_mismatch"
    return example, "ok"


def load_examples(paths: Sequence[Path], tokenizer, **row_kwargs) -> tuple[list[Example], dict]:
    """Read every file, keep the valid rows, and count each rejection reason."""
    examples: list[Example] = []
    counts: Counter[str] = Counter()
    files = []
    for path in paths:
        raw = path.read_bytes()
        lines = [line for line in raw.decode("utf-8").splitlines() if line.strip()]
        for line in lines:
            example, reason = prepare_row(line, tokenizer, **row_kwargs)
            counts[reason] += 1
            if example is not None:
                examples.append(example)
        # sha256_lf matches data/README.md on any OS; git with core.autocrlf checks files out with CRLF.
        files.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
                      "sha256_lf": hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest(), "rows": len(lines)})
    report = {"files": files, "used": counts["ok"], "rejected": {r: counts[r] for r in REJECT_REASONS}}
    return examples, report


def oversample_revise(examples: Sequence[Example], factor: int) -> list[Example]:
    """A new list in which every REVISE reviewer row appears `factor` times; other rows appear once."""
    if factor < 1:
        raise ValueError("factor must be >= 1")
    return [example for example in examples for _ in range(factor if example.revise else 1)]


# --- Schedule ------------------------------------------------------------------------------------


def plan_windows(n_examples: int, *, epochs: int, grad_accum: int, max_steps: int | None) -> list[int]:
    """Accumulation-window sizes, one per optimizer update.

    With max_steps, exactly max_steps full windows (epochs is ignored and rows repeat). Otherwise
    epochs * n_examples microsteps in one stream, so only the final window can be partial.
    """
    if n_examples <= 0 or epochs <= 0 or grad_accum <= 0 or (max_steps is not None and max_steps <= 0):
        raise ValueError("n_examples, epochs, grad_accum and max_steps must be positive")
    total = max_steps * grad_accum if max_steps else epochs * n_examples
    return [min(grad_accum, total - start) for start in range(0, total, grad_accum)]


def example_order(n_examples: int, seed: int) -> Iterator[int]:
    """Endless row indices: one seeded permutation per pass over the data."""
    for pass_index in range(sys.maxsize):
        generator = torch.Generator().manual_seed(seed + pass_index)
        yield from torch.randperm(n_examples, generator=generator).tolist()


def warmup_updates(total_updates: int) -> int:
    if total_updates <= 0:
        raise ValueError("total_updates must be positive")
    return min(total_updates, max(1, math.ceil(WARMUP_FRACTION * total_updates)))


def lr_factor(update: int, total_updates: int, warmup: int) -> float:
    """Linear warm-up over `warmup` updates, then linear decay; `update` is 0-based.

    Updates 0..total_updates-1 all get a positive factor. LambdaLR also evaluates update == total_updates
    after the last optimizer step; no update uses it, so it is 0 (this also covers warmup == total_updates).
    """
    if not 1 <= warmup <= total_updates:
        raise ValueError(f"warmup {warmup} must be between 1 and total_updates {total_updates}")
    if update >= total_updates:
        return 0.0
    if update < warmup:
        return (update + 1) / warmup
    return (total_updates - update) / (total_updates - warmup)


# --- Model ---------------------------------------------------------------------------------------


def make_lora_config(kind: str, role: str, rank: int, alpha: int, tokenizer) -> LoraConfig:
    return LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=LORA_DROPOUT,
        target_modules=list(TARGET_MODULES),
        task_type="CAUSAL_LM",
        alora_invocation_tokens=roles.invocation_tokens(tokenizer, role) if kind == "alora" else None,
    )


def load_base(model_name: str, device: str):
    base = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, attn_implementation="sdpa")
    base.config.use_cache = False
    return base.to(device)


def trainable_adapter_params(model, adapter_name: str, rank: int) -> list[torch.nn.Parameter]:
    """The parameters to optimize, after checking they are exactly the adapter's LoRA A/B matrices."""
    wrapped = [(name, module) for name, module in model.named_modules() if isinstance(module, LoraLayer)]
    expected_modules = len(TARGET_MODULES) * model.config.num_hidden_layers
    if len(wrapped) != expected_modules or any(n.rsplit(".", 1)[-1] not in TARGET_MODULES for n, _ in wrapped):
        raise TrainingError(f"wrapped {len(wrapped)} modules, expected {expected_modules} target projections")
    trainable = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    stray = [n for n, _ in trainable if f".lora_A.{adapter_name}." not in n and f".lora_B.{adapter_name}." not in n]
    if stray:
        raise TrainingError(f"unexpected trainable parameters, e.g. {stray[:3]}")
    if any(p.dtype != torch.float32 for _, p in trainable):
        raise TrainingError("adapter parameters are not fp32")
    expected = sum(rank * (m.in_features + m.out_features) for _, m in wrapped)
    count = sum(p.numel() for _, p in trainable)
    if count != expected:
        raise TrainingError(f"{count} trainable parameters, expected {expected}")
    return [p for _, p in trainable]


def accumulate_window(model, batch: Sequence[Example], device) -> list[float]:
    """Forward and backward each row in turn, scaled so the window's gradient is the mean row gradient."""
    losses = []
    for example in batch:
        input_ids = torch.tensor([example.input_ids], device=device)
        labels = torch.tensor([example.labels], device=device)
        loss = model(input_ids=input_ids, labels=labels, use_cache=False).loss
        if not torch.isfinite(loss):
            raise TrainingError(f"non-finite loss {loss.item()} on task {example.task_id}")
        (loss / len(batch)).backward()
        losses.append(loss.item())
    return losses


def grad_norm(params: Sequence[torch.nn.Parameter]) -> float:
    norms = [p.grad.detach().float().norm() for p in params if p.grad is not None]
    return torch.linalg.vector_norm(torch.stack(norms)).item() if norms else 0.0


def train(model, params, examples: Sequence[Example], windows: Sequence[int], args, log_file) -> list[dict]:
    """Run every accumulation window; one optimizer and scheduler step per window."""
    total, warmup = len(windows), warmup_updates(len(windows))
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=WEIGHT_DECAY)
    scheduler = LambdaLR(optimizer, lambda u: lr_factor(u, total, warmup))
    order = example_order(len(examples), args.seed)
    device = next(model.parameters()).device
    history, microsteps, start = [], 0, time.perf_counter()
    model.train()
    optimizer.zero_grad(set_to_none=True)
    for update, size in enumerate(windows):
        losses = accumulate_window(model, [examples[next(order)] for _ in range(size)], device)
        norm = grad_norm(params)
        if not math.isfinite(norm) or norm == 0.0:
            raise TrainingError(f"gradient norm {norm} at update {update}")
        lr = optimizer.param_groups[0]["lr"]
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        microsteps += size
        entry = {"update": update, "window_size": size, "microsteps_total": microsteps,
                 "loss": sum(losses) / size, "lr": lr, "grad_norm": norm,
                 "elapsed_s": round(time.perf_counter() - start, 3)}
        history.append(entry)
        log_file.write(json.dumps(entry) + "\n")
        log_file.flush()
        if update % 10 == 0 or update == total - 1:
            log.info("update %d/%d  loss %.4f  lr %.2e  grad_norm %.3g", update + 1, total, entry["loss"], lr, norm)
    return history


@torch.inference_mode()
def probe_logits(model, example: Example) -> torch.Tensor:
    """Logits of the final header, the target and up to PRE_HEADER_PROBE positions before the header.

    logits_to_keep keeps the full-context vocabulary logits from ever being materialized.
    """
    keep = min(len(example.input_ids), PRE_HEADER_PROBE + example.header_len + example.target_len)
    input_ids = torch.tensor([example.input_ids], device=next(model.parameters()).device)
    return model(input_ids=input_ids, use_cache=False, logits_to_keep=keep).logits[0].float()


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a - b).abs().max().item() if a.numel() else 0.0


def bf16_tolerance(logits: torch.Tensor) -> float:
    """The acceptance bound for comparing `logits` with a recomputation: BF16_STEP * max |logit|."""
    return BF16_STEP * logits.abs().max().item() if logits.numel() else 0.0


def measure_adapter(model, probe: Example) -> tuple[torch.Tensor, dict]:
    """Compare the trained adapter with the base model and with itself on the probe row.

    Returns the logits of the adapted positions (final header + target) on the CPU in bf16, which is all the
    reload check needs, and the measurements. Positions before the header are compared here and dropped.
    """
    trained = probe_logits(model, probe)
    repeat_err = max_abs(probe_logits(model, probe), trained)
    with model.disable_adapter():
        base = probe_logits(model, probe)
    split = trained.shape[0] - probe.header_len - probe.target_len  # positions before the final header
    stats = {
        "effect": max_abs(trained[split:], base[split:]),
        "repeat_err": repeat_err,
        "pre_header_diff": max_abs(trained[:split], base[:split]),
        "pre_header_tol": bf16_tolerance(base[:split]),
        "probe": {"task_id": probe.task_id, "adapted_positions": trained.shape[0] - split,
                  "pre_header_positions": split},
    }
    return trained[split:].to("cpu", torch.bfloat16), stats


# --- Run -----------------------------------------------------------------------------------------


def model_revision(model_name: str) -> str | None:
    """The cached Hugging Face snapshot (commit hash) the model was loaded from; None for a local directory."""
    if Path(model_name).is_dir():
        return None
    from huggingface_hub import try_to_load_from_cache

    path = try_to_load_from_cache(model_name, "config.json")
    return Path(path).parent.name if isinstance(path, str) else None


def git_state() -> dict:
    def run(*cmd: str) -> str:
        return subprocess.run(cmd, cwd=_ROOT, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"commit": None, "dirty": None, "error": str(exc)}


def train_and_save(args, rank: int, lora_config: LoraConfig, examples: Sequence[Example],
                   windows: Sequence[int], out_dir: Path, record: dict) -> torch.Tensor:
    """Train, measure the adapter's effect, save it; returns the probe logits (CPU).

    Every GPU object is local here, so it is released when this returns.
    """
    base = load_base(args.model, args.device)
    record["model_revision"] = model_revision(args.model)
    model = get_peft_model(base, lora_config, adapter_name=TRAIN_ADAPTER, autocast_adapter_dtype=True)
    params = trainable_adapter_params(model, TRAIN_ADAPTER, rank)
    record["trainable_params"] = sum(p.numel() for p in params)
    initial = [p.detach().to("cpu", copy=True) for p in params]
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    with (out_dir / "train_log.jsonl").open("w", encoding="utf-8") as log_file:
        history = train(model, params, examples, windows, args, log_file)
    record["updates"], record["microsteps"] = len(history), history[-1]["microsteps_total"]
    record["loss_first"], record["loss_last"] = history[0]["loss"], history[-1]["loss"]
    record["grad_norm_min"] = min(h["grad_norm"] for h in history)
    record["grad_norm_max"] = max(h["grad_norm"] for h in history)
    changed = [not torch.equal(p.detach().cpu(), p0) for p, p0 in zip(params, initial)]
    record["param_delta_l2"] = math.sqrt(sum((p.detach().cpu() - p0).pow(2).sum().item()
                                             for p, p0 in zip(params, initial)))
    record["changed_tensor_fraction"] = sum(changed) / len(changed)

    model.gradient_checkpointing_disable()  # with it on, a forward without backward raises in PEFT
    model.eval()
    adapted, stats = measure_adapter(model, examples[0])
    record.update(stats)
    model.save_pretrained(out_dir)
    record["peak_gpu_mb"] = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else None
    return adapted


def reload_logits(args, name: str, out_dir: Path, probe: Example) -> torch.Tensor:
    """The adapted positions' logits from the saved adapter, loaded into a fresh base model."""
    base = load_base(args.model, args.device)
    model = PeftModel.from_pretrained(base, str(out_dir), adapter_name=name, is_trainable=False)
    model.eval()
    return probe_logits(model, probe)[-(probe.header_len + probe.target_len):].to("cpu")


def release_gpu() -> float | None:
    gc.collect()
    if not torch.cuda.is_available():
        return None
    torch.cuda.empty_cache()
    return torch.cuda.memory_allocated() / 2**20


def check_gates(record: dict, kind: str) -> list[str]:
    """Correctness gates for every run. Finite loss and non-zero, finite gradients are enforced in train().

    The adapter's effect is recorded but not gated: a short or early run can legitimately move the logits
    very little. The smoke checks compare it with the measured noise instead.
    """
    failures = []
    if not record["param_delta_l2"] > 0:
        failures.append("adapter parameters did not change")
    if not record["reload_err"] <= record["reload_tol"]:
        failures.append(f"reload error {record['reload_err']:.3g} > tolerance {record['reload_tol']:.3g}")
    if kind == "alora" and not record["pre_header_diff"] <= record["pre_header_tol"]:
        failures.append(f"aLoRA changed logits before its header: {record['pre_header_diff']:.3g} "
                        f"> tolerance {record['pre_header_tol']:.3g}")
    return failures


def run(args, name: str, rank: int, alpha: int, out_dir: Path, tokenizer) -> dict:
    lora_config = make_lora_config(args.kind, args.role, rank, alpha, tokenizer)
    examples, data_report = load_examples(
        [Path(p) for p in args.data], tokenizer, role_filter=args.role, max_length=args.max_length,
        lora_config=lora_config)
    log.info("rows used %d, rejected %s", data_report["used"], {k: v for k, v in data_report["rejected"].items() if v})
    if not examples:
        raise TrainingError("no usable rows")
    data_report["revise_rows"] = sum(example.revise for example in examples)
    examples = oversample_revise(examples, args.oversample_revise)
    data_report["rows_after_oversampling"] = len(examples)
    if args.oversample_revise > 1:
        log.info("oversampled %d REVISE rows x%d: %d rows per pass", data_report["revise_rows"],
                 args.oversample_revise, len(examples))
    windows = plan_windows(len(examples), epochs=args.epochs, grad_accum=args.grad_accum, max_steps=args.max_steps)
    record = {
        "status": "running", "adapter": name, "kind": args.kind, "role": args.role,
        "model": args.model, "tokenizer": args.model,
        "invocation_tokens": lora_config.alora_invocation_tokens,
        "hyperparameters": {
            "rank": rank, "alpha": alpha, "dropout": LORA_DROPOUT, "lr": args.lr, "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW(betas=(0.9, 0.999), eps=1e-8)", "grad_clip": None,
            "grad_accum": args.grad_accum, "epochs": args.epochs, "max_steps": args.max_steps,
            "epochs_ignored": args.max_steps is not None, "warmup_fraction": WARMUP_FRACTION,
            "warmup_updates": warmup_updates(len(windows)), "schedule": "linear warm-up, then linear decay",
            "loss_averaging": "mean over each row's target tokens, then mean over the window's rows",
            "oversample_revise": args.oversample_revise,
            "microbatch": 1, "packing": False, "max_length": args.max_length,
            "target_modules": list(TARGET_MODULES), "seed": args.seed,
            "precision": "bf16 base, fp32 adapter, no autocast",
            "gradient_checkpointing": "non-reentrant",
        },
        "data": data_report, "planned_updates": len(windows), "planned_microsteps": sum(windows),
        "git": git_state(), "device": args.device,
        "gpu": torch.cuda.get_device_name(0) if args.device.startswith("cuda") else None,
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
    }
    start = time.perf_counter()
    try:
        if args.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        adapted = train_and_save(args, rank, lora_config, examples, windows, out_dir, record)
        record["train_wall_s"] = round(time.perf_counter() - start, 1)
        record["gpu_mb_after_release"] = release_gpu()
        reloaded = reload_logits(args, name, out_dir, examples[0])
        record["reload_err"] = max_abs(reloaded.float(), adapted.float())
        record["reload_tol"] = bf16_tolerance(adapted.float())
        # noise_floor is conservative: the larger of the measured errors and the acceptance bound.
        record["measured_err"] = max(record["repeat_err"], record["reload_err"])
        record["noise_floor"] = max(record["measured_err"], record["reload_tol"])
        if record["effect"] <= record["noise_floor"]:
            log.warning("adapter effect %.3g is within numerical noise %.3g on the probe row",
                        record["effect"], record["noise_floor"])
        failures = check_gates(record, args.kind)
        record["gate_failures"] = failures
        record["status"] = "failed" if failures else "passed"
    except Exception as exc:
        record["status"], record["error"] = "failed", f"{type(exc).__name__}: {exc}"
        raise
    finally:
        record["wall_s"] = round(time.perf_counter() - start, 1)
        (out_dir / "train_run.json").write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    if failures:
        raise TrainingError("; ".join(failures))
    return record


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--role", required=True, choices=["coder", "reviewer", "all"])
    parser.add_argument("--kind", required=True, choices=["lora", "alora"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, nargs="+")
    parser.add_argument("--out", required=True, help="adapter directory; the adapter goes to {out}/{kind}_{role}/")
    parser.add_argument("--rank", type=int, help="default 16 for lora, 32 for alora")
    parser.add_argument("--alpha", type=int, help="default 2 * rank")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--max-steps", type=int, help="optimizer updates; overrides --epochs")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--oversample-revise", type=int, default=1, metavar="N",
                        help="use each reviewer row whose target starts with REVISE N times per pass (default 1)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    if args.role == "all" and args.kind == "alora":
        parser.error("--role all needs --kind lora: an aLoRA is triggered by one role's header")
    for flag in ("rank", "alpha", "epochs", "grad_accum", "max_steps", "max_length", "lr"):
        value = getattr(args, flag)
        if value is not None and not value > 0:  # `not >` also rejects a NaN learning rate
            parser.error(f"--{flag.replace('_', '-')} must be positive")
    if args.seed < 0:
        parser.error("--seed must be >= 0")
    if args.oversample_revise < 1:
        parser.error("--oversample-revise must be >= 1")
    if args.oversample_revise > 1 and args.role == "coder":
        parser.error("--oversample-revise applies to reviewer rows; --role coder has none")
    missing = [p for p in args.data if not Path(p).is_file()]
    if missing:
        parser.error(f"data file not found: {missing}")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
    args = parse_args(argv)
    name = f"{args.kind}_{args.role}"
    rank = args.rank or DEFAULT_RANK[args.kind]
    alpha = args.alpha or 2 * rank
    out_dir = Path(args.out) / name
    if out_dir.exists():
        log.error("%s already exists; choose another --out (earlier runs are never overwritten)", out_dir)
        return 1
    if args.device.startswith("cuda") and not (torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        log.error("CUDA with bf16 is not available; pass --device cpu explicitly to run on the CPU")
        return 1
    torch.manual_seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    out_dir.mkdir(parents=True)
    try:
        record = run(args, name, rank, alpha, out_dir, tokenizer)
    except TrainingError as exc:
        log.error("run failed: %s (see %s)", exc, out_dir / "train_run.json")
        return 1
    log.info("passed: %s  loss %.4f -> %.4f  effect %.3g (noise %.3g)  reload_err %.3g  peak %.0f MB  %.1fs",
             out_dir, record["loss_first"], record["loss_last"], record["effect"], record["noise_floor"],
             record["reload_err"], record["peak_gpu_mb"] or 0, record["wall_s"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
