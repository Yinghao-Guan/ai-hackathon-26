"""Cost sweep: how much each method computes per session as the shared context grows.

Every session replays the same forced 4-turn conversation (buggy code, REVISE,
fixed code, APPROVE), so token counts depend only on the method and the context
length, not on the model, the adapters or the machine. Untrained adapters are
therefore enough: token counts are exact anywhere, and timings are meaningful
on the 4090.

    uv run python bench/cost_sweep.py --device cuda                 # 4090: 1.5B, bf16, timings
    uv run python bench/cost_sweep.py --device mps --model dev      # Mac: exact token counts
    uv run python bench/cost_sweep.py --summarize bench/results/cost_sweep_4090.jsonl

Writes the CONTRACT.md event log to --out and a markdown summary next to it.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

from rolekv import roles
from rolekv.costs import EventLog
from rolekv.runtime import (METHODS, ROLE_ADAPTERS, TARGET_MODULES, Engine, cast_adapters, load_model,
                            run_session, warmup)

TASK_TEXT = (
    "Write a function moving_average(values, window) that returns the list of averages "
    "of every contiguous window of the given size."
)
VISIBLE_TEST = "assert moving_average([1, 2, 3, 4], 2) == [1.5, 2.5, 3.5]"
BUGGY = '''```python
def moving_average(values, window):
    """Return the averages of every contiguous window of `window` values."""
    if window <= 0:
        raise ValueError("window must be positive")
    averages = []
    for i in range(len(values) - window):
        chunk = values[i:i + window]
        averages.append(sum(chunk) / window)
    return averages
```'''
FIXED = BUGGY.replace("range(len(values) - window)", "range(len(values) - window + 1)")
REVISE = (
    "REVISE: the loop stops one window early, so the last average is missing. "
    "Iterate over range(len(values) - window + 1)."
)
FORCED_TURNS = [BUGGY, REVISE, FIXED, "APPROVE"]
TOOL_OUTPUTS = [
    'Traceback (most recent call last):\n  File "test_visible.py", line 3, in <module>\n'
    f"    {VISIBLE_TEST}\nAssertionError",
    "1 passed",
]

# Filler for the shared "repository file": plausible small functions, numbered so none repeats.
_FILLER = [
    'def scale_{i}(values, factor={k}):\n    """Multiply every value by a fixed factor."""\n'
    "    return [v * factor for v in values]\n",
    'def count_{i}(items, target):\n    """Count how many items equal the target."""\n    total = 0\n'
    "    for item in items:\n        if item == target:\n            total += 1\n    return total\n",
    'def clamp_{i}(x, low=-{k}, high={k}):\n    """Clamp x into the closed interval [low, high]."""\n'
    "    if x < low:\n        return low\n    if x > high:\n        return high\n    return x\n",
    'def join_{i}(words, sep="-"):\n    """Join the non-empty words with a separator."""\n'
    "    return sep.join(w for w in words if w)\n",
]


def build_repo(tokenizer, target_tokens: int) -> str:
    """Deterministic filler code of at least `target_tokens` tokens."""
    functions: list[str] = []
    while True:
        repo = "\n\n".join(functions)
        if len(roles.segment_ids(tokenizer, repo)) >= target_tokens:
            return repo
        i = len(functions)
        functions.append(_FILLER[i % len(_FILLER)].format(i=i, k=i % 9 + 2))


def run_sweep(
    model,
    tokenizer,
    *,
    methods: Sequence[str],
    ctx_sizes: Sequence[int],
    repeats: int,
    log: EventLog,
    machine: str,
    run_prefix: str = "sweep",
    turns: Sequence[str] = FORCED_TURNS,
    tool_outputs: Sequence[str] = TOOL_OUTPUTS,
) -> list[dict]:
    """Run every (context size, method) `repeats` times, alternating methods, and log each session."""
    engines = {method: Engine(model, tokenizer, method) for method in methods}
    for engine in engines.values():
        warmup(engine)
    tasks = {size: roles.task_segment(build_repo(tokenizer, size), TASK_TEXT, VISIBLE_TEST) for size in ctx_sizes}
    records: list[dict] = []
    for rep in range(repeats):
        for size, task in tasks.items():
            for method, engine in engines.items():
                outputs = iter(tool_outputs)
                result = run_session(engine, task, lambda code: next(outputs), forced_outputs=turns)
                # One run_id per method and size, "sweep-A2-ctx4k": the dashboard reads the size from it
                # and keeps sweep runs out of quality numbers. `passed` only means the replay completed.
                for record in result.records(run_id=f"{run_prefix}-{method}-ctx{size / 1000:g}k", machine=machine,
                                             task_id=f"ctx{size}-r{rep}", passed=True):
                    log.write(record)
                    records.append(record)
        print(f"repeat {rep + 1}/{repeats} done", file=sys.stderr)
    return records


def summarize_records(records: Iterable[dict]) -> list[dict]:
    """One row per (context length, method): medians over repeats, plus ratios against A2."""
    tasks, turns = [], defaultdict(list)
    for r in records:
        if r["type"] == "task":
            tasks.append(r)
        else:
            turns[(r["run_id"], r["task_id"])].append(r)

    groups = defaultdict(list)
    for task in tasks:
        groups[(task["ctx_tokens"], task["method"])].append(task)

    rows = []
    for (ctx, method), group in sorted(groups.items(), key=lambda kv: (kv[0][0], METHODS.index(kv[0][1]))):
        session_turns = [t for task in group for t in turns[(task["run_id"], task["task_id"])]]
        switch_ttft = [t["ttft_ms"] for t in session_turns if t["turn"] >= 1]
        decode = [(t["turn_ms"] - t["ttft_ms"]) / (t["decode_tokens"] - 1) for t in session_turns
                  if t["decode_tokens"] > 1]
        computed = {task["prefill_tokens_total"] for task in group}
        if len(computed) != 1:
            raise ValueError(f"token counts differ across repeats for ctx={ctx} {method}: {sorted(computed)}")
        rows.append({
            "ctx_tokens": ctx,
            "method": method,
            "machine": group[0]["machine"],
            "repeats": len(group),
            "computed_tokens": computed.pop(),
            "canon_tokens": group[0]["canon_tokens_total"],
            "switch_ttft_ms": statistics.median(switch_ttft),
            "gpu_s": statistics.median(task["gpu_s"] for task in group),
            "decode_ms_per_token": statistics.median(decode),
        })

    by_key = {(row["ctx_tokens"], row["method"]): row for row in rows}
    for row in rows:
        a2 = by_key.get((row["ctx_tokens"], "A2"))
        row["compute_vs_a2"] = a2["computed_tokens"] / row["computed_tokens"] if a2 else None
        row["ttft_vs_a2"] = a2["switch_ttft_ms"] / row["switch_ttft_ms"] if a2 else None
    return rows


def format_table(rows: Sequence[dict], *, timings: bool = True) -> str:
    def ratio(x):
        return "–" if x is None else f"{x:.2f}×"

    head = ["Context tokens", "Method", "Tokens computed", "of which canon", "A2 ÷ method"]
    if timings:
        head += ["Switch TTFT (ms)", "A2 ÷ method", "GPU s / session", "Decode ms / token"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        cells = [f"{r['ctx_tokens']:,}", r["method"], f"{r['computed_tokens']:,}", f"{r['canon_tokens']:,}",
                 ratio(r["compute_vs_a2"])]
        if timings:
            cells += [f"{r['switch_ttft_ms']:.1f}", ratio(r["ttft_vs_a2"]), f"{r['gpu_s']:.2f}",
                      f"{r['decode_ms_per_token']:.1f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def model_with_random_adapters(base_model: str, methods: Sequence[str], device, dtype):
    """Base model plus untrained adapters at the planned ranks (LoRA r=16, aLoRA r=32).

    PEFT's default init makes the adapters output zero, but they still run their
    full computation, so costs match trained ones.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, attn_implementation="sdpa").to(device)
    names = sorted({name for m in methods for name in ROLE_ADAPTERS[m].values() if name})
    if not names:
        return model.eval(), tokenizer

    def config(name: str) -> LoraConfig:
        role = name.removeprefix("alora_") if name.startswith("alora_") else None
        rank = 32 if role else 16
        return LoraConfig(r=rank, lora_alpha=2 * rank, target_modules=TARGET_MODULES, task_type="CAUSAL_LM",
                          alora_invocation_tokens=roles.invocation_tokens(tokenizer, role) if role else None)

    model = get_peft_model(model, config(names[0]), adapter_name=names[0], autocast_adapter_dtype=False)
    for name in names[1:]:
        model.add_adapter(name, config(name), autocast_adapter_dtype=False)
    cast_adapters(model, dtype)  # adapters in the model dtype, as load_model does for trained ones
    return model.eval(), tokenizer


def default_machine(device: torch.device) -> str:
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        return next((gpu for gpu in ("4090", "4070") if gpu in name), "cuda")
    return "mac" if sys.platform == "darwin" else device.type


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=roles.BASE_MODEL,
                        help="Hugging Face id, or 'dev' for the 0.5B model (same tokenizer, so same token counts)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    parser.add_argument("--adapter-dir", help="trained adapters (CONTRACT.md section 5); default: untrained ones")
    parser.add_argument("--methods", default="A2,A5-H1,A5-H2", help="comma-separated, or 'all'")
    parser.add_argument("--ctx", default="1000,2000,4000,8000", help="comma-separated context sizes in tokens")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--machine", help="label for the log; default: detected (4090, 4070, mac, ...)")
    parser.add_argument("--out", type=Path, help="default: bench/results/cost_sweep_<machine>.jsonl")
    parser.add_argument("--overwrite", action="store_true", help="replace --out if it exists")
    parser.add_argument("--summarize", type=Path, metavar="JSONL", help="only print the summary of an existing log")
    args = parser.parse_args(argv)

    if args.summarize:
        print(format_table(summarize_records(read_jsonl(args.summarize))))
        return

    methods = list(METHODS) if args.methods == "all" else args.methods.split(",")
    if unknown := set(methods) - set(METHODS):
        parser.error(f"unknown methods {sorted(unknown)}; choose from {METHODS}")
    ctx_sizes = [int(size) for size in args.ctx.split(",")]
    base_model = roles.DEV_MODEL if args.model == "dev" else args.model
    device = torch.device(args.device)
    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    machine = args.machine or default_machine(device)
    out = args.out or Path(f"bench/results/cost_sweep_{machine}.jsonl")
    if out.exists() and not args.overwrite:
        parser.error(f"{out} exists; pass --overwrite to replace it")
    out.unlink(missing_ok=True)

    if args.adapter_dir:
        model, tokenizer = load_model(methods, base_model=base_model, adapter_dir=args.adapter_dir,
                                      device=device, dtype=dtype)
    else:
        model, tokenizer = model_with_random_adapters(base_model, methods, device, dtype)

    with EventLog(out) as log:
        records = run_sweep(model, tokenizer, methods=methods, ctx_sizes=ctx_sizes, repeats=args.repeats,
                            log=log, machine=machine)
    table = format_table(summarize_records(records))
    summary = (f"# Cost sweep: {machine}\n\nModel `{base_model}`, {args.dtype}, device `{device}`, "
               f"{args.repeats} repeats, adapters: {args.adapter_dir or 'untrained'}.\n"
               f"Every session replays the same forced 4-turn conversation.\n\n{table}\n")
    out.with_suffix(".md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
