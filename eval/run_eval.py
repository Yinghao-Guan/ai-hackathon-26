"""Evaluate RoleKV runtime methods and emit contract-compatible JSONL logs.

The runtime owner supplies a runner factory with this small interface:

``factory(method=..., model=..., adapters=...) -> runner``
``runner.run(task=..., task_segment=...) -> {"turns": [...], "final_code": "..."}``

Each returned turn supplies the per-turn fields from CONTRACT.md except the
fields added here (run ID, method, task ID, and turn index). This keeps test
scoring in Member B's shared harness while leaving cache mechanics in the
runtime module.
"""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Iterable, Mapping

from eval.testrun import run_mbpp_task
from rolekv.tasks import build_task_segment, load_mbpp_splits


METHODS = ("A0", "A1", "A2", "A3", "A5-H1", "A5-H2")
TURN_FIELDS = {"role", "prefill_tokens", "reused_tokens", "canon_tokens", "decode_tokens", "ttft_ms", "turn_ms", "kv_mb", "text"}


def _resolve_factory(spec: str) -> Callable[..., Any]:
    try:
        module_name, attribute = spec.split(":", maxsplit=1)
        factory = getattr(importlib.import_module(module_name), attribute)
    except (ValueError, ModuleNotFoundError, AttributeError) as exc:
        raise ValueError(f"runner must be specified as module:factory, got {spec!r}") from exc
    if not callable(factory):
        raise ValueError("runner factory must be callable")
    return factory


def _require_turn_fields(turn: Mapping[str, Any]) -> None:
    missing = TURN_FIELDS - set(turn)
    if missing:
        raise ValueError(f"runtime turn is missing fields: {', '.join(sorted(missing))}")
    if turn["role"] not in {"coder", "reviewer"}:
        raise ValueError("runtime turn role must be coder or reviewer")


def _write_jsonl_line(handle: Any, row: Mapping[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()


def evaluate(
    *,
    tasks: Iterable[Mapping[str, Any]],
    training_rows: Iterable[Mapping[str, Any]],
    runner: Any,
    method: str,
    run_id: str,
    machine: str,
    output_path: Path,
    ctx_funcs: int,
) -> int:
    """Run one method over tasks and return the number of emitted task rows."""

    if method not in METHODS:
        raise ValueError(f"unsupported method: {method}")
    training_rows = list(training_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for task in tasks:
            task_segment = build_task_segment(task, training_rows, ctx_funcs=ctx_funcs)
            started = time.monotonic()
            session = runner.run(task=task, task_segment=task_segment)
            turns = session.get("turns", [])
            if not turns or "final_code" not in session:
                raise ValueError("runtime session must contain non-empty turns and final_code")

            normalized_turns = []
            for index, turn in enumerate(turns):
                _require_turn_fields(turn)
                row = {
                    "type": "turn",
                    "run_id": run_id,
                    "machine": machine,
                    "method": method,
                    "task_id": task["task_id"],
                    "turn": index,
                    **dict(turn),
                }
                normalized_turns.append(row)
                _write_jsonl_line(handle, row)

            test_result = run_mbpp_task(session["final_code"], task)
            task_row = {
                "type": "task",
                "run_id": run_id,
                "machine": machine,
                "method": method,
                "task_id": task["task_id"],
                "ctx_tokens": session["ctx_tokens"],
                "passed": test_result.passed,
                "n_turns": len(normalized_turns),
                "prefill_tokens_total": sum(turn["prefill_tokens"] + turn["canon_tokens"] for turn in normalized_turns),
                "canon_tokens_total": sum(turn["canon_tokens"] for turn in normalized_turns),
                "decode_tokens_total": sum(turn["decode_tokens"] for turn in normalized_turns),
                "gpu_s": session.get("gpu_s", time.monotonic() - started),
                "peak_kv_mb": max(turn["kv_mb"] for turn in normalized_turns),
            }
            _write_jsonl_line(handle, task_row)
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runner", required=True, help="Runtime factory as module:factory.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--run-id", required=True,
                        help="prefix; each method is logged as <run-id>-<method>, since a run_id has one method")
    parser.add_argument("--machine", required=True, choices=("4090", "4070", "mac"))
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapters", type=Path)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--ctx-funcs", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.limit <= 100:
        raise SystemExit("--limit must be between 1 and 100")
    factory = _resolve_factory(args.runner)
    splits = load_mbpp_splits()
    tasks = [splits.evaluation[index] for index in range(args.limit)]
    for method in args.methods:
        runner = factory(method=method, model=args.model, adapters=args.adapters)
        emitted = evaluate(
            tasks=tasks,
            training_rows=splits.training,
            runner=runner,
            method=method,
            run_id=f"{args.run_id}-{method}",  # the dashboard requires one method per run_id
            machine=args.machine,
            output_path=args.out,
            ctx_funcs=args.ctx_funcs,
        )
        print(f"{method}: wrote {emitted} task rows to {args.out}")


if __name__ == "__main__":
    main()
