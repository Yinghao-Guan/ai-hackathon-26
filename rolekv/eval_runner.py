"""Runner factory for Member B's eval driver (eval/run_eval.py).

    uv run python -m eval.run_eval --runner rolekv.eval_runner:make_runner \\
        --model Qwen/Qwen2.5-Coder-1.5B-Instruct --adapters adapters/1.5b \\
        --methods A2 A5-H1 --machine 4090 --run-id eval-1.5b --out runs/eval.jsonl

The driver builds the TASK segment, scores the final code on all MBPP asserts,
and writes the event log; this module runs the session.

The TOOL turn runs the same check that data generation used
(`eval.testrun.run_mbpp_task`: every MBPP assert), so the Reviewer sees tool
output in the format it was trained on. That also means the agent sees the
asserts it is finally scored on: absolute pass@1 is optimistic, but every
method gets the same information, so comparisons between methods are fair.
Pass tool_asserts="visible" to run only the first assert instead.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from typing import Any

import torch

from rolekv.runtime import Engine, load_model, run_session, warmup


def _default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EvalRunner:
    def __init__(self, engine: Engine, *, tool_asserts: str = "all",
                 max_new_tokens: Mapping[str, int] | None = None):
        if tool_asserts not in ("all", "visible"):
            raise ValueError("tool_asserts must be 'all' or 'visible'")
        self.engine = engine
        self.tool_asserts = tool_asserts
        self.max_new_tokens = max_new_tokens

    def run(self, *, task: Mapping[str, Any], task_segment: str) -> dict:
        from eval.testrun import run_code, run_mbpp_task

        def run_tool(code: str) -> str:
            if self.tool_asserts == "visible":
                return run_code(code, [task["test_list"][0]], setup_code=task.get("test_setup_code", "")).output
            return run_mbpp_task(code, task).output

        result = run_session(self.engine, task_segment, run_tool, max_new_tokens=self.max_new_tokens)
        return {
            "turns": [asdict(t.cost) for t in result.turns],
            "final_code": result.final_code,
            "ctx_tokens": result.ctx_tokens,
            "gpu_s": result.cost.gpu_s,
        }


def make_runner(*, method: str, model: str, adapters=None, device=None, dtype=None,
                tool_asserts: str = "all", max_new_tokens: Mapping[str, int] | None = None) -> EvalRunner:
    """Load the model and the adapters `method` needs, warm up, and return a runner."""
    device = torch.device(device) if device is not None else _default_device()
    if dtype is None:
        dtype = torch.float32 if device.type == "cpu" else torch.bfloat16
    loaded, tokenizer = load_model([method], base_model=model, adapter_dir=adapters, device=device, dtype=dtype)
    engine = Engine(loaded, tokenizer, method)
    warmup(engine)
    return EvalRunner(engine, tool_asserts=tool_asserts, max_new_tokens=max_new_tokens)
