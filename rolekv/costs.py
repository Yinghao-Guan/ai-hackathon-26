"""Cost counters, timing and the JSONL event log (docs/CONTRACT.md section 3)."""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

MB = 2**20


@dataclass
class TurnCost:
    role: str
    prefill_tokens: int  # new tokens computed before the first decode step (excludes reused and canon)
    reused_tokens: int  # cache length at turn start, after any crop or reset
    canon_tokens: int  # previous turn's adapter KV recomputed with base weights in this turn (A5-H1)
    decode_tokens: int  # generated tokens, including the stop token
    ttft_ms: float  # turn start -> first generated token
    turn_ms: float
    kv_mb: float  # logical KV cache size at the end of the turn
    text: str
    truncated: bool = False  # hit max_new_tokens without a stop token
    verdict: str | None = None  # reviewer turns: APPROVE / REVISE / INVALID


@dataclass
class TaskCost:
    n_turns: int
    prefill_tokens_total: int  # sum of prefill + canon: every token computed outside decoding
    canon_tokens_total: int
    decode_tokens_total: int
    gpu_s: float
    peak_kv_mb: float
    ctx_tokens: int


def summarize(turns: Sequence[TurnCost], ctx_tokens: int) -> TaskCost:
    return TaskCost(
        n_turns=len(turns),
        prefill_tokens_total=sum(t.prefill_tokens + t.canon_tokens for t in turns),
        canon_tokens_total=sum(t.canon_tokens for t in turns),
        decode_tokens_total=sum(t.decode_tokens for t in turns),
        gpu_s=sum(t.turn_ms for t in turns) / 1000,
        peak_kv_mb=max((t.kv_mb for t in turns), default=0.0),
        ctx_tokens=ctx_tokens,
    )


class Clock:
    """Wall clock that waits for queued GPU work first, so timings cover the kernels themselves."""

    def __init__(self, device: torch.device | str):
        self.device = torch.device(device)

    def now(self) -> float:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()
        return time.perf_counter()


def kv_bytes(cache) -> int:
    """Logical size of all key/value tensors in a DynamicCache.

    A cropped cache still holds its old storage until the next update; this
    counts only the live tokens, the same way for every method.
    """
    return sum(layer.keys.nbytes + layer.values.nbytes for layer in cache.layers if layer.keys is not None)


def kv_bytes_per_token(config, dtype: torch.dtype) -> int:
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    itemsize = torch.empty((), dtype=dtype).element_size()
    return 2 * config.num_hidden_layers * config.num_key_value_heads * head_dim * itemsize


def turn_record(cost: TurnCost, *, run_id: str, machine: str, method: str, task_id, turn: int) -> dict:
    return {"type": "turn", "run_id": run_id, "machine": machine, "method": method, "task_id": task_id,
            "turn": turn, **asdict(cost)}


def task_record(cost: TaskCost, *, run_id: str, machine: str, method: str, task_id, passed: bool) -> dict:
    return {"type": "task", "run_id": run_id, "machine": machine, "method": method, "task_id": task_id,
            "passed": passed, **asdict(cost)}


class EventLog:
    """Append-only JSONL writer; each record is flushed so a crashed run keeps what it logged."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, record: dict) -> None:
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> EventLog:
        return self

    def __exit__(self, *exc) -> None:
        self.close()
