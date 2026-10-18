"""Role-switching engine for methods A0-A5 and the Coder/Reviewer session protocol.

Method policies (docs/PLAN.md has the full table):

| Method | At turn start         | Adapter for the turn | Adapter active from |
|--------|-----------------------|----------------------|---------------------|
| A0     | -                     | none (base)          | -                   |
| A1     | -                     | lora_all             | every token         |
| A2     | drop the whole cache  | lora_<role>          | every token         |
| A3     | -                     | lora_<role>          | every token fed now |
| A5-H1  | crop the last turn's  | alora_<role>         | the role header     |
|        | adapter KV (canon)    |                      |                     |
| A5-H2  | -                     | alora_<role>         | the role header     |

A5-H1 canonicalizes lazily: the previous turn's adapter KV is cropped at the
start of the next turn and recomputed with base weights in the same forward as
the new tokens, so that turn's TTFT includes it and the last turn never pays it.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

from rolekv import roles
from rolekv.cache import KVSession, is_alora
from rolekv.costs import MB, Clock, TaskCost, TurnCost, summarize, task_record, turn_record

METHODS = ("A0", "A1", "A2", "A3", "A5-H1", "A5-H2")
ROLE_ADAPTERS = {
    "A0": {"coder": None, "reviewer": None},
    "A1": {"coder": "lora_all", "reviewer": "lora_all"},
    "A2": {"coder": "lora_coder", "reviewer": "lora_reviewer"},
    "A3": {"coder": "lora_coder", "reviewer": "lora_reviewer"},
    "A5-H1": {"coder": "alora_coder", "reviewer": "alora_reviewer"},
    "A5-H2": {"coder": "alora_coder", "reviewer": "alora_reviewer"},
}
MAX_NEW_TOKENS = {"coder": 512, "reviewer": 128}
# The only modules adapters may target (CONTRACT.md section 5).
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass
class TurnResult:
    role: str
    text: str  # decoded output, without the stop token
    gen_ids: list[int]  # generated ids, including the stop token if there was one
    header_pos: int  # where this turn's role header starts in the session's ids
    cost: TurnCost


class Engine:
    """Runs one session at a time for one method. Not safe to share between threads."""

    def __init__(self, model, tokenizer, method: str):
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        self.model = model
        self.tokenizer = tokenizer
        self.method = method
        self.adapters = ROLE_ADAPTERS[method]
        self.clock = Clock(next(model.parameters()).device)
        self.im_end = tokenizer.convert_tokens_to_ids(roles.IM_END)
        self.stop_ids = {self.im_end, tokenizer.convert_tokens_to_ids("<|endoftext|>")}
        self.newline = roles.segment_ids(tokenizer, "\n")
        self.headers = {role: roles.invocation_tokens(tokenizer, role) for role in roles.ROLES}
        self.session: KVSession | None = None
        self.ctx_tokens = 0
        self._adapter_span: int | None = None  # A5-H1: where the last turn's adapter KV starts

    def start(self, task_segment: str, system_segment: str | None = None) -> None:
        """Begin a new session with a fresh cache."""
        system = roles.system_segment() if system_segment is None else system_segment
        task_ids = roles.segment_ids(self.tokenizer, task_segment)
        self.session = KVSession(self.model)
        self.session.extend(roles.segment_ids(self.tokenizer, system) + task_ids)
        self.ctx_tokens = len(task_ids)
        self._adapter_span = None

    def add_tool_output(self, output: str) -> None:
        self.session.extend(roles.segment_ids(self.tokenizer, roles.tool_segment(output)))

    def turn(self, role: str, max_new_tokens: int | None = None, forced: Sequence[int] | None = None) -> TurnResult:
        """Run one role turn: prefill everything pending plus the role header, then decode greedily.

        `forced` replaces each argmax with the given token (the argmax is still
        computed), which makes token counts deterministic for tests and sweeps.
        """
        s = self.session
        if s is None:
            raise RuntimeError("call start() before turn()")
        if forced is not None:
            max_new_tokens = len(forced)
        elif max_new_tokens is None:
            max_new_tokens = MAX_NEW_TOKENS[role]
        adapter = self.adapters[role]

        t0 = self.clock.now()
        if self.method == "A2":
            s.reset()
        canon = 0
        if self.method == "A5-H1" and self._adapter_span is not None:
            canon = s.crop_to(self._adapter_span)
            self._adapter_span = None
        reused = s.cache_len
        header_pos = len(s.ids)
        s.extend(self.headers[role])
        prefill = len(s.pending) - canon
        alora_start = header_pos if is_alora(self.model, adapter) else None

        gen: list[int] = []
        with s.adapter(adapter):
            logits = s.prefill(alora_start=alora_start)
            while True:
                choice = int(logits[-1].argmax())
                token = forced[len(gen)] if forced is not None else choice
                gen.append(token)
                s.extend([token])  # in ids now; fed to the model on the next step, if there is one
                if len(gen) == 1:
                    first_token = self.clock.now()
                if token in self.stop_ids or len(gen) >= max_new_tokens:
                    break
                logits = s.prefill(alora_start=alora_start)

        # Close the turn in contract format: BODY ends with <|im_end|>\n. The stop
        # token was never fed, so normalizing it costs nothing.
        truncated = gen[-1] not in self.stop_ids
        if truncated:
            s.extend([self.im_end])
        else:
            s.ids[-1] = self.im_end
        s.extend(self.newline)
        if self.method == "A5-H1":
            self._adapter_span = header_pos
        end = self.clock.now()

        text = self.tokenizer.decode(gen if truncated else gen[:-1])
        cost = TurnCost(
            role=role,
            prefill_tokens=prefill,
            reused_tokens=reused,
            canon_tokens=canon,
            decode_tokens=len(gen),
            ttft_ms=(first_token - t0) * 1000,
            turn_ms=(end - t0) * 1000,
            kv_mb=s.kv_bytes() / MB,
            text=text,
            truncated=truncated,
        )
        return TurnResult(role=role, text=text, gen_ids=gen, header_pos=header_pos, cost=cost)

    def canonicalize(self) -> int:
        """Apply the pending H1 recompute now instead of at the next turn (tests and debugging)."""
        if self._adapter_span is None:
            return 0
        s = self.session
        dropped = s.crop_to(self._adapter_span)
        self._adapter_span = None
        with s.adapter(None):
            s.prefill()
        return dropped


@dataclass
class SessionResult:
    method: str
    turns: list[TurnResult]
    final_code: str
    verdicts: list[str]
    ctx_tokens: int

    @property
    def cost(self) -> TaskCost:
        return summarize([t.cost for t in self.turns], self.ctx_tokens)

    def records(self, *, run_id: str, machine: str, task_id, passed: bool) -> list[dict]:
        """Event log lines for this session (CONTRACT.md section 3): one per turn, then the task line."""
        keys = {"run_id": run_id, "machine": machine, "method": self.method, "task_id": task_id}
        rows = [turn_record(t.cost, turn=i, **keys) for i, t in enumerate(self.turns)]
        rows.append(task_record(self.cost, passed=passed, **keys))
        return rows


def extract_code(text: str) -> str:
    """The Coder's code: the first fenced block, else the raw text minus an unclosed opening fence."""
    code = roles.parse_coder_output(text)
    if code is not None:
        return code
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    return "\n".join(lines)


def forced_ids(tokenizer, text: str) -> list[int]:
    """Token ids that make a forced turn produce `text` and stop, as the BODY segment tokenizes it."""
    return roles.segment_ids(tokenizer, text) + [tokenizer.convert_tokens_to_ids(roles.IM_END)]


def run_session(
    engine: Engine,
    task_segment: str,
    run_visible_test: Callable[[str], str],
    *,
    max_revise: int = 2,
    max_new_tokens: Mapping[str, int] | None = None,
    forced_outputs: Sequence[str] | None = None,
) -> SessionResult:
    """Coder -> visible test -> Reviewer, until APPROVE, an invalid verdict, or `max_revise` REVISEs.

    After the last allowed REVISE the Coder's next turn is final: no test run
    and no review, so a session has at most 2 * max_revise + 1 turns.
    `run_visible_test(code)` returns the raw stdout + stderr; truncation is
    applied here, through the TOOL segment.
    """
    engine.start(task_segment)
    outputs = iter(forced_outputs) if forced_outputs is not None else None
    turns: list[TurnResult] = []
    verdicts: list[str] = []

    def take(role: str) -> TurnResult:
        forced = forced_ids(engine.tokenizer, next(outputs)) if outputs is not None else None
        limit = max_new_tokens.get(role) if max_new_tokens else None
        result = engine.turn(role, max_new_tokens=limit, forced=forced)
        turns.append(result)
        return result

    revisions = 0
    while True:
        code = extract_code(take("coder").text)
        if revisions == max_revise:
            break
        engine.add_tool_output(run_visible_test(code))
        review = take("reviewer")
        verdict, _ = roles.parse_reviewer_verdict(review.text)
        review.cost.verdict = verdict
        verdicts.append(verdict)
        if verdict != "REVISE":
            break
        revisions += 1
    return SessionResult(engine.method, turns, code, verdicts, engine.ctx_tokens)


def warmup(engine: Engine) -> None:
    """One short session, so first-call overheads stay out of measured runs."""
    engine.start(roles.task_segment("", "Write a function f() that returns 1.", "assert f() == 1"))
    engine.turn("coder", max_new_tokens=4)
    engine.add_tool_output("1 passed")
    engine.turn("reviewer", max_new_tokens=2)


def load_model(
    methods: Iterable[str],
    *,
    base_model: str = roles.BASE_MODEL,
    adapter_dir: str | Path | None = None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
):
    """Load the base model plus every adapter the given methods need, ready for inference.

    Adapters live at {adapter_dir}/{alora_coder|alora_reviewer|lora_coder|lora_reviewer|lora_all}/,
    as written by save_pretrained. Returns (model, tokenizer).
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(base_model, dtype=dtype, attn_implementation="sdpa").to(device)
    if getattr(model.config, "use_sliding_window", False):
        raise ValueError("sliding-window attention is not supported: cache cropping assumes full attention")

    names = sorted({name for m in methods for name in ROLE_ADAPTERS[m].values() if name})
    if names:
        if adapter_dir is None:
            raise ValueError(f"methods {sorted(methods)} need adapters {names}; pass adapter_dir")
        root = Path(adapter_dir)
        model = PeftModel.from_pretrained(model, str(root / names[0]), adapter_name=names[0], is_trainable=False)
        for name in names[1:]:
            model.load_adapter(str(root / name), adapter_name=name, is_trainable=False)
        for name in names:
            tokens = list(getattr(model.peft_config[name], "alora_invocation_tokens", None) or [])
            if name.startswith("alora_"):
                expected = roles.invocation_tokens(tokenizer, name.removeprefix("alora_"))
                if tokens != expected:
                    raise ValueError(f"{name}: alora_invocation_tokens {tokens} != role header ids {expected}")
            elif tokens:
                raise ValueError(f"{name} is an aLoRA, but its name says it is a standard LoRA")

    base = model.get_base_model() if names else model
    if getattr(base, "is_gradient_checkpointing", False):
        base.gradient_checkpointing_disable()  # with it on, a second aLoRA forward raises in PEFT
    model.eval()
    return model, tokenizer
