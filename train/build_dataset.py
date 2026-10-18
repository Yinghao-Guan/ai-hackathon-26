"""Generate test-verified Coder/Reviewer SFT data from MBPP.

The script deliberately records the test runner's real output before asking the
Reviewer for feedback. It never asks the teacher to invent a tool result.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
import time
from typing import Any, Iterable, Mapping

from dotenv import load_dotenv

from eval.testrun import TestResult, run_mbpp_task
from rolekv.roles import (
    SYSTEM_PROMPT,
    body_segment,
    role_header,
    system_segment,
    tool_segment as format_tool_segment,
)
from rolekv.tasks import build_task_segment, load_mbpp_splits


DEFAULT_MODEL = "claude-sonnet-5-5"
CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def tool_segment(result: TestResult) -> str:
    return format_tool_segment(result.output)


def extract_python_code(text: str) -> str:
    """Extract exactly one Python fence from a Coder response."""

    matches = CODE_BLOCK_RE.findall(text)
    if len(matches) != 1:
        raise ValueError("Coder response must contain exactly one ```python code block")
    return matches[0].strip() + "\n"


def coder_body(code: str) -> str:
    """Return the contract-required Coder BODY: one Python fence only."""

    return body_segment(f"```python\n{code.strip()}\n```")


def _message_text(response: Any) -> str:
    text_parts = [block.text for block in response.content if getattr(block, "type", None) == "text"]
    if not text_parts:
        raise ValueError("teacher response contained no text")
    return "\n".join(text_parts).strip()


# USD per million tokens: (input, output). From the claude-api reference table cached 2026-10-06; check the live
# pricing page before a paid run. Prompt caching is not used here; if cache tokens ever appear they are charged
# conservatively, writes at 2x and reads at 1x the input price.
PRICES_PER_MTOK = {"claude-sonnet-5-5": (2.00, 10.00)}
# Upper bound on the request tokens that are not prompt text (role markers, message framing). Each prompt token
# covers at least one UTF-8 byte, so prompt bytes bound the prompt tokens.
REQUEST_OVERHEAD_TOKENS = 100


class SpendLimitReached(RuntimeError):
    """The next request could exceed the spend cap or the call cap, so it was not sent."""


class TeacherCallFailed(RuntimeError):
    """A metered teacher request failed after the client's bounded retries; it is logged and charged as reserved."""


def _call_purpose(prompt: str) -> str:
    for marker, purpose in (
        ("subtly buggy first attempt", "buggy"),
        ("still subtly wrong", "still_buggy_repair"),
        ("Return a corrected replacement", "repair"),
        ("Review this candidate", "review"),
        ("Solve the following task", "clean"),
    ):
        if marker in prompt:
            return purpose
    return "other"


class UsageLedger:
    """Logs every metered teacher request and refuses one whose worst-case cost could pass the caps.

    A metered client makes exactly one HTTP attempt per request (SDK retries are off, see Teacher), so every
    attempt is its own logged, charged record and a later success never hides an earlier, possibly billed,
    failure. Before a request it reserves the most it could cost: the prompt's byte-count upper bound on input
    tokens plus ``max_tokens`` of output (thinking tokens count against ``max_tokens``). A failed request is
    charged its full reservation, since whether it was billed is unknown. Each request is appended to a JSONL
    log as it happens.
    """

    def __init__(self, *, model: str, cap_usd: float, max_calls: int | None, log_path: Path) -> None:
        if model not in PRICES_PER_MTOK:
            raise ValueError(f"no price recorded for {model!r}; add it to PRICES_PER_MTOK before a metered run")
        if cap_usd <= 0 or (max_calls is not None and max_calls < 1):
            raise ValueError("cap_usd must be positive and max_calls at least 1")
        self.model = model
        self.input_price, self.output_price = (price / 1_000_000 for price in PRICES_PER_MTOK[model])
        self.cap_usd = cap_usd
        self.max_calls = max_calls
        self.log_path = log_path
        self.records: list[dict[str, Any]] = []
        self.current_task: int | None = None
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.open("x", encoding="utf-8").close()  # never overwrite an earlier run's log

    @property
    def spent_usd(self) -> float:
        return sum(record["charged_usd"] for record in self.records)

    def reservation(self, *, system: str, prompt: str, max_tokens: int) -> float:
        input_bound = len(system.encode("utf-8")) + len(prompt.encode("utf-8")) + REQUEST_OVERHEAD_TOKENS
        return input_bound * self.input_price + max_tokens * self.output_price

    def reserve(self, *, system: str, prompt: str, max_tokens: int) -> float:
        reserved = self.reservation(system=system, prompt=prompt, max_tokens=max_tokens)
        if self.max_calls is not None and len(self.records) >= self.max_calls:
            raise SpendLimitReached(f"call cap {self.max_calls} reached")
        if self.spent_usd + reserved > self.cap_usd:
            raise SpendLimitReached(
                f"next request could cost up to ${reserved:.4f}; ${self.spent_usd:.4f} of the ${self.cap_usd:.2f} cap is used"
            )
        return reserved

    def cost(self, usage: Any) -> float:
        tokens = {name: getattr(usage, name, None) or 0 for name in
                  ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")}
        return (
            (tokens["input_tokens"] + tokens["cache_read_input_tokens"] + 2 * tokens["cache_creation_input_tokens"])
            * self.input_price
            + tokens["output_tokens"] * self.output_price
        )

    def record(self, *, prompt: str, max_tokens: int, reserved: float, elapsed_s: float,
               response: Any = None, error: BaseException | None = None) -> None:
        usage = getattr(response, "usage", None)
        entry = {
            "call": len(self.records) + 1,
            "task_id": self.current_task,
            "purpose": _call_purpose(prompt),
            "model": self.model,
            "max_tokens": max_tokens,
            "request_id": getattr(response, "_request_id", None),
            "stop_reason": getattr(response, "stop_reason", None),
            "input_tokens": getattr(usage, "input_tokens", None),
            "output_tokens": getattr(usage, "output_tokens", None),
            "cache_creation_input_tokens": getattr(usage, "cache_creation_input_tokens", None),
            "cache_read_input_tokens": getattr(usage, "cache_read_input_tokens", None),
            "reserved_usd": reserved,
            "charged_usd": reserved if error is not None or usage is None else self.cost(usage),
            "error": None if error is None else f"{type(error).__name__}: {error}"[:500],
            "elapsed_s": round(elapsed_s, 3),
        }
        self.records.append(entry)
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(entry) + "\n")

    def summary(self, outcomes: Mapping[int, str]) -> dict[str, Any]:
        """Totals, with cost split by each task's outcome ('kept' or its rejection reason)."""

        by_outcome: Counter[str] = Counter()
        for entry in self.records:
            by_outcome[outcomes.get(entry["task_id"], "unfinished")] += entry["charged_usd"]
        return {
            "model": self.model,
            "price_per_mtok_usd": list(PRICES_PER_MTOK[self.model]),
            "cap_usd": self.cap_usd,
            "max_calls": self.max_calls,
            "http_attempts_per_call": 1,
            "calls": len(self.records),
            "errors": sum(1 for entry in self.records if entry["error"]),
            "input_tokens": sum(entry["input_tokens"] or 0 for entry in self.records),
            "output_tokens": sum(entry["output_tokens"] or 0 for entry in self.records),
            "charged_usd": round(self.spent_usd, 6),
            "stop_reasons": dict(sorted(Counter(str(entry["stop_reason"]) for entry in self.records).items())),
            "charged_usd_by_outcome": {key: round(value, 6) for key, value in sorted(by_outcome.items())},
        }


class Teacher:
    """Small synchronous wrapper around the Anthropic Messages API.

    With a ``ledger``, the client makes no automatic retries (``max_retries=0``): each request is one HTTP attempt
    that is reserved, logged and charged, and a failed request raises TeacherCallFailed instead of being retried
    behind the ledger's back. Without a ledger it behaves as it always has (the SDK's default retries).
    """

    def __init__(self, *, model: str, system_prompt: str, ledger: UsageLedger | None = None,
                 timeout: float | None = None, client: Any = None) -> None:
        if client is None:
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError("Install the 'anthropic' package to generate teacher data.") from exc
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("ANTHROPIC_API_KEY is required to generate teacher data.")
            options: dict[str, Any] = {}
            if ledger is not None:
                options["max_retries"] = 0
                if timeout is not None:
                    options["timeout"] = timeout
            client = Anthropic(**options)
        if ledger is not None and getattr(client, "max_retries", 0) != 0:
            raise ValueError("a metered Teacher needs a client with max_retries=0, so every attempt is accounted for")
        self._client = client
        self.model = model
        self.system_prompt = system_prompt
        self.ledger = ledger

    def ask(self, prompt: str, *, max_tokens: int) -> str:
        if self.ledger is None:
            return _message_text(self._create(prompt, max_tokens))
        reserved = self.ledger.reserve(system=self.system_prompt, prompt=prompt, max_tokens=max_tokens)
        start = time.monotonic()
        try:
            response = self._create(prompt, max_tokens)
        except Exception as exc:  # any failure of the request itself: log it, charge the reservation
            self.ledger.record(prompt=prompt, max_tokens=max_tokens, reserved=reserved,
                               elapsed_s=time.monotonic() - start, error=exc)
            raise TeacherCallFailed(str(exc)) from exc
        self.ledger.record(prompt=prompt, max_tokens=max_tokens, reserved=reserved,
                           elapsed_s=time.monotonic() - start, response=response)
        return _message_text(response)

    def _create(self, prompt: str, max_tokens: int) -> Any:
        return self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )


@dataclass(frozen=True)
class Turn:
    role: str
    segments: list[str]
    target: str


@dataclass(frozen=True)
class Trajectory:
    task_id: int
    turns: list[Turn]
    final_code: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "turns": [asdict(turn) for turn in self.turns],
            "final_code": self.final_code,
        }


def _coder_prompt(
    task_segment: str,
    *,
    request_bug: bool,
    correct_code: str | None = None,
    reviewer_feedback: str = "",
    tool_output: str = "",
) -> str:
    if request_bug:
        return (
            "Return a plausible but subtly buggy first attempt for the following task. "
            "It must contain exactly one Python code block and no prose outside it.\n\n" + task_segment
        )
    if correct_code is None:
        return (
            "Solve the following task. Return exactly one Python code block with the full function and no prose outside it.\n\n"
            + task_segment
        )
    return (
        "The previous attempt failed its real test. Return a corrected replacement. "
        "Return exactly one Python code block and no prose outside it.\n\n"
        + task_segment
        + f"\nPrevious code:\n```python\n{correct_code}```\n"
        + f"\nReviewer feedback:\n{reviewer_feedback}\n"
        + f"\nActual test output:\n<tool_output>\n{tool_output}\n</tool_output>"
    )


def _reviewer_prompt(code: str, result: TestResult) -> str:
    return (
        "Review this candidate Python solution using the actual test result below. "
        "Your first line must be exactly `APPROVE` when it passed, otherwise start with `REVISE: `.\n\n"
        f"```python\n{code}```\n\n<tool_output>\n{result.output}\n</tool_output>"
    )


def _still_buggy_repair_prompt(task_segment: str, *, previous_code: str, reviewer_feedback: str, tool_output: str) -> str:
    return (
        "The previous attempt failed its real test. Return a revised attempt that addresses the reviewer's "
        "feedback but is still subtly wrong, so that it fails at least one of the task's tests. "
        "Return exactly one Python code block and no prose outside it.\n\n"
        + task_segment
        + f"\nPrevious code:\n```python\n{previous_code}```\n"
        + f"\nReviewer feedback:\n{reviewer_feedback}\n"
        + f"\nActual test output:\n<tool_output>\n{tool_output}\n</tool_output>"
    )


def _correct_reviewer_label(code: str, result: TestResult, teacher: Teacher) -> str:
    if result.passed:
        return "APPROVE"
    feedback = teacher.ask(_reviewer_prompt(code, result), max_tokens=128)
    if not feedback.startswith("REVISE:"):
        raise ValueError("Reviewer did not return REVISE feedback for a failing solution")
    return feedback


@dataclass(frozen=True)
class Rejection:
    """Why a task produced no trajectory; the reasons are counted in the generation report."""

    reason: str


# Accepted trajectory shapes, keyed by their number of role turns. The session protocol allows at most two
# REVISE rounds, so the longest shape is five turns and its last turn is the Coder's.
SHAPES_BY_TURNS = {
    2: "clean",  # Coder (passes) -> APPROVE
    3: "repair",  # buggy Coder -> REVISE -> Coder (passes); the default, no closing review
    4: "repair_approved",  # buggy Coder -> REVISE -> Coder (passes) -> APPROVE  (--multiround)
    5: "two_revisions",  # buggy Coder -> REVISE -> Coder (fails) -> REVISE -> Coder (passes)  (--multiround)
}
MAX_TURNS = max(SHAPES_BY_TURNS)


def trajectory_shape(trajectory: Mapping[str, Any]) -> str:
    return SHAPES_BY_TURNS[len(trajectory["turns"])]


def _ask_code(teacher: Teacher, prompt: str) -> str | None:
    text = teacher.ask(prompt, max_tokens=512)  # teacher errors propagate; main() counts them as "error"
    try:
        return extract_python_code(text)
    except ValueError:
        return None


def _coder_turn_after_review(review: Turn) -> list[str]:
    return review.segments + [review.target, role_header("coder")]


def _review_turn(coder: Turn, code: str, result: TestResult, teacher: Teacher) -> tuple[Turn, str] | None:
    """The Reviewer turn after ``coder``, labelled from the real test result; None if the teacher's REVISE is unusable."""

    segments = coder.segments + [coder.target, tool_segment(result), role_header("reviewer")]
    try:
        label = _correct_reviewer_label(code, result, teacher)
    except ValueError:
        return None
    return Turn("reviewer", segments, body_segment(label)), label


def build_trajectory(
    task: Mapping[str, Any],
    training_rows: Iterable[Mapping[str, Any]],
    teacher: Teacher,
    *,
    ctx_funcs: int,
    request_bug: bool,
    multiround: bool = False,
    request_second_bug: bool = False,
) -> Trajectory | Rejection:
    """Generate one verified trajectory, or the reason it failed filtering.

    Without ``multiround`` this makes exactly the teacher calls and trajectories it always has. With it, a
    repair that passes gets a closing APPROVE review (no teacher call: the label follows the real result), and a
    repair that fails gets a second REVISE and a final Coder turn. ``request_second_bug`` asks the teacher for a
    still-failing first repair, so that the second shape is not left to chance. Every kept trajectory ends with
    code that passes all of the task's asserts.
    """

    if request_second_bug and not (multiround and request_bug):
        raise ValueError("request_second_bug needs multiround and request_bug")
    training_rows = list(training_rows)
    task_segment = build_task_segment(task, training_rows, ctx_funcs=ctx_funcs)
    # The student's context always uses the contract's system segment, the same one the runtime uses at eval.
    # The teacher's own system prompt is separate and never appears in the training data.
    initial_segments = [system_segment(), task_segment, role_header("coder")]
    task_id = int(task["task_id"])

    if not request_bug:
        final_code = _ask_code(teacher, _coder_prompt(task_segment, request_bug=False))
        if final_code is None:
            return Rejection("clean_no_code")
        final_result = run_mbpp_task(final_code, task)
        if not final_result.passed:
            return Rejection("clean_failed")
        coder_target = coder_body(final_code)
        review_segments = initial_segments + [coder_target, tool_segment(final_result), role_header("reviewer")]
        return Trajectory(
            task_id=task_id,
            turns=[
                Turn("coder", initial_segments, coder_target),
                Turn("reviewer", review_segments, body_segment("APPROVE")),
            ],
            final_code=final_code,
        )

    buggy_code = _ask_code(teacher, _coder_prompt(task_segment, request_bug=True))
    if buggy_code is None:
        return Rejection("buggy_no_code")
    buggy_result = run_mbpp_task(buggy_code, task)
    if buggy_result.passed:
        return Rejection("buggy_passed")
    turns = [Turn("coder", initial_segments, coder_body(buggy_code))]
    review = _review_turn(turns[-1], buggy_code, buggy_result, teacher)
    if review is None:
        return Rejection("review_not_revise")
    turns.append(review[0])

    if request_second_bug:
        repair_prompt = _still_buggy_repair_prompt(
            task_segment, previous_code=buggy_code, reviewer_feedback=review[1], tool_output=buggy_result.output
        )
    else:
        repair_prompt = _coder_prompt(
            task_segment,
            request_bug=False,
            correct_code=buggy_code,
            reviewer_feedback=review[1],
            tool_output=buggy_result.output,
        )
    repair_code = _ask_code(teacher, repair_prompt)
    if repair_code is None:
        return Rejection("repair_no_code")
    repair_result = run_mbpp_task(repair_code, task)
    turns.append(Turn("coder", _coder_turn_after_review(turns[-1]), coder_body(repair_code)))

    if not multiround:
        if not repair_result.passed:
            return Rejection("repair_failed")
        return Trajectory(task_id=task_id, turns=turns, final_code=repair_code)

    if repair_result.passed:
        if request_second_bug:
            return Rejection("second_attempt_passed")
        closing = _review_turn(turns[-1], repair_code, repair_result, teacher)
        if closing is None or closing[1] != "APPROVE":  # a passing result is labelled without the teacher
            raise RuntimeError("a passing repair must be labelled APPROVE")
        return Trajectory(task_id=task_id, turns=turns + [closing[0]], final_code=repair_code)

    second_review = _review_turn(turns[-1], repair_code, repair_result, teacher)
    if second_review is None:
        return Rejection("second_review_not_revise")
    turns.append(second_review[0])
    # After the second REVISE the next Coder turn is final (session protocol), so no review follows it.
    final_code = _ask_code(
        teacher,
        _coder_prompt(
            task_segment,
            request_bug=False,
            correct_code=repair_code,
            reviewer_feedback=second_review[1],
            tool_output=repair_result.output,
        ),
    )
    if final_code is None:
        return Rejection("final_no_code")
    if not run_mbpp_task(final_code, task).passed:
        return Rejection("final_failed")
    turns.append(Turn("coder", _coder_turn_after_review(turns[-1]), coder_body(final_code)))
    return Trajectory(task_id=task_id, turns=turns, final_code=final_code)


def generate_trajectory(
    task: Mapping[str, Any],
    training_rows: Iterable[Mapping[str, Any]],
    teacher: Teacher,
    *,
    ctx_funcs: int,
    request_bug: bool,
    multiround: bool = False,
    request_second_bug: bool = False,
) -> Trajectory | None:
    """Generate one verified trajectory, returning None when it fails filtering."""

    result = build_trajectory(
        task,
        training_rows,
        teacher,
        ctx_funcs=ctx_funcs,
        request_bug=request_bug,
        multiround=multiround,
        request_second_bug=request_second_bug,
    )
    return result if isinstance(result, Trajectory) else None


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    # newline="\n": LF on every OS, so the hashes recorded in README.md match the committed bytes.
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def export_sft(trajectories: Iterable[Mapping[str, Any]], output_dir: Path) -> dict[str, tuple[int, str]]:
    """Create one SFT JSONL file per role and return counts plus SHA-256 hashes."""

    by_role: dict[str, list[dict[str, Any]]] = {"coder": [], "reviewer": []}
    for trajectory in trajectories:
        for turn in trajectory["turns"]:
            by_role[turn["role"]].append(
                {"task_id": trajectory["task_id"], "role": turn["role"], "segments": turn["segments"], "target": turn["target"]}
            )

    results: dict[str, tuple[int, str]] = {}
    for role, rows in by_role.items():
        path = output_dir / f"sft_{role}.jsonl"
        count = write_jsonl(path, rows)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        results[role] = (count, digest)
    return results


def normalize_system_segments(trajectories: Iterable[dict[str, Any]]) -> int:
    """Replace each turn's system segment with the contract's; returns how many were changed.

    Data generated before the fix used the teacher's system prompt here, which
    differs from what the runtime puts in front of every session at eval.
    """
    expected = system_segment()
    changed = 0
    for trajectory in trajectories:
        for turn in trajectory["turns"]:
            first = turn["segments"][0]
            if first.startswith("<|im_start|>system\n") and first != expected:
                turn["segments"][0] = expected
                changed += 1
    return changed


def write_readme(path: Path, *, trajectories: int, stats: Mapping[str, tuple[int, str]]) -> None:
    lines = ["# RoleKV SFT data", "", f"Verified trajectories: {trajectories}", ""]
    for role in ("coder", "reviewer"):
        count, digest = stats[role]
        lines.append(f"- `sft_{role}.jsonl`: {count} rows; SHA-256 `{digest}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data"))
    parser.add_argument("--limit", type=int, default=20, help="Start with 20 to check API spend.")
    parser.add_argument("--ctx-funcs", type=int, default=30)
    parser.add_argument("--bug-rate", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--system-prompt", default=SYSTEM_PROMPT,
                        help="system prompt for the teacher model only; student data always uses roles.system_segment()")
    parser.add_argument("--reexport", action="store_true",
                        help="no API calls: rebuild the SFT files and README from <out>/trajectories.jsonl, "
                             "normalizing system segments")
    parser.add_argument("--multiround", action="store_true",
                        help="add second reviews: a passing repair gets a closing APPROVE, a failing one a second "
                             "REVISE and a final Coder turn (at most 5 turns); needs a new --out directory")
    parser.add_argument("--second-bug-rate", type=float, default=0.5,
                        help="with --multiround: share of buggy trajectories whose first repair the teacher is "
                             "asked to keep failing")
    parser.add_argument("--max-spend-usd", type=float,
                        help="meter the teacher: log every request to <out>/api_usage.jsonl and never send one whose "
                             "worst-case cost could pass this cap (required with --multiround)")
    parser.add_argument("--max-calls", type=int, help="with --max-spend-usd: at most this many teacher requests")
    parser.add_argument("--api-timeout", type=float,
                        help="with --max-spend-usd: client timeout in seconds; metered requests are never retried")
    parser.add_argument("--max-consecutive-api-errors", type=int, default=3,
                        help="with --max-spend-usd: stop after this many tasks in a row end on a failed request")
    return parser.parse_args()


OUTPUT_FILES = ("trajectories.jsonl", "sft_coder.jsonl", "sft_reviewer.jsonl", "README.md", "generation_report.json",
                "api_usage.jsonl")
USAGE_LOG = "api_usage.jsonl"


def draw_requests(
    count: int, *, seed: int, bug_rate: float, multiround: bool, second_bug_rate: float
) -> list[tuple[bool, bool]]:
    """Per selected task: (request a buggy first attempt, request a still-failing first repair).

    The first flag uses the same random stream as before --multiround existed, one draw per task, so the
    default run asks for exactly the same bugs. The second flag has its own stream.
    """

    rng = random.Random(seed)
    second_rng = random.Random(f"{seed}:second-bug")
    requests = []
    for _ in range(count):
        request_bug = rng.random() < bug_rate
        request_second_bug = multiround and request_bug and second_rng.random() < second_bug_rate
        requests.append((request_bug, request_second_bug))
    return requests


def generation_report(trajectories: Iterable[Mapping[str, Any]], rejections: Iterable[str], attempted: int) -> dict[str, Any]:
    kept = list(trajectories)
    return {
        "attempted": attempted,
        "kept": len(kept),
        "shapes": dict(sorted(Counter(trajectory_shape(t) for t in kept).items())),
        "rejections": dict(sorted(Counter(rejections).items())),
    }


def main() -> None:
    args = parse_args()
    if args.reexport and args.multiround:
        raise SystemExit("--reexport rewrites an existing dataset in place; it cannot be combined with --multiround")
    if args.reexport:
        path = args.out / "trajectories.jsonl"
        trajectories = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        changed = normalize_system_segments(trajectories)
        write_jsonl(path, trajectories)
        stats = export_sft(trajectories, args.out)
        write_readme(args.out / "README.md", trajectories=len(trajectories), stats=stats)
        print(f"Re-exported {len(trajectories)} trajectories; normalized {changed} system segments.")
        return
    _check_generation_args(args)
    load_dotenv()
    ledger = None
    if args.max_spend_usd is not None:
        ledger = UsageLedger(model=args.model, cap_usd=args.max_spend_usd, max_calls=args.max_calls,
                             log_path=args.out / USAGE_LOG)
        teacher = Teacher(model=args.model, system_prompt=args.system_prompt, ledger=ledger, timeout=args.api_timeout)
    else:
        teacher = Teacher(model=args.model, system_prompt=args.system_prompt)
    splits = load_mbpp_splits()
    selected = list(splits.training)
    random.Random(args.seed).shuffle(selected)
    selected = selected[: args.limit]
    requests = draw_requests(
        len(selected),
        seed=args.seed,
        bug_rate=args.bug_rate,
        multiround=args.multiround,
        second_bug_rate=args.second_bug_rate,
    )
    trajectories, outcomes, stopped = _generate(args, selected, requests, splits.training, teacher, ledger)

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "trajectories.jsonl", trajectories)
    stats = export_sft(trajectories, args.out)
    write_readme(args.out / "README.md", trajectories=len(trajectories), stats=stats)
    rejections = [outcome for outcome in outcomes.values() if outcome != "kept"]
    report = generation_report(trajectories, rejections, len(outcomes))
    if ledger is not None:
        report = {**report, "selected": len(selected), "stopped": stopped, "api": ledger.summary(outcomes)}
    if args.multiround:
        settings = {key: getattr(args, key) for key in ("seed", "limit", "bug_rate", "second_bug_rate", "ctx_funcs", "model",
                                                        "max_spend_usd", "max_calls", "api_timeout")}
        (args.out / "generation_report.json").write_text(
            json.dumps({**report, "settings": settings}, indent=2) + "\n", encoding="utf-8", newline="\n"
        )
    print(f"Kept {len(trajectories)} verified trajectories from {len(outcomes)} attempted tasks.")
    print(f"Shapes: {report['shapes']}; rejections: {report['rejections']}")
    if ledger is not None:
        api = report["api"]
        print(f"API: {api['calls']} requests, {api['errors']} failed, ${api['charged_usd']:.4f} charged"
              + (f"; stopped: {stopped}" if stopped else ""))


def _check_generation_args(args: argparse.Namespace) -> None:
    """Refuse bad settings before anything is written or any client is created."""

    if args.limit < 1 or not 0 <= args.bug_rate <= 1 or not 0 <= args.second_bug_rate <= 1:
        raise SystemExit("--limit must be positive and --bug-rate and --second-bug-rate must be between 0 and 1")
    if args.multiround and args.max_spend_usd is None:
        raise SystemExit("--multiround needs --max-spend-usd, so every teacher request is metered and capped")
    if args.max_spend_usd is not None:
        if args.model not in PRICES_PER_MTOK:
            raise SystemExit(f"no price recorded for {args.model!r}; cannot enforce --max-spend-usd")
        if args.max_spend_usd <= 0 or args.max_consecutive_api_errors < 1 or (
            args.max_calls is not None and args.max_calls < 1
        ) or (args.api_timeout is not None and args.api_timeout <= 0):
            raise SystemExit("--max-spend-usd and --api-timeout must be positive, "
                             "--max-calls and --max-consecutive-api-errors at least 1")
        if (args.out / USAGE_LOG).exists():
            raise SystemExit(f"{args.out / USAGE_LOG} already exists; choose a new --out")
    if args.multiround:
        existing = [name for name in OUTPUT_FILES if (args.out / name).exists()]
        if existing:
            raise SystemExit(f"--multiround never overwrites a dataset; {args.out} already has {', '.join(existing)}")


def _generate(
    args: argparse.Namespace,
    selected: list[Mapping[str, Any]],
    requests: list[tuple[bool, bool]],
    training_rows: Iterable[Mapping[str, Any]],
    teacher: Teacher,
    ledger: UsageLedger | None,
) -> tuple[list[dict[str, Any]], dict[int, str], str | None]:
    """Run the selected tasks; returns kept trajectories, each attempted task's outcome, and why it stopped early."""

    trajectories: list[dict[str, Any]] = []
    outcomes: dict[int, str] = {}
    consecutive_api_errors = 0
    for task, (request_bug, request_second_bug) in zip(selected, requests):
        task_id = int(task["task_id"])
        if ledger is not None:
            ledger.current_task = task_id
        try:
            result = build_trajectory(
                task,
                training_rows,
                teacher,
                ctx_funcs=args.ctx_funcs,
                request_bug=request_bug,
                multiround=args.multiround,
                request_second_bug=request_second_bug,
            )
        except SpendLimitReached as exc:  # only raised by a metered teacher, before the request is sent
            outcomes[task_id] = "spend_limit"
            return trajectories, outcomes, str(exc)
        except TeacherCallFailed:
            result = Rejection("api_error")
        except ValueError:
            result = Rejection("error")
        if isinstance(result, Rejection):
            outcomes[task_id] = result.reason
        else:
            outcomes[task_id] = "kept"
            trajectories.append(result.as_dict())
        consecutive_api_errors = consecutive_api_errors + 1 if outcomes[task_id] == "api_error" else 0
        if consecutive_api_errors >= args.max_consecutive_api_errors:
            return trajectories, outcomes, f"{consecutive_api_errors} tasks in a row ended on a failed request"
    return trajectories, outcomes, None


if __name__ == "__main__":
    main()
