"""Generate test-verified Coder/Reviewer SFT data from MBPP.

The script deliberately records the test runner's real output before asking the
Reviewer for feedback. It never asks the teacher to invent a tool result.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import re
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


class Teacher:
    """Small synchronous wrapper around the Anthropic Messages API."""

    def __init__(self, *, model: str, system_prompt: str) -> None:
        try:
            from anthropic import Anthropic
        except ImportError as exc:
            raise RuntimeError("Install the 'anthropic' package to generate teacher data.") from exc
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("ANTHROPIC_API_KEY is required to generate teacher data.")
        self._client = Anthropic()
        self.model = model
        self.system_prompt = system_prompt

    def ask(self, prompt: str, *, max_tokens: int) -> str:
        response = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=self.system_prompt,
            messages=[{"role": "user", "content": prompt}],
        )
        return _message_text(response)


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


def _correct_reviewer_label(code: str, result: TestResult, teacher: Teacher) -> str:
    if result.passed:
        return "APPROVE"
    feedback = teacher.ask(_reviewer_prompt(code, result), max_tokens=128)
    if not feedback.startswith("REVISE:"):
        raise ValueError("Reviewer did not return REVISE feedback for a failing solution")
    return feedback


def generate_trajectory(
    task: Mapping[str, Any],
    training_rows: Iterable[Mapping[str, Any]],
    teacher: Teacher,
    *,
    ctx_funcs: int,
    request_bug: bool,
) -> Trajectory | None:
    """Generate one verified trajectory, returning None when it fails filtering."""

    training_rows = list(training_rows)
    task_segment = build_task_segment(task, training_rows, ctx_funcs=ctx_funcs)
    # The student's context always uses the contract's system segment, the same one the runtime uses at eval.
    # The teacher's own system prompt is separate and never appears in the training data.
    initial_segments = [system_segment(), task_segment, role_header("coder")]

    if request_bug:
        buggy_text = teacher.ask(_coder_prompt(task_segment, request_bug=True), max_tokens=512)
        try:
            buggy_code = extract_python_code(buggy_text)
        except ValueError:
            return None
        buggy_result = run_mbpp_task(buggy_code, task)
        if buggy_result.passed:
            return None

        buggy_target = coder_body(buggy_code)
        review_segments = initial_segments + [buggy_target, tool_segment(buggy_result), role_header("reviewer")]
        try:
            review_text = _correct_reviewer_label(buggy_code, buggy_result, teacher)
        except ValueError:
            return None
        repair_segments = review_segments + [body_segment(review_text), role_header("coder")]
        final_text = teacher.ask(
            _coder_prompt(
                task_segment,
                request_bug=False,
                correct_code=buggy_code,
                reviewer_feedback=review_text,
                tool_output=buggy_result.output,
            ),
            max_tokens=512,
        )
        try:
            final_code = extract_python_code(final_text)
        except ValueError:
            return None
        final_result = run_mbpp_task(final_code, task)
        if not final_result.passed:
            return None
        return Trajectory(
            task_id=int(task["task_id"]),
            turns=[
                Turn("coder", initial_segments, buggy_target),
                Turn("reviewer", review_segments, body_segment(review_text)),
                Turn("coder", repair_segments, coder_body(final_code)),
            ],
            final_code=final_code,
        )

    final_text = teacher.ask(_coder_prompt(task_segment, request_bug=False), max_tokens=512)
    try:
        final_code = extract_python_code(final_text)
    except ValueError:
        return None
    final_result = run_mbpp_task(final_code, task)
    if not final_result.passed:
        return None
    coder_target = coder_body(final_code)
    review_segments = initial_segments + [coder_target, tool_segment(final_result), role_header("reviewer")]
    return Trajectory(
        task_id=int(task["task_id"]),
        turns=[
            Turn("coder", initial_segments, coder_target),
            Turn("reviewer", review_segments, body_segment("APPROVE")),
        ],
        final_code=final_code,
    )


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.reexport:
        path = args.out / "trajectories.jsonl"
        trajectories = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        changed = normalize_system_segments(trajectories)
        write_jsonl(path, trajectories)
        stats = export_sft(trajectories, args.out)
        write_readme(args.out / "README.md", trajectories=len(trajectories), stats=stats)
        print(f"Re-exported {len(trajectories)} trajectories; normalized {changed} system segments.")
        return
    if args.limit < 1 or not 0 <= args.bug_rate <= 1:
        raise SystemExit("--limit must be positive and --bug-rate must be between 0 and 1")
    load_dotenv()
    teacher = Teacher(model=args.model, system_prompt=args.system_prompt)
    splits = load_mbpp_splits()
    selected = list(splits.training)
    random.Random(args.seed).shuffle(selected)
    selected = selected[: args.limit]

    trajectories = []
    rng = random.Random(args.seed)
    for task in selected:
        try:
            trajectory = generate_trajectory(
                task, splits.training, teacher, ctx_funcs=args.ctx_funcs, request_bug=rng.random() < args.bug_rate
            )
        except ValueError:
            trajectory = None
        if trajectory is not None:
            trajectories.append(trajectory.as_dict())

    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "trajectories.jsonl", trajectories)
    stats = export_sft(trajectories, args.out)
    write_readme(args.out / "README.md", trajectories=len(trajectories), stats=stats)
    print(f"Kept {len(trajectories)} verified trajectories from {len(selected)} attempted tasks.")


if __name__ == "__main__":
    main()
