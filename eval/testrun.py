"""Run generated MBPP solutions in a short-lived Python subprocess.

Both dataset generation and final evaluation use this module so a teacher's
trajectory is verified by the exact same harness as the reported results.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import os
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping


DEFAULT_TIMEOUT_SECONDS = 5.0
DEFAULT_OUTPUT_LIMIT = 2_000


@dataclass(frozen=True)
class TestResult:
    """The normalized outcome consumed by data generation and evaluation."""

    passed: bool
    output: str
    stdout: str
    stderr: str
    timed_out: bool
    returncode: int | None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _combined_output(stdout: str, stderr: str, limit: int) -> str:
    if limit < 1:
        raise ValueError("output_limit must be at least 1")
    output = stdout
    if stdout and stderr:
        output += "\n"
    output += stderr
    if len(output) <= limit:
        return output

    suffix = "\n...[output truncated]"
    if limit <= len(suffix):
        return output[:limit]
    return output[: limit - len(suffix)] + suffix


def _assert_source(asserts: str | Iterable[str]) -> str:
    if isinstance(asserts, str):
        return asserts
    lines = list(asserts)
    if not lines or not all(isinstance(line, str) for line in lines):
        raise ValueError("asserts must be a string or a non-empty iterable of strings")
    return "\n".join(lines)


def run_code(
    code: str,
    asserts: str | Iterable[str],
    *,
    setup_code: str = "",
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    output_limit: int = DEFAULT_OUTPUT_LIMIT,
) -> TestResult:
    """Run Python source and assertions in an isolated temporary directory.

    ``-I`` prevents user-site imports and ignores environment Python settings.
    The subprocess boundary and timeout prevent a failed completion from taking
    down the data-generation or evaluation process. This is not a security
    sandbox for malicious code; use a container for untrusted submissions.
    """

    if not isinstance(code, str) or not isinstance(setup_code, str):
        raise ValueError("code and setup_code must be strings")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    assertion_source = _assert_source(asserts)
    source = "\n\n".join(part for part in (setup_code, code, assertion_source) if part) + "\n"

    with tempfile.TemporaryDirectory(prefix="rolekv-mbpp-") as temp_dir:
        script_path = Path(temp_dir) / "solution_test.py"
        script_path.write_text(source, encoding="utf-8")
        try:
            completed = subprocess.run(
                [sys.executable, "-I", str(script_path)],
                cwd=temp_dir,
                env={"PATH": os.environ.get("PATH", "")},
                text=True,
                capture_output=True,
                timeout=timeout_seconds,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout or ""
            stderr = exc.stderr or ""
            if isinstance(stdout, bytes):
                stdout = stdout.decode(errors="replace")
            if isinstance(stderr, bytes):
                stderr = stderr.decode(errors="replace")
            stderr = f"{stderr}\nTimed out after {timeout_seconds:g} seconds.".lstrip()
            return TestResult(
                passed=False,
                output=_combined_output(stdout, stderr, output_limit),
                stdout=stdout,
                stderr=stderr,
                timed_out=True,
                returncode=None,
            )

    return TestResult(
        passed=completed.returncode == 0,
        output=_combined_output(completed.stdout, completed.stderr, output_limit),
        stdout=completed.stdout,
        stderr=completed.stderr,
        timed_out=False,
        returncode=completed.returncode,
    )


def run_mbpp_task(
    code: str,
    task: Mapping[str, Any],
    *,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    output_limit: int = DEFAULT_OUTPUT_LIMIT,
) -> TestResult:
    """Run a solution against every visible MBPP assertion for one task."""

    try:
        asserts = task["test_list"]
    except KeyError as exc:
        raise ValueError("MBPP task must contain test_list") from exc
    setup_code = task.get("test_setup_code", "")
    return run_code(
        code,
        asserts,
        setup_code=setup_code,
        timeout_seconds=timeout_seconds,
        output_limit=output_limit,
    )
