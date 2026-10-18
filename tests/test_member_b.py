"""Fast tests for Member B modules (no Anthropic API calls)."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

from eval.testrun import TestResult as HarnessResult
from eval.testrun import run_code, run_mbpp_task
from rolekv.tasks import (
    EVAL_TASK_COUNT,
    build_task_segment,
    build_utils_py,
    select_evaluation_tasks,
)
from rolekv.roles import body_segment, role_header, system_segment, tool_segment
from train.build_dataset import coder_body, extract_python_code
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

ROLE_HEADER_RE = re.compile(r"^<\|im_start\|>assistant\n<\|role:(coder|reviewer)\|>\n$")
IM_END_SUFFIX = "<|im_end|>\n"


def _fake_training_rows(n: int = 5) -> list[dict]:
    return [
        {"task_id": i, "code": f"def helper_{i}(x):\n    return x + {i}\n", "text": f"task {i}", "test_list": [f"assert helper_{i}(1)"]}
        for i in range(100, 100 + n)
    ]


def validate_sft_row(row: dict) -> None:
    """Assert one training row matches docs/CONTRACT.md §2."""

    for key in ("task_id", "role", "segments", "target"):
        assert key in row, f"missing key {key!r}"
    assert row["role"] in {"coder", "reviewer"}
    assert isinstance(row["segments"], list) and row["segments"]
    assert all(isinstance(seg, str) for seg in row["segments"])
    assert row["target"].endswith(IM_END_SUFFIX), "target must be full BODY segment with trailing im_end"

    last = row["segments"][-1]
    assert ROLE_HEADER_RE.match(last), f"last segment must be role HEADER, got {last!r}"
    assert last.endswith(f"<|role:{row['role']}|>\n")

    assert row["segments"][0].startswith("<|im_start|>system\n")
    task_seg = row["segments"][1]
    assert "# Repository file: utils.py" in task_seg
    assert "# Task\n" in task_seg
    assert "# Visible test\n" in task_seg

    body = row["target"][: -len(IM_END_SUFFIX)]
    if row["role"] == "reviewer":
        first_line = body.split("\n", 1)[0]
        assert first_line == "APPROVE" or first_line.startswith("REVISE: ")
    else:
        assert re.fullmatch(r"```python\s*\n.*?```", body, flags=re.DOTALL), (
            "coder target must be exactly one python fence"
        )


class TestBuildDatasetHelpers:
    def test_segments_and_extract_code(self) -> None:
        sys_seg = system_segment("Hello")
        assert sys_seg.endswith(IM_END_SUFFIX)
        assert role_header("coder").endswith("<|role:coder|>\n")

        text = "```python\ndef f():\n    return 1\n```"
        assert extract_python_code(text).startswith("def f")
        assert coder_body(extract_python_code(text)).startswith("```python\n")

        with pytest.raises(ValueError):
            extract_python_code("no fence here")

        tr = HarnessResult(passed=False, output="fail", stdout="", stderr="fail", timed_out=False, returncode=1)
        assert "<tool_output>" in tool_segment(tr.output)


class TestTaskSegment:
    def test_build_utils_excludes_held_out_task(self) -> None:
        rows = _fake_training_rows(10)
        task = {"task_id": 105, "text": "Do something", "test_list": ["assert True"]}
        utils = build_utils_py(task, rows, ctx_funcs=3)
        assert "def helper_105" not in utils
        assert utils.count("def helper_") == 3

    def test_task_segment_contract_shape(self) -> None:
        rows = _fake_training_rows(10)
        task = {"task_id": 200, "text": "Return x.", "test_list": ["assert fn(1) == 1"]}
        seg = build_task_segment(task, rows, ctx_funcs=2)
        assert seg.startswith("<|im_start|>user\n")
        assert seg.endswith(IM_END_SUFFIX)
        assert "assert fn(1) == 1" in seg


class TestSelectEvaluation:
    def test_reproducible_subset(self) -> None:
        rows = [{"task_id": i} for i in range(200)]

        class FakeSplit:
            def __init__(self, data: list) -> None:
                self._data = data

            def __len__(self) -> int:
                return len(self._data)

            def __iter__(self):
                return iter(self._data)

            def select(self, indices: list[int]) -> list:
                return [self._data[i] for i in indices]

        split = FakeSplit(rows)
        a = select_evaluation_tasks(split, count=10, seed=0)
        b = select_evaluation_tasks(split, count=10, seed=0)
        assert [r["task_id"] for r in a] == [r["task_id"] for r in b]


class TestTestrun:
    def test_passing_code(self) -> None:
        result = run_code("def add(a, b):\n    return a + b", "assert add(1, 2) == 3")
        assert result.passed
        assert not result.timed_out

    def test_failing_code(self) -> None:
        result = run_code("def add(a, b):\n    return a - b", "assert add(1, 2) == 3")
        assert not result.passed

    def test_output_truncation(self) -> None:
        result = run_code("print('x' * 5000)", "", output_limit=100)
        assert "...[output truncated]" in result.output

    def test_run_mbpp_task(self) -> None:
        task = {"test_list": ["assert square(3) == 9"], "test_setup_code": ""}
        ok = run_mbpp_task("def square(n):\n    return n * n", task)
        assert ok.passed


@pytest.mark.parametrize("path", [DATA / "sft_coder.jsonl", DATA / "sft_reviewer.jsonl"])
def test_sft_files_sample_rows_match_contract(path: Path) -> None:
    if not path.is_file():
        pytest.skip(f"missing {path}")

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines, f"{path.name} is empty"
    # Spot-check first, middle, and last rows.
    indices = {0, len(lines) // 2, len(lines) - 1}
    for i in sorted(indices):
        validate_sft_row(json.loads(lines[i]))


def test_data_readme_row_counts() -> None:
    readme = DATA / "README.md"
    if not readme.is_file():
        pytest.skip("data/README.md missing")
    text = readme.read_text(encoding="utf-8")
    coder_lines = (DATA / "sft_coder.jsonl").read_text(encoding="utf-8").splitlines()
    reviewer_lines = (DATA / "sft_reviewer.jsonl").read_text(encoding="utf-8").splitlines()
    assert f"sft_coder.jsonl`: {len(coder_lines)} rows" in text
    assert f"sft_reviewer.jsonl`: {len(reviewer_lines)} rows" in text


@pytest.mark.skipif(os.environ.get("ROLEKV_SKIP_MBPP") == "1", reason="ROLEKV_SKIP_MBPP=1")
def test_load_mbpp_splits_no_leakage() -> None:
    pytest.importorskip("datasets")
    from rolekv.tasks import load_mbpp_splits

    splits = load_mbpp_splits(eval_count=EVAL_TASK_COUNT)
    assert len(splits.evaluation) == EVAL_TASK_COUNT
    train_ids = {row["task_id"] for row in splits.training}
    eval_ids = {row["task_id"] for row in splits.evaluation}
    assert train_ids.isdisjoint(eval_ids)


class TestRunEvalHelpers:
    def test_resolve_factory_errors(self) -> None:
        from eval.run_eval import _resolve_factory

        with pytest.raises(ValueError):
            _resolve_factory("not-a-valid-spec")
