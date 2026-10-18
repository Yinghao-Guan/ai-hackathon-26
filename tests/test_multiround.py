"""Multi-round SFT generation (train/build_dataset.py --multiround) and eval/reviewer_diagnostics.py.

A scripted fake teacher and a fake test runner stand in for the Anthropic API and the subprocess harness, so
these tests make no API calls and execute no generated code.
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

import pytest

from eval import reviewer_diagnostics as diag
from eval.testrun import TestResult as HarnessResult
from rolekv import roles
from rolekv.roles import body_segment, role_header, system_segment, tool_segment
from rolekv.tasks import build_task_segment
from train import build_dataset as bd

TASK = {"task_id": 7, "text": "Return x plus one.", "test_list": ["assert add_one(1) == 2", "assert add_one(2) == 3"]}
TRAINING_ROWS = [{"task_id": i, "code": f"def helper_{i}(x):\n    return x + {i}\n"} for i in range(100, 105)]
CTX_FUNCS = 2
TASK_SEGMENT = build_task_segment(TASK, TRAINING_ROWS, ctx_funcs=CTX_FUNCS)
INITIAL = [system_segment(), TASK_SEGMENT, role_header("coder")]


class FakeTeacher:
    """Returns scripted replies in order and records every prompt."""

    def __init__(self, *replies: str) -> None:
        self._replies = list(replies)
        self.prompts: list[str] = []

    def ask(self, prompt: str, *, max_tokens: int) -> str:
        self.prompts.append(prompt)
        assert self._replies, "unexpected teacher call"
        return self._replies.pop(0)

    @property
    def unused(self) -> int:
        return len(self._replies)


def code(name: str) -> str:
    """A Coder reply; the fake runner passes code whose name starts with 'good'."""
    return f"```python\ndef {name}(x):\n    return x\n```"


def extracted(name: str) -> str:
    return bd.extract_python_code(code(name))


def fake_run(source: str, task: dict) -> HarnessResult:
    assert task is TASK
    name = re.search(r"def (\w+)", source).group(1)
    passed = name.startswith("good")
    output = "" if passed else f"AssertionError in {name}"
    return HarnessResult(passed=passed, output=output, stdout="", stderr=output, timed_out=False,
                      returncode=0 if passed else 1)


@pytest.fixture(autouse=True)
def fake_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bd, "run_mbpp_task", fake_run)


def build(teacher: FakeTeacher, **kwargs) -> bd.Trajectory | bd.Rejection:
    return bd.build_trajectory(TASK, TRAINING_ROWS, teacher, ctx_funcs=CTX_FUNCS, **kwargs)


def tool(name: str) -> str:
    return tool_segment(fake_run(extracted(name), TASK).output)


# --- shapes -------------------------------------------------------------------------------------------------


def test_repair_approved_shape_has_exact_segments_and_a_test_derived_approve() -> None:
    teacher = FakeTeacher(code("bad1"), "REVISE: off by one", code("good_fix"))
    result = build(teacher, request_bug=True, multiround=True)

    assert isinstance(result, bd.Trajectory)
    assert len(teacher.prompts) == 3  # the closing APPROVE needs no teacher call
    assert bd.trajectory_shape(result.as_dict()) == "repair_approved"
    c0, r1, c2, r3 = result.turns
    assert [t.role for t in result.turns] == ["coder", "reviewer", "coder", "reviewer"]
    assert c0.segments == INITIAL
    assert r1.segments == INITIAL + [bd.coder_body(extracted("bad1")), tool("bad1"), role_header("reviewer")]
    assert r1.target == body_segment("REVISE: off by one")
    assert c2.segments == r1.segments + [r1.target, role_header("coder")]
    assert r3.segments == c2.segments + [bd.coder_body(extracted("good_fix")), tool("good_fix"),
                                         role_header("reviewer")]
    assert r3.target == body_segment("APPROVE")
    assert result.final_code == extracted("good_fix")


def test_two_revisions_shape_ends_with_the_final_coder_turn() -> None:
    teacher = FakeTeacher(code("bad1"), "REVISE: first", code("bad2"), "REVISE: second", code("good_final"))
    result = build(teacher, request_bug=True, multiround=True)

    assert isinstance(result, bd.Trajectory)
    assert teacher.unused == 0 and len(teacher.prompts) == 5
    assert bd.trajectory_shape(result.as_dict()) == "two_revisions"
    assert [t.role for t in result.turns] == ["coder", "reviewer", "coder", "reviewer", "coder"]
    _, r1, c2, r3, c4 = result.turns
    assert r3.segments == c2.segments + [bd.coder_body(extracted("bad2")), tool("bad2"), role_header("reviewer")]
    assert r3.target == body_segment("REVISE: second")
    assert c4.segments == r3.segments + [r3.target, role_header("coder")]
    assert c4.target == bd.coder_body(extracted("good_final"))
    assert c4.segments.count(role_header("reviewer")) == 2
    assert result.final_code == extracted("good_final")
    # The final repair is asked from the latest failing code, feedback and real output.
    assert "def bad2" in teacher.prompts[4] and "REVISE: second" in teacher.prompts[4]
    assert "AssertionError in bad2" in teacher.prompts[4]


def test_requested_second_bug_uses_the_still_wrong_prompt() -> None:
    teacher = FakeTeacher(code("bad1"), "REVISE: first", code("bad2"), "REVISE: second", code("good_final"))
    result = build(teacher, request_bug=True, multiround=True, request_second_bug=True)
    assert isinstance(result, bd.Trajectory) and len(result.turns) == 5
    assert "still subtly wrong" in teacher.prompts[2]
    assert "still subtly wrong" not in teacher.prompts[4]


def test_every_trajectory_is_within_the_protocol_limit() -> None:
    assert bd.MAX_TURNS == 5
    assert set(bd.SHAPES_BY_TURNS) == {2, 3, 4, 5}


# --- filtering: labels follow real test outcomes, kept trajectories end with passing code ----------------


@pytest.mark.parametrize(
    ("replies", "kwargs", "reason"),
    [
        ((code("good_first"),), {}, None),
        (("no fence",), {}, "clean_no_code"),
        ((code("bad_clean"),), {}, "clean_failed"),
        (("no fence",), {"request_bug": True}, "buggy_no_code"),
        ((code("good_not_buggy"),), {"request_bug": True}, "buggy_passed"),
        ((code("bad1"), "APPROVE"), {"request_bug": True}, "review_not_revise"),
        ((code("bad1"), "REVISE: x", "no fence"), {"request_bug": True}, "repair_no_code"),
        ((code("bad1"), "REVISE: x", code("bad2")), {"request_bug": True}, "repair_failed"),
        ((code("bad1"), "REVISE: x", code("good")), {"request_bug": True, "multiround": True,
                                                     "request_second_bug": True}, "second_attempt_passed"),
        ((code("bad1"), "REVISE: x", code("bad2"), "APPROVE"), {"request_bug": True, "multiround": True},
         "second_review_not_revise"),
        ((code("bad1"), "REVISE: x", code("bad2"), "REVISE: y", "no fence"), {"request_bug": True,
                                                                               "multiround": True}, "final_no_code"),
        ((code("bad1"), "REVISE: x", code("bad2"), "REVISE: y", code("bad3")), {"request_bug": True,
                                                                                 "multiround": True}, "final_failed"),
    ],
)
def test_filter_reasons(replies: tuple[str, ...], kwargs: dict, reason: str | None) -> None:
    teacher = FakeTeacher(*replies)
    result = build(teacher, **{"request_bug": False, **kwargs})
    assert teacher.unused == 0
    if reason is None:
        assert isinstance(result, bd.Trajectory)
    else:
        assert result == bd.Rejection(reason)
        assert bd.generate_trajectory(TASK, TRAINING_ROWS, FakeTeacher(*replies), ctx_funcs=CTX_FUNCS,
                                      **{"request_bug": False, **kwargs}) is None


def test_kept_trajectories_end_with_passing_code_and_labels_match_outcomes() -> None:
    cases = [
        (FakeTeacher(code("good")), {}),
        (FakeTeacher(code("bad1"), "REVISE: a", code("good")), {"request_bug": True}),
        (FakeTeacher(code("bad1"), "REVISE: a", code("good")), {"request_bug": True, "multiround": True}),
        (FakeTeacher(code("bad1"), "REVISE: a", code("bad2"), "REVISE: b", code("good")),
         {"request_bug": True, "multiround": True}),
    ]
    for teacher, kwargs in cases:
        result = build(teacher, **{"request_bug": False, **kwargs})
        assert fake_run(result.final_code, TASK).passed
        for index, turn in enumerate(result.turns):
            if turn.role != "reviewer":
                continue
            reviewed = result.turns[index - 1].target
            passed = re.search(r"def (\w+)", reviewed).group(1).startswith("good")
            assert turn.target.startswith("APPROVE" if passed else "REVISE: ")


def test_second_bug_request_requires_multiround() -> None:
    with pytest.raises(ValueError):
        build(FakeTeacher(), request_bug=True, request_second_bug=True)


# --- default compatibility ------------------------------------------------------------------------------


def test_default_repair_is_unchanged() -> None:
    teacher = FakeTeacher(code("bad1"), "REVISE: off by one", code("good_fix"))
    result = build(teacher, request_bug=True)

    assert [t.role for t in result.turns] == ["coder", "reviewer", "coder"]
    c0, r1, c2 = result.turns
    assert c0.segments == INITIAL and c0.target == bd.coder_body(extracted("bad1"))
    assert r1.segments == INITIAL + [c0.target, tool("bad1"), role_header("reviewer")]
    assert c2.segments == r1.segments + [body_segment("REVISE: off by one"), role_header("coder")]
    assert c2.target == bd.coder_body(extracted("good_fix"))
    assert teacher.prompts == [
        bd._coder_prompt(TASK_SEGMENT, request_bug=True),
        bd._reviewer_prompt(extracted("bad1"), fake_run(extracted("bad1"), TASK)),
        bd._coder_prompt(TASK_SEGMENT, request_bug=False, correct_code=extracted("bad1"),
                         reviewer_feedback="REVISE: off by one", tool_output="AssertionError in bad1"),
    ]


def test_default_clean_is_unchanged() -> None:
    teacher = FakeTeacher(code("good"))
    result = build(teacher, request_bug=False)
    c0, r1 = result.turns
    assert c0.segments == INITIAL
    assert r1.segments == INITIAL + [bd.coder_body(extracted("good")), tool("good"), role_header("reviewer")]
    assert r1.target == body_segment("APPROVE")
    assert teacher.prompts == [bd._coder_prompt(TASK_SEGMENT, request_bug=False)]


@pytest.mark.parametrize("multiround", [False, True])
def test_bug_draws_match_the_original_stream(multiround: bool) -> None:
    rng = random.Random(0)
    expected = [rng.random() < 0.5 for _ in range(200)]
    requests = bd.draw_requests(200, seed=0, bug_rate=0.5, multiround=multiround, second_bug_rate=0.5)
    assert [bug for bug, _ in requests] == expected
    seconds = [second for _, second in requests]
    if multiround:
        assert any(seconds) and not any(s and not b for b, s in requests)
    else:
        assert not any(seconds)


def test_write_jsonl_uses_lf(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    assert bd.write_jsonl(path, [{"a": "x\ny"}, {"b": 2}]) == 2
    data = path.read_bytes()
    assert b"\r" not in data and data.count(b"\n") == 2


def test_multiround_refuses_to_overwrite_a_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "trajectories.jsonl").write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["build_dataset", "--multiround", "--max-spend-usd", "5", "--out", str(tmp_path)])
    monkeypatch.setattr(bd, "Teacher", lambda **_: pytest.fail("no teacher before the overwrite check"))
    with pytest.raises(SystemExit, match="never overwrites"):
        bd.main()


def test_reexport_cannot_be_combined_with_multiround(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    files = {name: f"original {name}\r\n".encode() for name in
             ("trajectories.jsonl", "sft_coder.jsonl", "sft_reviewer.jsonl", "README.md")}
    for name, data in files.items():
        (tmp_path / name).write_bytes(data)
    monkeypatch.setattr("sys.argv", ["build_dataset", "--reexport", "--multiround", "--out", str(tmp_path)])
    monkeypatch.setattr(bd, "Teacher", lambda **_: pytest.fail("no teacher may be constructed"))
    for writer in ("write_jsonl", "export_sft", "write_readme", "normalize_system_segments"):
        monkeypatch.setattr(bd, writer, lambda *a, **k: pytest.fail("the reexport branch must not run"))
    with pytest.raises(SystemExit, match="cannot be combined"):
        bd.main()
    assert {path.name: path.read_bytes() for path in tmp_path.iterdir()} == files


# --- SFT export of multi-round trajectories keeps CONTRACT section 2 ----------------------------------------


def _two_revisions() -> bd.Trajectory:
    teacher = FakeTeacher(code("bad1"), "REVISE: first", code("bad2"), "REVISE: second", code("good_final"))
    return build(teacher, request_bug=True, multiround=True)


def test_export_keeps_the_contract_row_format(tmp_path: Path) -> None:
    approved = build(FakeTeacher(code("bad1"), "REVISE: a", code("good")), request_bug=True, multiround=True)
    stats = bd.export_sft([approved.as_dict(), _two_revisions().as_dict()], tmp_path)
    assert stats["coder"][0] == 5 and stats["reviewer"][0] == 4
    for role in roles.ROLES:
        for line in (tmp_path / f"sft_{role}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            assert set(row) == {"task_id", "role", "segments", "target"}
            assert row["segments"][0] == system_segment()
            assert row["segments"][-1] == role_header(role)
            assert row["target"].endswith("<|im_end|>\n")
    report = bd.generation_report([approved.as_dict(), _two_revisions().as_dict()], ["final_failed"], 3)
    assert report == {"attempted": 3, "kept": 2, "shapes": {"repair_approved": 1, "two_revisions": 1},
                      "rejections": {"final_failed": 1}}


def test_second_review_tokenizes_with_its_header_last(tokenizer) -> None:
    second_review = _two_revisions().turns[3]
    example = roles.tokenize_sft_example(tokenizer, second_review.segments, second_review.target)
    header = roles.invocation_tokens(tokenizer, "reviewer")
    ids = example["input_ids"]
    starts = [i for i in range(len(ids) - len(header) + 1) if ids[i:i + len(header)] == header]
    context_len = example["labels"].count(roles.IGNORE_INDEX)
    assert len(starts) == 2  # both reviewer headers survive tokenization
    assert starts[-1] == context_len - len(header)  # the aLoRA activates at the last one, right before the target


# --- reviewer diagnostics ------------------------------------------------------------------------------------


def turn(task_id: int, index: int, verdict: str | None = None, *, method: str = "A5-H1", truncated: bool = False,
         role: str | None = None) -> dict:
    return {"type": "turn", "run_id": f"r-{method}", "method": method, "task_id": task_id, "turn": index,
            "role": role or ("coder" if index % 2 == 0 else "reviewer"), "verdict": verdict, "truncated": truncated}


def task(task_id: int, passed: object, n_turns: int, *, method: str = "A5-H1") -> dict:
    return {"type": "task", "run_id": f"r-{method}", "method": method, "task_id": task_id, "passed": passed,
            "n_turns": n_turns}


def session(task_id: int, *verdicts: str, passed: object = True, method: str = "A5-H1") -> list[dict]:
    rows = []
    for review, verdict in enumerate(verdicts):
        rows += [turn(task_id, 2 * review, method=method), turn(task_id, 2 * review + 1, verdict, method=method)]
    if verdicts and verdicts[-1] == "REVISE":
        rows.append(turn(task_id, len(rows), method=method))
    return rows + [task(task_id, passed, len(rows), method=method)]


def write_log(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "log.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def run_diag(tmp_path: Path, rows: list[dict]) -> dict:
    turns, tasks, issues = diag.load_rows([write_log(tmp_path, rows)])
    report = diag.diagnose(turns, tasks)
    return {**report, "issues": issues + report["issues"]}


def test_diagnostics_counts_by_round_and_false_approvals(tmp_path: Path) -> None:
    rows = (
        session(1, "APPROVE", passed=True)
        + session(2, "APPROVE", passed=False)  # round-1 false approval
        + session(3, "REVISE", "APPROVE", passed=False)  # round-2 false approval
        + session(4, "REVISE", "APPROVE", passed=True)
        + session(5, "REVISE", "REVISE", passed=False)  # 5 turns, ends on the final Coder turn
        + session(6, "INVALID", passed=False)  # INVALID is never a false approval
        + session(7, "REVISE", "REVISE", passed=True, method="A2")
    )
    rows[1]["truncated"] = True  # task 1's review
    report = run_diag(tmp_path, rows)

    assert report["issues"] == []
    h1 = report["groups"]["r-A5-H1 A5-H1"]
    assert h1["sessions"] == 6 and h1["sessions_with_issues"] == 0
    assert h1["rounds"]["1"] == {"reviews": 6, "verdicts": {"APPROVE": 2, "INVALID": 1, "REVISE": 3},
                                 "truncated": 1, "terminal_approve_passed": 1, "false_approvals": 1,
                                 "unverified_terminal_approvals": 0}
    assert h1["rounds"]["2"] == {"reviews": 3, "verdicts": {"APPROVE": 2, "REVISE": 1}, "truncated": 0,
                                 "terminal_approve_passed": 1, "false_approvals": 1,
                                 "unverified_terminal_approvals": 0}
    assert report["groups"]["r-A2 A2"]["rounds"]["1"]["verdicts"] == {"REVISE": 1}
    assert report["groups"]["r-A2 A2"]["rounds"]["2"]["verdicts"] == {"REVISE": 1}


def test_diagnostics_flags_incomplete_and_inconsistent_records(tmp_path: Path) -> None:
    rows = (
        session(1, "APPROVE", passed=False)[:-1]  # APPROVE with no task record: not a false approval
        + session(2, "APPROVE", passed=False) + [task(2, False, 2)]  # duplicate task record
        + [turn(3, 0), turn(3, 1, "REVISE"), task(3, False, 2)]  # ends on REVISE
        + [turn(7, 0), turn(7, 1, "REVISE"), turn(7, 2), task(7, False, 3)]  # stops before the second review
        + [turn(4, 0), turn(4, 2, "APPROVE", role="reviewer"), task(4, True, 2)]  # gap in turn indices
        + [turn(5, 0), turn(5, 1, "APPROVE"), turn(5, 2), task(5, True, 2)]  # continues after APPROVE
        + [turn(6, 0), turn(6, 1, "MAYBE"), task(6, "yes", 2)]  # bad verdict, passed not bool
        + [task(8, True, 2)]  # task record without turns
        + [{"type": "turn", "run_id": "r-A5-H1", "task_id": 9}]  # missing fields
    )
    log = write_log(tmp_path, rows)
    log.write_text(log.read_text(encoding="utf-8") + "{not json\n", encoding="utf-8")
    turns, tasks, load_issues = diag.load_rows([log])
    report = diag.diagnose(turns, tasks)

    problems = {issue.get("task_id"): issue["problems"] for issue in report["issues"]}
    assert problems[1] == ["missing_task_record", "approve_without_single_task_record"]
    assert problems[2] == ["duplicate_task_record", "approve_without_single_task_record"]
    assert problems[3] == ["ends_on_revise"]
    assert "turn_indices_not_contiguous" in problems[4]
    assert problems[5] == ["n_turns_mismatch", "turn_1_session_continues_after_APPROVE", "ends_on_coder_before_final"]
    assert problems[6] == ["passed_not_bool", "turn_1_bad_verdict"]
    assert problems[7] == ["ends_on_coder_before_final"]
    assert problems[8] == ["task_without_turns"]
    assert sorted(issue["problem"] for issue in load_issues) == ["malformed_json", "missing_fields:method,turn,role,verdict,truncated"]
    group = report["groups"]["r-A5-H1 A5-H1"]
    assert sum(stats["false_approvals"] for stats in group["rounds"].values()) == 0
    assert sum(stats["terminal_approve_passed"] for stats in group["rounds"].values()) == 0
    assert group["rounds"]["1"]["unverified_terminal_approvals"] == 3  # tasks 1, 2 and 4
    assert group["rounds"]["1"]["verdicts"]["MAYBE"] == 1


def test_invalid_role_order_cannot_add_a_false_approval(tmp_path: Path) -> None:
    valid = session(1, "REVISE", "APPROVE", passed=False)  # a real round-2 false approval
    # Same verdicts and a failing task record, but the Reviewer speaks first.
    bad_order = [turn(2, 0, "REVISE", role="reviewer"), turn(2, 1, "APPROVE"), task(2, False, 2)]
    report = run_diag(tmp_path, valid + bad_order)

    group = report["groups"]["r-A5-H1 A5-H1"]
    assert [issue["problems"] for issue in report["issues"]] == [["turn_0_role_reviewer"]]
    assert group["rounds"]["2"]["false_approvals"] == 1  # only the valid session
    assert group["rounds"]["2"]["unverified_terminal_approvals"] == 1
    assert group["rounds"]["2"]["verdicts"] == {"APPROVE": 2}  # raw verdicts still include the flagged session
    assert group["rounds"]["1"]["verdicts"] == {"REVISE": 2}
    assert group["sessions_with_issues"] == 1


def test_diagnostics_reports_bad_types_instead_of_raising(tmp_path: Path) -> None:
    lines = [
        json.dumps({**turn(1, 0), "turn": "0"}),  # string turn index
        json.dumps({**turn(2, 0), "task_id": [2]}),  # unhashable task_id
        json.dumps({**task(3, True, 2), "task_id": {"id": 3}}),  # unhashable task_id on a task row
        json.dumps({**turn(4, 0), "truncated": "no", "verdict": 5}),
        json.dumps({**task(5, True, 2), "n_turns": -1}),
        json.dumps({**turn(6, 0), "task_id": True}),  # a bool is not a task id
        json.dumps({**turn(7, 0), "type": ["turn"]}),  # invalid row types
        json.dumps({**turn(8, 0), "type": 5}),
        json.dumps({**turn(9, 0), "type": "note"}),
        json.dumps([turn(10, 0)]),  # not an object
        *(json.dumps(row) for row in session(11, "APPROVE", passed=False)),  # one valid session
    ]
    log = tmp_path / "log.jsonl"
    log.write_text("\n".join(lines) + "\n", encoding="utf-8")

    turns, tasks, load_issues = diag.load_rows([log])
    assert [issue["problem"] for issue in load_issues] == [
        "bad_field_types:turn",
        "bad_field_types:task_id",
        "bad_field_types:task_id",
        "bad_field_types:verdict,truncated",
        "bad_field_types:n_turns",
        "bad_field_types:task_id",
        "unknown_row_type",
        "unknown_row_type",
        "unknown_row_type",
        "row_not_object",
    ]
    report = diag.diagnose(turns, tasks)
    assert report["issues"] == []
    assert report["groups"]["r-A5-H1 A5-H1"]["rounds"]["1"]["false_approvals"] == 1

    # diagnose() also checks rows it is handed directly, so a caller that skips load_rows cannot crash it.
    raw = [json.loads(line) for line in lines[:2]] + [turn(12, 0), [1]]
    direct = diag.diagnose(raw, [task(12, True, 1), {**task(13, True, 1), "task_id": [13]}, turn(14, 0)])
    assert [issue["problem"] for issue in direct["issues"] if "where" in issue] == [
        "bad_field_types:turn", "bad_field_types:task_id", "row_not_object",
        "bad_field_types:task_id", "expected_task_row",
    ]
    assert direct["groups"]["r-A5-H1 A5-H1"]["sessions"] == 1


def test_diagnostics_cli_never_overwrites(tmp_path: Path) -> None:
    log = write_log(tmp_path, session(1, "APPROVE"))
    out = tmp_path / "report.json"
    assert diag.main([str(log), "--json", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["groups"]["r-A5-H1 A5-H1"]["sessions"] == 1
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        diag.main([str(log), "--json", str(out)])
    assert diag.main([str(write_log(tmp_path, session(1, "REVISE")[:2])), "--strict"]) == 1
