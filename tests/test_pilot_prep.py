"""Metered teacher (usage log, spend guard, bounded retries) and eval/check_multiround_pilot.py.

A fake Messages client, a fake test runner and fixture MBPP splits stand in for the API, the subprocess harness and
the Hugging Face dataset: no API calls, no generated-code execution, no downloads.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from eval import check_multiround_pilot as pilot
from eval.testrun import TestResult as HarnessResult
from rolekv import roles
from train import build_dataset as bd

MODEL = "claude-sonnet-5-5"
IN_PRICE, OUT_PRICE = (price / 1_000_000 for price in bd.PRICES_PER_MTOK[MODEL])
TRAINING = [{"task_id": i, "code": f"def helper_{i}(x):\n    return x + {i}\n", "text": f"Task {i}.",
             "test_list": [f"assert f{i}(1) == 1"]} for i in range(1, 13)]
EVALUATION = [{"task_id": i, "code": "", "text": f"Eval {i}.", "test_list": ["assert True"]} for i in range(500, 503)]
SPLITS = SimpleNamespace(training=TRAINING, evaluation=EVALUATION)


def fake_run(code: str, task: dict) -> HarnessResult:
    passed = re.search(r"def (\w+)", code).group(1).startswith("good")
    output = "" if passed else f"Traceback (most recent call last):\nAssertionError: task {task['task_id']}"
    return HarnessResult(passed=passed, output=output, stdout="", stderr=output, timed_out=False,
                         returncode=0 if passed else 1)


def reply(prompt: str) -> str:
    """Teacher replies: a buggy first attempt (it passes for task 9), REVISE feedback, still-buggy or good repairs."""
    found = re.search(r"# Task\nTask (\d+)\.", prompt) or re.search(r"def \w+?_(\d+)\(", prompt)
    tid = int(found.group(1))
    if prompt.startswith("Review this candidate"):
        return f"REVISE: wrong for task {tid}"
    if "subtly buggy first attempt" in prompt:
        name = "good_anyway" if tid == 9 else "bad1"
    elif "still subtly wrong" in prompt:
        name = "bad2"
    else:
        name = "good"
    return f"```python\ndef {name}_{tid}(x):\n    return x\n```"


class FakeMessages:
    def __init__(self, fail: bool = False) -> None:
        self.calls: list[dict] = []
        self.fail = fail

    def create(self, **request) -> SimpleNamespace:
        self.calls.append(request)
        if self.fail:
            raise ConnectionError("network down")
        prompt = request["messages"][0]["content"]
        text = reply(prompt)
        usage = SimpleNamespace(input_tokens=len(prompt) // 4, output_tokens=len(text) // 4 + 5,
                                cache_creation_input_tokens=0, cache_read_input_tokens=0)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=text)], usage=usage,
                               stop_reason="end_turn", _request_id=f"req_{len(self.calls)}")


def ledger(tmp_path: Path, *, cap: float = 5.0, max_calls: int | None = None) -> bd.UsageLedger:
    return bd.UsageLedger(model=MODEL, cap_usd=cap, max_calls=max_calls, log_path=tmp_path / "api_usage.jsonl")


def metered_teacher(tmp_path: Path, messages: FakeMessages, **kwargs) -> bd.Teacher:
    return bd.Teacher(model=MODEL, system_prompt="teacher system", ledger=ledger(tmp_path, **kwargs),
                      client=SimpleNamespace(messages=messages))


PROMPT = "Solve the following task. Return exactly one Python code block.\n\n# Task\nTask 3.\n"


# --- usage ledger and spend guard ---------------------------------------------------------------------------


def test_each_request_is_logged_and_charged_from_its_usage(tmp_path: Path) -> None:
    messages = FakeMessages()
    teacher = metered_teacher(tmp_path, messages)
    teacher.ledger.current_task = 3
    assert teacher.ask(PROMPT, max_tokens=512).startswith("```python")

    (entry,) = [json.loads(line) for line in (tmp_path / "api_usage.jsonl").read_text(encoding="utf-8").splitlines()]
    expected = entry["input_tokens"] * IN_PRICE + entry["output_tokens"] * OUT_PRICE
    assert entry["charged_usd"] == pytest.approx(expected)
    assert entry["reserved_usd"] == pytest.approx(teacher.ledger.reservation(
        system="teacher system", prompt=PROMPT, max_tokens=512))
    assert (entry["task_id"], entry["purpose"], entry["stop_reason"], entry["request_id"], entry["error"]) == (
        3, "clean", "end_turn", "req_1", None)
    assert b"\r" not in (tmp_path / "api_usage.jsonl").read_bytes()


def test_reservation_bounds_the_request_by_bytes_and_max_tokens(tmp_path: Path) -> None:
    book = ledger(tmp_path)
    prompt = "é" * 10  # 20 UTF-8 bytes: bytes, not characters, bound the tokens
    expected = (len("sys") + 20 + bd.REQUEST_OVERHEAD_TOKENS) * IN_PRICE + 128 * OUT_PRICE
    assert book.reservation(system="sys", prompt=prompt, max_tokens=128) == pytest.approx(expected)


def test_guard_refuses_before_sending(tmp_path: Path) -> None:
    messages = FakeMessages()
    teacher = metered_teacher(tmp_path, messages, cap=0.001)  # below one request's reservation
    with pytest.raises(bd.SpendLimitReached, match="cap"):
        teacher.ask(PROMPT, max_tokens=512)
    assert messages.calls == [] and (tmp_path / "api_usage.jsonl").read_text(encoding="utf-8") == ""


def test_call_cap(tmp_path: Path) -> None:
    messages = FakeMessages()
    teacher = metered_teacher(tmp_path, messages, max_calls=2)
    teacher.ask(PROMPT, max_tokens=64)
    teacher.ask(PROMPT, max_tokens=64)
    with pytest.raises(bd.SpendLimitReached, match="call cap 2"):
        teacher.ask(PROMPT, max_tokens=64)
    assert len(messages.calls) == 2


def test_failed_request_is_logged_and_charged_its_reservation(tmp_path: Path) -> None:
    teacher = metered_teacher(tmp_path, FakeMessages(fail=True))
    with pytest.raises(bd.TeacherCallFailed, match="network down"):
        teacher.ask(PROMPT, max_tokens=512)
    (entry,) = teacher.ledger.records
    assert entry["error"] == "ConnectionError: network down"
    assert entry["charged_usd"] == entry["reserved_usd"] > 0


def test_metered_client_never_retries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import anthropic

    created: list[dict] = []
    monkeypatch.setattr(anthropic, "Anthropic",
                        lambda **options: created.append(options) or SimpleNamespace(max_retries=options.get("max_retries", 2)))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-not-a-key")
    bd.Teacher(model=MODEL, system_prompt="s")
    bd.Teacher(model=MODEL, system_prompt="s", ledger=ledger(tmp_path), timeout=30.0)
    assert created == [{}, {"max_retries": 0, "timeout": 30.0}]  # the unmetered client keeps the SDK defaults
    retrying = SimpleNamespace(messages=FakeMessages(), max_retries=2)
    with pytest.raises(ValueError, match="max_retries=0"):
        bd.Teacher(model=MODEL, system_prompt="s", ledger=bd.UsageLedger(
            model=MODEL, cap_usd=1, max_calls=None, log_path=tmp_path / "other.jsonl"), client=retrying)


def test_a_later_success_never_erases_an_earlier_failed_attempt(tmp_path: Path) -> None:
    messages = FakeMessages(fail=True)
    teacher = metered_teacher(tmp_path, messages)
    with pytest.raises(bd.TeacherCallFailed):
        teacher.ask(PROMPT, max_tokens=512)
    messages.fail = False
    teacher.ask(PROMPT, max_tokens=512)
    failed, succeeded = teacher.ledger.records
    assert len(messages.calls) == 2 and failed["error"] and succeeded["error"] is None
    assert failed["charged_usd"] == failed["reserved_usd"]
    assert teacher.ledger.spent_usd == pytest.approx(failed["reserved_usd"] + succeeded["charged_usd"])
    logged = [json.loads(line) for line in (tmp_path / "api_usage.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [entry["call"] for entry in logged] == [1, 2]


def test_ledger_refuses_unknown_prices_and_existing_logs(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no price"):
        bd.UsageLedger(model="some-other-model", cap_usd=1, max_calls=None, log_path=tmp_path / "a")
    (tmp_path / "b").write_text("old\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        bd.UsageLedger(model=MODEL, cap_usd=1, max_calls=None, log_path=tmp_path / "b")
    assert (tmp_path / "b").read_text(encoding="utf-8") == "old\n"


# --- main() with the metered teacher ----------------------------------------------------------------------------


def run_main(monkeypatch: pytest.MonkeyPatch, out: Path, *extra: str, fail: bool = False) -> FakeMessages:
    messages = FakeMessages(fail=fail)
    real_teacher = bd.Teacher
    monkeypatch.setattr(bd, "Teacher", lambda **kwargs: real_teacher(**kwargs, client=SimpleNamespace(messages=messages)))
    monkeypatch.setattr(bd, "load_mbpp_splits", lambda: SPLITS)
    monkeypatch.setattr(bd, "run_mbpp_task", fake_run)
    monkeypatch.setattr(bd, "load_dotenv", lambda: None)
    monkeypatch.setattr(sys, "argv", ["build_dataset", "--multiround", "--bug-rate", "1", "--second-bug-rate", "0.5",
                                      "--seed", "0", "--limit", "8", "--ctx-funcs", "2", "--out", str(out),
                                      "--max-spend-usd", "5", *extra])
    bd.main()
    return messages


def read_report(out: Path) -> dict:
    return json.loads((out / "generation_report.json").read_text(encoding="utf-8"))


def test_metered_run_accounts_for_kept_and_rejected_tasks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "run"
    messages = run_main(monkeypatch, out)
    report = read_report(out)
    log = [json.loads(line) for line in (out / "api_usage.jsonl").read_text(encoding="utf-8").splitlines()]

    assert report["api"]["calls"] == len(log) == len(messages.calls)
    assert report["api"]["charged_usd"] == pytest.approx(sum(entry["charged_usd"] for entry in log))
    by_outcome = report["api"]["charged_usd_by_outcome"]
    assert set(by_outcome) == {"kept", "buggy_passed"} and by_outcome["buggy_passed"] > 0  # task 9's request counts
    assert report["stopped"] is None and report["attempted"] == report["selected"] == 8
    assert report["settings"]["max_spend_usd"] == 5.0 and report["api"]["http_attempts_per_call"] == 1
    assert {entry["purpose"] for entry in log} == {"buggy", "review", "repair", "still_buggy_repair"}
    assert all(request["model"] == MODEL for request in messages.calls)


def test_call_cap_stops_the_run_and_keeps_finished_work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "run"
    messages = run_main(monkeypatch, out, "--max-calls", "7")
    report = read_report(out)
    assert len(messages.calls) == 7 and "call cap 7" in report["stopped"]
    assert report["rejections"].get("spend_limit") == 1
    assert report["attempted"] == report["kept"] + sum(report["rejections"].values()) < report["selected"]
    assert len((out / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()) == report["kept"]


def test_repeated_api_errors_stop_the_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    out = tmp_path / "run"
    run_main(monkeypatch, out, fail=True)
    report = read_report(out)
    assert report["rejections"] == {"api_error": 3} and report["attempted"] == 3
    assert report["api"]["errors"] == 3 and "3 tasks in a row" in report["stopped"]
    assert report["api"]["charged_usd_by_outcome"] == {"api_error": pytest.approx(report["api"]["charged_usd"])}


def test_multiround_requires_a_spend_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bd, "Teacher", lambda **_: pytest.fail("no teacher before the argument check"))
    monkeypatch.setattr(sys, "argv", ["build_dataset", "--multiround", "--out", str(tmp_path / "run")])
    with pytest.raises(SystemExit, match="needs --max-spend-usd"):
        bd.main()
    monkeypatch.setattr(sys, "argv", ["build_dataset", "--multiround", "--model", "unknown-model", "--max-spend-usd",
                                      "5", "--out", str(tmp_path / "run")])
    with pytest.raises(SystemExit, match="no price recorded"):
        bd.main()
    assert not (tmp_path / "run").exists()


def test_default_run_stays_unmetered(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    constructed: list[dict] = []

    class Scripted:
        def __init__(self, **kwargs) -> None:
            constructed.append(kwargs)

        def ask(self, prompt: str, *, max_tokens: int) -> str:
            return reply(prompt)

    monkeypatch.setattr(bd, "Teacher", Scripted)
    monkeypatch.setattr(bd, "load_mbpp_splits", lambda: SPLITS)
    monkeypatch.setattr(bd, "run_mbpp_task", fake_run)
    monkeypatch.setattr(bd, "load_dotenv", lambda: None)
    out = tmp_path / "default"
    monkeypatch.setattr(sys, "argv", ["build_dataset", "--limit", "4", "--ctx-funcs", "2", "--out", str(out)])
    bd.main()
    assert constructed == [{"model": MODEL, "system_prompt": roles.SYSTEM_PROMPT}]
    assert sorted(path.name for path in out.iterdir()) == ["README.md", "sft_coder.jsonl", "sft_reviewer.jsonl",
                                                            "trajectories.jsonl"]


# --- pilot checker ---------------------------------------------------------------------------------------------


@pytest.fixture
def pilot_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "run"
    run_main(monkeypatch, out)
    return out


def check(path: Path, tokenizer, **kwargs) -> dict:
    return pilot.check_pilot(path, tokenizer=tokenizer, splits=SPLITS, **kwargs)


def rewrite(out: Path, trajectories: list[dict]) -> None:
    """Rewrite a run's files consistently around edited trajectories, so one problem can be isolated."""
    bd.write_jsonl(out / "trajectories.jsonl", trajectories)
    stats = bd.export_sft(trajectories, out)
    bd.write_readme(out / "README.md", trajectories=len(trajectories), stats=stats)


def load(out: Path) -> list[dict]:
    return [json.loads(line) for line in (out / "trajectories.jsonl").read_text(encoding="utf-8").splitlines()]


def test_checker_accepts_a_valid_run_and_counts_second_reviews(pilot_dir: Path, tokenizer) -> None:
    result = check(pilot_dir, tokenizer, require_second_round_labels=True)
    assert result["ok"], result["errors"]
    assert result["warnings"] == []
    rounds = result["reviewer_by_round"]
    assert rounds["1"]["REVISE"]["all"] == result["kept"]  # every kept run started with a buggy attempt
    assert rounds["2"]["APPROVE"]["within_limit"] >= 1 and rounds["2"]["REVISE"]["within_limit"] >= 1
    assert result["lengths"]["reviewer"]["over_limit"] == 0
    assert result["separation"]["duplicate_task_ids"] == []


def test_length_limit_is_reported_before_and_after_never_truncated(pilot_dir: Path, tokenizer) -> None:
    rows = [json.loads(line) for line in (pilot_dir / "sft_reviewer.jsonl").read_text(encoding="utf-8").splitlines()]
    lengths = [len(roles.tokenize_sft_example(tokenizer, r["segments"], r["target"])["input_ids"]) for r in rows]
    limit = sorted(lengths)[len(lengths) // 2]
    result = check(pilot_dir, tokenizer, max_length=limit)
    assert result["lengths"]["reviewer"]["max"] == max(lengths)  # measured on the untruncated rows
    assert result["lengths"]["reviewer"]["over_limit"] == sum(n > limit for n in lengths) > 0
    for review_round, labels in result["reviewer_by_round"].items():
        for label, cell in labels.items():
            picked = [n for r, n in zip(rows, lengths) if str(r["segments"].count(roles.role_header("reviewer")))
                      == review_round and r["target"].startswith(label)]
            assert cell == {"all": len(picked), "within_limit": sum(n <= limit for n in picked)}


def test_checker_requires_both_second_round_labels(pilot_dir: Path, tokenizer) -> None:
    kept = [t for t in load(pilot_dir) if len(t["turns"]) != 4]  # drop every second-round APPROVE
    rewrite(pilot_dir, kept)
    report = read_report(pilot_dir)
    report["kept"], report["shapes"] = len(kept), {"two_revisions": len(kept)}
    report["rejections"]["dropped_for_test"] = report["attempted"] - len(kept) - sum(report["rejections"].values())
    (pilot_dir / "generation_report.json").write_text(json.dumps(report), encoding="utf-8")
    assert check(pilot_dir, tokenizer)["ok"]
    result = check(pilot_dir, tokenizer, require_second_round_labels=True)
    assert result["errors"] == ["second-round coverage missing: need at least one APPROVE and one REVISE within the limit"]


def test_checker_catches_hash_and_line_ending_problems(pilot_dir: Path, tokenizer) -> None:
    path = pilot_dir / "sft_coder.jsonl"
    path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    errors = check(pilot_dir, tokenizer)["errors"]
    assert "sft_coder.jsonl contains CR bytes; generated JSONL must be LF-only" in errors
    assert any(e.startswith("README.md does not list the SHA-256 of sft_coder.jsonl") for e in errors)


def test_checker_catches_schema_and_export_mismatches(pilot_dir: Path, tokenizer) -> None:
    rows = (pilot_dir / "sft_reviewer.jsonl").read_text(encoding="utf-8").splitlines()
    first = {**json.loads(rows[0]), "label": "extra"}
    data = "\n".join([json.dumps(first)] + rows[1:]) + "\n"
    (pilot_dir / "sft_reviewer.jsonl").write_text(data, encoding="utf-8", newline="\n")
    readme = pilot_dir / "README.md"
    readme.write_text(readme.read_text(encoding="utf-8") + f"\n{__import__('hashlib').sha256(data.encode()).hexdigest()}\n",
                      encoding="utf-8")
    assert any("fields must be exactly" in e for e in check(pilot_dir, tokenizer)["errors"])


def test_checker_catches_protocol_breaks(pilot_dir: Path, tokenizer) -> None:
    trajectories = load(pilot_dir)
    broken = trajectories[0]
    broken["turns"][1]["segments"] = broken["turns"][1]["segments"][:-2] + [broken["turns"][1]["segments"][-1]]
    rewrite(pilot_dir, trajectories)
    errors = check(pilot_dir, tokenizer)["errors"]
    assert "trajectory 0 turn 1: context does not extend the previous turn with the contractual segments" in errors


def test_checker_catches_evaluation_tasks_and_foreign_task_text(pilot_dir: Path, tokenizer) -> None:
    trajectories = load(pilot_dir)
    held_out = trajectories[0]["task_id"]
    splits = SimpleNamespace(training=TRAINING, evaluation=EVALUATION + [{"task_id": held_out}])
    errors = pilot.check_pilot(pilot_dir, tokenizer=tokenizer, splits=splits)["errors"]
    assert f"task {held_out} is a held-out evaluation task" in errors

    report = read_report(pilot_dir)
    report["settings"]["ctx_funcs"] = 3  # the TASK segments were built with 2
    (pilot_dir / "generation_report.json").write_text(json.dumps(report), encoding="utf-8")
    errors = check(pilot_dir, tokenizer)["errors"]
    assert f"task {held_out}: TASK segment differs from the one built from the training rows" in errors


def test_checker_catches_report_and_usage_log_mismatches(pilot_dir: Path, tokenizer) -> None:
    log = pilot_dir / "api_usage.jsonl"
    log.write_text("".join(log.read_text(encoding="utf-8").splitlines(keepends=True)[:-1]), encoding="utf-8")
    report = read_report(pilot_dir)
    report["kept"] += 1
    (pilot_dir / "generation_report.json").write_text(json.dumps(report), encoding="utf-8")
    errors = check(pilot_dir, tokenizer)["errors"]
    assert "api_usage.jsonl does not add up to the report's API totals" in errors
    assert any(e.startswith("generation_report.json says kept=") for e in errors)
    assert "generation_report.json: kept + rejections != attempted" in errors


def test_checker_cli_never_overwrites(pilot_dir: Path, tokenizer, monkeypatch: pytest.MonkeyPatch) -> None:
    import transformers

    import rolekv.tasks

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_DATASETS_OFFLINE", "1")

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **k: tokenizer)
    monkeypatch.setattr(rolekv.tasks, "load_mbpp_splits", lambda: SPLITS)
    out = pilot_dir / "check.json"
    assert pilot.main([str(pilot_dir), "--json", str(out), "--require-second-round-labels"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["ok"] is True
    with pytest.raises(SystemExit, match="refusing to overwrite"):
        pilot.main([str(pilot_dir), "--json", str(out)])


def _drop(key: str):
    return lambda t: t["turns"][0].pop(key)


def _set_turn(key: str, value):
    return lambda t: t["turns"][0].__setitem__(key, value)


def _set(key: str, value):
    return lambda t: t.__setitem__(key, value)


@pytest.mark.parametrize(("mutate", "message"), [
    (_drop("segments"), "turn 0: fields must be exactly"),
    (_drop("target"), "turn 0: fields must be exactly"),
    (_set_turn("segments", "not a list"), "turn 0: bad role, segments or target types"),
    (_set_turn("segments", [1, 2]), "turn 0: bad role, segments or target types"),
    (_set_turn("segments", []), "turn 0: bad role, segments or target types"),
    (_set_turn("target", 5), "turn 0: bad role, segments or target types"),
    (_set_turn("role", ["coder"]), "turn 0: bad role, segments or target types"),
    (lambda t: t["turns"].__setitem__(0, "not a turn"), "turn 0: fields must be exactly"),
    (_set("task_id", "7"), "needs an integer task_id"),
    (_set("task_id", True), "needs an integer task_id"),
    (_set("turns", "x"), "needs an integer task_id, a non-empty turns list"),
    (_set("turns", []), "needs an integer task_id, a non-empty turns list"),
    (_set("final_code", None), "a final_code string"),
    (lambda t: t.pop("final_code"), "fields must be exactly ['final_code', 'task_id', 'turns']"),
    (_set("extra", 1), "fields must be exactly ['final_code', 'task_id', 'turns']"),
])
def test_checker_reports_malformed_trajectories_without_raising(pilot_dir: Path, tokenizer, mutate, message) -> None:
    trajectories = load(pilot_dir)
    mutate(trajectories[0])
    bd.write_jsonl(pilot_dir / "trajectories.jsonl", trajectories)  # the SFT files are left as they were
    result = check(pilot_dir, tokenizer)
    assert not result["ok"]
    assert any(error.startswith("trajectory 0") and message in error for error in result["errors"]), result["errors"]


def test_checker_reports_a_non_object_trajectory_line(pilot_dir: Path, tokenizer) -> None:
    path = pilot_dir / "trajectories.jsonl"
    path.write_text("[1, 2]\n" + path.read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    result = check(pilot_dir, tokenizer)
    assert "trajectory 0: fields must be exactly ['final_code', 'task_id', 'turns']" in result["errors"]


def test_checker_rejects_final_code_that_differs_from_the_last_coder_turn(pilot_dir: Path, tokenizer) -> None:
    trajectories = load(pilot_dir)
    assert {len(t["turns"]) for t in trajectories} == {4, 5}  # covers a Reviewer-final and a Coder-final shape
    assert check(pilot_dir, tokenizer)["ok"]
    trajectories[0]["final_code"] = trajectories[0]["final_code"].replace("return x", "return x + 1")
    rewrite(pilot_dir, trajectories)
    errors = check(pilot_dir, tokenizer)["errors"]
    assert "trajectory 0: final_code does not match the code in the last Coder turn" in errors


@pytest.mark.parametrize(("content", "message"), [
    ("{not json", "generation_report.json is not valid JSON"),
    ("[1]", "generation_report.json: must be a JSON object"),
    (json.dumps({"kept": "3", "attempted": 8, "shapes": {}, "rejections": {}}),
     "generation_report.json: kept and attempted must be integers"),
    (json.dumps({"kept": 3, "attempted": 8, "shapes": {}, "rejections": {"x": "1"}}),
     "generation_report.json: shapes and rejections must be objects, rejections with integer counts"),
    (json.dumps({"kept": 3, "attempted": 8, "shapes": {}, "rejections": {}, "settings": {"ctx_funcs": "2"}}),
     "generation_report.json: settings must be an object with an integer ctx_funcs"),
    (json.dumps({"kept": 3, "attempted": 8, "shapes": {}, "rejections": {}, "api": {"calls": 1}}),
     "generation_report.json: api needs integer calls and numeric charged_usd and cap_usd"),
])
def test_checker_reports_a_malformed_report_without_raising(pilot_dir: Path, tokenizer, content, message) -> None:
    (pilot_dir / "generation_report.json").write_text(content, encoding="utf-8")
    assert message in check(pilot_dir, tokenizer)["errors"]


def test_checker_reports_usage_records_without_a_cost(pilot_dir: Path, tokenizer) -> None:
    log = pilot_dir / "api_usage.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    first = json.loads(lines[0])
    del first["charged_usd"]
    log.write_text("\n".join([json.dumps(first)] + lines[1:]) + "\n", encoding="utf-8", newline="\n")
    assert "api_usage.jsonl has a record without a numeric charged_usd" in check(pilot_dir, tokenizer)["errors"]
