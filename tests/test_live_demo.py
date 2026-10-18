"""The live demo's session runner (demo/live/server.py) keeps run_session's protocol and reports the engine's costs."""

import importlib.util
from pathlib import Path

import pytest

from eval import testrun
from eval.testrun import run_mbpp_task
from rolekv import roles
from rolekv.costs import summarize
from rolekv.runtime import Engine, run_session
from rolekv.tasks import build_task_segment

SERVER = Path(__file__).resolve().parents[1] / "demo" / "live" / "server.py"
TASK = {
    "task_id": 999,
    "text": "Write a function square(x) that returns x squared.",
    "test_list": ["assert square(3) == 9", "assert square(0) == 0", "assert square(-2) == 4"],
    "test_setup_code": "",
}
TRAINING_ROWS = [
    {"task_id": 1, "code": "def add(a, b):\n    return a + b"},
    {"task_id": 2, "code": "def sub(a, b):\n    return a - b"},
]
LIMITS = {"coder": 6, "reviewer": 4}
COUNTERS = ("prefill_tokens", "reused_tokens", "canon_tokens", "decode_tokens", "text", "truncated", "verdict")


@pytest.fixture(scope="module")
def live():
    spec = importlib.util.spec_from_file_location("rolekv_live_server", SERVER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def engines(model, tokenizer):
    return {m: Engine(model, tokenizer, m) for m in ("A2", "A5-H1")}


def segment() -> str:
    return build_task_segment(TASK, TRAINING_ROWS, ctx_funcs=2)


def record_turns(engine: Engine) -> list:
    """Wrap engine.turn so the test sees every TurnResult the runner got."""
    results, turn = [], engine.turn

    def wrapped(*args, **kwargs):
        result = turn(*args, **kwargs)
        results.append(result)
        return result

    engine.turn = wrapped
    return results


def fixed_tool(code: str) -> testrun.TestResult:
    """A deterministic tool: real test runs print a random temp path, which changes token counts."""
    passed = run_mbpp_task(code, TASK).passed
    output = "" if passed else "AssertionError: square(3) != 9"
    return testrun.TestResult(passed=passed, output=output, stdout="", stderr=output, timed_out=False,
                      returncode=0 if passed else 1)


def check_stream(events, engines, recorded, tokenizer, *, order=("A2", "A5-H1")):
    """Event order, turn counters == the engine's TurnCost, final totals == summarize(); returns turn events."""
    assert events[0]["type"] == "start" and events[-1]["type"] == "done"
    assert events[0]["ctx_tokens"] == engines["A2"].ctx_tokens > 0
    body = events[1:-1]
    turns = {m: [e for e in body if e["type"] == "turn" and e["method"] == m] for m in engines}
    for m, rows in turns.items():
        assert [e["turn"] for e in rows] == list(range(len(recorded[m]))) and rows

    # One round = each unfinished method's next turn, in `order`; every turn is announced first.
    expected, round_ = [], 0
    while any(round_ < len(recorded[m]) for m in order):
        expected += [(m, round_) for m in order if round_ < len(recorded[m])]
        round_ += 1
    seen = [(e["method"], e["turn"]) for e in body if e["type"] == "turn"]
    assert seen == expected
    for i, event in enumerate(body):
        if event["type"] == "turn":
            start = body[i - 1]
            assert start["type"] == "turn_start"
            assert (start["method"], start["turn"], start["role"]) == (event["method"], event["turn"], event["role"])

    header = {role: len(roles.invocation_tokens(tokenizer, role)) for role in roles.ROLES}
    for m, rows in turns.items():
        for event, result in zip(rows, recorded[m], strict=True):
            cost = result.cost
            assert {k: event[k] for k in COUNTERS} == {k: getattr(cost, k) for k in COUNTERS}
            assert event["computed_tokens"] == cost.prefill_tokens + cost.canon_tokens
            assert event["ttft_ms"] == pytest.approx(cost.ttft_ms, abs=0.01)
            assert event["turn_ms"] == pytest.approx(cost.turn_ms, abs=0.01)
            # CONTRACT.md section 3: reused + canon + prefill = the session up to this turn's role header.
            assert cost.reused_tokens + cost.canon_tokens + cost.prefill_tokens == result.header_pos + header[event["role"]]
            assert (event["tool"] is not None) == (event["role"] == "coder" and not event["last"])
        assert rows[-1]["last"] and not any(e["last"] for e in rows[:-1])

        finals = [e for e in body if e["type"] == "final" and e["method"] == m]
        assert len(finals) == 1
        final = finals[0]
        assert body.index(final) == body.index(rows[-1]) + 1  # right after the method's last turn
        total = summarize([r.cost for r in recorded[m]], engines[m].ctx_tokens)
        assert final["n_turns"] == total.n_turns == len(rows)
        assert final["prefill_tokens_total"] == total.prefill_tokens_total
        assert final["canon_tokens_total"] == total.canon_tokens_total
        assert final["decode_tokens_total"] == total.decode_tokens_total
        assert final["gpu_s"] == pytest.approx(total.gpu_s, abs=1e-4)
        assert final["usd"] == pytest.approx(total.gpu_s * final["usd_per_gpu_hour"] / 3600)
        assert final["passed"] == run_mbpp_task(final["final_code"], TASK).passed
    assert all(e["reused_tokens"] == 0 for e in turns["A2"])  # A2 re-reads everything at every turn
    return turns


def test_live_session_streams_engine_costs(live, engines, tokenizer):
    recorded = {m: record_turns(e) for m, e in engines.items()}
    events = list(live.compare_sessions(engines, TASK, segment(), max_new_tokens=LIMITS, usd_per_gpu_hour=0.5))
    turns = check_stream(events, engines, recorded, tokenizer)
    a5 = turns["A5-H1"]
    assert all(e["reused_tokens"] > 0 for e in a5[1:])  # RoleKV serves the shared context from the cache
    for a2_turn, a5_turn in zip(turns["A2"][1:], a5[1:]):
        assert a5_turn["computed_tokens"] < a2_turn["computed_tokens"]


def test_live_log_records_what_the_eval_would(live, engines, tmp_path):
    """demo/live/live_log.py writes the same records as rolekv.costs.turn_record / task_record, so the dashboard
    reads a live session exactly like an eval run. Floats are rounded in the stream, hence approx."""
    import json

    from rolekv.costs import task_record, turn_record

    recorded = {m: record_turns(e) for m, e in engines.items()}
    request = {"task": TASK, "ctx_funcs": 2, "mirror": False, "usd_per_gpu_hour": 0.5}
    log = live.LiveLog.for_session(tmp_path, "4090")
    for event in live.compare_sessions(engines, TASK, segment(), max_new_tokens=LIMITS, run_tool=fixed_tool):
        log.observe(event, request)
    rows = [json.loads(line) for line in log.path.read_text(encoding="utf-8").splitlines()]

    for m, results in recorded.items():
        run_id = log.run_id(m, 2, False)
        mine = [r for r in rows if r["run_id"] == run_id]
        ids = {"run_id": run_id, "machine": "4090", "method": m, "task_id": TASK["task_id"]}
        expected = [turn_record(r.cost, **ids, turn=i) for i, r in enumerate(results)]
        cost = summarize([r.cost for r in results], engines[m].ctx_tokens)
        expected.append(task_record(cost, **ids, passed=mine[-1]["passed"]))
        assert len(mine) == len(expected)
        for got, want in zip(mine, expected, strict=True):
            assert set(got) == set(want)
            for key, value in want.items():
                if isinstance(value, float):
                    assert got[key] == pytest.approx(value, abs=0.01), key
                else:
                    assert got[key] == value, key
        assert isinstance(mine[-1]["passed"], bool)


def test_protocol_matches_run_session(live, engines, model, tokenizer):
    """Two REVISEs, then a final Coder turn: same turns, verdicts and token counts as run_session."""
    wrong = "```python\ndef square(x):\n    return x + x\n```"
    outputs = [wrong, "REVISE: square, do not double", wrong, "REVISE: use x * x", "```python\ndef square(x):\n    return x * x\n```"]
    recorded = {m: record_turns(e) for m, e in engines.items()}
    events = list(live.compare_sessions(engines, TASK, segment(), forced_outputs={m: outputs for m in engines},
                                        run_tool=fixed_tool))
    turns = check_stream(events, engines, recorded, tokenizer)

    for method, rows in turns.items():
        assert [e["role"] for e in rows] == ["coder", "reviewer"] * 2 + ["coder"]
        assert [e["verdict"] for e in rows] == [None, "REVISE", None, "REVISE", None]
        assert [e["tool"]["passed"] if e["tool"] else None for e in rows] == [False, None, False, None, None]
        final = next(e for e in events if e["type"] == "final" and e["method"] == method)
        assert final["passed"] and final["end"] == "LAST_REVISION" and final["verdicts"] == ["REVISE", "REVISE"]

        reference = run_session(Engine(model, tokenizer, method), segment(), lambda code: fixed_tool(code).output,
                                forced_outputs=outputs)
        assert reference.final_code == final["final_code"]
        keys = ("role", "prefill_tokens", "reused_tokens", "canon_tokens", "decode_tokens", "verdict")
        assert [{k: e[k] for k in keys} for e in rows] == [{k: getattr(t.cost, k) for k in keys} for t in reference.turns]
    assert turns["A2"][-1]["computed_tokens"] > 3 * turns["A5-H1"][-1]["computed_tokens"]


def test_mirror_replays_the_leader(live, engines, tokenizer):
    recorded = {m: record_turns(e) for m, e in engines.items()}
    events = list(live.compare_sessions(engines, TASK, segment(), max_new_tokens=LIMITS, mirror=True))
    turns = check_stream(events, engines, recorded, tokenizer, order=("A5-H1", "A2"))
    for a2_turn, a5_turn in zip(turns["A2"], turns["A5-H1"], strict=True):
        assert (a2_turn["text"], a2_turn["decode_tokens"], a2_turn["tool"]) == (a5_turn["text"], a5_turn["decode_tokens"], a5_turn["tool"])
    assert [r.gen_ids for r in recorded["A2"]] == [r.gen_ids for r in recorded["A5-H1"]]


def test_should_stop_cancels_between_turns(live, engines):
    calls = []

    def stop() -> bool:
        calls.append(1)
        return len(calls) > 1  # let exactly one turn run

    events = list(live.compare_sessions(engines, TASK, segment(), max_new_tokens=LIMITS, should_stop=stop))
    assert [e["type"] for e in events] == ["start", "turn_start", "turn", "cancelled"]


def test_run_requests(live):
    tasks = {TASK["task_id"]: TASK}
    req = live.parse_run_request({"task_id": "999", "ctx_funcs": 60, "usd_per_gpu_hour": 1.2}, tasks, 30)
    assert req["task"] is TASK and req["ctx_funcs"] == 60 and req["usd_per_gpu_hour"] == 1.2 and not req["mirror"]
    assert live.parse_run_request({"task_id": 999}, tasks, 30)["ctx_funcs"] == 30

    custom = live.parse_run_request({"custom": {"text": " Add two numbers. ", "asserts": "assert add(1, 2) == 3\n\n assert add(0, 0) == 0 "}},
                                    tasks, 30)["task"]
    assert custom["text"] == "Add two numbers." and custom["test_list"] == ["assert add(1, 2) == 3", "assert add(0, 0) == 0"]
    assert build_task_segment(custom, TRAINING_ROWS, ctx_funcs=2).count("assert add(1, 2) == 3") == 1  # visible test

    for bad in ({"task_id": 5}, {"task_id": None}, {"task_id": 999, "ctx_funcs": 500},
                {"custom": {"text": "", "asserts": "assert 1"}}, {"custom": {"text": "x", "asserts": ""}},
                {"custom": {"text": "x", "asserts": "print(1)"}}):
        with pytest.raises(ValueError):
            live.parse_run_request(bad, tasks, 30)
