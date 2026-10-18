"""The runtime's runner factory plugs into Member B's eval driver end to end."""

import json

import pytest

from rolekv import roles

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


@pytest.mark.parametrize("method", ["A2", "A5-H1"])
def test_member_b_driver_runs_runtime_sessions(model, tmp_path, method):
    from eval.run_eval import evaluate

    from rolekv.eval_runner import make_runner

    adapter_dir = tmp_path / "adapters"
    model.save_pretrained(adapter_dir)
    runner = make_runner(method=method, model=roles.DEV_MODEL, adapters=adapter_dir, device="cpu",
                         max_new_tokens={"coder": 6, "reviewer": 4})
    out = tmp_path / "eval.jsonl"
    emitted = evaluate(tasks=[TASK], training_rows=TRAINING_ROWS, runner=runner, method=method, run_id="t",
                       machine="mac", output_path=out, ctx_funcs=2)

    rows = [json.loads(line) for line in out.read_text().splitlines()]
    turns, task_row = rows[:-1], rows[-1]
    assert emitted == 1 and task_row["type"] == "task" and task_row["n_turns"] == len(turns) >= 2
    assert task_row["prefill_tokens_total"] == sum(t["prefill_tokens"] + t["canon_tokens"] for t in turns)
    assert all(t["method"] == method and "verdict" in t for t in turns)
    assert task_row["ctx_tokens"] > 0 and task_row["gpu_s"] > 0
