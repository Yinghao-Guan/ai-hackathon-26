import pytest

from bench.cost_sweep import build_repo, format_table, run_sweep, summarize_records
from rolekv import roles
from rolekv.costs import EventLog

SHORT_TURNS = [
    "```python\ndef f(x):\n    return x\n```",
    "REVISE: return x + 1",
    "```python\ndef f(x):\n    return x + 1\n```",
    "APPROVE",
]


def test_build_repo_reaches_target(tokenizer):
    repo = build_repo(tokenizer, 300)
    assert len(roles.segment_ids(tokenizer, repo)) >= 300
    assert repo == build_repo(tokenizer, 300)  # deterministic


def test_cost_sweep_smoke(model, tokenizer, tmp_path):
    methods = ["A2", "A5-H1", "A5-H2"]
    with EventLog(tmp_path / "sweep.jsonl") as log:
        records = run_sweep(model, tokenizer, methods=methods, ctx_sizes=[100, 400], repeats=2, log=log,
                            machine="mac", turns=SHORT_TURNS)
    assert len(records) == 2 * 2 * len(methods) * 5  # repeats x sizes x methods x (4 turns + 1 task line)
    assert len((tmp_path / "sweep.jsonl").read_text().splitlines()) == len(records)
    # one run_id per method and size, which the dashboard needs to place sweep points
    assert {r["run_id"] for r in records} == {f"sweep-{m}-ctx{k}k" for m in methods for k in ("0.1", "0.4")}

    rows = summarize_records(records)  # raises if token counts differ across repeats
    assert len(rows) == 2 * len(methods)
    sizes = sorted({r["ctx_tokens"] for r in rows})
    for size in sizes:
        row = {r["method"]: r for r in rows if r["ctx_tokens"] == size}
        a2, h1, h2 = (row[m]["computed_tokens"] for m in methods)
        assert a2 > h1 > h2
        assert h1 - h2 == row["A5-H1"]["canon_tokens"]  # same prefill each turn, plus the recompute

    def saving(size):
        return next(r["compute_vs_a2"] for r in rows if r["ctx_tokens"] == size and r["method"] == "A5-H1")

    assert saving(sizes[1]) > saving(sizes[0]) > 1  # the longer the shared context, the bigger the saving
    assert "A5-H1" in format_table(rows)
