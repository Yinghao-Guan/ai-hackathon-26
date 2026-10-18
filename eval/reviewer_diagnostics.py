"""Reviewer verdicts by method and review round, read from existing event logs (CONTRACT.md section 3).

    uv run --no-sync python -m eval.reviewer_diagnostics eval/results/eval-1.5b-4090-202610180231.jsonl

It only reads logs: no model, no generated-code execution, no network. Review round r is a session's r-th
Reviewer turn. A *false approval* is a session whose last turn is a Reviewer APPROVE, whose single matching
task record says ``passed: false``, and which has no protocol or completeness issue. A terminal APPROVE in a
flagged session (for example one without exactly one matching task record) is counted as *unverified*, never as
passed or failed; its verdicts still appear in the raw verdict counts. Rows whose fields have the wrong type are
reported and left out before any grouping.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping, Sequence


VERDICTS = ("APPROVE", "REVISE", "INVALID")
MAX_TURNS = 5  # session protocol: Coder, then at most two REVISE rounds, so at most 5 turns
TURN_FIELDS = ("run_id", "method", "task_id", "turn", "role", "verdict", "truncated")
TASK_FIELDS = ("run_id", "method", "task_id", "passed", "n_turns")

SessionKey = tuple[str, str, Any]  # (run_id, method, task_id)


def _is_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


# Type checks for the fields used to group, sort and count; ``passed`` is checked per session instead, so a
# bad value there leaves the record visible as an issue of its session.
FIELD_TYPES = {
    "run_id": lambda v: isinstance(v, str),
    "method": lambda v: isinstance(v, str),
    "task_id": lambda v: isinstance(v, str) or (isinstance(v, int) and not isinstance(v, bool)),
    "turn": _is_count,
    "role": lambda v: isinstance(v, str),
    "verdict": lambda v: v is None or isinstance(v, str),
    "truncated": lambda v: isinstance(v, bool),
    "n_turns": _is_count,
    "passed": lambda v: True,
}


def load_rows(paths: Iterable[Path]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return (turn rows, task rows, issues) from JSONL event logs."""

    turns: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for path in paths:
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            where = f"{path}:{number}"
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                issues.append({"where": where, "problem": "malformed_json"})
                continue
            problem = row_problem(row)
            if problem is not None:
                issues.append({"where": where, "problem": problem})
                continue
            (turns if row["type"] == "turn" else tasks).append(row)
    return turns, tasks, issues


def row_problem(row: Any) -> str | None:
    """Why a parsed row can't be used, or None; checked before any grouping, sorting or counting."""

    if not isinstance(row, dict):
        return "row_not_object"
    kind = row.get("type")
    if kind not in ("turn", "task"):  # also rejects unhashable or non-string types without raising
        return "unknown_row_type"
    fields = TURN_FIELDS if kind == "turn" else TASK_FIELDS
    missing = [field for field in fields if field not in row]
    if missing:
        return f"missing_fields:{','.join(missing)}"
    bad = [field for field in fields if not FIELD_TYPES[field](row[field])]
    if bad:
        return f"bad_field_types:{','.join(bad)}"
    return None


def _key(row: Mapping[str, Any]) -> SessionKey:
    return (row["run_id"], row["method"], row["task_id"])


def session_problems(turns: Sequence[Mapping[str, Any]], task_records: Sequence[Mapping[str, Any]]) -> list[str]:
    """Protocol and completeness problems of one session; ``turns`` sorted by turn index."""

    problems = []
    if not turns:
        return ["task_without_turns"]
    if len(task_records) == 0:
        problems.append("missing_task_record")
    elif len(task_records) > 1:
        problems.append("duplicate_task_record")
    elif task_records[0]["n_turns"] != len(turns):
        problems.append("n_turns_mismatch")
    if len(task_records) == 1 and not isinstance(task_records[0]["passed"], bool):
        problems.append("passed_not_bool")
    if [turn["turn"] for turn in turns] != list(range(len(turns))):
        problems.append("turn_indices_not_contiguous")
    if len(turns) > MAX_TURNS:
        problems.append("too_many_turns")
    for index, turn in enumerate(turns):
        expected_role = "coder" if index % 2 == 0 else "reviewer"
        if turn["role"] != expected_role:
            problems.append(f"turn_{index}_role_{turn['role']}")
        elif expected_role == "coder" and turn["verdict"] is not None:
            problems.append(f"turn_{index}_coder_verdict")
        elif expected_role == "reviewer" and turn["verdict"] not in VERDICTS:
            problems.append(f"turn_{index}_bad_verdict")
        if turn["role"] == "reviewer" and turn["verdict"] in ("APPROVE", "INVALID") and index != len(turns) - 1:
            problems.append(f"turn_{index}_session_continues_after_{turn['verdict']}")
    if turns[-1]["role"] == "reviewer" and turns[-1]["verdict"] == "REVISE":
        problems.append("ends_on_revise")
    revisions = sum(1 for turn in turns if turn["role"] == "reviewer" and turn["verdict"] == "REVISE")
    if turns[-1]["role"] == "coder" and revisions < 2:
        problems.append("ends_on_coder_before_final")  # only the Coder turn after the second REVISE is final
    return problems


def _empty_group() -> dict[str, Any]:
    return {
        "sessions": 0,
        "verdicts": defaultdict(Counter),  # round -> verdict -> count
        "truncated_reviews": Counter(),  # round -> count
        "truncated_coder_turns": 0,
        "terminal_approvals": defaultdict(Counter),  # round -> {"passed", "failed"} -> count
        "sessions_with_issues": 0,
    }


def diagnose(turns: Iterable[Mapping[str, Any]], tasks: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Group sessions by (run_id, method) and count Reviewer verdicts, truncation and false approvals."""

    issues = []
    turns_by_session: dict[SessionKey, list[Mapping[str, Any]]] = defaultdict(list)
    tasks_by_session: dict[SessionKey, list[Mapping[str, Any]]] = defaultdict(list)
    for kind, rows, by_session in (("turn", turns, turns_by_session), ("task", tasks, tasks_by_session)):
        for row in rows:
            problem = row_problem(row)
            if problem is None and row["type"] != kind:
                problem = f"expected_{kind}_row"
            if problem is not None:
                issues.append({"where": f"{kind} input", "problem": problem})
                continue
            by_session[_key(row)].append(row)

    groups: dict[tuple[str, str], dict[str, Any]] = defaultdict(_empty_group)
    for key in sorted(set(turns_by_session) | set(tasks_by_session), key=str):
        session = sorted(turns_by_session.get(key, []), key=lambda row: row["turn"])
        records = tasks_by_session.get(key, [])
        group = groups[key[:2]]
        group["sessions"] += 1
        problems = session_problems(session, records)
        _count_session(group, session)  # raw verdicts stay visible even for flagged sessions
        if session and session[-1]["role"] == "reviewer" and session[-1]["verdict"] == "APPROVE":
            review_round = sum(1 for turn in session if turn["role"] == "reviewer")
            if not (len(records) == 1 and isinstance(records[0]["passed"], bool)):
                problems.append("approve_without_single_task_record")
            # Only a session with no protocol or completeness problem is a verified outcome.
            if problems:
                outcome = "unverified"
            else:
                outcome = "passed" if records[0]["passed"] else "failed"
            group["terminal_approvals"][review_round][outcome] += 1
        if problems:
            group["sessions_with_issues"] += 1
            issues.append({"run_id": key[0], "method": key[1], "task_id": key[2], "problems": problems})

    methods_by_run = defaultdict(set)
    for run_id, method in groups:
        methods_by_run[run_id].add(method)
    for run_id, methods in sorted(methods_by_run.items()):
        if len(methods) > 1:
            issues.append({"run_id": run_id, "problems": [f"run_id_has_methods:{','.join(sorted(methods))}"]})

    return {
        "groups": {f"{run_id} {method}": _plain(group, method) for (run_id, method), group in sorted(groups.items())},
        "issues": issues,
    }


def _count_session(group: dict[str, Any], session: Sequence[Mapping[str, Any]]) -> None:
    review_round = 0
    for turn in session:
        if turn["role"] == "reviewer":
            review_round += 1
            group["verdicts"][review_round][str(turn["verdict"])] += 1
            if turn["truncated"]:
                group["truncated_reviews"][review_round] += 1
        elif turn["truncated"]:
            group["truncated_coder_turns"] += 1


def _plain(group: Mapping[str, Any], method: str) -> dict[str, Any]:
    """JSON-ready copy of a group; rounds become string keys."""

    rounds = sorted(set(group["verdicts"]) | set(group["truncated_reviews"]) | set(group["terminal_approvals"]))
    return {
        "method": method,
        "sessions": group["sessions"],
        "sessions_with_issues": group["sessions_with_issues"],
        "truncated_coder_turns": group["truncated_coder_turns"],
        "rounds": {
            str(r): {
                "reviews": sum(group["verdicts"][r].values()),
                "verdicts": {v: group["verdicts"][r][v] for v in sorted(group["verdicts"][r])},
                "truncated": group["truncated_reviews"][r],
                "terminal_approve_passed": group["terminal_approvals"][r]["passed"],
                "false_approvals": group["terminal_approvals"][r]["failed"],
                "unverified_terminal_approvals": group["terminal_approvals"][r]["unverified"],
            }
            for r in rounds
        },
    }


def markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "| Run / method | Sessions | Review | Reviews | APPROVE / REVISE / INVALID | Truncated reviews "
        "| False approvals | Unverified terminal APPROVEs | Truncated Coder turns | Sessions with issues |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, group in report["groups"].items():
        for index, (review_round, stats) in enumerate(group["rounds"].items()):
            verdicts = stats["verdicts"]
            other = sum(n for v, n in verdicts.items() if v not in VERDICTS)
            counts = " / ".join(str(verdicts.get(v, 0)) for v in VERDICTS) + (f" (+{other} other)" if other else "")
            first = index == 0
            lines.append(
                f"| {name if first else ''} | {group['sessions'] if first else ''} | {review_round} "
                f"| {stats['reviews']} | {counts} | {stats['truncated']} | {stats['false_approvals']} "
                f"| {stats['unverified_terminal_approvals']} "
                f"| {group['truncated_coder_turns'] if first else ''} | {group['sessions_with_issues'] if first else ''} |"
            )
    lines.append("")
    lines.append(f"Issues: {len(report['issues'])}")
    for issue in report["issues"][:50]:
        lines.append(f"- {json.dumps(issue)}")
    if len(report["issues"]) > 50:
        lines.append(f"- ... {len(report['issues']) - 50} more (see --json)")
    return "\n".join(lines)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("logs", nargs="+", type=Path, help="event-log JSONL files (CONTRACT.md section 3)")
    parser.add_argument("--json", type=Path, help="also write the full report here; never overwrites a file")
    parser.add_argument("--strict", action="store_true", help="exit with status 1 if any issue is found")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    turns, tasks, load_issues = load_rows(args.logs)
    report = diagnose(turns, tasks)
    report = {**report, "issues": load_issues + report["issues"], "logs": [str(path) for path in args.logs]}
    if args.json is not None:
        if args.json.exists():
            raise SystemExit(f"refusing to overwrite {args.json}")
        args.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(markdown(report))
    return 1 if args.strict and report["issues"] else 0


if __name__ == "__main__":
    sys.exit(main())
