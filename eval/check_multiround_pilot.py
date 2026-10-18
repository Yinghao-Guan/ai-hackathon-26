"""Check a generated multi-round SFT directory before anyone trains on it.

    uv run --no-sync python -m eval.check_multiround_pilot outputs/data/<run> --require-second-round-labels

Offline and read-only: it loads MBPP and the Qwen tokenizer from the local caches only, runs no model and executes
no generated code. It checks
- files: the SFT hashes in README.md match the bytes, every JSONL file is LF-only;
- schema: each SFT row is exactly CONTRACT.md section 2, and the rows are exactly the trajectories' turns in order;
- protocol: role order, at most two REVISEs and five turns, each turn's context extends the previous one with the
  contractual segments, APPROVE only as the last turn;
- separation: every task is in MBPP's training splits, none is an evaluation task, and each TASK segment is
  exactly the one built from the training rows;
- report: generation_report.json agrees with the files, and api_usage.jsonl with the report's API totals;
- length: contractual token counts per row against the training limit, never truncating; Reviewer rows are
  counted by review round and label before and after the limit.
Labels are derived by the generator from real test results; this checker cannot re-run code, so it only flags
labels that disagree with the recorded tool output (warnings).
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Mapping, Sequence

from rolekv import roles

SFT_FIELDS = {"task_id", "role", "segments", "target"}
TOOL_PREFIX = "<|im_start|>user\n<tool_output>\n"
CODER_BODY_RE = re.compile(r"```python\n.*\n```<\|im_end\|>\n", re.DOTALL)
FAILURE_MARKERS = ("Traceback (most recent call last)", "AssertionError", "Error:", "Timed out after")
MAX_TURNS = 5


def _read_jsonl(path: Path, errors: list[str]) -> list[Any]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                errors.append(f"{path.name}:{number}: malformed JSON")
    return rows


def check_files(data_dir: Path, errors: list[str]) -> dict[str, str]:
    """LF-only JSONL, and README.md lists each SFT file's real SHA-256."""

    readme = (data_dir / "README.md").read_text(encoding="utf-8") if (data_dir / "README.md").exists() else ""
    if not readme:
        errors.append("README.md is missing or empty")
    hashes = {}
    for name in ("trajectories.jsonl", "sft_coder.jsonl", "sft_reviewer.jsonl"):
        path = data_dir / name
        if not path.exists():
            errors.append(f"{name} is missing")
            continue
        data = path.read_bytes()
        if b"\r" in data:
            errors.append(f"{name} contains CR bytes; generated JSONL must be LF-only")
        hashes[name] = hashlib.sha256(data).hexdigest()
    for name in ("sft_coder.jsonl", "sft_reviewer.jsonl"):
        if name in hashes and hashes[name] not in readme:
            errors.append(f"README.md does not list the SHA-256 of {name} ({hashes[name][:12]}...)")
    return hashes


def _is_task_id(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _shape_problem(row: Any, fields: set[str]) -> str | None:
    """Why a turn or SFT row can't be read safely, or None. Checked before any field is used."""

    if not isinstance(row, dict) or set(row) != fields:
        return f"fields must be exactly {sorted(fields)}"
    if "task_id" in fields and not _is_task_id(row["task_id"]):
        return "task_id must be an integer"
    segments = row["segments"]
    if row["role"] not in roles.ROLES or not isinstance(segments, list) or not segments \
            or not all(isinstance(s, str) for s in segments) or not isinstance(row["target"], str):
        return "bad role, segments or target types"
    return None


def check_sft_row(row: Any, where: str, errors: list[str]) -> None:
    problem = _shape_problem(row, SFT_FIELDS)
    if problem is not None:
        errors.append(f"{where}: {problem}")
        return
    role, segments, target = row["role"], row["segments"], row["target"]
    if segments[0] != roles.system_segment():
        errors.append(f"{where}: first segment is not roles.system_segment()")
    if segments[-1] != roles.role_header(role):
        errors.append(f"{where}: last segment is not the {role} header")
    if not target.endswith(roles.IM_END + "\n") or not target[: -len(roles.IM_END) - 1].strip():
        errors.append(f"{where}: target must be a non-empty BODY ending with <|im_end|>\\n")
    if role == "coder" and not CODER_BODY_RE.fullmatch(target):
        errors.append(f"{where}: coder target must be exactly one ```python block")
    if role == "reviewer" and not (target.startswith("APPROVE") or target.startswith("REVISE: ")):
        errors.append(f"{where}: reviewer target must start with APPROVE or 'REVISE: '")


def _tool_output(segment: str) -> str:
    return segment[len(TOOL_PREFIX):] if segment.startswith(TOOL_PREFIX) else ""


def check_trajectory(trajectory: Any, where: str, errors: list[str], warnings: list[str]) -> None:
    """Role order, protocol limits, how each context grows, and labels against the recorded tool output."""

    if not isinstance(trajectory, dict) or set(trajectory) != {"task_id", "turns", "final_code"}:
        errors.append(f"{where}: fields must be exactly ['final_code', 'task_id', 'turns']")
        return
    turns = trajectory["turns"]
    if not _is_task_id(trajectory["task_id"]) or not isinstance(turns, list) or not turns \
            or not isinstance(trajectory["final_code"], str):
        errors.append(f"{where}: needs an integer task_id, a non-empty turns list and a final_code string")
        return
    for index, turn in enumerate(turns):  # every turn must be readable before any is compared with another
        problem = _shape_problem(turn, {"role", "segments", "target"})
        if problem is not None:
            errors.append(f"{where} turn {index}: {problem}")
            return
    if not 2 <= len(turns) <= MAX_TURNS:
        errors.append(f"{where}: {len(turns)} turns; the protocol allows 2 to {MAX_TURNS}")
    _check_final_code(trajectory, where, errors)
    revisions = 0
    for index, turn in enumerate(turns):
        here = f"{where} turn {index}"
        expected = "coder" if index % 2 == 0 else "reviewer"
        if turn["role"] != expected:
            errors.append(f"{here}: expected a {expected} turn")
            return
        check_sft_row({"task_id": trajectory["task_id"], **turn}, here, errors)
        if index == 0:
            if len(turn["segments"]) != 3:
                errors.append(f"{here}: the first turn's context is SYSTEM, TASK, coder header")
            continue
        previous = turns[index - 1]
        if expected == "reviewer":
            middle = turn["segments"][len(previous["segments"]) + 1:-1]
            grows = len(middle) == 1 and middle[0].startswith(TOOL_PREFIX)
        else:
            grows = len(turn["segments"]) == len(previous["segments"]) + 2
        if not grows or turn["segments"][: len(previous["segments"]) + 1] != previous["segments"] + [previous["target"]]:
            errors.append(f"{here}: context does not extend the previous turn with the contractual segments")
        if expected == "reviewer":
            approve = turn["target"].startswith("APPROVE")
            revisions += not approve
            if approve and index != len(turns) - 1:
                errors.append(f"{here}: APPROVE must end the session")
            if not approve and index == len(turns) - 1:
                errors.append(f"{here}: a session cannot end on REVISE")
            output = _tool_output(turn["segments"][-2])
            failed = any(marker in output for marker in FAILURE_MARKERS)
            if approve and failed:
                warnings.append(f"{here}: APPROVE after tool output that looks like a failure")
            if not approve and not output.strip():
                warnings.append(f"{here}: REVISE after empty tool output")
    if revisions > 2:
        errors.append(f"{where}: more than two REVISE rounds")
    if turns[-1]["role"] == "coder" and revisions < 2 and len(turns) != 3:
        errors.append(f"{where}: a session may end on a Coder turn only after its REVISE limit or the default repair")


def _check_final_code(trajectory: Mapping[str, Any], where: str, errors: list[str]) -> None:
    """final_code must be exactly the code the generator extracts from the last Coder turn."""

    from train.build_dataset import extract_python_code

    coder_targets = [turn["target"] for turn in trajectory["turns"] if turn["role"] == "coder"]
    if not coder_targets:
        errors.append(f"{where}: no Coder turn to take final_code from")
        return
    body = coder_targets[-1].removesuffix(roles.IM_END + "\n")
    try:
        code = extract_python_code(body)
    except ValueError:
        errors.append(f"{where}: the last Coder turn has no single ```python block")
        return
    if code != trajectory["final_code"]:
        errors.append(f"{where}: final_code does not match the code in the last Coder turn")


def expected_sft_rows(trajectories: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    rows: dict[str, list[dict[str, Any]]] = {role: [] for role in roles.ROLES}
    for trajectory in trajectories:
        for turn in trajectory["turns"]:
            rows[turn["role"]].append({"task_id": trajectory["task_id"], "role": turn["role"],
                                       "segments": turn["segments"], "target": turn["target"]})
    return rows


def check_separation(trajectories: Sequence[Mapping[str, Any]], splits: Any, ctx_funcs: int | None,
                     errors: list[str]) -> dict[str, Any]:
    from rolekv.tasks import build_task_segment

    training = {int(row["task_id"]): row for row in splits.training}
    evaluation = {int(row["task_id"]) for row in splits.evaluation}
    ids = [trajectory["task_id"] for trajectory in trajectories]
    for task_id in ids:
        if task_id in evaluation:
            errors.append(f"task {task_id} is a held-out evaluation task")
        elif task_id not in training:
            errors.append(f"task {task_id} is not in MBPP's training splits")
        elif ctx_funcs is not None:
            try:
                expected = build_task_segment(training[task_id], list(splits.training), ctx_funcs=ctx_funcs)
            except ValueError as exc:
                errors.append(f"task {task_id}: cannot rebuild its TASK segment ({exc})")
                continue
            if any(turn["segments"][1] != expected for turn in next(t for t in trajectories if t["task_id"] == task_id)["turns"]):
                errors.append(f"task {task_id}: TASK segment differs from the one built from the training rows")
    return {"task_ids": ids, "duplicate_task_ids": sorted(task for task, n in Counter(ids).items() if n > 1)}


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _report_problem(report: Any) -> str | None:
    """Why the report's fields can't be used, or None."""

    if not isinstance(report, dict):
        return "must be a JSON object"
    if not (_is_task_id(report.get("kept")) and _is_task_id(report.get("attempted"))):
        return "kept and attempted must be integers"
    if not isinstance(report.get("shapes"), dict) or not isinstance(report.get("rejections"), dict) \
            or not all(_is_task_id(n) for n in report["rejections"].values()):
        return "shapes and rejections must be objects, rejections with integer counts"
    settings = report.get("settings", {})
    if not isinstance(settings, dict) or ("ctx_funcs" in settings and not _is_task_id(settings["ctx_funcs"])):
        return "settings must be an object with an integer ctx_funcs"
    api = report.get("api")
    if api is not None and not (isinstance(api, dict) and _is_task_id(api.get("calls"))
                                and _is_number(api.get("charged_usd")) and _is_number(api.get("cap_usd"))):
        return "api needs integer calls and numeric charged_usd and cap_usd"
    return None


def check_report(data_dir: Path, trajectories: Sequence[Mapping[str, Any]], errors: list[str]) -> dict[str, Any]:
    from train.build_dataset import SHAPES_BY_TURNS

    path = data_dir / "generation_report.json"
    if not path.exists():
        errors.append("generation_report.json is missing")
        return {}
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        errors.append("generation_report.json is not valid JSON")
        return {}
    problem = _report_problem(report)
    if problem is not None:
        errors.append(f"generation_report.json: {problem}")
        return {}
    shapes = dict(sorted(Counter(SHAPES_BY_TURNS.get(len(t["turns"]), "invalid") for t in trajectories).items()))
    if report["kept"] != len(trajectories) or report["shapes"] != shapes:
        errors.append(f"generation_report.json says kept={report['kept']} shapes={report['shapes']}; "
                      f"the files have {len(trajectories)} {shapes}")
    if report["kept"] + sum(report["rejections"].values()) != report["attempted"]:
        errors.append("generation_report.json: kept + rejections != attempted")
    usage_path = data_dir / "api_usage.jsonl"
    if "api" in report:
        records = _read_jsonl(usage_path, errors) if usage_path.exists() else None
        if records is None:
            errors.append("the report has API totals but api_usage.jsonl is missing")
        elif not all(isinstance(r, dict) and _is_number(r.get("charged_usd")) for r in records):
            errors.append("api_usage.jsonl has a record without a numeric charged_usd")
        else:
            charged = round(sum(record["charged_usd"] for record in records), 6)
            if len(records) != report["api"]["calls"] or abs(charged - report["api"]["charged_usd"]) > 1e-6:
                errors.append("api_usage.jsonl does not add up to the report's API totals")
            if report["api"]["charged_usd"] > report["api"]["cap_usd"]:
                errors.append("charged cost is above the spend cap")
    return report


def check_lengths(rows: Mapping[str, Sequence[Mapping[str, Any]]], tokenizer, max_length: int) -> dict[str, Any]:
    """Contractual token counts per row; Reviewer rows by review round and label, before and after the limit."""

    lengths: dict[str, Any] = {}
    reviewer: dict[str, dict[str, dict[str, int]]] = {}
    for role, role_rows in rows.items():
        counts = [len(roles.tokenize_sft_example(tokenizer, row["segments"], row["target"])["input_ids"])
                  for row in role_rows]
        lengths[role] = {
            "rows": len(counts),
            "min": min(counts, default=None),
            "median": int(statistics.median(counts)) if counts else None,
            "max": max(counts, default=None),
            "over_limit": sum(n > max_length for n in counts),
        }
        if role != "reviewer":
            continue
        for row, n in zip(role_rows, counts):
            review_round = str(row["segments"].count(roles.role_header("reviewer")))
            label = "APPROVE" if row["target"].startswith("APPROVE") else (
                "REVISE" if row["target"].startswith("REVISE") else "other")
            cell = reviewer.setdefault(review_round, {}).setdefault(label, {"all": 0, "within_limit": 0})
            cell["all"] += 1
            cell["within_limit"] += n <= max_length
    return {"max_length": max_length, "lengths": lengths, "reviewer_by_round": reviewer}


def check_pilot(data_dir: Path, *, tokenizer, splits: Any, max_length: int = 4096,
                require_second_round_labels: bool = False) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    hashes = check_files(data_dir, errors)
    if errors and len(hashes) < 3:
        return {"ok": False, "errors": errors, "warnings": warnings}
    trajectories = _read_jsonl(data_dir / "trajectories.jsonl", errors)
    sft = {role: _read_jsonl(data_dir / f"sft_{role}.jsonl", errors) for role in roles.ROLES}
    for index, trajectory in enumerate(trajectories):
        check_trajectory(trajectory, f"trajectory {index}", errors, warnings)
    for role, role_rows in sft.items():
        for index, row in enumerate(role_rows):
            check_sft_row(row, f"sft_{role}.jsonl row {index}", errors)
    if errors:  # later checks assume well-formed rows
        return {"ok": False, "errors": errors, "warnings": warnings, "hashes": hashes}
    if sft != expected_sft_rows(trajectories):
        errors.append("the SFT files are not exactly the trajectories' turns in order")
    report = check_report(data_dir, trajectories, errors)
    separation = check_separation(trajectories, splits, report.get("settings", {}).get("ctx_funcs"), errors)
    coverage = check_lengths(sft, tokenizer, max_length)
    second = coverage["reviewer_by_round"].get("2", {})
    if require_second_round_labels and not all(second.get(label, {}).get("within_limit", 0) for label in
                                               ("APPROVE", "REVISE")):
        errors.append("second-round coverage missing: need at least one APPROVE and one REVISE within the limit")
    return {
        "ok": not errors,
        "errors": errors,
        "warnings": warnings,
        "hashes": hashes,
        "kept": len(trajectories),
        "rejections": report.get("rejections"),
        "api": report.get("api"),
        "separation": separation,
        **coverage,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("--tokenizer", default=roles.BASE_MODEL, help="loaded from the local Hugging Face cache only")
    parser.add_argument("--revision", help="tokenizer snapshot, e.g. the model_revision of the training runs")
    parser.add_argument("--max-length", type=int, default=4096, help="the trainer's --max-length")
    parser.add_argument("--require-second-round-labels", action="store_true",
                        help="fail unless round-2 APPROVE and REVISE both survive the length limit")
    parser.add_argument("--json", type=Path, help="also write the full result here; never overwrites a file")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from transformers import AutoTokenizer

    from rolekv.tasks import load_mbpp_splits

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, revision=args.revision, local_files_only=True)
    result = check_pilot(args.data_dir, tokenizer=tokenizer, splits=load_mbpp_splits(), max_length=args.max_length,
                         require_second_round_labels=args.require_second_round_labels)
    result = {**result, "tokenizer": args.tokenizer, "revision": args.revision}
    if args.json is not None:
        if args.json.exists():
            raise SystemExit(f"refusing to overwrite {args.json}")
        args.json.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps({key: result.get(key) for key in ("ok", "kept", "rejections", "lengths", "reviewer_by_round",
                                                        "errors", "warnings")}, indent=2))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
