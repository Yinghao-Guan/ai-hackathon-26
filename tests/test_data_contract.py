"""The committed SFT data follows CONTRACT.md section 2 and uses the runtime's exact context format."""

import hashlib
import json
from pathlib import Path

import pytest

from rolekv import roles
from train.build_dataset import normalize_system_segments

DATA = Path(__file__).resolve().parents[1] / "data"


def read_rows(name: str) -> list[dict]:
    return [json.loads(line) for line in (DATA / name).read_text(encoding="utf-8").splitlines() if line.strip()]


@pytest.mark.parametrize("role", roles.ROLES)
def test_sft_rows_follow_contract(role):
    rows = read_rows(f"sft_{role}.jsonl")
    assert rows
    for row in rows:
        assert row["role"] == role
        # The same system segment the runtime puts in front of every eval session.
        assert row["segments"][0] == roles.system_segment()
        assert row["segments"][-1] == roles.role_header(role)
        assert row["target"].endswith(roles.IM_END + "\n")


def test_trajectories_match_sft_files():
    turns = [turn for trajectory in read_rows("trajectories.jsonl") for turn in trajectory["turns"]]
    for role in roles.ROLES:
        assert sum(turn["role"] == role for turn in turns) == len(read_rows(f"sft_{role}.jsonl"))
    assert all(turn["segments"][0] == roles.system_segment() for turn in turns)


def test_readme_hashes_match_files():
    readme = (DATA / "README.md").read_text(encoding="utf-8")
    for role in roles.ROLES:
        # README records the committed LF bytes; Git may check them out as CRLF on Windows.
        data = (DATA / f"sft_{role}.jsonl").read_bytes().replace(b"\r\n", b"\n")
        assert hashlib.sha256(data).hexdigest() in readme


def test_normalize_replaces_only_the_system_segment():
    header = roles.role_header("coder")
    trajectories = [{"task_id": 1, "turns": [
        {"role": "coder", "segments": ["<|im_start|>system\nold prompt<|im_end|>\n", "TASK", header], "target": "x"},
    ]}]
    assert normalize_system_segments(trajectories) == 1
    assert trajectories[0]["turns"][0]["segments"] == [roles.system_segment(), "TASK", header]
    assert normalize_system_segments(trajectories) == 0  # idempotent
