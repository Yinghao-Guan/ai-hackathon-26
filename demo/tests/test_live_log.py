"""Tests for demo/live/live_log.py. Stdlib only (no torch, no model):

    python -m unittest discover -s demo/tests -p "test_*.py"

Events are shaped like demo/live/server.py's _turn_event and _final_event. tests/test_live_demo.py checks
the real events against the engine; here the point is what reaches the dashboard.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

DEMO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DEMO / "live"))
from live_log import LiveLog  # noqa: E402

CTX = 1800


def turn_event(method: str, i: int, role: str, prefill: int, reused: int, canon: int, decode: int, text: str) -> dict:
    return {"type": "turn", "method": method, "turn": i, "role": role, "text": text,
            "prefill_tokens": prefill, "reused_tokens": reused, "canon_tokens": canon,
            "computed_tokens": prefill + canon, "decode_tokens": decode, "ttft_ms": 120.5, "turn_ms": 900.25,
            "kv_mb": 50.5, "truncated": False, "verdict": "APPROVE" if role == "reviewer" else None,
            "header_pos": 0, "session_tokens": 0, "last": role == "reviewer", "tool": None}


def session(method: str) -> tuple[list[dict], dict]:
    """A Coder -> Reviewer session with consistent accounting for one method."""
    if method == "A2":  # a fresh cache per turn: everything is prefilled again
        turns = [turn_event(method, 0, "coder", CTX, 0, 0, 60, "```python\ndef f(x):\n    return x\n```"),
                 turn_event(method, 1, "reviewer", CTX + 120, 0, 0, 3, "APPROVE")]
    else:  # one shared cache; the coder's header + output is re-read at the handoff
        turns = [turn_event(method, 0, "coder", CTX, 0, 70, 60, "```python\ndef f(x):\n    return x\n```"),
                 turn_event(method, 1, "reviewer", 60, CTX + 60, 0, 3, "APPROVE")]
    final = {"type": "final", "method": method, "passed": True, "final_code": "def f(x): return x",
             "n_turns": 2, "prefill_tokens_total": sum(t["prefill_tokens"] + t["canon_tokens"] for t in turns),
             "canon_tokens_total": sum(t["canon_tokens"] for t in turns),
             "decode_tokens_total": sum(t["decode_tokens"] for t in turns), "gpu_s": 1.8, "peak_kv_mb": 52.0,
             "ctx_tokens": CTX, "end": "APPROVE", "verdicts": ["APPROVE"]}
    return turns, final


def request(task_id=11, ctx_funcs=30, mirror=False) -> dict:
    return {"task": {"task_id": task_id}, "ctx_funcs": ctx_funcs, "mirror": mirror, "usd_per_gpu_hour": 0.5}


def play(log: LiveLog, req: dict, *, finish=("A2", "A5-H1"), end="done") -> None:
    """Feed one interleaved run, as compare_sessions yields it, finishing only the methods in `finish`."""
    log.observe({"type": "start"}, req)
    sessions = {m: session(m) for m in ("A2", "A5-H1")}
    for i in range(2):
        for m, (turns, _) in sessions.items():
            log.observe({"type": "turn_start", "method": m, "turn": i}, req)
            log.observe(turns[i], req)
    for m in finish:
        log.observe(sessions[m][1], req)
    log.observe({"type": end}, req)


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


class LiveLogTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.log = LiveLog.for_session(self.tmp, "4090", now=0)

    def test_session_file_name_and_dashboard_link(self) -> None:
        self.assertEqual(self.log.path.parent, self.tmp / "live")
        self.assertRegex(self.log.path.name, r"^live-4090-\d{12}\.jsonl$")
        s = self.log.summary()
        self.assertEqual(s["path"], f"data/live/{self.log.path.name}")
        self.assertEqual(s["dashboard"], f"../dashboard.html?log=data/live/{self.log.path.name}")
        self.assertEqual((s["sessions"], s["tasks"]), (0, 0))
        self.assertFalse(self.log.path.exists())

    def test_finished_run_is_written_in_the_contract_shape(self) -> None:
        play(self.log, request())
        records = read(self.log.path)
        self.assertEqual([r["type"] for r in records], ["turn", "turn", "task", "turn", "turn", "task"])
        prefix = self.log.run_prefix
        self.assertEqual({r["run_id"] for r in records}, {f"{prefix}-f30-A2", f"{prefix}-f30-A5-H1"})
        self.assertTrue(all(r["machine"] == "4090" and r["task_id"] == 11 for r in records))
        turn = records[0]
        self.assertEqual(set(turn), {"type", "run_id", "machine", "method", "task_id", "turn", "role", "prefill_tokens",
                                     "reused_tokens", "canon_tokens", "decode_tokens", "ttft_ms", "turn_ms", "kv_mb",
                                     "text", "truncated", "verdict"})
        task = records[2]
        self.assertEqual(set(task), {"type", "run_id", "machine", "method", "task_id", "passed", "n_turns",
                                     "prefill_tokens_total", "canon_tokens_total", "decode_tokens_total", "gpu_s",
                                     "peak_kv_mb", "ctx_tokens"})
        self.assertEqual((self.log.summary()["sessions"], self.log.summary()["tasks"]), (2, 1))

    def test_rerunning_a_task_replaces_it(self) -> None:
        play(self.log, request(task_id=11))
        play(self.log, request(task_id=12))
        play(self.log, request(task_id=11))
        tasks = [(r["method"], r["task_id"]) for r in read(self.log.path) if r["type"] == "task"]
        self.assertEqual(tasks, [("A2", 12), ("A5-H1", 12), ("A2", 11), ("A5-H1", 11)])
        self.assertEqual(self.log.summary()["tasks"], 2)

    def test_context_size_and_mirror_runs_are_separate_runs(self) -> None:
        play(self.log, request(ctx_funcs=30))
        play(self.log, request(ctx_funcs=60))
        play(self.log, request(ctx_funcs=30, mirror=True))
        prefix = self.log.run_prefix
        self.assertEqual({r["run_id"] for r in read(self.log.path) if r["method"] == "A2"},
                         {f"{prefix}-f30-A2", f"{prefix}-f60-A2", f"{prefix}-f30-mirror-A2"})
        self.assertEqual(self.log.summary()["tasks"], 3)

    def test_unfinished_sessions_are_not_written(self) -> None:
        play(self.log, request(), finish=(), end="cancelled")
        self.assertFalse(self.log.path.exists())
        play(self.log, request(), finish=("A5-H1",), end="cancelled")  # A2 was stopped mid-session
        self.assertEqual({r["method"] for r in read(self.log.path)}, {"A5-H1"})

    def test_an_error_drops_the_run_in_progress(self) -> None:
        self.log.observe({"type": "start"}, request())
        self.log.observe(session("A2")[0][0], request())
        self.log.observe({"type": "error"}, request())
        self.log.observe(session("A2")[1], request())  # a stray final after the error is ignored
        self.assertFalse(self.log.path.exists())

    def test_custom_tasks_get_their_own_ids(self) -> None:
        play(self.log, request(task_id="custom"))
        play(self.log, request(task_id="custom"))
        ids = {r["task_id"] for r in read(self.log.path)}
        self.assertEqual(ids, {"custom-1", "custom-2"})

    def test_machines_outside_the_contract_are_not_recorded(self) -> None:
        log = LiveLog.for_session(self.tmp, "cuda", now=0)
        play(log, request())
        self.assertFalse(log.path.exists())
        s = log.summary()
        self.assertFalse(s["enabled"])
        self.assertIn("--machine", s["reason"])

    @unittest.skipUnless(shutil.which("node"), "needs Node to run the dashboard validator")
    def test_the_dashboard_accepts_the_recording(self) -> None:
        for task_id in (11, 12, 13):
            play(self.log, request(task_id=task_id))
        result = subprocess.run(["node", str(DEMO / "check_log.mjs"), str(self.log.path)], capture_output=True, text=True)
        summary = json.loads(result.stdout)
        self.assertEqual(result.returncode, 0, result.stdout)
        self.assertEqual((summary["accepted"], summary["rejected"], summary["auditIssues"]), (18, 0, 0))
        self.assertEqual(summary["headline"]["pairedTasks"], 3)


if __name__ == "__main__":
    unittest.main()
