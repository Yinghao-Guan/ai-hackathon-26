"""Tests for demo/collect_logs.py. Stdlib only (no torch), so they run anywhere:

    python -m unittest discover -s demo/tests -p "test_*.py"
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import collect_logs as cl  # noqa: E402


def record(run_id: str, machine: str = "4090", **extra) -> str:
    return json.dumps({"type": "task", "run_id": run_id, "machine": machine, **extra})


class TempDirTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)

    def write(self, name: str, lines: list[str], newline: str = "\n") -> Path:
        path = self.tmp / name
        path.write_bytes("".join(f"{ln}{newline}" for ln in lines).encode("utf-8"))
        return path


class DefaultSources(TempDirTest):
    def test_newest_eval_is_picked_by_timestamp_for_the_machine(self) -> None:
        for name in ["eval-1.5b-4090-202610180231.jsonl", "eval-1.5b-4090-202610171200.jsonl",
                     "eval-1.5b-4070-202610190000.jsonl", "eval-1.5b-4090-202610180231.summary.json",
                     "smoke-1.5b.jsonl"]:
            (self.tmp / name).write_text("")
        self.assertEqual(cl.newest_eval("4090", self.tmp).name, "eval-1.5b-4090-202610180231.jsonl")
        self.assertEqual(cl.newest_eval("4070", self.tmp).name, "eval-1.5b-4070-202610190000.jsonl")
        self.assertIsNone(cl.newest_eval("mac", self.tmp))

    def test_sweep_default_is_per_machine_and_optional(self) -> None:
        (self.tmp / "cost_sweep_4090.jsonl").write_text("")
        self.assertEqual(cl.default_sweep("4090", self.tmp).name, "cost_sweep_4090.jsonl")
        self.assertIsNone(cl.default_sweep("mac", self.tmp))


class Collect(TempDirTest):
    def test_concatenates_sources_in_order_and_describes_them(self) -> None:
        a = self.write("a.jsonl", [record("eval-A2"), record("eval-A2"), record("eval-A5-H1")])
        b = self.write("b.jsonl", [record("sweep-A2-ctx1k")])
        events, manifest = cl.collect([a, b], "4090")
        self.assertEqual(events.splitlines(), [record("eval-A2"), record("eval-A2"), record("eval-A5-H1"), record("sweep-A2-ctx1k")])
        self.assertEqual([s["lines"] for s in manifest["sources"]], [3, 1])
        self.assertEqual(manifest["sources"][0]["run_ids"], ["eval-A2", "eval-A5-H1"])
        self.assertEqual(manifest["events"]["lines"], 4)
        self.assertEqual(manifest["events"]["sha256"], hashlib.sha256(events.encode()).hexdigest())
        self.assertNotIn("_lines", manifest["sources"][0])

    def test_output_and_hashes_do_not_depend_on_line_endings(self) -> None:
        lf = self.write("lf.jsonl", [record("r1"), record("r2")])
        crlf = self.write("crlf.jsonl", [record("r1"), record("r2")], newline="\r\n")
        ev_lf, m_lf = cl.collect([lf], "4090")
        ev_crlf, m_crlf = cl.collect([crlf], "4090")
        self.assertEqual(ev_lf, ev_crlf)
        self.assertNotIn("\r", ev_crlf)
        self.assertEqual(m_lf["sources"][0]["sha256"], m_crlf["sources"][0]["sha256"])

    def test_rejects_mock_runs(self) -> None:
        src = self.write("m.jsonl", [record("eval-A2"), record("mock-a2-1.5b-ctx3k")])
        with self.assertRaisesRegex(cl.CollectError, r"m\.jsonl:2: mock run"):
            cl.collect([src], "4090")

    def test_rejects_other_machines(self) -> None:
        src = self.write("x.jsonl", [record("eval-A2", machine="4070")])
        with self.assertRaisesRegex(cl.CollectError, "machine '4070', expected '4090'"):
            cl.collect([src], "4090")

    def test_rejects_a_run_in_two_sources(self) -> None:
        a = self.write("a.jsonl", [record("eval-A2")])
        b = self.write("b.jsonl", [record("eval-A2")])
        with self.assertRaisesRegex(cl.CollectError, "counted twice"):
            cl.collect([a, b], "4090")

    def test_rejects_malformed_json_and_missing_run_id(self) -> None:
        with self.assertRaisesRegex(cl.CollectError, r"bad\.jsonl:2: malformed JSON"):
            cl.collect([self.write("bad.jsonl", [record("r1"), "{oops"])], "4090")
        with self.assertRaisesRegex(cl.CollectError, "no run_id"):
            cl.collect([self.write("noid.jsonl", ['{"type": "task"}'])], "4090")

    def test_requires_at_least_one_existing_source(self) -> None:
        with self.assertRaisesRegex(cl.CollectError, "no source logs"):
            cl.collect([], "4090")
        with self.assertRaisesRegex(cl.CollectError, "does not exist"):
            cl.collect([self.tmp / "missing.jsonl"], "4090")


@unittest.skipUnless(shutil.which("node"), "needs Node to run the dashboard validator")
class AgainstTheRepo(TempDirTest):
    """The committed demo/data must be exactly what the collector builds from the committed results."""

    def test_committed_dashboard_log_is_up_to_date(self) -> None:
        self.assertEqual(cl.main(["--check"]), 0)

    def test_check_flags_a_stale_log(self) -> None:
        self.assertEqual(cl.main(["--out-dir", str(self.tmp)]), 0)
        self.assertEqual(cl.main(["--check", "--out-dir", str(self.tmp)]), 0)
        events = self.tmp / "events.jsonl"
        events.write_text(events.read_text(encoding="utf-8").split("\n", 1)[1], encoding="utf-8")
        self.assertEqual(cl.main(["--check", "--out-dir", str(self.tmp)]), 1)


if __name__ == "__main__":
    unittest.main()
