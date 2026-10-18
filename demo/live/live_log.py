"""Record live-demo runs as a CONTRACT.md event log, so the dashboard can show their stats afterwards.

server.py feeds every event that `compare_sessions` yields to `LiveLog.observe`. When a method's session
ends (its `final` event), its turns and task result are written to one JSONL file per server session,
demo/data/live/live-<machine>-<YYYYMMDDHHMM>.jsonl, with the same fields as rolekv.costs.turn_record and
task_record. The dashboard opens it with dashboard.html?log=data/live/<file>.

- run_id is "<prefix>-f<ctx_funcs>-<method>", plus "-mirror" for same-transcript runs, so runs at
  different context sizes, and mirror runs (where A2 copies RoleKV's output, so its pass rate is not its
  own), never share a run.
- Running a task again replaces its earlier result in that run instead of counting it twice.
- A method whose session did not finish (stopped, or the server failed) writes nothing.
- Custom tasks get ids custom-1, custom-2, ... in the order they are run.

Stdlib only, so it is testable without the model (demo/tests/test_live_log.py).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

# The machines the event-log contract allows; other devices still run live but are not logged.
CONTRACT_MACHINES = ("4090", "4070", "mac")
TURN_FIELDS = ("role", "prefill_tokens", "reused_tokens", "canon_tokens", "decode_tokens",
               "ttft_ms", "turn_ms", "kv_mb", "text", "truncated", "verdict")
TASK_FIELDS = ("n_turns", "prefill_tokens_total", "canon_tokens_total", "decode_tokens_total",
               "gpu_s", "peak_kv_mb", "ctx_tokens")


class LiveLog:
    def __init__(self, path: Path, machine: str, *, run_prefix: str | None = None, page_path: str | None = None):
        self.path = Path(path)
        self.machine = machine
        self.run_prefix = run_prefix or self.path.stem
        self.page_path = page_path or self.path.name  # how the dashboard fetches it, relative to demo/
        self.enabled = machine in CONTRACT_MACHINES
        self.reason = None if self.enabled else (
            f"machine {machine!r} is not one the event log allows ({', '.join(CONTRACT_MACHINES)}); start the "
            "server with --machine to record live runs")
        # (run_id, task_id) -> {"config": (ctx_funcs, mirror), "records": turn records + the task record}
        self._sessions: dict[tuple[str, str], dict] = {}
        self._custom = 0
        self._run: dict[str, Any] | None = None  # the run in progress: config, task id, turns per method

    @classmethod
    def for_session(cls, data_dir: Path, machine: str, *, now: float | None = None) -> LiveLog:
        stamp = time.strftime("%Y%m%d%H%M", time.localtime(now))
        name = f"live-{machine}-{stamp}"
        return cls(Path(data_dir) / "live" / f"{name}.jsonl", machine, run_prefix=name,
                   page_path=f"data/live/{name}.jsonl")

    def run_id(self, method: str, ctx_funcs: int, mirror: bool) -> str:
        return f"{self.run_prefix}-f{ctx_funcs}{'-mirror' if mirror else ''}-{method}"

    def observe(self, event: dict, request: dict) -> bool:
        """Feed one compare_sessions event. Returns True when it wrote a finished session to the file."""
        if not self.enabled:
            return False
        kind = event.get("type")
        if kind == "start":
            task_id = request["task"]["task_id"]
            if task_id == "custom":
                self._custom += 1
                task_id = f"custom-{self._custom}"
            self._run = {"task_id": task_id, "ctx_funcs": request["ctx_funcs"], "mirror": bool(request["mirror"]),
                         "turns": {}}
        elif self._run is None:
            return False
        elif kind == "turn":
            self._run["turns"].setdefault(event["method"], []).append(event)
        elif kind == "final":
            self._commit(event["method"], event)
            return True
        elif kind in ("done", "cancelled", "error"):
            self._run = None
        return False

    def _commit(self, method: str, final: dict) -> None:
        run = self._run
        run_id = self.run_id(method, run["ctx_funcs"], run["mirror"])
        base = {"run_id": run_id, "machine": self.machine, "method": method, "task_id": run["task_id"]}
        records = [{"type": "turn", **base, "turn": t["turn"], **{k: t.get(k) for k in TURN_FIELDS}}
                   for t in run["turns"].get(method, [])]
        records.append({"type": "task", **base, "passed": bool(final["passed"]),
                        **{k: final.get(k) for k in TASK_FIELDS}})
        key = (run_id, str(run["task_id"]))
        self._sessions.pop(key, None)  # a re-run replaces the earlier result and moves it to the end
        self._sessions[key] = {"config": (run["ctx_funcs"], run["mirror"]), "records": records}
        self._write()

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        text = "".join(json.dumps(r, ensure_ascii=False) + "\n"
                       for s in self._sessions.values() for r in s["records"])
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(text, encoding="utf-8", newline="\n")
        try:
            os.replace(tmp, self.path)
        except PermissionError:  # Windows: the page may be reading the file right now
            self.path.write_text(text, encoding="utf-8", newline="\n")
            tmp.unlink(missing_ok=True)

    def summary(self) -> dict:
        """What the live page needs to link to the dashboard."""
        tasks = {(s["config"], tid) for (_, tid), s in self._sessions.items()}
        return {
            "enabled": self.enabled,
            "reason": self.reason,
            "path": self.page_path,
            "dashboard": f"../dashboard.html?log={self.page_path}",
            "sessions": len(self._sessions),  # one per method per task run
            "tasks": len(tasks),  # distinct tasks run, per context size and mirror setting
        }
