"""Live side-by-side demo: one coding task, one model, A2 (per-role LoRA) next to A5-H1 (RoleKV).

    uv run python demo/live/server.py --model Qwen/Qwen2.5-Coder-1.5B-Instruct --adapters adapters/1.5b --device cuda

then open http://127.0.0.1:8001/. The server serves demo/ (the page is /live/) plus a small JSON API, and
streams every finished turn as a Server-Sent Event:

    GET  /api/info     model, device, adapters, defaults
    GET  /api/tasks    the 100 MBPP eval tasks
    POST /api/run      {"task_id": 443 | "custom": {"text", "asserts"}, "ctx_funcs", "usd_per_gpu_hour", "mirror"}
                       -> text/event-stream: start, (turn_start, turn)..., final per method, done
    POST /api/cancel   stop the current run after the turn in progress

Both sessions run on one model, interleaved turn by turn (A2's turn, then A5-H1's), in a single worker
thread behind a lock: adapter switching is global state, so two sessions never run at once
(CONTRACT.md section 6). `compare_sessions` follows `rolekv.runtime.run_session`'s protocol exactly and is
what tests/test_live_demo.py drives. Stdlib only on top of the project's locked environment.
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
DEMO_DIR = ROOT / "demo"
for _path in (ROOT, Path(__file__).resolve().parent):  # the repo, and live_log.py next to this file
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import torch  # noqa: E402

from eval.testrun import TestResult, run_mbpp_task  # noqa: E402
from live_log import LiveLog  # noqa: E402
from rolekv import roles  # noqa: E402
from rolekv.costs import summarize  # noqa: E402
from rolekv.runtime import MAX_NEW_TOKENS, Engine, TurnResult, extract_code, forced_ids  # noqa: E402

METHODS = ("A2", "A5-H1")  # display and interleaving order
LEADER = "A5-H1"  # in mirror mode A2 replays this method's tokens
LABELS = {"A2": "Per-role LoRA", "A5-H1": "RoleKV"}
CTX_OPTIONS = (30, 60, 120)
# An assumption, shown as one in the UI: roughly what a rented RTX 4090 costs per hour in 2026.
DEFAULT_USD_PER_GPU_HOUR = 0.50
MAX_REVISE = 2
SMOKE_UPDATES = 50  # adapters trained for fewer optimizer updates than this are labelled smoke adapters
CUSTOM_TASK_ID = "custom"
HEARTBEAT_S = 2.0


# --- The session runner -----------------------------------------------------------------------------------


class _Lane:
    """One method's session, advanced one turn at a time with run_session's protocol.

    Coder -> tool output -> Reviewer. APPROVE or INVALID ends the session; after the `max_revise`-th REVISE
    the next Coder turn is final (no test run, no review).
    """

    def __init__(self, engine: Engine, task_segment: str, run_tool: Callable[[str], TestResult], *,
                 max_revise: int, max_new_tokens: Mapping[str, int] | None):
        self.engine = engine
        self.run_tool = run_tool
        self.max_revise = max_revise
        self.max_new_tokens = max_new_tokens
        self.role = "coder"  # the role of the next turn
        self.revisions = 0
        self.turns: list[TurnResult] = []
        self.tools: list[TestResult | None] = []  # per turn: the test run after a Coder turn, else None
        self.verdicts: list[str] = []
        self.code = ""
        self.done = False
        engine.start(task_segment)

    def step(self, forced: Sequence[int] | None = None, tool: TestResult | None = None) -> TurnResult:
        """Run the next turn. `tool` replaces this lane's own test run (mirror mode shares the leader's)."""
        role = self.role
        limit = self.max_new_tokens.get(role) if self.max_new_tokens else None
        result = self.engine.turn(role, max_new_tokens=limit, forced=forced)
        self.turns.append(result)
        if role == "coder":
            self.code = extract_code(result.text)
            if self.revisions == self.max_revise:
                self.tools.append(None)
                self.done = True
            else:
                tool = tool if tool is not None else self.run_tool(self.code)
                self.engine.add_tool_output(tool.output)
                self.tools.append(tool)
                self.role = "reviewer"
        else:
            verdict, _ = roles.parse_reviewer_verdict(result.text)
            result.cost.verdict = verdict
            self.verdicts.append(verdict)
            self.tools.append(None)
            if verdict == "REVISE":
                self.revisions += 1
                self.role = "coder"
            else:
                self.done = True
        return result

    @property
    def end_reason(self) -> str:
        if self.verdicts and self.turns[-1].role == "reviewer":
            return self.verdicts[-1]  # APPROVE or INVALID
        return "LAST_REVISION"  # the Coder turn after the last allowed REVISE


def _tool_event(tool: TestResult | None) -> dict | None:
    if tool is None:
        return None
    return {"passed": tool.passed, "timed_out": tool.timed_out, "output": tool.output}


def _turn_event(method: str, index: int, lane: _Lane) -> dict:
    result = lane.turns[index]
    c = result.cost
    return {
        "type": "turn",
        "method": method,
        "turn": index,
        "role": c.role,
        "text": c.text,
        "prefill_tokens": c.prefill_tokens,
        "reused_tokens": c.reused_tokens,
        "canon_tokens": c.canon_tokens,
        "computed_tokens": c.prefill_tokens + c.canon_tokens,  # CONTRACT.md section 3: "recomputed"
        "decode_tokens": c.decode_tokens,
        "ttft_ms": round(c.ttft_ms, 2),
        "turn_ms": round(c.turn_ms, 2),
        "kv_mb": round(c.kv_mb, 2),
        "truncated": c.truncated,
        "verdict": c.verdict,
        "header_pos": result.header_pos,
        "session_tokens": len(lane.engine.session.ids),
        "last": lane.done,
        "tool": _tool_event(lane.tools[index]),
    }


def _final_event(method: str, lane: _Lane, task: Mapping[str, Any], usd_per_gpu_hour: float) -> dict:
    cost = summarize([t.cost for t in lane.turns], lane.engine.ctx_tokens)
    check = run_mbpp_task(lane.code, task)
    return {
        "type": "final",
        "method": method,
        "final_code": lane.code,
        "passed": check.passed,
        "test_output": check.output,
        "n_asserts": len(task["test_list"]),
        "end": lane.end_reason,
        "verdicts": list(lane.verdicts),
        "n_turns": cost.n_turns,
        "prefill_tokens_total": cost.prefill_tokens_total,
        "canon_tokens_total": cost.canon_tokens_total,
        "decode_tokens_total": cost.decode_tokens_total,
        "gpu_s": round(cost.gpu_s, 4),
        "peak_kv_mb": round(cost.peak_kv_mb, 2),
        "ctx_tokens": cost.ctx_tokens,
        "usd_per_gpu_hour": usd_per_gpu_hour,
        "usd": cost.gpu_s * usd_per_gpu_hour / 3600,  # an estimate from an assumed price
    }


def compare_sessions(
    engines: Mapping[str, Engine],
    task: Mapping[str, Any],
    task_segment: str,
    *,
    max_new_tokens: Mapping[str, int] | None = None,
    max_revise: int = MAX_REVISE,
    usd_per_gpu_hour: float = DEFAULT_USD_PER_GPU_HOUR,
    mirror: bool = False,
    forced_outputs: Mapping[str, Sequence[str]] | None = None,
    run_tool: Callable[[str], TestResult] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> Iterator[dict]:
    """Run one session per engine on the same task, interleaved turn by turn, yielding demo events.

    Events: `start`; per turn `turn_start` then `turn`; `final` when a method's session ends; `done`
    (or `cancelled` if `should_stop()` turns true between turns). Each turn's counters are the engine's
    own TurnCost. The tool output is `eval.testrun.run_mbpp_task` on all of the task's asserts, as in
    the evals (rolekv/eval_runner.py).

    `mirror=True` runs A5-H1 first in each round and has A2 replay its generated tokens and share its tool
    output, so both sessions are token for token the same conversation (A2's argmax is still computed,
    as in `run_session(forced_outputs=...)`). `forced_outputs` replays fixed texts per method, and
    `run_tool(code) -> TestResult` replaces the test run; both are for tests.
    """
    started = time.perf_counter()
    order = list(engines)
    if mirror:
        if set(order) != {LEADER, "A2"}:
            raise ValueError(f"mirror mode needs exactly the engines A2 and {LEADER}")
        order = [LEADER, "A2"]

    if run_tool is None:
        def run_tool(code: str) -> TestResult:
            return run_mbpp_task(code, task)

    lanes = {m: _Lane(engines[m], task_segment, run_tool, max_revise=max_revise, max_new_tokens=max_new_tokens)
             for m in order}
    forced = {m: iter(texts) for m, texts in forced_outputs.items()} if forced_outputs else {}
    limits = {role: (max_new_tokens or {}).get(role, MAX_NEW_TOKENS[role]) for role in roles.ROLES}
    yield {
        "type": "start",
        "methods": list(engines),
        "order": order,
        "task": {"task_id": task["task_id"], "text": task["text"], "test_list": list(task["test_list"])},
        "ctx_tokens": lanes[order[0]].engine.ctx_tokens,
        "context_tokens": len(lanes[order[0]].engine.session.ids),  # system + task, before the first turn
        "max_new_tokens": limits,
        "max_revise": max_revise,
        "mirror": mirror,
        "usd_per_gpu_hour": usd_per_gpu_hour,
    }

    while not all(lane.done for lane in lanes.values()):
        for method in order:
            lane = lanes[method]
            if lane.done:
                continue
            if should_stop is not None and should_stop():
                yield {"type": "cancelled", "wall_s": round(time.perf_counter() - started, 3)}
                return
            index = len(lane.turns)
            yield {"type": "turn_start", "method": method, "turn": index, "role": lane.role}
            ids, tool = None, None
            if method in forced:
                ids = forced_ids(lane.engine.tokenizer, next(forced[method]))
            elif mirror and method != LEADER:
                lead = lanes[LEADER]
                ids, tool = lead.turns[index].gen_ids, lead.tools[index]
            lane.step(forced=ids, tool=tool)
            yield _turn_event(method, index, lane)
            if lane.done:
                yield _final_event(method, lane, task, usd_per_gpu_hour)
    yield {"type": "done", "wall_s": round(time.perf_counter() - started, 3)}


# --- Requests -------------------------------------------------------------------------------------------


def custom_task(text: str, asserts: str | Sequence[str]) -> dict:
    """An MBPP-shaped task from free text and one or more assert lines (blank lines are ignored)."""
    lines = asserts.splitlines() if isinstance(asserts, str) else list(asserts)
    lines = [line.strip() for line in lines if line.strip()]
    text = text.strip()
    if not text:
        raise ValueError("the custom task needs a description")
    if len(text) > 4000:
        raise ValueError("the task description is too long (4,000 characters at most)")
    if not lines:
        raise ValueError("the custom task needs at least one assert line")
    if len(lines) > 20 or any(len(line) > 1000 for line in lines):
        raise ValueError("at most 20 assert lines of 1,000 characters each")
    bad = [line for line in lines if not line.startswith("assert")]
    if bad:
        raise ValueError(f"every test line must start with 'assert': {bad[0]!r}")
    return {"task_id": CUSTOM_TASK_ID, "text": text, "test_list": lines, "test_setup_code": ""}


def parse_run_request(body: Mapping[str, Any], tasks_by_id: Mapping[Any, Mapping[str, Any]],
                      default_ctx_funcs: int) -> dict:
    """Validate a /api/run body. Returns {task, ctx_funcs, usd_per_gpu_hour, mirror}; raises ValueError."""
    if not isinstance(body, Mapping):
        raise ValueError("the request body must be a JSON object")
    if body.get("custom") is not None:
        custom = body["custom"]
        if not isinstance(custom, Mapping):
            raise ValueError("custom must be an object with text and asserts")
        task = custom_task(str(custom.get("text", "")), custom.get("asserts", ""))
    else:
        try:
            task = tasks_by_id[int(body.get("task_id"))]
        except (TypeError, ValueError, KeyError):
            raise ValueError(f"unknown task_id {body.get('task_id')!r}; pick one of the 100 eval tasks") from None
    try:
        ctx_funcs = int(body.get("ctx_funcs", default_ctx_funcs))
        rate = float(body.get("usd_per_gpu_hour", DEFAULT_USD_PER_GPU_HOUR))
    except (TypeError, ValueError):
        raise ValueError("ctx_funcs must be an integer and usd_per_gpu_hour a number") from None
    if not 0 <= ctx_funcs <= 200:
        raise ValueError("ctx_funcs must be between 0 and 200")
    if not 0 <= rate <= 1000:
        raise ValueError("usd_per_gpu_hour must be between 0 and 1000")
    return {"task": task, "ctx_funcs": ctx_funcs, "usd_per_gpu_hour": rate, "mirror": bool(body.get("mirror"))}


# --- One worker thread owns the model ---------------------------------------------------------------------


class Busy(Exception):
    pass


class Job:
    def __init__(self, request: dict):
        self.request = request
        self.events: queue.Queue = queue.Queue()  # None marks the end
        self.cancel = threading.Event()


class Worker:
    """Runs jobs one at a time on the shared model. `lock` guards every use of the model."""

    def __init__(self, engines: Mapping[str, Engine], training_rows: Sequence[Mapping[str, Any]], *,
                 max_new_tokens: Mapping[str, int] | None = None, log: LiveLog | None = None):
        self.engines = dict(engines)
        self.training_rows = training_rows
        self.max_new_tokens = max_new_tokens
        self.log = log  # records finished sessions for the dashboard (live_log.py)
        self.lock = threading.Lock()
        self._jobs: queue.Queue = queue.Queue()
        self._state = threading.Lock()
        self.current: Job | None = None
        threading.Thread(target=self._loop, name="rolekv-model", daemon=True).start()

    def submit(self, request: dict) -> Job:
        with self._state:
            if self.current is not None:
                raise Busy("a run is in progress")
            job = self.current = Job(request)
        self._jobs.put(job)
        return job

    def cancel_current(self) -> bool:
        with self._state:
            job = self.current
        if job is None:
            return False
        job.cancel.set()
        return True

    @property
    def busy(self) -> bool:
        return self.current is not None

    def _loop(self) -> None:
        from rolekv.tasks import build_task_segment

        while True:
            job = self._jobs.get()
            try:
                req = job.request
                segment = build_task_segment(req["task"], self.training_rows, ctx_funcs=req["ctx_funcs"])
                with self.lock:
                    events = compare_sessions(self.engines, req["task"], segment,
                                              max_new_tokens=self.max_new_tokens,
                                              usd_per_gpu_hour=req["usd_per_gpu_hour"], mirror=req["mirror"],
                                              should_stop=job.cancel.is_set)
                    for event in events:
                        if event["type"] == "start":
                            event["ctx_funcs"] = req["ctx_funcs"]
                        if self.log is not None:
                            self.log.observe(event, req)
                            if event["type"] in ("done", "cancelled"):
                                event["log"] = self.log.summary()  # where the page finds this run's stats
                        job.events.put(event)
            except Exception as exc:  # report to the page, keep serving
                if self.log is not None:
                    self.log.observe({"type": "error"}, job.request)  # drop the unfinished run
                job.events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
                print(f"run failed: {type(exc).__name__}: {exc}", file=sys.stderr)
            finally:
                job.events.put(None)
                with self._state:
                    self.current = None


# --- HTTP ----------------------------------------------------------------------------------------------


class App:
    def __init__(self, worker: Worker, tasks: Sequence[Mapping[str, Any]], info: dict, default_ctx_funcs: int):
        self.worker = worker
        self.tasks = list(tasks)
        self.tasks_by_id = {int(t["task_id"]): t for t in self.tasks}
        self.info = info
        self.default_ctx_funcs = default_ctx_funcs


class Handler(SimpleHTTPRequestHandler):
    server_version = "RoleKVLive/1"
    app: App  # set on the subclass by make_server

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(DEMO_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")  # never a stale page or script mid-demo
        super().end_headers()

    def log_message(self, format: str, *args) -> None:
        if getattr(self, "path", "/api/").startswith("/api/"):  # no path yet when the request line is bad
            super().log_message(format, *args)

    def _json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/":
            self.send_response(HTTPStatus.FOUND)
            self.send_header("Location", "/live/")
            self.end_headers()
        elif path == "/api/info":
            log = self.app.worker.log
            self._json({**self.app.info, "busy": self.app.worker.busy, "live_log": log.summary() if log else None})
        elif path == "/api/tasks":
            self._json([{"task_id": t["task_id"], "text": t["text"], "test_list": list(t["test_list"])}
                        for t in self.app.tasks])
        else:
            super().do_GET()

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/cancel":
            self._json({"cancelled": self.app.worker.cancel_current()})
        elif path == "/api/run":
            self._run()
        else:
            self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _run(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > 64 * 1024:
                raise ValueError("request too large")
            body = json.loads(self.rfile.read(length) or b"{}")
            request = parse_run_request(body, self.app.tasks_by_id, self.app.default_ctx_funcs)
            job = self.app.worker.submit(request)
        except Busy as exc:
            self._json({"error": str(exc)}, HTTPStatus.CONFLICT)
            return
        except ValueError as exc:  # includes json.JSONDecodeError
            self._json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("X-Accel-Buffering", "no")
        self.end_headers()
        try:
            while True:
                try:
                    event = job.events.get(timeout=HEARTBEAT_S)
                except queue.Empty:
                    self.wfile.write(b": working\n\n")  # also detects a closed page between turns
                    self.wfile.flush()
                    continue
                if event is None:
                    break
                data = json.dumps(event, ensure_ascii=False)
                self.wfile.write(f"event: {event['type']}\ndata: {data}\n\n".encode())
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            job.cancel.set()  # the page went away: stop after the current turn


def make_server(app: App, host: str, port: int) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"app": app})
    server = ThreadingHTTPServer((host, port), handler)
    server.daemon_threads = True
    return server


# --- Startup ---------------------------------------------------------------------------------------------


def default_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def device_name(device: torch.device) -> str:
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return "Apple GPU (MPS)"
    return "CPU"


def default_machine(device: torch.device) -> str:
    """The event log's machine label, detected as bench/cost_sweep.py does (4090, 4070, mac, ...)."""
    if device.type == "cuda":
        name = torch.cuda.get_device_name(device)
        return next((gpu for gpu in ("4090", "4070") if gpu in name), "cuda")
    return "mac" if sys.platform == "darwin" else device.type


def adapter_info(adapter_dir: Path) -> list[dict]:
    """What the training runs recorded about each adapter (train/train_role_adapter.py's train_run.json)."""
    rows = []
    for name in sorted({n for m in METHODS for n in _role_adapters(m)}):
        path = adapter_dir / name / "train_run.json"
        try:
            run = json.loads(path.read_text())
        except (OSError, ValueError):
            rows.append({"name": name, "updates": None, "model": None})
            continue
        rows.append({"name": name, "updates": run.get("updates"), "model": run.get("model")})
    return rows


def _role_adapters(method: str) -> list[str]:
    from rolekv.runtime import ROLE_ADAPTERS

    return [name for name in ROLE_ADAPTERS[method].values() if name]


def warm_up(engines: Mapping[str, Engine], segments: Sequence[str]) -> None:
    """A short Coder -> tool -> Reviewer exchange per engine at each context size the page offers.

    rolekv.runtime.warmup uses a tiny prompt. The first large prefill still pays one-time allocator and
    kernel costs, and A2 runs first in every round, so without this its first turn would look slower.
    """
    for segment in segments:
        for engine in engines.values():
            engine.start(segment)
            engine.turn("coder", max_new_tokens=2)
            engine.add_tool_output("warm-up")
            engine.turn("reviewer", max_new_tokens=2)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoleKV live demo: A2 and A5-H1 side by side.")
    parser.add_argument("--model", default=roles.BASE_MODEL)
    parser.add_argument("--adapters", type=Path, default=Path("adapters/1.5b"),
                        help="folder with lora_coder, lora_reviewer, alora_coder, alora_reviewer")
    parser.add_argument("--device", default=None, help="default: mps if available, else cuda, else cpu")
    parser.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default=None,
                        help="default: fp32 on cpu, else bf16 (as the evals)")
    parser.add_argument("--host", default="127.0.0.1",
                        help="custom tasks run model-written code with your asserts: keep this local")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--ctx-funcs", type=int, default=30, help="utils.py functions in the context (eval: 30)")
    parser.add_argument("--max-coder-tokens", type=int, default=MAX_NEW_TOKENS["coder"])
    parser.add_argument("--max-reviewer-tokens", type=int, default=MAX_NEW_TOKENS["reviewer"])
    parser.add_argument("--machine", default=None,
                        help="machine label for the recorded runs (4090, 4070, mac); default: detected")
    parser.add_argument("--no-record", action="store_true",
                        help="don't write live runs to demo/data/live/ for the dashboard")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    from rolekv.runtime import load_model, warmup
    from rolekv.tasks import build_task_segment, load_mbpp_splits

    args = parse_args(argv)
    device = torch.device(args.device or default_device())
    dtype_name = args.dtype or ("fp32" if device.type == "cpu" else "bf16")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[dtype_name]
    if not args.adapters.is_dir():
        raise SystemExit(f"adapter folder {args.adapters} not found (see demo/README.md, 'Live demo')")

    print("loading MBPP splits ...", flush=True)
    splits = load_mbpp_splits()
    tasks = [dict(row) for row in splits.evaluation]
    training_rows = [dict(row) for row in splits.training]

    print(f"loading {args.model} + adapters from {args.adapters} on {device} ({dtype_name}) ...", flush=True)
    model, tokenizer = load_model(list(METHODS), base_model=args.model, adapter_dir=args.adapters,
                                  device=device, dtype=dtype)
    engines = {m: Engine(model, tokenizer, m) for m in METHODS}
    ctx_options = sorted({*CTX_OPTIONS, args.ctx_funcs})
    print(f"warming up at {', '.join(map(str, ctx_options))} context functions ...", flush=True)
    for engine in engines.values():
        warmup(engine)
    warm_up(engines, [build_task_segment(tasks[0], training_rows, ctx_funcs=n) for n in ctx_options])

    limits = {"coder": args.max_coder_tokens, "reviewer": args.max_reviewer_tokens}
    log = None if args.no_record else LiveLog.for_session(DEMO_DIR / "data", args.machine or default_machine(device))
    if log is not None:
        print(f"recording live runs to {log.path}" if log.enabled else f"not recording live runs: {log.reason}",
              flush=True)
    worker = Worker(engines, training_rows, max_new_tokens=limits, log=log)
    adapters = adapter_info(args.adapters)
    updates = [a["updates"] for a in adapters if a["updates"] is not None]
    info = {
        "model": args.model,
        "device": device.type,
        "device_name": device_name(device),
        "dtype": dtype_name,
        "adapters": str(args.adapters),
        "adapter_runs": adapters,
        "smoke_adapters": bool(updates) and min(updates) < SMOKE_UPDATES,
        "methods": list(METHODS),
        "labels": LABELS,
        "ctx_options": ctx_options,
        "default_ctx_funcs": args.ctx_funcs,
        "default_usd_per_gpu_hour": DEFAULT_USD_PER_GPU_HOUR,
        "max_new_tokens": limits,
        "max_revise": MAX_REVISE,
    }
    server = make_server(App(worker, tasks, info, args.ctx_funcs), args.host, args.port)
    print(f"RoleKV live demo: http://{args.host}:{args.port}/  (Ctrl+C to stop)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
