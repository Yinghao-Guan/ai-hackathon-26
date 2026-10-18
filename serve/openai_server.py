"""OpenAI-compatible chat server on RoleKV's KVSession, so agent harnesses (DeepSeek Harness) can drive our model.

    uv run --no-sync python -m serve.openai_server --device mps --port 8765 --dump-dir tmp/serve_logs/dumps

Endpoints: POST /v1/chat/completions (streaming or not) and GET /v1/models, on 127.0.0.1 only.

The model id picks the KV policy:

| Model id         | Prompt KV                                                        |
|------------------|------------------------------------------------------------------|
| rolekv-reuse     | reuse the longest common token prefix with a kept session        |
| rolekv-noreuse   | prefill the whole prompt on every request (per-role LoRA baseline) |

Requests are rendered with the tokenizer's chat template (Qwen2.5 has a tool format), and Qwen's
<tool_call>{json}</tool_call> blocks are parsed back into OpenAI `tool_calls`. Decoding is greedy.

Reuse keeps up to --max-sessions KVSessions. A request continues the session that shares the longest
token prefix with its prompt: the cache is cropped to that prefix and only the rest is computed. When
the shared part is under half of that session (a side request that only shares the system prompt and
tools), the session is forked instead, so the main conversation keeps its cache.

Every request is logged as one JSONL line (--log): prompt, reused and computed tokens, TTFT, decode
tokens, wall time and how the tool calls parsed. --dump-dir also keeps each request and response body,
which serve/replay.py replays against either policy.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import threading
import time
import uuid
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import torch

from rolekv.cache import KVSession

MODELS = {"rolekv-reuse": True, "rolekv-noreuse": False}  # model id -> reuse the prompt KV
CONTEXT_WINDOW = 32768  # Qwen2.5-Coder-1.5B-Instruct's max_position_embeddings
TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"


# --- Prompt rendering ------------------------------------------------------------


def message_text(content) -> str:
    """OpenAI content (a string, None, or a list of parts) as plain text; non-text parts are dropped."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")


def template_messages(messages: Sequence[dict]) -> list[dict]:
    """OpenAI chat messages in the shape Qwen's chat template expects.

    `developer` becomes `system`; tool-call arguments, a JSON string on the wire, become objects so the
    template renders them as JSON rather than as a quoted string.
    """
    out = []
    for m in messages:
        role = "system" if m.get("role") == "developer" else m.get("role")
        msg = {"role": role, "content": message_text(m.get("content"))}
        calls = []
        for call in m.get("tool_calls") or []:
            fn = call.get("function", {})
            args = fn.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args) if args.strip() else {}
                except json.JSONDecodeError:
                    pass  # rendered as a string; the model sees what it sent
            calls.append({"type": "function", "function": {"name": fn.get("name", ""), "arguments": args}})
        if calls:
            msg["tool_calls"] = calls
        if role == "tool" and m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        out.append(msg)
    return out


def render(tokenizer, messages: Sequence[dict], tools: Sequence[dict] | None, *, generation_prompt: bool = True) -> list[int]:
    text = tokenizer.apply_chat_template(
        template_messages(messages), tools=list(tools) if tools else None,
        add_generation_prompt=generation_prompt, tokenize=False,
    )
    return tokenizer(text, add_special_tokens=False)["input_ids"]


# --- Tool-call parsing -------------------------------------------------------------

_TAGGED = re.compile(re.escape(TOOL_OPEN) + r"(.*?)(?:" + re.escape(TOOL_CLOSE) + r"|$)", re.DOTALL)
_FENCE = re.compile(r"^```(?:json)?[ \t]*\n?(.*?)(?:```|$)", re.DOTALL)


def _first_json_object(blob: str):
    """The first JSON value starting at the first '{', ignoring trailing text (small models add a stray '}')."""
    start = blob.find("{")
    if start < 0:
        return None
    try:
        value, _ = json.JSONDecoder().raw_decode(blob[start:])
    except json.JSONDecodeError:
        return None
    return value


def _as_call(obj) -> dict | None:
    """{"name", "arguments": <JSON string>} from a parsed {"name": ..., "arguments": ...}, or None."""
    if not isinstance(obj, dict) or not isinstance(obj.get("name"), str) or not obj["name"]:
        return None
    args = obj.get("arguments", obj.get("parameters", {}))
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            return None
    if not isinstance(args, dict):
        return None
    return {"name": obj["name"], "arguments": json.dumps(args, ensure_ascii=False)}


@dataclass
class ParsedReply:
    content: str | None
    calls: list[dict]  # [{"name", "arguments"}], arguments as a JSON string
    mode: str  # "text", "tagged" (Qwen's format), "bare" (an untagged JSON call, --lenient only)
    errors: list[str] = field(default_factory=list)  # <tool_call> blocks that did not parse


def parse_reply(text: str, tool_names: Sequence[str] = (), *, lenient: bool = True) -> ParsedReply:
    """Split a generated reply into content and tool calls.

    Qwen's format is one or more <tool_call>{"name": ..., "arguments": {...}}</tool_call> blocks; content
    is the text before the first one. A block that is cut off (no closing tag) still counts if its JSON
    parses. With `lenient`, a reply that is only a JSON call (bare or in a ```json fence) naming one of
    `tool_names` also counts, since small models often drop the tags.
    """
    cut = text.find(TOOL_OPEN)
    if cut >= 0:
        calls, errors = [], []
        for block in _TAGGED.findall(text[cut:]):
            call = _as_call(_first_json_object(block))
            if call is None:
                errors.append(block.strip()[:200])
            else:
                calls.append(call)
        content = text[:cut].strip() or None
        if calls:
            return ParsedReply(content, calls, "tagged", errors)
        return ParsedReply(text.strip() or None, [], "text", errors)
    if lenient and tool_names:
        stripped = text.strip()
        fence = _FENCE.match(stripped)
        blob = fence.group(1) if fence else stripped
        if blob.lstrip().startswith("{"):
            call = _as_call(_first_json_object(blob))
            if call is not None and call["name"] in tool_names:
                return ParsedReply(None, [call], "bare")
    return ParsedReply(text.strip() or None, [], "text")


def check_calls(calls: Sequence[dict], tools: Sequence[dict] | None) -> list[dict]:
    """Per call: is the tool known, and are its required arguments present (the validity numbers in the log)."""
    schemas = {t.get("function", {}).get("name"): t.get("function", {}).get("parameters") or {} for t in tools or []}
    checked = []
    for call in calls:
        schema = schemas.get(call["name"])
        args = json.loads(call["arguments"])
        missing = [k for k in (schema or {}).get("required", []) if k not in args] if schema is not None else []
        checked.append({"name": call["name"], "known": schema is not None, "missing": missing})
    return checked


class ContentStream:
    """Which part of the reply decoded so far is safe to stream as content.

    Holds back text that may still turn into a tool call: everything after <tool_call>, a suffix that
    could be the start of the tag, and (for the lenient parser) a reply that opens with '{' or '```'.
    Leading and trailing whitespace is never streamed, so the streamed content equals parse_reply's.
    """

    def __init__(self):
        self.sent = 0  # end of the streamed part of the text; 0 until something was streamed

    def _safe_end(self, text: str) -> int:
        lead = text.lstrip()
        if self.sent == 0 and (not lead or lead[0] in "{`" or (lead[0] == "<" and (
                lead.startswith(TOOL_OPEN) or TOOL_OPEN.startswith(lead)))):
            return 0
        cut = text.find(TOOL_OPEN)
        end = cut if cut >= 0 else len(text)
        if cut < 0:
            for k in range(min(len(TOOL_OPEN) - 1, len(text)), 0, -1):
                if TOOL_OPEN.startswith(text[-k:]):
                    end -= k
                    break
        while end > 0 and text[end - 1] == "�":  # an incomplete multi-byte character
            end -= 1
        return max(len(text[:end].rstrip()), self.sent)  # trailing whitespace may come before a tool call

    def update(self, text: str) -> str:
        end = self._safe_end(text)
        start = self.sent or len(text) - len(text.lstrip())
        if end <= start:
            return ""
        self.sent = end
        return text[start:end]

    def finish(self, text: str, parsed: ParsedReply) -> str:
        """The rest of the content once the reply is complete."""
        if self.sent == 0:
            return parsed.content or ""
        cut = text.find(TOOL_OPEN)
        end = cut if cut >= 0 and parsed.calls else len(text)
        out = text[self.sent : end].rstrip()
        self.sent = max(self.sent, end)
        return out


# --- Generation with prefix reuse ------------------------------------------------


def common_prefix(a: Sequence[int], b: Sequence[int]) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


@dataclass
class Generation:
    gen_ids: list[int]  # generated ids, including the stop token if there was one
    text: str  # decoded reply without the stop token
    finish: str  # "stop" or "length"
    prompt_tokens: int
    reused_tokens: int  # prompt tokens whose KV came from an earlier request
    computed_tokens: int  # prompt tokens prefilled for this request
    ttft_ms: float
    decode_ms: float
    session: str  # "new", "continued" or "forked"
    aborted: bool = False


class Backend:
    """One model, a pool of KV sessions and a lock: requests run one at a time."""

    def __init__(self, model, tokenizer, *, max_sessions: int = 4, chunk: int = 1024, adapter: str | None = None):
        self.model = model
        self.tokenizer = tokenizer
        self.max_sessions = max_sessions
        self.chunk = chunk
        self.adapter = adapter
        self.sessions: list[KVSession] = []  # least recently used first
        self.lock = threading.Lock()
        self.stop_ids = {tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.convert_tokens_to_ids("<|endoftext|>")}
        self.device = next(model.parameters()).device

    def _sync(self) -> float:
        if self.device.type == "mps":
            torch.mps.synchronize()
        elif self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def take_session(self, prompt_ids: Sequence[int], reuse: bool) -> tuple[KVSession, int, str]:
        """A session whose cache holds prompt_ids[:n] and whose ids equal it. Returns (session, n, how)."""
        if not reuse:
            return KVSession(self.model), 0, "new"
        best, best_n = None, 0
        for s in self.sessions:
            n = common_prefix(s.ids[: s.cache_len], prompt_ids)
            if n > best_n:
                best, best_n = s, n
        n = min(best_n, len(prompt_ids) - 1)  # at least one token must be fed to get logits
        if best is None or n <= 0:
            s, how, n = KVSession(self.model), "new", 0
        elif n >= best.cache_len // 2:
            s, how = best, "continued"
            self.sessions.remove(s)
            s.crop_to(n)
        else:
            s, how = KVSession(self.model), "forked"
            s.cache = copy.deepcopy(best.cache)
            s.crop_to(n)
        s.ids = list(prompt_ids[:n])
        self.sessions.append(s)
        while len(self.sessions) > self.max_sessions:
            self.sessions.pop(0)
        return s, n, how

    def prefill(self, s: KVSession, prompt_ids: Sequence[int]) -> torch.Tensor:
        """Feed prompt_ids[cache_len:] in chunks; returns the last position's logits."""
        logits = None
        pos = s.cache_len
        while pos < len(prompt_ids):
            end = min(len(prompt_ids), pos + self.chunk)
            s.extend(prompt_ids[pos:end])
            logits = s.prefill()
            pos = end
        return logits[-1]

    def generate(
        self, prompt_ids: Sequence[int], *, reuse: bool, max_new_tokens: int,
        on_token: Callable[[list[int]], bool] | None = None,
    ) -> Generation:
        """Greedy decoding. `on_token(gen_ids)` runs after each token; returning False stops (client gone)."""
        t0 = self._sync()
        s, reused, how = self.take_session(prompt_ids, reuse)
        gen: list[int] = []
        aborted = False
        try:
            with s.adapter(self.adapter):
                logits = self.prefill(s, prompt_ids)
                while True:
                    token = int(logits.argmax())
                    if not gen:
                        t1 = time.perf_counter()  # argmax() above waited for the GPU
                    gen.append(token)
                    if token in self.stop_ids or len(gen) >= max_new_tokens:
                        break
                    if on_token is not None and not on_token(gen):
                        aborted = True
                        break
                    s.extend([token])
                    logits = s.prefill()[-1]
        except BaseException:
            if s in self.sessions:
                self.sessions.remove(s)  # a failed forward may leave some layers holding the new tokens
            raise
        if not reuse and s in self.sessions:
            self.sessions.remove(s)
        s.extend(gen[-1:])  # ids = prompt + every generated token; the cache lacks the last one
        t2 = self._sync()
        stopped = gen[-1] in self.stop_ids
        text = self.tokenizer.decode(gen[:-1] if stopped else gen)
        return Generation(
            gen_ids=gen, text=text, finish="stop" if stopped or aborted else "length",
            prompt_tokens=len(prompt_ids), reused_tokens=reused, computed_tokens=len(prompt_ids) - reused,
            ttft_ms=(t1 - t0) * 1000, decode_ms=(t2 - t1) * 1000, session=how, aborted=aborted,
        )


# --- HTTP --------------------------------------------------------------------------


class BadRequest(Exception):
    pass


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, backend: Backend, *, max_new_tokens: int = 1024, lenient: bool = True,
                 log_path: Path | None = None, dump_dir: Path | None = None, quiet: bool = False):
        super().__init__(address, Handler)
        self.backend = backend
        self.max_new_tokens = max_new_tokens
        self.lenient = lenient
        self.log_path = log_path
        self.dump_dir = dump_dir
        self.quiet = quiet
        self.counter = 0
        self.counter_lock = threading.Lock()
        for p in (log_path.parent if log_path else None, dump_dir):
            if p:
                p.mkdir(parents=True, exist_ok=True)

    def next_id(self) -> int:
        with self.counter_lock:
            self.counter += 1
            return self.counter


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server: Server

    def log_message(self, fmt, *args):
        if not self.server.quiet:
            super().log_message(fmt, *args)

    # -- plumbing

    def _send_json(self, status: int, obj: dict) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str, kind: str = "invalid_request_error") -> None:
        self._send_json(status, {"error": {"message": message, "type": kind, "code": None}})

    def _chunk(self, data: bytes) -> None:
        self.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
        self.wfile.flush()

    def _sse(self, obj) -> None:
        payload = obj if isinstance(obj, str) else json.dumps(obj, ensure_ascii=False)
        self._chunk(f"data: {payload}\n\n".encode())

    # -- routes

    def do_GET(self):
        if self.path.rstrip("/") in ("/v1/models", "/models"):
            now = int(time.time())
            data = [{"id": m, "object": "model", "created": now, "owned_by": "rolekv", "context_length": CONTEXT_WINDOW}
                    for m in MODELS]
            self._send_json(200, {"object": "list", "data": data})
        elif self.path.rstrip("/") in ("", "/health"):
            self._send_json(200, {"ok": True})
        else:
            self._error(404, f"no route {self.path}")

    def do_POST(self):
        if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
            self._error(404, f"no route {self.path}")
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            self._complete(body)
        except BadRequest as e:
            self._error(400, str(e))
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _complete(self, body: dict) -> None:
        srv, backend = self.server, self.server.backend
        received = time.perf_counter()
        rid = srv.next_id()
        model_id = body.get("model")
        if model_id not in MODELS:
            raise BadRequest(f"unknown model {model_id!r}; expected one of {sorted(MODELS)}")
        messages, tools = body.get("messages") or [], body.get("tools") or None
        if not messages:
            raise BadRequest("messages is empty")
        tool_names = [t.get("function", {}).get("name") for t in tools or []]
        prompt_ids = render(backend.tokenizer, messages, tools)
        head_tokens = 0  # the system prompt and the tool schemas: the part every request of a session repeats
        if messages[0].get("role") in ("system", "developer"):
            head_tokens = len(render(backend.tokenizer, messages[:1], tools, generation_prompt=False))
        room = CONTEXT_WINDOW - len(prompt_ids)
        if room <= 0:
            raise BadRequest(f"prompt is {len(prompt_ids)} tokens; the context window is {CONTEXT_WINDOW}")
        asked = body.get("max_completion_tokens") or body.get("max_tokens") or srv.max_new_tokens
        max_new = max(1, min(int(asked), srv.max_new_tokens, room))
        stream = bool(body.get("stream"))
        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        cid, created = f"chatcmpl-{uuid.uuid4().hex[:24]}", int(time.time())

        def chunk(delta: dict, finish: str | None = None) -> dict:
            return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}

        content_stream = ContentStream()
        on_token = None
        if stream:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            self._sse(chunk({"role": "assistant", "content": ""}))

            def on_token(gen_ids: list[int]) -> bool:
                text = backend.tokenizer.decode(gen_ids)
                piece = content_stream.update(text)
                if piece:
                    try:
                        self._sse(chunk({"content": piece}))
                    except (BrokenPipeError, ConnectionResetError):
                        return False
                return True

        with backend.lock:
            waited_ms = (time.perf_counter() - received) * 1000
            g = backend.generate(prompt_ids, reuse=MODELS[model_id], max_new_tokens=max_new, on_token=on_token)
        parsed = parse_reply(g.text, tool_names, lenient=srv.lenient)
        finish = "tool_calls" if parsed.calls else g.finish
        tool_calls = [{"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                       "function": {"name": c["name"], "arguments": c["arguments"]}} for c in parsed.calls]
        usage = {"prompt_tokens": g.prompt_tokens, "completion_tokens": len(g.gen_ids),
                 "total_tokens": g.prompt_tokens + len(g.gen_ids),
                 "prompt_tokens_details": {"cached_tokens": g.reused_tokens}}
        response = None
        if stream:
            if not g.aborted:
                rest = content_stream.finish(g.text, parsed)
                if rest:
                    self._sse(chunk({"content": rest}))
                for i, call in enumerate(tool_calls):
                    self._sse(chunk({"tool_calls": [{"index": i, **call}]}))
                self._sse(chunk({}, finish))
                if include_usage:
                    self._sse({"id": cid, "object": "chat.completion.chunk", "created": created, "model": model_id,
                               "choices": [], "usage": usage})
                self._sse("[DONE]")
                self._chunk(b"")
        else:
            message = {"role": "assistant", "content": parsed.content}
            if tool_calls:
                message["tool_calls"] = tool_calls
            response = {"id": cid, "object": "chat.completion", "created": created, "model": model_id,
                        "choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage}
            self._send_json(200, response)

        record = {
            "request": rid, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "model": model_id, "stream": stream,
            "messages": len(messages), "tools": len(tools or []), "prompt_tokens": g.prompt_tokens,
            "head_tokens": head_tokens, "reused_tokens": g.reused_tokens, "computed_tokens": g.computed_tokens,
            "session": g.session, "ttft_ms": round(g.ttft_ms, 1), "decode_tokens": len(g.gen_ids),
            "decode_ms": round(g.decode_ms, 1), "wait_ms": round(waited_ms, 1),
            "wall_ms": round((time.perf_counter() - received) * 1000, 1), "finish": finish, "aborted": g.aborted,
            "max_new_tokens": max_new, "parse_mode": parsed.mode, "tool_calls": check_calls(parsed.calls, tools),
            "parse_errors": parsed.errors, "text": g.text[:2000],
        }
        if srv.log_path:
            with open(srv.log_path, "a") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        if srv.dump_dir:
            (srv.dump_dir / f"{rid:04d}.request.json").write_text(json.dumps(body, ensure_ascii=False, indent=1))
            out = response or {"streamed": True, "content": parsed.content, "tool_calls": tool_calls,
                               "finish_reason": finish, "usage": usage, "raw_text": g.text}
            (srv.dump_dir / f"{rid:04d}.response.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
        if not srv.quiet:
            calls = ",".join(c["name"] for c in parsed.calls) or "-"
            print(f"[{rid}] {model_id} prompt={g.prompt_tokens} reused={g.reused_tokens} computed={g.computed_tokens} "
                  f"ttft={g.ttft_ms:.0f}ms decode={len(g.gen_ids)} wall={record['wall_ms']:.0f}ms "
                  f"finish={finish} calls={calls}", flush=True)


def main(argv: Sequence[str] | None = None) -> None:
    from rolekv import roles
    from rolekv.runtime import load_model

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--model", default=roles.BASE_MODEL)
    p.add_argument("--device", default="mps")
    p.add_argument("--dtype", choices=["bf16", "fp32"], default="bf16")
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--max-sessions", type=int, default=4)
    p.add_argument("--chunk", type=int, default=1024, help="prefill chunk size in tokens")
    p.add_argument("--strict", action="store_true", help="accept only <tool_call> blocks, not untagged JSON calls")
    p.add_argument("--log", type=Path, default=Path("tmp/serve_logs/requests.jsonl"))
    p.add_argument("--dump-dir", type=Path, default=None, help="keep every request and response body here")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    dtype = {"bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    model, tokenizer = load_model(["A0"], base_model=args.model, device=args.device, dtype=dtype)
    backend = Backend(model, tokenizer, max_sessions=args.max_sessions, chunk=args.chunk)
    srv = Server(("127.0.0.1", args.port), backend, max_new_tokens=args.max_new_tokens, lenient=not args.strict,
                 log_path=args.log, dump_dir=args.dump_dir, quiet=args.quiet)
    print(f"serving {args.model} on http://127.0.0.1:{srv.server_address[1]}/v1 as {', '.join(MODELS)}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
