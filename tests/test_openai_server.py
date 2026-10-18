import json
import threading
import urllib.request

import pytest
from conftest import assert_close, assert_matches, full_logits

from rolekv import roles
from serve.openai_server import (
    Backend,
    ContentStream,
    Server,
    check_calls,
    common_prefix,
    model_specs,
    parse_reply,
    render,
    template_messages,
)

TOOLS = [
    {"type": "function", "function": {"name": "write", "description": "Write a file", "parameters": {
        "type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
        "required": ["path", "content"]}}},
    {"type": "function", "function": {"name": "bash", "description": "Run a command", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}},
]
NAMES = ["write", "bash"]
CALL = '<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>'
SYSTEM = {"role": "system", "content": "You are a coding agent."}


# --- Tool-call parsing -------------------------------------------------------------


def test_parse_tagged_call():
    r = parse_reply(CALL, NAMES)
    assert r.mode == "tagged" and r.content is None
    assert r.calls == [{"name": "bash", "arguments": '{"command": "ls"}'}]


def test_parse_content_then_several_calls():
    text = "Let me look.\n" + CALL + "\n" + CALL.replace("ls", "pwd")
    r = parse_reply(text, NAMES)
    assert r.content == "Let me look."
    assert [json.loads(c["arguments"])["command"] for c in r.calls] == ["ls", "pwd"]


def test_parse_unclosed_call_and_stray_brace():
    r = parse_reply('<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}}', NAMES)
    assert r.mode == "tagged" and json.loads(r.calls[0]["arguments"]) == {"command": "ls"}


def test_parse_string_arguments_are_kept_as_json():
    r = parse_reply('<tool_call>{"name": "bash", "arguments": "{\\"command\\": \\"ls\\"}"}</tool_call>', NAMES)
    assert json.loads(r.calls[0]["arguments"]) == {"command": "ls"}


def test_parse_bad_json_falls_back_to_text():
    text = '<tool_call>\n{"name": "bash", "arguments": {"command": }\n</tool_call>'
    r = parse_reply(text, NAMES)
    assert r.mode == "text" and r.calls == [] and r.content == text and len(r.errors) == 1


def test_parse_lenient_fenced_json():
    text = '```json\n{\n  "name": "write",\n  "arguments": {"path": "a.py", "content": "x = 1\\n"}\n}\n```'
    r = parse_reply(text, NAMES)
    assert r.mode == "bare" and r.content is None and r.calls[0]["name"] == "write"
    assert parse_reply(text, NAMES, lenient=False).mode == "text"
    assert parse_reply(text.replace('"write"', '"unknown"'), NAMES).mode == "text"  # only known tools


def test_plain_text_and_code_are_content():
    assert parse_reply("Done.", NAMES).content == "Done."
    code = "```python\nprint(1)\n```"
    assert parse_reply(code, NAMES).mode == "text"


def test_check_calls_reports_unknown_tools_and_missing_arguments():
    calls = [{"name": "write", "arguments": '{"path": "a.py"}'}, {"name": "nope", "arguments": "{}"}]
    assert check_calls(calls, TOOLS) == [
        {"name": "write", "known": True, "missing": ["content"]},
        {"name": "nope", "known": False, "missing": []},
    ]


def test_template_messages_converts_the_openai_shapes():
    msgs = [
        {"role": "developer", "content": [{"type": "text", "text": "sys"}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}}]},
        {"role": "tool", "tool_call_id": "c1", "content": "a.py"},
    ]
    out = template_messages(msgs)
    assert out[0] == {"role": "system", "content": "sys"}
    assert out[1]["tool_calls"][0]["function"]["arguments"] == {"command": "ls"}
    assert out[2] == {"role": "tool", "content": "a.py", "tool_call_id": "c1"}


def test_rendered_tool_call_round_trips(tokenizer):
    """A tagged call the model emits renders back the same way, so its tokens can be reused next request."""
    msgs = [SYSTEM, {"role": "user", "content": "list files"}]
    prompt = render(tokenizer, msgs, TOOLS)
    r = parse_reply(CALL, NAMES)
    history = msgs + [{"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": r.calls[0]}]}]
    rendered = render(tokenizer, history, TOOLS, generation_prompt=False)
    generated = tokenizer(CALL + "<|im_end|>", add_special_tokens=False)["input_ids"]
    assert rendered[: len(prompt) + len(generated)] == prompt + generated


def test_raw_history_repeats_the_generated_text(tokenizer):
    msgs = [SYSTEM, {"role": "user", "content": "list files"}]
    prompt = render(tokenizer, msgs, TOOLS)
    reply = '```json\n{"name": "bash", "arguments": {"command": "ls"}}\n```'
    call = {"id": "c1", "type": "function", "function": parse_reply(reply, NAMES).calls[0]}
    history = msgs + [{"role": "assistant", "content": None, "tool_calls": [call]},
                      {"role": "tool", "tool_call_id": "c1", "content": "a.py"}]
    generated = tokenizer(reply + "<|im_end|>", add_special_tokens=False)["input_ids"]
    n = len(prompt) + len(generated)
    assert render(tokenizer, history, TOOLS, raw={"c1": reply})[:n] == prompt + generated
    assert render(tokenizer, history, TOOLS)[:n] != prompt + generated  # the template writes <tool_call> tags
    assert render(tokenizer, history, TOOLS, raw={"other": reply}) == render(tokenizer, history, TOOLS)


# --- Streaming ---------------------------------------------------------------------


def stream_all(text: str) -> str:
    s, out = ContentStream(), ""
    for i in range(1, len(text) + 1):
        out += s.update(text[:i])
    return out + s.finish(text, parse_reply(text, NAMES))


def test_stream_never_shows_tool_call_markup():
    text = "Checking.\n" + CALL
    assert stream_all(text) == "Checking."


def test_stream_holds_back_a_partial_tag():
    s = ContentStream()
    assert s.update("ok <tool") == "ok"
    assert s.update("ok <tools are fine") == " <tools are fine"


def test_stream_holds_json_replies_until_the_end():
    text = '{"name": "bash", "arguments": {"command": "ls"}}'
    s = ContentStream()
    assert all(s.update(text[:i]) == "" for i in range(1, len(text) + 1))
    assert s.finish(text, parse_reply(text, NAMES)) == ""
    assert stream_all("plain answer") == "plain answer"


# --- Prefix reuse --------------------------------------------------------------------


def conversation_messages(n_turns: int) -> list[dict]:
    msgs = [SYSTEM, {"role": "user", "content": "Create is_prime.py and test it."}]
    for i in range(n_turns):
        msgs.append({"role": "assistant", "content": None, "tool_calls": [
            {"id": f"c{i}", "type": "function", "function": {"name": "bash", "arguments": json.dumps({"command": f"ls {i}"})}}]})
        msgs.append({"role": "tool", "tool_call_id": f"c{i}", "content": f"file_{i}.py"})
    return msgs


def conversation(tokenizer, n_turns: int) -> list[int]:
    return render(tokenizer, conversation_messages(n_turns), TOOLS)


def role_prompt(tokenizer, n_turns: int, role: str) -> tuple[list[int], int]:
    """A role request as the server builds it: the conversation, then the role header. Returns (ids, header_pos)."""
    ids = render(tokenizer, conversation_messages(n_turns), TOOLS, generation_prompt=False)
    return ids + roles.invocation_tokens(tokenizer, role), len(ids)


def test_common_prefix():
    assert common_prefix([1, 2, 3], [1, 2, 4, 5]) == 2
    assert common_prefix([], [1]) == 0
    assert common_prefix([1, 2], [1, 2]) == 2


def test_reuse_continues_the_session_and_matches_a_full_prefill(model, tokenizer):
    backend = Backend(model, tokenizer)
    p1, p2 = conversation(tokenizer, 1), conversation(tokenizer, 2)
    g1 = backend.generate(p1, reuse=True, max_new_tokens=3)
    assert (g1.reused_tokens, g1.computed_tokens, g1.session) == (0, len(p1), "new")
    cached = p1 + g1.gen_ids[:-1]  # the last generated token is never fed

    s, n, how = backend.take_session(p2, reuse=True)
    assert how == "continued" and n == common_prefix(cached, p2) >= len(p1) - 1
    assert s.cache_len == n and s.ids == p2[:n]
    with s.adapter(None):
        logits = backend.prefill(s, p2)
    assert s.cache_len == len(p2)
    assert_close(logits, full_logits(model, p2, None)[-1])


def test_reuse_and_noreuse_give_the_same_reply(model, tokenizer):
    p1, p2 = conversation(tokenizer, 1), conversation(tokenizer, 2)
    reuse, fresh = Backend(model, tokenizer), Backend(model, tokenizer)
    reuse.generate(p1, reuse=True, max_new_tokens=3)
    a = reuse.generate(p2, reuse=True, max_new_tokens=4)
    b = fresh.generate(p2, reuse=False, max_new_tokens=4)
    assert a.reused_tokens > 0 and b.reused_tokens == 0 and b.computed_tokens == len(p2)
    assert a.gen_ids == b.gen_ids
    assert fresh.sessions == []  # the baseline keeps nothing


def test_side_request_forks_and_keeps_the_main_session(model, tokenizer):
    backend = Backend(model, tokenizer)
    main = conversation(tokenizer, 15)  # long enough that the system prompt and tools are under half of it
    backend.generate(main, reuse=True, max_new_tokens=2)
    main_session = backend.sessions[-1]
    kept = main_session.cache_len
    side = render(tokenizer, [SYSTEM, {"role": "user", "content": "Give this session a title."}], TOOLS)
    g = backend.generate(side, reuse=True, max_new_tokens=2)
    assert g.session == "forked" and 0 < g.reused_tokens < kept // 2
    assert main_session.cache_len == kept and main_session in backend.sessions


def test_identical_prompt_recomputes_one_token(model, tokenizer):
    backend = Backend(model, tokenizer)
    p = conversation(tokenizer, 1)
    backend.generate(p, reuse=True, max_new_tokens=1)
    g = backend.generate(p, reuse=True, max_new_tokens=1)
    assert g.computed_tokens == 1 and g.reused_tokens == len(p) - 1


def test_role_switch_reuses_only_base_kv_and_matches_a_full_prefill(model, tokenizer):
    backend = Backend(model, tokenizer)
    p1, h1 = role_prompt(tokenizer, 1, "coder")
    g1 = backend.generate(p1, reuse=True, max_new_tokens=2, adapter="alora_coder", alora_start=h1)
    assert g1.reused_tokens == 0
    p2, h2 = role_prompt(tokenizer, 2, "reviewer")
    # The history repeats "<|im_start|>assistant\n" after h1, but the coder's adapter computed that KV.
    assert common_prefix(p1, p2) > h1
    s, n, how = backend.take_session(p2, reuse=True)
    assert (n, how) == (h1, "continued")
    with s.adapter("alora_reviewer"):
        logits = backend.prefill(s, p2, alora_start=h2)
    assert_matches(logits, full_logits(model, p2, "alora_reviewer")[-1], full_logits(model, p2, None)[-1])


def test_base_request_after_a_role_request_stops_at_the_header(model, tokenizer):
    backend = Backend(model, tokenizer)
    p1, h1 = role_prompt(tokenizer, 1, "coder")
    backend.generate(p1, reuse=True, max_new_tokens=2, adapter="alora_coder", alora_start=h1)
    g = backend.generate(conversation(tokenizer, 2), reuse=True, max_new_tokens=1)
    assert g.reused_tokens == h1


def test_model_specs():
    assert list(model_specs()) == ["rolekv-reuse", "rolekv-noreuse"]
    specs = model_specs(roles.ROLES)
    assert specs["rolekv-reviewer"].role == "reviewer" and specs["rolekv-reviewer"].reuse
    assert not specs["rolekv-coder-noreuse"].reuse


def test_chunked_prefill_matches_one_chunk(model, tokenizer):
    p = conversation(tokenizer, 2)
    small = Backend(model, tokenizer, chunk=16)
    s, _, _ = small.take_session(p, reuse=False)
    with s.adapter(None):
        logits = small.prefill(s, p)
    assert_close(logits, full_logits(model, p, None)[-1])


def test_pool_is_bounded(model, tokenizer):
    backend = Backend(model, tokenizer, max_sessions=2)
    for word in ("alpha", "beta", "gamma"):
        backend.generate(tokenizer(f"{word} " * 5, add_special_tokens=False)["input_ids"], reuse=True, max_new_tokens=1)
    assert len(backend.sessions) == 2


# --- HTTP ----------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server(model, tokenizer):
    srv = Server(("127.0.0.1", 0), Backend(model, tokenizer), max_new_tokens=4, quiet=True)
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()


def post(url: str, body: dict) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_http_models(server):
    with urllib.request.urlopen(server + "/v1/models") as resp:
        ids = [m["id"] for m in json.loads(resp.read())["data"]]
    assert ids == ["rolekv-reuse", "rolekv-noreuse"]


def test_http_completion_and_cached_tokens(server):
    body = {"model": "rolekv-reuse", "messages": [SYSTEM, {"role": "user", "content": "hi"}], "tools": TOOLS}
    status, raw = post(server + "/v1/chat/completions", body)
    first = json.loads(raw)
    assert status == 200 and first["object"] == "chat.completion"
    assert first["choices"][0]["message"]["role"] == "assistant"
    assert first["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    body["messages"] = body["messages"] + [{"role": "assistant", "content": "hello"}, {"role": "user", "content": "again"}]
    second = json.loads(post(server + "/v1/chat/completions", body)[1])
    assert second["usage"]["prompt_tokens_details"]["cached_tokens"] > 0


def test_http_stream(server):
    body = {"model": "rolekv-noreuse", "messages": [{"role": "user", "content": "hi"}], "stream": True,
            "stream_options": {"include_usage": True}}
    status, raw = post(server + "/v1/chat/completions", body)
    events = [line[len("data: "):] for line in raw.decode().split("\n") if line.startswith("data: ")]
    assert status == 200 and events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-2]["choices"][0]["finish_reason"] in ("stop", "length")
    assert chunks[-1]["choices"] == [] and chunks[-1]["usage"]["prompt_tokens_details"]["cached_tokens"] == 0


def test_http_rejects_unknown_model(server):
    status, raw = post(server + "/v1/chat/completions", {"model": "gpt-4", "messages": [{"role": "user", "content": "x"}]})
    assert status == 400 and "unknown model" in json.loads(raw)["error"]["message"]


def test_http_role_models_share_the_prefix(model, tokenizer):
    srv = Server(("127.0.0.1", 0), Backend(model, tokenizer), max_new_tokens=3, quiet=True, roles=roles.ROLES)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}/v1/chat/completions"
    try:
        msgs = conversation_messages(1)
        first = json.loads(post(url, {"model": "rolekv-coder", "messages": msgs, "tools": TOOLS})[1])
        later = msgs + [{"role": "assistant", "content": "done"}, {"role": "user", "content": "review it"}]
        second = json.loads(post(url, {"model": "rolekv-reviewer", "messages": later, "tools": TOOLS})[1])
        base = json.loads(post(url, {"model": "rolekv-reviewer-noreuse", "messages": later, "tools": TOOLS})[1])
    finally:
        srv.shutdown()
    assert first["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert second["usage"]["prompt_tokens_details"]["cached_tokens"] == first["usage"]["prompt_tokens"] - len(
        roles.invocation_tokens(tokenizer, "coder"))
    assert base["usage"]["prompt_tokens_details"]["cached_tokens"] == 0
    assert second["choices"][0]["message"] == base["choices"][0]["message"]
