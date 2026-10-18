import copy
import json

import pytest
import torch
from conftest import DEVICE, DTYPE, KV_RTOL, MIN_EFFECT, assert_close, assert_matches, full_logits
from transformers import DynamicCache

from rolekv import roles
from rolekv.cache import KVSession
from rolekv.costs import MB, kv_bytes_per_token
from rolekv.runtime import METHODS, Engine, forced_ids, load_model, run_session

TASK = roles.task_segment(
    "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b",
    "Write a function square(x) that returns x squared.",
    "assert square(3) == 9",
)
CODE = "```python\ndef square(x):\n    return x * x\n```"
CODE_BAD = "```python\ndef square(x):\n    return x + x\n```"
TOOL_OUTPUTS = ["AssertionError: square(3) returned 6", "1 passed"]
FORCED = [CODE_BAD, "REVISE: use x * x", CODE, "APPROVE"]


def context_segments(role: str) -> list[str]:
    """What precedes a turn by `role`; the reviewer's context includes an earlier coder turn."""
    segments = [roles.system_segment(), TASK]
    if role == "reviewer":
        segments += [
            roles.role_header("coder"),
            roles.body_segment("```python\ndef square(x):\n    return x + x\n```"),
            roles.tool_segment("AssertionError: square(3) returned 6"),
        ]
    return segments


def turn_tail(tokenizer, role: str) -> list[int]:
    body = CODE if role == "coder" else "REVISE: use x * x"
    return roles.tokenize_segments(tokenizer, [roles.role_header(role), roles.body_segment(body)])


# --- T1: cache equivalence ------------------------------------------------------


@pytest.mark.parametrize("role", roles.ROLES)
def test_cache_equivalence(model, tokenizer, role):
    adapter = f"alora_{role}"
    ctx = roles.tokenize_segments(tokenizer, context_segments(role))
    tail = turn_tail(tokenizer, role)
    ids, n, hp = ctx + tail, len(tail), len(ctx)
    ref, base = full_logits(model, ids, adapter), full_logits(model, ids, None)

    # (a) Base-prefill the context, then prefill header + body with the aLoRA.
    s = KVSession(model)
    s.extend(ctx)
    with s.adapter(None):
        s.prefill()
    s.extend(tail)
    with s.adapter(adapter):
        assert_matches(s.prefill(alora_start=hp, keep=n), ref[-n:], base[-n:])

    # (b) Same tail in three chunks; the first one ends inside the role header.
    s = KVSession(model)
    s.extend(ctx)
    with s.adapter(adapter):
        s.prefill()  # alora_start=None: adapter off for the context
        bounds = [hp, hp + 3, hp + n // 2, len(ids)]
        for lo, hi in zip(bounds, bounds[1:]):
            s.extend(ids[lo:hi])
            assert_matches(s.prefill(alora_start=hp, keep=hi - lo), ref[lo:hi], base[lo:hi])

    # (c) Context and tail in one forward: base rows stay exactly base.
    s = KVSession(model)
    s.extend(ids)
    with s.adapter(adapter):
        got = s.prefill(alora_start=hp, keep=0)
    assert_matches(got[-n:], ref[-n:], base[-n:])
    assert_close(got[:hp], base[:hp])


# --- T3: adapter stays on during decode -------------------------------------------


def test_adapter_stays_on_during_decode(model, tokenizer):
    adapter = "alora_coder"
    ctx = roles.tokenize_segments(tokenizer, context_segments("coder"))
    hp = len(ctx)
    s = KVSession(model)
    s.extend(ctx + roles.invocation_tokens(tokenizer, "coder"))
    with s.adapter(adapter):
        logits = s.prefill(alora_start=hp)
        for _ in range(4):
            s.extend([int(logits[-1].argmax())])
            logits = s.prefill(alora_start=hp)
            assert_matches(logits[-1], full_logits(model, s.ids, adapter)[-1], full_logits(model, s.ids, None)[-1])

        # Negative control, documenting the PEFT pitfall: a one-token forward without
        # explicit offsets runs with the adapter OFF.
        token = int(logits[-1].argmax())
        s.extend([token])
        x = torch.tensor([[token]], device=DEVICE)
        with torch.inference_mode():
            raw = model(input_ids=x, past_key_values=copy.deepcopy(s.cache)).logits[0, -1]
            off = model(input_ids=x, past_key_values=copy.deepcopy(s.cache), alora_offsets=[0]).logits[0, -1]
        on = s.prefill(alora_start=hp)[-1]
    assert_close(raw, off)
    assert (raw.float() - on.float()).abs().max().item() > MIN_EFFECT


# --- T2: canonicalization and positions -------------------------------------------


def base_cache(model, ids: list[int]) -> DynamicCache:
    """Reference cache: one base-model forward over `ids`."""
    cache = DynamicCache(config=model.config)
    with torch.inference_mode(), model.disable_adapter():
        model(input_ids=torch.tensor([ids], device=DEVICE), past_key_values=cache)
    return cache


def kv_rel_errors(cache, ref, lo: int, hi: int) -> list[float]:
    """Relative error of every layer's keys and values over positions [lo, hi)."""
    errors = []
    for layer, ref_layer in zip(cache.layers, ref.layers, strict=True):
        for got, want in ((layer.keys, ref_layer.keys), (layer.values, ref_layer.values)):
            got, want = got[..., lo:hi, :].float(), want[..., lo:hi, :].float()
            errors.append(((got - want).norm() / want.norm()).item())
    return errors


def test_canonicalize_restores_base_cache(model, tokenizer):
    engine = Engine(model, tokenizer, "A5-H1")
    engine.start(TASK)
    result = engine.turn("coder", max_new_tokens=6)
    s, hp = engine.session, result.header_pos
    header_len = len(roles.invocation_tokens(tokenizer, "coder"))
    with torch.inference_mode():
        before = copy.deepcopy(s.cache)
    before_len = before.get_seq_length()
    assert before_len == hp + header_len + len(result.gen_ids) - 1  # the last token is never fed

    assert engine.canonicalize() == before_len - hp
    assert s.cache_len == len(s.ids)
    ref = base_cache(model, s.ids)
    # K is cached after RoPE, so matching keys also proves the positions survived the crop.
    assert max(kv_rel_errors(s.cache, ref, 0, len(s.ids))) < KV_RTOL
    # Before the recompute: the shared prefix was base, the turn's own span was not (non-vacuity).
    assert max(kv_rel_errors(before, ref, 0, hp)) < KV_RTOL
    assert max(kv_rel_errors(before, ref, hp, before_len)) > 10 * KV_RTOL

    # Continue with the reviewer on top of the repaired cache.
    reviewer_hp = len(s.ids)
    header = roles.invocation_tokens(tokenizer, "reviewer")
    s.extend(header)
    with s.adapter("alora_reviewer"):
        got = s.prefill(alora_start=reviewer_hp, keep=len(header))
    n = len(header)
    assert_matches(got, full_logits(model, s.ids, "alora_reviewer")[-n:], full_logits(model, s.ids, None)[-n:])


def test_lazy_canonicalization_happens_at_next_turn(model, tokenizer):
    engine = Engine(model, tokenizer, "A5-H1")
    engine.start(TASK)
    coder = engine.turn("coder", forced=forced_ids(tokenizer, CODE_BAD))
    engine.add_tool_output(TOOL_OUTPUTS[0])
    review = engine.turn("reviewer", forced=forced_ids(tokenizer, "REVISE: use x * x"))

    header_len = len(roles.invocation_tokens(tokenizer, "coder"))
    assert review.cost.canon_tokens == header_len + len(coder.gen_ids) - 1
    ref = base_cache(model, engine.session.ids[: review.header_pos])
    assert max(kv_rel_errors(engine.session.cache, ref, 0, review.header_pos)) < KV_RTOL


# --- T4: cost accounting ------------------------------------------------------------


def count_fed_tokens(model) -> tuple[list[int], object]:
    """Independent oracle: count every token id that reaches the embedding layer."""
    fed: list[int] = []
    handle = model.get_input_embeddings().register_forward_pre_hook(lambda module, args: fed.append(args[0].numel()))
    return fed, handle


@pytest.mark.parametrize("method", METHODS)
def test_cost_accounting(model, tokenizer, method):
    def seg(text: str) -> int:
        return len(roles.segment_ids(tokenizer, text))

    P = seg(roles.system_segment()) + seg(TASK)
    Hc, Hr = (len(roles.invocation_tokens(tokenizer, r)) for r in ("coder", "reviewer"))
    U = [seg(roles.tool_segment(out)) for out in TOOL_OUTPUTS]
    g = [len(forced_ids(tokenizer, text)) for text in FORCED]
    H = [Hc, Hr, Hc, Hr]
    # L[i] = session length when turn i starts, before its header; each turn adds header + body + "\n".
    L = [P, P + Hc + g[0] + 1 + U[0]]
    L.append(L[1] + Hr + g[1] + 1)
    L.append(L[2] + Hc + g[2] + 1 + U[1])

    calls: list[str] = []

    def runner(code: str) -> str:
        calls.append(code)
        return TOOL_OUTPUTS[len(calls) - 1]

    engine = Engine(model, tokenizer, method)
    fed, handle = count_fed_tokens(model)
    try:
        result = run_session(engine, TASK, runner, forced_outputs=FORCED)
    finally:
        handle.remove()
    costs = [t.cost for t in result.turns]
    assert [c.role for c in costs] == ["coder", "reviewer", "coder", "reviewer"]

    bytes_per_token = kv_bytes_per_token(model.config, DTYPE)
    for i, c in enumerate(costs):
        cached_after_prev = 0 if i == 0 else L[i - 1] + H[i - 1] + g[i - 1] - 1
        if method == "A2":
            expected = (0, 0, L[i] + H[i])
        elif method == "A5-H1" and i > 0:
            canon = H[i - 1] + g[i - 1] - 1
            expected = (L[i - 1], canon, L[i] + H[i] - L[i - 1] - canon)
        else:
            expected = (cached_after_prev, 0, L[i] + H[i] - cached_after_prev)
        assert (c.reused_tokens, c.canon_tokens, c.prefill_tokens) == expected, f"turn {i}"
        assert c.reused_tokens + c.canon_tokens + c.prefill_tokens == L[i] + H[i]
        assert c.decode_tokens == g[i]
        assert c.kv_mb == (L[i] + H[i] + g[i] - 1) * bytes_per_token / MB
        assert c.turn_ms >= c.ttft_ms > 0
        assert not c.truncated

    # Every token the model saw is accounted for: prefill + canon, plus g - 1 fed decode steps per turn.
    assert sum(fed) == sum(c.prefill_tokens + c.canon_tokens for c in costs) + sum(n - 1 for n in g)

    # The session's ids are exactly the contract tokenization, as in training data.
    segments = [roles.system_segment(), TASK]
    for i, (role, text) in enumerate(zip(["coder", "reviewer", "coder", "reviewer"], FORCED)):
        segments += [roles.role_header(role), roles.body_segment(text)]
        if role == "coder":
            segments.append(roles.tool_segment(TOOL_OUTPUTS[i // 2]))
    assert engine.session.ids == roles.tokenize_segments(tokenizer, segments)

    task = result.cost
    assert task.n_turns == 4
    assert task.prefill_tokens_total == sum(c.prefill_tokens + c.canon_tokens for c in costs)
    assert task.decode_tokens_total == sum(g)
    assert task.peak_kv_mb == costs[-1].kv_mb
    assert task.ctx_tokens == seg(TASK)
    assert result.verdicts == ["REVISE", "APPROVE"]
    assert calls == [roles.parse_coder_output(CODE_BAD), roles.parse_coder_output(CODE)]
    assert result.final_code == roles.parse_coder_output(CODE)

    rows = result.records(run_id="test", machine="mac", task_id=11, passed=True)
    turn_keys = {"type", "run_id", "machine", "method", "task_id", "turn", "role", "prefill_tokens", "reused_tokens",
                 "canon_tokens", "decode_tokens", "ttft_ms", "turn_ms", "kv_mb", "text"}
    task_keys = {"type", "run_id", "machine", "method", "task_id", "ctx_tokens", "passed", "n_turns",
                 "prefill_tokens_total", "decode_tokens_total", "gpu_s", "peak_kv_mb"}
    assert [r["type"] for r in rows] == ["turn"] * 4 + ["task"]
    assert all(turn_keys <= r.keys() for r in rows[:-1]) and task_keys <= rows[-1].keys()
    json.dumps(rows)  # serializable as JSONL


# --- T5: session protocol -----------------------------------------------------------


def test_protocol_stops_after_max_revise_and_on_invalid(model, tokenizer):
    calls: list[str] = []

    def runner(code: str) -> str:
        calls.append(code)
        return "AssertionError"

    result = run_session(Engine(model, tokenizer, "A5-H1"), TASK, runner,
                         forced_outputs=[CODE_BAD, "REVISE: a", CODE_BAD, "REVISE: b", CODE])
    assert [t.role for t in result.turns] == ["coder", "reviewer", "coder", "reviewer", "coder"]
    assert result.verdicts == ["REVISE", "REVISE"]
    assert len(calls) == 2  # no test run after the final coder turn
    assert result.final_code == roles.parse_coder_output(CODE)

    result = run_session(Engine(model, tokenizer, "A0"), TASK, runner, forced_outputs=[CODE, "Looks fine to me"])
    assert len(result.turns) == 2
    assert result.verdicts == ["INVALID"]


# --- T6: smoke test with real decoding ------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
def test_smoke_real_decoding(model, tokenizer, method):
    engine = Engine(model, tokenizer, method)
    fed, handle = count_fed_tokens(model)
    try:
        result = run_session(engine, TASK, lambda code: "1 passed", max_new_tokens={"coder": 6, "reviewer": 4})
    finally:
        handle.remove()
    costs = [t.cost for t in result.turns]
    assert sum(fed) == sum(c.prefill_tokens + c.canon_tokens + c.decode_tokens - 1 for c in costs)
    assert engine.session.ids[-2:] == [tokenizer.convert_tokens_to_ids(roles.IM_END)] + roles.segment_ids(tokenizer, "\n")
    json.dumps(result.records(run_id="smoke", machine="mac", task_id=0, passed=False))


# --- T7: load_model round trip ----------------------------------------------------------


def test_load_model_round_trip(model, tokenizer, tmp_path):
    model.save_pretrained(tmp_path)  # one subdirectory per adapter, the layout load_model expects
    loaded, _ = load_model(METHODS, base_model=roles.DEV_MODEL, adapter_dir=tmp_path, device=DEVICE, dtype=DTYPE)
    ids = roles.tokenize_segments(tokenizer, context_segments("reviewer")) + turn_tail(tokenizer, "reviewer")
    for name in ("alora_coder", "alora_reviewer", "lora_coder", "lora_reviewer", "lora_all"):
        assert_close(full_logits(loaded, ids, name), full_logits(model, ids, name))
    del loaded

    config_path = tmp_path / "alora_coder" / "adapter_config.json"
    config = json.loads(config_path.read_text())
    config["alora_invocation_tokens"] = roles.invocation_tokens(tokenizer, "reviewer")
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="alora_invocation_tokens"):
        load_model(["A5-H1"], base_model=roles.DEV_MODEL, adapter_dir=tmp_path, device=DEVICE, dtype=DTYPE)
