import pytest
import torch
from peft import LoraConfig
from peft.tuners.lora.variants import calculate_alora_offsets
from transformers import AutoTokenizer

from rolekv import roles


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(roles.DEV_MODEL)


def _segments(*turns: tuple[str, str]) -> list[str]:
    """System + task, then (role, body) turns with a tool turn after each coder body."""
    segments = [
        roles.system_segment(),
        roles.task_segment("def add(a, b):\n    return a + b", "Write square(x).", "assert square(3) == 9"),
    ]
    for role, body in turns:
        segments.append(roles.role_header(role))
        if body is None:
            break
        segments.append(roles.body_segment(body))
        if role == "coder":
            segments.append(roles.tool_segment("1 passed"))
    return segments


def _alora_offset(tokenizer, role: str, ids: list[int]) -> int:
    config = {role: LoraConfig(alora_invocation_tokens=roles.invocation_tokens(tokenizer, role), task_type="CAUSAL_LM")}
    (offset,) = calculate_alora_offsets(config, role, torch.tensor([ids]))
    return offset


def test_headers_are_distinct_and_plain(tokenizer):
    coder = roles.invocation_tokens(tokenizer, "coder")
    reviewer = roles.invocation_tokens(tokenizer, "reviewer")
    assert coder != reviewer
    for ids in (coder, reviewer):
        assert ids[0] == tokenizer.convert_tokens_to_ids(roles.IM_START)
        assert tokenizer.unk_token_id is None or tokenizer.unk_token_id not in ids
        assert tokenizer.decode(ids) in (roles.role_header("coder"), roles.role_header("reviewer"))


def test_dev_and_base_models_share_header_ids(tokenizer):
    # We test on the 0.5B model and train the 1.5B one, so their header ids must match.
    base = AutoTokenizer.from_pretrained(roles.BASE_MODEL)
    for role in roles.ROLES:
        assert roles.invocation_tokens(base, role) == roles.invocation_tokens(tokenizer, role)


def test_prompt_ends_with_invocation(tokenizer):
    ids = roles.tokenize_segments(tokenizer, _segments(("coder", None)))
    inv = roles.invocation_tokens(tokenizer, "coder")
    assert ids[-len(inv) :] == inv


def test_peft_activates_at_last_header_of_role(tokenizer):
    segments = _segments(("coder", "```python\ndef square(x):\n    return x + x\n```"), ("reviewer", "REVISE: use x * x"), ("coder", None))
    ids = roles.tokenize_segments(tokenizer, segments)

    # Coder: active from the last coder header, not the first one.
    assert _alora_offset(tokenizer, "coder", ids) == len(roles.invocation_tokens(tokenizer, "coder"))

    # Reviewer: active from its header through everything after it.
    after_reviewer_header = roles.tokenize_segments(tokenizer, segments[-3:])
    assert _alora_offset(tokenizer, "reviewer", ids) == len(after_reviewer_header)


def test_sft_example_masks_context(tokenizer):
    segments = _segments(("coder", None))
    target = roles.body_segment("```python\ndef square(x):\n    return x * x\n```")
    example = roles.tokenize_sft_example(tokenizer, segments, target)

    context_len = len(roles.tokenize_segments(tokenizer, segments))
    target_ids = roles.segment_ids(tokenizer, target)
    assert example["input_ids"][context_len:] == target_ids
    assert example["labels"][:context_len] == [roles.IGNORE_INDEX] * context_len
    assert example["labels"][context_len:] == target_ids
    assert tokenizer.convert_tokens_to_ids(roles.IM_END) in target_ids

    with pytest.raises(ValueError):
        roles.tokenize_sft_example(tokenizer, segments[:-1], target)


def test_tool_segment_keeps_tail():
    output = "x" * 5000 + "AssertionError: square(3) returned 6"
    segment = roles.tool_segment(output)
    inner = segment.split("<tool_output>\n", 1)[1].split("\n</tool_output>", 1)[0]
    assert len(inner) == roles.TOOL_OUTPUT_LIMIT
    assert inner.endswith("AssertionError: square(3) returned 6")
    assert roles.tool_segment("1 passed").count("1 passed") == 1


def test_parse_coder_output():
    assert roles.parse_coder_output("```python\ndef f():\n    return 1\n```") == "def f():\n    return 1\n"
    assert roles.parse_coder_output("no code here") is None


def test_parse_reviewer_verdict():
    assert roles.parse_reviewer_verdict("APPROVE") == ("APPROVE", "")
    assert roles.parse_reviewer_verdict("REVISE: use x * x\nand handle 0") == ("REVISE", "use x * x\nand handle 0")
    assert roles.parse_reviewer_verdict("Looks fine") == ("INVALID", "Looks fine")


def test_unknown_role_is_rejected():
    with pytest.raises(ValueError):
        roles.role_header("planner")
