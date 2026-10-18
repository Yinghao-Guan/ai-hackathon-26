"""Prompt segments, role headers and tokenization helpers.

This module is the code side of docs/CONTRACT.md sections 1-2. Data generation,
training and the runtime all build prompts through it, so each role header --
which doubles as that role's aLoRA invocation sequence -- tokenizes to the same
ids everywhere.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

ROLES = ("coder", "reviewer")

BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
DEV_MODEL = "Qwen/Qwen2.5-Coder-0.5B-Instruct"

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"

TOOL_OUTPUT_LIMIT = 2000
_TRUNCATED = "...[truncated]\n"

# Label value that the loss ignores (the PyTorch / transformers convention).
IGNORE_INDEX = -100

# One system prompt for both roles: it is the first shared segment, so it must
# not depend on which role speaks next. It also gives A0 (no adapters) the
# instructions each role needs.
SYSTEM_PROMPT = (
    "You are one member of a two-person coding team working in a Python repository.\n"
    "Each assistant turn starts with a role tag, <|role:coder|> or <|role:reviewer|>, "
    "naming the member who speaks.\n"
    "The coder replies with exactly one ```python code block that contains the requested "
    "function and the imports it needs, and nothing else.\n"
    "The reviewer checks the coder's latest code against the task and the test output, "
    "then replies APPROVE on the first line if the code is correct, or REVISE: followed "
    "by short, specific feedback."
)


# --- Segments (CONTRACT.md section 1) -----------------------------------------


def system_segment(prompt: str = SYSTEM_PROMPT) -> str:
    return f"{IM_START}system\n{prompt}{IM_END}\n"


def task_segment(repo: str, text: str, visible_test: str) -> str:
    return (
        f"{IM_START}user\n"
        f"# Repository file: utils.py\n```python\n{repo}\n```\n"
        f"# Task\n{text}\n"
        f"# Visible test\n{visible_test}{IM_END}\n"
    )


def role_header(role: str) -> str:
    """The segment that opens a role's turn. It is also the role's aLoRA invocation sequence."""
    if role not in ROLES:
        raise ValueError(f"unknown role {role!r}; expected one of {ROLES}")
    return f"{IM_START}assistant\n<|role:{role}|>\n"


def body_segment(output: str) -> str:
    return f"{output}{IM_END}\n"


def tool_segment(output: str) -> str:
    """Wrap test output as a user turn, keeping at most TOOL_OUTPUT_LIMIT characters.

    When the output is too long its tail is kept, because tracebacks and
    assertion errors end there.
    """
    if len(output) > TOOL_OUTPUT_LIMIT:
        output = _TRUNCATED + output[-(TOOL_OUTPUT_LIMIT - len(_TRUNCATED)) :]
    return f"{IM_START}user\n<tool_output>\n{output}\n</tool_output>{IM_END}\n"


def is_role_header(segment: str) -> bool:
    return any(segment == role_header(role) for role in ROLES)


# --- Tokenization (CONTRACT.md rules 1-2) --------------------------------------


def segment_ids(tokenizer, segment: str) -> list[int]:
    return tokenizer(segment, add_special_tokens=False)["input_ids"]


def tokenize_segments(tokenizer, segments: Iterable[str]) -> list[int]:
    """Tokenize each segment on its own and concatenate the ids.

    Never tokenize the joined string instead: BPE can merge across a segment
    boundary and change the role header's tokens.
    """
    ids: list[int] = []
    for segment in segments:
        ids.extend(segment_ids(tokenizer, segment))
    return ids


def invocation_tokens(tokenizer, role: str) -> list[int]:
    """The ids to pass as LoraConfig(alora_invocation_tokens=...) for this role."""
    return segment_ids(tokenizer, role_header(role))


def tokenize_sft_example(tokenizer, segments: Sequence[str], target: str) -> dict[str, list[int]]:
    """input_ids and labels for one SFT row (CONTRACT.md section 2).

    `segments` must end with a role header, and `target` is the full BODY
    segment. The loss falls on the target tokens only.
    """
    if not segments or not is_role_header(segments[-1]):
        raise ValueError("the last segment of an SFT example must be a role header")
    context = tokenize_segments(tokenizer, segments)
    target_ids = segment_ids(tokenizer, target)
    return {
        "input_ids": context + target_ids,
        "labels": [IGNORE_INDEX] * len(context) + target_ids,
    }


# --- Parsing role output (CONTRACT.md rules 3-4) -------------------------------

_CODE_BLOCK = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)```", re.DOTALL)


def parse_coder_output(text: str) -> str | None:
    """The code inside the first fenced block of a Coder reply, or None if there is none.

    `text` is the reply without the trailing <|im_end|>.
    """
    match = _CODE_BLOCK.search(text)
    return match.group(1) if match else None


def parse_reviewer_verdict(text: str) -> tuple[str, str]:
    """Classify a Reviewer reply.

    Returns ("APPROVE", ""), ("REVISE", feedback), or ("INVALID", text) when
    the first line is neither. `text` is the reply without the trailing <|im_end|>.
    """
    stripped = text.strip()
    first_line, _, rest = stripped.partition("\n")
    if first_line.startswith("APPROVE"):
        return "APPROVE", ""
    if first_line.startswith("REVISE"):
        feedback = first_line[len("REVISE") :].lstrip(":").strip()
        if rest:
            feedback = f"{feedback}\n{rest}".strip()
        return "REVISE", feedback
    return "INVALID", stripped
