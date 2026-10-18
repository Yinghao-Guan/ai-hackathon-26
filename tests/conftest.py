"""Shared fixtures: the dev model with five random, non-zero adapters.

RT_DEVICE (default cpu) and RT_DTYPE (fp32 | bf16, default fp32) select where
the runtime tests run. fp32 checks equality to a tight tolerance; bf16 (for a
smoke run on the 4090) checks that the error is small next to the adapter's effect.
"""

import os

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from rolekv import roles

DEVICE = torch.device(os.environ.get("RT_DEVICE", "cpu"))
DTYPE = {"fp32": torch.float32, "bf16": torch.bfloat16}[os.environ.get("RT_DTYPE", "fp32")]
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
ADAPTERS = {
    "alora_coder": "coder",
    "alora_reviewer": "reviewer",
    "lora_coder": None,
    "lora_reviewer": None,
    "lora_all": None,
}
MIN_EFFECT = 0.1  # an adapter must move some logit by at least this much, or a test could pass vacuously
# fp32 tolerance relative to the logit scale. Mathematically equal schedules are not bit-identical on CPU:
# with the 209-token test context, a base-only chunked prefill differs from one full forward by 3.9e-5
# (logits up to 28), the random adapters amplify that to ~6e-4, and four one-token decode steps on top
# reach ~1.7e-3. Wiring bugs (adapter on the wrong tokens, or off) show up at the scale of the adapter's
# effect, > 0.1 and usually 15-30.
FP32_RTOL = 5e-4
# Relative (Frobenius) tolerance for comparing cached K/V tensors against a fresh base prefill.
KV_RTOL = 1e-4 if DTYPE == torch.float32 else 2e-2


@pytest.fixture(scope="session")
def tokenizer():
    return AutoTokenizer.from_pretrained(roles.DEV_MODEL)


def adapter_config(tokenizer, role: str | None) -> LoraConfig:
    return LoraConfig(
        r=8,
        lora_alpha=8,
        target_modules=TARGET_MODULES,
        init_lora_weights=False,  # random A and B, so every adapter actually changes the output
        task_type="CAUSAL_LM",
        alora_invocation_tokens=roles.invocation_tokens(tokenizer, role) if role else None,
    )


@pytest.fixture(scope="session")
def model(tokenizer):
    torch.manual_seed(0)
    base = AutoModelForCausalLM.from_pretrained(roles.DEV_MODEL, dtype=DTYPE, attn_implementation="sdpa").to(DEVICE)
    names = list(ADAPTERS)
    peft_model = get_peft_model(base, adapter_config(tokenizer, ADAPTERS[names[0]]), adapter_name=names[0])
    for name in names[1:]:
        peft_model.add_adapter(name, adapter_config(tokenizer, ADAPTERS[name]))
    peft_model.eval()
    return peft_model


@torch.inference_mode()
def full_logits(model, ids: list[int], adapter: str | None) -> torch.Tensor:
    """Reference: one forward over the whole sequence, no cache, PEFT finding the invocation itself."""
    input_ids = torch.tensor([ids], device=DEVICE)
    if adapter is None:
        with model.disable_adapter():
            return model(input_ids=input_ids, use_cache=False).logits[0].float()
    model.set_adapter(adapter, inference_mode=True)
    return model(input_ids=input_ids, use_cache=False).logits[0].float()


def _fp32_limit(expected: torch.Tensor) -> float:
    return FP32_RTOL * max(1.0, expected.abs().max().item())


def assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    actual, expected = actual.float(), expected.float()
    error = (actual - expected).abs().max().item()
    limit = _fp32_limit(expected) if DTYPE == torch.float32 else 1e-2 * expected.abs().max().item()
    assert error < limit, f"max abs error {error:.3g} (limit {limit:.3g})"


def assert_matches(actual: torch.Tensor, expected: torch.Tensor, base: torch.Tensor) -> None:
    """actual == expected, and expected differs from the base-model logits (non-vacuity)."""
    actual, expected, base = actual.float(), expected.float(), base.float()
    effect = (expected - base).abs().max().item()
    assert effect > MIN_EFFECT, f"adapter barely changes the logits ({effect:.3g}); the test would be vacuous"
    error = (actual - expected).abs().max().item()
    if DTYPE == torch.float32:
        limit = _fp32_limit(expected)
        assert error < limit, f"max abs error {error:.3g} (limit {limit:.3g}, adapter effect {effect:.3g})"
    else:
        assert error < 0.1 * effect, f"max abs error {error:.3g} vs adapter effect {effect:.3g}"
