"""A faster aLoRA forward for PEFT 0.21.2: slice the adapted suffix instead of masking it.

PEFT's ALoraLinearVariant.forward selects the adapted rows with a boolean mask
(`res_flat[mask] += ...`). The gather calls nonzero(), which waits for the GPU
on every LoRA layer of every forward: 168 syncs per token on the 0.5B model.
On the Mac GPU this made aLoRA decoding 6-15x slower than plain LoRA.

The adapted rows are always a contiguous suffix whose length is a Python int
(the offset), so a slice gives the same result with no sync. tests/test_fast_alora.py
checks it against PEFT's original on random inputs.
"""

from __future__ import annotations

from peft.tuners.lora import variants

_ORIGINAL_FORWARD = variants.ALoraLinearVariant.forward


def _sliced_forward(module, active_adapter, x, result, **kwargs):
    alora_offsets = kwargs.get("alora_offsets", None)
    if alora_offsets is None:
        return result  # adapter off everywhere (PEFT computes a zero-row update here)
    lora_A = module.lora_A[active_adapter]
    lora_B = module.lora_B[active_adapter]
    dropout = module.lora_dropout[active_adapter]
    scaling = module.scaling[active_adapter]
    x = x.to(lora_A.weight.dtype)

    # Same shape handling as PEFT: [B, T, D] for sequences, [B, D] for a single position.
    if result.dim() == 2:
        x, rows = x.unsqueeze(1), result.unsqueeze(1)
    else:
        rows = result
    T = rows.shape[1]
    offsets = [0 if o is None else min(int(o), T) for o in alora_offsets]

    if all(o == offsets[0] for o in offsets):
        k = offsets[0]
        if k:
            rows[:, T - k :] += lora_B(lora_A(dropout(x[:, T - k :]))) * scaling
    else:
        for i, k in enumerate(offsets):
            if k:
                rows[i, T - k :] += lora_B(lora_A(dropout(x[i, T - k :]))) * scaling
    return result


def install() -> None:
    """Use the sliced forward for every aLoRA layer, existing and future. Idempotent."""
    variants.ALoraLinearVariant.forward = staticmethod(_sliced_forward)


def uninstall() -> None:
    variants.ALoraLinearVariant.forward = staticmethod(_ORIGINAL_FORWARD)
