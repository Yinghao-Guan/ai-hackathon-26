import pytest
import torch
from types import SimpleNamespace

from rolekv import fast_alora


def fake_layer(d_in=16, d_out=24, r=4, seed=0):
    """Just the attributes the variant forward reads, with random non-zero weights."""
    torch.manual_seed(seed)
    lora_A, lora_B = torch.nn.Linear(d_in, r, bias=False), torch.nn.Linear(r, d_out, bias=False)
    return SimpleNamespace(lora_A={"a": lora_A}, lora_B={"a": lora_B}, lora_dropout={"a": torch.nn.Identity()},
                           scaling={"a": 0.5})


@pytest.mark.parametrize(
    "shape, offsets",
    [
        ((1, 7), [3]),  # header part-way through the chunk
        ((1, 7), [7]),  # whole chunk adapted
        ((1, 7), [20]),  # offset beyond the chunk is clipped
        ((1, 7), [0]),  # adapter off
        ((1, 7), [None]),  # invocation not found: adapter off
        ((1, 1), [1]),  # one-token decode step
        ((3, 5), [2, 2, 2]),  # batch, same offset
        ((3, 5), [1, 0, 5]),  # batch, different offsets
        ((2,), [1, 1]),  # 2-D input: one position per row
    ],
)
@torch.inference_mode()
def test_sliced_forward_matches_peft(shape, offsets):
    layer = fake_layer()
    x = torch.randn(*shape, 16)
    base = torch.randn(*shape, 24)
    expected = fast_alora._ORIGINAL_FORWARD(layer, "a", x.clone(), base.clone(), alora_offsets=offsets)
    got = fast_alora._sliced_forward(layer, "a", x.clone(), base.clone(), alora_offsets=offsets)
    torch.testing.assert_close(got, expected, rtol=0, atol=1e-6)


@torch.inference_mode()
def test_no_offsets_means_adapter_off():
    layer = fake_layer()
    x, base = torch.randn(1, 4, 16), torch.randn(1, 4, 24)
    torch.testing.assert_close(fast_alora._sliced_forward(layer, "a", x, base.clone()), base)
