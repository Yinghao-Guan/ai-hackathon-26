"""One KV cache per session, fed chunk by chunk with explicit aLoRA activation.

Invariant: `ids` is the session's full logical token sequence and the cache
always holds exactly `ids[:cache_len]`. Tokens not yet computed are
`ids[cache_len:]` (`pending`); they are never tracked separately.

aLoRA activation is given as an absolute position, `alora_start` (where the
role header starts in `ids`). Each forward turns it into PEFT's per-call
offset, so a merged prefill, a chunked prefill and a one-token decode step all
use the same rule. Never let PEFT search for the invocation itself: a one-token
decode chunk does not contain it, and PEFT then silently turns the adapter off.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, nullcontext

import torch
from transformers import DynamicCache

from rolekv import fast_alora
from rolekv.costs import kv_bytes


def is_peft(model) -> bool:
    return hasattr(model, "peft_config")


def is_alora(model, adapter: str | None) -> bool:
    if adapter is None or not is_peft(model):
        return False
    return bool(getattr(model.peft_config[adapter], "alora_invocation_tokens", None))


def clear_alora_hooks(model) -> int:
    """Remove aLoRA offset hooks that PEFT leaves behind when a forward raises.

    PEFT registers them per call and removes them only on a clean exit; a stale
    hook would feed old offsets to later calls. Returns how many were removed.
    """
    from peft.tuners.lora.model import _alora_offsets_pre_forward_hook

    removed = 0
    for module in model.modules():
        hooks = module._forward_pre_hooks
        for hook_id, hook in list(hooks.items()):
            if isinstance(hook, functools.partial) and hook.func is _alora_offsets_pre_forward_hook:
                del hooks[hook_id]
                module._forward_pre_hooks_with_kwargs.pop(hook_id, None)
                removed += 1
    return removed


class KVSession:
    def __init__(self, model, config=None):
        fast_alora.install()  # PEFT's masked aLoRA forward syncs with the GPU on every layer
        self.model = model
        self.config = config if config is not None else model.config
        self.device = next(model.parameters()).device
        self.ids: list[int] = []
        self.cache = DynamicCache(config=self.config)
        self.active: str | None = None  # adapter used by prefill() inside the current adapter() block
        self.forward_tokens = 0  # every token fed to the model, for cross-checking the cost counters
        self.broken = False

    @property
    def cache_len(self) -> int:
        return self.cache.get_seq_length()

    @property
    def pending(self) -> list[int]:
        return self.ids[self.cache_len :]

    def extend(self, ids: Sequence[int]) -> None:
        self.ids.extend(ids)

    @contextmanager
    def adapter(self, name: str | None) -> Iterator[None]:
        """Run prefill() calls inside with `name` active; None means the base model.

        Enter once per turn: switching walks every module, so it must stay out of
        the decode loop.
        """
        if name is None:
            context = self.model.disable_adapter() if is_peft(self.model) else nullcontext()
        else:
            if self.model.active_adapter != name:
                # inference_mode=True keeps the adapter frozen; the default would set requires_grad.
                self.model.set_adapter(name, inference_mode=True)
            context = nullcontext()
        previous, self.active = self.active, name
        try:
            with context:
                yield
        finally:
            self.active = previous

    @torch.inference_mode()
    def prefill(self, *, alora_start: int | None = None, keep: int = 1) -> torch.Tensor:
        """Feed all pending tokens through the model in one forward.

        With an aLoRA adapter active, the adapter applies only to tokens at
        positions >= `alora_start`; with `alora_start=None` it is off entirely.
        Returns the logits of the last `keep` positions, shape [keep, vocab];
        keep=0 returns every position of this chunk.
        """
        if self.broken:
            raise RuntimeError("session is broken after a failed forward; start a new one")
        pending = self.pending
        if not pending:
            raise ValueError("nothing to prefill")
        start, end = self.cache_len, len(self.ids)
        kwargs = {}
        if is_alora(self.model, self.active):
            offset = 0 if alora_start is None else max(0, end - max(alora_start, start))
            kwargs["alora_offsets"] = [offset]
        input_ids = torch.tensor([pending], dtype=torch.long, device=self.device)
        try:
            out = self.model(input_ids=input_ids, past_key_values=self.cache, use_cache=True,
                             logits_to_keep=keep, **kwargs)
        except BaseException:
            clear_alora_hooks(self.model)
            self.broken = True  # some layers may already hold the new tokens
            raise
        self.forward_tokens += len(pending)
        return out.logits[0]

    def crop_to(self, n: int) -> int:
        """Truncate the cache to its first `n` tokens; `ids` is kept. Returns tokens dropped."""
        drop = self.cache_len - n
        if drop < 0:
            raise ValueError(f"cannot crop a cache of {self.cache_len} tokens to {n}")
        if drop:
            self.cache.crop(-drop)  # DynamicCache.crop takes minus the number of tokens to remove
        return drop

    def reset(self) -> None:
        """Drop the cache entirely (A2); `ids` is kept and becomes all pending."""
        self.cache = DynamicCache(config=self.config)

    def kv_bytes(self) -> int:
        return kv_bytes(self.cache)
