"""Qwen module conversion for the frozen W3A8 path."""

from __future__ import annotations

import torch
from torch import nn

from .linear import CachedLUTLinear
from .weight import quantize_weight


_ELIGIBLE_LINEAR_NAMES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)


def convert_qwen_to_w3a8(model: nn.Module) -> nn.Module:
    """Replace eligible Transformer Linear modules with cached LUT-W3 modules."""

    def replace_children(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if name in _ELIGIBLE_LINEAR_NAMES and isinstance(child, nn.Linear):
                qweight = quantize_weight(child.weight.detach().float())
                replacement = CachedLUTLinear(qweight, child.bias)
                replacement.to(device=child.weight.device)
                setattr(parent, name, replacement)
            else:
                replace_children(child)

    replace_children(model)
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model
