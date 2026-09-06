"""Qwen module conversion for the W3A8 and W3A16 paths."""

from __future__ import annotations

import torch
from torch import nn

from .linear import CachedLUTLinear
from .weight import quantize_weight


_ELIGIBLE_LINEAR_NAMES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)


def _convert_qwen(model: nn.Module, *, quantize_activations: bool) -> nn.Module:
    """Replace eligible Transformer Linear modules with cached LUT-W3 modules."""

    def replace_children(parent: nn.Module) -> None:
        for name, child in list(parent.named_children()):
            if name in _ELIGIBLE_LINEAR_NAMES and isinstance(child, nn.Linear):
                qweight = quantize_weight(child.weight.detach().float())
                replacement = CachedLUTLinear(
                    qweight,
                    child.bias,
                    quantize_activations=quantize_activations,
                )
                replacement.to(device=child.weight.device)
                setattr(parent, name, replacement)
            else:
                replace_children(child)

    replace_children(model)
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model


def convert_qwen_to_w3a8(model: nn.Module) -> nn.Module:
    """Convert Qwen eligible Linear modules to W3 weights with A8 activations."""

    return _convert_qwen(model, quantize_activations=True)


def convert_qwen_to_w3a16(model: nn.Module) -> nn.Module:
    """Convert Qwen eligible Linear modules to W3 weights with BF16 activations."""

    return _convert_qwen(model, quantize_activations=False)
