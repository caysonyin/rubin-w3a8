"""Correctness and cached dense Linear paths."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .activation import fake_quantize_activation_k64
from .grouping import to_n8k64_tiles
from .weight import LUTWeight, reconstruct_weight


def _lookup_weight(qweight: LUTWeight) -> torch.Tensor:
    """Explicit index -> LUT lookup, materialized for the reference path."""

    tile_indices = to_n8k64_tiles(qweight.indices).to(torch.long)
    luts = qweight.luts.reshape(-1, 8)
    values = torch.gather(
        luts[:, None, :].expand(-1, 512, -1),
        2,
        tile_indices.reshape(-1, 512, 1),
    ).squeeze(-1)
    values = values.reshape(-1, 8, 64) * qweight.scales.reshape(-1, 8, 1)
    n_pad, k_pad = qweight.padded_shape
    n_tiles, k_tiles = n_pad // 8, k_pad // 64
    dense_padded = values.reshape(n_tiles, k_tiles, 8, 64).permute(0, 2, 1, 3).reshape(
        n_pad, k_pad
    )
    n, k = qweight.original_shape
    return dense_padded[:n, :k]


def reference_linear_lookup(
    x: torch.Tensor, qweight: LUTWeight, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """Run explicit LUT lookup plus K64 activation quantization."""

    if x.shape[-1] != qweight.original_shape[1]:
        raise ValueError("x K dimension does not match qweight")
    input_dtype = x.dtype
    x_hat = fake_quantize_activation_k64(x)
    weight_hat = _lookup_weight(qweight)
    bias_fp32 = bias.to(torch.float32) if bias is not None else None
    y_fp32 = F.linear(x_hat.to(torch.float32), weight_hat.to(torch.float32), bias_fp32)
    return y_fp32.to(input_dtype)


class CachedLUTLinear(nn.Module):
    """A Linear module with one-time dense LUT-W3 reconstruction."""

    def __init__(self, qweight: LUTWeight, bias: torch.Tensor | None = None) -> None:
        super().__init__()
        self.register_buffer("indices", qweight.indices.detach().clone())
        self.register_buffer("luts", qweight.luts.detach().clone())
        self.register_buffer("scales", qweight.scales.detach().clone())
        self.register_buffer("weight_hat", reconstruct_weight(qweight).detach())
        if bias is not None:
            self.register_buffer("bias", bias.detach().to(torch.float32).clone())
        else:
            self.bias = None
        self.original_shape = qweight.original_shape
        self.padded_shape = qweight.padded_shape

    @property
    def qweight(self) -> LUTWeight:
        return LUTWeight(
            indices=self.indices,
            luts=self.luts,
            scales=self.scales,
            original_shape=self.original_shape,
            padded_shape=self.padded_shape,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x_hat = fake_quantize_activation_k64(x)
        y_fp32 = F.linear(
            x_hat.to(torch.float32),
            self.weight_hat.to(torch.float32),
            self.bias,
        )
        return y_fp32.to(input_dtype)
