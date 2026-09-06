"""Project-defined FP32 max-absolute K64 scales."""

from __future__ import annotations

import torch


def max_abs_scale(values: torch.Tensor, dim: int = -1) -> torch.Tensor:
    max_abs = values.abs().amax(dim=dim)
    return torch.where(
        max_abs == 0,
        torch.ones_like(max_abs, dtype=torch.float32),
        max_abs.to(torch.float32) / 448.0,
    )


def weight_scales_k64(tiles: torch.Tensor) -> torch.Tensor:
    """Compute [T, 8] scales for [T, 8, 64] weight tiles."""

    if tiles.ndim != 3 or tiles.shape[1:] != (8, 64):
        raise ValueError("tiles must have shape [T, 8, 64]")
    return max_abs_scale(tiles, dim=-1)


def activation_scales_k64(padded: torch.Tensor) -> torch.Tensor:
    """Compute [M, K_tiles] scales for a padded [M, K] activation matrix."""

    if padded.ndim != 2 or padded.shape[1] % 64:
        raise ValueError("padded activations must have shape [M, K_pad]")
    blocks = padded.reshape(padded.shape[0], -1, 64)
    return max_abs_scale(blocks, dim=-1)
