"""N8 x K64 tensor grouping helpers."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def padded_shape(n: int, k: int) -> tuple[int, int]:
    if n < 0 or k < 0:
        raise ValueError("matrix dimensions must be non-negative")
    return ((n + 7) // 8 * 8, (k + 63) // 64 * 64)


def pad_matrix(x: torch.Tensor, shape: tuple[int, int] | None = None) -> torch.Tensor:
    """Zero-pad a rank-2 matrix to N8 x K64 boundaries."""

    if x.ndim != 2:
        raise ValueError("x must be rank 2")
    n, k = x.shape
    n_pad, k_pad = shape or padded_shape(n, k)
    if n_pad < n or k_pad < k:
        raise ValueError("target shape cannot crop x")
    return F.pad(x, (0, k_pad - k, 0, n_pad - n), value=0.0)


def to_n8k64_tiles(x: torch.Tensor) -> torch.Tensor:
    """Convert a padded [N, K] matrix to [N_tiles*K_tiles, 8, 64]."""

    if x.ndim != 2 or x.shape[0] % 8 or x.shape[1] % 64:
        raise ValueError("x must be padded to multiples of 8 and 64")
    n_pad, k_pad = x.shape
    n_tiles, k_tiles = n_pad // 8, k_pad // 64
    return x.reshape(n_tiles, 8, k_tiles, 64).permute(0, 2, 1, 3).reshape(
        n_tiles * k_tiles, 8, 64
    )


def from_n8k64_tiles(tiles: torch.Tensor, padded: tuple[int, int]) -> torch.Tensor:
    """Convert [N_tiles*K_tiles, 8, 64] back to a padded [N, K] matrix."""

    if tiles.ndim != 3 or tiles.shape[1:] != (8, 64):
        raise ValueError("tiles must have shape [T, 8, 64]")
    n_pad, k_pad = padded
    n_tiles, k_tiles = n_pad // 8, k_pad // 64
    if tiles.shape[0] != n_tiles * k_tiles:
        raise ValueError("tile count does not match padded shape")
    return tiles.reshape(n_tiles, k_tiles, 8, 64).permute(0, 2, 1, 3).reshape(
        n_pad, k_pad
    )


def tile_valid_mask(
    original: tuple[int, int], padded: tuple[int, int], *, device: torch.device
) -> torch.Tensor:
    """Return a valid-element mask in flattened N8 x K64 tile order."""

    n, k = original
    n_pad, k_pad = padded
    rows = torch.arange(n_pad, device=device)[:, None] < n
    cols = torch.arange(k_pad, device=device)[None, :] < k
    return to_n8k64_tiles(rows & cols)
