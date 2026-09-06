"""Batched E4M3-constrained Lloyd-Max fitting."""

from __future__ import annotations

import torch

from .e4m3 import quantize_e4m3


_INIT_PROBABILITIES = torch.tensor(
    [0.0625, 0.1875, 0.3125, 0.4375, 0.5625, 0.6875, 0.8125, 0.9375],
    dtype=torch.float32,
)


def _quantile_initialization(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    counts = valid.sum(dim=1)
    safe_counts = counts.clamp_min(1)
    sorted_values = torch.sort(
        torch.where(valid, values, torch.full_like(values, float("inf"))), dim=1
    ).values
    positions = _INIT_PROBABILITIES.to(values.device)[None, :] * (safe_counts[:, None] - 1)
    lower_index = positions.floor().to(torch.long)
    upper_index = positions.ceil().to(torch.long)
    lower = torch.gather(sorted_values, 1, lower_index)
    upper = torch.gather(sorted_values, 1, upper_index)
    centers = lower + (upper - lower) * (positions - lower_index.to(positions.dtype))
    return torch.where(counts[:, None] > 0, centers, torch.zeros_like(centers))


def _assign(values: torch.Tensor, valid: torch.Tensor, centers: torch.Tensor) -> torch.Tensor:
    distances = (values[:, :, None] - centers[:, None, :]).square()
    distances = distances.masked_fill(~valid[:, :, None], float("inf"))
    return distances.argmin(dim=-1)


def fit_lut_tiles(
    normalized_tiles: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    max_iters: int = 50,
    tile_chunk_size: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fit one shared eight-entry E4M3 LUT for each N8 x K64 tile.

    The implementation loops only over tile chunks. All values, assignments,
    and centroid updates inside a chunk are tensorized.
    """

    if normalized_tiles.ndim != 3 or normalized_tiles.shape[1:] != (8, 64):
        raise ValueError("normalized_tiles must have shape [T, 8, 64]")
    if valid_mask.shape != normalized_tiles.shape:
        raise ValueError("valid_mask must match normalized_tiles")
    if max_iters < 1 or tile_chunk_size < 1:
        raise ValueError("max_iters and tile_chunk_size must be positive")

    tile_count = normalized_tiles.shape[0]
    all_luts: list[torch.Tensor] = []
    all_indices: list[torch.Tensor] = []
    for start in range(0, tile_count, tile_chunk_size):
        values = normalized_tiles[start : start + tile_chunk_size].reshape(-1, 512).float()
        valid = valid_mask[start : start + tile_chunk_size].reshape(-1, 512).bool()
        centers = quantize_e4m3(_quantile_initialization(values, valid))
        previous: torch.Tensor | None = None
        for _ in range(max_iters):
            assignments = _assign(values, valid, centers)
            if previous is not None and torch.equal(assignments, previous):
                break
            previous = assignments
            counts = torch.zeros_like(centers)
            totals = torch.zeros_like(centers)
            counts.scatter_add_(1, assignments, valid.to(centers.dtype))
            totals.scatter_add_(1, assignments, torch.where(valid, values, torch.zeros_like(values)))
            updated = torch.where(counts > 0, totals / counts.clamp_min(1), centers)
            centers = quantize_e4m3(updated)
        final_indices = _assign(values, valid, centers)
        all_luts.append(centers)
        all_indices.append(final_indices.reshape(-1, 8, 64).to(torch.uint8))
    if not all_luts:
        return (
            normalized_tiles.new_empty((0, 8), dtype=torch.float32),
            normalized_tiles.new_empty((0, 8, 64), dtype=torch.uint8),
        )
    return torch.cat(all_luts, dim=0), torch.cat(all_indices, dim=0)
