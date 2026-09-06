"""Tensorized LUT-W3 weight representation and reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .grouping import (
    from_n8k64_tiles,
    pad_matrix,
    padded_shape,
    tile_valid_mask,
    to_n8k64_tiles,
)
from .lloyd_max import fit_lut_tiles
from .scales import weight_scales_k64


@dataclass
class LUTWeight:
    indices: torch.Tensor
    luts: torch.Tensor
    scales: torch.Tensor
    original_shape: tuple[int, int]
    padded_shape: tuple[int, int]


def quantize_weight(weight: torch.Tensor, *, tile_chunk_size: int = 1024) -> LUTWeight:
    """Quantize a PyTorch [N, K] weight into tensorized LUT-W3 form."""

    if weight.ndim != 2:
        raise ValueError("weight must have shape [N, K]")
    original = (int(weight.shape[0]), int(weight.shape[1]))
    padded = padded_shape(*original)
    padded_weight = pad_matrix(weight.to(torch.float32), padded)
    tiles = to_n8k64_tiles(padded_weight)
    scales_flat = weight_scales_k64(tiles)
    normalized = tiles / scales_flat[..., None]
    valid = tile_valid_mask(original, padded, device=weight.device)
    luts_flat, indices_flat = fit_lut_tiles(
        normalized,
        valid,
        tile_chunk_size=tile_chunk_size,
    )
    n_tiles, k_tiles = padded[0] // 8, padded[1] // 64
    return LUTWeight(
        indices=from_n8k64_tiles(indices_flat, padded).contiguous(),
        luts=luts_flat.reshape(n_tiles, k_tiles, 8).contiguous(),
        scales=scales_flat.reshape(n_tiles, k_tiles, 8).contiguous(),
        original_shape=original,
        padded_shape=padded,
    )


def reconstruct_weight(qweight: LUTWeight) -> torch.Tensor:
    """Materialize the dense FP32 weight represented by ``qweight``."""

    n_pad, k_pad = qweight.padded_shape
    n_tiles, k_tiles = n_pad // 8, k_pad // 64
    tile_indices = to_n8k64_tiles(qweight.indices).to(torch.long)
    luts = qweight.luts.reshape(-1, 8)
    values = torch.gather(luts[:, None, :].expand(-1, 512, -1), 2, tile_indices.reshape(-1, 512, 1))
    values = values.squeeze(-1).reshape(-1, 8, 64)
    values = values * qweight.scales.reshape(-1, 8, 1)
    dense_padded = from_n8k64_tiles(values, qweight.padded_shape)
    n, k = qweight.original_shape
    return dense_padded[:n, :k].contiguous()
