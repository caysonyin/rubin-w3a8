"""A small, deterministic E4M3 finite-number quantizer."""

from __future__ import annotations

from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _positive_grid() -> tuple[torch.Tensor, torch.Tensor]:
    """Return positive E4M3FN values and mantissa-even tie flags."""

    values: list[float] = [0.0]
    even: list[bool] = [True]
    # E=0 is handled as zero and subnormal values. E=15, m=7 is NaN in
    # E4M3FN, so the finite grid ends at m=6 (448).
    for mantissa in range(1, 8):
        values.append(mantissa * 2.0 ** -9)
        even.append(mantissa % 2 == 0)
    for exponent in range(1, 15):
        for mantissa in range(8):
            values.append((1.0 + mantissa / 8.0) * 2.0 ** (exponent - 7))
            even.append(mantissa % 2 == 0)
    for mantissa in range(7):
        values.append((1.0 + mantissa / 8.0) * 2.0**8)
        even.append(mantissa % 2 == 0)
    order = sorted(range(len(values)), key=values.__getitem__)
    return (
        torch.tensor([values[i] for i in order], dtype=torch.float32),
        torch.tensor([even[i] for i in order], dtype=torch.bool),
    )


def quantize_e4m3(x: torch.Tensor) -> torch.Tensor:
    """Quantize ``x`` to the finite E4M3 numerical grid and return FP32.

    E4M3FN has no infinities in its finite grid. Non-finite positive/negative
    infinity inputs saturate to +/-448; NaN is preserved as NaN. Ties use the
    even mantissa of the upper candidate, which also handles the zero and
    exponent-boundary ties deterministically.
    """

    if not isinstance(x, torch.Tensor):
        raise TypeError("x must be a torch.Tensor")
    values, even_flags = _positive_grid()
    values = values.to(device=x.device)
    even_flags = even_flags.to(device=x.device)

    xf = x.to(dtype=torch.float32)
    nan_mask = torch.isnan(xf)
    sign_mask = torch.signbit(xf)
    magnitude = torch.where(nan_mask, torch.zeros_like(xf), xf.abs())
    positions = torch.searchsorted(values, magnitude, right=False)
    lower_idx = (positions - 1).clamp(min=0, max=values.numel() - 1)
    upper_idx = positions.clamp(min=0, max=values.numel() - 1)
    lower = values[lower_idx]
    upper = values[upper_idx]
    lower_distance = (magnitude - lower).abs()
    upper_distance = (upper - magnitude).abs()
    choose_upper = upper_distance < lower_distance
    ties = upper_distance == lower_distance
    choose_upper = choose_upper | (ties & even_flags[upper_idx])
    quantized = torch.where(choose_upper, upper, lower)
    quantized = torch.where(sign_mask, -quantized, quantized)
    return torch.where(nan_mask, torch.full_like(quantized, float("nan")), quantized)


def e4m3_values(device: torch.device | str | None = None) -> torch.Tensor:
    """Expose the positive finite grid for tests and diagnostics."""

    values, _ = _positive_grid()
    return values.to(device=device)
