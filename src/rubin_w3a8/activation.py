"""K64 E4M3 fake quantization for Linear activations."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .e4m3 import quantize_e4m3
from .scales import activation_scales_k64


def fake_quantize_activation_k64(x: torch.Tensor) -> torch.Tensor:
    """Fake-quantize each activation row x K64 block and return FP32."""

    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    k = x.shape[-1]
    if k == 0:
        raise ValueError("the K dimension must be non-empty")
    flat = x.to(torch.float32).reshape(-1, k)
    k_pad = (k + 63) // 64 * 64
    padded = F.pad(flat, (0, k_pad - k), value=0.0)
    scales = activation_scales_k64(padded)
    normalized = padded.reshape(flat.shape[0], -1, 64) / scales[..., None]
    quantized = quantize_e4m3(normalized)
    restored = (quantized * scales[..., None]).reshape(flat.shape[0], k_pad)
    return restored[:, :k].reshape(x.shape)
