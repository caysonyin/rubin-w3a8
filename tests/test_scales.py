import torch

from rubin_w3a8.activation import fake_quantize_activation_k64
from rubin_w3a8.scales import activation_scales_k64, weight_scales_k64


def test_zero_scale_is_one() -> None:
    tiles = torch.zeros(2, 8, 64)
    scales = weight_scales_k64(tiles)
    assert torch.equal(scales, torch.ones_like(scales))


def test_weight_and_activation_k64_scales() -> None:
    tiles = torch.zeros(1, 8, 64)
    tiles[0, 3, 10] = 224.0
    assert weight_scales_k64(tiles)[0, 3].item() == 0.5
    activations = torch.zeros(2, 128)
    activations[1, 1] = 448.0
    scales = activation_scales_k64(activations)
    assert scales[1, 0].item() == 1.0
    assert torch.all(scales[0] == 1)
    quantized = fake_quantize_activation_k64(activations.to(torch.bfloat16))
    assert quantized.shape == activations.shape
    assert quantized.dtype == torch.float32
    assert torch.isfinite(quantized).all()
