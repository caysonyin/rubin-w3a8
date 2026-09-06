import torch

from rubin_w3a8.weight import quantize_weight, reconstruct_weight


def test_weight_quantize_reconstruct_non_multiple_shape() -> None:
    torch.manual_seed(0)
    weight = torch.randn(9, 65, dtype=torch.float32) * 0.1
    qweight = quantize_weight(weight, tile_chunk_size=2)
    assert qweight.indices.dtype == torch.uint8
    assert qweight.indices.shape == (16, 128)
    assert qweight.luts.shape == (2, 2, 8)
    assert qweight.scales.shape == (2, 2, 8)
    reconstructed = reconstruct_weight(qweight)
    assert reconstructed.shape == weight.shape
    assert torch.isfinite(reconstructed).all()
    assert torch.max(reconstructed.abs()) <= torch.max(weight.abs()) * 1.1


def test_zero_weight_is_finite_and_zero() -> None:
    weight = torch.zeros(8, 64)
    qweight = quantize_weight(weight)
    assert torch.equal(reconstruct_weight(qweight), weight)
