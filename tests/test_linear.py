import torch

from rubin_w3a8.linear import CachedLUTLinear, reference_linear_lookup
from rubin_w3a8.weight import quantize_weight, reconstruct_weight


def test_reference_matches_dense_reconstruction_and_cached_module() -> None:
    torch.manual_seed(1)
    weight = torch.randn(16, 128, dtype=torch.float32) * 0.05
    bias = torch.randn(16, dtype=torch.float32)
    qweight = quantize_weight(weight)
    x = torch.randn(4, 128, dtype=torch.bfloat16)
    y_lookup = reference_linear_lookup(x, qweight, bias)
    x_hat = __import__("rubin_w3a8").fake_quantize_activation_k64(x)
    y_dense = torch.nn.functional.linear(x_hat, reconstruct_weight(qweight), bias).to(x.dtype)
    y_cached = CachedLUTLinear(qweight, bias)(x)
    torch.testing.assert_close(y_lookup, y_dense, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(y_cached, y_dense, rtol=1e-5, atol=1e-6)
    assert y_cached.dtype == x.dtype
    assert torch.isfinite(y_cached).all()
