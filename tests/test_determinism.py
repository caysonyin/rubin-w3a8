import torch
from torch import nn

from src.linear import CachedLUTLinear
from src.weight import quantize_weight


def test_cached_quantized_forward_is_bit_exact_on_cpu() -> None:
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        torch.manual_seed(42)
        weight = torch.randn(16, 64, dtype=torch.float32) * 0.05
        bias = torch.randn(16, dtype=torch.float32) * 0.01
        model = nn.Sequential(CachedLUTLinear(quantize_weight(weight), bias)).eval()
        inputs = torch.randn(2, 64, dtype=torch.bfloat16)

        with torch.inference_mode():
            output1 = model(inputs)
            output2 = model(inputs)

        torch.testing.assert_close(output1, output2, rtol=0, atol=0)
    finally:
        torch.use_deterministic_algorithms(previous_deterministic)
