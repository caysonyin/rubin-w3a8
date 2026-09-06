import pytest
import torch

from rubin_w3a8.e4m3 import e4m3_values, quantize_e4m3


def test_grid_and_saturation() -> None:
    values = e4m3_values()
    assert values[0].item() == 0.0
    assert values[-1].item() == 448.0
    assert torch.all(values[1:] > values[:-1])
    x = torch.tensor([-1000.0, -448.0, -1.0, 0.0, 1.0, 448.0, 1000.0], dtype=torch.float32)
    expected = torch.tensor([-448.0, -448.0, -1.0, 0.0, 1.0, 448.0, 448.0])
    torch.testing.assert_close(quantize_e4m3(x), expected, rtol=0, atol=0)


def test_nan_and_bfloat16_return_fp32() -> None:
    x = torch.tensor([float("nan"), 0.125], dtype=torch.bfloat16)
    y = quantize_e4m3(x)
    assert y.dtype == torch.float32
    assert torch.isnan(y[0])
    assert y[1].item() == 0.125


def test_matches_torch_float8_when_available() -> None:
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("PyTorch build has no float8_e4m3fn")
    x = torch.linspace(-448.0, 448.0, 4097, dtype=torch.float32)
    try:
        expected = x.to(torch.float8_e4m3fn).to(torch.float32)
    except (RuntimeError, NotImplementedError):
        pytest.skip("CPU build does not support FP8 cast")
    torch.testing.assert_close(quantize_e4m3(x), expected, rtol=0, atol=0)
