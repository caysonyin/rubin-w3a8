import torch
from torch import nn

from src.linear import CachedLUTLinear
from src.qwen import convert_qwen_to_w3a8


class TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(64, 64)
        self.gate_proj = nn.Linear(64, 128)
        self.other = nn.Linear(64, 64).to(torch.bfloat16)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.other(self.q_proj(x))


def test_only_eligible_linear_names_are_replaced() -> None:
    model = TinyTransformer()
    converted = convert_qwen_to_w3a8(model)
    assert isinstance(converted.q_proj, CachedLUTLinear)
    assert isinstance(converted.gate_proj, CachedLUTLinear)
    assert isinstance(converted.other, nn.Linear)
    x = torch.randn(2, 64, dtype=torch.bfloat16)
    y = converted(x)
    assert y.dtype == torch.bfloat16
    assert torch.isfinite(y).all()
