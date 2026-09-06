"""Exercise the explicit lookup path against dense reconstruction."""

from __future__ import annotations

import argparse

import torch

from src.activation import fake_quantize_activation_k64
from src.linear import reference_linear_lookup
from src.weight import quantize_weight, reconstruct_weight


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    for m in (1, 4):
        for n in (8, 16, 1024):
            for k in (64, 128, 1024):
                weight = torch.randn(n, k, dtype=torch.float32) * 0.08
                bias = torch.randn(n, dtype=torch.float32) * 0.01
                qweight = quantize_weight(weight)
                x = torch.randn(m, k, dtype=torch.bfloat16)
                y_lookup = reference_linear_lookup(x, qweight, bias)
                x_hat = fake_quantize_activation_k64(x)
                y_dense = torch.nn.functional.linear(x_hat, reconstruct_weight(qweight), bias).to(x.dtype)
                torch.testing.assert_close(y_lookup, y_dense, rtol=1e-5, atol=1e-6)
                assert torch.isfinite(y_lookup).all()
    print("sanity_linear: PASS")


if __name__ == "__main__":
    main()
