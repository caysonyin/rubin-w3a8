"""FP32 signed orthogonal K64 transforms (OSCAR recipe adaptation)."""
import hashlib
import torch


def make_signs(k: int, alias: str, seed: int = 20260813) -> torch.Tensor:
    if k <= 0 or k % 64:
        raise ValueError("Hadamard K must be positive and divisible by 64")
    digest = hashlib.sha256(f"{seed}:{alias}".encode()).digest()
    generator = torch.Generator(device="cpu").manual_seed(
        int.from_bytes(digest[:8], "little") % (2**63 - 1)
    )
    return torch.randint(0, 2, (k // 64, 64), generator=generator, dtype=torch.int8) * 2 - 1


def rotate(x: torch.Tensor, signs: torch.Tensor) -> torch.Tensor:
    if x.shape[-1] % 64 or signs.shape != (x.shape[-1] // 64, 64):
        raise ValueError("Hadamard signs and K64 input dimensions mismatch")
    if not torch.all((signs == 1) | (signs == -1)):
        raise ValueError("Hadamard signs must be +/-1")
    shape = x.shape
    blocks = x.float().reshape(-1, shape[-1] // 64, 64) * signs.float()
    for step in (1, 2, 4, 8, 16, 32):
        paired = blocks.reshape(-1, shape[-1] // 64, 64 // (2 * step), 2, step)
        left, right = paired[..., 0, :], paired[..., 1, :]
        blocks = torch.stack((left + right, left - right), dim=-2).reshape_as(blocks)
    return (blocks / 8).reshape(shape)
