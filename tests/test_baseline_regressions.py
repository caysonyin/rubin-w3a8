"""CPU numerical-contract checks without models or calibration data."""

import math

import pytest
import torch

import src.activation as activation
import src.lloyd_max as lloyd
from src.e4m3 import quantize_e4m3
from src.linear import CachedLUTLinear, reference_linear_lookup
from src.weight import quantize_weight, reconstruct_weight


def _positive_grid() -> list[float]:
    # Decode finite byte patterns independently of the emulator's grid builder.
    return [
        (code & 7) * 2.0**-9
        if code >> 3 == 0
        else (1 + (code & 7) / 8) * 2.0 ** ((code >> 3) - 7)
        for code in range(127)
    ]


def _oracle(x: torch.Tensor) -> torch.Tensor:
    grid = _positive_grid()

    def convert(value: float) -> float:
        if math.isnan(value):
            return float("nan")
        magnitude = min(abs(value), 448.0)
        # At a midpoint, the even least-significant mantissa bit wins.
        code = min(range(len(grid)), key=lambda c: (abs(magnitude - grid[c]), c & 1))
        return math.copysign(grid[code], value)

    return torch.tensor(
        [convert(value) for value in x.float().reshape(-1).tolist()],
        dtype=torch.float32,
    ).reshape(x.shape)


def _assert_exact(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    zero = expected == 0
    assert torch.equal(torch.signbit(actual[zero]), torch.signbit(expected[zero]))


def _boundary_inputs() -> torch.Tensor:
    grid = torch.tensor(_positive_grid())
    midpoints = (grid[:-1] + grid[1:]) / 2
    below = torch.nextafter(midpoints, torch.full_like(midpoints, -float("inf")))
    above = torch.nextafter(midpoints, torch.full_like(midpoints, float("inf")))
    positive = torch.cat([grid, below, midpoints, above])
    special = torch.tensor(
        [500.0, -500.0, 1e30, -1e30, float("inf"), -float("inf"), float("nan")]
    )
    return torch.cat([positive, -positive, special])


def _native_quantize(x: torch.Tensor) -> torch.Tensor:
    return x.float().clamp(-448.0, 448.0).to(torch.float8_e4m3fn).float()


@pytest.fixture
def native_quantizer():
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("PyTorch has no float8_e4m3fn; independent oracle checks remain active")
    try:
        _native_quantize(torch.tensor([0.0]))
    except (RuntimeError, NotImplementedError):
        pytest.skip("PyTorch CPU build does not support FP8 casts")
    return _native_quantize


def test_known_representable_values() -> None:
    x = torch.tensor([0.5, 1.0, 1.5, 2.0, -1.0, 2.0**-6])
    _assert_exact(quantize_e4m3(x), x)


def test_all_finite_codes() -> None:
    positive = torch.tensor(_positive_grid())
    x = torch.cat([positive, -positive])
    _assert_exact(quantize_e4m3(x), x)


def test_rounding_midpoints_neighbors_and_special_values() -> None:
    x = _boundary_inputs()
    _assert_exact(quantize_e4m3(x), _oracle(x))


def test_boundary_inputs_match_native_fp8(native_quantizer) -> None:
    x = _boundary_inputs()
    _assert_exact(quantize_e4m3(x), native_quantizer(x))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_different_magnitudes_and_noncontiguous_input(dtype) -> None:
    generator = torch.Generator().manual_seed(20261008)
    x = torch.randn(8, 128, generator=generator)
    x *= torch.tensor([2.0**e for e in [-12, -9, -6, -3, 0, 3, 6, 10]])[:, None]
    x = x.to(dtype).t()
    result = quantize_e4m3(x)
    assert result.dtype == torch.float32
    assert result.device == x.device and result.shape == x.shape
    _assert_exact(result, _oracle(x))


def test_scalar_and_empty_tensor() -> None:
    scalar = torch.tensor(-0.0001)
    _assert_exact(quantize_e4m3(scalar), _oracle(scalar))
    empty = torch.empty(0, 3)
    assert quantize_e4m3(empty).shape == empty.shape


@pytest.mark.parametrize(
    "n,k", [(8, 64), (9, 64), (8, 65), (9, 65), (1, 1), (3, 17), (10, 100)]
)
def test_padding_shapes(n: int, k: int) -> None:
    weight = torch.linspace(-1.0, 1.0, n * k).reshape(n, k)
    qweight = quantize_weight(weight)
    restored = reconstruct_weight(qweight)
    assert restored.shape == weight.shape
    assert torch.isfinite(restored).all() and torch.isfinite(qweight.luts).all()
    assert qweight.indices.dtype == torch.uint8
    assert int(qweight.indices.min()) >= 0 and int(qweight.indices.max()) <= 7
    positive = torch.tensor(_positive_grid())
    assert torch.isin(qweight.luts, torch.cat([positive, -positive])).all()


@pytest.mark.parametrize("chunk_size", [1, 4])
def test_variable_valid_counts_and_mask_exclusion(chunk_size: int) -> None:
    tiles = torch.linspace(-300.0, 300.0, 4 * 512).reshape(4, 8, 64)
    mask = torch.zeros_like(tiles, dtype=torch.bool)
    for i, count in enumerate([512, 65, 1, 0]):
        mask[i].reshape(-1)[:count] = True
    changed = tiles.clone()
    changed[~mask] = torch.linspace(500.0, 1500.0, int((~mask).sum()))
    lut1, idx1 = lloyd.fit_lut_tiles(tiles, mask, tile_chunk_size=chunk_size)
    lut2, idx2 = lloyd.fit_lut_tiles(changed, mask, tile_chunk_size=chunk_size)
    _assert_exact(lut1, lut2)
    assert torch.equal(idx1[mask], idx2[mask])
    assert torch.isfinite(lut1).all()
    assert torch.equal(lut1[-1], torch.zeros(8))
    assert torch.equal(idx1[-1], torch.zeros(8, 64, dtype=torch.uint8))


def test_zero_and_duplicate_centroids() -> None:
    tiles = torch.stack([torch.zeros(8, 64), torch.full((8, 64), 4.0)])
    lut, idx = lloyd.fit_lut_tiles(tiles, torch.ones_like(tiles, dtype=torch.bool))
    _assert_exact(lut[0], torch.zeros(8))
    _assert_exact(lut[1], torch.full((8,), 4.0))
    assert torch.equal(idx, torch.zeros_like(idx))


def test_empty_clusters_retain_centroids(monkeypatch) -> None:
    initial = torch.tensor([[-8.0, -4.0, -2.0, -1.0, 1.0, 2.0, 4.0, 8.0]])
    monkeypatch.setattr(lloyd, "_quantile_initialization", lambda values, valid: initial)
    tiles = torch.zeros(1, 8, 64)
    lut, idx = lloyd.fit_lut_tiles(
        tiles, torch.ones_like(tiles, dtype=torch.bool), max_iters=1
    )
    expected = initial.clone()
    expected[0, 3] = 0.0
    _assert_exact(lut, expected)
    assert torch.equal(idx, torch.full_like(idx, 3))


@pytest.mark.parametrize("cap", [1, 50])
def test_returned_indices_match_final_lut(cap: int) -> None:
    tiles = torch.randn(2, 8, 64, generator=torch.Generator().manual_seed(42)) * 75
    lut, idx = lloyd.fit_lut_tiles(
        tiles, torch.ones_like(tiles, dtype=torch.bool), max_iters=cap
    )
    distances = (tiles.reshape(2, 512, 1) - lut[:, None, :]).square()
    nearest = distances.argmin(dim=-1).reshape_as(idx).to(torch.uint8)
    assert torch.equal(idx, nearest)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("zero", [False, True])
def test_quantization_and_linear_native_equivalence(
    dtype, zero: bool, monkeypatch, native_quantizer
) -> None:
    weight = torch.linspace(-1.0, 1.0, 9 * 65).reshape(9, 65)
    x = torch.linspace(-2.0, 2.0, 3 * 65).reshape(3, 65).to(dtype)
    bias = torch.linspace(-0.1, 0.1, 9)
    if zero:
        weight.zero_()
        x.zero_()
    observations = []
    # Native casts are a test oracle, not a production execution backend.
    for quantizer in [quantize_e4m3, native_quantizer]:
        monkeypatch.setattr(lloyd, "quantize_e4m3", quantizer)
        monkeypatch.setattr(activation, "quantize_e4m3", quantizer)
        qweight = quantize_weight(weight)
        lookup = reference_linear_lookup(x, qweight, bias)
        cached = CachedLUTLinear(qweight, bias)(x)
        _assert_exact(lookup, cached)
        observations.append(
            (qweight.scales, qweight.luts, qweight.indices, reconstruct_weight(qweight),
             activation.fake_quantize_activation_k64(x), lookup, cached)
        )
    for software, native in zip(*observations):
        _assert_exact(software, native)
