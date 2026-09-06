import torch

from src.lloyd_max import fit_lut_tiles


def test_lut_fit_shapes_validity_and_zero_tile() -> None:
    tiles = torch.zeros(3, 8, 64, dtype=torch.float32)
    tiles[0] = torch.linspace(-1, 1, 8 * 64).reshape(8, 64)
    valid = torch.ones_like(tiles, dtype=torch.bool)
    valid[2, 2:] = False
    luts, indices = fit_lut_tiles(tiles, valid, tile_chunk_size=2)
    assert luts.shape == (3, 8)
    assert indices.shape == (3, 8, 64)
    assert indices.dtype == torch.uint8
    assert torch.isfinite(luts).all()
    assert torch.equal(luts[1], torch.zeros(8))
    assert torch.equal(indices[1], torch.zeros((8, 64), dtype=torch.uint8))
    assert torch.all((indices <= 7))
