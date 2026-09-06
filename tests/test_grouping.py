import torch

from src.grouping import (
    from_n8k64_tiles,
    pad_matrix,
    padded_shape,
    tile_valid_mask,
    to_n8k64_tiles,
)


def test_n8_k64_round_trip_and_padding_mask() -> None:
    x = torch.arange(9 * 65, dtype=torch.float32).reshape(9, 65)
    shape = padded_shape(*x.shape)
    assert shape == (16, 128)
    padded = pad_matrix(x, shape)
    tiles = to_n8k64_tiles(padded)
    assert tiles.shape == (4, 8, 64)
    assert torch.equal(from_n8k64_tiles(tiles, shape), padded)
    valid = tile_valid_mask(tuple(x.shape), shape, device=x.device)
    assert int(valid.sum()) == x.numel()
    assert not valid[-1, -1, -1]
