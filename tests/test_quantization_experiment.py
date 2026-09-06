import copy
import torch
import pytest
from torch import nn
from src.hadamard import rotate, make_signs
from src.gptq import sweep, inverse_cholesky, quantize_gptq, refit
from src.weight import quantize_weight, reconstruct_weight
from src.linear import CachedLUTLinear, reference_linear_lookup
from src.experiment import (alias_for, collect_hessians, calibration_metadata, build_payload,
                            apply_payload, tensor_hash, payload_hash, FORMAT, CONFIG)


def test_hadamard_orientation_and_aliases():
    torch.manual_seed(3)
    signs = make_signs(128, 'layer')
    assert torch.equal(signs, make_signs(128, 'layer'))
    r = rotate(torch.eye(128), signs)
    torch.testing.assert_close(r.T @ r, torch.eye(128), rtol=1e-5, atol=1e-5)
    x, w = torch.randn(3, 128) * .1, torch.randn(9, 128) * .1
    torch.testing.assert_close(rotate(x, signs) @ rotate(w, signs).T, x @ w.T, rtol=1e-5, atol=1e-5)
    assert alias_for('a.self_attn.k_proj') == alias_for('a.self_attn.q_proj')
    assert alias_for('a.mlp.up_proj') == alias_for('a.mlp.gate_proj')
    with pytest.raises(ValueError): make_signs(65, 'x')


def test_gptq_diagonal_and_scalar_oracle():
    torch.manual_seed(4)
    tiles = torch.randn(1, 1, 8, 64)
    centers = torch.tensor([[[-3., -2., -1., 0., 1., 2., 3., 4.]]])
    scales = torch.ones(1, 1, 8)
    h = torch.eye(64).unsqueeze(0)
    upper, dead = inverse_cholesky(h)
    q, ix = sweep(tiles, centers, scales, upper, dead)
    nearest = (tiles[..., None] - centers[:, :, None, None]).abs().argmin(-1)
    assert torch.equal(ix.long(), nearest)
    a = torch.randn(64, 64)
    upper, dead = inverse_cholesky((a @ a.T + torch.eye(64)).unsqueeze(0))
    q, ix = sweep(tiles, centers, scales, upper, dead)
    # Independent per-row scalar compensation oracle.
    oracle = tiles.clone()
    expected = torch.empty_like(ix)
    for row in range(8):
        for col in range(64):
            value = float(oracle[0, 0, row, col])
            index = min(range(8), key=lambda i: abs(value - float(centers[0, 0, i])))
            expected[0, 0, row, col] = index
            error = (value - float(centers[0, 0, index])) / float(upper[0, col, col])
            for target in range(col, 64):
                oracle[0, 0, row, target] -= error * upper[0, col, target]
    assert torch.equal(ix, expected)


def test_refit_diagonal_solution_and_empty_codes():
    # Only center 0 is occupied; ridge-adjusted closed form is known.
    original = torch.full((1, 1, 8, 64), 2.)
    idx = torch.zeros_like(original, dtype=torch.uint8)
    centers = torch.arange(8).float().reshape(1, 1, 8)
    new = refit(original, idx, centers, torch.ones(1, 1, 8), torch.eye(64)[None], torch.ones(1, 1, 8, 1))
    assert new.flatten().tolist() == [1., 2., 2., 3., 4., 5., 6., 7.]


@pytest.mark.parametrize('zero', [False, True])
def test_payload_reconstruction_padding_refit_and_rotation(zero):
    torch.manual_seed(8)
    w = torch.zeros(9, 64) if zero else torch.randn(9, 64) * .05
    signs = make_signs(64, 'q_proj')
    x = torch.randn(32, 64)
    xr = rotate(x, signs)
    h = (xr.T @ xr / 32)[None]  # singular calibration with damping
    if zero: h.zero_()
    qw, trace = quantize_gptq(rotate(w, signs), h)
    assert all(b < a for a, b in zip(trace['proxy_objective_trace'], trace['proxy_objective_trace'][1:]))
    assert torch.isfinite(reconstruct_weight(qw)).all()
    if zero: assert torch.count_nonzero(reconstruct_weight(qw)) == 0
    for a8 in (False, True):
        cached = CachedLUTLinear(qw, signs=signs, quantize_activations=a8)
        expected = reference_linear_lookup(x, qw, signs=signs, quantize_activations=a8)
        torch.testing.assert_close(cached(x), expected, rtol=0, atol=0)
    # Index/LUT payload directly reconstructs the final sweep, without nearest recoding.
    from src.grouping import to_n8k64_tiles, pad_matrix
    tiles = to_n8k64_tiles(pad_matrix(rotate(w, signs), qw.padded_shape)).reshape(2, 1, 8, 64)
    upper, dead = inverse_cholesky(h)
    final_q, final_ix = sweep(tiles, qw.luts, qw.scales, upper, dead)
    assert torch.equal(to_n8k64_tiles(qw.indices).reshape_as(final_ix), final_ix)
    torch.testing.assert_close(reconstruct_weight(qw), final_q.reshape(16, 64)[:9], rtol=0, atol=0)


class Mini(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(8, 64)
        self.q_proj = nn.Linear(64, 8)
        self.k_proj = nn.Linear(64, 8)
    def forward(self, input_ids, use_cache=False):
        x = self.emb(input_ids)
        return self.q_proj(x) + self.k_proj(x)


@pytest.mark.parametrize('method', ['plain_lloyd_max', 'h64_gptq_refit'])
def test_calibration_and_serialization(tmp_path, method):
    torch.manual_seed(42)
    model = Mini().bfloat16().eval()
    tokens = torch.arange(512) % 8
    hs, signs = collect_hessians(model, tokens)
    assert hs['q_proj'] is hs['k_proj']
    xr = rotate(model.emb(tokens), signs['q_proj'])
    torch.testing.assert_close(hs['q_proj'][0], xr.T @ xr / 512)
    assert not model.q_proj._forward_pre_hooks
    meta = dict(weight_method=method, calibration=calibration_metadata(tokens) if method != 'plain_lloyd_max' else None, format=FORMAT, config=CONFIG)
    payload = build_payload(model, meta, tokens)
    if method == 'plain_lloyd_max':
        old = quantize_weight(model.q_proj.weight.float())
        for key in ('indices', 'luts', 'scales'):
            assert torch.equal(payload['layers']['q_proj'][key], getattr(old, key))
    path = tmp_path / 'weights.pt'
    torch.save(payload, path)
    loaded = torch.load(path, weights_only=True)
    assert payload_hash(loaded) == payload['fingerprint']
    for a8 in (False, True):
        converted = apply_payload(copy.deepcopy(model), loaded, meta, quantize_activations=a8)
        y = converted(tokens[None])
        assert torch.isfinite(y).all()
        assert tensor_hash(converted.q_proj.indices) == tensor_hash(payload['layers']['q_proj']['indices'])
    with pytest.raises(ValueError): apply_payload(copy.deepcopy(model), loaded, {**meta, 'extra': 1}, quantize_activations=False)
    broken = copy.deepcopy(loaded)
    broken['layers']['q_proj']['indices'][0, 0] = 9
    with pytest.raises(ValueError): apply_payload(model, broken, meta, quantize_activations=False)


def test_invalid_inputs_and_hook_cleanup():
    with pytest.raises(ValueError): calibration_metadata(torch.arange(512), split='test')
    with pytest.raises(ValueError): calibration_metadata(torch.arange(511))
    with pytest.raises(ValueError): quantize_gptq(torch.randn(8, 64), torch.eye(32)[None])
    with pytest.raises(ValueError): quantize_gptq(torch.randn(8, 65), torch.eye(64)[None])
    indefinite = torch.eye(64)[None]
    indefinite[0, 0, 1] = indefinite[0, 1, 0] = 100
    with pytest.raises(ValueError): inverse_cholesky(indefinite)
    model = Mini()
    with pytest.raises(IndexError): collect_hessians(model, torch.full((512,), 99))
    assert not model.q_proj._forward_pre_hooks


def test_optimized_determinism_and_invalid_configuration():
    torch.manual_seed(31)
    weight = torch.randn(8, 128) * .02
    x = torch.randn(512, 2, 64)
    h = torch.einsum('tki,tkj->kij', x, x) / 512
    first, trace1 = quantize_gptq(weight, h)
    second, trace2 = quantize_gptq(weight, h)
    for key in ('indices', 'luts', 'scales'):
        assert torch.equal(getattr(first, key), getattr(second, key))
    assert trace1 == trace2
    with pytest.raises(ValueError): build_payload(Mini(), dict(weight_method='h64_gptq_refit', config={}, format=FORMAT))


def test_refit_against_independent_normal_equations():
    from src.e4m3 import quantize_e4m3
    torch.manual_seed(91)
    w = torch.randn(1, 1, 8, 64)
    ix = torch.randint(0, 8, w.shape, dtype=torch.uint8)
    centers = torch.arange(8).float().reshape(1, 1, 8)
    scales = torch.rand(1, 1, 8) + .5
    a = torch.randn(64, 64)
    h = (a @ a.T)[None]
    normal, target = torch.zeros(8, 8), torch.zeros(8)
    for row in range(8):
        design = torch.zeros(64, 8)
        for col in range(64):
            design[col, int(ix[0, 0, row, col])] = scales[0, 0, row]
        normal += design.T @ h[0] @ design
        target += design.T @ h[0] @ w[0, 0, row]
    ridge = 1e-4 * normal.diag().mean().clamp_min(1e-8)
    expected = quantize_e4m3(torch.linalg.solve(normal + ridge * torch.eye(8), target)).sort().values
    actual = refit(w, ix, centers, scales, h, torch.ones(1, 1, 8, 1))
    torch.testing.assert_close(actual.flatten(), expected, rtol=0, atol=0)
