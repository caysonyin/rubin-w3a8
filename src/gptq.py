"""K64 block GPTQ and Hessian LUT refitting with fixed project scales.

Adapted algorithmically from FutureMLS-Lab/OSCAR, commit
658d6539153530fd63a7bd8017bec619f6604539, rubin/w3a8_lut_sim.py.
No hardware kernels, UE8M0 scaling, or source runtime dependencies.
"""
import torch
import torch.nn.functional as F
from .e4m3 import quantize_e4m3
from .grouping import pad_matrix, to_n8k64_tiles, from_n8k64_tiles
from .weight import LUTWeight, quantize_weight


def inverse_cholesky(hessian: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    h = hessian.float().clone()
    diagonal = h.diagonal(dim1=-2, dim2=-1)
    live = diagonal > 1e-12
    mean = ((diagonal * live).sum(-1) / live.sum(-1).clamp_min(1)).clamp_min(1e-8)
    diagonal.add_(0.01 * mean[:, None])
    diagonal.masked_fill_(~live, 1)
    eye = torch.eye(64)
    for jitter in (0., 1e-6, 1e-4, 1e-2):
        try:
            chol = torch.linalg.cholesky(h + jitter * eye)
            return torch.linalg.cholesky(torch.cholesky_inverse(chol), upper=True), ~live
        except torch.linalg.LinAlgError:
            pass
    raise ValueError("GPTQ Hessian Cholesky failed after damping and jitter")


def sweep(tiles, centers, scales, upper, dead):
    """Tiles [Ntiles,Ktiles,8,64]; preserve sequential GPTQ assignments."""
    work = tiles.clone().masked_fill(dead[None, :, None, :], 0)
    indices = torch.empty_like(work, dtype=torch.uint8)
    quantized = torch.empty_like(work)
    for col in range(64):
        values = work[..., col]
        normalized = (values / scales).clamp(-448, 448)
        idx = (normalized[..., None] - centers[:, :, None, :]).abs().argmin(-1)
        restored = centers.gather(-1, idx) * scales
        indices[..., col] = idx.to(torch.uint8)
        quantized[..., col] = restored
        error = (values - restored) / upper[None, :, None, col, col]
        work[..., col:] -= error[..., None] * upper[None, :, None, col, col:]
    return quantized, indices


def objective(original, quantized, hessian, valid_rows=None):
    error = original - quantized
    if valid_rows is not None:
        error = error * valid_rows
    return float(torch.einsum('akni,kij,aknj->', error, hessian, error))


def refit(original, indices, centers, scales, hessian, valid_rows, chunk_size=128):
    nt, kt = original.shape[:2]
    flat = original.reshape(-1, 8, 64)
    idx = indices.reshape(-1, 8, 64).long()
    old = centers.reshape(-1, 8)
    scale = scales.reshape(-1, 8)
    valid = valid_rows.expand_as(original).reshape(-1, 8, 64)
    result = torch.empty_like(old)
    for start in range(0, len(flat), chunk_size):
        stop = min(start + chunk_size, len(flat))
        onehot = F.one_hot(idx[start:stop], 8).float() * valid[start:stop, ..., None]
        design = onehot * scale[start:stop, :, None, None]
        h = hessian[torch.arange(start, stop) % kt]
        normal = torch.einsum('cnka,ckl,cnlb->cab', design, h, design)
        target = torch.einsum('cnka,ckl,cnl->ca', design, h, flat[start:stop])
        ridge = 1e-4 * normal.diagonal(dim1=-2, dim2=-1).mean(-1).clamp_min(1e-8)
        regularized = normal + ridge[:, None, None] * torch.eye(8)
        try:
            proposal = torch.linalg.solve(regularized, target[..., None]).squeeze(-1)
        except torch.linalg.LinAlgError:
            proposal = (torch.linalg.pinv(regularized) @ target[..., None]).squeeze(-1)
        proposal = torch.where(onehot.sum((1, 2)) > 0, proposal, old[start:stop])
        result[start:stop] = quantize_e4m3(proposal).sort(-1).values
    return result.reshape(nt, kt, 8)


def quantize_gptq(weight: torch.Tensor, hessian: torch.Tensor) -> tuple[LUTWeight, dict]:
    if weight.ndim != 2 or min(weight.shape) <= 0 or weight.shape[1] % 64:
        raise ValueError("GPTQ requires [N,K] with K divisible by 64")
    ktiles = weight.shape[1] // 64
    if hessian.shape != (ktiles, 64, 64) or not torch.isfinite(hessian).all():
        raise ValueError("Missing, nonfinite or mismatched Hessian")
    if not torch.allclose(hessian, hessian.transpose(-1, -2), rtol=1e-5, atol=1e-6):
        raise ValueError("Hessian must be symmetric")
    if not torch.isfinite(weight).all() or weight.device.type != 'cpu' or hessian.device.type != 'cpu':
        raise ValueError("Finite CPU tensors required")
    hessian = hessian.float()
    if torch.any(hessian.diagonal(dim1=-2, dim2=-1) < 0):
        raise ValueError("Hessian diagonal must be nonnegative")
    base = quantize_weight(weight)
    nt = base.padded_shape[0] // 8
    tiles = to_n8k64_tiles(pad_matrix(weight.float(), base.padded_shape)).reshape(nt, ktiles, 8, 64)
    valid = (torch.arange(base.padded_shape[0]).reshape(nt, 1, 8, 1) < weight.shape[0]).float()
    upper, dead = inverse_cholesky(hessian)
    centers, scales = base.luts, base.scales
    quantized, indices = sweep(tiles, centers, scales, upper, dead)
    current = objective(tiles, quantized, hessian, valid)
    signal = max(objective(tiles, torch.zeros_like(tiles), hessian, valid), 1e-30)
    trace = [current / signal]
    for _ in range(3):
        proposal = refit(tiles, indices, centers, scales, hessian, valid)
        best = (current, centers, quantized, indices)
        for alpha in (1., .5, .25):
            candidate = quantize_e4m3(centers + alpha * (proposal - centers)).sort(-1).values
            if torch.equal(candidate, centers):
                continue
            q, ix = sweep(tiles, candidate, scales, upper, dead)
            loss = objective(tiles, q, hessian, valid)
            if loss < best[0]:
                best = (loss, candidate, q, ix)
        improvement = (current - best[0]) / max(abs(current), 1e-30)
        if improvement <= 1e-5:
            break
        current, centers, quantized, indices = best
        trace.append(current / signal)
    payload = LUTWeight(from_n8k64_tiles(indices.reshape(-1, 8, 64), base.padded_shape),
                        centers.contiguous(), scales, base.original_shape, base.padded_shape)
    return payload, {'proxy_objective_trace': trace, 'accepted_iterations': len(trace) - 1,
                     'dead_channels': int(dead.sum())}
