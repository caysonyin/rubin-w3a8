"""Isolated Linear diagnostics using unchanged BF16 teacher forwards."""
from __future__ import annotations

import torch
from torch.nn import functional as F

from scripts.diagnostic_protocol import (
    SCHEMA_VERSION, METHODS, PROJECTIONS, module_coordinates, row_order,
)
from .experiment import eligible, tensor_hash, validate_payload
from .hadamard import rotate, inverse_rotate
from .linear import CachedLUTLinear, _lookup_weight
from .weight import LUTWeight


def squared_error(actual, reference):
    if actual.shape != reference.shape:
        raise ValueError("Error tensors have different shapes")
    a, b = actual.detach().double(), reference.detach().double()
    if not torch.isfinite(a).all() or not torch.isfinite(b).all():
        raise ValueError("Nonfinite diagnostic tensor")
    return float((a - b).square().sum())


def energy(tensor):
    value = tensor.detach().double()
    if not torch.isfinite(value).all():
        raise ValueError("Nonfinite reference energy")
    return float(value.square().sum())


def qweight_from_entry(entry):
    return LUTWeight(**{key: entry[key] for key in (
        "indices", "luts", "scales", "original_shape", "padded_shape")})


def weight_statistics(module, reconstructed, signs):
    original = module.weight.detach().float()
    effective = inverse_rotate(reconstructed, signs) if signs is not None else reconstructed
    if effective.shape != original.shape:
        raise ValueError("Weight statistics must exclude padded entries")
    return dict(in_features=module.in_features, out_features=module.out_features,
                weight_numel=original.numel(), weight_sse=squared_error(effective, original),
                weight_reference_energy=energy(original))


def validate_teacher(model, *, canonical=True):
    modules = eligible(model)
    if model.training or not modules or any(isinstance(m, CachedLUTLinear) for m in model.modules()):
        raise ValueError("Diagnostics require an original unquantized eval teacher")
    if any(m.weight.dtype != torch.bfloat16 or m.weight.device.type != "cpu" for m in modules.values()):
        raise ValueError("Diagnostics require BF16 CPU teacher weights")
    for name in modules:
        module_coordinates(name)
    if canonical:
        coordinates = {(block, projection) for block in range(28) for projection in PROJECTIONS}
        if len(modules) != 196 or {module_coordinates(name) for name in modules} != coordinates:
            raise ValueError("Expected 28 blocks with seven projections each")
    return modules


def collect_recipe(model, payload, tokens, *, block_length=1024, canonical=True):
    """One teacher pass per token block; each local A16/A8 pair shares one reconstruction."""
    modules = validate_teacher(model, canonical=canonical)
    validate_payload(payload, payload["metadata"], model)
    method = payload["metadata"]["weight_method"]
    if method not in METHODS:
        raise ValueError("Unsupported diagnostic recipe")
    if (tokens.ndim != 1 or tokens.dtype != torch.long or tokens.device.type != "cpu"
            or block_length < 1 or tokens.numel() == 0 or tokens.numel() % block_length):
        raise ValueError("Expected complete CPU token blocks")
    if canonical and (block_length != 1024 or tokens.numel() not in (1024, 4096)):
        raise ValueError("Noncanonical token workload")
    records, weights, handles = [], {}, []
    active_block = 0
    observed = set()

    def hook_for(name):
        def hook(module, inputs, output):
            if name in observed:
                raise ValueError(f"Repeated target observation: {name}")
            observed.add(name)
            x = inputs[0].detach()
            y = output.detach()
            if x.dtype != torch.bfloat16 or y.dtype != torch.bfloat16 or x.device.type != "cpu":
                raise ValueError("Teacher input/output must be BF16 CPU tensors")
            if x.numel() // module.in_features != block_length or y.numel() != block_length * module.out_features:
                raise ValueError(f"Unexpected token count or output shape: {name}")
            entry = payload["layers"][name]
            qweight = qweight_from_entry(entry)
            cached = CachedLUTLinear(qweight, module.bias, quantize_activations=False, signs=entry["signs"])
            # This is the only LUT reconstruction in this A16/A8 pair.
            y16 = cached(x)
            cached.quantize_activations = True
            y8 = cached(x)
            if (name, method) not in weights:
                weights[name, method] = weight_statistics(module, cached.weight_hat, entry["signs"])
            bias = module.bias.detach().float() if module.bias is not None else None
            control = F.linear(x.float(), module.weight.detach().float(), bias).to(x.dtype)
            block, projection = module_coordinates(name)
            row = dict(schema_version=SCHEMA_VERSION, module_name=name, transformer_block=block,
                       projection=projection, weight_method=method, payload_fingerprint=payload["fingerprint"],
                       token_block_index=active_block, num_tokens=block_length, output_numel=y.numel(),
                       teacher_input_hash=tensor_hash(x), teacher_output_hash=tensor_hash(y),
                       teacher_output_energy=energy(y), sse_a16=squared_error(y16, y),
                       sse_a8=squared_error(y8, y), sse_a8_vs_a16=squared_error(y8, y16),
                       sse_fp32_control=squared_error(control, y), fp32_control_output_hash=tensor_hash(control),
                       sse_h64_control=None, h64_control_output_hash=None)
            if method == METHODS[1]:
                rotated_control = F.linear(rotate(x, entry["signs"]),
                                           rotate(module.weight.detach(), entry["signs"]), bias).to(x.dtype)
                row.update(sse_h64_control=squared_error(rotated_control, y),
                           h64_control_output_hash=tensor_hash(rotated_control))
            records.append(row)
            # Returning None leaves the native BF16 teacher output unchanged.
        return hook

    try:
        for name, module in modules.items():
            handles.append(module.register_forward_hook(hook_for(name)))
        with torch.inference_mode():
            for active_block, start in enumerate(range(0, tokens.numel(), block_length)):
                observed.clear()
                result = model(input_ids=tokens[start:start + block_length].unsqueeze(0), use_cache=False)
                if observed != set(modules):
                    raise ValueError("Missing teacher target observations")
                logits = getattr(result, "logits", result if isinstance(result, torch.Tensor) else None)
                if logits is not None and not torch.isfinite(logits).all():
                    raise ValueError("Nonfinite teacher final output")
                del result
                print(f"diagnostics recipe={method} block={active_block + 1}/{tokens.numel() // block_length}", flush=True)
    finally:
        for handle in handles:
            handle.remove()
    records.sort(key=lambda row: row_order("block_stats.csv", row))
    return records, weights


def independent_weight_statistics(model, payload, *, canonical=True):
    """Recompute weight errors directly from the frozen payload, without a forward or Cached module."""
    modules = validate_teacher(model, canonical=canonical)
    validate_payload(payload, payload["metadata"], model)
    method = payload["metadata"]["weight_method"]
    result = {}
    with torch.inference_mode():
        for name, module in modules.items():
            entry = payload["layers"][name]
            # Separate index->LUT lookup implementation; do not reuse recorded weight scalars.
            reconstructed = _lookup_weight(qweight_from_entry(entry))
            result[name, method] = weight_statistics(module, reconstructed, entry["signs"])
    return result
