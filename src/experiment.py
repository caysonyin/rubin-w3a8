"""Offline calibration, reproducible quantization payloads and conversion."""
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch
from torch import nn
from .hadamard import make_signs, rotate
from .gptq import quantize_gptq
from .linear import CachedLUTLinear
from .qwen import _ELIGIBLE_LINEAR_NAMES
from .weight import LUTWeight, quantize_weight

METHODS = ('plain_lloyd_max', 'h64_gptq_refit')
FORMAT = {'tile': [8, 64], 'lut': 'e4m3', 'scale': 'fp32_max_abs_div448_k64',
          'activation': 'existing_fp32_scale_e4m3_k64', 'compute': 'fp32_output_input_dtype'}
CONFIG = {'version': 1, 'hadamard_seed': 20260813, 'lm_max_iters': 50,
          'gptq_damp': .01, 'refit_damp': 1e-4, 'refit_iters': 3,
          'refit_tolerance': 1e-5, 'refit_alphas': [1., .5, .25],
          'source_commit': '658d6539153530fd63a7bd8017bec619f6604539'}


def tensor_hash(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def files_hash(paths):
    digest = hashlib.sha256()
    for path in sorted(map(Path, paths)):
        digest.update(path.name.encode())
        with path.open('rb') as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b''):
                digest.update(block)
    return digest.hexdigest()


def eligible(model):
    return {name: module for name, module in model.named_modules()
            if name.rsplit('.', 1)[-1] in _ELIGIBLE_LINEAR_NAMES and isinstance(module, nn.Linear)}


def alias_for(name):
    parent, _, leaf = name.rpartition('.')
    representative = 'q_proj' if leaf in ('q_proj', 'k_proj', 'v_proj') else 'gate_proj' if leaf in ('gate_proj', 'up_proj') else leaf
    return f'{parent}.{representative}' if parent else representative


def calibration_metadata(tokens, *, split='train'):
    if split != 'train' or tokens.ndim != 1 or tokens.numel() != 512:
        raise ValueError('Calibration requires exactly 512 train tokens')
    return {'dataset': 'Salesforce/wikitext', 'config': 'wikitext-2-raw-v1',
            'split': split, 'tokens': 512, 'token_hash': tensor_hash(tokens),
            'source': 'unquantized_bf16_inputs_fp32_rotation'}


def collect_hessians(model, tokens):
    calibration_metadata(tokens)
    modules = eligible(model)
    if not modules or any(isinstance(m, CachedLUTLinear) for m in model.modules()):
        raise ValueError('Calibration requires an unquantized model')
    sums, counts, handles = {}, {}, []
    representatives = {}
    signs = {}
    for name, module in modules.items():
        alias = alias_for(name)
        signs[name] = make_signs(module.in_features, alias)
        representatives.setdefault(alias, name)
    def hook_for(alias, sign):
        def hook(module, inputs):
            x = rotate(inputs[0].detach(), sign).reshape(-1, sign.shape[0], 64)
            gram = torch.einsum('tki,tkj->kij', x, x)
            sums[alias] = sums.get(alias, torch.zeros_like(gram)) + gram
            counts[alias] = counts.get(alias, 0) + len(x)
        return hook
    try:
        for alias, name in representatives.items():
            handles.append(modules[name].register_forward_pre_hook(hook_for(alias, signs[name])))
        with torch.inference_mode():
            model(input_ids=tokens.unsqueeze(0), use_cache=False)
    finally:
        for handle in handles:
            handle.remove()
    if set(sums) != set(representatives):
        raise ValueError('Missing calibration activations')
    normalized = {alias: h / counts[alias] for alias, h in sums.items()}
    return {name: normalized[alias_for(name)] for name in modules}, signs


def payload_hash(payload):
    digest = hashlib.sha256(json.dumps(payload['metadata'], sort_keys=True).encode())
    for name, entry in sorted(payload['layers'].items()):
        digest.update(name.encode())
        for key, value in sorted(entry.items()):
            digest.update(key.encode())
            digest.update((tensor_hash(value) if isinstance(value, torch.Tensor) else json.dumps(value, sort_keys=True)).encode())
    return digest.hexdigest()


def validate_metadata(metadata):
    if metadata.get('weight_method') not in METHODS or metadata.get('format') != FORMAT or metadata.get('config') != CONFIG:
        raise ValueError('Unknown or incompatible quantization configuration')
    if metadata['weight_method'] == METHODS[0] and metadata.get('calibration') is not None:
        raise ValueError('Plain quantization does not use calibration')


def build_payload(model, metadata, tokens=None):
    validate_metadata(metadata)
    method = metadata['weight_method']
    if method not in METHODS:
        raise ValueError('Unknown weight method')
    if method == METHODS[1]:
        if tokens is None or metadata['calibration'] != calibration_metadata(tokens):
            raise ValueError('Calibration metadata mismatch')
        hessians, signs = collect_hessians(model, tokens)
    else:
        hessians, signs = {}, {}
    layers, traces = {}, {}
    modules = eligible(model)
    if not modules:
        raise ValueError('No eligible Linear modules')
    with torch.inference_mode():
        for index, (name, module) in enumerate(modules.items()):
            weight = module.weight.detach().float()
            if method == METHODS[1]:
                qweight, trace = quantize_gptq(rotate(weight, signs[name]), hessians[name])
                traces[name] = trace
            else:
                qweight = quantize_weight(weight)
            layers[name] = {**asdict(qweight), 'signs': signs.get(name)}
            print(f'quantized={index + 1}/{len(modules)} module={name}', flush=True)
    payload = {'metadata': metadata, 'layers': layers, 'traces': traces}
    payload['fingerprint'] = payload_hash(payload)
    return payload


def validate_payload(payload, metadata, model):
    validate_metadata(metadata)
    if payload['metadata'] != metadata or payload['fingerprint'] != payload_hash(payload):
        raise ValueError('Payload metadata or fingerprint mismatch')
    modules = eligible(model)
    if set(payload['layers']) != set(modules):
        raise ValueError('Payload layer set mismatch')
    for name, entry in payload['layers'].items():
        n, k = modules[name].weight.shape
        npad, kpad = ((n + 7) // 8 * 8, (k + 63) // 64 * 64)
        if tuple(entry['original_shape']) != (n, k) or tuple(entry['padded_shape']) != (npad, kpad):
            raise ValueError('Payload shape mismatch')
        ix, lut, scale = entry['indices'], entry['luts'], entry['scales']
        if ix.dtype != torch.uint8 or ix.shape != (npad, kpad) or torch.any(ix > 7):
            raise ValueError('Invalid payload indices')
        if lut.dtype != torch.float32 or scale.dtype != torch.float32 or lut.shape != (npad // 8, kpad // 64, 8) or scale.shape != lut.shape:
            raise ValueError('Invalid payload LUT/scale shape')
        from .e4m3 import quantize_e4m3
        if not torch.isfinite(lut).all() or not torch.equal(lut, quantize_e4m3(lut)) or not torch.isfinite(scale).all() or not torch.all(scale > 0):
            raise ValueError('Invalid payload LUT/scale values')
        expected = make_signs(k, alias_for(name)) if metadata['weight_method'] == METHODS[1] else None
        actual = entry['signs']
        if (expected is None) != (actual is None) or (expected is not None and not torch.equal(expected, actual)):
            raise ValueError('Invalid payload signs')


def apply_payload(model, payload, metadata, *, quantize_activations):
    validate_payload(payload, metadata, model)
    for name, module in eligible(model).items():
        entry = payload['layers'][name]
        qweight = LUTWeight(**{key: entry[key] for key in ('indices', 'luts', 'scales', 'original_shape', 'padded_shape')})
        parent_name, _, leaf = name.rpartition('.')
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, leaf, CachedLUTLinear(qweight, module.bias,
                quantize_activations=quantize_activations, signs=entry['signs']))
    return model
