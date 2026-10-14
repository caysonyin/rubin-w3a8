import copy
import json
import math
import sys

import pytest
import torch
from torch import nn

from src.experiment import (
    build_payload, FORMAT, CONFIG, recipe_config, calibration_metadata, apply_payload, payload_hash,
)
from src.gptq import quantize_gptq, sweep, inverse_cholesky
from src.grouping import to_n8k64_tiles, pad_matrix
from src.hadamard import rotate
from src.weight import quantize_weight
from src.diagnostics import collect_recipe, independent_weight_statistics
from scripts.ablation_protocol import (
    METHODS, SCHEMA_VERSION, validate_evaluations, evaluation_summaries, local_increments, reduce_inputs,
)
from scripts.diagnostic_protocol import (
    PROJECTIONS, reduce_tables, row_order, write_csv, read_verified_tables, assert_tables_equal,
)
from scripts import run_ablation


class Mini(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(8, 64)
        self.q_proj = nn.Linear(64, 8)
        self.k_proj = nn.Linear(64, 8)

    def forward(self, input_ids, use_cache=False):
        x = self.emb(input_ids)
        return self.q_proj(x) + self.k_proj(x)


def metadata(method, tokens):
    return dict(weight_method=method, format=FORMAT, config=recipe_config(method),
                calibration=calibration_metadata(tokens) if method in (METHODS[2], METHODS[3]) else None)


@pytest.mark.parametrize("method,expected", [
    (METHODS[0], "1b5f80bc29a755b6714d701c39689f19955faea28f6411c741f8f12a2b914a2b"),
    (METHODS[3], "5d117233e2bf44cbe40e75036f2d4e39f985e9b67cd9567561a9e67f6073274f"),
])
def test_original_recipe_fingerprints_from_merged_main(method, expected):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    model = Mini().bfloat16().eval()
    tokens = torch.arange(512) % 8
    payload = build_payload(model, metadata(method, tokens), tokens)
    assert payload["fingerprint"] == expected
    assert payload["metadata"]["config"] == CONFIG


def test_gptq_without_refit_is_exact_initial_sweep(monkeypatch):
    import src.gptq as gptq
    torch.manual_seed(67)
    weight = torch.randn(9, 128) * .03
    x = torch.randn(96, 2, 64)
    h = torch.einsum("tki,tkj->kij", x, x) / 96
    def forbidden(*args, **kwargs):
        raise AssertionError("Refit must not run in A2")
    monkeypatch.setattr(gptq, "refit", forbidden)
    actual, trace = quantize_gptq(weight, h, refit_iterations=0)
    initial = quantize_weight(weight)
    tiles = to_n8k64_tiles(pad_matrix(weight, initial.padded_shape)).reshape(2, 2, 8, 64)
    upper, dead = inverse_cholesky(h)
    _, expected = sweep(tiles, initial.luts, initial.scales, upper, dead)
    assert torch.equal(to_n8k64_tiles(actual.indices).reshape_as(expected), expected)
    assert torch.equal(actual.luts, initial.luts) and torch.equal(actual.scales, initial.scales)
    assert trace["accepted_iterations"] == 0 and len(trace["proxy_objective_trace"]) == 1
    with pytest.raises(ValueError):
        quantize_gptq(weight, h, refit_iterations=1)


def test_ablation_factors_and_a1_requires_no_calibration(tmp_path, monkeypatch):
    import src.experiment as experiment
    torch.manual_seed(42)
    teacher = Mini().bfloat16().eval()
    tokens = torch.arange(512) % 8
    original_collect = experiment.collect_hessians
    def forbidden(*args, **kwargs):
        raise AssertionError("A1 must not collect calibration")
    monkeypatch.setattr(experiment, "collect_hessians", forbidden)
    a1 = build_payload(teacher, metadata(METHODS[1], tokens))
    entry = a1["layers"]["q_proj"]
    independent = quantize_weight(rotate(teacher.q_proj.weight.float(), entry["signs"]))
    for field in ("indices", "luts", "scales"):
        assert torch.equal(entry[field], getattr(independent, field))
    monkeypatch.setattr(experiment, "collect_hessians", original_collect)
    payloads = {METHODS[1]: a1}
    for method in METHODS[2:]:
        payloads[method] = build_payload(teacher, metadata(method, tokens), tokens)
    paths = {}
    for method, payload in payloads.items():
        paths[method] = tmp_path / f"{method}.pt"
        torch.save(payload, paths[method])
        for a8 in (False, True):
            converted = apply_payload(copy.deepcopy(teacher), payload, metadata(method, tokens), quantize_activations=a8)
            assert torch.isfinite(converted(tokens[:8][None])).all()
    monkeypatch.setitem(run_ablation.EXPECTED_PAYLOADS, METHODS[3], payloads[METHODS[3]]["fingerprint"])
    checks = run_ablation.verify_factors(paths, {m: metadata(m, tokens) for m in METHODS}, teacher)
    assert checks["layers"] == 2 and checks["a2_refit_rounds"] == 0
    broken = copy.deepcopy(payloads[METHODS[2]])
    broken["layers"]["q_proj"]["luts"][0, 0, 0] = 0
    broken["fingerprint"] = payload_hash(broken)
    torch.save(broken, paths[METHODS[2]])
    with pytest.raises(ValueError, match="initial LUT"):
        run_ablation.verify_factors(paths, {m: metadata(m, tokens) for m in METHODS}, teacher)


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(17, 64)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.layers[0]
        layer.self_attn, layer.mlp = nn.Module(), nn.Module()
        for p in PROJECTIONS:
            setattr(layer.self_attn if p in PROJECTIONS[:4] else layer.mlp, p, nn.Linear(64, 9))
        self.calls = 0

    def forward(self, input_ids, use_cache=False):
        self.calls += 1
        x = self.embedding(input_ids)
        layer = self.model.layers[0]
        return torch.stack([getattr(layer.self_attn if p in PROJECTIONS[:4] else layer.mlp, p)(x) for p in PROJECTIONS]).sum(0)


def test_four_recipe_diagnostics_share_controls_and_input_observer_has_no_forward(tmp_path):
    torch.manual_seed(311)
    teacher = TinyTeacher().bfloat16().eval()
    calibration = torch.arange(512) % 17
    payloads = {m: build_payload(teacher, metadata(m, calibration), calibration if m in (METHODS[2], METHODS[3]) else None) for m in METHODS}
    before = teacher.calls
    blocks, weights, independent, inputs = [], {}, {}, []
    for method in METHODS:
        rows, statistics = collect_recipe(teacher, payloads[method], calibration[:8], block_length=4, canonical=False,
            methods=METHODS, schema_version=SCHEMA_VERSION,
            observer=run_ablation.input_observer(inputs) if method == METHODS[1] else None)
        blocks.extend(rows)
        weights.update(statistics)
        independent.update(independent_weight_statistics(teacher, payloads[method], canonical=False))
    assert teacher.calls - before == 8
    assert len(inputs) == 28 and len(reduce_inputs(inputs)) == 14
    blocks.sort(key=lambda r: row_order("block_stats.csv", r, methods=METHODS))
    tables = reduce_tables(blocks, weights, methods=METHODS, schema_version=SCHEMA_VERSION)
    assert len(tables["layer_errors.csv"]) == 28 and len(tables["precision_controls.csv"]) == 14
    assert all(r["cross_recipe_check"] == "PASS" for r in tables["precision_controls.csv"])
    for name, rows in tables.items():
        write_csv(tmp_path / name, rows, methods=METHODS, schema_version=SCHEMA_VERSION)
    checked = read_verified_tables(tmp_path, independent, methods=METHODS, schema_version=SCHEMA_VERSION)
    assert_tables_equal(tables, checked)
    with pytest.raises(ValueError):
        read_verified_tables(tmp_path)
    layer_steps, projection_steps = local_increments(tables)
    assert len(layer_steps) == 21 and len(projection_steps) == 21
    corrupted = copy.deepcopy(blocks)
    next(r for r in corrupted if r["weight_method"] == METHODS[2])["h64_control_output_hash"] = "a" * 64
    with pytest.raises(ValueError, match="H64 control"):
        reduce_tables(corrupted, weights, methods=METHODS, schema_version=SCHEMA_VERSION)
    for module in teacher.modules():
        assert not module._forward_hooks


@pytest.mark.parametrize("method", METHODS[1:3])
def test_prepare_only_never_loads_test_tokens(tmp_path, monkeypatch, method):
    from scripts import eval_qwen
    model_path = tmp_path / "model"
    model_path.mkdir()
    for name in ("config.json", "model.safetensors", "tokenizer.json"):
        (model_path / name).write_text("{}")
    class Tokenizer:
        def __call__(self, text, **kwargs):
            return {"input_ids": (torch.arange(512) % 8).tolist()}
    monkeypatch.setattr(eval_qwen.AutoTokenizer, "from_pretrained", lambda *a, **k: Tokenizer())
    monkeypatch.setattr(eval_qwen.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: Mini().bfloat16().eval())
    requests = []
    def dataset(*args, split, **kwargs):
        requests.append(split)
        return {"text": ["calibration"]}
    monkeypatch.setattr(eval_qwen, "load_dataset", dataset)
    def forbidden(*args, **kwargs):
        raise AssertionError("Prepare-only cannot load test data")
    monkeypatch.setattr(eval_qwen, "_load_tokens", forbidden)
    monkeypatch.setattr(sys, "argv", ["eval", "--mode", "w3a16", "--weight-method", method,
        "--model-path", str(model_path), "--prepare-only", "--save-payload", str(tmp_path / "weights.pt"),
        "--output", str(tmp_path / "prepare.json")])
    eval_qwen.main()
    assert requests == ([] if method == METHODS[1] else ["train"])


def evaluation_fixture(tmp_path):
    cal = calibration_metadata(torch.arange(512))
    bf16 = dict(mean_nll=2., ppl=math.exp(2.), eval_token_hash="c" * 64,
                python_version="3.14", torch_version="cpu", transformers_version="4.57")
    meta, paths = {}, {}
    for i, method in enumerate(METHODS):
        meta[method] = dict(model_fingerprint="a" * 64, tokenizer_fingerprint="b" * 64,
                           calibration=cal if method in (METHODS[2], METHODS[3]) else None)
        for mode, offset in (("w3a16", 0), ("w3a8", .02)):
            nll = 3 - i * .1 + offset
            row = dict(spec_version="reference-v1", experiment_version="ablation-comparison-v1" if i in (1, 2) else "quantization-comparison-v1",
                model_path="local", device="cpu", dataset="Salesforce/wikitext", dataset_config="wikitext-2-raw-v1",
                split="test", seq_len=1024, num_eval_tokens=32768, num_predicted_tokens=32736, seed=42,
                mode=mode, weight_method=method, full_model_smoke="PASS", cpu_threads=20, mean_nll=nll, ppl=math.exp(nll),
                quantization_format=FORMAT, recipe_config=recipe_config(method), payload_fingerprint=str(i + 1) * 64,
                **{k: v for k, v in bf16.items() if k not in ("mean_nll", "ppl")}, **meta[method])
            path = tmp_path / f"{i}_{mode}.json"
            path.write_text(json.dumps(row))
            paths[method, mode] = path
    return paths, meta, bf16


def test_conditional_nll_deltas_telescope_without_ppl_attribution(tmp_path):
    paths, meta, bf16 = evaluation_fixture(tmp_path)
    results = validate_evaluations(paths, meta, bf16)
    summary, increments, activation = evaluation_summaries(results, bf16)
    assert len(summary) == 8 and len(increments) == 6 and len(activation) == 4
    assert sum(r["delta_mean_nll"] for r in increments if r["mode"] == "w3a8") == pytest.approx(-.3)
    assert all(r["delta_mean_nll_a8_minus_a16"] == pytest.approx(.02) for r in activation)


@pytest.mark.parametrize("field,value", [
    ("eval_token_hash", "f" * 64), ("recipe_config", CONFIG), ("calibration", None),
    ("cpu_threads", 4), ("ppl", 1.), ("payload_fingerprint", "invalid"), ("experiment_version", "unknown"),
])
def test_ablation_comparison_rejects_unfair_rows(tmp_path, field, value):
    paths, meta, bf16 = evaluation_fixture(tmp_path)
    path = paths[METHODS[2], "w3a8"]
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        validate_evaluations(paths, meta, bf16)


def test_plan_a_reuse_checks_merged_source_and_keeps_original_tables():
    manifest, stages = run_ablation.verify_plan_a(run_ablation.ROOT / "results/layerwise_diagnostics/20261014")
    assert manifest["experiment_acceptance"] == "PASS"
    assert len(stages["full"]["block_stats.csv"]) == 1568


def test_missing_resources_block_formal_acceptance_without_execution(tmp_path, monkeypatch):
    monkeypatch.setattr(run_ablation.sys, "argv", ["ablation", "--model-path", str(tmp_path / "missing"),
        "--output-dir", str(tmp_path / "run")])
    def forbidden(*args, **kwargs):
        raise AssertionError("Missing resources must not run an experiment")
    monkeypatch.setattr(run_ablation, "execute", forbidden)
    assert run_ablation.main() == 2
    record = json.loads((tmp_path / "run/manifest.json").read_text())
    assert record["status"] == "BLOCKED" and record["experiment_acceptance"] == "BLOCKED"
    assert record["evaluated_rows"] == []
