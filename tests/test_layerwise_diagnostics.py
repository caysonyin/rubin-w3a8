import copy
import csv
import json
import math
import re
import shutil

import pytest
import torch
from torch import nn

from scripts import diagnose_layers as runner
from scripts.diagnostic_protocol import (
    METHODS, PROJECTIONS, SCHEMAS, EPSILON, COMPARISON_ATOL, COMPARISON_RTOL,
    classify_change, reduce_tables, row_order, read_csv, write_csv,
    read_verified_tables, validate_rows, output_metrics, validate_canonical_tables,
)
from src.diagnostics import collect_recipe, independent_weight_statistics, squared_error, energy
from src.experiment import build_payload, calibration_metadata, tensor_hash, FORMAT, CONFIG
from src.hadamard import rotate, inverse_rotate, make_signs
from src.linear import CachedLUTLinear, reference_linear_lookup


class TinyTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(17, 64)
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Module()])
        layer = self.model.layers[0]
        layer.self_attn, layer.mlp = nn.Module(), nn.Module()
        for projection in PROJECTIONS:
            parent = layer.self_attn if projection in PROJECTIONS[:4] else layer.mlp
            setattr(parent, projection, nn.Linear(64, 9, bias=True))
        self.calls = 0

    def forward(self, input_ids, use_cache=False):
        assert not use_cache
        self.calls += 1
        x = self.embedding(input_ids)
        layer = self.model.layers[0]
        outputs = [getattr(layer.self_attn if p in PROJECTIONS[:4] else layer.mlp, p)(x) for p in PROJECTIONS]
        return torch.stack(outputs).sum(0)


@pytest.fixture(scope="module")
def fixture():
    torch.set_num_threads(4)
    torch.manual_seed(1729)
    teacher = TinyTeacher().bfloat16().eval()
    calibration = torch.arange(512) % 17
    payloads = {}
    for method in METHODS:
        meta = dict(weight_method=method, format=FORMAT, config=CONFIG,
                    calibration=calibration_metadata(calibration) if method == METHODS[1] else None)
        payloads[method] = build_payload(teacher, meta, calibration if method == METHODS[1] else None)
    return teacher, payloads, torch.arange(8) % 17


def small_tables(fixture):
    teacher, payloads, tokens = fixture
    rows, weights, independent = [], {}, {}
    for method in METHODS:
        blocks, w = collect_recipe(teacher, payloads[method], tokens, block_length=4, canonical=False)
        rows.extend(blocks)
        weights.update(w)
        independent.update(independent_weight_statistics(teacher, payloads[method], canonical=False))
    rows.sort(key=lambda row: row_order("block_stats.csv", row))
    return reduce_tables(rows, weights), independent


def test_inverse_rotation_orientation():
    signs = make_signs(128, "nontrivial-signs")
    matrix = rotate(torch.eye(128), signs)
    # Dyadic inputs make both accumulation paths exact, isolating orientation.
    x = torch.arange(384).reshape(3, 128).float() / 32
    torch.testing.assert_close(inverse_rotate(x, signs), x @ matrix.T, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(inverse_rotate(rotate(x, signs), signs), x, rtol=1e-5, atol=1e-6)
    assert not torch.allclose(rotate(rotate(x, signs), signs), x)
    with pytest.raises(ValueError):
        inverse_rotate(x, torch.zeros_like(signs))


def test_inverse_rotation_cancellation_has_fp32_error_bound():
    signs = make_signs(128, "nontrivial-signs")
    matrix = rotate(torch.eye(128), signs)
    x = torch.arange(384).reshape(3, 128).float() / 37
    oracle = x.double() @ matrix.double().T
    actual = inverse_rotate(x, signs).double()
    # Six FP32 butterfly additions, followed by exact division by a power of two.
    block_l1 = x.double().reshape(3, 2, 64).abs().sum(-1, keepdim=True)
    bound = (6 * torch.finfo(torch.float32).eps * block_l1 / 8).expand(3, 2, 64).reshape_as(actual)
    assert torch.all((actual - oracle).abs() <= bound)


@pytest.mark.parametrize("method", METHODS)
def test_shared_reconstruction_matches_lookup_and_keeps_teacher(fixture, monkeypatch, method):
    import src.linear as linear
    teacher, payloads, tokens = fixture
    before = {name: tensor_hash(value) for name, value in teacher.state_dict().items()}
    with torch.inference_mode():
        native = teacher(tokens[None]).clone()
    reconstruct, original_forward = linear.reconstruct_weight, CachedLUTLinear.forward
    calls, pointers = [], []

    def count_reconstruction(qweight):
        calls.append(qweight.original_shape)
        return reconstruct(qweight)

    def check_forward(module, x):
        pointers.append((id(module), module.weight_hat.data_ptr(), module.quantize_activations))
        actual = original_forward(module, x)
        expected = reference_linear_lookup(x, module.qweight, module.bias, signs=module.signs,
                                          quantize_activations=module.quantize_activations)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        return actual

    monkeypatch.setattr(linear, "reconstruct_weight", count_reconstruction)
    monkeypatch.setattr(CachedLUTLinear, "forward", check_forward)
    start_calls = teacher.calls
    rows, weights = collect_recipe(teacher, payloads[method], tokens, block_length=4, canonical=False)
    assert teacher.calls - start_calls == 2
    assert len(calls) == 14 and len(pointers) == 28
    for a, b in zip(pointers[::2], pointers[1::2]):
        assert a[:2] == b[:2] and a[2] is False and b[2] is True
    assert len(rows) == 14 and len(weights) == 7
    assert all(row["output_numel"] == 36 for row in rows)
    assert all(w["weight_numel"] == 576 for w in weights.values())
    with torch.inference_mode():
        torch.testing.assert_close(teacher(tokens[None]), native, rtol=0, atol=0)
    assert before == {name: tensor_hash(value) for name, value in teacher.state_dict().items()}
    assert all(not module._forward_hooks for module in teacher.modules())


def test_fp64_subtraction_before_squares():
    a, b = torch.tensor([1e20, .00390625], dtype=torch.bfloat16), torch.tensor([-1e20, 0], dtype=torch.bfloat16)
    expected = sum((float(x) - float(y)) ** 2 for x, y in zip(a, b))
    assert math.isfinite(squared_error(a, b)) and squared_error(a, b) == expected
    assert energy(torch.zeros(3)) == 0
    with pytest.raises(ValueError):
        squared_error(torch.tensor([float("nan")]), torch.zeros(1))
    with pytest.raises(ValueError):
        squared_error(torch.zeros(1), torch.zeros(2))


@pytest.mark.parametrize("s16,s8,delta", [(12., 20., 2.), (20., 12., -2.), (12., 12., 0.)])
def test_teacher_mse_delta_is_signed(s16, s8, delta):
    result = output_metrics(dict(output_numel=4, teacher_output_energy=0.,
                                 sse_a16=s16, sse_a8=s8, sse_a8_vs_a16=100.))
    assert result["a8_teacher_mse_delta"] == delta
    assert result["activation_increment_mse"] == 25.
    assert result["relative_error_a16"] == math.sqrt(s16 / EPSILON)


def test_comparison_boundaries_preserve_raw_delta():
    boundary = COMPARISON_ATOL / (1 - COMPARISON_RTOL)
    assert boundary == COMPARISON_ATOL + COMPARISON_RTOL * boundary
    cases = [(0., 0., "tied"), (0., boundary, "tied"), (boundary, 0., "tied"),
             (0., math.nextafter(boundary, math.inf), "worsened"),
             (math.nextafter(boundary, math.inf), 0., "improved"), (1., 1. + 5e-7, "tied"),
             (1., 2., "worsened"), (2., 1., "improved")]
    for b0, b1, expected in cases:
        delta, tolerance, classification = classify_change(b0, b1)
        assert classification == expected and delta == b1 - b0
        assert tolerance == COMPARISON_ATOL + COMPARISON_RTOL * max(abs(b0), abs(b1))


def test_control_dedup_and_independent_recomputation(fixture, tmp_path):
    tables, independent = small_tables(fixture)
    assert len(tables["precision_controls.csv"]) == 14
    for name, rows in tables.items():
        write_csv(tmp_path / name, rows)
        assert read_csv(tmp_path / name) == rows
    assert read_verified_tables(tmp_path, independent) == tables
    for control in tables["precision_controls.csv"]:
        method = METHODS[0] if control["control_type"] == "fp32_unquantized" else METHODS[1]
        field = "sse_fp32_control" if method == METHODS[0] else "sse_h64_control"
        blocks = [r for r in tables["block_stats.csv"] if r["module_name"] == control["module_name"] and r["weight_method"] == method]
        assert control["control_sse"] == sum(r[field] for r in blocks)
    altered = copy.deepcopy(independent)
    altered[next(iter(altered))]["weight_sse"] += 1
    with pytest.raises(ValueError, match="recomputation mismatch"):
        read_verified_tables(tmp_path, altered)


@pytest.mark.parametrize("field", ["teacher_input_hash", "teacher_output_hash", "fp32_control_output_hash",
                                  "sse_fp32_control", "teacher_output_energy", "output_numel"])
def test_cross_recipe_mismatch_rejected(fixture, field):
    tables, weights = small_tables(fixture)
    rows = copy.deepcopy(tables["block_stats.csv"])
    b1 = next(r for r in rows if r["weight_method"] == METHODS[1])
    b1[field] = "a" * 64 if field.endswith("hash") else b1[field] + 1
    with pytest.raises(ValueError, match="mismatch"):
        reduce_tables(rows, weights)


def test_schema_matches_accepted_document():
    fence = chr(96) * 3
    headers = re.findall(fence + r"csv\n([^\n]+)\n" + fence, (runner.ROOT / "docs/update_plan_a.md").read_text())
    assert [tuple(next(csv.reader([header]))) for header in headers] == list(SCHEMAS.values())


def test_duplicate_keys_nonfinite_and_header_change_rejected(fixture, tmp_path):
    tables, _ = small_tables(fixture)
    controls = tables["precision_controls.csv"]
    with pytest.raises(ValueError, match="duplicate"):
        validate_rows("precision_controls.csv", [controls[0], controls[0]])
    altered = copy.deepcopy(controls)
    altered[0]["control_sse"] = float("nan")
    with pytest.raises(ValueError):
        write_csv(tmp_path / "precision_controls.csv", altered)
    path = tmp_path / "precision_controls.csv"
    write_csv(path, controls)
    path.write_text(path.read_text().replace("control_sse", "wrong_sse", 1))
    with pytest.raises(ValueError, match="header"):
        read_csv(path)


def test_hook_cleanup_after_failure(fixture):
    teacher, payloads, _ = fixture
    with pytest.raises((IndexError, RuntimeError)):
        collect_recipe(teacher, payloads[METHODS[0]], torch.full((4,), 99), block_length=4, canonical=False)
    assert all(not module._forward_hooks for module in teacher.modules())


def test_payload_self_validation_does_not_accept_new_baseline(fixture, tmp_path):
    payload = fixture[1][METHODS[0]]
    path = tmp_path / "plain.pt"
    torch.save(payload, path)
    with pytest.raises(ValueError, match="Historical payload fingerprint") as exc:
        runner.check_payload(path, payload["metadata"], "0" * 64)
    assert exc.value.payload_details["actual_fingerprint"] == payload["fingerprint"]
    loaded, _ = runner.check_payload(path, payload["metadata"], payload["fingerprint"])
    assert loaded["fingerprint"] == payload["fingerprint"]


def test_historical_records_tampering_rejected(tmp_path):
    assert set(runner.historical_metadata()) == set(METHODS)
    for path in runner.HISTORICAL.glob("*.json"):
        shutil.copy(path, tmp_path / path.name)
    path = tmp_path / "plain_w3a8.quantization.json"
    data = json.loads(path.read_text())
    data["payload_fingerprint"] = "0" * 64
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Historical quantization"):
        runner.historical_metadata(tmp_path)


def test_missing_runtime_preflight_is_blocked(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "dependency_check", lambda: ("BLOCKED", "Missing torch", {}))
    monkeypatch.setattr(runner, "load_teacher", lambda *_: pytest.fail("Preflight must not load/forward teacher"))
    monkeypatch.setattr(runner.sys, "argv", ["diagnose", "--output-dir", str(tmp_path / "run"),
                         "--model-path", str(tmp_path / "missing"), "--preflight-only"])
    assert runner.main() == 2
    manifest = json.loads((tmp_path / "run/manifest.json").read_text())
    assert manifest["historical_baseline_acceptance"] == "BLOCKED"
    assert manifest["experiment_acceptance"] == "BLOCKED" and manifest["smoke_acceptance"] == "NOT_RUN"
    assert not (tmp_path / "run/full").exists()


def test_refuses_overwrite_and_conflicting_flags(tmp_path, monkeypatch):
    existing = tmp_path / "run"
    existing.mkdir()
    (existing / "sentinel").write_text("preserve")
    monkeypatch.setattr(runner.sys, "argv", ["diagnose", "--output-dir", str(existing)])
    with pytest.raises(SystemExit):
        runner.main()
    assert (existing / "sentinel").read_text() == "preserve"
    monkeypatch.setattr(runner.sys, "argv", ["diagnose", "--output-dir", str(tmp_path / "other"),
                         "--preflight-only", "--prepare-payloads"])
    with pytest.raises(SystemExit):
        runner.main()


def test_smoke_prefix_checked(fixture):
    tables, _ = small_tables(fixture)
    smoke = {"block_stats.csv": [r for r in tables["block_stats.csv"] if r["token_block_index"] == 0]}
    runner.check_smoke_prefix(smoke, tables)
    altered = copy.deepcopy(smoke)
    altered["block_stats.csv"][0]["sse_a8"] += 1
    with pytest.raises(ValueError):
        runner.check_smoke_prefix(altered, tables)


def canonical_tables():
    blocks, weights = [], {}
    for block in range(28):
        for i, projection in enumerate(PROJECTIONS):
            name = f"model.layers.{block}.{'self_attn' if projection in PROJECTIONS[:4] else 'mlp'}.{projection}"
            for j, method in enumerate(METHODS):
                weights[name, method] = dict(in_features=64, out_features=9, weight_numel=576,
                                             weight_sse=2., weight_reference_energy=10.)
                blocks.append(dict(schema_version="layerwise-diagnostics-csv-v1", module_name=name,
                    transformer_block=block, projection=projection, weight_method=method,
                    payload_fingerprint=runner.EXPECTED_PAYLOADS[method], token_block_index=0,
                    num_tokens=1024, output_numel=9216, teacher_input_hash="a"*64, teacher_output_hash="b"*64,
                    teacher_output_energy=100., sse_a16=float(i + 1), sse_a8=float(i + 2),
                    sse_a8_vs_a16=4., sse_fp32_control=1., fp32_control_output_hash="c"*64,
                    sse_h64_control=1. if j else None, h64_control_output_hash="d"*64 if j else None))
    return reduce_tables(blocks, weights)


def test_canonical_coverage_and_png_pdf(tmp_path):
    pytest.importorskip("matplotlib")
    from scripts.plot_layer_diagnostics import render_figures
    tables = canonical_tables()
    validate_canonical_tables(tables, 1024, runner.EXPECTED_PAYLOADS)
    render_figures(tables["layer_errors.csv"], tmp_path)
    for stem in ("figure_a1_relative_output_error", "figure_a2_error_change"):
        assert (tmp_path / f"{stem}.png").read_bytes().startswith(b"\x89PNG")
        assert (tmp_path / f"{stem}.pdf").read_bytes().startswith(b"%PDF")
    tables["layer_errors.csv"].pop()
    with pytest.raises(ValueError, match="coverage"):
        validate_canonical_tables(tables, 1024, runner.EXPECTED_PAYLOADS)
