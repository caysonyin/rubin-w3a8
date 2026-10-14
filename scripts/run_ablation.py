"""Execute frozen Plan B with reused historical baselines and eight fresh test rows."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

from scripts.ablation_protocol import (
    BASE_COMMIT, METHODS, IDS, SCHEMA_VERSION, validate_evaluations,
    evaluation_summaries, local_increments, reduce_inputs,
)
from scripts.diagnose_layers import (
    ROOT, EXPECTED_PAYLOADS, dependency_check, historical_metadata, check_payload,
    configure_numeric, offline_environment, load_data, load_teacher,
    source_fingerprints, implementation_evidence, write_json, relative_path, check_smoke_prefix,
)
from scripts.diagnostic_protocol import (
    read_verified_tables, reduce_tables, validate_canonical_tables, row_order, write_csv,
)
from scripts.compare_results import _write_csv, _load_and_validate


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_plan_a(directory):
    manifest = json.loads((directory / "manifest.json").read_text())
    if any(manifest.get(field) != "PASS" for field in (
        "implementation_acceptance", "historical_baseline_acceptance", "smoke_acceptance", "experiment_acceptance"
    )):
        raise ValueError("Plan A has not passed all acceptance gates")
    for path, expected in manifest["source_fingerprints"].items():
        original = subprocess.check_output(["git", "show", f"{BASE_COMMIT}:{path}"], cwd=ROOT)
        if hashlib.sha256(original).hexdigest() != expected:
            raise ValueError(f"Plan A source does not match merged baseline: {path}")
    stages = {}
    for name, count in (("smoke", 1024), ("full", 4096)):
        stages[name] = read_verified_tables(directory / name)
        validate_canonical_tables(stages[name], count, EXPECTED_PAYLOADS)
    check_smoke_prefix(stages["smoke"], stages["full"])
    return manifest, stages


def verify_factors(paths, metadata, teacher):
    """Verify the only removed components; no unrelated signs, scales, or LUT changes."""
    import torch
    from src.experiment import validate_payload, payload_hash
    payloads = {}
    for method in METHODS[1:]:
        payload = torch.load(paths[method], map_location="cpu", weights_only=True)
        validate_payload(payload, metadata[method], teacher)
        if payload["fingerprint"] != payload_hash(payload):
            raise ValueError("Invalid ablation payload fingerprint")
        if method in EXPECTED_PAYLOADS and payload["fingerprint"] != EXPECTED_PAYLOADS[method]:
            raise ValueError("Historical B1 identity changed")
        payloads[method] = payload
    for name in payloads[METHODS[1]]["layers"]:
        entries = [payloads[method]["layers"][name] for method in METHODS[1:]]
        for field in ("signs", "scales"):
            if any(not torch.equal(entries[0][field], entry[field]) for entry in entries[1:]):
                raise ValueError(f"Unfair ablation {field}: {name}")
        if not torch.equal(entries[0]["luts"], entries[1]["luts"]):
            raise ValueError(f"A2 initial LUT differs from A1: {name}")
        trace = payloads[METHODS[2]]["traces"][name]
        baseline_trace = payloads[METHODS[3]]["traces"][name]
        if (trace["accepted_iterations"] != 0 or len(trace["proxy_objective_trace"]) != 1
                or trace["proxy_objective_trace"][0] != baseline_trace["proxy_objective_trace"][0]):
            raise ValueError(f"A2 differs from B1 initial sweep: {name}")
    return dict(layers=len(payloads[METHODS[1]]["layers"]), signs_equal=True, scales_equal=True, a1_a2_luts_equal=True,
                a2_matches_b1_initial_proxy=True, a2_refit_rounds=0)


def input_observer(records):
    from src.activation import fake_quantize_activation_k64
    from src.hadamard import rotate
    from src.diagnostics import squared_error, energy
    from src.experiment import tensor_hash
    from scripts.diagnostic_protocol import module_coordinates

    def observe(name, module, x, y, entry, block):
        transformer, projection = module_coordinates(name)
        for basis, value in (("plain", x.float()), ("h64", rotate(x, entry["signs"]))):
            quantized = fake_quantize_activation_k64(value)
            records.append(dict(module_name=name, transformer_block=transformer, projection=projection,
                                basis=basis, token_block_index=block, num_tokens=x.numel() // module.in_features,
                                input_numel=value.numel(), input_energy=energy(value),
                                input_sse=squared_error(quantized, value),
                                teacher_input_hash=tensor_hash(x), input_hash=tensor_hash(value),
                                quantized_input_hash=tensor_hash(quantized)))
    return observe


def execute(command, log, threads):
    print(f"Running {' '.join(command[2:])}", flush=True)
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=offline_environment(threads),
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Execution failed ({result.returncode}): {relative_path(log)}")


def report(manifest, summary=None, increments=None):
    lines = ["# Plan B execution record", "", f"Status: {manifest['status']}", "",
             f"Base main: {BASE_COMMIT}", "",
             "Historical BF16 and Plan A local statistics are explicitly reused; all eight quantized test rows are freshly evaluated."]
    if summary:
        lines += ["", "| Recipe | Mode | Mean NLL | PPL |", "|---|---|---:|---:|"]
        for row in summary:
            lines.append(f"| {row['recipe_id']} | {row['mode']} | {row['mean_nll']:.12f} | {row['ppl']:.12f} |")
    if increments:
        lines += ["", "| Conditional step | Mode | ΔMean NLL | ΔPPL | PPL change % |",
                  "|---|---|---:|---:|---:|"]
        for row in increments:
            lines.append(f"| {row['before_id']}→{row['after_id']} | {row['mode']} | {row['delta_mean_nll']:.12f} | {row['delta_ppl']:.12f} | {row['ppl_change_percent']:.6f} |")
    if manifest.get("error"):
        lines += ["", f"Stopped: {manifest['error']}"]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/Qwen3-0.6B-Base")
    parser.add_argument("--baseline-payload-dir", type=Path, default=ROOT / "results/historical_rerun/20261013/quantization_comparison")
    parser.add_argument("--plan-a-dir", type=Path, default=ROOT / "results/layerwise_diagnostics/20261014")
    parser.add_argument("--implementation-dir", type=Path, default=ROOT / "results/ablation/implementation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Plan B requires a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest = dict(status="NOT_RUN", experiment_version="ablation-comparison-v1", schema_version=SCHEMA_VERSION,
                    base_commit=BASE_COMMIT, execution_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                    source_fingerprints=source_fingerprints(), protocol_sha256=sha256(ROOT / "docs/ablation_protocol.md"),
                    output_dir=relative_path(out), recipe_order=list(METHODS), preparation_threads=4, cpu_threads=20,
                    implementation_acceptance="NOT_RUN", historical_baseline_acceptance="NOT_RUN",
                    smoke_acceptance="NOT_RUN", experiment_acceptance="NOT_RUN", evaluated_rows=[])
    summary = increments = None
    try:
        subprocess.run(["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"], cwd=ROOT, check=True)
        status, reason, versions = dependency_check()
        manifest["software_versions"] = versions
        if status == "BLOCKED":
            raise FileNotFoundError(reason)
        if status != "PASS":
            raise ValueError(reason)
        status, evidence = implementation_evidence(args.implementation_dir)
        manifest.update(implementation_acceptance=status, implementation_evidence=evidence)
        torch = configure_numeric()
        from src.experiment import files_hash, recipe_config, FORMAT, calibration_metadata, validate_payload, payload_hash
        from src.diagnostics import collect_recipe, independent_weight_statistics
        historical = historical_metadata()
        model_path = args.model_path.resolve()
        model_hash = files_hash([*model_path.glob("*.safetensors"), model_path / "config.json"])
        tokenizer_hash = files_hash([p for pattern in ("tokenizer*", "vocab*", "merges*", "special_tokens*") for p in model_path.glob(pattern)])
        if (model_hash != historical[METHODS[0]]["model_fingerprint"]
                or tokenizer_hash != historical[METHODS[0]]["tokenizer_fingerprint"]):
            raise ValueError("Historical checkpoint/tokenizer mismatch")
        paths = {method: args.baseline_payload_dir.resolve() / f"{method}.pt" for method in EXPECTED_PAYLOADS}
        for method in EXPECTED_PAYLOADS:
            payload, details = check_payload(paths[method], historical[method], EXPECTED_PAYLOADS[method])
            manifest.setdefault("verified_payloads", {})[method] = details
            del payload
        plan_a_manifest, plan_a = verify_plan_a(args.plan_a_dir.resolve())
        tokens, calibration, token_info = load_data(model_path, "validation", need_calibration=True)
        if (token_info["token_hash"] != plan_a_manifest["diagnostic_tokens"]["token_hash"]
                or calibration_metadata(calibration) != historical[METHODS[3]]["calibration"]):
            raise ValueError("Historical diagnostic/calibration token mismatch")
        metadata = {method: historical[method] if method in historical else
                    dict(model_fingerprint=model_hash, tokenizer_fingerprint=tokenizer_hash, weight_method=method,
                         format=FORMAT, config=recipe_config(method),
                         calibration=calibration_metadata(calibration) if method == METHODS[2] else None)
                    for method in METHODS}
        bf16_path = ROOT / "results/quantization_comparison/bf16.json"
        bf16 = _load_and_validate(bf16_path, "bf16")
        manifest.update(historical_baseline_acceptance="PASS", recipe_metadata=metadata,
                        model_fingerprint=model_hash, tokenizer_fingerprint=tokenizer_hash,
                        diagnostic_tokens={k: v for k, v in token_info.items() if k != "token_ids"},
                        baseline_local_source=relative_path(args.plan_a_dir),
                        baseline_local_manifest_sha256=sha256(args.plan_a_dir / "manifest.json"),
                        bf16_reference=dict(path=relative_path(bf16_path), sha256=sha256(bf16_path), reused=True),
                        preflight="PASS")
        write_json(out / "diagnostic_tokens.json", token_info)
        if args.preflight_only:
            manifest["status"] = "PREFLIGHT_PASS"
            return 0
        if status != "PASS":
            raise FileNotFoundError("Current independent implementation acceptance is required before execution")
        weights_dir = out / "weights"
        weights_dir.mkdir()
        for method in METHODS[1:3]:
            paths[method] = weights_dir / f"{method}.pt"
            execute([sys.executable, "-m", "scripts.eval_qwen", "--mode", "w3a16",
                     "--weight-method", method, "--model-path", relative_path(model_path), "--prepare-only",
                     "--save-payload", relative_path(paths[method]), "--output", relative_path(weights_dir / f"prepare_{IDS[method]}.json")],
                    weights_dir / f"prepare_{IDS[method]}.log", 4)
        teacher = load_teacher(model_path)
        manifest["factor_checks"] = verify_factors(paths, metadata, teacher)
        fingerprints = {}
        for method in METHODS:
            payload = torch.load(paths[method], map_location="cpu", weights_only=True)
            validate_payload(payload, metadata[method], teacher)
            fingerprints[method] = payload["fingerprint"]
            if fingerprints[method] != payload_hash(payload):
                raise ValueError("Payload self-fingerprint mismatch")
            manifest.setdefault("verified_payloads", {})[method] = dict(path=relative_path(paths[method]),
                                                                       actual_fingerprint=fingerprints[method])
            del payload
        manifest["payload_fingerprints"] = fingerprints
        write_json(out / "manifest.json", manifest)
        smoke, inputs = None, []
        for stage, count in (("smoke", 1024), ("full", 4096)):
            manifest["active_stage"] = stage
            old = plan_a[stage]
            blocks = [{**row, "schema_version": SCHEMA_VERSION} for row in old["block_stats.csv"]]
            weights = {(r["module_name"], r["weight_method"]): {key: r[key] for key in
                       ("in_features", "out_features", "weight_numel", "weight_sse", "weight_reference_energy")}
                       for r in old["layer_errors.csv"]}
            independent = {}
            for method in METHODS:
                payload = torch.load(paths[method], map_location="cpu", weights_only=True)
                if method in METHODS[1:3]:
                    rows, statistics = collect_recipe(teacher, payload, tokens[:count], methods=METHODS,
                        schema_version=SCHEMA_VERSION, observer=input_observer(inputs) if stage == "full" and method == METHODS[1] else None)
                    blocks.extend(rows)
                    weights.update(statistics)
                independent.update(independent_weight_statistics(teacher, payload))
                del payload
            blocks.sort(key=lambda r: row_order("block_stats.csv", r, methods=METHODS))
            tables = reduce_tables(blocks, weights, methods=METHODS, schema_version=SCHEMA_VERSION)
            validate_canonical_tables(tables, count, fingerprints, methods=METHODS)
            stage_dir = out / stage
            stage_dir.mkdir()
            for name, rows in tables.items():
                write_csv(stage_dir / name, rows, methods=METHODS, schema_version=SCHEMA_VERSION)
            verified = read_verified_tables(stage_dir, independent, methods=METHODS, schema_version=SCHEMA_VERSION)
            if smoke is None:
                smoke = verified
            else:
                check_smoke_prefix(smoke, verified)
            manifest["smoke_acceptance" if stage == "smoke" else "diagnostic_acceptance"] = "PASS"
            manifest.setdefault("stages", {})[stage] = dict(num_tokens=count, row_counts={k: len(v) for k, v in verified.items()},
                output_statistics_recomputed=True, weight_errors_independently_recomputed=True,
                teacher_and_fp32_controls_exact=True, h64_controls_exact=True, baseline_local_statistics_reused=True)
            write_json(stage_dir / "manifest.json", manifest)
            write_json(out / "manifest.json", manifest)
        input_rows = reduce_inputs(inputs)
        if len(inputs) != 1568 or len(input_rows) != 392:
            raise ValueError("Incomplete input diagnostic coverage")
        _write_csv(out / "full/input_block_stats.csv", inputs)
        _write_csv(out / "full/input_errors.csv", input_rows)
        layer_steps, projection_steps = local_increments(verified)
        _write_csv(out / "full/layer_increments.csv", layer_steps)
        _write_csv(out / "full/projection_increments.csv", projection_steps)
        del teacher
        evaluations = out / "evaluations"
        evaluations.mkdir()
        result_paths = {}
        for method in METHODS:
            for mode in ("w3a16", "w3a8"):
                path = evaluations / f"{IDS[method]}_{mode}.json"
                execute([sys.executable, "-m", "scripts.eval_qwen", "--mode", mode, "--weight-method", method,
                         "--model-path", relative_path(model_path), "--load-payload", relative_path(paths[method]),
                         "--output", relative_path(path)], path.with_suffix(".log"), 20)
                current = json.loads(path.read_text())
                if current["payload_fingerprint"] != fingerprints[method]:
                    raise ValueError("Evaluation used a different payload")
                if method in EXPECTED_PAYLOADS:
                    stem = "plain" if method == METHODS[0] else method
                    reference = json.loads((ROOT / f"results/quantization_comparison/{stem}_{mode}.json").read_text())
                    regression = dict(delta_mean_nll=current["mean_nll"] - reference["mean_nll"],
                                      delta_ppl=current["ppl"] - reference["ppl"])
                    if any(value != 0 for value in regression.values()):
                        raise ValueError(f"Historical baseline drift: {IDS[method]} {mode} {regression}")
                    manifest.setdefault("baseline_regression", {})[f"{IDS[method]}_{mode}"] = regression
                result_paths[method, mode] = path
                manifest["evaluated_rows"].append(dict(recipe_id=IDS[method], mode=mode, path=relative_path(path),
                                                       sha256=sha256(path), mean_nll=current["mean_nll"], ppl=current["ppl"], reused=False))
                write_json(out / "manifest.json", manifest)
        results = validate_evaluations(result_paths, metadata, bf16)
        summary, increments, activation = evaluation_summaries(results, bf16)
        for name, rows in (("summary.csv", summary), ("conditional_increments.csv", increments), ("activation_increments.csv", activation)):
            _write_csv(out / name, rows)
        from scripts.plot_ablation import render
        render(out, summary, increments, verified, input_rows, layer_steps, bf16)
        if source_fingerprints() != manifest["source_fingerprints"]:
            raise ValueError("Experiment source changed during execution")
        manifest.update(status="PASS", experiment_acceptance="PASS", test_acceptance="PASS",
                        source_unchanged_during_execution=True, test_hyperparameter_tuning=False)
    except FileNotFoundError as exc:
        manifest.update(status="BLOCKED", experiment_acceptance="BLOCKED", error=str(exc).replace(str(ROOT), "."))
    except Exception as exc:
        manifest.update(status="FAIL", experiment_acceptance="FAIL", error=str(exc).replace(str(ROOT), "."))
    finally:
        manifest.update(runtime_seconds=time.perf_counter() - started,
                        peak_rss_gib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024),
                        child_peak_rss_gib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (1024 * 1024))
        write_json(out / "manifest.json", manifest)
        (out / "report.md").write_text(report(manifest, summary, increments))
    print(f"Plan B: {manifest['status']}", flush=True)
    return 0 if manifest["status"] in ("PASS", "PREFLIGHT_PASS") else 2 if manifest["status"] == "BLOCKED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
