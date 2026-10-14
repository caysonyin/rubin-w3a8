"""Run Plan A with strict historical provenance and separate acceptance gates."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import resource
import subprocess
import sys
import tempfile
import time
import tomllib

from scripts.diagnostic_protocol import (
    METHODS, SCHEMA_VERSION, EPSILON, COMPARISON_ATOL, COMPARISON_RTOL,
    SCHEMAS, reduce_tables, row_order, write_csv, read_verified_tables,
    validate_canonical_tables,
)

ROOT = Path(__file__).resolve().parents[1]
HISTORICAL = ROOT / "results/quantization_comparison"
EXPECTED_PAYLOADS = {
    METHODS[0]: "b51d1413b50282f1062fe02645caacb6893a3d37d34d8daeca582ba173453de9",
    METHODS[1]: "93da19247272e6ce0b6a7f0fd17c9c49b6f8462c34b3ddd8b3de842d8c68f043",
}


class ProtocolMismatch(ValueError):
    """Existing inputs violate the fixed experimental protocol."""


def relative_path(path):
    return Path(os.path.relpath(Path(path).resolve(), ROOT)).as_posix()


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, path)


def source_fingerprints():
    paths = sorted([*ROOT.glob("src/*.py"), *ROOT.glob("scripts/*.py"), *ROOT.glob("tests/*.py"),
                    ROOT / "pyproject.toml", ROOT / "uv.lock", ROOT / "docs/update_plan_a.md"])
    return {relative_path(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def file_fingerprint(paths):
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(path.name.encode())
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def offline_environment(threads=20):
    return {**os.environ, "HF_HUB_OFFLINE": "1", "HF_DATASETS_OFFLINE": "1",
            "OMP_NUM_THREADS": str(threads), "MKL_NUM_THREADS": str(threads)}


def configure_numeric(threads=20):
    os.environ.update({key: value for key, value in offline_environment(threads).items()
                       if key in ("HF_HUB_OFFLINE", "HF_DATASETS_OFFLINE", "OMP_NUM_THREADS", "MKL_NUM_THREADS")})
    import numpy as np
    import torch
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision("highest")
    return torch


def historical_metadata(directory=HISTORICAL):
    from scripts.compare_quantization import compare
    names = ("bf16", "plain_w3a16", "plain_w3a8", "h64_gptq_refit_w3a16", "h64_gptq_refit_w3a8")
    compare([directory / f"{name}.json" for name in names])
    records = [json.loads((directory / f"{name}.json").read_text()) for name in names]
    metadata = {}
    for method, index in ((METHODS[0], 1), (METHODS[1], 3)):
        record = records[index]
        if record["payload_fingerprint"] != EXPECTED_PAYLOADS[method]:
            raise ValueError(f"Historical record does not match fixed {method} fingerprint")
        for paired_index in (index, index + 1):
            quant = json.loads((directory / f"{names[paired_index]}.quantization.json").read_text())
            expected = dict(model_fingerprint=record["model_fingerprint"],
                            tokenizer_fingerprint=record["tokenizer_fingerprint"],
                            weight_method=method, format=record["quantization_format"],
                            config=record["recipe_config"], calibration=record["calibration"])
            if quant["metadata"] != expected or quant["payload_fingerprint"] != record["payload_fingerprint"]:
                raise ValueError("Historical quantization metadata mismatch")
        metadata[method] = expected
    return metadata


def dependency_check():
    locked = tomllib.loads((ROOT / "uv.lock").read_text())
    targets = ("torch", "numpy", "transformers", "datasets", "pytest")
    actual = {}
    for name in (*targets, "matplotlib"):
        try:
            actual[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "BLOCKED", f"Missing test/runtime dependency: {name}", actual
        if name in targets:
            versions = {package["version"] for package in locked["package"] if package["name"] == name}
            expected = {"2.14.0+cpu"} if name == "torch" else versions
            if actual[name] not in expected:
                return "FAIL", f"Locked dependency mismatch: {name}={actual[name]}", actual
    return "PASS", "Locked CPU runtime and plotting dependencies available", actual


def load_data(model_path, source, *, need_calibration=False):
    from datasets import load_dataset
    from transformers import AutoTokenizer
    from src.experiment import tensor_hash
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)

    def tokens_for(split):
        try:
            dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
        except Exception as exc:
            raise FileNotFoundError(f"Selected offline WikiText split unavailable: {split}") from exc
        return tokenizer("\n\n".join(dataset["text"]), add_special_tokens=False)["input_ids"]

    torch = sys.modules["torch"]
    split, start = ("validation", 0) if source == "validation" else ("train", 512)
    ids = tokens_for(split)
    if len(ids) < start + 4096:
        raise ProtocolMismatch("Diagnostic source has fewer than the fixed 4096 tokens")
    diagnostic = torch.tensor(ids[start:start + 4096], dtype=torch.long)
    calibration = None
    if need_calibration:
        train = ids if split == "train" else tokens_for("train")
        if len(train) < 512:
            raise ProtocolMismatch("Calibration requires exactly 512 train tokens")
        calibration = torch.tensor(train[:512], dtype=torch.long)
    info = dict(dataset="Salesforce/wikitext", dataset_config="wikitext-2-raw-v1",
                split=split, token_start=start, token_end=start + 4096,
                num_tokens=4096, token_hash=tensor_hash(diagnostic),
                smoke_token_hash=tensor_hash(diagnostic[:1024]), token_ids=diagnostic.tolist())
    return diagnostic, calibration, info


def load_teacher(model_path):
    from transformers import AutoModelForCausalLM
    from src.diagnostics import validate_teacher
    torch = sys.modules["torch"]
    model = AutoModelForCausalLM.from_pretrained(str(model_path), local_files_only=True,
                                               torch_dtype=torch.bfloat16).cpu().eval()
    validate_teacher(model)
    return model


def check_payload(path, metadata, expected, model=None):
    import torch
    from src.experiment import payload_hash, validate_metadata, validate_payload
    payload = torch.load(path, map_location="cpu", weights_only=True)
    actual = payload.get("fingerprint")
    details = dict(path=relative_path(path), expected_fingerprint=expected, actual_fingerprint=actual)
    try:
        validate_metadata(payload["metadata"])
        if payload["metadata"] != metadata or actual != payload_hash(payload):
            raise ValueError("Payload metadata or self-fingerprint mismatch")
        if actual != expected:
            raise ValueError("Historical payload fingerprint mismatch; rebuilt/platform-specific weights are not accepted")
        if model is not None:
            validate_payload(payload, metadata, model)
    except Exception as exc:
        error = ValueError(str(exc))
        error.payload_details = details
        raise error from exc
    return payload, details


def implementation_evidence():
    path = ROOT / "results/layerwise_diagnostics/implementation_acceptance.json"
    if not path.is_file():
        return "NOT_RUN", "No current independent implementation test evidence"
    evidence = json.loads(path.read_text())
    if (evidence.get("status") != "PASS" or evidence.get("source_fingerprints") != source_fingerprints()
            or evidence.get("returncode") != 0):
        return "NOT_RUN", "Independent test evidence is missing, stale or failed"
    log = ROOT / evidence["log"]
    if not log.is_file() or hashlib.sha256(log.read_bytes()).hexdigest() != evidence.get("log_sha256"):
        return "NOT_RUN", "Independent test log fingerprint mismatch"
    return "PASS", dict(evidence=relative_path(path), log=evidence["log"])


def preflight(args, manifest, *, supplied_payload_paths=None):
    """Do not load a model or execute forwards; tokenization is allowed."""
    gates = []

    def gate(name, status, details):
        gates.append(dict(check=name, status=status, details=details))

    status, message, versions = dependency_check()
    gate("dependencies", status, message)
    manifest["software_versions"] = versions
    meta = None
    if status == "PASS":
        configure_numeric()
        try:
            meta = historical_metadata()
            gate("historical_records", "PASS", "Five historical rows and paired quantization metadata verified")
        except Exception as exc:
            gate("historical_records", "BLOCKED" if isinstance(exc, FileNotFoundError) else "FAIL", str(exc))
    else:
        gate("historical_records", "BLOCKED", "Cannot validate numerical metadata without runtime dependencies")
    model_path = args.model_path.resolve()
    weights = list(model_path.glob("*.safetensors"))
    config = model_path / "config.json"
    tokenizer_files = sorted({p for pattern in ("tokenizer*", "vocab*", "merges*", "special_tokens*")
                              for p in model_path.glob(pattern) if p.is_file()})
    model_ready = False
    if not weights or not config.is_file() or not tokenizer_files:
        gate("checkpoint", "BLOCKED", dict(path=relative_path(model_path), reason="Complete local checkpoint/tokenizer missing"))
    elif meta is None:
        gate("checkpoint", "BLOCKED", "Historical metadata unavailable for fingerprint comparison")
    else:
        model_hash, tokenizer_hash = file_fingerprint([*weights, config]), file_fingerprint(tokenizer_files)
        manifest.update(model_fingerprint=model_hash, tokenizer_fingerprint=tokenizer_hash)
        model_ready = (model_hash == meta[METHODS[0]]["model_fingerprint"]
                       and tokenizer_hash == meta[METHODS[0]]["tokenizer_fingerprint"])
        gate("checkpoint", "PASS" if model_ready else "FAIL",
             dict(path=relative_path(model_path), expected_model=meta[METHODS[0]]["model_fingerprint"],
                  actual_model=model_hash, expected_tokenizer=meta[METHODS[0]]["tokenizer_fingerprint"],
                  actual_tokenizer=tokenizer_hash))
    payload_paths = supplied_payload_paths or {method: args.payload_dir / f"{method}.pt" for method in METHODS}
    missing = [method for method, path in payload_paths.items() if not path.is_file()]
    payload_failed = False
    if meta is not None:
        for method, path in payload_paths.items():
            if not path.is_file():
                gate(f"payload:{method}", "BLOCKED", dict(path=relative_path(path), preparation_requested=args.prepare_payloads))
                continue
            try:
                payload, details = check_payload(path, meta[method], EXPECTED_PAYLOADS[method])
                del payload
                gate(f"payload:{method}", "PASS", details)
            except Exception as exc:
                payload_failed = True
                gate(f"payload:{method}", "FAIL", dict(reason=str(exc), **getattr(exc, "payload_details", {})))
    else:
        for method in METHODS:
            gate(f"payload:{method}", "BLOCKED", "Historical metadata/runtime unavailable")
    data = calibration = data_info = None
    if model_ready and status == "PASS" and not payload_failed:
        try:
            data, calibration, data_info = load_data(model_path, args.diagnostic_source,
                                                   need_calibration=args.prepare_payloads and METHODS[1] in missing)
            if calibration is not None:
                from src.experiment import calibration_metadata
                if calibration_metadata(calibration) != meta[METHODS[1]]["calibration"]:
                    raise ProtocolMismatch("Historical train calibration token hash mismatch")
            gate("diagnostic_data", "PASS", {key: value for key, value in data_info.items() if key != "token_ids"})
            if calibration is not None:
                gate("calibration_data", "PASS", meta[METHODS[1]]["calibration"])
        except Exception as exc:
            # Offline cache errors are resource blockers; bad fixed token counts/hashes are failures.
            gate("diagnostic_data", "FAIL" if isinstance(exc, ProtocolMismatch) else "BLOCKED", str(exc))
    else:
        gate("diagnostic_data", "BLOCKED", "Cannot freeze selected diagnostic tokens before checkpoint/runtime readiness")
    gate("execution_configuration", "PASS", dict(device="cpu", seed=42, preparation_threads=4,
                                                diagnostic_threads=20, use_cache=False,
                                                diagnostic_source=args.diagnostic_source))
    manifest["preflight"] = gates
    ready = not any(g["status"] == "FAIL" for g in gates) and data is not None and (
        not missing or args.prepare_payloads)
    if not ready:
        failed = any(g["status"] == "FAIL" for g in gates)
        manifest["historical_baseline_acceptance"] = "FAIL" if failed else "BLOCKED"
        manifest["experiment_acceptance"] = "FAIL" if failed else "BLOCKED"
    return ready, meta, payload_paths, missing, data, calibration, data_info


def prepare_worker(job_path):
    job = json.loads(Path(job_path).read_text())
    torch = configure_numeric(4)
    from src.experiment import build_payload, calibration_metadata
    teacher = load_teacher(ROOT / job["model_path"])
    tokens = torch.tensor(job["calibration_ids"], dtype=torch.long) if job["calibration_ids"] is not None else None
    if tokens is not None and calibration_metadata(tokens) != job["metadata"]["calibration"]:
        raise ValueError("Worker calibration mismatch")
    payload = build_payload(teacher, job["metadata"], tokens)
    details = dict(expected_fingerprint=job["expected"], actual_fingerprint=payload["fingerprint"])
    write_json(job["result_path"], details)
    if payload["fingerprint"] != job["expected"]:
        raise ValueError("Rebuilt fingerprint does not match historical baseline")
    destination = ROOT / job["destination"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise ValueError("Refusing to overwrite existing prepared payload")
    torch.save(payload, destination)


def prepare_missing(args, metadata, paths, missing, calibration, manifest):
    for method in missing:
        destination = args.output_dir / "payloads" / f"{method}.pt"
        with tempfile.TemporaryDirectory(prefix="rubin-diagnostic-preparation-") as temporary:
            job_path = Path(temporary) / "job.json"
            result_path = Path(temporary) / "result.json"
            write_json(job_path, dict(model_path=relative_path(args.model_path), metadata=metadata[method],
                                     expected=EXPECTED_PAYLOADS[method], destination=relative_path(destination),
                                     calibration_ids=calibration.tolist() if method == METHODS[1] else None,
                                     result_path=str(result_path)))
            log = args.output_dir / f"prepare_{method}.log"
            with log.open("w") as handle:
                completed = subprocess.run(
                    [sys.executable, "-c", "import sys; from scripts.diagnose_layers import prepare_worker; prepare_worker(sys.argv[1])", str(job_path)],
                    cwd=ROOT, env=offline_environment(4), stdout=handle, stderr=subprocess.STDOUT)
            details = json.loads(result_path.read_text()) if result_path.is_file() else {}
            manifest.setdefault("preparation", []).append(dict(weight_method=method, log=relative_path(log),
                                                               returncode=completed.returncode, **details))
            if completed.returncode:
                manifest["historical_baseline_acceptance"] = "FAIL"
                raise ValueError(f"Preparation failed: {method}; see {relative_path(log)}")
        paths[method] = destination


def save_stage(out, tables):
    out.mkdir()
    for name, rows in tables.items():
        write_csv(out / name, rows)


def check_smoke_prefix(smoke, full):
    prefix = [row for row in full["block_stats.csv"] if row["token_block_index"] == 0]
    if prefix != smoke["block_stats.csv"]:
        raise ValueError("Smoke/full first-block statistics or hashes mismatch")


def report_text(manifest, tables=None):
    lines = ["# Plan A execution report", "", f"Implementation Acceptance: {manifest['implementation_acceptance']}",
             f"Historical Baseline Acceptance: {manifest['historical_baseline_acceptance']}",
             f"Smoke Acceptance: {manifest['smoke_acceptance']}",
             f"Experiment Acceptance: {manifest['experiment_acceptance']}", ""]
    if manifest.get("error"):
        lines += [f"Execution stopped: {manifest['error']}", ""]
    blocked = [gate for gate in manifest.get("preflight", []) if gate["status"] != "PASS"]
    for gate in blocked:
        lines.append(f"- {gate['check']}: {gate['status']} — {json.dumps(gate['details'], ensure_ascii=False)}")
    if tables:
        lines += ["", "Local relative output errors and signed A8 Teacher MSE changes:", "",
                  "| Recipe | Projection | A16 relative error | A8 relative error | Activation MSE | A8 Teacher MSE delta |",
                  "|---|---|---:|---:|---:|---:|"]
        for row in tables["projection_summary.csv"]:
            lines.append(f"| {row['weight_method']} | {row['projection']} | {row['relative_error_a16']:.8g} | "
                         f"{row['relative_error_a8']:.8g} | {row['activation_increment_mse']:.8g} | {row['a8_teacher_mse_delta']:.8g} |")
        for mode in ("a16", "a8"):
            group = [r for r in tables["layer_errors.csv"] if r["weight_method"] == METHODS[1]]
            counts = {kind: sum(r[f"comparison_{mode}"] == kind for r in group) for kind in ("improved", "worsened", "tied")}
            lines += ["", f"B1 − B0 {mode}: {counts}"]
        for method in METHODS:
            ranked = sorted((r for r in tables["layer_errors.csv"] if r["weight_method"] == method),
                            key=lambda r: (-r["activation_increment_mse"], r["module_name"]))[:10]
            lines += ["", f"Largest A8 activation increments for {method}:", ""]
            lines.extend(f"- {r['module_name']}: activation MSE={r['activation_increment_mse']:.8g}, "
                         f"A8 Teacher MSE delta={r['a8_teacher_mse_delta']:.8g}" for r in ranked)
        lines += ["", "Local errors do not establish layer PPL sensitivity, attribution to individual H64/GPTQ/refit steps, or hardware performance."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=ROOT / "models/Qwen3-0.6B-Base")
    parser.add_argument("--payload-dir", type=Path, default=HISTORICAL)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--diagnostic-source", choices=("validation", "train-holdout"), default="validation")
    parser.add_argument("--prepare-payloads", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.preflight_only and (args.prepare_payloads or args.smoke_only):
        parser.error("--preflight-only cannot prepare or execute")
    args.model_path, args.payload_dir, args.output_dir = (
        path.resolve() for path in (args.model_path, args.payload_dir, args.output_dir))
    if args.output_dir.exists() and (not args.output_dir.is_dir() or any(args.output_dir.iterdir())):
        parser.error("Refusing to overwrite a nonempty output directory")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    impl_status, impl_evidence = implementation_evidence()
    manifest = dict(experiment_version="layerwise-diagnostics-v1", schema_version=SCHEMA_VERSION,
                    epsilon=EPSILON, comparison_atol=COMPARISON_ATOL, comparison_rtol=COMPARISON_RTOL,
                    implementation_acceptance=impl_status, implementation_evidence=impl_evidence,
                    historical_baseline_acceptance="NOT_RUN", smoke_acceptance="NOT_RUN",
                    experiment_acceptance="NOT_RUN", source_fingerprints=source_fingerprints(),
                    model_path=relative_path(args.model_path), payload_dir=relative_path(args.payload_dir),
                    output_dir=relative_path(args.output_dir), python_version=platform.python_version(),
                    platform=platform.platform(), work_budget=dict(max_preparations_per_recipe=1,
                    teacher_token_blocks=10, diagnostic_tokens=4096, calibration_tokens=512,
                    wall_time_limit=None, memory_limit=None, gpu_count=0))
    tables = None
    exit_code = 0
    try:
        ready, metadata, paths, missing, tokens, calibration, token_info = preflight(args, manifest)
        if not ready:
            exit_code = 2
        elif not args.preflight_only and impl_status != "PASS":
            manifest["preflight"].append(dict(check="implementation_acceptance", status="BLOCKED",
                details="Current independent implementation test evidence is required before diagnostic forwards"))
            manifest["experiment_acceptance"] = "BLOCKED"
            exit_code = 2
        elif not args.preflight_only:
            if missing:
                prepare_missing(args, metadata, paths, missing, calibration, manifest)
                ready, metadata, paths, missing, tokens, calibration, token_info = preflight(
                    args, manifest, supplied_payload_paths=paths)
                if not ready or missing:
                    raise ValueError("Post-preparation preflight did not pass")
            torch = configure_numeric()
            teacher = load_teacher(args.model_path)
            manifest["execution"] = dict(cpu_threads=torch.get_num_threads(), mkldnn_enabled=torch.backends.mkldnn.enabled,
                                          mkl_available=torch.backends.mkl.is_available(),
                                          cpu_capability=torch.backends.cpu.get_cpu_capability(),
                                          float32_matmul_precision=torch.get_float32_matmul_precision(),
                                          device="cpu", seed=42, use_cache=False)
            from src.diagnostics import collect_recipe, independent_weight_statistics
            for method in METHODS:
                payload, details = check_payload(paths[method], metadata[method], EXPECTED_PAYLOADS[method], teacher)
                manifest.setdefault("verified_payloads", {})[method] = details
                del payload
            manifest["historical_baseline_acceptance"] = "PASS"
            manifest["recipe_metadata"] = metadata
            write_json(args.output_dir / "diagnostic_tokens.json", token_info)
            manifest["diagnostic_tokens"] = {key: value for key, value in token_info.items() if key != "token_ids"}
            smoke = None
            for stage, count in (("smoke", 1024), ("full", 4096)):
                if stage == "full" and args.smoke_only:
                    break
                manifest["active_stage"] = stage
                blocks, weights, independent = [], {}, {}
                for method in METHODS:
                    payload, _ = check_payload(paths[method], metadata[method], EXPECTED_PAYLOADS[method], teacher)
                    block_rows, weight_rows = collect_recipe(teacher, payload, tokens[:count])
                    blocks.extend(block_rows)
                    weights.update(weight_rows)
                    independent.update(independent_weight_statistics(teacher, payload))
                    del payload
                blocks.sort(key=lambda row: row_order("block_stats.csv", row))
                tables = reduce_tables(blocks, weights)
                validate_canonical_tables(tables, count, EXPECTED_PAYLOADS)
                stage_dir = args.output_dir / stage
                save_stage(stage_dir, tables)
                verified = read_verified_tables(stage_dir, independent)
                validate_canonical_tables(verified, count, EXPECTED_PAYLOADS)
                if smoke is not None:
                    check_smoke_prefix(smoke, verified)
                else:
                    smoke = verified
                from scripts.plot_layer_diagnostics import render_figures
                render_figures(verified["layer_errors.csv"], stage_dir)
                manifest[f"{stage}_acceptance" if stage == "smoke" else "experiment_acceptance"] = "PASS"
                manifest.setdefault("stages", {})[stage] = dict(num_tokens=count, num_token_blocks=count // 1024,
                    row_counts={name: len(rows) for name, rows in tables.items()},
                    output_statistics_recomputed=True, weight_errors_independently_recomputed=True,
                    precision_controls_deduplicated=True)
                write_json(stage_dir / "manifest.json", manifest)
                (stage_dir / "report.md").write_text(report_text(manifest, tables))
                write_json(args.output_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest["error"] = str(exc).replace(str(ROOT), ".")
        if manifest["historical_baseline_acceptance"] != "PASS":
            manifest["historical_baseline_acceptance"] = "FAIL"
        if manifest.get("active_stage") == "smoke":
            manifest["smoke_acceptance"] = "FAIL"
        manifest["experiment_acceptance"] = "FAIL"
        exit_code = 1
    finally:
        manifest["runtime_seconds"] = time.perf_counter() - started
        manifest["peak_rss_gib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
        manifest["child_peak_rss_gib"] = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / (1024 * 1024)
        write_json(args.output_dir / "manifest.json", manifest)
        (args.output_dir / "report.md").write_text(report_text(manifest, tables))
    print(json.dumps({key: manifest[key] for key in ("implementation_acceptance", "historical_baseline_acceptance",
                      "smoke_acceptance", "experiment_acceptance")}, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
