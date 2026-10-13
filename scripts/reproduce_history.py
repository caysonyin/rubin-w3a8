"""Rerun the three original evaluations and the five frozen comparison rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from scripts.diagnose_layers import (
    ROOT, HISTORICAL, EXPECTED_PAYLOADS, configure_numeric, historical_metadata,
    check_payload, offline_environment, relative_path, source_fingerprints, write_json,
)


def run(command, log, threads):
    print(" ".join(command), flush=True)
    with log.open("w") as handle:
        result = subprocess.run(command, cwd=ROOT, env=offline_environment(threads),
                                stdout=handle, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f"Historical command failed: {relative_path(log)}")


def validate_reproduction(current, reference):
    checks = dict(delta_mean_nll=current["mean_nll"] - reference["mean_nll"],
                  delta_ppl=current["ppl"] - reference["ppl"])
    checks["pass"] = abs(checks["delta_mean_nll"]) <= 1e-5 and abs(checks["delta_ppl"]) <= 1e-3
    if not checks["pass"]:
        raise ValueError(f"Historical NLL/PPL drift: {checks}")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path, default=Path("models/Qwen3-0.6B-Base"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        parser.error("Historical reruns require a fresh output directory")
    out.mkdir(parents=True, exist_ok=True)
    legacy, comparison = out / "original", out / "quantization_comparison"
    legacy.mkdir()
    comparison.mkdir()
    started = time.perf_counter()
    manifest = dict(status="NOT_RUN", output_dir=relative_path(out), cpu_threads=20,
                    preparation_threads=4, source_fingerprints=source_fingerprints(),
                    executed_rows=[], payload_fingerprints={})
    try:
        configure_numeric()
        metadata = historical_metadata()
        # Freeze and verify the exact original inputs before any historical forward.
        from src.experiment import files_hash, tensor_hash, calibration_metadata
        from transformers import AutoTokenizer
        from datasets import load_dataset
        import torch
        model = args.model_path.resolve()
        model_hash = files_hash([*model.glob("*.safetensors"), model / "config.json"])
        tokenizer_hash = files_hash([p for pattern in ("tokenizer*", "vocab*", "merges*", "special_tokens*")
                                     for p in model.glob(pattern)])
        if (model_hash != metadata["plain_lloyd_max"]["model_fingerprint"]
                or tokenizer_hash != metadata["plain_lloyd_max"]["tokenizer_fingerprint"]):
            raise ValueError("Historical checkpoint/tokenizer fingerprint mismatch")
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
        hashes = {}
        for split, count in (("train", 512), ("test", 32768)):
            dataset = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
            ids = tokenizer("\n\n".join(dataset["text"]), add_special_tokens=False)["input_ids"][:count]
            if len(ids) != count:
                raise ValueError("Historical dataset has insufficient tokens")
            tokens = torch.tensor(ids, dtype=torch.long)
            hashes[split] = tensor_hash(tokens)
            if split == "train" and calibration_metadata(tokens) != metadata["h64_gptq_refit"]["calibration"]:
                raise ValueError("Historical calibration token hash mismatch")
        historical_bf16 = json.loads((HISTORICAL / "bf16.json").read_text())
        if hashes["test"] != historical_bf16["eval_token_hash"]:
            raise ValueError("Historical evaluation token hash mismatch")
        manifest["inputs"] = dict(model_fingerprint=model_hash, tokenizer_fingerprint=tokenizer_hash, token_hashes=hashes)
        write_json(out / "manifest.json", manifest)
        model_argument = relative_path(model)
        base = [sys.executable, "-m", "scripts.eval_qwen", "--model-path", model_argument]
        plain = comparison / "plain_lloyd_max.pt"
        run(base + ["--mode", "w3a16", "--prepare-only", "--save-payload", relative_path(plain),
                    "--output", relative_path(comparison / "prepare_plain.json")],
            comparison / "prepare_plain.log", 4)
        payload, details = check_payload(plain, metadata["plain_lloyd_max"], EXPECTED_PAYLOADS["plain_lloyd_max"])
        manifest["payload_fingerprints"]["plain_lloyd_max"] = details
        del payload
        for mode in ("bf16", "w3a16", "w3a8"):
            result = legacy / f"{mode}.json"
            command = base + ["--mode", mode, "--output", relative_path(result)]
            if mode != "bf16":
                command += ["--load-payload", relative_path(plain)]
            run(command, legacy / f"{mode}.log", 20)
            current = json.loads(result.read_text())
            reference = json.loads((ROOT / "results" / f"{mode}.json").read_text())
            check = validate_reproduction(current, reference)
            manifest["executed_rows"].append(dict(group="original", mode=mode, result=relative_path(result), **check))
            write_json(out / "manifest.json", manifest)
            print(f"Original {mode}: PPL={current['ppl']:.10f}, historical reproduction PASS", flush=True)
        run([sys.executable, "-m", "scripts.compare_results", relative_path(legacy / "bf16.json"),
             relative_path(legacy / "w3a8.json"), "--w3a16-json", relative_path(legacy / "w3a16.json"),
             "--output", relative_path(legacy / "summary.csv"),
             "--attribution-output", relative_path(legacy / "attribution.csv")],
            legacy / "comparison.log", 20)
        optimized = comparison / "h64_gptq_refit.pt"
        run(base + ["--mode", "w3a16", "--weight-method", "h64_gptq_refit", "--prepare-only",
                    "--save-payload", relative_path(optimized), "--output", relative_path(comparison / "prepare_h64.json")],
            comparison / "prepare_h64.log", 4)
        payload, details = check_payload(optimized, metadata["h64_gptq_refit"], EXPECTED_PAYLOADS["h64_gptq_refit"])
        manifest["payload_fingerprints"]["h64_gptq_refit"] = details
        del payload
        run([sys.executable, "-m", "scripts.run_quantization_comparison", "--model-path", model_argument,
             "--output-dir", relative_path(comparison), "--reuse-payloads"],
            out / "five_row_runner.log", 20)
        for name in ("bf16", "plain_w3a16", "plain_w3a8", "h64_gptq_refit_w3a16", "h64_gptq_refit_w3a8"):
            current = json.loads((comparison / f"{name}.json").read_text())
            reference = json.loads((HISTORICAL / f"{name}.json").read_text())
            check = validate_reproduction(current, reference)
            manifest["executed_rows"].append(dict(group="comparison", row=name,
                                                  result=relative_path(comparison / f"{name}.json"), **check))
        manifest["status"] = "PASS"
    except Exception as exc:
        manifest["status"] = "FAIL"
        manifest["error"] = str(exc).replace(str(ROOT), ".")
    finally:
        manifest["runtime_seconds"] = time.perf_counter() - started
        write_json(out / "manifest.json", manifest)
        lines = ["# Historical experiment reproduction", "", f"Acceptance: {manifest['status']}", "",
                 "| Group | Row | Δmean NLL | ΔPPL |", "|---|---|---:|---:|"]
        for row in manifest["executed_rows"]:
            lines.append(f"| {row['group']} | {row.get('mode', row.get('row'))} | {row['delta_mean_nll']:.12g} | {row['delta_ppl']:.12g} |")
        if manifest.get("error"):
            lines += ["", f"Stopped: {manifest['error']}"]
        (out / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Historical reproduction: {manifest['status']}", flush=True)
    return 0 if manifest["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
