"""Plan B recipe order, canonical evaluation checks, and derived increments."""
from __future__ import annotations

import math
from scripts.diagnostic_protocol import PROJECTIONS, EPSILON, classify_change
from scripts.compare_results import _load_and_validate, _validate_metadata_match
from scripts.compare_quantization import is_sha256
from src.experiment import FORMAT, recipe_config, GPTQ_METHODS

BASE_COMMIT = "e6a3adc1ac7f52fa37fb47d00f87c418dcdc5b31"
METHODS = ("plain_lloyd_max", "h64_lloyd_max", "h64_gptq", "h64_gptq_refit")
IDS = dict(zip(METHODS, ("B0", "A1", "A2", "B1")))
SCHEMA_VERSION = "ablation-diagnostics-csv-v1"
EDGES = tuple(zip(METHODS[:-1], METHODS[1:], ("H64_given_plain", "GPTQ_given_H64", "refit_given_H64_GPTQ")))
CANONICAL = dict(device="cpu", dataset="Salesforce/wikitext", dataset_config="wikitext-2-raw-v1",
                 split="test", seq_len=1024, num_eval_tokens=32768, num_predicted_tokens=32736,
                 seed=42, cpu_threads=20)


def validate_evaluations(paths, metadata, bf16):
    results = {}
    for method in METHODS:
        for mode in ("w3a16", "w3a8"):
            row = _load_and_validate(paths[method, mode], mode)
            if any(row.get(key) != value for key, value in CANONICAL.items()):
                raise ValueError("Noncanonical ablation evaluation")
            if (row.get("weight_method") != method or row.get("full_model_smoke") != "PASS"
                    or row.get("quantization_format") != FORMAT
                    or row.get("recipe_config") != recipe_config(method)):
                raise ValueError("Unexpected ablation recipe or smoke")
            for field in ("model_fingerprint", "tokenizer_fingerprint"):
                if row.get(field) != metadata[method][field]:
                    raise ValueError("Ablation checkpoint/tokenizer mismatch")
            for field in ("eval_token_hash", "python_version", "torch_version", "transformers_version"):
                if row.get(field) != bf16.get(field):
                    raise ValueError(f"Incompatible evaluation {field}")
            if row.get("calibration") != metadata[method]["calibration"]:
                raise ValueError("Calibration mismatch")
            version = "ablation-comparison-v1" if method in METHODS[1:3] else "quantization-comparison-v1"
            if row.get("experiment_version") != version:
                raise ValueError("Experiment version mismatch")
            if method in GPTQ_METHODS and (row["calibration"]["split"] != "train" or row["calibration"]["tokens"] != 512):
                raise ValueError("Invalid calibration split or count")
            results[method, mode] = row
        a, b = (results[method, mode] for mode in ("w3a16", "w3a8"))
        if not is_sha256(a.get("payload_fingerprint")) or a["payload_fingerprint"] != b["payload_fingerprint"]:
            raise ValueError("Paired A16/A8 weight payload mismatch")
    _validate_metadata_match([(str(paths[key]), row) for key, row in results.items()])
    return results


def evaluation_summaries(results, bf16):
    summary, increments, activation = [], [], []
    for method in METHODS:
        for mode in ("w3a16", "w3a8"):
            row = results[method, mode]
            summary.append(dict(recipe_id=IDS[method], weight_method=method, mode=mode,
                                mean_nll=row["mean_nll"], ppl=row["ppl"],
                                delta_mean_nll_vs_bf16=row["mean_nll"] - bf16["mean_nll"],
                                delta_ppl_vs_bf16=row["ppl"] - bf16["ppl"],
                                payload_fingerprint=row["payload_fingerprint"]))
        a, b = (results[method, mode] for mode in ("w3a16", "w3a8"))
        activation.append(dict(recipe_id=IDS[method], weight_method=method,
                               delta_mean_nll_a8_minus_a16=b["mean_nll"] - a["mean_nll"],
                               delta_ppl_a8_minus_a16=b["ppl"] - a["ppl"]))
    for before, after, contrast in EDGES:
        for mode in ("w3a16", "w3a8"):
            a, b = results[before, mode], results[after, mode]
            increments.append(dict(contrast=contrast, before_id=IDS[before], after_id=IDS[after], mode=mode,
                                   delta_mean_nll=b["mean_nll"] - a["mean_nll"],
                                   delta_ppl=b["ppl"] - a["ppl"], ppl_change_percent=100 * (b["ppl"] / a["ppl"] - 1)))
    return summary, increments, activation


def local_increments(tables):
    layers = tables["layer_errors.csv"]
    projections = tables["projection_summary.csv"]
    by_layer = {(r["module_name"], r["weight_method"]): r for r in layers}
    by_projection = {(r["projection"], r["weight_method"]): r for r in projections}
    layer_rows, projection_rows = [], []
    for before, after, contrast in EDGES:
        for name in dict.fromkeys(r["module_name"] for r in layers):
            a, b = by_layer[name, before], by_layer[name, after]
            row = dict(contrast=contrast, before_id=IDS[before], after_id=IDS[after], module_name=name,
                       transformer_block=a["transformer_block"], projection=a["projection"])
            for mode in ("a16", "a8"):
                delta, tolerance, decision = classify_change(a[f"relative_error_{mode}"], b[f"relative_error_{mode}"])
                row.update({f"before_relative_error_{mode}": a[f"relative_error_{mode}"],
                            f"after_relative_error_{mode}": b[f"relative_error_{mode}"],
                            f"delta_relative_error_{mode}": delta, f"tolerance_{mode}": tolerance,
                            f"comparison_{mode}": decision})
            for field in ("relative_weight_error", "activation_increment_mse", "a8_teacher_mse_delta"):
                row[f"delta_{field}"] = b[field] - a[field]
            layer_rows.append(row)
        for projection in PROJECTIONS:
            a, b = by_projection[projection, before], by_projection[projection, after]
            row = dict(contrast=contrast, before_id=IDS[before], after_id=IDS[after], projection=projection)
            for mode in ("a16", "a8"):
                row.update({f"before_relative_error_{mode}": a[f"relative_error_{mode}"],
                            f"after_relative_error_{mode}": b[f"relative_error_{mode}"],
                            f"delta_relative_error_{mode}": b[f"relative_error_{mode}"] - a[f"relative_error_{mode}"]})
                for kind in ("improved", "worsened", "tied"):
                    row[f"{kind}_layers_{mode}"] = sum(r[f"comparison_{mode}"] == kind for r in layer_rows
                                                       if r["contrast"] == contrast and r["projection"] == projection)
            for field in ("relative_weight_error", "activation_increment_mse", "a8_teacher_mse_delta"):
                row[f"delta_{field}"] = b[field] - a[field]
            projection_rows.append(row)
    return layer_rows, projection_rows


def reduce_inputs(blocks):
    """Pool input SSE/energy; do not average block ratios."""
    groups = {}
    keys = set()
    for row in blocks:
        key = (row["module_name"], row["basis"], row["token_block_index"])
        if key in keys or row["basis"] not in ("plain", "h64"):
            raise ValueError("Duplicate or invalid input observation")
        keys.add(key)
        for field in ("input_sse", "input_energy", "input_numel", "num_tokens"):
            if not math.isfinite(row[field]) or row[field] < 0:
                raise ValueError("Invalid input statistic")
        if row["input_numel"] <= 0 or row["num_tokens"] <= 0:
            raise ValueError("Empty input observation")
        groups.setdefault(key[:2], []).append(row)
    rows = []
    for (module, basis), group in groups.items():
        indices = sorted(r["token_block_index"] for r in group)
        if indices != list(range(len(group))):
            raise ValueError("Missing input token block")
        first = group[0]
        row = {field: first[field] for field in ("module_name", "projection", "transformer_block", "basis")}
        for field in ("input_sse", "input_energy", "input_numel", "num_tokens"):
            row[field] = sum(r[field] for r in group)
        row.update(num_token_blocks=len(group), input_mse=row["input_sse"] / row["input_numel"],
                   relative_input_error=math.sqrt(row["input_sse"] / (row["input_energy"] + EPSILON)))
        rows.append(row)
    rows.sort(key=lambda r: (r["transformer_block"], PROJECTIONS.index(r["projection"]), r["basis"]))
    return rows
