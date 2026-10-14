"""Frozen Plan A tables and reductions; no torch or plotting imports."""
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
import re
import statistics

SCHEMA_VERSION = "layerwise-diagnostics-csv-v1"
METHODS = ("plain_lloyd_max", "h64_gptq_refit")
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
EPSILON = 1e-12
COMPARISON_ATOL = 1e-8
COMPARISON_RTOL = 1e-6
RECOMPUTE_ATOL = 1e-12
RECOMPUTE_RTOL = 1e-10
HEADERS = {
    "layer_errors.csv": "schema_version,module_name,transformer_block,projection,weight_method,payload_fingerprint,in_features,out_features,num_token_blocks,num_tokens,output_numel,teacher_output_energy,teacher_zero_energy,sse_a16,mse_a16,relative_error_a16,sse_a8,mse_a8,relative_error_a8,sse_a8_vs_a16,activation_increment_mse,a8_teacher_mse_delta,weight_numel,weight_sse,weight_reference_energy,weight_zero_energy,relative_weight_error,delta_relative_error_a16,comparison_tolerance_a16,comparison_a16,delta_relative_error_a8,comparison_tolerance_a8,comparison_a8",
    "projection_summary.csv": "schema_version,projection,weight_method,num_layers,num_token_blocks_per_layer,num_tokens_per_layer,output_numel,teacher_output_energy,sse_a16,mse_a16,relative_error_a16,median_relative_error_a16,sse_a8,mse_a8,relative_error_a8,median_relative_error_a8,sse_a8_vs_a16,activation_increment_mse,a8_teacher_mse_delta,weight_numel,weight_sse,weight_reference_energy,relative_weight_error,median_relative_weight_error,improved_layers_a16,worsened_layers_a16,tied_layers_a16,improved_layers_a8,worsened_layers_a8,tied_layers_a8",
    "precision_controls.csv": "schema_version,module_name,transformer_block,projection,control_type,num_token_blocks,num_tokens,output_numel,teacher_output_energy,teacher_zero_energy,control_sse,control_mse,control_relative_error,cross_recipe_check",
    "block_stats.csv": "schema_version,module_name,transformer_block,projection,weight_method,payload_fingerprint,token_block_index,num_tokens,output_numel,teacher_input_hash,teacher_output_hash,teacher_output_energy,sse_a16,sse_a8,sse_a8_vs_a16,sse_fp32_control,fp32_control_output_hash,sse_h64_control,h64_control_output_hash",
}
SCHEMAS = {name: tuple(header.split(",")) for name, header in HEADERS.items()}
KEYS = {
    "layer_errors.csv": ("module_name", "weight_method"),
    "projection_summary.csv": ("projection", "weight_method"),
    "precision_controls.csv": ("module_name", "control_type"),
    "block_stats.csv": ("module_name", "weight_method", "token_block_index"),
}
STRINGS = {
    "schema_version", "module_name", "projection", "weight_method", "payload_fingerprint",
    "teacher_input_hash", "teacher_output_hash", "fp32_control_output_hash",
    "h64_control_output_hash", "control_type", "cross_recipe_check",
    "comparison_a16", "comparison_a8",
}
INTEGERS = {
    "transformer_block", "in_features", "out_features", "num_token_blocks", "num_tokens",
    "output_numel", "weight_numel", "num_layers", "num_token_blocks_per_layer",
    "num_tokens_per_layer", "token_block_index", "teacher_zero_energy", "weight_zero_energy",
    *(f"{kind}_layers_{mode}" for kind in ("improved", "worsened", "tied") for mode in ("a16", "a8")),
}
SIGNED_FLOATS = {"a8_teacher_mse_delta", "delta_relative_error_a16", "delta_relative_error_a8"}
COMPARE_FIELDS = tuple(field for field in SCHEMAS["layer_errors.csv"] if field.startswith(("delta_relative", "comparison_")))
COUNT_FIELDS = tuple(field for field in SCHEMAS["projection_summary.csv"] if "_layers_a" in field)
H64_FIELDS = ("sse_h64_control", "h64_control_output_hash")


def module_coordinates(name):
    match = re.fullmatch(r"model\.layers\.(\d+)\.(self_attn|mlp)\.(\w+)", name)
    if not match or match[3] not in PROJECTIONS:
        raise ValueError(f"Invalid target module name: {name}")
    block, parent, projection = int(match[1]), match[2], match[3]
    if block >= 28 or parent != ("self_attn" if projection in PROJECTIONS[:4] else "mlp"):
        raise ValueError(f"Invalid projection location: {name}")
    return block, projection


def classify_change(b0, b1):
    if not all(math.isfinite(x) and x >= 0 for x in (b0, b1)):
        raise ValueError("Comparison requires finite nonnegative errors")
    delta = b1 - b0
    tolerance = COMPARISON_ATOL + COMPARISON_RTOL * max(abs(b0), abs(b1))
    classification = "improved" if delta < -tolerance else "worsened" if delta > tolerance else "tied"
    return delta, tolerance, classification


def relative_error(sse, energy):
    return math.sqrt(sse / (energy + EPSILON))


def output_metrics(row):
    count, energy = row["output_numel"], row["teacher_output_energy"]
    if count <= 0:
        raise ValueError("Empty output statistics")
    return {
        "mse_a16": row["sse_a16"] / count,
        "relative_error_a16": relative_error(row["sse_a16"], energy),
        "mse_a8": row["sse_a8"] / count,
        "relative_error_a8": relative_error(row["sse_a8"], energy),
        "activation_increment_mse": row["sse_a8_vs_a16"] / count,
        "a8_teacher_mse_delta": (row["sse_a8"] - row["sse_a16"]) / count,
    }


def nullable(name, row):
    if row.get("weight_method") != METHODS[0]:
        return set()
    return set(COMPARE_FIELDS if name == "layer_errors.csv" else
               COUNT_FIELDS if name == "projection_summary.csv" else
               H64_FIELDS if name == "block_stats.csv" else ())


def validate_row(name, row, *, methods=METHODS, schema_version=SCHEMA_VERSION):
    if set(row) != set(SCHEMAS[name]):
        raise ValueError(f"{name}: fields do not match frozen schema")
    allowed_empty = nullable(name, row)
    for field, value in row.items():
        if value is None:
            if field not in allowed_empty:
                raise ValueError(f"{name}: missing {field}")
            continue
        if field in allowed_empty:
            raise ValueError(f"{name}: unexpected B0 value for {field}")
        if field in STRINGS:
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name}: invalid string {field}")
        elif field in INTEGERS:
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name}: invalid integer {field}")
        elif not isinstance(value, (float, int)) or not math.isfinite(value) or (field not in SIGNED_FLOATS and value < 0):
            raise ValueError(f"{name}: invalid finite statistic {field}")
        if field.endswith(("hash", "fingerprint")) and not re.fullmatch("[0-9a-f]{64}", value):
            raise ValueError(f"{name}: invalid SHA256 {field}")
    if row["schema_version"] != schema_version or row["projection"] not in PROJECTIONS:
        raise ValueError(f"{name}: unsupported version or projection")
    if "weight_method" in row and row["weight_method"] not in methods:
        raise ValueError(f"{name}: unsupported recipe")
    if "module_name" in row and module_coordinates(row["module_name"]) != (row["transformer_block"], row["projection"]):
        raise ValueError(f"{name}: inconsistent module coordinates")
    for field in ("comparison_a16", "comparison_a8"):
        if field in row and row[field] is not None and row[field] not in ("improved", "worsened", "tied"):
            raise ValueError(f"{name}: invalid comparison")
    for field in ("teacher_zero_energy", "weight_zero_energy"):
        energy = "teacher_output_energy" if field.startswith("teacher") else "weight_reference_energy"
        if field in row and row[field] != int(row[energy] == 0):
            raise ValueError(f"{name}: inconsistent zero energy marker")
    if "control_type" in row and row["control_type"] not in ("fp32_unquantized", "h64_unquantized"):
        raise ValueError("Invalid control type")
    if "cross_recipe_check" in row and row["cross_recipe_check"] != (
        "PASS" if row["control_type"] == "fp32_unquantized" or len(methods) > 2 else "NOT_APPLICABLE"
    ):
        raise ValueError("Invalid cross-recipe control status")


def row_order(name, row, *, methods=METHODS):
    projection = PROJECTIONS.index(row["projection"])
    if name == "projection_summary.csv":
        return projection, methods.index(row["weight_method"])
    tail = (("fp32_unquantized", "h64_unquantized").index(row["control_type"])
            if name == "precision_controls.csv" else methods.index(row["weight_method"]))
    return row["transformer_block"], projection, tail, row.get("token_block_index", 0)


def validate_rows(name, rows, *, methods=METHODS, schema_version=SCHEMA_VERSION):
    seen = set()
    for row in rows:
        validate_row(name, row, methods=methods, schema_version=schema_version)
        key = tuple(row[field] for field in KEYS[name])
        if key in seen:
            raise ValueError(f"{name}: duplicate primary key {key}")
        seen.add(key)
    if rows != sorted(rows, key=lambda row: row_order(name, row, methods=methods)):
        raise ValueError(f"{name}: noncanonical row order")


def write_csv(path, rows, *, methods=METHODS, schema_version=SCHEMA_VERSION):
    path = Path(path)
    name = path.name
    validate_rows(name, rows, methods=methods, schema_version=schema_version)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".csv.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCHEMAS[name])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: "" if value is None else format(value, ".17g")
                             if field not in STRINGS | INTEGERS else value for field, value in row.items()})
    os.replace(temporary, path)


def read_csv(path, *, methods=METHODS, schema_version=SCHEMA_VERSION):
    path = Path(path)
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != SCHEMAS[path.name]:
            raise ValueError(f"{path.name}: header mismatch")
        rows = []
        for raw in reader:
            if set(raw) != set(SCHEMAS[path.name]) or any(value is None for value in raw.values()):
                raise ValueError(f"{path.name}: malformed row")
            rows.append({field: None if value == "" else value if field in STRINGS
                         else int(value) if field in INTEGERS else float(value) for field, value in raw.items()})
    validate_rows(path.name, rows, methods=methods, schema_version=schema_version)
    return rows


def validate_block_pairs(rows, *, methods=METHODS, schema_version=SCHEMA_VERSION):
    if len(methods) < 2 or len(set(methods)) != len(methods) or methods[0] != METHODS[0]:
        raise ValueError("Diagnostics require unique recipes with plain B0 first")
    validate_rows("block_stats.csv", rows, methods=methods, schema_version=schema_version)
    pairs = {}
    for row in rows:
        pairs.setdefault((row["module_name"], row["token_block_index"]), {})[row["weight_method"]] = row
    common = ("teacher_input_hash", "teacher_output_hash", "num_tokens", "output_numel",
              "teacher_output_energy", "sse_fp32_control", "fp32_control_output_hash")
    for key, pair in pairs.items():
        if set(pair) != set(methods):
            raise ValueError(f"Missing paired recipe for {key}")
        if any(pair[methods[0]][field] != pair[method][field] for field in common for method in methods[1:]):
            raise ValueError(f"Teacher or FP32 control mismatch for {key}")
        if any(pair[methods[1]][field] != pair[method][field]
               for field in H64_FIELDS for method in methods[2:]):
            raise ValueError(f"Repeated H64 control mismatch for {key}")


def reduce_tables(blocks, weights, *, methods=METHODS, schema_version=SCHEMA_VERSION):
    """Rebuild tables from block sufficient statistics and one weight record per layer."""
    validate_block_pairs(blocks, methods=methods, schema_version=schema_version)
    groups = {}
    for row in blocks:
        groups.setdefault((row["module_name"], row["weight_method"]), []).append(row)
    if set(weights) != set(groups):
        raise ValueError("Independent weight records do not match observed layers")
    layers = []
    for key, group in groups.items():
        first, weight = group[0], weights[key]
        if [row["token_block_index"] for row in group] != list(range(len(group))):
            raise ValueError("Missing token block or nonzero starting index")
        for row in group:
            if (row["payload_fingerprint"] != first["payload_fingerprint"]
                    or row["num_tokens"] <= 0 or row["output_numel"] != row["num_tokens"] * weight["out_features"]):
                raise ValueError("Payload identity or output element count mismatch")
        layer = {field: first[field] for field in ("schema_version", "module_name", "transformer_block", "projection", "weight_method", "payload_fingerprint")}
        layer.update(weight)
        layer.update(num_token_blocks=len(group), num_tokens=sum(row["num_tokens"] for row in group))
        for field in ("output_numel", "teacher_output_energy", "sse_a16", "sse_a8", "sse_a8_vs_a16"):
            layer[field] = sum(row[field] for row in group)
        layer.update(output_metrics(layer))
        layer["teacher_zero_energy"] = int(layer["teacher_output_energy"] == 0)
        layer["weight_zero_energy"] = int(layer["weight_reference_energy"] == 0)
        if layer["weight_numel"] != layer["in_features"] * layer["out_features"]:
            raise ValueError("Weight element count mismatch")
        layer["relative_weight_error"] = math.sqrt(layer["weight_sse"]) / (math.sqrt(layer["weight_reference_energy"]) + EPSILON)
        layer.update({field: None for field in COMPARE_FIELDS})
        layers.append(layer)
    by_key = {(row["module_name"], row["weight_method"]): row for row in layers}
    for row in layers:
        if row["weight_method"] != methods[0]:
            baseline = by_key[row["module_name"], METHODS[0]]
            for mode in ("a16", "a8"):
                delta, tolerance, classification = classify_change(baseline[f"relative_error_{mode}"], row[f"relative_error_{mode}"])
                row.update({f"delta_relative_error_{mode}": delta,
                            f"comparison_tolerance_{mode}": tolerance, f"comparison_{mode}": classification})
    layers.sort(key=lambda row: row_order("layer_errors.csv", row, methods=methods))
    summaries = []
    for projection in PROJECTIONS:
        for method in methods:
            group = [row for row in layers if row["projection"] == projection and row["weight_method"] == method]
            if not group:
                continue
            if len({(row["num_tokens"], row["num_token_blocks"]) for row in group}) != 1:
                raise ValueError("Inconsistent projection token coverage")
            summary = dict(schema_version=schema_version, projection=projection, weight_method=method,
                           num_layers=len(group), num_token_blocks_per_layer=group[0]["num_token_blocks"],
                           num_tokens_per_layer=group[0]["num_tokens"])
            for field in ("output_numel", "teacher_output_energy", "sse_a16", "sse_a8", "sse_a8_vs_a16",
                          "weight_numel", "weight_sse", "weight_reference_energy"):
                summary[field] = sum(row[field] for row in group)
            summary.update(output_metrics(summary))
            summary["relative_weight_error"] = math.sqrt(summary["weight_sse"]) / (math.sqrt(summary["weight_reference_energy"]) + EPSILON)
            summary["median_relative_weight_error"] = statistics.median(row["relative_weight_error"] for row in group)
            for mode in ("a16", "a8"):
                summary[f"median_relative_error_{mode}"] = statistics.median(row[f"relative_error_{mode}"] for row in group)
                for classification in ("improved", "worsened", "tied"):
                    summary[f"{classification}_layers_{mode}"] = (
                        sum(row[f"comparison_{mode}"] == classification for row in group) if method != methods[0] else None)
            summaries.append(summary)
    controls = []
    for module in dict.fromkeys(row["module_name"] for row in layers):
        for kind, method, sse in (("fp32_unquantized", METHODS[0], "sse_fp32_control"),
                                   ("h64_unquantized", methods[1], "sse_h64_control")):
            group = groups[module, method]
            layer = by_key[module, method]
            control = {field: layer[field] for field in ("schema_version", "module_name", "transformer_block", "projection",
                       "num_token_blocks", "num_tokens", "output_numel", "teacher_output_energy", "teacher_zero_energy")}
            total = sum(row[sse] for row in group)
            control.update(control_type=kind, control_sse=total, control_mse=total / layer["output_numel"],
                           control_relative_error=relative_error(total, layer["teacher_output_energy"]),
                           cross_recipe_check="PASS" if method == methods[0] or len(methods) > 2 else "NOT_APPLICABLE")
            controls.append(control)
    controls.sort(key=lambda row: row_order("precision_controls.csv", row))
    result = {"layer_errors.csv": layers, "projection_summary.csv": summaries,
              "precision_controls.csv": controls, "block_stats.csv": blocks}
    for name, rows in result.items():
        validate_rows(name, rows, methods=methods, schema_version=schema_version)
    return result


def assert_tables_equal(actual, expected):
    for name in SCHEMAS:
        if len(actual[name]) != len(expected[name]):
            raise ValueError(f"{name}: row count mismatch")
        for a, b in zip(actual[name], expected[name]):
            for field in SCHEMAS[name]:
                av, bv = a[field], b[field]
                equal = math.isclose(av, bv, rel_tol=RECOMPUTE_RTOL, abs_tol=RECOMPUTE_ATOL) if (
                    av is not None and bv is not None and field not in STRINGS | INTEGERS) else av == bv
                if not equal:
                    raise ValueError(f"{name}: recomputation mismatch for {field}")


def read_verified_tables(directory, independent_weights=None, *, methods=METHODS, schema_version=SCHEMA_VERSION):
    directory = Path(directory)
    actual = {name: read_csv(directory / name, methods=methods, schema_version=schema_version) for name in SCHEMAS}
    weights = independent_weights if independent_weights is not None else {
        (row["module_name"], row["weight_method"]): {field: row[field] for field in (
            "in_features", "out_features", "weight_numel", "weight_sse", "weight_reference_energy")}
        for row in actual["layer_errors.csv"]
    }
    assert_tables_equal(actual, reduce_tables(actual["block_stats.csv"], weights, methods=methods, schema_version=schema_version))
    return actual


def validate_canonical_tables(tables, num_tokens, payload_fingerprints, *, methods=METHODS):
    names = {f"model.layers.{block}.{'self_attn' if projection in PROJECTIONS[:4] else 'mlp'}.{projection}"
             for block in range(28) for projection in PROJECTIONS}
    expected = {(name, method) for name in names for method in methods}
    if {(row["module_name"], row["weight_method"]) for row in tables["layer_errors.csv"]} != expected:
        raise ValueError("Incomplete canonical 196-layer coverage")
    if len(tables["projection_summary.csv"]) != 7 * len(methods) or len(tables["precision_controls.csv"]) != 392:
        raise ValueError("Incomplete canonical summary/control coverage")
    if len(tables["block_stats.csv"]) != 196 * len(methods) * (num_tokens // 1024):
        raise ValueError("Incomplete canonical token block coverage")
    for row in tables["layer_errors.csv"]:
        if (row["num_tokens"] != num_tokens or row["num_token_blocks"] != num_tokens // 1024
                or row["payload_fingerprint"] != payload_fingerprints[row["weight_method"]]):
            raise ValueError("Noncanonical diagnostic tokens or payload")
    if any(row["num_tokens"] != 1024 for row in tables["block_stats.csv"]):
        raise ValueError("Noncanonical token block length")
