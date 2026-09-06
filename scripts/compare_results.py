"""Compare BF16 and W3A8 result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


REQUIRED_METADATA = (
    "spec_version",
    "model_path",
    "device",
    "dataset",
    "dataset_config",
    "split",
    "seq_len",
    "num_eval_tokens",
    "num_predicted_tokens",
    "seed",
)


def _finite_float(result: dict, field: str, path: Path) -> float:
    try:
        value = float(result[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {field} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"{path}: {field} must be finite")
    return value


def _load_and_validate(path: Path, expected_mode: str) -> dict:
    try:
        result = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"could not read valid JSON from {path}") from exc
    if not isinstance(result, dict):
        raise ValueError(f"{path}: result must be a JSON object")

    required_fields = (*REQUIRED_METADATA, "mode", "mean_nll", "ppl")
    missing = [field for field in required_fields if field not in result]
    if missing:
        raise ValueError(f"{path}: missing required fields: {', '.join(missing)}")
    if result["mode"] != expected_mode:
        raise ValueError(f"{path}: expected mode {expected_mode!r}, got {result['mode']!r}")
    mean_nll = _finite_float(result, "mean_nll", path)
    ppl = _finite_float(result, "ppl", path)
    try:
        expected_ppl = math.exp(mean_nll)
    except OverflowError as exc:
        raise ValueError(f"{path}: mean_nll is too large for a finite PPL") from exc
    tolerance = 1e-6 * max(1.0, ppl)
    if abs(ppl - expected_ppl) >= tolerance:
        raise ValueError(
            f"{path}: ppl={ppl} is inconsistent with exp(mean_nll)={expected_ppl}"
        )
    return result


def _validate_metadata_match(bf16: dict, w3a8: dict) -> None:
    mismatched = [field for field in REQUIRED_METADATA if bf16[field] != w3a8[field]]
    if mismatched:
        details = ", ".join(
            f"{field}: {bf16[field]!r} != {w3a8[field]!r}" for field in mismatched
        )
        raise ValueError(f"metadata mismatch; comparison refused ({details})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bf16_json", type=Path)
    parser.add_argument("w3a8_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bf16 = _load_and_validate(args.bf16_json, "bf16")
    w3a8 = _load_and_validate(args.w3a8_json, "w3a8")
    _validate_metadata_match(bf16, w3a8)
    baseline = float(bf16["ppl"])
    quantized = float(w3a8["ppl"])
    baseline_nll = float(bf16["mean_nll"])
    quantized_nll = float(w3a8["mean_nll"])
    delta = quantized - baseline
    relative = delta / baseline * 100.0
    rows = [
        {
            "mode": "BF16",
            "mean_nll": baseline_nll,
            "ppl": baseline,
            "delta_ppl": "",
            "delta_ppl_percent": "",
        },
        {
            "mode": "W3A8",
            "mean_nll": quantized_nll,
            "ppl": quantized,
            "delta_ppl": delta,
            "delta_ppl_percent": relative,
        },
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"BF16 mean_nll={baseline_nll:.8f} PPL={baseline:.8f}")
    print(
        f"W3A8 mean_nll={quantized_nll:.8f} PPL={quantized:.8f} "
        f"delta={delta:.8f} delta_percent={relative:.4f}%"
    )


if __name__ == "__main__":
    main()
