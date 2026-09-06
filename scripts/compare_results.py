"""Compare BF16, W3A16, and W3A8 result JSON files."""

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


def _validate_metadata_match(named_results: list[tuple[str, dict]]) -> None:
    reference_name, reference = named_results[0]
    mismatches: list[str] = []
    for name, result in named_results[1:]:
        for field in REQUIRED_METADATA:
            if reference[field] != result[field]:
                mismatches.append(
                    f"{field}: {reference_name}={reference[field]!r} != "
                    f"{name}={result[field]!r}"
                )
    if mismatches:
        raise ValueError(
            "metadata mismatch; comparison refused (" + ", ".join(mismatches) + ")"
        )


def _relative_delta(delta: float, baseline: float) -> float | str:
    if baseline == 0:
        return ""
    return delta / baseline * 100.0


def _summary_rows(results: list[tuple[str, dict]]) -> list[dict]:
    baseline_name, baseline_result = results[0]
    baseline_nll = float(baseline_result["mean_nll"])
    baseline_ppl = float(baseline_result["ppl"])
    rows = []
    for display_name, result in results:
        mean_nll = float(result["mean_nll"])
        ppl = float(result["ppl"])
        if display_name == baseline_name:
            delta_nll: float | str = ""
            delta_ppl: float | str = ""
            delta_nll_percent: float | str = ""
            delta_ppl_percent: float | str = ""
        else:
            delta_nll_value = mean_nll - baseline_nll
            delta_ppl_value = ppl - baseline_ppl
            delta_nll = delta_nll_value
            delta_ppl = delta_ppl_value
            delta_nll_percent = _relative_delta(delta_nll_value, baseline_nll)
            delta_ppl_percent = _relative_delta(delta_ppl_value, baseline_ppl)
        rows.append(
            {
                "mode": display_name,
                "mean_nll": mean_nll,
                "ppl": ppl,
                "delta_mean_nll": delta_nll,
                "delta_mean_nll_percent": delta_nll_percent,
                "delta_ppl": delta_ppl,
                "delta_ppl_percent": delta_ppl_percent,
            }
        )
    return rows


def _attribution_rows(
    bf16: dict, w3a16: dict, w3a8: dict
) -> list[dict[str, str | float]]:
    bf16_nll = float(bf16["mean_nll"])
    w3a16_nll = float(w3a16["mean_nll"])
    w3a8_nll = float(w3a8["mean_nll"])
    total_delta = w3a8_nll - bf16_nll

    def percent_of_total(delta: float) -> float | str:
        return _relative_delta(delta, total_delta)

    return [
        {
            "component": "weight_quantization",
            "from_mode": "BF16",
            "to_mode": "W3A16",
            "mean_nll_delta": w3a16_nll - bf16_nll,
            "mean_nll_delta_percent_of_total": percent_of_total(w3a16_nll - bf16_nll),
        },
        {
            "component": "activation_quantization_on_w3_weights",
            "from_mode": "W3A16",
            "to_mode": "W3A8",
            "mean_nll_delta": w3a8_nll - w3a16_nll,
            "mean_nll_delta_percent_of_total": percent_of_total(w3a8_nll - w3a16_nll),
        },
        {
            "component": "total_quantization",
            "from_mode": "BF16",
            "to_mode": "W3A8",
            "mean_nll_delta": total_delta,
            "mean_nll_delta_percent_of_total": percent_of_total(total_delta),
        },
    ]


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bf16_json", type=Path)
    parser.add_argument("w3a8_json", type=Path)
    parser.add_argument(
        "--w3a16-json",
        type=Path,
        help="Optional W3A16 result; when supplied, generate the three-mode comparison.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--attribution-output",
        type=Path,
        help="CSV path for mean-NLL attribution; defaults beside --output in three-mode runs.",
    )
    args = parser.parse_args()

    bf16 = _load_and_validate(args.bf16_json, "bf16")
    w3a8 = _load_and_validate(args.w3a8_json, "w3a8")
    named_results: list[tuple[str, dict]] = [("BF16", bf16)]
    if args.w3a16_json is not None:
        w3a16 = _load_and_validate(args.w3a16_json, "w3a16")
        named_results.append(("W3A16", w3a16))
    elif args.attribution_output is not None:
        raise ValueError("--attribution-output requires --w3a16-json")
    named_results.append(("W3A8", w3a8))
    _validate_metadata_match(
        [
            (args.bf16_json.name, bf16),
            *([(args.w3a16_json.name, w3a16)] if args.w3a16_json is not None else []),
            (args.w3a8_json.name, w3a8),
        ]
    )

    summary_rows = _summary_rows(named_results)
    _write_csv(args.output, summary_rows)

    if args.w3a16_json is not None:
        attribution_output = args.attribution_output or args.output.with_name("attribution.csv")
        _write_csv(attribution_output, _attribution_rows(bf16, w3a16, w3a8))

    for row in summary_rows:
        suffix = ""
        if row["delta_mean_nll"] != "":
            suffix = (
                f" delta_mean_nll={float(row['delta_mean_nll']):.8f}"
                f" delta_ppl={float(row['delta_ppl']):.8f}"
            )
        print(
            f"{row['mode']} mean_nll={float(row['mean_nll']):.8f} "
            f"PPL={float(row['ppl']):.8f}{suffix}"
        )


if __name__ == "__main__":
    main()
