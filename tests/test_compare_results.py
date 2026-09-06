import csv
import math
import json
import subprocess
import sys
from pathlib import Path

import pytest


def _result(mode: str, **overrides: object) -> dict:
    result = {
        "spec_version": "reference-v1",
        "model_path": "models/Qwen3-0.6B-Base",
        "mode": mode,
        "device": "cpu",
        "dataset": "Salesforce/wikitext",
        "dataset_config": "wikitext-2-raw-v1",
        "split": "test",
        "seq_len": 1024,
        "num_eval_tokens": 32768,
        "num_predicted_tokens": 32736,
        "seed": 42,
        "mean_nll": 2.0,
        "ppl": math.exp(2.0),
    }
    result.update(overrides)
    return result


def _run_compare(
    tmp_path: Path,
    bf16: dict,
    w3a8: dict,
    w3a16: dict | None = None,
) -> subprocess.CompletedProcess:
    bf16_path = tmp_path / "bf16.json"
    w3a8_path = tmp_path / "w3a8.json"
    w3a16_path = tmp_path / "w3a16.json"
    output_path = tmp_path / "summary.csv"
    attribution_path = tmp_path / "attribution.csv"
    bf16_path.write_text(json.dumps(bf16), encoding="utf-8")
    w3a8_path.write_text(json.dumps(w3a8), encoding="utf-8")
    command = [
        sys.executable,
        str(Path(__file__).parents[1] / "scripts" / "compare_results.py"),
        str(bf16_path),
        str(w3a8_path),
    ]
    if w3a16 is not None:
        w3a16_path.write_text(json.dumps(w3a16), encoding="utf-8")
        command.extend(["--w3a16-json", str(w3a16_path)])
    command.extend(["--output", str(output_path)])
    if w3a16 is not None:
        command.extend(["--attribution-output", str(attribution_path)])
    return subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )


def test_compare_three_modes_writes_summary_and_attribution(tmp_path: Path) -> None:
    completed = _run_compare(
        tmp_path,
        _result("bf16", mean_nll=2.0, ppl=math.exp(2.0)),
        _result("w3a8", mean_nll=2.4, ppl=math.exp(2.4)),
        _result("w3a16", mean_nll=2.2, ppl=math.exp(2.2)),
    )

    assert completed.returncode == 0, completed.stderr
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["mode"] for row in rows] == ["BF16", "W3A16", "W3A8"]
    assert float(rows[1]["delta_mean_nll"]) == pytest.approx(0.2)
    assert float(rows[2]["delta_mean_nll"]) == pytest.approx(0.4)

    with (tmp_path / "attribution.csv").open(newline="", encoding="utf-8") as handle:
        attribution_rows = list(csv.DictReader(handle))
    assert [row["component"] for row in attribution_rows] == [
        "weight_quantization",
        "activation_quantization_on_w3_weights",
        "total_quantization",
    ]
    assert float(attribution_rows[0]["mean_nll_delta"]) == pytest.approx(0.2)
    assert float(attribution_rows[1]["mean_nll_delta"]) == pytest.approx(0.2)


def test_compare_two_modes_remains_compatible(tmp_path: Path) -> None:
    completed = _run_compare(
        tmp_path,
        _result("bf16"),
        _result("w3a8", mean_nll=2.4, ppl=math.exp(2.4)),
    )

    assert completed.returncode == 0, completed.stderr
    with (tmp_path / "summary.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert [row["mode"] for row in rows] == ["BF16", "W3A8"]
    assert not (tmp_path / "attribution.csv").exists()


def test_compare_result_rejects_metadata_mismatch(tmp_path: Path) -> None:
    completed = _run_compare(
        tmp_path,
        _result("bf16"),
        _result("w3a8"),
        _result("w3a16", seed=7),
    )
    assert completed.returncode != 0
    assert "metadata mismatch" in completed.stderr
    assert not (tmp_path / "summary.csv").exists()
    assert not (tmp_path / "attribution.csv").exists()


def test_compare_result_rejects_w3a8_metadata_mismatch(tmp_path: Path) -> None:
    completed = _run_compare(
        tmp_path,
        _result("bf16"),
        _result("w3a8", dataset_config="other-config"),
        _result("w3a16"),
    )
    assert completed.returncode != 0
    assert "metadata mismatch" in completed.stderr
    assert not (tmp_path / "summary.csv").exists()
    assert not (tmp_path / "attribution.csv").exists()


def test_compare_result_rejects_inconsistent_ppl(tmp_path: Path) -> None:
    completed = _run_compare(
        tmp_path,
        _result("bf16", ppl=99.0),
        _result("w3a8"),
        _result("w3a16"),
    )
    assert completed.returncode != 0
    assert "inconsistent" in completed.stderr
    assert not (tmp_path / "summary.csv").exists()
    assert not (tmp_path / "attribution.csv").exists()


def test_compare_result_rejects_wrong_w3a16_mode(tmp_path: Path) -> None:
    completed = _run_compare(
        tmp_path,
        _result("bf16"),
        _result("w3a8"),
        _result("w3a8"),
    )
    assert completed.returncode != 0
    assert "expected mode 'w3a16'" in completed.stderr
    assert not (tmp_path / "summary.csv").exists()
