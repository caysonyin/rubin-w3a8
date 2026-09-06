import math
import json
import subprocess
import sys
from pathlib import Path


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


def _run_compare(tmp_path: Path, bf16: dict, w3a8: dict) -> subprocess.CompletedProcess:
    bf16_path = tmp_path / "bf16.json"
    w3a8_path = tmp_path / "w3a8.json"
    output_path = tmp_path / "summary.csv"
    bf16_path.write_text(json.dumps(bf16), encoding="utf-8")
    w3a8_path.write_text(json.dumps(w3a8), encoding="utf-8")
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).parents[1] / "scripts" / "compare_results.py"),
            str(bf16_path),
            str(w3a8_path),
            "--output",
            str(output_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )


def test_compare_result_rejects_metadata_mismatch(tmp_path: Path) -> None:
    completed = _run_compare(tmp_path, _result("bf16"), _result("w3a8", seed=7))
    assert completed.returncode != 0
    assert "metadata mismatch" in completed.stderr
    assert not (tmp_path / "summary.csv").exists()


def test_compare_result_rejects_inconsistent_ppl(tmp_path: Path) -> None:
    completed = _run_compare(tmp_path, _result("bf16", ppl=99.0), _result("w3a8"))
    assert completed.returncode != 0
    assert "inconsistent" in completed.stderr
    assert not (tmp_path / "summary.csv").exists()
