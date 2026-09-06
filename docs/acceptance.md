# Reproducibility and Acceptance Record

This record summarizes the repository's reproducibility checks and recorded
evaluation artifacts. The runtime is Python 3.14 with `uv.lock`, and all model
evaluation is CPU-only.

## Executed gates

| Gate | Evidence | Status |
|---|---|---|
| A — Environment | `.python-version`, `requires-python`, `uv sync --locked`, runtime `Python 3.14.4` | PASS |
| B — Regression | [`results/pytest.log`](../results/pytest.log) | PASS |
| C — CPU reproducibility | `--device {cpu}` CLI plus deterministic smoke test | PASS |
| D — BF16 evaluation | [`results/bf16.json`](../results/bf16.json) | PASS |
| E — W3A8 evaluation | [`results/w3a8.json`](../results/w3a8.json) | PASS |
| F — Result integrity | metadata and `PPL = exp(mean NLL)` validation in `compare_results.py` | PASS |
| G — Accuracy comparison | [`results/summary.csv`](../results/summary.csv) | PASS |
| H — Documentation | [`README.md`](../README.md) and [`docs/final_report.md`](final_report.md) | PASS |

## Reproduction commands

```bash
uv sync --locked
uv run pytest -q
uv run python scripts/sanity_linear.py
uv run python scripts/sanity_block.py \
  --model-path models/Qwen3-0.6B-Base --device cpu
```

The two evaluator commands and the comparison command are documented in the
README. The evaluator records model, protocol, runtime, and repository metadata
in each JSON result. Historical PPL values are reproduced within the specified
tolerances, with the earlier records available through Git history.

## Scope stop

No layer-sensitivity study, accuracy-recovery method, CUDA/PTX/Triton path,
physical packing, performance benchmark, additional model, or additional
downstream benchmark is included in the current reference.
