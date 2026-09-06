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
| E — W3A16 evaluation | [`results/w3a16.json`](../results/w3a16.json) | PASS |
| F — W3A8 evaluation | [`results/w3a8.json`](../results/w3a8.json) | PASS |
| G — Result integrity | metadata and `PPL = exp(mean NLL)` validation in `compare_results.py` | PASS |
| H — Accuracy comparison and NLL attribution | [`results/summary.csv`](../results/summary.csv), [`results/attribution.csv`](../results/attribution.csv) | PASS |
| I — Documentation | [`README.md`](../README.md) and [`docs/final_report.md`](final_report.md) | PASS |

## Reproduction commands

```bash
uv sync --locked
uv run pytest -q
uv run python scripts/sanity_linear.py
uv run python scripts/sanity_block.py \
  --model-path models/Qwen3-0.6B-Base --device cpu
```

The three evaluator commands and the comparison command are documented in the
README. The evaluator records model, protocol, runtime, and repository metadata
in each JSON result. The comparison records three rows and uses mean NLL for
the W3 weight and activation quantization attribution.

## Historical Phase 2/3 scope stop

No W16A8, layer-sensitivity study, accuracy-recovery method, CUDA/PTX/Triton
path, physical packing, performance benchmark, additional model, or additional
downstream benchmark was included in the original Phase 2/3 acceptance scope.

The subsequently authorized H64 + GPTQ + LUT-refit comparison is documented
separately in [quantization_comparison.md](quantization_comparison.md); it
does not retroactively change the gates above.
