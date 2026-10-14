# Reproducibility and Acceptance Record

This record summarizes the repository's reproducibility checks and recorded
evaluation artifacts. The runtime is Python 3.14 with `uv.lock`, and all model
evaluation is CPU-only.

## Original reference acceptance

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

## Current acceptance completed on 2026-10-14

The subsequent Plan A work is accepted independently of the historical
Phase 2/3 gates. Its [final protocol](update_plan_a.md) fixes the numerical
definitions, dataset protocol, recipes, experiment size, and deliverables.
The [diagnostic result report](layerwise_diagnostics.md) records the measured
outcomes and their interpretation.

| Gate | Result | Evidence |
|---|---|---|
| Implementation Acceptance | PASS; 91 tests, zero skips, PNG/PDF test passed | [Engineering record](../results/layerwise_diagnostics/implementation_acceptance.json), [test log](../results/layerwise_diagnostics/implementation_tests.log) |
| Original sanity checks | PASS; linear and full Qwen path | [Linear log](../results/historical_rerun/20261013/sanity_linear.log), [block log](../results/historical_rerun/20261013/sanity_block.log) |
| Full historical reproduction | PASS; all eight NLL/PPL deltas exactly zero | [Historical report](../results/historical_rerun/20261013/report.md), [manifest](../results/historical_rerun/20261013/manifest.json) |
| Historical Baseline Acceptance | PASS; exact checkpoint, tokenizer, calibration/test tokens, and both payload identities | [Preflight and execution manifest](../results/layerwise_diagnostics/20261014/manifest.json) |
| Smoke Acceptance | PASS; 1,024 validation tokens, full fixed layer coverage | [Smoke manifest](../results/layerwise_diagnostics/20261014/smoke/manifest.json) |
| Experiment Acceptance | PASS; 4,096 validation tokens, fixed CSV coverage, exact controls and smoke-prefix checks, independent output reductions and weight reconstruction | [Full manifest](../results/layerwise_diagnostics/20261014/full/manifest.json) |
| Recorded deliverables | Four CSVs, two PNG/PDF figures, tokens, manifests and reports | [Accepted run](../results/layerwise_diagnostics/20261014/) |

The original three modes and canonical five-row comparison were evaluated
again in full; the latter includes the shared BF16 and both recipes in A16/A8.
The original reference result files remain the comparison source. Numeric
dependencies retain their historical locked versions; the optional `plots`
extra adds the diagnostic figure renderer.

Implementation Acceptance depends on independent engineering evidence and its
source fingerprints. It can pass when model, data, or payload resources are
unavailable. Experiment Acceptance requires successful preflight, strict
historical identities, a passing smoke stage, and full diagnostics. Missing
resources produce BLOCKED status, and fingerprint mismatch fails historical
acceptance. A rebuilt or cross-platform payload cannot be silently accepted
as the original baseline. The
[earlier blocked preflight](../results/layerwise_diagnostics/20261013-preflight/manifest.json)
remains a separate record and is not the accepted experiment.

Output metrics are recomputed from `block_stats.csv` sufficient statistics;
this checks reductions without claiming an independent forward. Weight error
is independently rebuilt from the original checkpoint and frozen payload,
including B1 inverse rotation. Both use the fixed recomputation tolerance,
separate from the B1−B0 improvement classification threshold.

Current scope is the fixed two-recipe diagnostic experiment. It does not
establish layer PPL sensitivity, individual H64/GPTQ/refit attribution, a new
recovery method, an additional model/benchmark, or a hardware performance
result. Reproduction commands are in the [README](../README.md).

## Plan B acceptance completed on 2026-10-14

Plan B is the separately authorized four-recipe sequential ablation on merged
main `e6a3adc`. The [frozen protocol](ablation_protocol.md) and
[research report](ablation_report.md) define its scope and conclusions.
Historical Plan A engineering evidence above remains a record of that execution.

| Gate | Result | Evidence |
|---|---|---|
| Implementation Acceptance | PASS; 108 tests, zero skips; original recipe golden fingerprints unchanged | [Engineering record](../results/ablation/implementation/implementation_acceptance.json), [log](../results/ablation/implementation/implementation_tests.log) |
| Resource and historical acceptance | PASS; original model/tokenizer/calibration/diagnostic inputs and both historical payloads match | [Preflight](../results/ablation/20261014-preflight/manifest.json), [run manifest](../results/ablation/20261014/manifest.json) |
| Recipe fairness | PASS; 196 matching signs/scales, identical A1/A2 initial LUTs, A2 proxy matches B1 initial sweep, zero A2 refit rounds | [Factor checks](../results/ablation/20261014/manifest.json) |
| Smoke Acceptance | PASS; 1,024 validation tokens | [Smoke manifest](../results/ablation/20261014/smoke/manifest.json) |
| Diagnostic Acceptance | PASS; 4,096 validation tokens, four-recipe coverage, independent reductions/weight reconstruction, exact shared controls and smoke prefix | [Full manifest](../results/ablation/20261014/full/manifest.json) |
| Historical result regression | PASS; all four B0/B1 NLL/PPL differences exactly zero | [Baseline regression](../results/ablation/20261014/manifest.json) |
| Test and Experiment Acceptance | PASS; all eight quantized rows freshly evaluated with frozen payloads | [Summary](../results/ablation/20261014/summary.csv), [logs and JSON](../results/ablation/20261014/evaluations/) |
| Source and artifacts | PASS; source unchanged during execution, three PNG/PDF figure sets, relative document links | [Manifest](../results/ablation/20261014/manifest.json), [output directory](../results/ablation/20261014/) |

The historical BF16 score and B0/B1 local diagnostic statistics are explicitly
reused with provenance; their reuse is separate from the freshly measured
quantized Test rows. A1/A2 and all conditional comparisons preserve the original
model, quantization format, data isolation, and thread counts. No Test tuning,
new model, new quantization format, or GPU kernel was introduced.

Experiment acceptance verifies execution and measurement integrity. It does
not require each added recipe to improve accuracy: H64 alone is a valid
negative result in this setting. GPTQ given H64 provides the largest
conditional benefit, and refitting provides a smaller further benefit.
Independent component causality, cross-seed generality, and per-layer PPL
sensitivity remain outside the supported conclusions.
