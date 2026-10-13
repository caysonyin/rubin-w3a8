# Rubin LUT-W3A8/W3A16 CPU Numerical Reference — Original Reference Report

This report preserves the original three-mode reference experiment and its
interpretation. Subsequent accepted work is recorded in the
[two-recipe comparison](quantization_comparison.md) and
[Plan A diagnostic results](layerwise_diagnostics.md).
The latest execution completed on 2026-10-14 and reproduced all eight
historical evaluation rows exactly; see the [acceptance record](acceptance.md).

## 1. Project scope and contents

This repository contains a CPU numerical reference for LUT-based W3A8 and
W3A16 configurations, together with its numerical specification, tests,
reproducibility checks, and recorded Qwen evaluation results.

The scope is deliberately limited to a CPU numerical reference. It does not
model any particular hardware implementation or benchmark hardware performance.

## 2. Rubin LUT-W3A8/W3A16 numerical semantics

The logical representation stores a 3-bit index for each weight value. The
index selects one of eight E4M3 values in a tile-local LUT:

```text
3-bit index -> 8-entry E4M3 LUT
```

The implementation models LUT lookup rather than an INT3 × INT8 path. It
preserves PyTorch `nn.Linear.weight` orientation, uses logical N8 × K64 tiles,
and reconstructs each weight as `scale[row] * LUT[index]`.

## 3. Reference boundary

The reference models the following target concepts:

- 3-bit Matrix B indices;
- an eight-entry E4M3 lookup table;
- E4M3 lookup semantics;
- a LUT-based matrix multiplication context.

The following are project-defined numerical choices and are not intended as a
specification of any production implementation:

- FP32 K64 max-absolute scales;
- padding and valid-value masking;
- the E4M3 software emulator;
- quantile initialization and E4M3-constrained Lloyd-Max fitting;
- fake-quantized activations;
- CPU reconstruction and the Qwen evaluation protocol.

Lloyd-Max and the scale recipe apply only to this reference implementation.

## 4. CPU numerical simulator

The implemented chain is:

```text
E4M3 emulator
    -> N8 x K64 grouping
    -> FP32 weight and activation scales
    -> E4M3-constrained Lloyd-Max LUT fitting
    -> 3-bit LUT indices
    -> dense FP32 reconstruction for the Qwen path
    -> BF16 activations for W3A16 or fake-quantized activations for W3A8
    -> FP32 accumulation
```

The explicit index-to-LUT implementation remains a correctness oracle for
small matrices. The Qwen path reconstructs each quantized weight once and
caches the dense FP32 weight. Only `q_proj`, `k_proj`, `v_proj`, `o_proj`,
`gate_proj`, `up_proj`, and `down_proj` are replaced.

## 5. Recorded evaluation protocol

| Field | Frozen value |
|---|---|
| Model | local `models/Qwen3-0.6B-Base` |
| Device | CPU |
| Dataset | `Salesforce/wikitext` |
| Config / split | `wikitext-2-raw-v1` / `test` |
| Sequence length | 1024 |
| Max evaluation tokens | 32768 |
| Predicted tokens | 32736 |
| Seed | 42 |
| Cache | `use_cache=False` |

The environment is Python 3.14 with dependencies from `uv.lock`, installed by
`uv sync --locked`. All modes tokenize the same joined WikiText-2 text and use
summed FP32 cross-entropy to compute mean NLL and PPL. W3A16 preserves the
model's BF16 activation values and only promotes them to FP32 for accumulation;
it is not an IEEE FP16 path.

## 6. Results

The recorded evaluation produced the following reference results:

| Mode | mean NLL | PPL | Δmean NLL | ΔPPL |
|---|---:|---:|---:|---:|
| BF16 | 2.64401770896576 | 14.069617833591494 | — | — |
| LUT-W3A16 | 3.3644417383337535 | 28.917349344449057 | 0.7204240293679933 | 14.847731510857564 |
| LUT-W3A8 | 3.3695883653031067 | 29.066559790365766 | 0.7255706563373465 | 14.996941956774272 |

The machine-readable results are [`results/bf16.json`](../results/bf16.json),
[`results/w3a16.json`](../results/w3a16.json),
[`results/w3a8.json`](../results/w3a8.json),
[`results/summary.csv`](../results/summary.csv), and
[`results/attribution.csv`](../results/attribution.csv). Runtime is recorded as
execution metadata only; it is not a Rubin performance comparison.

The result comparison first checks shared experiment metadata and verifies
`PPL = exp(mean NLL)` within the configured numerical tolerance. The
W3A16 → W3A8 activation quantization increment is `0.005146626969353196`
mean NLL, compared with the total BF16 → W3A8 increment of
`0.7255706563373465`. The `attribution.csv` file uses mean NLL differences to
separate BF16 → W3A16 weight quantization from the additional W3A16 → W3A8
activation quantization; PPL differences are descriptive and are not treated
as additive.

## 7. Interpretation

The three-mode interpretation is restricted to the fixed
Qwen3-0.6B-Base / WikiText-2 protocol. W3A16 isolates the effect of W3 weight
quantization, while the W3A16 → W3A8 difference measures the additional effect
of activation quantization under the same W3 weight path. In this run, weight
quantization accounts for `99.29067873343539%` of the total mean-NLL increase,
while the activation quantization increment accounts for
`0.7093212665646067%`. These observations apply only to the recorded numerical
recipe and experiment; they should not be generalized to other quantizers,
models, datasets, or hardware implementations.

## 8. Limitations

- The implementation is a CPU numerical simulation and does not model a
  particular hardware implementation.
- It does not model PTX, Tensor Core behavior, physical SMEM layout, packing,
  throughput, latency, or energy.
- FP32 scales and Lloyd-Max are implementation-specific choices for this
  reference.
- No accuracy-recovery method was evaluated.
- Results cover one local model, one dataset, one split, and one fixed token
  protocol.

## 9. Subsequent accepted work

The original scope above remains the basis of the three-mode interpretation.
The repository now also contains the separately authorized fixed H64 + GPTQ +
LUT-refit comparison and Plan A layerwise diagnostics. Their acceptance,
evidence, and limitations are recorded separately below.

## Additional quantization recipe comparison

A separately authorized two-recipe experiment now compares the original
plain Lloyd-Max recipe with H64 + K64 block GPTQ + Hessian-refitted LUTs,
under the same local model, CPU, scale format, and test protocol. See
[the experiment report](quantization_comparison.md) and
[`results/quantization_comparison/`](../results/quantization_comparison/).
This extension does not revise the historical Phase 2/3 acceptance results.

## Historical rerun and Plan A diagnostics

The [complete historical rerun](../results/historical_rerun/20261013/report.md)
executed the three original rows and all five recipe-comparison rows.
All mean-NLL and PPL deltas from the historical artifacts were exactly zero.
Rebuilt payloads matched both recorded fingerprints, and the original linear
and full-model sanity checks passed. The current engineering record contains
91 passing tests with zero skips.

[Plan A](update_plan_a.md) subsequently measured local outputs on fixed
BF16-teacher inputs, using 1,024-token smoke and 4,096-token validation stages.
Implementation, historical baseline, smoke, and experiment acceptance all
passed. B1 lowers relative output error in 170/196 layers in both A16/A8;
24 `o_proj` and two `down_proj` layers worsen, with no ties.
Relative weight reconstruction error and activation increment MSE increase
in all 196 matched layers. The [diagnostic report](layerwise_diagnostics.md)
contains projection summaries, the two heatmaps, signed A8 Teacher MSE changes,
and independent verification sources.

The local-output results concern the combined recipe under the fixed
teacher-input protocol. They do not measure layer PPL sensitivity or isolate
the effects of individual H64, GPTQ, and refitting steps.
