# Rubin LUT-W3A8 CPU Numerical Reference — Final Report

## 1. Project scope and contents

This repository contains a CPU numerical reference for a LUT-based W3A8
configuration, together with its numerical specification, tests,
reproducibility checks, and recorded Qwen evaluation results.

The scope is deliberately limited to a CPU numerical reference. It does not
model any particular hardware implementation or benchmark hardware performance.

## 2. Rubin LUT-W3A8 numerical semantics

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
    -> fake-quantized activations
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
`uv sync --locked`. Both modes tokenize the same joined WikiText-2 text and
use summed FP32 cross-entropy to compute mean NLL and PPL.

## 6. Results

The recorded evaluation produced the following reference results:

| Mode | mean NLL | PPL | ΔPPL | ΔPPL % |
|---|---:|---:|---:|---:|
| BF16 | 2.64401770896576 | 14.069617833591494 | — | — |
| LUT-W3A8 | 3.3695883653031067 | 29.066559790365766 | 14.996941956774272 | 106.59096881060104% |

The machine-readable results are [`results/bf16.json`](../results/bf16.json),
[`results/w3a8.json`](../results/w3a8.json), and
[`results/summary.csv`](../results/summary.csv). Runtime is
recorded as execution metadata only; it is not a Rubin performance comparison.

The BF16 result agrees with the historical reference PPL `14.0696178336`
within the specified tolerance. The W3A8 result agrees with the historical
reference PPL `29.0665597904` within the specified tolerance. The result
comparison first checks the shared experiment metadata and verifies
`PPL = exp(mean NLL)` within the configured numerical tolerance.

## 7. Interpretation

In this fixed run, the W3A8 configuration reports a higher PPL than the BF16
configuration on the Qwen3-0.6B-Base / WikiText-2 protocol. This observation
applies to the recorded numerical recipe and experiment; it should not be
generalized to other quantizers, models, datasets, or hardware
implementations.

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

## 9. Future work

Possible extensions include W3A16/W16A8 diagnostics, layer-sensitivity
definitions, RMS²-weighted Lloyd fitting, Hadamard transforms, GPTQ, LUT
refitting, K32 studies, mixed precision, and a native Rubin kernel. These are
not part of the current reference implementation.
