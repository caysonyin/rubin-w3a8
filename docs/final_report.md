# Phase 2 Final Report

## 1. Numerical semantics

Weights remain in PyTorch `[N, K]` orientation. Each logical `N8 x K64` tile
has eight row-wise FP32 K64 scales and one shared eight-entry E4M3 LUT. Each
weight stores a uint8 logical index in `[0, 7]`; reconstruction is
`scale[row] * LUT[index]`. Padding to multiples of 8 and 64 is excluded from
the fitting objective and cropped after reconstruction.

Activations use one FP32 max-absolute scale for each activation row x K64
block, E4M3 fake quantization, and scale restoration. Linear accumulation is
FP32 and the result is cast to the incoming hidden-state dtype, normally BF16.

## 2. Quantizer

The project baseline is FP32 K64 max-absolute scaling plus
E4M3-constrained Lloyd–Max. It is a project-defined CPU numerical recipe, not
NVIDIA's official quantizer. Full-model weights are stored in tensorized
`LUTWeight` form and reconstructed once into a cached dense weight. The
explicit index-to-LUT path is reserved for correctness tests.

## 3. Correctness

All unit tests pass, including E4M3 reference validation when the installed CPU
PyTorch build supports the FP8 cast, non-multiple padding, zero tiles,
Lloyd–Max fitting, lookup-vs-dense reconstruction, cached Linear behavior, and
eligible Qwen module replacement.

The fixed linear sanity suite covers `M = 1, 4`, `N = 8, 16, 1024`,
`K = 64, 128, 1024`, plus padding. Full Qwen CPU forward completed with finite
logits of shape `(1, 12, 151936)` and BF16 output dtype.

## 4. Qwen accuracy

Evaluation used the local `models/Qwen3-0.6B-Base`, CPU, seed 42, WikiText-2
test text joined with two newlines, one tokenization, 32768 tokens, and 1024
token blocks. Each block predicts 1023 next tokens, for 32736 total predicted
tokens.

| Mode | PPL | ΔPPL | ΔPPL % |
|---|---:|---:|---:|
| BF16 | 14.0696178336 | — | — |
| W3A8 | 29.0665597904 | 14.9969419568 | 106.5909688106% |

The machine-readable results are [`results/bf16.json`](../results/bf16.json),
[`results/w3a8.json`](../results/w3a8.json), and
[`results/summary.csv`](../results/summary.csv).

## 5. Limitations

- This is a CPU numerical simulation and does not implement Rubin hardware.
- Runtime measurements are experiment metadata, not Rubin performance claims.
- FP32 scales and Lloyd–Max are project-defined approximations.
- Lloyd–Max is not NVIDIA's official quantization recipe.
- No accuracy-recovery methods such as GPTQ, Hadamard, AWQ, or layer
  sensitivity were implemented.
- Therefore, the PPL penalty alone cannot establish that Rubin LUT-W3 itself
  is unsuitable.

Phase 2 Core stops here as specified.
