# Rubin LUT-W3A8 CPU Numerical Specification v1

## Scope and evidence boundary

This project is a CPU-only numerical reference for a Rubin-style LUT-W3A8
model. Public ISA constraints motivate the logical organization: 3-bit Matrix
B indices select one of eight E4M3 values, and the CPU reference uses an
N8 x K64 logical tile. This organization is an ISA-consistent logical tile;
it is not a claim about NVIDIA's undisclosed production quantizer or physical
SMEM layout.

The project-defined parts are the FP32 scale representation, max-absolute
scaling, padding, E4M3 software emulator, Lloyd initialization, and the Qwen
evaluation protocol. Lloyd-Max and the scale recipe are not NVIDIA's official
quantizer specification.

Out of scope: W3A16, W16A8, GPTQ, AWQ, Hadamard, Fisher, RMS2, K32, mixed
precision, physical 3-bit packing, CUDA, PTX, Triton, C++ extensions,
TensorRT, K3V3, MoE, Qwen3-4B, and performance claims.

## Weight orientation and logical tile

All offline weight APIs preserve PyTorch `nn.Linear.weight` orientation:

\[
W \in \mathbb{R}^{N\times K},\qquad Y=XW^T.
\]

A logical tile is `W[n:n+8, k:k+64]`, with shape `[8, 64]`. The quantizer
never transposes a weight internally. Dimensions are padded to multiples of 8
and 64 with zero, but padded values never enter a fitting objective and the
reconstructed weight is cropped to the original shape.

## Weight scale and LUT fitting

For each output row and K64 block:

\[
s_B[n,h] = \max_{k\in K_h}|W[n,k]|/448.
\]

An all-zero block uses `s_B = 1`. Scales are FP32. Each of the eight rows in
an N8 x K64 tile is normalized by its own scale. The resulting 512 valid
normalized values share one eight-entry LUT. Every LUT value is constrained to
the E4M3 numerical grid.

LUT fitting minimizes squared error over valid values only:

\[
\min_{C,Z}\sum_{i\in valid}(w_i-C_{z_i})^2,
\quad z_i\in\{0,\ldots,7\},\quad C_j\in E4M3.
\]

Initialization uses the quantile-bin midpoints
`[0.0625, 0.1875, ..., 0.9375]`, computed per tile and immediately snapped to
E4M3. Each iteration assigns to the nearest centroid, updates non-empty
clusters by their mean, snaps the means to E4M3, and then reassigns. Empty
clusters retain their previous centroid; duplicate snapped centroids are valid.
The iteration limit is 50 and assignments are checked for stability.

The full representation is tensorized:

```text
indices: uint8  [N_pad, K_pad]
luts:    float32 [N_tiles, K_tiles, 8]
scales:  float32 [N_tiles, K_tiles, 8]
```

## E4M3 emulator

`quantize_e4m3` accepts FP32 or BF16 tensors and returns FP32 values on the
E4M3 numerical grid. It models sign, normal numbers, subnormals, zero,
round-to-nearest-even, and finite saturation. The finite range is `[-448,448]`;
values outside it saturate to the corresponding endpoint. Where the installed
PyTorch CPU build supports `torch.float8_e4m3fn`, tests compare the emulator
against an in-range cast reference. Overflow behavior is tested separately.

## Activation scaling

For an input tensor with arbitrary leading dimensions, flatten leading
dimensions to `[M, K]`. For each row and K64 block use:

\[
s_A[m,h]=\max_{k\in K_h}|A[m,k]|/448,
\]

with `s_A = 1` for an all-zero block. Normalize, E4M3-quantize, restore the
FP32 scale, and crop away K padding. The returned activation is FP32.

## Linear and Qwen paths

The numerical linear model is FP32 accumulation of fake-quantized activations
and reconstructed weights plus FP32 bias. The result is cast back to the
incoming input dtype, normally BF16. Bias is not W3-quantized.

The explicit index-to-LUT path is a correctness oracle for small matrices. The
Qwen path reconstructs each quantized weight once and caches a dense FP32
weight. Every forward only fake-quantizes activations and calls `F.linear`
with the cached weight; it never performs Python-level per-forward LUT lookup.
Only `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
`down_proj` inside transformer layers are replaced. Embeddings, RMSNorm, RoPE,
softmax, and `lm_head` remain high precision.

## Evaluation and limitations

The frozen model is the local `models/Qwen3-0.6B-Base` and all final runs use
CPU. WikiText-2 test text entries are joined with `"\\n\\n"`, tokenized once
without special tokens, truncated to 32768 tokens, and cropped to complete
1024-token blocks. Each block predicts 1023 next tokens with `use_cache=False`.
BF16 and W3A8 use the same sequence and FP32 cross-entropy metric.

This is a numerical CPU simulation, not a Rubin hardware implementation and
not a performance benchmark. FP32 scales and the Lloyd-Max recipe are
project-defined approximations. No accuracy-recovery method from supplemental
materials is implemented, so a PPL penalty cannot by itself establish that
Rubin LUT-W3 is unsuitable.
