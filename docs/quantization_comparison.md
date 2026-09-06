# Quantization comparison: fixed K64 CPU adaptation

This experiment adds the user-authorized H64 + GPTQ + refitted LUT recipe to
plain Lloyd-Max, separately from the historical Phase 2/3 artifacts. It does
not introduce per-step ablations, K32 scales, UE8M0, AWQ, Fisher, KV cache,
additional models, or hardware/performance claims.

## Reproduce

Use the locked Python 3.14 CPU environment and the same local
`models/Qwen3-0.6B-Base` checkpoint. WikiText-2 raw train and test must already
be cached. Both Hugging Face offline flags are enforced by the evaluator;
missing data is an error, never a split/model substitution.

```bash
uv run python -m scripts.run_quantization_comparison
```

The runner executes BF16, plain W3A16/W3A8, and optimized W3A16/W3A8, using
four CPU threads for offline payload preparation and 20 CPU threads for all
five evaluations. Each complete full-model path is smoke-tested before its
32 evaluation blocks. Each recipe creates one local `.pt` payload in a
preparation stage; W3A16 and W3A8 both load that same payload. Large payloads are ignored by Git.
Existing `results/bf16.json`, `results/w3a16.json`, and `results/w3a8.json`
remain historical references. All new output is under
`results/quantization_comparison/`.

If the shared dataset cache is read-only, copy its `Salesforce___wikitext`
directory to a writable cache and set `HF_DATASETS_CACHE` to that parent.
The recorded run used `/tmp/rubin-quantization-datasets`. This only relocates
cached data/locks; evaluation and calibration token SHA256 values identify
the actual inputs.

Individual runs accept `--weight-method plain_lloyd_max|h64_gptq_refit` and
mutually exclusive `--save-payload PATH` / `--load-payload PATH` on
`scripts/eval_qwen.py`. The original default remains plain Lloyd-Max. `--prepare-only` saves the
payload and its metadata without evaluating test NLL. The runner also accepts
`--reuse-payloads` to skip preparation and validate existing local payloads. BF16
rejects payload options and the optimized method. Calibration flags accept
only `--calibration-split train --calibration-tokens 512`.

## Frozen numerical recipe

Common: N8×K64, eight E4M3 LUT values, per-row K64 FP32 max-abs/448 scales,
existing E4M3 rounding, FP32 reconstruction/accumulation, output cast to the
input dtype. Only q/k/v/o/gate/up/down projections are replaced. Embeddings,
normalization, and the tied language-model head retain their original policy.

For the optimized path, normalized signed H64 rotates both weights and
inputs on the right: `R = D H64 / 8`, `W' = W R`, `x' = x R`.
Signs use SHA256 of `20260813:module_alias`; q/k/v and gate/up respectively
share aliases. K must be divisible by 64. All rotation results remain FP32;
there is no additional BF16 rounding of the rotated weights or activations.
A8 quantization follows rotation; W3A16 uses the FP32 rotation of BF16 input.

Calibration uses the first 512 train tokens after joining text with two
newlines and tokenizing without special tokens. Hooks observe original,
unquantized BF16-model inputs, rotate them in FP32, and accumulate
`H_b = X'_b.T X'_b / 512` independently per K64 block. Shared-input modules
reuse Hessians. No test data, propagated quantization calibration, backward
pass, cross-block Hessian, or act-order is used.

Initialization reuses the project's original masked E4M3-constrained
Lloyd-Max (50 iterations maximum) and scales. Scales remain fixed thereafter.
GPTQ uses damping 0.01 times the mean live diagonal, dead-channel threshold
1e-12, and Cholesky jitter candidates 0, 1e-6, 1e-4, 1e-2. Dead working
columns become zero and their damped Hessian diagonal becomes one, following
the reference. A failed factorization raises an error.

For fixed assignments, each tile's scale-weighted design matrix solves the
eight-center Hessian least-squares normal equations with ridge
`1e-4 * max(mean(diag(normal)), 1e-8)`. Empty codes retain their old center
before sorting; E4M3 projection follows the solve. Up to three refit rounds
try interpolation factors 1, 0.5, 0.25. Each candidate runs a fresh GPTQ sweep
from the original rotated weights. The whole-layer unregularized Hessian
proxy selects candidates; accept only relative improvement >1e-5, otherwise
retain the previous payload and stop. Padded N rows never affect fitting or
the objective. Final GPTQ assignments are stored directly, not recomputed by
nearest-neighbor encoding.

## Provenance and interpretation

Reference: [supplement](https://www.zhongzhuzhou.org/verarubin-short/) and
[fixed implementation](https://github.com/FutureMLS-Lab/OSCAR/blob/658d6539153530fd63a7bd8017bec619f6604539/rubin/w3a8_lut_sim.py).
See `THIRD_PARTY_NOTICES.md` for the MIT notice.

This adapts the non-sensitivity `inter_lloymax_gptq_hadamard` family. Differences
include the local 0.6B model, original seven-projection policy, existing
FP32 K64 scales/A8 rules, existing constrained Lloyd-Max initialization,
masked padding, FP32 rotations, and strictly train-only teacher-input
calibration. It does not reproduce the source's selected native Rubin rows.

The five runs share test sequence length 1024, 32768 input tokens, 32736
predicted tokens, seed 42, and `use_cache=False`. JSON records checkpoint,
tokenizer, test-token, and payload fingerprints plus software versions.
The comparison checks common metadata and validates recipe-specific
calibration/payload identity separately. The same recipe must use identical
weights for W3A16 and W3A8.

`summary.csv` reports mean NLL/PPL and differences from BF16;
`contrasts.csv` reports optimized minus plain at matched activation precision
and A8 minus A16 within each recipe. These are descriptive differences:
PPL deltas are not additive attribution, and a two-recipe comparison cannot
attribute gains to any single optimization step. Calibration-proxy reduction
does not guarantee lower held-out PPL. No tuning uses test results.

## Thread-count diagnosis and final execution conditions

The first run used four threads for both preparation and evaluation. BF16
matched the historical result exactly, but plain W3A16 exceeded the historical
reproduction tolerance. Before accepting any comparison, the original
evaluator from repository commit `56521288eb3d200cbb4f659c7d63c8c8a88664c6`
was rerun under the same four-thread environment, and the identical saved
plain payload was separately evaluated with 20 threads.

| W3A16 execution | Threads | mean NLL | PPL |
|---|---:|---:|---:|
| Historical recorded reference | Not recorded | 3.3644417383337535 | 28.917349344449057 |
| Original evaluator, independent rerun | 4 | 3.363885829176022 | 28.901278392539698 |
| New payload evaluator | 4 | 3.363885829176022 | 28.901278392539698 |
| Same payload, final evaluation setting | 20 | 3.3644417383337535 | 28.917349344449057 |

Thus the new plain conversion reproduces the original entrypoint exactly at
matched thread count, and 20-thread evaluation reproduces the historical
W3A16 value exactly. The controlled rerun demonstrates thread-sensitive
numerical behavior; it does not establish the unrecorded thread setting of
the historical process. Toggling `config.use_cache` did not change the first
block result. No quantizer, test data, or accuracy hyperparameter was tuned.

Final evaluations all use `OMP_NUM_THREADS=20 MKL_NUM_THREADS=20`; offline
payload preparation and train Hessian collection use four threads. The
comparison rejects other evaluation thread counts. Both saved payloads were
prepared once and retained. Four-thread diagnostics, including the original
entrypoint rerun, are preserved under
`results/quantization_comparison/diagnostics/threads4/`. The four-thread optimized child process completed after its original runner
was interrupted; its result is retained only as a diagnostic. Resume validation
rejected that late row because its thread count did not match the final run,
preventing accidental inclusion in the final comparison.

The recorded final evaluation command reuses those payloads and the completed
20-thread W3A16 row:

```bash
HF_DATASETS_CACHE=/tmp/rubin-quantization-datasets \
  uv run python -m scripts.run_quantization_comparison \
  --reuse-payloads --resume-evaluations
```

`--resume-evaluations` checks each reused row's mode, recipe, and thread count;
all five rows must still pass the complete metadata/payload comparison before
summary output. The default command without these options performs fresh
four-thread preparation and all five 20-thread evaluations.

## Recorded final results

| Recipe | Activation | mean NLL | PPL |
|---|---|---:|---:|
| none | bf16 | 2.644017708966 | 14.069617833591 |
| plain_lloyd_max | w3a16 | 3.364441738334 | 28.917349344449 |
| plain_lloyd_max | w3a8 | 3.369588365303 | 29.066559790366 |
| h64_gptq_refit | w3a16 | 3.084551277049 | 21.857656609021 |
| h64_gptq_refit | w3a8 | 3.101196535172 | 22.224527807600 |

Compared with plain Lloyd-Max, the combined recipe reduces PPL by 24.41% in W3A16 and 23.54% in W3A8.
The matched-recipe NLL differences (optimized minus plain) are -0.279890461285 and -0.268391830131, respectively.

The A8-minus-A16 NLL increment is 0.005146626969 for plain and 0.016645258123 for the combined recipe. Thus this run's activation increment is larger after the recipe change, while its overall W3A8 PPL remains lower. The comparison supports a benefit for the combined recipe on this fixed model/protocol, not attribution to individual H64/GPTQ/refit steps or a general Rubin hardware conclusion.

All 196 optimized layers have finite, strictly decreasing accepted proxy traces. Accepted refit rounds: 194 layers accepted three, one accepted two, and one accepted none. This proxy check is separate from the held-out PPL result.

## Acceptance

| Check | Result | Evidence |
|---|---|---|
| Regression and numerical tests | PASS — 42 tests | `tests.log` |
| Five full-model smoke checks | PASS | Each result JSON and run log |
| Common metadata and PPL/NLL consistency | PASS | `comparison.log` |
| Paired W3A16/W3A8 payload identity | PASS for both recipes | Result and quantization JSON fingerprints |
| Historical BF16/W3A16/W3A8 reproduction | PASS — all deltas exactly zero | `baseline_reproduction.json` |
| Source and environment provenance | PASS | `source_manifest.json`, `environment.json` |

Artifact-level and engineering acceptance: PASS for this CPU adaptation and the frozen two-recipe comparison. No test-based hyperparameter tuning was performed. The observed thread sensitivity is resolved by the documented 20-thread evaluation setting; reproducibility claims concern this recorded CPU/runtime setup.

All evidence paths above are relative to `results/quantization_comparison/`. The source manifest identifies the modified working-tree code in addition to the base Git commit recorded by the evaluator. Diagnostic runs are separated from the five final rows; expected rejection traces in `run.log` precede the final successful resumed run.
