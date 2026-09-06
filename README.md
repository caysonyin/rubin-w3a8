# Rubin LUT-W3A8 CPU Numerical Reference

This repository implements the frozen Phase 2 CPU reference for an
ISA-consistent N8 x K64 logical LUT-W3A8 organization and validates it on the
local Qwen3-0.6B-Base model.

The numerical definition is documented in
[`docs/numerical_spec_v1.md`](docs/numerical_spec_v1.md). The measured result
and limitations are in [`docs/final_report.md`](docs/final_report.md).

## Run

```bash
uv sync
uv run pytest -q
uv run python scripts/sanity_linear.py
uv run python scripts/sanity_block.py --model-path models/Qwen3-0.6B-Base
```

The final evaluation uses CPU only and the fixed WikiText-2 protocol from the
plan:

```bash
uv run python scripts/eval_qwen.py --mode bf16 \
  --model-path models/Qwen3-0.6B-Base --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 --split test --seq-len 1024 \
  --max-eval-tokens 32768 --device cpu --seed 42 \
  --output results/bf16.json

uv run python scripts/eval_qwen.py --mode w3a8 \
  --model-path models/Qwen3-0.6B-Base --dataset Salesforce/wikitext \
  --dataset-config wikitext-2-raw-v1 --split test --seq-len 1024 \
  --max-eval-tokens 32768 --device cpu --seed 42 \
  --output results/w3a8.json

uv run python scripts/compare_results.py results/bf16.json results/w3a8.json \
  --output results/summary.csv
```

The project pins PyTorch to its CPU wheel source through `uv`; no CUDA, PTX,
Triton, or custom extension is used.

## Acceptance evidence

The independent Gate A-D evidence is included in the repository:

- Numerical specification: [`docs/numerical_spec_v1.md`](docs/numerical_spec_v1.md)
- Source implementation: [`src/rubin_w3a8/`](src/rubin_w3a8/)
- Tests: [`tests/`](tests/)
- Pytest evidence: [`results/pytest.log`](results/pytest.log)
- Linear correctness evidence: [`results/sanity_linear.log`](results/sanity_linear.log)
- Full Qwen forward evidence: [`results/sanity_block.log`](results/sanity_block.log)
- Final PPL evidence: [`docs/final_report.md`](docs/final_report.md) and [`results/`](results/)

The 1.2 GB model weight file is intentionally not committed to Git; the local
model directory is ignored and must be supplied separately at
`models/Qwen3-0.6B-Base`.
