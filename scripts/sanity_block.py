"""Load and run the local Qwen model through the converted CPU path."""

from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from rubin_w3a8 import convert_qwen_to_w3a8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="models/Qwen3-0.6B-Base")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(args.device).eval()
    model = convert_qwen_to_w3a8(model)
    inputs = tokenizer("Rubin LUT-W3A8 CPU sanity test.", return_tensors="pt").to(args.device)
    with torch.inference_mode():
        outputs = model(**inputs, use_cache=False)
    assert outputs.logits.ndim == 3
    assert outputs.logits.shape[:2] == inputs.input_ids.shape
    assert torch.isfinite(outputs.logits.float()).all()
    print(f"sanity_block: PASS logits_shape={tuple(outputs.logits.shape)} dtype={outputs.logits.dtype}")


if __name__ == "__main__":
    main()
