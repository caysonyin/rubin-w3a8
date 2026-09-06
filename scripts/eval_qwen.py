"""Run the frozen BF16 or LUT-W3A8 WikiText-2 perplexity protocol."""

from __future__ import annotations

import argparse
import json
import math
import platform
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.qwen import convert_qwen_to_w3a8


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _commit_sha() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip()


def _load_tokens(args: argparse.Namespace, tokenizer) -> torch.Tensor:
    dataset = load_dataset(args.dataset, args.dataset_config, split=args.split)
    full_text = "\n\n".join(dataset["text"])
    tokenized = tokenizer(full_text, add_special_tokens=False)
    tokens = torch.tensor(tokenized["input_ids"], dtype=torch.long)
    tokens = tokens[: args.max_eval_tokens]
    return tokens[: (tokens.numel() // args.seq_len) * args.seq_len]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("bf16", "w3a8"), required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--dataset", default="Salesforce/wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--split", default="test")
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--max-eval-tokens", type=int, default=32768)
    parser.add_argument(
        "--device",
        choices=("cpu",),
        default="cpu",
        help="Canonical evaluation is CPU-only.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _set_seed(args.seed)
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(args.device).eval()
    if args.mode == "w3a8":
        model = convert_qwen_to_w3a8(model)
    tokens = _load_tokens(args, tokenizer)
    num_blocks = tokens.numel() // args.seq_len
    total_nll = 0.0
    total_predicted = 0
    with torch.inference_mode():
        for block_index in range(num_blocks):
            start = block_index * args.seq_len
            block = tokens[start : start + args.seq_len].unsqueeze(0).to(args.device)
            outputs = model(input_ids=block, use_cache=False)
            logits = outputs.logits[:, :-1, :].float()
            targets = block[:, 1:]
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
                reduction="sum",
            )
            total_nll += float(nll)
            total_predicted += targets.numel()
    mean_nll = total_nll / total_predicted if total_predicted else float("nan")
    ppl = math.exp(mean_nll) if math.isfinite(mean_nll) else float("nan")
    result = {
        "spec_version": "reference-v1",
        "model_path": str(Path(args.model_path).expanduser()),
        "mode": args.mode,
        "device": args.device,
        "dataset": args.dataset,
        "dataset_config": args.dataset_config,
        "split": args.split,
        "seq_len": args.seq_len,
        "num_eval_tokens": int(tokens.numel()),
        "num_predicted_tokens": int(total_predicted),
        "seed": args.seed,
        "ppl": ppl,
        "mean_nll": mean_nll,
        "runtime_seconds": time.perf_counter() - started,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "commit_sha": _commit_sha(),
    }
    if not math.isfinite(ppl):
        raise RuntimeError(f"non-finite PPL result: {result}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
