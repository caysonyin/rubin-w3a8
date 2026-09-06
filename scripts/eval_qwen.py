"""Run the frozen BF16, LUT-W3A16, or LUT-W3A8 WikiText-2 protocol."""

from __future__ import annotations

import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"

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

from src.experiment import (METHODS, FORMAT, CONFIG, files_hash, tensor_hash,
    calibration_metadata, build_payload, apply_payload)


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
    parser.add_argument("--mode", choices=("bf16", "w3a16", "w3a8"), required=True)
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
    parser.add_argument("--weight-method", choices=METHODS, default=METHODS[0])
    parser.add_argument("--calibration-split", choices=("train",), default="train")
    parser.add_argument("--calibration-tokens", type=int, choices=(512,), default=512)
    payload_group = parser.add_mutually_exclusive_group()
    payload_group.add_argument("--save-payload", type=Path)
    payload_group.add_argument("--load-payload", type=Path)
    parser.add_argument("--prepare-only", action="store_true", help="Save a quantization payload without evaluating test NLL")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.prepare_only and (args.mode == "bf16" or args.save_payload is None):
        raise ValueError("--prepare-only requires a quantized mode and --save-payload")
    _set_seed(args.seed)
    started = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
    ).to(args.device).eval()
    if args.seq_len < 2 or args.max_eval_tokens < args.seq_len:
        raise ValueError("Evaluation needs at least one block of length >= 2")
    if args.mode == "bf16" and (args.save_payload or args.load_payload or args.weight_method != METHODS[0]):
        raise ValueError("BF16 does not accept quantization payloads or an optimized recipe")
    model_root = Path(args.model_path).expanduser()
    model_files = list(model_root.glob("*.safetensors")) + [model_root / "config.json"]
    if len(model_files) < 2:
        raise ValueError("Local safetensors model missing")
    model_fingerprint = files_hash(model_files)
    tokenizer_fingerprint = files_hash(list(model_root.glob("tokenizer*")) + list(model_root.glob("vocab*")) + list(model_root.glob("merges*")) + list(model_root.glob("special_tokens*")))
    tokens = _load_tokens(args, tokenizer)
    calibration = None
    calibration_ids = None
    if args.mode != "bf16" and args.weight_method == METHODS[1]:
        if args.split != "test" or args.dataset != "Salesforce/wikitext" or args.dataset_config != "wikitext-2-raw-v1":
            raise ValueError("Optimized comparison requires WikiText-2 test evaluation and train calibration")
        dataset = load_dataset(args.dataset, args.dataset_config, split=args.calibration_split)
        calibration_ids = torch.tensor(tokenizer("\n\n".join(dataset["text"]), add_special_tokens=False)["input_ids"][:512], dtype=torch.long)
        calibration = calibration_metadata(calibration_ids, split=args.calibration_split)
    metadata = {'model_fingerprint': model_fingerprint, 'tokenizer_fingerprint': tokenizer_fingerprint,
                'weight_method': args.weight_method, 'format': FORMAT, 'config': CONFIG,
                'calibration': calibration}
    payload_fingerprint = None
    if args.mode != "bf16":
        if args.load_payload:
            payload = torch.load(args.load_payload, map_location="cpu", weights_only=True)
        else:
            payload = build_payload(model, metadata, calibration_ids)
        model = apply_payload(model, payload, metadata, quantize_activations=args.mode == "w3a8")
        payload_fingerprint = payload['fingerprint']
        if args.save_payload:
            args.save_payload.parent.mkdir(parents=True, exist_ok=True)
            torch.save(payload, args.save_payload)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.with_suffix('.quantization.json').write_text(json.dumps(
            {'metadata': metadata, 'payload_fingerprint': payload_fingerprint, 'traces': payload['traces']}, indent=2) + "\n")
        del payload
    if args.prepare_only:
        print("payload_preparation: PASS", flush=True)
        return
    # Exercise the exact loaded full-model path before the measured evaluation.
    with torch.inference_mode():
        smoke = model(input_ids=tokens[:16].unsqueeze(0), use_cache=False).logits
        if smoke.shape[:2] != (1, min(16, tokens.numel())) or not torch.isfinite(smoke).all():
            raise ValueError("Full model smoke failed")
    print("full_model_smoke: PASS", flush=True)
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
            print(f"block={block_index + 1}/{num_blocks} mean_nll={total_nll / total_predicted:.8f}", flush=True)
    mean_nll = total_nll / total_predicted if total_predicted else float("nan")
    ppl = math.exp(mean_nll) if math.isfinite(mean_nll) else float("nan")
    result = {
        "spec_version": "reference-v1",
        "experiment_version": "quantization-comparison-v1",
        "weight_method": args.weight_method if args.mode != "bf16" else "none",
        "model_fingerprint": model_fingerprint,
        "tokenizer_fingerprint": tokenizer_fingerprint,
        "eval_token_hash": tensor_hash(tokens),
        "payload_fingerprint": payload_fingerprint,
        "quantization_format": FORMAT,
        "recipe_config": CONFIG,
        "calibration": calibration,
        "full_model_smoke": "PASS",
        "cpu_threads": torch.get_num_threads(),
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
