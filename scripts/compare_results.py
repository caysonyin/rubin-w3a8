"""Compare BF16 and W3A8 result JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bf16_json", type=Path)
    parser.add_argument("w3a8_json", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    bf16 = json.loads(args.bf16_json.read_text(encoding="utf-8"))
    w3a8 = json.loads(args.w3a8_json.read_text(encoding="utf-8"))
    baseline = float(bf16["ppl"])
    quantized = float(w3a8["ppl"])
    delta = quantized - baseline
    relative = delta / baseline * 100.0
    rows = [
        {"mode": "BF16", "ppl": baseline, "delta_ppl": "", "delta_ppl_percent": ""},
        {"mode": "W3A8", "ppl": quantized, "delta_ppl": delta, "delta_ppl_percent": relative},
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"BF16 PPL={baseline:.8f}")
    print(f"W3A8 PPL={quantized:.8f} delta={delta:.8f} delta_percent={relative:.4f}%")


if __name__ == "__main__":
    main()
