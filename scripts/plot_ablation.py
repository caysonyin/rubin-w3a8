"""Render Plan B results without model forwards."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import tempfile

from scripts.ablation_protocol import METHODS, IDS, SCHEMA_VERSION, EDGES
from scripts.diagnostic_protocol import PROJECTIONS, read_verified_tables


def render(directory, summary, increments, tables, inputs, steps, bf16):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rubin-plan-b-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    directory = Path(directory)
    colors = ("#59636f", "#327bb2", "#df9441", "#268575")
    names = [IDS[m] for m in METHODS]
    def save(fig, stem):
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"{stem}.{extension}", dpi=300)
        plt.close(fig)

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 9}):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
        for mode, marker in (("w3a16", "o"), ("w3a8", "s")):
            values = [next(r["ppl"] for r in summary if r["weight_method"] == m and r["mode"] == mode) for m in METHODS]
            axes[0].plot(names, values, marker=marker, label=mode.upper())
            for x, y in zip(names, values):
                axes[0].annotate(f"{y:.3f}", (x, y), xytext=(0, -14 if mode == "w3a16" else 7),
                                 textcoords="offset points", ha="center", fontsize=8)
        axes[0].axhline(bf16["ppl"], color="#777777", linestyle="--", label=f"BF16 {bf16['ppl']:.3f}")
        axes[0].set_ylabel("Test perplexity (lower is better)")
        axes[0].set_title("Four frozen recipes")
        axes[0].legend()
        x = np.arange(3)
        for offset, mode in ((-.18, "w3a16"), (.18, "w3a8")):
            values = [r["delta_mean_nll"] for r in increments if r["mode"] == mode]
            axes[1].bar(x + offset, values, width=.36, label=mode.upper())
        axes[1].set_xticks(x, ("B0→A1\nH64", "A1→A2\nGPTQ | H64", "A2→B1\nRefit | H64+GPTQ"))
        axes[1].axhline(0, color="#777777", linewidth=.8)
        axes[1].set_ylabel("Conditional ΔMean NLL (after − before)")
        axes[1].set_title("Fixed-order conditional increments")
        axes[1].legend()
        fig.suptitle("Plan B — Qwen3-0.6B-Base / WikiText-2 test", fontsize=13)
        save(fig, "figure_b1_test_ablation")

        matrices = {}
        for contrast in (edge[2] for edge in EDGES):
            lookup = {(r["transformer_block"], r["projection"]): r for r in steps if r["contrast"] == contrast}
            for mode in ("a16", "a8"):
                matrices[contrast, mode] = np.array([[lookup[b, p][f"delta_relative_error_{mode}"] for p in PROJECTIONS] for b in range(28)])
        maximum = max(max(float(np.abs(v).max()) for v in matrices.values()), 1e-12)
        fig, axes = plt.subplots(2, 3, figsize=(15, 12), layout="constrained")
        for col, (before, after, contrast) in enumerate(EDGES):
            rows = [r for r in steps if r["contrast"] == contrast]
            for row_index, mode in enumerate(("a16", "a8")):
                ax = axes[row_index, col]
                im = ax.imshow(matrices[contrast, mode], aspect="auto", cmap="RdBu_r", vmin=-maximum, vmax=maximum)
                counts = [sum(r[f"comparison_{mode}"] == k for r in rows) for k in ("improved", "worsened", "tied")]
                ax.set_title(f"{IDS[before]}→{IDS[after]} — W3{mode.upper()}\nImproved {counts[0]} / Worsened {counts[1]} / Tied {counts[2]}")
                ax.set_xticks(range(7), PROJECTIONS, rotation=45, ha="right")
                ax.set_yticks(range(28))
                ax.set_ylabel("Transformer block")
        fig.colorbar(im, ax=axes.ravel().tolist(), label="After − before relative output error")
        fig.suptitle("Conditional local output error changes\nFixed BF16 Teacher inputs; common symmetric scale", fontsize=13)
        save(fig, "figure_b2_conditional_output_error")

        fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
        for ax, mode in zip(axes[0], ("a16", "a8")):
            for method, color in zip(METHODS, colors):
                rows = sorted((r for r in tables["layer_errors.csv"] if r["projection"] == "o_proj" and r["weight_method"] == method),
                              key=lambda r: r["transformer_block"])
                ax.plot(range(28), [r[f"relative_error_{mode}"] for r in rows], label=IDS[method], color=color)
            ax.set_title(f"o_proj — W3{mode.upper()}")
            ax.set_xlabel("Transformer block")
            ax.set_ylabel("Relative output error")
            ax.legend(ncol=4)
        for method, color in zip(METHODS, colors):
            values = [next(r["activation_increment_mse"] for r in tables["projection_summary.csv"]
                           if r["projection"] == p and r["weight_method"] == method) for p in PROJECTIONS]
            axes[1, 0].plot(range(7), values, marker="o", label=IDS[method], color=color)
        axes[1, 0].set_yscale("log")
        axes[1, 0].set_ylabel("Pooled ||Y_A8 − Y_A16||² / N")
        axes[1, 0].set_title("Output activation increment (log scale)")
        axes[1, 0].legend(ncol=4)
        for basis, label in (("plain", "Original input"), ("h64", "H64 input")):
            values = []
            for projection in PROJECTIONS:
                rows = [r for r in inputs if r["basis"] == basis and r["projection"] == projection]
                values.append((sum(r["input_sse"] for r in rows) / (sum(r["input_energy"] for r in rows) + 1e-12)) ** .5)
            axes[1, 1].plot(range(7), values, marker="o", label=label)
        axes[1, 1].set_ylabel("Pooled relative A8 input error")
        axes[1, 1].set_title("Input perturbation before applying W3 weights")
        axes[1, 1].legend()
        for ax in axes[1]:
            ax.set_xticks(range(7), PROJECTIONS, rotation=35, ha="right")
        fig.suptitle("o_proj and A8 perturbation diagnostics — validation", fontsize=13)
        save(fig, "figure_b3_o_proj_and_a8")


def load_numeric(path):
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in row.items():
            if key.startswith(("delta_", "before_relative_", "after_relative_", "tolerance_", "input_", "num_", "relative_input_")) or key in ("ppl", "mean_nll", "ppl_change_percent", "transformer_block"):
                row[key] = float(value)
        if "transformer_block" in row:
            row["transformer_block"] = int(row["transformer_block"])
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    directory = args.input_dir
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest.get("experiment_acceptance") != "PASS":
        parser.error("Plan B has not passed experiment acceptance")
    tables = read_verified_tables(directory / "full", methods=METHODS, schema_version=SCHEMA_VERSION)
    bf16 = json.loads((Path(manifest["bf16_reference"]["path"])).read_text())
    render(directory, load_numeric(directory / "summary.csv"), load_numeric(directory / "conditional_increments.csv"),
           tables, load_numeric(directory / "full/input_errors.csv"), load_numeric(directory / "full/layer_increments.csv"), bf16)


if __name__ == "__main__":
    main()
