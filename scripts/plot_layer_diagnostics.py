"""Export Plan A heatmaps from verified frozen CSV tables."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from scripts.diagnostic_protocol import METHODS, PROJECTIONS, read_verified_tables, validate_canonical_tables


def render_figures(layers, directory):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "rubin-plan-a-matplotlib"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    by_key = {(row["transformer_block"], row["projection"], row["weight_method"]): row for row in layers}
    if len(by_key) != 392:
        raise ValueError("Figures require complete 196-layer paired recipe coverage")
    matrices = {
        (method, mode): np.array([[by_key[block, projection, method][f"relative_error_{mode}"]
                                  for projection in PROJECTIONS] for block in range(28)])
        for method in METHODS for mode in ("a16", "a8")
    }
    if not all(np.isfinite(matrix).all() for matrix in matrices.values()):
        raise ValueError("Cannot plot nonfinite errors")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    palette = dict(font="DejaVu Sans", font_size=8)
    with plt.rc_context({"font.family": palette["font"], "font.size": palette["font_size"]}):
        fig, axes = plt.subplots(2, 2, figsize=(11, 13), layout="constrained")
        maximum = max(max(float(matrix.max()) for matrix in matrices.values()), 1e-12)
        for i, mode in enumerate(("a16", "a8")):
            for j, method in enumerate(METHODS):
                ax = axes[i, j]
                im = ax.imshow(matrices[method, mode], vmin=0, vmax=maximum, cmap="viridis", aspect="auto")
                ax.set_title(f"{'B0' if j == 0 else 'B1'} — W3{mode.upper()}")
                ax.set_xticks(range(7), PROJECTIONS, rotation=45, ha="right")
                ax.set_yticks(range(28))
                ax.set_ylabel("Transformer block")
        fig.colorbar(im, ax=axes.ravel().tolist(), label="Relative output error")
        fig.suptitle("Isolated Linear Output Error\nBF16 Teacher inputs; shared color scale", fontsize=12)
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"figure_a1_relative_output_error.{extension}", dpi=300)
        plt.close(fig)

        differences = {mode: np.array([[by_key[block, projection, METHODS[1]][f"delta_relative_error_{mode}"]
                                       for projection in PROJECTIONS] for block in range(28)])
                       for mode in ("a16", "a8")}
        maximum = max(max(float(np.abs(matrix).max()) for matrix in differences.values()), 1e-12)
        fig, axes = plt.subplots(1, 2, figsize=(11, 8), layout="constrained")
        for ax, mode in zip(axes, ("a16", "a8")):
            im = ax.imshow(differences[mode], vmin=-maximum, vmax=maximum, cmap="RdBu_r", aspect="auto")
            rows = [row for row in layers if row["weight_method"] == METHODS[1]]
            counts = {kind: sum(row[f"comparison_{mode}"] == kind for row in rows)
                      for kind in ("improved", "worsened", "tied")}
            ax.set_title(f"W3{mode.upper()}\nImproved {counts['improved']} / Worsened {counts['worsened']} / Tied {counts['tied']}")
            ax.set_xticks(range(7), PROJECTIONS, rotation=45, ha="right")
            ax.set_yticks(range(28))
            ax.set_ylabel("Transformer block")
        fig.colorbar(im, ax=axes.tolist(), label="B1 − B0 relative output error")
        fig.suptitle("Per-layer Error Change\nRaw differences; fixed numerical classification threshold", fontsize=12)
        for extension in ("png", "pdf"):
            fig.savefig(directory / f"figure_a2_error_change.{extension}", dpi=300)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads((args.input_dir / "manifest.json").read_text())
    stage = manifest.get("active_stage")
    status = manifest.get("smoke_acceptance" if stage == "smoke" else "experiment_acceptance")
    if stage not in ("smoke", "full") or status != "PASS":
        parser.error("The selected stage has not passed its experiment checks")
    tables = read_verified_tables(args.input_dir)
    fingerprints = {method: item["actual_fingerprint"] for method, item in manifest["verified_payloads"].items()}
    validate_canonical_tables(tables, 1024 if stage == "smoke" else 4096, fingerprints)
    render_figures(tables["layer_errors.csv"], args.input_dir)


if __name__ == "__main__":
    main()
