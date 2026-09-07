"""Render publication-style PNG figures from the recorded comparison results.

The renderer uses a small PostScript backend so the repository does not need a
plotting dependency for the CPU reference itself; Ghostscript is used only
for the final PNG export.
"""

from __future__ import annotations

import csv
import math
import shutil
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "results" / "quantization_comparison" / "summary.csv"
OUTPUT_DIR = ROOT / "results" / "figures"

# Restrained research-figure palette: charcoal, quiet gray, and one blue root.
INK = (0.12, 0.15, 0.18)
MID = (0.36, 0.40, 0.44)
LIGHT = (0.82, 0.84, 0.86)
PALE = (0.94, 0.95, 0.96)
ACCENT = (0.16, 0.37, 0.50)
WHITE = (1.0, 1.0, 1.0)


def load_rows() -> list[dict[str, str]]:
    with SUMMARY_PATH.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    expected = [
        ("none", "bf16"),
        ("plain_lloyd_max", "w3a16"),
        ("plain_lloyd_max", "w3a8"),
        ("h64_gptq_refit", "w3a16"),
        ("h64_gptq_refit", "w3a8"),
    ]
    actual = [(row["weight_method"], row["mode"]) for row in rows]
    if actual != expected:
        raise ValueError(f"Unexpected rows in {SUMMARY_PATH}: {actual}")
    return rows


def value(rows: list[dict[str, str]], method: str, mode: str, field: str) -> float:
    for row in rows:
        if row["weight_method"] == method and row["mode"] == mode:
            return float(row[field])
    raise KeyError((method, mode, field))


def rgb(color: tuple[float, float, float]) -> str:
    return " ".join(f"{component:.4f}" for component in color)


def ps_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class Figure:
    """Small top-left-coordinate PostScript drawing helper."""

    def __init__(self, width: float, height: float):
        self.width = width
        self.height = height
        self.commands = [
            "%!PS-Adobe-3.0 EPSF-3.0",
            f"%%BoundingBox: 0 0 {int(width)} {int(height)}",
            "%%Pages: 1",
            "%%EndComments",
            "1 setlinejoin 1 setlinecap",
        ]

    def y(self, top: float) -> float:
        return self.height - top

    def add(self, command: str) -> None:
        self.commands.append(command)

    def fill_page(self) -> None:
        self.rect(0, 0, self.width, self.height, fill=WHITE, stroke=None)

    def line(self, x1: float, y1: float, x2: float, y2: float,
             color: tuple[float, float, float] = INK, width: float = 0.8,
             dash: tuple[float, float] | None = None) -> None:
        dash_cmd = "[] 0 setdash" if dash is None else f"[{dash[0]} {dash[1]}] 0 setdash"
        self.add(
            f"{rgb(color)} setrgbcolor {width:.2f} setlinewidth {dash_cmd} "
            f"newpath {x1:.2f} {self.y(y1):.2f} moveto {x2:.2f} {self.y(y2):.2f} lineto stroke"
        )

    def arrow(self, x1: float, y1: float, x2: float, y2: float,
              color: tuple[float, float, float] = INK, width: float = 0.9) -> None:
        self.line(x1, y1, x2, y2, color, width)
        angle = math.atan2(y2 - y1, x2 - x1)
        head = 5.0
        wing = 2.3
        p1 = (x2 - head * math.cos(angle) + wing * math.sin(angle),
              y2 - head * math.sin(angle) - wing * math.cos(angle))
        p2 = (x2 - head * math.cos(angle) - wing * math.sin(angle),
              y2 - head * math.sin(angle) + wing * math.cos(angle))
        self.line(p1[0], p1[1], x2, y2, color, width)
        self.line(p2[0], p2[1], x2, y2, color, width)

    def rect(self, x: float, y: float, width: float, height: float,
             fill: tuple[float, float, float] | None = None,
             stroke: tuple[float, float, float] | None = INK,
             line_width: float = 0.8) -> None:
        fill_cmd = "" if fill is None else f"{rgb(fill)} setrgbcolor fill"
        stroke_cmd = "" if stroke is None else f"{rgb(stroke)} setrgbcolor {line_width:.2f} setlinewidth stroke"
        self.add(
            f"newpath {x:.2f} {self.y(y + height):.2f} moveto "
            f"{width:.2f} 0 rlineto 0 {height:.2f} rlineto "
            f"{-width:.2f} 0 rlineto closepath gsave {fill_cmd} grestore "
            f"gsave {stroke_cmd} grestore"
        )

    def text(self, x: float, y: float, content: str, size: float,
             font: str = "Helvetica", color: tuple[float, float, float] = INK,
             align: str = "left") -> None:
        alignment = {
            "left": "pop",
            "center": "2 div neg 0 rmoveto",
            "right": "neg 0 rmoveto",
        }[align]
        # Alignment is handled by the PostScript stringwidth operator.  The
        # first pop removes the string height; the alignment command consumes
        # the remaining width while leaving the original string for show.
        self.add(
            f"{rgb(color)} setrgbcolor /{font} findfont {size:.2f} scalefont setfont "
            f"{x:.2f} {self.y(y):.2f} moveto "
            f"({ps_string(content)}) dup stringwidth pop {alignment} show"
        )

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.commands + ["showpage", "%%EOF", ""]), encoding="ascii")


def header(fig: Figure, title: str, subtitle: str) -> None:
    fig.text(36, 28, title, 13, "Helvetica-Bold")
    fig.text(36, 45, subtitle, 7.6, "Helvetica", MID)


def box(fig: Figure, x: float, y: float, width: float, height: float,
        label: str, detail: str | None = None, accent: bool = False) -> None:
    fig.rect(x, y, width, height, fill=WHITE, stroke=ACCENT if accent else INK, line_width=1.0)
    fig.text(x + width / 2, y + 20, label, 9.2, "Helvetica-Bold", ACCENT if accent else INK, "center")
    if detail:
        fig.text(x + width / 2, y + 34, detail, 6.4, "Helvetica", MID, "center")


def architecture_figure() -> Figure:
    fig = Figure(720, 250)
    fig.fill_page()
    header(fig, "Architecture / Method Flow", "CPU numerical reference pipeline")

    center_y = 111
    node_y = 91
    node_h = 40
    box(fig, 34, node_y, 76, node_h, "BF16", "input")
    box(fig, 132, node_y, 84, node_h, "N8 x K64", "grouping")
    box(fig, 238, node_y, 82, node_h, "Scale", "FP32")
    box(fig, 509, node_y, 80, node_h, "LUTWeight", "3-bit index + LUT")
    box(fig, 613, node_y, 70, node_h, "Qwen", "inference path")

    fig.arrow(110, center_y, 128, center_y)
    fig.arrow(216, center_y, 234, center_y)
    fig.arrow(320, center_y, 345, center_y)

    # Recipe alternatives are drawn as a compact branch, not as UI-style cards.
    box(fig, 345, 70, 144, 29, "Lloyd-Max", accent=False)
    box(fig, 345, 123, 144, 29, "H64 + GPTQ + Refit", accent=True)
    fig.line(345, center_y, 333, center_y, INK, 0.8)
    fig.line(333, center_y, 345, 84.5, INK, 0.8)
    fig.line(333, center_y, 345, 137.5, INK, 0.8)
    fig.line(489, 84.5, 498, 84.5, INK, 0.8)
    fig.line(498, 84.5, 498, center_y, INK, 0.8)
    fig.line(489, 137.5, 498, 137.5, INK, 0.8)
    fig.line(498, 137.5, 498, center_y, INK, 0.8)
    fig.arrow(498, center_y, 507, center_y)
    fig.arrow(589, center_y, 611, center_y)

    fig.line(36, 190, 684, 190, LIGHT, 0.6)
    fig.text(36, 207, "A16: BF16 activations", 7.6, "Helvetica", INK)
    fig.text(190, 207, "A8: activation fake quantization", 7.6, "Helvetica", MID)
    fig.text(420, 207, "FP32 accumulation", 7.6, "Helvetica", MID)
    fig.text(36, 224, "The recipe branch changes weight construction; activation precision is selected downstream.", 7.0, "Helvetica-Oblique", MID)
    return fig


def table_figure(rows: list[dict[str, str]]) -> Figure:
    fig = Figure(720, 300)
    fig.fill_page()
    header(fig, "Main Held-out Results", "Qwen3-0.6B-Base | WikiText-2 raw test | 32,736 predicted tokens")

    x0, x1, x2, x3, x4 = 45, 284, 400, 545, 675
    top = 75
    row_h = 31
    fig.text(x0, top, "Recipe", 8.4, "Helvetica-Bold")
    fig.text(x1, top, "Mode", 8.4, "Helvetica-Bold")
    fig.text(x3, top, "Mean NLL", 8.4, "Helvetica-Bold", align="right")
    fig.text(x4, top, "PPL", 8.4, "Helvetica-Bold", align="right")
    fig.line(x0, top + 9, x4, top + 9, INK, 0.9)

    labels = [
        ("BF16 reference", "BF16", False),
        ("Plain Lloyd-Max", "W3A16", False),
        ("Plain Lloyd-Max", "W3A8", False),
        ("H64 + GPTQ + Refit", "W3A16", True),
        ("H64 + GPTQ + Refit", "W3A8", True),
    ]
    for index, (row, label) in enumerate(zip(rows, labels, strict=True)):
        recipe, mode, accent = label
        y = top + 31 + index * row_h
        fig.line(x0, y + 9, x4, y + 9, PALE if index < 4 else INK, 0.55)
        if accent:
            fig.line(x0 - 8, y - 12, x0 - 8, y + 8, ACCENT, 2.0)
        fig.text(x0, y, recipe, 8.0, "Helvetica-Bold" if index == 0 else "Helvetica", ACCENT if accent else INK)
        fig.text(x1, y, mode, 8.0, "Helvetica", MID)
        fig.text(x3, y, f'{float(row["mean_nll"]):.4f}', 8.1, "Courier", INK, "right")
        fig.text(x4, y, f'{float(row["ppl"]):.4f}', 8.1, "Courier-Bold", ACCENT if accent else INK, "right")

    fig.text(x0, 272, "A16 preserves BF16 activations; A8 applies activation fake quantization. All rows use FP32 accumulation.", 6.8, "Helvetica-Oblique", MID)
    return fig


def chart_figure(rows: list[dict[str, str]]) -> Figure:
    fig = Figure(720, 370)
    fig.fill_page()
    header(fig, "Perplexity (PPL) on WikiText-2 Test", "Lower is better; the W3A8 comparison is highlighted")

    left, right, top, bottom = 76, 684, 86, 284
    max_ppl = 32.0

    def y_for(ppl: float) -> float:
        return bottom - (ppl / max_ppl) * (bottom - top)

    for tick in range(0, 33, 8):
        y = y_for(tick)
        fig.line(left, y, right, y, LIGHT, 0.55)
        fig.text(left - 10, y + 2.5, str(tick), 7.0, "Courier", MID, "right")
    fig.line(left, top, left, bottom, INK, 0.9)
    fig.line(left, bottom, right, bottom, INK, 0.9)
    fig.text(30, 186, "PPL", 8.0, "Helvetica-Bold", INK, "center")

    bars = [
        ("BF16", 235, [("BF16", value(rows, "none", "bf16", "ppl"), PALE, MID)]),
        ("W3A16", 432, [
            ("Plain", value(rows, "plain_lloyd_max", "w3a16", "ppl"), WHITE, MID),
            ("H64 + GPTQ + Refit", value(rows, "h64_gptq_refit", "w3a16", "ppl"), ACCENT, ACCENT),
        ]),
        ("W3A8", 610, [
            ("Plain", value(rows, "plain_lloyd_max", "w3a8", "ppl"), WHITE, MID),
            ("H64 + GPTQ + Refit", value(rows, "h64_gptq_refit", "w3a8", "ppl"), ACCENT, ACCENT),
        ]),
    ]
    bar_w, gap = 29, 9
    bar_tops: dict[str, float] = {}
    for group, group_x, group_bars in bars:
        total_w = len(group_bars) * bar_w + (len(group_bars) - 1) * gap
        start_x = group_x - total_w / 2
        for index, (label, ppl, fill, stroke) in enumerate(group_bars):
            x = start_x + index * (bar_w + gap)
            y = y_for(ppl)
            fig.rect(x, y, bar_w, bottom - y, fill=fill, stroke=stroke, line_width=0.8)
            fig.text(x + bar_w / 2, y - 7, f"{ppl:.2f}", 7.3, "Courier-Bold", stroke, "center")
            bar_tops[f"{group}:{label}"] = y
        fig.text(group_x, 304, group, 8.4, "Helvetica-Bold", INK, "center")

    # Group labels and one compact recipe key; the key prevents repeated
    # labels from colliding beneath the narrow W3A16/W3A8 groups.
    fig.text(235, 324, "reference", 6.9, "Helvetica-Oblique", MID, "center")
    fig.rect(330, 332, 7, 7, fill=WHITE, stroke=MID, line_width=0.7)
    fig.text(343, 339, "Plain Lloyd-Max", 6.8, "Helvetica", MID)
    fig.rect(463, 332, 7, 7, fill=ACCENT, stroke=ACCENT, line_width=0.7)
    fig.text(476, 339, "H64 + GPTQ + Refit", 6.8, "Helvetica", ACCENT)

    # Minimal callout for the requested transition, with a drawn arrow.
    fig.text(610, 60, "W3A8", 7.5, "Helvetica-Bold", ACCENT, "center")
    fig.text(570, 73, "29.07", 8.2, "Courier-Bold", ACCENT, "right")
    fig.arrow(578, 70, 619, 70, ACCENT, 0.8)
    fig.text(630, 73, "22.22", 8.2, "Courier-Bold", ACCENT)
    fig.line(570, 78, 595, bar_tops["W3A8:Plain"] - 4, ACCENT, 0.55, (3, 3))
    fig.line(630, 78, 627, bar_tops["W3A8:H64 + GPTQ + Refit"] - 4, ACCENT, 0.55, (3, 3))
    fig.text(610, 45, "-23.54%", 7.0, "Helvetica", ACCENT, "center")

    # Explicit BF16 reference annotation.
    fig.text(235, 116, "BF16 = 14.07", 7.6, "Helvetica-Bold", MID, "center")
    fig.line(235, 120, 235, bar_tops["BF16:BF16"] - 4, MID, 0.55, (3, 3))
    return fig


def export_figure(fig: Figure, stem: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if shutil.which("gs") is None:
        raise RuntimeError("Ghostscript (gs) is required to export PNG figures")
    with tempfile.TemporaryDirectory(prefix="rubin-figures-") as temp_dir:
        ps_path = Path(temp_dir) / f"{stem}.eps"
        fig.save(ps_path)
        png_path = OUTPUT_DIR / f"{stem}.png"
        common = ["gs", "-q", "-dSAFER", "-dBATCH", "-dNOPAUSE", "-dEPSCrop"]
        subprocess.run(common + ["-sDEVICE=pngalpha", "-r300", f"-sOutputFile={png_path}", str(ps_path)], check=True)
        print(png_path)


def main() -> None:
    rows = load_rows()
    # The old SVG exports are intentionally not part of the publication asset set.
    for old in OUTPUT_DIR.glob("*.svg"):
        old.unlink()
    export_figure(architecture_figure(), "architecture_method_flow")
    export_figure(table_figure(rows), "main_results_table")
    export_figure(chart_figure(rows), "ppl_comparison")


if __name__ == "__main__":
    main()
