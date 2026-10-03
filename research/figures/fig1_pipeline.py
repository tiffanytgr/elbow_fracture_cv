"""Figure 1 -- the complete clinical pipeline, and representative processing.

  A  schematic of the framework end to end: paired AP and LAT input, the
     quality-control front-end, the two parallel tracks (hierarchical cascade
     and geometric reconstruction), and grade fusion with discordance flagging
  B  representative AP processing: as acquired, markers removed, standardised
  C  representative LAT processing: the six-step standardisation, as acquired
     through to the aligned radiograph used downstream

The schematic is drawn in code rather than imported as artwork so it stays in
step with the pipeline: the node labels come from this module's constants, and
the figure is regenerated from the same command as the rest.

Representative images are read from ``<results>/images/`` (see IMAGE_STRIPS
below for the filenames). Missing images render as labelled placeholders.

    python -m research.figures.fig1_pipeline --results research/outputs
    python -m research.figures.fig1_pipeline --demo
"""
from __future__ import annotations

import logging
import sys

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from .cli import build_parser, setup
from .data import Results
from .panels import image_panel
from .style import (
    CASCADE,
    GEOMETRIC,
    GRID,
    INK,
    INK_SECONDARY,
    MULTICLASS,
    STATUS_CRITICAL,
    WIDTH_DOUBLE,
    panel_label,
    save_figure,
)

log = logging.getLogger(__name__)

#: Cascade nodes, in routing order. Kept here so the schematic and the text
#: cannot drift apart.
NODES = [
    ("Node 1", "Fracture vs normal", "AP"),
    ("Node 2", "Grade 3 vs 1/2A/2B", "AP"),
    ("Node 3", "Grade 2 vs Grade 1", "LAT"),
    ("Node 4", "Grade 2B vs 2A", "LAT"),
]

#: Geometric measurements, with the node each one informs.
MEASUREMENTS = [
    ("AHL bisection", "→ Node 3"),
    ("Cortical-width ratio", "→ Node 4"),
    ("Baumann angle", "→ Node 2"),
]

IMAGE_STRIPS = {
    "AP": [
        ("ap_original.png", "As acquired"),
        ("ap_markers_removed.png", "Markers removed"),
        ("ap_standardised.png", "Standardised"),
    ],
    "LAT": [
        ("lat_original.png", "As acquired"),
        ("lat_foreground.png", "Otsu foreground"),
        ("lat_orientation.png", "Orientation"),
        ("lat_pca_axis.png", "PCA shaft axis"),
        ("lat_standardised.png", "Rotated and cropped"),
    ],
}


def _box(ax, x, y, w, h, text, *, facecolor, edgecolor, fontsize=6.0, weight="regular",
         textcolor=INK):
    """Rounded box with centred, wrapped text, in 0-100 schematic coordinates."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=1.6",
            linewidth=0.8, facecolor=facecolor, edgecolor=edgecolor, zorder=3,
        )
    )
    ax.text(
        x + w / 2, y + h / 2, text, ha="center", va="center",
        fontsize=fontsize, color=textcolor, zorder=4, linespacing=1.35,
        fontweight=weight,
    )


def _arrow(ax, start, end, color=INK_SECONDARY, style="-|>", linestyle="-"):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle=style, mutation_scale=7,
            linewidth=0.8, color=color, linestyle=linestyle,
            shrinkA=0, shrinkB=0, zorder=2,
        )
    )


def _group_frame(ax, x, y, w, h, label, color):
    """Dashed grouping frame with a small label in its top-left corner."""
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0,rounding_size=1.6",
            linewidth=0.7, facecolor="none", edgecolor=color,
            linestyle=(0, (3, 2)), zorder=1,
        )
    )
    ax.text(
        x + 1.4, y + h - 1.0, label, ha="left", va="top",
        fontsize=5.8, color=color, fontweight="bold", zorder=4,
    )


def _schematic(ax) -> None:
    """Draw the pipeline flow.

    Coordinates are a 0-100 grid in both axes. Boxes are laid out in four
    columns -- input, quality control, the two parallel tracks, output -- and
    the dashed grouping frames leave 5 units of head room so their labels sit
    above the first box rather than on it.
    """
    ax.set_xlim(0, 100)
    ax.set_ylim(11, 99)
    ax.axis("off")

    tint_cascade = "#E7F0F8"
    tint_geometric = "#E6F2EB"
    tint_qc = "#FDF0E2"
    tint_neutral = "#F2F2F0"

    # --- column 1: input -----------------------------------------------------
    _box(ax, 1, 68, 15, 15, "Paired\nradiographs\n\nAP + LAT",
         facecolor=tint_neutral, edgecolor=GRID, fontsize=6.0, weight="bold")

    # --- column 2: quality-control front-end --------------------------------
    _group_frame(ax, 19, 59, 24, 38, "QUALITY CONTROL", MULTICLASS)
    qc_steps = [
        ("Marker removal", 85.0),
        ("LAT standardisation\n(6-step alignment)", 76.0),
        ("DRUE uncertainty gate\nabstain above threshold", 67.0),
    ]
    for text, y in qc_steps:
        _box(ax, 21, y, 20, 7.0, text, facecolor=tint_qc,
             edgecolor=MULTICLASS, fontsize=5.4)
        _arrow(ax, (16, 75.5), (21, y + 3.5))

    # Abstention is a pipeline output, not a silent failure.
    _box(ax, 21, 50.5, 20, 6.5, "Withheld from\nautomated grading",
         facecolor="#FFFFFF", edgecolor=STATUS_CRITICAL, fontsize=5.3,
         textcolor=STATUS_CRITICAL)
    _arrow(ax, (31, 67), (31, 57), color=STATUS_CRITICAL, linestyle=(0, (2, 1.5)))

    # --- column 3, upper: hierarchical cascade ------------------------------
    _group_frame(ax, 46, 59, 27, 38, "HIERARCHICAL CASCADE", CASCADE)
    for i, (node, question, view) in enumerate(NODES):
        y = 85.0 - i * 7.5
        _box(ax, 48, y, 23, 6.0, f"{node} ({view}) · {question}",
             facecolor=tint_cascade, edgecolor=CASCADE, fontsize=5.3)
        if i:
            _arrow(ax, (59.5, y + 7.5), (59.5, y + 6.0))
    _arrow(ax, (43, 77), (48, 88.0))

    # --- column 3, lower: geometric reconstruction --------------------------
    _group_frame(ax, 46, 14, 27, 33, "GEOMETRIC RECONSTRUCTION", GEOMETRIC)
    _box(ax, 48, 34.5, 23, 7.0, "SAM2 humerus + capitellum\nsegmentation",
         facecolor=tint_geometric, edgecolor=GEOMETRIC, fontsize=5.3)
    measurement_text = "\n".join(f"{name}  {feeds}" for name, feeds in MEASUREMENTS)
    _box(ax, 48, 17.0, 23, 13.0, measurement_text,
         facecolor=tint_geometric, edgecolor=GEOMETRIC, fontsize=5.1)
    _arrow(ax, (59.5, 34.5), (59.5, 30.0))
    _arrow(ax, (43, 70), (48, 41.5))

    # --- column 4: fusion and output ----------------------------------------
    _box(ax, 76, 70, 23, 13, "Grade fusion\nCNN grade vs geometric grade",
         facecolor=tint_neutral, edgecolor=GRID, fontsize=5.6, weight="bold")
    _arrow(ax, (71, 82), (76, 79.0))
    _arrow(ax, (71, 30), (76, 73.0))

    _box(ax, 76, 50, 23, 14,
         "Gartland grade\nNormal · 1 · 2A · 2B · 3\n\n+ inspectable measurements",
         facecolor="#FFFFFF", edgecolor=INK, fontsize=5.6, weight="bold")
    _arrow(ax, (87.5, 70), (87.5, 64))

    _box(ax, 76, 36, 23, 8, "Discordance flagged\nfor clinician review",
         facecolor="#FFFFFF", edgecolor=STATUS_CRITICAL, fontsize=5.4,
         textcolor=STATUS_CRITICAL)
    _arrow(ax, (87.5, 50), (87.5, 44), color=STATUS_CRITICAL,
           linestyle=(0, (2, 1.5)))


def _image_strip(fig, grid_row, results: Results, view: str, letter: str, title: str):
    """One row of representative processing steps for a single view."""
    strip = IMAGE_STRIPS[view]
    sub = grid_row.subgridspec(1, len(strip), wspace=0.12)
    first = None
    for i, (filename, caption) in enumerate(strip):
        ax = fig.add_subplot(sub[0, i])
        image_panel(
            ax, results.image(filename), caption=f"({i + 1}) {caption}",
            placeholder=f"{view}\n{caption.lower()}",
        )
        if i == 0:
            first = ax
            panel_label(ax, letter, dx=-0.10, dy=1.04)
            ax.text(
                -0.10, 1.30, title, transform=ax.transAxes, ha="left", va="top",
                fontsize=7, color=INK,
            )
    return first


def build_figure(results: Results):
    fig = plt.figure(figsize=(WIDTH_DOUBLE, 7.4))
    grid = fig.add_gridspec(
        3, 1, height_ratios=[1.45, 0.72, 0.72], hspace=0.60,
        left=0.03, right=0.985, top=0.965, bottom=0.04,
    )

    ax_schematic = fig.add_subplot(grid[0])
    _schematic(ax_schematic)
    panel_label(ax_schematic, "A", dx=-0.005, dy=0.99)
    ax_schematic.text(
        0.025, 1.0, "Clinical pipeline", transform=ax_schematic.transAxes,
        ha="left", va="top", fontsize=7, color=INK,
    )

    _image_strip(fig, grid[1], results, "AP", "B",
                 "Representative AP processing")
    _image_strip(fig, grid[2], results, "LAT", "C",
                 "Representative LAT standardisation")
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("Render Figure 1 (pipeline overview).", epilog=__doc__)
    args = parser.parse_args(argv)
    results = setup(args)
    fig = build_figure(results)
    save_figure(fig, args.out_dir, "figure1_pipeline",
                formats=tuple(args.formats), demo=args.demo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
