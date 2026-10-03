"""Figure 4 -- anatomical explainability: measurements, not saliency.

Top row, one representative radiograph per measurement; bottom row, the
group-level distribution of that same measurement, so each column reads as
"here is what the pipeline draws, and here is what it gives across the cohort".

  A / D  anterior humeral line bisection of the capitellum (Grade 1 vs 2)
  B / E  cortical-width match ratio (Grade 2A vs 2B)
  C / F  Baumann angle (Grade 3 vs non-Grade 3)

This is the figure that distinguishes the framework from attribution maps: each
panel is a quantity in the institutional grading pathway that a clinician can
re-measure on the radiograph, not a heatmap.

Representative images are read from ``<results>/images/``:
``ahl_example.png``, ``cortical_width_example.png``, ``baumann_example.png``.
Missing images render as labelled placeholders so the composition can still be
reviewed.

    python -m research.figures.fig4_anatomy --results research/outputs
    python -m research.figures.fig4_anatomy --demo
"""
from __future__ import annotations

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from research.multiclass_baseline.labels import GRADES
from research.multiclass_baseline.metrics import wilson_interval

from .cli import build_parser, setup
from .data import MissingInput, Results
from .panels import grouped_bar_panel, image_panel, strip_distribution_panel
from .style import (
    GEOMETRIC,
    INK_SECONDARY,
    WIDTH_DOUBLE,
    grade_colors,
    panel_label,
    save_figure,
)

log = logging.getLogger(__name__)

#: Reported normal range for the Baumann angle, shaded in panel F.
BAUMANN_NORMAL_RANGE = (64.0, 81.0)


def _group(measurement: dict[str, np.ndarray], order: list[str]) -> dict[str, np.ndarray]:
    """Split per-case values by grade, keeping the requested ordinal order."""
    grouped: dict[str, np.ndarray] = {}
    for grade in order:
        mask = measurement["grade"] == grade
        if mask.any():
            grouped[grade] = measurement["value"][mask]
    if not grouped:
        raise MissingInput(
            f"no rows for grades {order}; found {sorted(set(measurement['grade']))}"
        )
    return grouped


def _mann_whitney(a: np.ndarray, b: np.ndarray) -> str:
    """Two-sided Mann-Whitney U, formatted for a panel annotation."""
    if a.size < 2 or b.size < 2:
        return "P not computed (n < 2)"
    p = float(stats.mannwhitneyu(a, b, alternative="two-sided").pvalue)
    return "P < .001" if p < 0.001 else f"P = {p:.3f}".replace("0.", ".")


def _panel_d(ax, bisection: dict[str, tuple[int, int]]) -> None:
    order = [g for g in GRADES if g in bisection]
    rates = [bisection[g][0] / bisection[g][1] * 100 for g in order]
    cis = [wilson_interval(*bisection[g]) for g in order]
    supports = [bisection[g][1] for g in order]
    grouped_bar_panel(
        ax,
        categories=[g.replace("Grade ", "G") for g in order],
        series={"AHL bisects capitellum": rates},
        colors=[GEOMETRIC],
        errors={"AHL bisects capitellum": cis},
        ylabel="Radiographs with bisection, %",
        title="AHL bisection by grade",
        supports=supports,
        ylim=(0, 124),
    )
    ax.get_legend().remove()  # single series: the title names it
    ax.text(
        0.5, 0.955, "95% CI, Wilson", transform=ax.transAxes, ha="center", va="top",
        fontsize=5.8, color=INK_SECONDARY,
    )


def _panel_e(ax, cortical: dict[str, np.ndarray]) -> None:
    grouped = _group(cortical, ["Grade 2A", "Grade 2B"])
    annotation = ""
    if len(grouped) == 2:
        a, b = grouped["Grade 2A"], grouped["Grade 2B"]
        pooled_sd = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2) if min(a.size, b.size) > 1 else np.nan
        d = (a.mean() - b.mean()) / pooled_sd if np.isfinite(pooled_sd) and pooled_sd > 0 else np.nan
        annotation = f"{_mann_whitney(a, b)}"
        if np.isfinite(d):
            annotation += f", Cohen's d = {d:.2f}"
    grouped = {k.replace("Grade ", "G"): v for k, v in grouped.items()}
    strip_distribution_panel(
        ax,
        grouped,
        colors=grade_colors(5)[2:4],
        ylabel="Cortical-width match ratio",
        title="Match ratio, Grade 2A vs 2B",
        annotation=annotation,
    )


def _panel_f(ax, baumann: dict[str, np.ndarray]) -> None:
    grouped = _group(baumann, [g for g in GRADES if g != "Normal"])
    annotation = ""
    if "Grade 3" in grouped:
        g3 = grouped["Grade 3"]
        rest = np.concatenate([v for k, v in grouped.items() if k != "Grade 3"])
        if rest.size:
            annotation = f"Grade 3 vs non-Grade 3: {_mann_whitney(g3, rest)}"
    grouped = {k.replace("Grade ", "G"): v for k, v in grouped.items()}
    strip_distribution_panel(
        ax,
        grouped,
        colors=grade_colors(5)[1:],
        ylabel="Baumann angle, degrees",
        title="Baumann angle by grade",
        annotation=annotation,
        reference_band=BAUMANN_NORMAL_RANGE,
        reference_label=f"Reported normal range ({BAUMANN_NORMAL_RANGE[0]:.0f}–"
                        f"{BAUMANN_NORMAL_RANGE[1]:.0f}°)",
    )


def build_figure(results: Results):
    bisection = results.ahl_bisection()
    cortical = results.measurements("cortical_width.csv", "match_ratio")
    baumann = results.measurements("baumann.csv", "baumann_angle")

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 5.3))
    grid = fig.add_gridspec(
        2, 3, hspace=0.42, wspace=0.34, height_ratios=[0.85, 1.0],
        left=0.075, right=0.985, top=0.95, bottom=0.085,
    )

    images = [
        ("A", "ahl_example.png",
         "Anterior humeral line and capitellum",
         "AHL fitted along the anterior cortex;\nbisection of the capitellum\nseparates Grade 1 from Grade 2"),
        ("B", "cortical_width_example.png",
         "Cortical-width sampling",
         "40 cross-sections perpendicular to the\nPCA shaft axis over the 30 px\nproximal to the capitellum"),
        ("C", "baumann_example.png",
         "Baumann angle",
         "Shaft axis and fitted physeal line\non the AP view"),
    ]
    for col, (letter, filename, caption, placeholder) in enumerate(images):
        ax = fig.add_subplot(grid[0, col])
        image_panel(ax, results.image(filename), caption=caption, placeholder=placeholder)
        panel_label(ax, letter, dx=-0.04, dy=1.02)

    ax_d = fig.add_subplot(grid[1, 0])
    _panel_d(ax_d, bisection)
    panel_label(ax_d, "D", dx=-0.30, dy=1.04)

    ax_e = fig.add_subplot(grid[1, 1])
    _panel_e(ax_e, cortical)
    panel_label(ax_e, "E", dx=-0.26, dy=1.04)

    ax_f = fig.add_subplot(grid[1, 2])
    _panel_f(ax_f, baumann)
    panel_label(ax_f, "F", dx=-0.26, dy=1.04)

    return fig


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("Render Figure 4 (anatomical explainability).", epilog=__doc__)
    args = parser.parse_args(argv)
    results = setup(args)
    try:
        fig = build_figure(results)
    except MissingInput as exc:
        log.error("cannot render Figure 4: %s", exc)
        log.error("run with --demo to check the layout without the real artefacts")
        return 1
    save_figure(fig, args.out_dir, "figure4_anatomy",
                formats=tuple(args.formats), demo=args.demo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
