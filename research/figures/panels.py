"""Reusable panel renderers shared by the main and appendix figures.

Each function draws into an Axes the caller supplies and returns nothing, so
panels compose freely into whatever grid a figure needs.
"""
from __future__ import annotations

import logging

import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

from research.multiclass_baseline.labels import OPERATIVE_BOUNDARY, SHORT_GRADES

from .style import (
    CASCADE,
    GRADE_CMAP,
    GRID,
    HATCHES,
    INK,
    INK_MUTED,
    INK_SECONDARY,
    MULTICLASS,
    STATUS_CRITICAL,
    annotate_n,
    readable_text_color,
    strip_spines,
    subtle_grid,
)

log = logging.getLogger(__name__)


def confusion_panel(
    ax,
    cm: np.ndarray,
    title: str = "",
    class_labels: tuple[str, ...] = SHORT_GRADES,
    show_colorbar: bool = False,
    mark_operative_boundary: bool = True,
) -> None:
    """Confusion matrix with counts in the cells, shaded by row-wise recall.

    Shading is row-normalised (recall) rather than raw count, because the grade
    supports differ by an order of magnitude -- a count-shaded matrix would show
    only that Normal is the largest class. Counts stay printed, so the reader
    still sees the denominators.
    """
    cm = np.asarray(cm)
    row_totals = cm.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        recall = np.where(row_totals > 0, cm / row_totals, 0.0)

    ax.imshow(recall, cmap=GRADE_CMAP, vmin=0, vmax=1, aspect="equal")

    n_classes = cm.shape[0]
    for i in range(n_classes):
        for j in range(n_classes):
            value = int(cm[i, j])
            if row_totals[i, 0] == 0:
                label = "-"
            elif i == j:
                label = f"{value}\n{recall[i, j] * 100:.0f}%"
            else:
                label = str(value) if value else ""
            if not label:
                continue
            ax.text(
                j, i, label, ha="center", va="center",
                fontsize=6 if i == j else 6.5,
                fontweight="bold" if i == j else "regular",
                color=readable_text_color(GRADE_CMAP(recall[i, j])),
                linespacing=1.1,
            )

    # Outline the 2A<->2B cells: the operative boundary the study is about.
    if mark_operative_boundary:
        a, b = OPERATIVE_BOUNDARY
        for i, j in ((a, b), (b, a)):
            ax.add_patch(
                Rectangle(
                    (j - 0.5, i - 0.5), 1, 1, fill=False,
                    edgecolor=STATUS_CRITICAL, linewidth=1.3, zorder=5,
                )
            )

    ax.set_xticks(range(n_classes), labels=list(class_labels))
    ax.set_yticks(
        range(n_classes),
        labels=[f"{g}\n(n={int(t)})" for g, t in zip(class_labels, row_totals.ravel())],
    )
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("Reference standard")
    if title:
        ax.set_title(title, pad=4)
    ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.grid(which="minor", color=GRID, linewidth=0.5)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    annotate_n(ax, int(cm.sum()), prefix="n = ")

    if show_colorbar:
        bar = ax.figure.colorbar(
            ax.images[0], ax=ax, fraction=0.045, pad=0.03, ticks=[0, 0.5, 1]
        )
        bar.ax.set_yticklabels(["0", "50", "100"])
        bar.set_label("Per-grade recall, %", fontsize=6.5)
        bar.outline.set_visible(False)
        bar.ax.tick_params(labelsize=6, length=2)


def grouped_bar_panel(
    ax,
    categories: list[str],
    series: dict[str, list[float]],
    colors: list[str],
    errors: dict[str, list[tuple[float, float]]] | None = None,
    ylabel: str = "",
    title: str = "",
    value_labels: bool = True,
    supports: list[int] | None = None,
    ylim: tuple[float, float] | None = None,
) -> None:
    """Grouped bars with optional asymmetric CI whiskers and direct labels.

    Bars carry hatch texture as a second channel alongside hue, so the panel
    survives greyscale printing and the 6-8 CVD band.
    """
    n_series = len(series)
    if n_series == 0:
        raise ValueError("grouped_bar_panel needs at least one series")
    x = np.arange(len(categories), dtype=float)
    width = min(0.38, 0.82 / n_series)

    for s, (name, values) in enumerate(series.items()):
        offset = (s - (n_series - 1) / 2) * width
        yerr = None
        if errors and name in errors:
            lows = [max(0.0, v - lo) for v, (lo, _) in zip(values, errors[name])]
            highs = [max(0.0, hi - v) for v, (_, hi) in zip(values, errors[name])]
            yerr = np.array([lows, highs])
        ax.bar(
            x + offset, values, width * 0.92, label=name,
            color=colors[s % len(colors)],
            hatch=HATCHES[s % len(HATCHES)],
            edgecolor="white", linewidth=0.6,
            yerr=yerr, error_kw={"elinewidth": 0.7, "capsize": 1.6,
                                 "ecolor": INK_SECONDARY},
            zorder=3,
        )
        if value_labels:
            # Sit the label above the whisker, not above the bar, or the two
            # collide wherever the interval is wide (which is exactly the
            # minority grades the reader most needs to read).
            tops = values if yerr is None else [v + h for v, h in zip(values, yerr[1])]
            for xi, value, top in zip(x + offset, values, tops):
                if not np.isfinite(value):
                    continue
                ax.text(
                    xi, top + 1.8, f"{value:.0f}", ha="center", va="bottom",
                    fontsize=5.8, color=INK_SECONDARY, zorder=4,
                )

    labels = list(categories)
    if supports is not None:
        labels = [f"{c}\n(n={n})" for c, n in zip(categories, supports)]
    ax.set_xticks(x, labels=labels)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=4)
    ax.set_ylim(*(ylim or (0, 108)))
    subtle_grid(ax, axis="y")
    strip_spines(ax)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=n_series)


def risk_coverage_panel(
    ax,
    curves: dict[str, dict[str, np.ndarray]],
    colors: list[str],
    operating_points: dict[str, tuple[float, float]] | None = None,
    title: str = "",
) -> None:
    """Error rate against coverage, abstaining on lowest confidence first.

    A flat curve means the confidence score carries no information about which
    cases are wrong; a curve that rises towards full coverage means abstention
    is buying accuracy.
    """
    for i, (name, curve) in enumerate(curves.items()):
        colour = colors[i % len(colors)]
        ax.plot(
            curve["coverage"] * 100, curve["risk"] * 100,
            color=colour, label=name, linewidth=1.4,
            linestyle=("-", "--", "-.")[i % 3],
        )
        if operating_points and name in operating_points:
            cov, risk = operating_points[name]
            ax.plot(
                cov, risk, marker="o", markersize=4.5, color=colour,
                markeredgecolor="white", markeredgewidth=0.8, zorder=5,
            )
            ax.annotate(
                f"{cov:.0f}% coverage",
                xy=(cov, risk), xytext=(-7, -11), textcoords="offset points",
                fontsize=5.8, color=colour, ha="right", va="top",
            )
    ax.set_xlabel("Coverage, %")
    ax.set_ylabel("Error rate on retained cases, %")
    ax.set_xlim(20, 101)
    if title:
        ax.set_title(title, pad=4)
    subtle_grid(ax, axis="both")
    strip_spines(ax)
    ax.legend(loc="lower right")


def score_distribution_panel(
    ax,
    values_by_group: dict[str, np.ndarray],
    colors: list[str],
    xlabel: str = "",
    title: str = "",
    auroc_value: float | None = None,
    log_x: bool = False,
) -> None:
    """Overlaid histograms for two groups, e.g. correct vs incorrect cases.

    Step outlines rather than filled bars, so the overlap region stays readable
    where the two distributions sit on top of one another -- and the overlap is
    the point being made.
    """
    finite = np.concatenate([v[np.isfinite(v)] for v in values_by_group.values()])
    if finite.size == 0:
        raise ValueError("score_distribution_panel: no finite values")
    if log_x:
        positive = finite[finite > 0]
        bins = np.geomspace(positive.min(), positive.max(), 26)
    else:
        bins = np.linspace(finite.min(), finite.max(), 26)

    for i, (name, values) in enumerate(values_by_group.items()):
        colour = colors[i % len(colors)]
        values = values[np.isfinite(values)]
        ax.hist(
            values, bins=bins, density=True, histtype="stepfilled",
            color=colour, alpha=0.18, zorder=2,
        )
        ax.hist(
            values, bins=bins, density=True, histtype="step",
            color=colour, linewidth=1.3, label=f"{name} (n={values.size})", zorder=3,
        )
        ax.axvline(
            float(np.median(values)), color=colour, linestyle=":",
            linewidth=1.0, zorder=4,
        )
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")
    if title:
        ax.set_title(title, pad=4)
    if auroc_value is not None and np.isfinite(auroc_value):
        ax.text(
            0.97, 0.72, f"AUC = {auroc_value:.2f}", transform=ax.transAxes,
            ha="right", va="top", fontsize=6.5, color=INK,
        )
    ax.set_yticks([])
    subtle_grid(ax, axis="x")
    strip_spines(ax, keep=("bottom",))
    ax.legend(loc="upper right")


def calibration_panel(
    ax,
    bin_centre: np.ndarray,
    bin_accuracy: np.ndarray,
    bin_confidence: np.ndarray,
    bin_count: np.ndarray,
    ece: float,
    color: str = CASCADE,
    title: str = "",
) -> None:
    """Reliability diagram: observed accuracy against predicted confidence.

    Marker area is proportional to the number of cases in the bin, so sparse
    bins cannot be mistaken for well-estimated ones.
    """
    valid = np.isfinite(bin_accuracy) & (bin_count > 0)
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=0.8, linestyle="--",
            label="Perfect calibration", zorder=2)
    ax.vlines(
        bin_confidence[valid], bin_confidence[valid], bin_accuracy[valid],
        color=color, linewidth=0.8, alpha=0.5, zorder=3,
    )
    sizes = 8 + 42 * (bin_count[valid] / max(bin_count.max(), 1))
    ax.scatter(
        bin_confidence[valid], bin_accuracy[valid], s=sizes, color=color,
        edgecolor="white", linewidth=0.6, zorder=4, label="Observed",
    )
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Observed accuracy")
    ax.set_xlim(0, 1.02)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    if title:
        ax.set_title(title, pad=4)
    ax.text(
        0.04, 0.95, f"ECE = {ece:.3f}", transform=ax.transAxes,
        ha="left", va="top", fontsize=6.5, color=INK,
    )
    ax.text(
        0.04, 0.87, "marker area ∝ cases in bin", transform=ax.transAxes,
        ha="left", va="top", fontsize=5.5, color=INK_MUTED,
    )
    subtle_grid(ax, axis="both")
    strip_spines(ax)
    ax.legend(loc="lower right")


def strip_distribution_panel(
    ax,
    grade_values: dict[str, np.ndarray],
    colors: list[str],
    ylabel: str = "",
    title: str = "",
    annotation: str | None = None,
    reference_band: tuple[float, float] | None = None,
    reference_label: str = "",
) -> None:
    """Per-grade jittered points with a mean +/- SD marker.

    Individual points are shown rather than a box plot: at n=8 for Grade 2B a
    box plot's quartiles imply precision the sample does not support.
    """
    rng = np.random.default_rng(7)
    names = list(grade_values)

    if reference_band is not None:
        ax.axhspan(
            *reference_band, color=INK_MUTED, alpha=0.1, zorder=1,
            label=reference_label or None,
        )

    for i, name in enumerate(names):
        values = np.asarray(grade_values[name], dtype=float)
        values = values[np.isfinite(values)]
        if values.size == 0:
            continue
        colour = colors[i % len(colors)]
        jitter = rng.uniform(-0.17, 0.17, values.size)
        ax.scatter(
            np.full(values.size, i) + jitter, values, s=9, color=colour,
            alpha=0.55, edgecolor="none", zorder=3,
        )
        mean, sd = float(values.mean()), float(values.std(ddof=1)) if values.size > 1 else 0.0
        ax.errorbar(
            i, mean, yerr=sd, fmt="_", color=INK, markersize=14,
            elinewidth=1.1, capsize=3, capthick=1.0, zorder=4,
        )
        ax.text(
            i + 0.26, mean, f"{mean:.2f}", fontsize=5.8, color=INK_SECONDARY,
            va="center", ha="left", zorder=5,
        )

    ax.set_xticks(
        range(len(names)),
        labels=[
            f"{n}\n(n={np.isfinite(np.asarray(grade_values[n], dtype=float)).sum()})"
            for n in names
        ],
    )
    ax.set_xlim(-0.6, len(names) - 0.4)
    ax.set_ylabel(ylabel)
    # The annotation sits above the axes, under the title: inside the axes it
    # collides with the legend on every panel where the spread is wide.
    if title:
        ax.set_title(title, pad=13 if annotation else 4)
    if annotation:
        ax.text(
            0.5, 1.012, annotation, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=6, color=INK,
        )
    subtle_grid(ax, axis="y")
    strip_spines(ax)
    handles, _ = ax.get_legend_handles_labels()
    marker = Line2D([], [], color=INK, marker="_", linestyle="none",
                    markersize=9, label="Mean ± SD")
    ax.legend(handles=[*handles, marker], loc="upper right")


def profile_panel(
    ax,
    position: np.ndarray,
    curves_by_grade: dict[str, np.ndarray],
    colors: list[str],
    ylabel: str = "Normalised cortical width",
    xlabel: str = "Position proximal to capitellum",
    title: str = "",
) -> None:
    """Population mean profile with an inter-quartile ribbon, per grade."""
    for i, (grade, curves) in enumerate(curves_by_grade.items()):
        curves = np.atleast_2d(np.asarray(curves, dtype=float))
        colour = colors[i % len(colors)]
        mean = np.nanmean(curves, axis=0)
        lower = np.nanpercentile(curves, 25, axis=0)
        upper = np.nanpercentile(curves, 75, axis=0)
        ax.fill_between(position, lower, upper, color=colour, alpha=0.16, linewidth=0, zorder=2)
        ax.plot(
            position, mean, color=colour, linewidth=1.5, zorder=3,
            linestyle=("-", "--")[i % 2],
            label=f"{grade} (n={curves.shape[0]})",
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, pad=4)
    ax.text(
        0.99, 0.02, "ribbon = IQR", transform=ax.transAxes, ha="right", va="bottom",
        fontsize=5.5, color=INK_MUTED,
    )
    subtle_grid(ax, axis="both")
    strip_spines(ax)
    ax.legend(loc="upper right")


def error_taxonomy_panel(
    ax,
    breakdown: dict[str, dict[str, int]],
    title: str = "",
    colors: tuple[str, ...] = (CASCADE, MULTICLASS),
) -> None:
    """Stacked error composition per model: adjacent, distant, boundary crossings.

    Counts rather than proportions, because the absolute number of crossings of
    the operative boundary is what a clinician weighs.
    """
    models = list(breakdown)
    kinds = ["Adjacent (non-boundary)", "2A↔2B boundary", "Distant (≥2 grades)"]
    fills = ["#B9CFE5", STATUS_CRITICAL, "#5E6B77"]
    hatches = ["", "", "///"]

    x = np.arange(len(models), dtype=float)
    bottoms = np.zeros(len(models))
    for kind, fill, hatch in zip(kinds, fills, hatches):
        values = np.array([breakdown[m].get(kind, 0) for m in models], dtype=float)
        ax.bar(
            x, values, 0.5, bottom=bottoms, label=kind, color=fill,
            hatch=hatch, edgecolor="white", linewidth=0.8, zorder=3,
        )
        for xi, value, base in zip(x, values, bottoms):
            if value > 0:
                ax.text(
                    xi, base + value / 2, f"{int(value)}", ha="center", va="center",
                    fontsize=6, color=readable_text_color(fill), zorder=4,
                )
        bottoms += values

    for xi, total in zip(x, bottoms):
        ax.text(xi, total + 0.6, f"{int(total)} errors", ha="center", va="bottom",
                fontsize=6, color=INK_SECONDARY)

    ax.set_xticks(x, labels=models)
    ax.set_ylabel("Misclassified patients")
    # Headroom for the legend, which would otherwise sit on top of the bars.
    ax.set_ylim(0, bottoms.max() * 1.62 if bottoms.max() else 1)
    if title:
        ax.set_title(title, pad=4)
    subtle_grid(ax, axis="y")
    strip_spines(ax)
    ax.legend(loc="upper right", ncol=1)


def error_breakdown(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, int]:
    """Count errors by kind, for error_taxonomy_panel."""
    from research.multiclass_baseline.labels import (
        crosses_operative_boundary,
        is_adjacent,
    )

    counts = {"Adjacent (non-boundary)": 0, "2A↔2B boundary": 0, "Distant (≥2 grades)": 0}
    for t, p in zip(np.asarray(y_true), np.asarray(y_pred)):
        t, p = int(t), int(p)
        if t == p:
            continue
        if crosses_operative_boundary(t, p):
            counts["2A↔2B boundary"] += 1
        elif is_adjacent(t, p):
            counts["Adjacent (non-boundary)"] += 1
        else:
            counts["Distant (≥2 grades)"] += 1
    return counts


def image_panel(ax, image_path, caption: str = "", placeholder: str = "") -> None:
    """Draw a representative radiograph, or a labelled placeholder box.

    The placeholder keeps the composition reviewable before the final
    representative cases have been chosen, and is obvious enough that it cannot
    be mistaken for a radiograph.
    """
    import matplotlib.image as mpimg

    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(GRID)
        spine.set_linewidth(0.6)

    if image_path is not None:
        try:
            ax.imshow(mpimg.imread(str(image_path)), cmap="gray")
        except (FileNotFoundError, OSError) as exc:
            log.warning("could not read %s: %s", image_path, exc)
            image_path = None

    if image_path is None:
        ax.set_facecolor("#F4F4F2")
        ax.text(
            0.5, 0.55, placeholder or "representative\nradiograph", transform=ax.transAxes,
            ha="center", va="center", fontsize=6, color=INK_MUTED, linespacing=1.4,
        )
        ax.text(
            0.5, 0.2, "[placeholder]", transform=ax.transAxes,
            ha="center", va="center", fontsize=5.5, color=STATUS_CRITICAL,
        )
    if caption:
        ax.set_xlabel(caption, fontsize=6, color=INK_SECONDARY, labelpad=2)
