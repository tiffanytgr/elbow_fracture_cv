"""Figure 3 -- uncertainty-based quality control.

  A  risk-coverage curves: error rate on the retained subset as cases are
     withheld in order of increasing uncertainty, with the deployed operating
     point marked
  B  discrimination: DRUE score distributions for correctly and incorrectly
     graded patients, with the AUROC
  C  calibration: reliability diagram and expected calibration error
  D  four-condition ablation: raw, + standardisation, + QC filtering, and both,
     per cascade node

Panel A ranks by two scores on one axis -- the classifier's own max-softmax
confidence and the DRUE reconstruction score -- because the question the panel
answers is which of the two is the better abstention signal.

    python -m research.figures.fig3_uncertainty --results research/outputs
    python -m research.figures.fig3_uncertainty --demo
"""
from __future__ import annotations

import logging
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

from research.multiclass_baseline.metrics import (
    auroc,
    expected_calibration_error,
    risk_coverage_curve,
    selective_risk_auc,
)

from .cli import build_parser, setup
from .data import HELD_OUT_FILTERED_N, HELD_OUT_N, MissingInput, Results
from .panels import (
    calibration_panel,
    grouped_bar_panel,
    risk_coverage_panel,
    score_distribution_panel,
)
from .style import (
    ACCENT,
    CASCADE,
    GEOMETRIC,
    INK_SECONDARY,
    MULTICLASS,
    WIDTH_DOUBLE,
    panel_label,
    save_figure,
)

log = logging.getLogger(__name__)

#: Deployed coverage, from Table 3: 178 of 232 held-out patients retained.
DEPLOYED_COVERAGE = HELD_OUT_FILTERED_N / HELD_OUT_N * 100


def _panel_a(ax, uncertainty: dict[str, np.ndarray]) -> None:
    correct = uncertainty["correct"]
    curves: dict[str, dict[str, np.ndarray]] = OrderedDict()

    # DRUE: higher score = more uncertain, so rank by the negated score.
    curves["DRUE reconstruction score"] = risk_coverage_curve(
        correct, -uncertainty["drue_score"]
    )
    if "confidence" in uncertainty:
        curves["Classifier max-softmax"] = risk_coverage_curve(
            correct, uncertainty["confidence"]
        )

    operating: dict[str, tuple[float, float]] = {}
    drue_curve = curves["DRUE reconstruction score"]
    idx = int(np.argmin(np.abs(drue_curve["coverage"] * 100 - DEPLOYED_COVERAGE)))
    operating["DRUE reconstruction score"] = (
        drue_curve["coverage"][idx] * 100,
        drue_curve["risk"][idx] * 100,
    )

    risk_coverage_panel(
        ax, curves, colors=[CASCADE, MULTICLASS],
        operating_points=operating, title="Risk versus coverage",
    )
    aucs = " · ".join(
        f"{name.split()[0]} AURC {selective_risk_auc(correct, s):.3f}"
        for name, s in (
            ("DRUE", -uncertainty["drue_score"]),
            *( (("Softmax", uncertainty["confidence"]),) if "confidence" in uncertainty else () ),
        )
    )
    ax.text(0.02, 0.96, aucs, transform=ax.transAxes, fontsize=5.8,
            color=INK_SECONDARY, va="top", ha="left")


def _panel_b(ax, uncertainty: dict[str, np.ndarray]) -> None:
    correct = uncertainty["correct"]
    score = uncertainty["drue_score"]
    score_distribution_panel(
        ax,
        {
            "Correctly graded": score[correct],
            "Misgraded": score[~correct],
        },
        colors=[CASCADE, MULTICLASS],
        xlabel="DRUE uncertainty score",
        title="Uncertainty and error",
        auroc_value=auroc(score, ~correct),
        log_x=True,
    )
    ax.text(
        0.02, 0.96,
        "dotted lines = medians",
        transform=ax.transAxes, fontsize=5.5, color=INK_SECONDARY, va="top", ha="left",
    )


def _panel_c(ax, uncertainty: dict[str, np.ndarray]) -> None:
    if "confidence" not in uncertainty:
        raise MissingInput(
            "panel C needs a 'confidence' column in uncertainty.csv (or a "
            "confidence column in cascade_predictions.csv)"
        )
    calibration = expected_calibration_error(
        uncertainty["correct"], uncertainty["confidence"], n_bins=10
    )
    calibration_panel(
        ax,
        calibration["bin_centre"],
        calibration["bin_accuracy"],
        calibration["bin_confidence"],
        calibration["bin_count"],
        calibration["ece"],
        color=CASCADE,
        title="Calibration",
    )


def _panel_d(ax, ablation: list[dict]) -> None:
    conditions = list(dict.fromkeys(row["condition"] for row in ablation))
    nodes = list(dict.fromkeys(row["node"] for row in ablation))
    lookup = {(r["condition"], r["node"]): r["recall"] for r in ablation}

    missing = [(c, n) for c in conditions for n in nodes if (c, n) not in lookup]
    if missing:
        log.warning("ablation.csv has no row for %d condition/node pair(s); "
                    "those bars are omitted", len(missing))

    series = {
        condition: [lookup.get((condition, node), np.nan) for node in nodes]
        for condition in conditions
    }
    grouped_bar_panel(
        ax,
        categories=nodes,
        series=series,
        colors=[CASCADE, MULTICLASS, GEOMETRIC, ACCENT],
        ylabel="Recall, %",
        title="Preprocessing and quality-control ablation",
        value_labels=False,
        ylim=(0, 118),
    )
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=2)


def build_figure(results: Results):
    uncertainty = results.uncertainty()
    ablation = results.ablation()

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 5.9))
    grid = fig.add_gridspec(
        2, 2, hspace=0.46, wspace=0.30,
        left=0.085, right=0.975, top=0.93, bottom=0.085,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    _panel_a(ax_a, uncertainty)
    panel_label(ax_a, "A", dx=-0.19)

    ax_b = fig.add_subplot(grid[0, 1])
    _panel_b(ax_b, uncertainty)
    panel_label(ax_b, "B", dx=-0.13)

    ax_c = fig.add_subplot(grid[1, 0])
    _panel_c(ax_c, uncertainty)
    panel_label(ax_c, "C", dx=-0.19)

    ax_d = fig.add_subplot(grid[1, 1])
    _panel_d(ax_d, ablation)
    panel_label(ax_d, "D", dx=-0.13)

    return fig


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("Render Figure 3 (uncertainty analysis).", epilog=__doc__)
    args = parser.parse_args(argv)
    results = setup(args)
    try:
        fig = build_figure(results)
    except MissingInput as exc:
        log.error("cannot render Figure 3: %s", exc)
        log.error("run with --demo to check the layout without the real artefacts")
        return 1
    save_figure(fig, args.out_dir, "figure3_uncertainty",
                formats=tuple(args.formats), demo=args.demo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
