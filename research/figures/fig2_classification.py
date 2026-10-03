"""Figure 2 -- end-to-end classification results.

Four panels on a double-column canvas:

  A  5x5 confusion matrix, hierarchical cascade, KKH held-out cohort
  B  5x5 confusion matrix, end-to-end multiclass baseline, same patients
  C  per-grade recall with 95% Wilson CIs, cascade vs baseline, and the
     paired summary statistics (McNemar, kappa difference)
  D  error composition: adjacent, 2A<->2B operative-boundary, and distant

Panels A and B are drawn on the same patients so the reader can read one
against the other; panel C's tests are paired for the same reason.

    python -m research.figures.fig2_classification --results research/outputs
    python -m research.figures.fig2_classification --demo    # layout check
"""
from __future__ import annotations

import logging
import sys

import matplotlib.pyplot as plt
import numpy as np

from research.multiclass_baseline.labels import GRADES, SHORT_GRADES
from research.multiclass_baseline.metrics import (
    classification_report,
    mcnemar_exact,
    paired_kappa_difference,
    wilson_interval,
)

from .cli import build_parser, setup
from .data import MissingInput, Results
from .panels import (
    confusion_panel,
    error_breakdown,
    error_taxonomy_panel,
    grouped_bar_panel,
)
from .style import (
    CASCADE,
    INK_SECONDARY,
    MULTICLASS,
    WIDTH_DOUBLE,
    panel_label,
    save_figure,
)

log = logging.getLogger(__name__)

CASCADE_NAME = "Hierarchical cascade"
MULTICLASS_NAME = "End-to-end multiclass"


def _align(results: Results) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Reduce both prediction sets to the patients they share."""
    cascade = results.predictions("cascade")
    multi = results.predictions("multiclass")
    shared = sorted(set(cascade["case_id"]) & set(multi["case_id"]))
    if not shared:
        raise MissingInput(
            "cascade and multiclass prediction files share no case_id -- panels "
            "B-D need both models scored on the same patients"
        )
    c_idx = {c: i for i, c in enumerate(cascade["case_id"])}
    m_idx = {c: i for i, c in enumerate(multi["case_id"])}

    y_true = np.array([cascade["y_true"][c_idx[c]] for c in shared])
    y_true_m = np.array([multi["y_true"][m_idx[c]] for c in shared])
    if not np.array_equal(y_true, y_true_m):
        raise MissingInput(
            "the two prediction files disagree on the reference standard for at "
            "least one shared case; the comparison would be meaningless"
        )
    pred_cascade = np.array([cascade["y_pred"][c_idx[c]] for c in shared])
    pred_multi = np.array([multi["y_pred"][m_idx[c]] for c in shared])
    log.info("Figure 2 drawn on %d shared patients", len(shared))
    return y_true, pred_cascade, pred_multi


def _recall_series(
    y_true: np.ndarray, y_pred: np.ndarray
) -> tuple[list[float], list[tuple[float, float]], list[int]]:
    recalls: list[float] = []
    cis: list[tuple[float, float]] = []
    supports: list[int] = []
    for idx in range(len(GRADES)):
        mask = y_true == idx
        support = int(mask.sum())
        hits = int((y_pred[mask] == idx).sum())
        supports.append(support)
        recalls.append(hits / support * 100 if support else np.nan)
        cis.append(wilson_interval(hits, support))
    return recalls, cis, supports


def _summary_text(y_true, pred_cascade, pred_multi) -> str:
    cascade = classification_report(y_true, pred_cascade, n_boot=0)
    multi = classification_report(y_true, pred_multi, n_boot=0)
    mcnemar = mcnemar_exact(y_true == pred_cascade, y_true == pred_multi)
    kappa = paired_kappa_difference(y_true, pred_cascade, pred_multi, n_boot=1000)
    return (
        f"Exact-grade accuracy: {cascade.overall_accuracy:.1f}% vs "
        f"{multi.overall_accuracy:.1f}%\n"
        f"Quadratic-weighted κ: {cascade.quadratic_kappa:.3f} vs "
        f"{multi.quadratic_kappa:.3f} "
        f"(Δ {kappa['difference']:+.3f}; 95% CI {kappa['ci'][0]:+.3f}, {kappa['ci'][1]:+.3f})\n"
        f"Exact McNemar on per-patient correctness: P = {mcnemar['p_value']:.3f} "
        f"({mcnemar['a_only_correct']} vs {mcnemar['b_only_correct']} discordant)"
    )


def build_figure(results: Results):
    y_true, pred_cascade, pred_multi = _align(results)

    fig = plt.figure(figsize=(WIDTH_DOUBLE, 6.5))
    grid = fig.add_gridspec(
        2, 2, hspace=0.52, wspace=0.34,
        height_ratios=[1.0, 0.92], left=0.085, right=0.97, top=0.94, bottom=0.07,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    confusion_panel(
        ax_a,
        _confusion(y_true, pred_cascade),
        title=CASCADE_NAME,
        class_labels=SHORT_GRADES,
    )
    panel_label(ax_a, "A", dx=-0.26)

    ax_b = fig.add_subplot(grid[0, 1])
    confusion_panel(
        ax_b,
        _confusion(y_true, pred_multi),
        title=MULTICLASS_NAME,
        class_labels=SHORT_GRADES,
        show_colorbar=True,
    )
    panel_label(ax_b, "B", dx=-0.26)

    ax_c = fig.add_subplot(grid[1, 0])
    c_recall, c_ci, supports = _recall_series(y_true, pred_cascade)
    m_recall, m_ci, _ = _recall_series(y_true, pred_multi)
    grouped_bar_panel(
        ax_c,
        categories=list(SHORT_GRADES),
        series={CASCADE_NAME: c_recall, MULTICLASS_NAME: m_recall},
        colors=[CASCADE, MULTICLASS],
        errors={CASCADE_NAME: c_ci, MULTICLASS_NAME: m_ci},
        ylabel="Per-grade recall, %",
        title="Cascade versus multiclass",
        supports=supports,
        ylim=(0, 118),
    )
    panel_label(ax_c, "C", dx=-0.19)
    ax_c.text(
        0.0, -0.40, _summary_text(y_true, pred_cascade, pred_multi),
        transform=ax_c.transAxes, fontsize=6, color=INK_SECONDARY,
        va="top", ha="left", linespacing=1.6,
    )

    ax_d = fig.add_subplot(grid[1, 1])
    error_taxonomy_panel(
        ax_d,
        {
            CASCADE_NAME: error_breakdown(y_true, pred_cascade),
            MULTICLASS_NAME: error_breakdown(y_true, pred_multi),
        },
        title="Error composition",
    )
    ax_d.set_xticks(
        range(2), labels=["Cascade", "Multiclass"],
    )
    panel_label(ax_d, "D", dx=-0.19)
    ax_d.text(
        0.0, -0.40,
        "Adjacent errors differ by one Gartland grade. The 2A↔2B subset is\n"
        "shown separately because it is the boundary that changes management\n"
        "from immobilisation to operative fixation.",
        transform=ax_d.transAxes, fontsize=6, color=INK_SECONDARY,
        va="top", ha="left", linespacing=1.6,
    )
    return fig


def _confusion(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    from research.multiclass_baseline.metrics import confusion_matrix

    return confusion_matrix(y_true, y_pred)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("Render Figure 2 (classification results).", epilog=__doc__)
    args = parser.parse_args(argv)
    results = setup(args)
    try:
        fig = build_figure(results)
    except MissingInput as exc:
        log.error("cannot render Figure 2: %s", exc)
        log.error("run with --demo to check the layout without the real artefacts")
        return 1
    save_figure(fig, args.out_dir, "figure2_classification",
                formats=tuple(args.formats), demo=args.demo)
    return 0


if __name__ == "__main__":
    sys.exit(main())
