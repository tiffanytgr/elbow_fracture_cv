"""Supplementary figures: the material moved out of the main four.

  S1  per-node ROC curves (KKH held-out)
  S2  training curves: autoencoder pretraining and the baseline's two stages
  S3  Grad-CAM examples, a grid of representative cases
  S4  full regularisation ablation (label smoothing, MixUp, dropout)
  S5  failure cases, with the reference and predicted grade

Each renders independently and is skipped with a message if its inputs are
absent, so a partial results directory still produces the figures it can.

Inputs, all under ``--results``:

  roc_curves.csv       node,fpr,tpr[,auc]
  training_curve.csv   epoch,stage,train_loss,val_loss,val_accuracy (from train.py)
  autoencoder_curve.csv epoch,train_loss,val_loss
  ablation_full.csv    configuration,node,metric_value[,seed]
  images/gradcam/*.png  Grad-CAM overlays; filename is used as the caption
  failure_cases.csv    case_id,true_grade,pred_grade[,confidence,note]
  images/failures/<case_id>.png

    python -m research.figures.appendix_figures --results research/outputs
    python -m research.figures.appendix_figures --demo --only s2
"""
from __future__ import annotations

import logging
import sys
from collections import OrderedDict

import matplotlib.pyplot as plt
import numpy as np

from .cli import build_parser, setup
from .data import MissingInput, Results
from .panels import grouped_bar_panel, image_panel
from .style import (
    ACCENT,
    CASCADE,
    GEOMETRIC,
    INK_MUTED,
    INK_SECONDARY,
    MULTICLASS,
    WIDTH_DOUBLE,
    WIDTH_ONE_HALF,
    panel_label,
    save_figure,
    strip_spines,
    subtle_grid,
)

log = logging.getLogger(__name__)

SERIES_COLORS = [CASCADE, MULTICLASS, GEOMETRIC, ACCENT]

#: ROC needs a little more than one column to fit its legend.
WIDTH_ROC = 3.6


# --- S1: ROC ------------------------------------------------------------------


def figure_s1_roc(results: Results):
    rows = results.table("roc_curves.csv")
    for key in ("node", "fpr", "tpr"):
        if key not in rows[0]:
            raise MissingInput(f"roc_curves.csv has no {key!r} column")

    curves: dict[str, dict[str, list[float]]] = OrderedDict()
    aucs: dict[str, float] = {}
    for row in rows:
        node = row["node"]
        curve = curves.setdefault(node, {"fpr": [], "tpr": []})
        curve["fpr"].append(float(row["fpr"]))
        curve["tpr"].append(float(row["tpr"]))
        if row.get("auc"):
            aucs[node] = float(row["auc"])

    fig, ax = plt.subplots(figsize=(WIDTH_ROC, 3.2))
    ax.plot([0, 1], [0, 1], color=INK_MUTED, linewidth=0.8, linestyle="--",
            label="Chance", zorder=2)
    for i, (node, curve) in enumerate(curves.items()):
        order = np.argsort(curve["fpr"])
        label = node if node not in aucs else f"{node} (AUC {aucs[node]:.2f})"
        ax.plot(
            np.array(curve["fpr"])[order], np.array(curve["tpr"])[order],
            color=SERIES_COLORS[i % len(SERIES_COLORS)],
            linestyle=("-", "--", "-.", ":")[i % 4],
            label=label, zorder=3,
        )
    ax.set_xlabel("1 − specificity")
    ax.set_ylabel("Sensitivity")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title("Per-node ROC, KKH held-out cohort")
    subtle_grid(ax, axis="both")
    strip_spines(ax)
    ax.legend(loc="lower right")
    return fig


# --- S2: training curves ------------------------------------------------------


def figure_s2_training(results: Results):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH_ONE_HALF, 2.5))

    drawn = False
    try:
        rows = results.table("autoencoder_curve.csv")
        epochs = [float(r["epoch"]) for r in rows]
        axes[0].plot(epochs, [float(r["train_loss"]) for r in rows],
                     color=CASCADE, label="Training")
        axes[0].plot(epochs, [float(r["val_loss"]) for r in rows],
                     color=MULTICLASS, linestyle="--", label="Validation")
        best = int(np.argmin([float(r["val_loss"]) for r in rows]))
        axes[0].axvline(epochs[best], color=INK_MUTED, linewidth=0.7, linestyle=":")
        axes[0].annotate(
            f"best epoch {epochs[best]:.0f}", xy=(epochs[best], float(rows[best]["val_loss"])),
            xytext=(5, 14), textcoords="offset points", fontsize=5.8, color=INK_SECONDARY,
        )
        drawn = True
    except MissingInput as exc:
        log.warning("S2 left panel skipped: %s", exc)
        axes[0].text(0.5, 0.5, "autoencoder_curve.csv\nnot found", ha="center",
                     va="center", transform=axes[0].transAxes, fontsize=6, color=INK_MUTED)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Reconstruction loss (MSE)")
    axes[0].set_title("Autoencoder pretraining")
    if drawn:
        axes[0].legend(loc="upper right")
    subtle_grid(axes[0], axis="both")
    strip_spines(axes[0])
    panel_label(axes[0], "A", dx=-0.24)

    try:
        rows = results.table("training_curve.csv")
        epochs = [float(r["epoch"]) for r in rows]
        axes[1].plot(epochs, [float(r["train_loss"]) for r in rows],
                     color=CASCADE, label="Training loss")
        axes[1].plot(epochs, [float(r["val_loss"]) for r in rows],
                     color=MULTICLASS, linestyle="--", label="Validation loss")
        if "stage" in rows[0]:
            stages = [int(float(r["stage"])) for r in rows]
            boundary = next(
                (e for e, s in zip(epochs, stages) if s == 2), None
            )
            if boundary is not None:
                axes[1].axvline(boundary - 0.5, color=INK_MUTED, linewidth=0.7,
                                linestyle=":")
                axes[1].text(
                    boundary - 0.5, axes[1].get_ylim()[1], " stage 2: unfrozen",
                    fontsize=5.8, color=INK_SECONDARY, va="top", ha="left",
                )
        axes[1].legend(loc="upper right")
    except MissingInput as exc:
        log.warning("S2 right panel skipped: %s", exc)
        axes[1].text(0.5, 0.5, "training_curve.csv\nnot found", ha="center",
                     va="center", transform=axes[1].transAxes, fontsize=6, color=INK_MUTED)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Multiclass baseline")
    subtle_grid(axes[1], axis="both")
    strip_spines(axes[1])
    panel_label(axes[1], "B", dx=-0.24)

    fig.tight_layout()
    return fig


# --- S3: Grad-CAM grid --------------------------------------------------------


def figure_s3_gradcam(results: Results, columns: int = 4):
    directory = results.root / "images" / "gradcam"
    paths = sorted(p for p in directory.glob("*.png")) if directory.is_dir() else []
    if not paths:
        raise MissingInput(f"no PNGs in {directory}")
    paths = paths[: columns * 3]  # at most three rows

    rows = (len(paths) + columns - 1) // columns
    fig, axes = plt.subplots(
        rows, columns, figsize=(WIDTH_DOUBLE, 1.9 * rows), squeeze=False
    )
    for ax, path in zip(axes.ravel(), paths):
        image_panel(ax, path, caption=path.stem.replace("_", " "))
    for ax in axes.ravel()[len(paths):]:
        ax.axis("off")
    fig.suptitle("Grad-CAM overlays for representative cases", fontsize=7.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# --- S4: full regularisation ablation ----------------------------------------


def figure_s4_ablation(results: Results):
    rows = results.table("ablation_full.csv")
    for key in ("configuration", "node", "metric_value"):
        if key not in rows[0]:
            raise MissingInput(f"ablation_full.csv has no {key!r} column")

    configurations = list(dict.fromkeys(r["configuration"] for r in rows))
    nodes = list(dict.fromkeys(r["node"] for r in rows))
    if len(configurations) > len(SERIES_COLORS):
        log.warning(
            "%d configurations but only %d distinct hues; the ablation is "
            "drawn as a heatmap instead of grouped bars",
            len(configurations), len(SERIES_COLORS),
        )
        return _ablation_heatmap(rows, configurations, nodes)

    # Average over seeds where several are present.
    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault(
            (row["configuration"], row["node"]), []
        ).append(float(row["metric_value"]))

    fig, ax = plt.subplots(figsize=(WIDTH_ONE_HALF, 2.8))
    grouped_bar_panel(
        ax,
        categories=nodes,
        series={
            config: [
                float(np.mean(grouped.get((config, node), [np.nan])))
                for node in nodes
            ]
            for config in configurations
        },
        colors=SERIES_COLORS,
        ylabel="Recall, %",
        title="Regularisation ablation",
        value_labels=False,
    )
    fig.tight_layout()
    return fig


def _ablation_heatmap(rows, configurations, nodes):
    """Fallback for more configurations than the categorical palette allows."""
    from .style import GRADE_CMAP, readable_text_color

    matrix = np.full((len(configurations), len(nodes)), np.nan)
    for row in rows:
        i = configurations.index(row["configuration"])
        j = nodes.index(row["node"])
        value = float(row["metric_value"])
        matrix[i, j] = value if np.isnan(matrix[i, j]) else (matrix[i, j] + value) / 2

    fig, ax = plt.subplots(figsize=(WIDTH_ONE_HALF, 0.28 * len(configurations) + 1.4))
    finite = matrix[np.isfinite(matrix)]
    image = ax.imshow(
        matrix, cmap=GRADE_CMAP, aspect="auto",
        vmin=float(finite.min()) if finite.size else 0,
        vmax=float(finite.max()) if finite.size else 1,
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if np.isfinite(matrix[i, j]):
                ax.text(
                    j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=5.5,
                    color=readable_text_color(image.cmap(image.norm(matrix[i, j]))),
                )
    ax.set_xticks(range(len(nodes)), labels=nodes)
    ax.set_yticks(range(len(configurations)), labels=configurations)
    ax.set_title("Regularisation ablation")
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, ax=ax, fraction=0.04, pad=0.02)
    bar.set_label("Recall, %", fontsize=6.5)
    bar.outline.set_visible(False)
    fig.tight_layout()
    return fig


# --- S5: failure cases --------------------------------------------------------


def figure_s5_failures(results: Results, columns: int = 4):
    rows = results.table("failure_cases.csv")
    for key in ("case_id", "true_grade", "pred_grade"):
        if key not in rows[0]:
            raise MissingInput(f"failure_cases.csv has no {key!r} column")
    rows = rows[: columns * 2]

    n_rows = (len(rows) + columns - 1) // columns
    fig, axes = plt.subplots(
        n_rows, columns, figsize=(WIDTH_DOUBLE, 2.1 * n_rows), squeeze=False
    )
    for ax, row in zip(axes.ravel(), rows):
        path = results.root / "images" / "failures" / f"{row['case_id']}.png"
        caption = f"Reference {row['true_grade']} → predicted {row['pred_grade']}"
        if row.get("confidence"):
            caption += f"\nconfidence {float(row['confidence']):.2f}"
        image_panel(
            ax, path if path.exists() else None, caption=caption,
            placeholder=row["case_id"],
        )
    for ax in axes.ravel()[len(rows):]:
        ax.axis("off")
    fig.suptitle("Representative failure cases", fontsize=7.5, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


FIGURES = {
    "s1": ("supplementary_s1_roc", figure_s1_roc),
    "s2": ("supplementary_s2_training", figure_s2_training),
    "s3": ("supplementary_s3_gradcam", figure_s3_gradcam),
    "s4": ("supplementary_s4_ablation", figure_s4_ablation),
    "s5": ("supplementary_s5_failures", figure_s5_failures),
}


def main(argv: list[str] | None = None) -> int:
    parser = build_parser("Render the supplementary figures.", epilog=__doc__)
    parser.add_argument(
        "--only", nargs="+", choices=sorted(FIGURES),
        help="Render only these supplementary figures (default: all available).",
    )
    args = parser.parse_args(argv)
    results = setup(args)

    wanted = args.only or sorted(FIGURES)
    rendered, skipped = 0, []
    for key in wanted:
        stem, builder = FIGURES[key]
        try:
            fig = builder(results)
        except MissingInput as exc:
            skipped.append(f"{key}: {exc}")
            continue
        save_figure(fig, args.out_dir, stem, formats=tuple(args.formats), demo=args.demo)
        rendered += 1

    for message in skipped:
        log.warning("skipped %s", message)
    log.info("rendered %d of %d supplementary figures", rendered, len(wanted))
    # Missing optional inputs are not a failure: a partial results directory is
    # the normal case while the analysis is still being assembled.
    return 0


if __name__ == "__main__":
    sys.exit(main())
