"""Metrics for the end-to-end 5-class evaluation and the cascade comparison.

Only numpy and scipy are required, so this module imports without torch and can
be used by the figure scripts on a machine with no GPU stack installed.

Conventions
-----------
* ``y_true`` / ``y_pred`` are integer arrays of class indices in ordinal order
  (see labels.GRADES).
* Confidence intervals are two-sided 95% by default.
* Proportion CIs use the Wilson score interval, which behaves sensibly at the
  small denominators in this study (the Grade 2B held-out subset is n=6, where
  a normal-approximation interval is meaningless).
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field

import numpy as np
from scipy import stats

from .labels import (
    GRADES,
    NUM_CLASSES,
    OPERATIVE_GRADES,
    crosses_operative_boundary,
    is_adjacent,
)

# --- interval estimation ------------------------------------------------------


def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, as a percentage."""
    if total <= 0:
        return (float("nan"), float("nan"))
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p = successes / total
    denom = 1 + z**2 / total
    centre = (p + z**2 / (2 * total)) / denom
    half = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denom
    return (max(0.0, centre - half) * 100, min(1.0, centre + half) * 100)


def bootstrap_ci(
    statistic,
    *arrays: np.ndarray,
    n_boot: int = 2000,
    confidence: float = 0.95,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap CI for a statistic of paired per-case arrays.

    Resamples cases (not predictions) so the interval reflects patient-level
    sampling variability. Resamples that make the statistic undefined -- a draw
    containing a single class, for which kappa has no value -- are skipped.
    """
    arrays = tuple(np.asarray(a) for a in arrays)
    n = len(arrays[0])
    if n == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            value = float(statistic(*(a[idx] for a in arrays)))
        except (ValueError, ZeroDivisionError):
            continue
        if np.isfinite(value):
            values.append(value)
    if not values:
        return (float("nan"), float("nan"))
    lower = float(np.percentile(values, (1 - confidence) / 2 * 100))
    upper = float(np.percentile(values, (1 + confidence) / 2 * 100))
    return (lower, upper)


# --- agreement ----------------------------------------------------------------


def confusion_matrix(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = NUM_CLASSES
) -> np.ndarray:
    """Rows are reference standard, columns are prediction."""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)):
        cm[t, p] += 1
    return cm


def quadratic_weighted_kappa(
    y_true: Sequence[int], y_pred: Sequence[int], num_classes: int = NUM_CLASSES
) -> float:
    """Cohen's kappa with quadratic weights, over the fixed ordinal scale.

    The weight matrix spans all ``num_classes`` levels regardless of which
    appear in the sample, so values are comparable between cohorts and between
    the cascade and the baseline.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        raise ValueError("quadratic_weighted_kappa on an empty sample")
    levels = np.arange(num_classes)
    weights = (levels[:, None] - levels[None, :]) ** 2 / (num_classes - 1) ** 2

    observed = confusion_matrix(y_true, y_pred, num_classes).astype(float)
    observed /= observed.sum()
    true_marginal = np.bincount(y_true, minlength=num_classes) / y_true.size
    pred_marginal = np.bincount(y_pred, minlength=num_classes) / y_pred.size
    expected = np.outer(true_marginal, pred_marginal)

    denom = float((weights * expected).sum())
    if denom == 0:
        raise ValueError("quadratic_weighted_kappa undefined: zero expected disagreement")
    return 1.0 - float((weights * observed).sum()) / denom


# --- per-class and aggregate summaries ---------------------------------------


@dataclass
class ClassMetrics:
    grade: str
    support: int
    recall: float
    recall_ci: tuple[float, float]
    precision: float
    precision_ci: tuple[float, float]
    f1: float
    true_positives: int
    predicted_positives: int


@dataclass
class ClassificationReport:
    n: int
    overall_accuracy: float
    overall_accuracy_ci: tuple[float, float]
    macro_recall: float
    quadratic_kappa: float
    quadratic_kappa_ci: tuple[float, float]
    weighted_f1: float
    confusion_matrix: list[list[int]]
    per_class: list[ClassMetrics]
    adjacent_error_rate: float
    distant_error_rate: float
    operative_boundary_errors: int
    operative_collapse_accuracy: float
    operative_collapse_sensitivity: float
    operative_collapse_specificity: float
    coverage: float | None = None
    n_total: int | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def classification_report(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int = NUM_CLASSES,
    n_total: int | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> ClassificationReport:
    """Full end-to-end report for one cohort configuration.

    ``n_total`` is the size of the cohort *before* any abstention, so that
    accuracy on a retained subset is always reported next to its coverage --
    the report's Table 3 caption makes the same point.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}")
    n = int(y_true.size)
    if n == 0:
        raise ValueError("classification_report on an empty cohort")

    cm = confusion_matrix(y_true, y_pred, num_classes)
    correct = int(np.trace(cm))

    per_class: list[ClassMetrics] = []
    recalls: list[float] = []
    for idx in range(num_classes):
        support = int(cm[idx].sum())
        predicted = int(cm[:, idx].sum())
        tp = int(cm[idx, idx])
        recall = tp / support if support else float("nan")
        precision = tp / predicted if predicted else float("nan")
        if np.isnan(recall) or np.isnan(precision) or (recall + precision) == 0:
            f1 = float("nan")
        else:
            f1 = 2 * recall * precision / (recall + precision)
        if support:
            recalls.append(recall)
        per_class.append(
            ClassMetrics(
                grade=GRADES[idx],
                support=support,
                recall=recall * 100 if support else float("nan"),
                recall_ci=wilson_interval(tp, support),
                precision=precision * 100 if predicted else float("nan"),
                precision_ci=wilson_interval(tp, predicted),
                f1=f1,
                true_positives=tp,
                predicted_positives=predicted,
            )
        )

    supports = np.array([c.support for c in per_class], dtype=float)
    f1s = np.array([0.0 if np.isnan(c.f1) else c.f1 for c in per_class])
    weighted_f1 = float((f1s * supports).sum() / supports.sum()) if supports.sum() else float("nan")

    errors = y_true != y_pred
    n_errors = int(errors.sum())
    adjacent = sum(
        1 for t, p in zip(y_true[errors], y_pred[errors]) if is_adjacent(int(t), int(p))
    )
    boundary = sum(
        1
        for t, p in zip(y_true[errors], y_pred[errors])
        if crosses_operative_boundary(int(t), int(p))
    )

    op_true = np.isin(y_true, list(OPERATIVE_GRADES))
    op_pred = np.isin(y_pred, list(OPERATIVE_GRADES))
    op_tp = int((op_true & op_pred).sum())
    op_tn = int((~op_true & ~op_pred).sum())
    op_pos = int(op_true.sum())
    op_neg = int((~op_true).sum())

    kappa_ci = bootstrap_ci(
        lambda t, p: quadratic_weighted_kappa(t, p, num_classes),
        y_true,
        y_pred,
        n_boot=n_boot,
        seed=seed,
    )

    return ClassificationReport(
        n=n,
        overall_accuracy=correct / n * 100,
        overall_accuracy_ci=wilson_interval(correct, n),
        macro_recall=float(np.mean(recalls)) * 100 if recalls else float("nan"),
        quadratic_kappa=quadratic_weighted_kappa(y_true, y_pred, num_classes),
        quadratic_kappa_ci=kappa_ci,
        weighted_f1=weighted_f1,
        confusion_matrix=cm.tolist(),
        per_class=per_class,
        adjacent_error_rate=adjacent / n_errors * 100 if n_errors else 0.0,
        distant_error_rate=(n_errors - adjacent) / n_errors * 100 if n_errors else 0.0,
        operative_boundary_errors=boundary,
        operative_collapse_accuracy=(op_tp + op_tn) / n * 100,
        operative_collapse_sensitivity=op_tp / op_pos * 100 if op_pos else float("nan"),
        operative_collapse_specificity=op_tn / op_neg * 100 if op_neg else float("nan"),
        coverage=(n / n_total * 100) if n_total else None,
        n_total=n_total,
    )


# --- paired model comparison --------------------------------------------------


def mcnemar_exact(model_a_correct: Sequence[bool], model_b_correct: Sequence[bool]) -> dict:
    """Exact (binomial) McNemar test on paired exact-grade correctness.

    The exact test is used rather than the chi-square approximation because the
    discordant count at this sample size is small.
    """
    a = np.asarray(model_a_correct, dtype=bool)
    b = np.asarray(model_b_correct, dtype=bool)
    if a.shape != b.shape:
        raise ValueError("McNemar requires paired arrays of equal length")
    a_only = int((a & ~b).sum())
    b_only = int((~a & b).sum())
    discordant = a_only + b_only
    if discordant == 0:
        p_value = 1.0
    else:
        p_value = float(
            min(1.0, 2 * stats.binom.cdf(min(a_only, b_only), discordant, 0.5))
        )
    return {
        "both_correct": int((a & b).sum()),
        "both_wrong": int((~a & ~b).sum()),
        "a_only_correct": a_only,
        "b_only_correct": b_only,
        "discordant": discordant,
        "p_value": p_value,
        "test": "exact McNemar (two-sided binomial)",
    }


def paired_kappa_difference(
    y_true: Sequence[int],
    y_pred_a: Sequence[int],
    y_pred_b: Sequence[int],
    num_classes: int = NUM_CLASSES,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Bootstrap CI for kappa(a) - kappa(b) on the same cases.

    Resampling the same case indices for both models preserves the pairing, so
    the interval is for the *difference*, not the difference of two independent
    intervals.
    """
    y_true = np.asarray(y_true, dtype=int)
    a = np.asarray(y_pred_a, dtype=int)
    b = np.asarray(y_pred_b, dtype=int)
    point = quadratic_weighted_kappa(y_true, a, num_classes) - quadratic_weighted_kappa(
        y_true, b, num_classes
    )
    lower, upper = bootstrap_ci(
        lambda t, pa, pb: (
            quadratic_weighted_kappa(t, pa, num_classes)
            - quadratic_weighted_kappa(t, pb, num_classes)
        ),
        y_true,
        a,
        b,
        n_boot=n_boot,
        seed=seed,
    )
    return {"difference": point, "ci": (lower, upper)}


# --- selective prediction and calibration ------------------------------------


def risk_coverage_curve(
    correct: Sequence[bool], confidence: Sequence[float]
) -> dict[str, np.ndarray]:
    """Error rate as a function of coverage, abstaining on lowest confidence first.

    Returns coverage (fraction retained), risk (error rate on the retained
    subset) and the confidence threshold at each operating point.
    """
    correct = np.asarray(correct, dtype=bool)
    confidence = np.asarray(confidence, dtype=float)
    if correct.shape != confidence.shape:
        raise ValueError("correct and confidence must have the same length")
    order = np.argsort(-confidence, kind="stable")  # most confident first
    ranked = correct[order]
    n = ranked.size
    kept = np.arange(1, n + 1)
    errors = np.cumsum(~ranked)
    return {
        "coverage": kept / n,
        "risk": errors / kept,
        "threshold": confidence[order],
    }


def selective_risk_auc(correct: Sequence[bool], confidence: Sequence[float]) -> float:
    """Area under the risk-coverage curve. Lower is better; 0 is perfect ranking."""
    curve = risk_coverage_curve(correct, confidence)
    return float(np.trapezoid(curve["risk"], curve["coverage"]))


def auroc(scores: Sequence[float], positive: Sequence[bool]) -> float:
    """Rank-based AUROC, ties averaged. Used for uncertainty-vs-error discrimination."""
    scores = np.asarray(scores, dtype=float)
    positive = np.asarray(positive, dtype=bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = stats.rankdata(scores)
    return float((ranks[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def expected_calibration_error(
    correct: Sequence[bool], confidence: Sequence[float], n_bins: int = 10
) -> dict:
    """Equal-width-bin ECE plus the per-bin values needed for a reliability plot."""
    correct = np.asarray(correct, dtype=bool)
    confidence = np.asarray(confidence, dtype=float)
    n = correct.size
    if n == 0:
        raise ValueError("expected_calibration_error on an empty sample")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Bin 0 collects everything at or below the first edge; the last bin is closed.
    bin_index = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, n_bins - 1)

    centres, accuracies, confidences, counts = [], [], [], []
    ece = 0.0
    for b in range(n_bins):
        mask = bin_index == b
        count = int(mask.sum())
        centres.append((edges[b] + edges[b + 1]) / 2)
        counts.append(count)
        if count == 0:
            accuracies.append(float("nan"))
            confidences.append(float("nan"))
            continue
        acc = float(correct[mask].mean())
        conf = float(confidence[mask].mean())
        accuracies.append(acc)
        confidences.append(conf)
        ece += count / n * abs(acc - conf)
    return {
        "ece": ece,
        "bin_centre": np.array(centres),
        "bin_accuracy": np.array(accuracies),
        "bin_confidence": np.array(confidences),
        "bin_count": np.array(counts),
        "n_bins": n_bins,
    }
