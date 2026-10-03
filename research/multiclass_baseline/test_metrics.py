"""Tests for the metric and manifest code behind the reported numbers.

Deliberately torch-free, so they run anywhere:

    python -m pytest research/multiclass_baseline/test_metrics.py -q
    python research/multiclass_baseline/test_metrics.py        # no pytest needed
"""
from __future__ import annotations

import numpy as np
import pytest

from research.multiclass_baseline.labels import (
    GRADES,
    UnknownGradeError,
    crosses_operative_boundary,
    grade_to_index,
    is_adjacent,
    normalise_grade,
)
from research.multiclass_baseline.manifest import detect_view, extract_case_id
from research.multiclass_baseline.metrics import (
    auroc,
    classification_report,
    confusion_matrix,
    expected_calibration_error,
    mcnemar_exact,
    quadratic_weighted_kappa,
    risk_coverage_curve,
    selective_risk_auc,
    wilson_interval,
)

# --- labels -------------------------------------------------------------------


def test_grade_aliases_map_to_canonical_vocabulary():
    assert normalise_grade("SC Grade 2a") == "Grade 2A"
    assert normalise_grade("  grade   2B ") == "Grade 2B"
    assert normalise_grade("Normal") == "Normal"
    assert grade_to_index("SC Grade 3") == 4


def test_flexion_is_excluded_not_mislabelled():
    assert normalise_grade("Flexion type") is None
    assert grade_to_index("flexion") is None


def test_unknown_grade_raises_rather_than_dropping_silently():
    with pytest.raises(UnknownGradeError):
        normalise_grade("Grade 4")


def test_ordinal_helpers():
    g2a, g2b = GRADES.index("Grade 2A"), GRADES.index("Grade 2B")
    assert is_adjacent(g2a, g2b)
    assert crosses_operative_boundary(g2a, g2b)
    # 2A -> Grade 3 also changes management, but it is a distant error and is
    # counted separately.
    assert not crosses_operative_boundary(g2a, GRADES.index("Grade 3"))
    assert not is_adjacent(0, 2)


# --- agreement ----------------------------------------------------------------


def test_confusion_matrix_orientation_is_truth_by_prediction():
    cm = confusion_matrix([0, 0, 1], [0, 1, 1])
    assert cm[0, 0] == 1  # true Normal, predicted Normal
    assert cm[0, 1] == 1  # true Normal, predicted Grade 1
    assert cm[1, 1] == 1


def test_kappa_is_one_for_perfect_agreement():
    y = [0, 1, 2, 3, 4, 2, 1]
    assert quadratic_weighted_kappa(y, y) == pytest.approx(1.0)


def test_kappa_rewards_adjacent_errors_over_distant_ones():
    y_true = [0, 1, 2, 3, 4] * 4
    adjacent = [min(t + 1, 4) for t in y_true]
    distant = [(t + 3) % 5 for t in y_true]
    assert quadratic_weighted_kappa(y_true, adjacent) > quadratic_weighted_kappa(
        y_true, distant
    )


def test_kappa_uses_the_full_scale_even_when_classes_are_absent():
    # Only three of five grades present; the weight matrix must still span
    # all five, or values stop being comparable between cohorts.
    a = quadratic_weighted_kappa([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 2, 1])
    assert -1.0 <= a <= 1.0


def test_kappa_rejects_an_empty_sample():
    with pytest.raises(ValueError):
        quadratic_weighted_kappa([], [])


# --- intervals ----------------------------------------------------------------


def test_wilson_interval_brackets_the_point_estimate():
    low, high = wilson_interval(3, 6)
    assert low < 50.0 < high


def test_wilson_interval_stays_inside_zero_to_one_hundred():
    assert wilson_interval(0, 6)[0] == pytest.approx(0.0)
    assert wilson_interval(6, 6)[1] == pytest.approx(100.0)


def test_wilson_interval_is_wider_at_small_denominators():
    narrow = wilson_interval(50, 100)
    wide = wilson_interval(3, 6)
    assert (wide[1] - wide[0]) > (narrow[1] - narrow[0])


def test_wilson_interval_on_zero_denominator_is_nan():
    low, high = wilson_interval(0, 0)
    assert np.isnan(low) and np.isnan(high)


# --- report -------------------------------------------------------------------


def test_classification_report_counts_boundary_and_adjacent_errors():
    g2a, g2b = GRADES.index("Grade 2A"), GRADES.index("Grade 2B")
    y_true = [g2a, g2a, g2b, 0, 0]
    y_pred = [g2a, g2b, g2a, 0, 4]  # one 2A->2B, one 2B->2A, one Normal->Grade 3
    report = classification_report(y_true, y_pred, n_boot=0)
    assert report.operative_boundary_errors == 2
    assert report.n == 5
    assert report.overall_accuracy == pytest.approx(40.0)
    # Two of three errors are adjacent (the boundary pair); one is distant.
    assert report.adjacent_error_rate == pytest.approx(200 / 3)


def test_classification_report_coverage_is_relative_to_the_full_cohort():
    report = classification_report([0, 1, 2], [0, 1, 2], n_total=6, n_boot=0)
    assert report.coverage == pytest.approx(50.0)


def test_macro_recall_ignores_absent_classes():
    # Grades 2B and 3 absent: macro recall averages over the three present.
    report = classification_report([0, 0, 1, 2], [0, 1, 1, 2], n_boot=0)
    assert report.macro_recall == pytest.approx((50.0 + 100.0 + 100.0) / 3)


def test_operative_collapse_separates_operative_from_non_operative():
    # Grade 2B and Grade 3 are operative; Normal, 1, 2A are not.
    y_true = [0, 1, 2, 3, 4]
    y_pred = [0, 1, 2, 3, 4]
    report = classification_report(y_true, y_pred, n_boot=0)
    assert report.operative_collapse_sensitivity == pytest.approx(100.0)
    assert report.operative_collapse_specificity == pytest.approx(100.0)


def test_classification_report_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        classification_report([0, 1], [0], n_boot=0)


# --- paired comparison --------------------------------------------------------

def test_mcnemar_counts_discordant_pairs():
    a = [True, True, False, False, True]
    b = [True, False, True, False, False]
    result = mcnemar_exact(a, b)
    assert result["a_only_correct"] == 2
    assert result["b_only_correct"] == 1
    assert result["both_correct"] == 1
    assert result["both_wrong"] == 1


def test_mcnemar_on_identical_models_is_p_one():
    a = [True, False, True]
    assert mcnemar_exact(a, a)["p_value"] == pytest.approx(1.0)


def test_mcnemar_is_significant_when_all_discordance_is_one_sided():
    a = [True] * 12 + [False] * 2
    b = [False] * 12 + [False] * 2
    assert mcnemar_exact(a, b)["p_value"] < 0.01


# --- selective prediction and calibration ------------------------------------


def test_risk_coverage_ends_at_the_full_error_rate():
    correct = np.array([True, True, False, True])
    confidence = np.array([0.9, 0.8, 0.4, 0.7])
    curve = risk_coverage_curve(correct, confidence)
    assert curve["coverage"][-1] == pytest.approx(1.0)
    assert curve["risk"][-1] == pytest.approx(0.25)


def test_risk_coverage_abstains_on_the_least_confident_first():
    # The single error carries the lowest confidence, so risk is 0 until the
    # last case is admitted.
    correct = np.array([True, True, True, False])
    confidence = np.array([0.95, 0.9, 0.85, 0.1])
    curve = risk_coverage_curve(correct, confidence)
    assert curve["risk"][:3] == pytest.approx(0.0)
    assert selective_risk_auc(correct, confidence) < 0.1


def test_auroc_is_one_for_perfect_separation_and_half_for_none():
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    assert auroc(scores, np.array([False, False, True, True])) == pytest.approx(1.0)
    # Positives at the extremes: of the four positive/negative pairs, two rank
    # the right way round and two the wrong way.
    assert auroc(scores, np.array([True, False, False, True])) == pytest.approx(0.5)
    assert auroc(scores, np.array([True, True, False, False])) == pytest.approx(0.0)


def test_auroc_is_nan_when_one_class_is_missing():
    assert np.isnan(auroc([0.1, 0.2], [False, False]))


def test_ece_is_zero_for_a_perfectly_calibrated_model():
    # Every case confident at 0.95 and correct 95% of the time.
    correct = np.array([True] * 95 + [False] * 5)
    confidence = np.full(100, 0.95)
    result = expected_calibration_error(correct, confidence, n_bins=10)
    assert result["ece"] == pytest.approx(0.0, abs=1e-9)


def test_ece_detects_overconfidence():
    correct = np.array([True] * 50 + [False] * 50)
    confidence = np.full(100, 0.99)  # 99% confident, 50% right
    result = expected_calibration_error(correct, confidence, n_bins=10)
    assert result["ece"] == pytest.approx(0.49, abs=0.01)


def test_ece_bin_counts_sum_to_the_sample_size():
    rng = np.random.default_rng(0)
    confidence = rng.uniform(0.2, 1.0, 200)
    correct = rng.random(200) < confidence
    result = expected_calibration_error(correct, confidence, n_bins=10)
    assert int(result["bin_count"].sum()) == 200


# --- manifest -----------------------------------------------------------------


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("A3517.png", "A3517"),
        ("A326 AP.png", "A326"),
        ("A162 AP2.png", "A162"),
        ("A35 LAT.png", "A35"),
        # A241 and A241A are different patients with different grades.
        ("A241 AP.png", "A241"),
        ("A241A AP.png", "A241A"),
        ("A594B AP.png", "A594B"),
        # Left and right elbows are separate studies.
        ("A118 (L).png", "A118(L)"),
        ("A118(R).png", "A118(R)"),
    ],
)
def test_case_id_extraction_matches_the_kkh_filename_grammar(filename, expected):
    assert extract_case_id(filename) == expected


def test_case_id_is_none_when_the_filename_has_no_id():
    assert extract_case_id("notes.png") is None


@pytest.mark.parametrize(
    "path,expected",
    [
        (r"...\SC Grade 2a\SC Grade 2a AP\A3517.png", "AP"),
        (r"...\SC Grade 2a\SC Grade 2a LAT\A3517.png", "LAT"),
        ("Normal/Normal LAT/A118.png", "LAT"),
        # The filename decides when the folder does not say.
        ("images/A326 AP.png", "AP"),
    ],
)
def test_view_detection(path, expected):
    assert detect_view(path) == expected


def test_view_detection_returns_none_when_undetermined():
    assert detect_view("images/A326.png") is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
