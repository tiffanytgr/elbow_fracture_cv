"""Canonical 5-class Gartland label vocabulary.

Single source of truth for the end-to-end multiclass baseline so that the
baseline, the cascade comparison, and the figure scripts all agree on class
order. Ordinal order matters: quadratic-weighted kappa and the
adjacent-vs-distant error analysis both assume GRADES is sorted by severity.
"""
from __future__ import annotations

import re

# Ordinal severity order. Index == integer label used everywhere downstream.
GRADES: tuple[str, ...] = ("Normal", "Grade 1", "Grade 2A", "Grade 2B", "Grade 3")
GRADE_TO_IDX: dict[str, int] = {g: i for i, g in enumerate(GRADES)}
NUM_CLASSES = len(GRADES)

# Short forms for axis tick labels, where "Grade 2A" is too wide.
SHORT_GRADES: tuple[str, ...] = ("Normal", "G1", "G2A", "G2B", "G3")

# The operative boundary of clinical interest: Grade 2A is typically managed
# non-operatively, Grade 2B operatively. Errors that cross this pair are
# reported separately in Figure 2C.
OPERATIVE_BOUNDARY: tuple[int, int] = (GRADE_TO_IDX["Grade 2A"], GRADE_TO_IDX["Grade 2B"])

# Grades for which the clinical pathway is operative. Used for the
# operative-vs-nonoperative collapse reported alongside the 5x5 matrix.
OPERATIVE_GRADES: frozenset[int] = frozenset(
    {GRADE_TO_IDX["Grade 2B"], GRADE_TO_IDX["Grade 3"]}
)

# Raw label strings seen in the KKH manifests ("SC Grade 2a", folder names,
# free-text variants) mapped onto the canonical vocabulary. Keys are matched
# after casefolding and whitespace collapsing.
_ALIASES: dict[str, str] = {
    "normal": "Normal",
    "sc normal": "Normal",
    "no fracture": "Normal",
    "grade 1": "Grade 1",
    "sc grade 1": "Grade 1",
    "g1": "Grade 1",
    "type 1": "Grade 1",
    "grade 2a": "Grade 2A",
    "sc grade 2a": "Grade 2A",
    "g2a": "Grade 2A",
    "type 2a": "Grade 2A",
    "grade 2b": "Grade 2B",
    "sc grade 2b": "Grade 2B",
    "g2b": "Grade 2B",
    "type 2b": "Grade 2B",
    "grade 3": "Grade 3",
    "sc grade 3": "Grade 3",
    "g3": "Grade 3",
    "type 3": "Grade 3",
}

# Flexion-type fractures are excluded from the Gartland extension-type scheme
# used in this study (7 KKH radiographs, per Table 1).
_EXCLUDED = {"flexion", "flexion type", "sc flexion", "sc flexion type"}


class UnknownGradeError(ValueError):
    """Raised when a manifest label cannot be mapped onto GRADES."""


def normalise_grade(raw: str) -> str | None:
    """Map a raw manifest/folder label onto the canonical vocabulary.

    Returns None for labels that are deliberately excluded from the 5-class
    task (flexion type). Raises UnknownGradeError for anything unrecognised
    rather than silently dropping it, so a manifest typo fails loudly.
    """
    key = re.sub(r"\s+", " ", str(raw).strip().casefold())
    if not key:
        raise UnknownGradeError("empty grade label")
    if key in _EXCLUDED:
        return None
    if key in _ALIASES:
        return _ALIASES[key]
    raise UnknownGradeError(
        f"unrecognised grade label {raw!r}; add an alias in labels.py if this is "
        f"a legitimate variant of {GRADES}"
    )


def grade_to_index(raw: str) -> int | None:
    """Canonicalise then convert to the integer label, or None if excluded."""
    grade = normalise_grade(raw)
    return None if grade is None else GRADE_TO_IDX[grade]


def is_adjacent(true_idx: int, pred_idx: int) -> bool:
    """True when a confusion is between neighbouring severity grades."""
    return abs(true_idx - pred_idx) == 1


def crosses_operative_boundary(true_idx: int, pred_idx: int) -> bool:
    """True when a confusion crosses the Grade 2A / Grade 2B boundary.

    Only the 2A<->2B pair counts. A 2A->Grade 3 error also changes management
    but is a distant error, counted separately in the figure.
    """
    return {true_idx, pred_idx} == set(OPERATIVE_BOUNDARY)
