"""Input contracts for the panel figures, plus a synthetic demo mode.

Every figure script reads from a results directory and never recomputes a
statistic that ``multiclass_baseline.evaluate`` / ``.compare`` already wrote.
That keeps the number in a figure identical to the number in the text.

Expected layout of ``--results`` (each file optional; a figure skips the panels
it has no data for and says so):

    cascade_predictions.csv      case_id,true_idx,pred_idx[,confidence]
    multiclass_predictions.csv   written by evaluate.py
    cascade_metrics.json         written by evaluate.py for the cascade
    multiclass_metrics.json      written by evaluate.py for the baseline
    comparison.json              written by compare.py
    uncertainty.csv              case_id,drue_score[,correct,confidence]
    ablation.csv                 condition,node,recall,weighted_f1[,coverage]
    cortical_width.csv           case_id,grade,match_ratio
    cortical_profiles.csv        case_id,grade,position,normalised_width
    baumann.csv                  case_id,grade,baumann_angle
    ahl.csv                      case_id,grade,bisects_capitellum
    images/                      representative PNGs for the photo panels

``--demo`` replaces all of it with data synthesised from the published summary
statistics, so layout can be reviewed before the real artefacts exist. Demo
renders carry a diagonal "SYNTHETIC DEMO DATA" stamp and a ``_DEMO`` filename
suffix; they are for checking composition only.
"""
from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from research.multiclass_baseline.labels import GRADES, NUM_CLASSES

log = logging.getLogger(__name__)


# --- published summary statistics (report Tables 1 and 3) ---------------------
# Used only to seed --demo so the synthetic layout has realistic magnitudes.
HELD_OUT_GRADE_COUNTS = {"Normal": 75, "Grade 1": 69, "Grade 2A": 53, "Grade 2B": 8, "Grade 3": 27}
HELD_OUT_N = sum(HELD_OUT_GRADE_COUNTS.values())  # 232
HELD_OUT_FILTERED_N = 178
PUBLISHED = {
    "unfiltered": {"overall_accuracy": 65.9, "macro_recall": 56.8, "kappa": 0.755},
    "filtered": {"overall_accuracy": 66.9, "macro_recall": 61.5, "kappa": 0.796},
}


class MissingInput(Exception):
    """Raised when a figure cannot be drawn because its input file is absent."""


@dataclass
class Results:
    """Lazily-read bundle of result artefacts for one results directory."""

    root: Path
    demo: bool = False
    _cache: dict = field(default_factory=dict, repr=False)

    # --- generic readers ----------------------------------------------------

    def path(self, name: str) -> Path:
        return self.root / name

    def has(self, name: str) -> bool:
        return self.demo or self.path(name).exists()

    def json(self, name: str) -> dict:
        if name not in self._cache:
            path = self.path(name)
            if not path.exists():
                raise MissingInput(f"{path} not found")
            self._cache[name] = json.loads(path.read_text(encoding="utf-8-sig"))
        return self._cache[name]

    def table(self, name: str) -> list[dict[str, str]]:
        """Read a CSV as a list of dicts, with keys casefolded and stripped."""
        if name not in self._cache:
            path = self.path(name)
            if not path.exists():
                raise MissingInput(f"{path} not found")
            with path.open(newline="", encoding="utf-8-sig") as handle:
                reader = csv.DictReader(handle)
                rows = [
                    {(k or "").strip().casefold(): (v or "").strip() for k, v in row.items()}
                    for row in reader
                ]
            if not rows:
                raise MissingInput(f"{path} is empty")
            self._cache[name] = rows
        return self._cache[name]

    def column(self, name: str, key: str, cast=float) -> np.ndarray:
        rows = self.table(name)
        if key not in rows[0]:
            raise MissingInput(f"{self.path(name)} has no {key!r} column "
                               f"(found {sorted(rows[0])})")
        return np.array([cast(r[key]) for r in rows if r[key] != ""])

    # --- predictions --------------------------------------------------------

    def predictions(self, which: str) -> dict[str, np.ndarray]:
        """``which`` is "cascade" or "multiclass"."""
        key = f"_pred_{which}"
        if key in self._cache:
            return self._cache[key]
        if self.demo:
            value = _demo_predictions(which)
        else:
            rows = self.table(f"{which}_predictions.csv")
            value = {
                "case_id": np.array([r["case_id"] for r in rows]),
                "y_true": np.array([int(float(r["true_idx"])) for r in rows]),
                "y_pred": np.array([int(float(r["pred_idx"])) for r in rows]),
            }
            if "confidence" in rows[0] and rows[0]["confidence"]:
                value["confidence"] = np.array([float(r["confidence"]) for r in rows])
        self._cache[key] = value
        return value

    def confusion(self, which: str) -> np.ndarray:
        pred = self.predictions(which)
        cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)
        for t, p in zip(pred["y_true"], pred["y_pred"]):
            cm[t, p] += 1
        return cm

    # --- demo-aware accessors ----------------------------------------------

    def comparison(self) -> dict:
        if self.demo:
            return _demo_comparison()
        return self.json("comparison.json")

    def uncertainty(self) -> dict[str, np.ndarray]:
        if self.demo:
            return _demo_uncertainty()
        rows = self.table("uncertainty.csv")
        out = {
            "case_id": np.array([r["case_id"] for r in rows]),
            "drue_score": np.array([float(r["drue_score"]) for r in rows]),
        }
        pred = self.predictions("cascade")
        lookup = dict(zip(pred["case_id"], pred["y_true"] == pred["y_pred"]))
        if "correct" in rows[0] and rows[0]["correct"] != "":
            out["correct"] = np.array([bool(int(float(r["correct"]))) for r in rows])
        else:
            missing = [r["case_id"] for r in rows if r["case_id"] not in lookup]
            if missing:
                raise MissingInput(
                    f"uncertainty.csv has no 'correct' column and {len(missing)} case(s) "
                    f"are absent from cascade_predictions.csv (e.g. {missing[:5]})"
                )
            out["correct"] = np.array([lookup[r["case_id"]] for r in rows])
        if "confidence" in rows[0] and rows[0]["confidence"] != "":
            out["confidence"] = np.array([float(r["confidence"]) for r in rows])
        return out

    def ablation(self) -> list[dict]:
        if self.demo:
            return _demo_ablation()
        rows = self.table("ablation.csv")
        required = {"condition", "node", "recall"}
        missing = required - set(rows[0])
        if missing:
            raise MissingInput(
                f"{self.path('ablation.csv')} needs columns {sorted(required)}; "
                f"missing {sorted(missing)}"
            )
        return [
            {
                "condition": r["condition"],
                "node": r["node"],
                "recall": float(r["recall"]),
                "weighted_f1": float(r["weighted_f1"]) if r.get("weighted_f1") else None,
                "coverage": float(r["coverage"]) if r.get("coverage") else None,
            }
            for r in rows
        ]

    def measurements(self, name: str, value_key: str) -> dict[str, np.ndarray]:
        """Per-case geometric measurement grouped by grade (Figure 4)."""
        if self.demo:
            return _demo_measurement(value_key)
        rows = self.table(name)
        for key in ("grade", value_key):
            if key not in rows[0]:
                raise MissingInput(
                    f"{self.path(name)} has no {key!r} column (found {sorted(rows[0])})"
                )
        return {
            "grade": np.array([r["grade"] for r in rows]),
            "value": np.array([float(r[value_key]) for r in rows if r[value_key] != ""]),
            "case_id": np.array([r.get("case_id", "") for r in rows]),
        }

    def ahl_bisection(self) -> dict[str, tuple[int, int]]:
        """``grade -> (cases where the AHL bisects the capitellum, total)``."""
        if self.demo:
            return _demo_ahl()
        rows = self.table("ahl.csv")
        for key in ("grade", "bisects_capitellum"):
            if key not in rows[0]:
                raise MissingInput(
                    f"{self.path('ahl.csv')} has no {key!r} column (found {sorted(rows[0])})"
                )
        counts: dict[str, list[int]] = {}
        for row in rows:
            flag = row["bisects_capitellum"].strip().casefold()
            if flag in {"", "na", "nan", "none"}:
                continue  # not computable on this radiograph
            if flag not in {"0", "1", "true", "false", "yes", "no"}:
                raise MissingInput(
                    f"{self.path('ahl.csv')}: bisects_capitellum must be boolean, got {flag!r}"
                )
            bucket = counts.setdefault(row["grade"], [0, 0])
            bucket[1] += 1
            if flag in {"1", "true", "yes"}:
                bucket[0] += 1
        if not counts:
            raise MissingInput(f"{self.path('ahl.csv')} has no evaluable rows")
        return {g: (hits, total) for g, (hits, total) in counts.items()}

    def cortical_profiles(self) -> dict[str, np.ndarray]:
        """``position`` plus one (n_cases x 40) array of profiles per grade."""
        if self.demo:
            return demo_cortical_profiles()
        rows = self.table("cortical_profiles.csv")
        for key in ("case_id", "grade", "position", "normalised_width"):
            if key not in rows[0]:
                raise MissingInput(
                    f"{self.path('cortical_profiles.csv')} has no {key!r} column "
                    f"(found {sorted(rows[0])})"
                )
        by_case: dict[tuple[str, str], dict[float, float]] = {}
        for row in rows:
            key = (row["grade"], row["case_id"])
            by_case.setdefault(key, {})[float(row["position"])] = float(row["normalised_width"])
        positions = sorted({p for case in by_case.values() for p in case})
        out: dict[str, np.ndarray] = {"position": np.array(positions)}
        for grade in sorted({g for g, _ in by_case}):
            curves = [
                [case.get(p, np.nan) for p in positions]
                for (g, _), case in by_case.items() if g == grade
            ]
            out[grade] = np.array(curves, dtype=float)
        return out

    def image(self, name: str) -> Path | None:
        """Representative radiograph for a photo panel, or None if absent."""
        if self.demo:
            return None
        path = self.root / "images" / name
        return path if path.exists() else None


# --- demo generators ----------------------------------------------------------
# Each is seeded and deterministic, and reproduces the published summary
# statistics to within a point or so, so the demo layout is realistically
# shaped without implying any new result.


def _demo_rng(tag: str) -> np.random.Generator:
    return np.random.default_rng(abs(hash(tag)) % (2**32))


# Per-grade recall targets for the demo. The cascade row is chosen to land on
# the published unfiltered figures (65.9% overall, 56.8% macro, Table 3); the
# baseline row is set a little lower purely so the panels have something to
# compare, and is not a prediction of how the baseline will actually do.
_DEMO_RECALL = {
    "cascade": {"Normal": 0.83, "Grade 1": 0.68, "Grade 2A": 0.54,
                "Grade 2B": 0.25, "Grade 3": 0.46},
    "multiclass": {"Normal": 0.80, "Grade 1": 0.64, "Grade 2A": 0.50,
                   "Grade 2B": 0.13, "Grade 3": 0.44},
}


def _fixed_labels() -> np.ndarray:
    """The held-out cohort's reference grades, in a fixed shuffled order.

    Both demo models see the identical label vector, so the demo comparison is
    paired in the same way the real one is.
    """
    y_true = np.concatenate(
        [np.full(count, GRADES.index(grade)) for grade, count in HELD_OUT_GRADE_COUNTS.items()]
    )
    _demo_rng("labels").shuffle(y_true)
    return y_true


def _sample_confusion(rng, recall_by_grade: dict[str, float], adjacent_bias: float = 0.78) -> dict:
    """Draw predictions hitting the given per-grade recall, with ordinal errors.

    Errors land on a neighbouring grade with probability ``adjacent_bias``,
    which is what makes quadratic-weighted kappa meaningfully higher than
    exact-grade accuracy -- the pattern the real cascade shows.
    """
    y_true = _fixed_labels()
    y_pred = y_true.copy()
    for i, t in enumerate(y_true):
        target = recall_by_grade[GRADES[int(t)]]
        if rng.random() < target:
            continue
        neighbours = [c for c in (t - 1, t + 1) if 0 <= c < NUM_CLASSES]
        distant = [c for c in range(NUM_CLASSES) if abs(c - t) > 1]
        pool = neighbours if (rng.random() < adjacent_bias or not distant) else distant
        y_pred[i] = rng.choice(pool)
    confidence = np.clip(
        rng.beta(6, 2, size=y_true.size) + np.where(y_true == y_pred, 0.08, -0.12),
        0.21, 0.999,
    )
    return {
        "case_id": np.array([f"DEMO{i:04d}" for i in range(y_true.size)]),
        "y_true": y_true,
        "y_pred": y_pred,
        "confidence": confidence,
    }


def _demo_predictions(which: str) -> dict[str, np.ndarray]:
    if which not in _DEMO_RECALL:
        raise ValueError(f"unknown prediction set {which!r}")
    return _sample_confusion(_demo_rng(which), _DEMO_RECALL[which])


def _demo_comparison() -> dict:
    from research.multiclass_baseline.compare import compare

    cascade = _demo_predictions("cascade")
    multi = _demo_predictions("multiclass")
    result = compare(cascade["y_true"], cascade["y_pred"], multi["y_pred"], n_boot=400)
    result["demo"] = True
    return result


def _demo_uncertainty() -> dict[str, np.ndarray]:
    pred = _demo_predictions("cascade")
    rng = _demo_rng("drue")
    correct = pred["y_true"] == pred["y_pred"]
    # Errors sit at higher reconstruction disagreement, with heavy overlap.
    score = rng.gamma(shape=4.0, scale=0.0032, size=correct.size)
    score = score + np.where(correct, 0.0, 0.0042) + rng.normal(0, 0.0015, correct.size)
    return {
        "case_id": pred["case_id"],
        "drue_score": np.clip(score, 1e-4, None),
        "correct": correct,
        "confidence": pred["confidence"],
    }


def _demo_ablation() -> list[dict]:
    rng = _demo_rng("ablation")
    conditions = ["Raw", "+ Standardisation", "+ QC filtering", "+ Std. + QC"]
    anchors = {
        "Node 1": [88.0, 91.0, 90.2, 92.4],
        "Node 2": [48.5, 51.9, 54.0, 57.9],
        "Node 3": [45.8, 81.4, 52.1, 81.0],
        "Node 4": [80.0, 42.9, 66.7, 50.0],
    }
    rows = []
    for node, values in anchors.items():
        for condition, recall in zip(conditions, values):
            rows.append(
                {
                    "condition": condition,
                    "node": node,
                    "recall": float(recall),
                    "weighted_f1": float(np.clip(recall / 100 * 0.95 + rng.normal(0, 0.02), 0, 1)),
                    "coverage": 100.0 if "QC" not in condition else float(rng.uniform(78, 92)),
                }
            )
    return rows


def _demo_measurement(value_key: str) -> dict[str, np.ndarray]:
    rng = _demo_rng(f"measure-{value_key}")
    if value_key == "match_ratio":
        # Report section 4.3: 0.59 +/- 0.29 (2A) vs 0.47 +/- 0.34 (2B).
        spec = {"Grade 2A": (0.59, 0.29, 53), "Grade 2B": (0.47, 0.34, 8)}
    elif value_key == "baumann_angle":
        # Report section 3: median 74 (Grade 3) vs 66 (non-Grade 3).
        spec = {"Grade 1": (66, 8, 69), "Grade 2A": (66, 8, 53),
                "Grade 2B": (67, 9, 8), "Grade 3": (74, 9, 27)}
    else:
        spec = {g: (0.5, 0.2, 20) for g in GRADES[1:]}
    grades, values = [], []
    for grade, (mean, sd, n) in spec.items():
        grades += [grade] * n
        drawn = rng.normal(mean, sd, n)
        if value_key == "match_ratio":
            drawn = np.clip(drawn, 0.0, 1.0)  # a ratio of widths cannot be negative
        values += list(drawn)
    return {
        "grade": np.array(grades),
        "value": np.array(values),
        "case_id": np.array([f"DEMO{i:04d}" for i in range(len(grades))]),
    }


def _demo_ahl() -> dict[str, tuple[int, int]]:
    """AHL bisection rates: high for Grade 1, falling as displacement increases."""
    return {
        "Normal": (71, 75),
        "Grade 1": (58, 69),
        "Grade 2A": (14, 53),
        "Grade 2B": (1, 8),
        "Grade 3": (2, 27),
    }


def demo_cortical_profiles() -> dict[str, np.ndarray]:
    """40-point normalised width profiles for the Figure 4 profile panel."""
    rng = _demo_rng("profiles")
    position = np.linspace(0, 1, 40)
    out: dict[str, np.ndarray] = {"position": position}
    for grade, slope, noise in (("Grade 2A", -0.22, 0.06), ("Grade 2B", -0.40, 0.09)):
        n = 40 if grade == "Grade 2A" else 12
        curves = (
            1.08
            + slope * position[None, :]
            + rng.normal(0, noise, size=(n, position.size)).cumsum(axis=1) * 0.04
        )
        out[grade] = curves
    return out
