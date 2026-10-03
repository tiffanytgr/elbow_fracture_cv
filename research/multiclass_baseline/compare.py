"""Paired comparison of the hierarchical cascade against the multiclass baseline.

Both models must have been evaluated on the *same* patients, so the comparison
is restricted to the intersection of the two prediction files by ``case_id``
and every test is paired. Reported:

  * exact-grade accuracy, macro recall and quadratic-weighted kappa for each
    model, with the kappa difference and a paired bootstrap CI
  * exact McNemar on per-patient exact-grade correctness
  * per-grade recall for each model, with the difference
  * operative-boundary (2A<->2B) error counts, and the operative /
    non-operative collapse
  * adjacent vs distant error split

Input is two CSVs with at least ``case_id,true_idx,pred_idx``; the files written
by ``evaluate.py`` satisfy this, and a cascade prediction file only needs those
three columns. If the cascade file carries ``true_grade``/``pred_grade`` text
instead of indices, those are canonicalised through labels.py.

Example
-------
    python -m research.multiclass_baseline.compare \
        --cascade research/outputs/cascade_predictions.csv \
        --multiclass research/outputs/multiclass_dual_seed0/eval/predictions.csv \
        --out research/outputs/cascade_vs_multiclass.json
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

from .labels import GRADES, NUM_CLASSES, grade_to_index
from .metrics import (
    classification_report,
    mcnemar_exact,
    paired_kappa_difference,
    wilson_interval,
)

log = logging.getLogger(__name__)


def read_predictions(path: Path, label: str) -> dict[str, tuple[int, int]]:
    """Read ``case_id -> (true_idx, pred_idx)`` from a prediction CSV.

    Accepts integer index columns or grade-name columns. An abstention
    (``pred_idx`` of -1 or an empty prediction) is dropped with a count in the
    log: a withheld case has no grade to compare, and silently scoring it as
    wrong would misrepresent selective prediction.
    """
    out: dict[str, tuple[int, int]] = {}
    abstained = 0
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        cols = {c.strip().casefold(): c for c in (reader.fieldnames or [])}
        if "case_id" not in cols:
            raise ValueError(f"{path} has no case_id column (found {reader.fieldnames})")

        def resolve(row: dict, idx_key: str, name_key: str, line_no: int) -> int | None:
            if idx_key in cols:
                raw = (row[cols[idx_key]] or "").strip()
                if raw == "" or raw == "-1":
                    return None
                try:
                    value = int(float(raw))
                except ValueError as exc:
                    raise ValueError(f"{path}:{line_no}: bad {idx_key} {raw!r}") from exc
                if not 0 <= value < NUM_CLASSES:
                    raise ValueError(
                        f"{path}:{line_no}: {idx_key}={value} outside 0..{NUM_CLASSES - 1}"
                    )
                return value
            if name_key in cols:
                raw = (row[cols[name_key]] or "").strip()
                if not raw:
                    return None
                return grade_to_index(raw)
            raise ValueError(
                f"{path} needs {idx_key} or {name_key} (found {reader.fieldnames})"
            )

        for line_no, row in enumerate(reader, start=2):
            case_id = (row[cols["case_id"]] or "").strip().upper()
            if not case_id:
                continue
            true_idx = resolve(row, "true_idx", "true_grade", line_no)
            pred_idx = resolve(row, "pred_idx", "pred_grade", line_no)
            if true_idx is None:
                raise ValueError(f"{path}:{line_no}: missing reference grade for {case_id}")
            if pred_idx is None:
                abstained += 1
                continue
            if case_id in out:
                raise ValueError(f"{path}: duplicate case_id {case_id}")
            out[case_id] = (true_idx, pred_idx)
    log.info("%s: %d scored cases%s", label, len(out),
             f" ({abstained} abstentions dropped)" if abstained else "")
    if not out:
        raise ValueError(f"{path} contained no scorable predictions")
    return out


def align(
    cascade: dict[str, tuple[int, int]], multiclass: dict[str, tuple[int, int]]
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray]:
    """Intersect on case_id and check the two files agree on the reference standard."""
    shared = sorted(set(cascade) & set(multiclass))
    if not shared:
        raise ValueError(
            "the two prediction files share no case_id; check that both use the "
            "same case identifiers"
        )
    only_cascade = sorted(set(cascade) - set(multiclass))
    only_multi = sorted(set(multiclass) - set(cascade))
    for name, cases in (("cascade only", only_cascade), ("multiclass only", only_multi)):
        if cases:
            log.warning("%d case(s) %s, excluded from the paired comparison: %s",
                        len(cases), name, ", ".join(cases[:10]))

    mismatched = [c for c in shared if cascade[c][0] != multiclass[c][0]]
    if mismatched:
        raise ValueError(
            f"{len(mismatched)} case(s) have different reference grades in the two "
            f"files (e.g. {mismatched[:5]}); the comparison would be meaningless"
        )

    y_true = np.array([cascade[c][0] for c in shared])
    pred_cascade = np.array([cascade[c][1] for c in shared])
    pred_multi = np.array([multiclass[c][1] for c in shared])
    log.info("paired comparison on %d shared cases", len(shared))
    return shared, y_true, pred_cascade, pred_multi


def compare(
    y_true: np.ndarray,
    pred_cascade: np.ndarray,
    pred_multiclass: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    cascade_report = classification_report(y_true, pred_cascade, n_boot=n_boot, seed=seed)
    multi_report = classification_report(y_true, pred_multiclass, n_boot=n_boot, seed=seed)

    correct_cascade = y_true == pred_cascade
    correct_multi = y_true == pred_multiclass

    per_grade = []
    for idx, grade in enumerate(GRADES):
        mask = y_true == idx
        support = int(mask.sum())
        if support == 0:
            continue
        c_hits = int(correct_cascade[mask].sum())
        m_hits = int(correct_multi[mask].sum())
        per_grade.append(
            {
                "grade": grade,
                "support": support,
                "cascade_recall": c_hits / support * 100,
                "cascade_recall_ci": wilson_interval(c_hits, support),
                "multiclass_recall": m_hits / support * 100,
                "multiclass_recall_ci": wilson_interval(m_hits, support),
                "difference": (c_hits - m_hits) / support * 100,
            }
        )

    accuracy_delta = (
        cascade_report.overall_accuracy - multi_report.overall_accuracy
    )
    return {
        "n": int(y_true.size),
        "classes": list(GRADES),
        "cascade": cascade_report.to_dict(),
        "multiclass": multi_report.to_dict(),
        "per_grade_recall": per_grade,
        "paired_tests": {
            "mcnemar_exact_grade": mcnemar_exact(correct_cascade, correct_multi),
            "kappa_difference_cascade_minus_multiclass": paired_kappa_difference(
                y_true, pred_cascade, pred_multiclass, n_boot=n_boot, seed=seed
            ),
            "accuracy_difference_cascade_minus_multiclass": accuracy_delta,
        },
        "operative_boundary": {
            "cascade_errors": cascade_report.operative_boundary_errors,
            "multiclass_errors": multi_report.operative_boundary_errors,
        },
    }


def log_comparison(result: dict) -> None:
    cascade, multi = result["cascade"], result["multiclass"]
    log.info("--- cascade vs end-to-end multiclass (n=%d) ---", result["n"])
    log.info("%-26s %12s %12s", "metric", "cascade", "multiclass")
    rows = [
        ("exact-grade accuracy, %", cascade["overall_accuracy"], multi["overall_accuracy"]),
        ("macro recall, %", cascade["macro_recall"], multi["macro_recall"]),
        ("weighted F1", cascade["weighted_f1"], multi["weighted_f1"]),
        ("quadratic-weighted kappa", cascade["quadratic_kappa"], multi["quadratic_kappa"]),
        ("operative collapse acc, %", cascade["operative_collapse_accuracy"],
         multi["operative_collapse_accuracy"]),
        ("adjacent errors, %", cascade["adjacent_error_rate"], multi["adjacent_error_rate"]),
    ]
    for name, a, b in rows:
        log.info("%-26s %12.3f %12.3f", name, a, b)

    tests = result["paired_tests"]
    mc = tests["mcnemar_exact_grade"]
    kappa = tests["kappa_difference_cascade_minus_multiclass"]
    log.info(
        "McNemar: cascade-only correct %d, multiclass-only correct %d, P=%.3f",
        mc["a_only_correct"], mc["b_only_correct"], mc["p_value"],
    )
    log.info(
        "kappa difference (cascade - multiclass): %+.3f (95%% CI %+.3f, %+.3f)",
        kappa["difference"], *kappa["ci"],
    )
    log.info("per-grade recall")
    for row in result["per_grade_recall"]:
        log.info(
            "  %-9s n=%-4d cascade %5.1f%%  multiclass %5.1f%%  diff %+5.1f pp",
            row["grade"], row["support"], row["cascade_recall"],
            row["multiclass_recall"], row["difference"],
        )
    log.info(
        "2A<->2B boundary errors: cascade %d, multiclass %d",
        result["operative_boundary"]["cascade_errors"],
        result["operative_boundary"]["multiclass_errors"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paired cascade vs multiclass comparison.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--cascade", type=Path, required=True,
                        help="Cascade predictions CSV (case_id,true_idx,pred_idx).")
    parser.add_argument("--multiclass", type=Path, required=True,
                        help="Baseline predictions CSV from evaluate.py.")
    parser.add_argument("--out", type=Path, required=True, help="Output JSON.")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    cascade = read_predictions(args.cascade, "cascade")
    multiclass = read_predictions(args.multiclass, "multiclass")
    case_ids, y_true, pred_cascade, pred_multi = align(cascade, multiclass)
    result = compare(y_true, pred_cascade, pred_multi, n_boot=args.n_boot, seed=args.seed)
    result["case_ids"] = case_ids
    result["inputs"] = {"cascade": str(args.cascade), "multiclass": str(args.multiclass)}
    log_comparison(result)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    log.info("wrote %s", args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
