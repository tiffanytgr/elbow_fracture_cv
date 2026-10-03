"""Evaluate the end-to-end 5-class baseline on a held-out cohort.

Writes everything the reviewer's Figure 2 needs:

  predictions.csv  one row per patient: true grade, predicted grade, the five
                   class probabilities, max-softmax confidence, correctness
  metrics.json     5x5 confusion matrix, per-grade recall/precision with Wilson
                   CIs, overall and macro accuracy, quadratic-weighted kappa
                   with a bootstrap CI, adjacent-vs-distant error split,
                   operative-boundary error count, and the operative /
                   non-operative collapse

``--coverage-from`` applies the same DRUE-retained patient list the cascade's
filtered row uses, so "filtered" means the same 178 patients for both models
and the comparison is paired. Without it, metrics are unfiltered over every
case in the split.

Example
-------
    python -m research.multiclass_baseline.evaluate \
        --manifest research/outputs/paired_manifest.csv \
        --checkpoint research/outputs/multiclass_dual_seed0/best_model.pth \
        --split test \
        --coverage-from research/outputs/drue_retained_cases.txt \
        --out-dir research/outputs/multiclass_dual_seed0/eval
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

from .dataset import build_loader, read_manifest
from .labels import GRADES, NUM_CLASSES
from .metrics import (
    auroc,
    classification_report,
    expected_calibration_error,
    risk_coverage_curve,
    selective_risk_auc,
)
from .model import build_model

log = logging.getLogger(__name__)

PREDICTION_COLUMNS = (
    "case_id",
    "true_idx",
    "true_grade",
    "pred_idx",
    "pred_grade",
    "confidence",
    "correct",
    *(f"prob_{g.replace(' ', '_').lower()}" for g in GRADES),
)


def load_model(checkpoint: Path, device: torch.device, architecture: str | None = None):
    """Rebuild the baseline from a training checkpoint and load its weights."""
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if "model_state_dict" not in payload:
        raise ValueError(
            f"{checkpoint} is not a baseline training checkpoint "
            f"(expected a 'model_state_dict' key, found {sorted(payload)[:6]})"
        )
    arch = architecture or payload.get("architecture", "dual")
    stored_classes = payload.get("classes")
    if stored_classes and list(stored_classes) != list(GRADES):
        raise ValueError(
            f"checkpoint class order {stored_classes} does not match labels.GRADES "
            f"{list(GRADES)}; metrics would be silently wrong"
        )
    model = build_model(
        architecture=arch,
        num_classes=NUM_CLASSES,
        encoder_checkpoint=None,
        dropout=float(payload.get("dropout", 0.3)),
        imagenet_fallback=False,
    )
    model.load_state_dict(payload["model_state_dict"])
    model.to(device).eval()
    log.info("loaded %s checkpoint from epoch %s", arch, payload.get("epoch", "?"))
    return model, payload


@torch.inference_mode()
def predict(model, loader, device: torch.device) -> list[dict]:
    rows: list[dict] = []
    for ap, lat, target, case_ids in loader:
        probs = torch.softmax(model(ap.to(device), lat.to(device)), dim=1).cpu().numpy()
        for case_id, true_idx, prob in zip(case_ids, target.numpy(), probs):
            pred_idx = int(prob.argmax())
            row = {
                "case_id": case_id,
                "true_idx": int(true_idx),
                "true_grade": GRADES[int(true_idx)],
                "pred_idx": pred_idx,
                "pred_grade": GRADES[pred_idx],
                "confidence": float(prob.max()),
                "correct": int(pred_idx == int(true_idx)),
            }
            for grade, value in zip(GRADES, prob):
                row[f"prob_{grade.replace(' ', '_').lower()}"] = float(value)
            rows.append(row)
    return rows


def read_case_list(path: Path) -> set[str]:
    """Read a retained-case list: one case ID per line, or a CSV with case_id."""
    text = path.read_text(encoding="utf-8-sig").strip()
    if not text:
        raise ValueError(f"{path} is empty")
    first = text.splitlines()[0]
    if "," in first or first.strip().casefold() == "case_id":
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            cols = {c.strip().casefold(): c for c in (reader.fieldnames or [])}
            if "case_id" not in cols:
                raise ValueError(f"{path} has no case_id column (found {reader.fieldnames})")
            return {row[cols["case_id"]].strip().upper() for row in reader if row[cols["case_id"]].strip()}
    return {line.strip().upper() for line in text.splitlines() if line.strip()}


def write_predictions(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(PREDICTION_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def summarise(rows: list[dict], n_total: int | None, n_boot: int, seed: int) -> dict:
    """Build the metrics payload for one cohort configuration."""
    y_true = np.array([r["true_idx"] for r in rows])
    y_pred = np.array([r["pred_idx"] for r in rows])
    confidence = np.array([r["confidence"] for r in rows])
    correct = y_true == y_pred

    report = classification_report(
        y_true, y_pred, n_total=n_total, n_boot=n_boot, seed=seed
    )
    payload = report.to_dict()
    payload["classes"] = list(GRADES)
    payload["selective_prediction"] = {
        "risk_coverage_auc": selective_risk_auc(correct, confidence),
        "confidence_error_auroc": auroc(-confidence, ~correct),
        "mean_confidence_correct": float(confidence[correct].mean()) if correct.any() else None,
        "mean_confidence_incorrect": (
            float(confidence[~correct].mean()) if (~correct).any() else None
        ),
    }
    calibration = expected_calibration_error(correct, confidence)
    payload["calibration"] = {
        "ece": calibration["ece"],
        "n_bins": calibration["n_bins"],
        "bin_centre": calibration["bin_centre"].tolist(),
        "bin_accuracy": [None if np.isnan(v) else v for v in calibration["bin_accuracy"]],
        "bin_confidence": [None if np.isnan(v) else v for v in calibration["bin_confidence"]],
        "bin_count": calibration["bin_count"].tolist(),
    }
    curve = risk_coverage_curve(correct, confidence)
    payload["risk_coverage_curve"] = {
        "coverage": curve["coverage"].tolist(),
        "risk": curve["risk"].tolist(),
        "threshold": curve["threshold"].tolist(),
    }
    return payload


def log_report(title: str, payload: dict) -> None:
    log.info("--- %s ---", title)
    coverage = payload.get("coverage")
    log.info(
        "n=%d  overall acc=%.1f%% [%.1f, %.1f]  macro acc=%.1f%%  QWK=%.3f [%.2f, %.2f]%s",
        payload["n"], payload["overall_accuracy"], *payload["overall_accuracy_ci"],
        payload["macro_recall"], payload["quadratic_kappa"], *payload["quadratic_kappa_ci"],
        f"  coverage={coverage:.0f}%" if coverage else "",
    )
    log.info("%-10s %6s %22s %22s", "grade", "n", "recall % [95% CI]", "precision % [95% CI]")
    for cls in payload["per_class"]:
        log.info(
            "%-10s %6d   %5.1f [%4.1f, %5.1f]   %5.1f [%4.1f, %5.1f]",
            cls["grade"], cls["support"], cls["recall"], *cls["recall_ci"],
            cls["precision"], *cls["precision_ci"],
        )
    log.info(
        "errors: %.0f%% adjacent, %.0f%% distant | 2A<->2B boundary errors: %d",
        payload["adjacent_error_rate"], payload["distant_error_rate"],
        payload["operative_boundary_errors"],
    )
    log.info(
        "operative collapse: acc=%.1f%% sens=%.1f%% spec=%.1f%%",
        payload["operative_collapse_accuracy"],
        payload["operative_collapse_sensitivity"],
        payload["operative_collapse_specificity"],
    )
    log.info("confusion matrix (rows=reference, cols=prediction)")
    header = " " * 10 + "".join(f"{g:>8}" for g in payload["classes"])
    log.info(header)
    for grade, row in zip(payload["classes"], payload["confusion_matrix"]):
        log.info(f"{grade:<10}" + "".join(f"{v:>8}" for v in row))


def evaluate(args: argparse.Namespace) -> dict:
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available()
                          else "cpu" if args.device == "auto" else args.device)
    cases = read_manifest(args.manifest, split=args.split)
    log.info("evaluating %d paired cases from split %r", len(cases), args.split)

    model, payload = load_model(args.checkpoint, device, args.architecture)
    loader = build_loader(
        cases, batch_size=args.batch_size, train=False,
        input_size=int(payload.get("input_size", args.input_size)),
        num_workers=args.num_workers, image_root=args.image_root,
    )
    rows = predict(model, loader, device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_predictions(rows, args.out_dir / "predictions.csv")

    results = {
        "checkpoint": str(args.checkpoint),
        "manifest": str(args.manifest),
        "split": args.split,
        "architecture": payload.get("architecture", args.architecture or "dual"),
        "unfiltered": summarise(rows, n_total=None, n_boot=args.n_boot, seed=args.seed),
    }
    log_report(f"unfiltered ({len(rows)} cases)", results["unfiltered"])

    if args.coverage_from:
        retained = read_case_list(args.coverage_from)
        kept = [r for r in rows if r["case_id"].upper() in retained]
        missing = retained - {r["case_id"].upper() for r in rows}
        if not kept:
            raise ValueError(
                f"{args.coverage_from} retained no case present in split {args.split!r}; "
                f"check that the case IDs match the manifest"
            )
        if missing:
            log.warning(
                "%d retained case ID(s) are not in this split and were ignored: %s",
                len(missing), ", ".join(sorted(missing)[:10]),
            )
        results["filtered"] = summarise(
            kept, n_total=len(rows), n_boot=args.n_boot, seed=args.seed
        )
        results["filtered"]["retained_case_list"] = str(args.coverage_from)
        log_report(f"DRUE filtered ({len(kept)}/{len(rows)} cases)", results["filtered"])

    (args.out_dir / "metrics.json").write_text(json.dumps(results, indent=2))
    log.info("wrote %s and %s", args.out_dir / "predictions.csv", args.out_dir / "metrics.json")
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the end-to-end 5-class baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--image-root", type=Path)
    parser.add_argument("--architecture", choices=("dual", "two-channel"),
                        help="Override the architecture recorded in the checkpoint.")
    parser.add_argument("--coverage-from", type=Path,
                        help="Retained case IDs (one per line, or a CSV with case_id) "
                             "defining the DRUE-filtered subset.")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(message)s",
    )
    evaluate(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
