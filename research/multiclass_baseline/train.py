"""Train the end-to-end 5-class baseline with the cascade's own training recipe.

Two-stage transfer learning, matching report section 2.7.1 so that the only
difference between the baseline and the cascade is the decision structure:

  Stage 1  encoders frozen (batch-norm in eval mode), head only, Adam,
           constant ``--head-lr``, ``--stage1-epochs`` epochs.
  Stage 2  everything unfrozen, discriminative learning rates with the
           backbone at ``--backbone-lr-scale`` x the head rate, cosine
           annealing over ``--stage2-epochs`` epochs.

Checkpoint selection is on validation accuracy, as in the cascade. Note the
limitation the report already records in section 4.6: validation *accuracy*
selects the checkpoint while recall is the primary clinical metric. Pass
``--select-on macro_recall`` to select on balanced recall instead; the default
reproduces the cascade's behaviour so the comparison is like-for-like.

Example
-------
    python -m research.multiclass_baseline.train \
        --manifest research/outputs/paired_manifest.csv \
        --encoder-checkpoint experiments/checkpoints/autoencoder/encoder.pth \
        --out-dir research/outputs/multiclass_dual_seed0 \
        --seed 0
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import CosineAnnealingLR

from .dataset import (
    PairedCase,
    build_loader,
    class_counts,
    inverse_frequency_weights,
    read_manifest,
)
from .labels import GRADES, NUM_CLASSES
from .metrics import classification_report
from .model import ARCHITECTURES, build_model

log = logging.getLogger(__name__)


@dataclass
class EpochRecord:
    epoch: int
    stage: int
    train_loss: float
    val_loss: float
    val_accuracy: float
    val_macro_recall: float
    val_kappa: float
    head_lr: float
    backbone_lr: float


def set_seed(seed: int) -> torch.Generator:
    """Seed every RNG the run touches and return a loader-shuffle generator."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def mixup_batch(
    ap: torch.Tensor,
    lat: torch.Tensor,
    targets: torch.Tensor,
    alpha: float,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """MixUp on both views with one shared lambda and permutation per batch.

    A shared permutation keeps each mixed sample a mixture of the *same* two
    patients in both views; independent permutations would pair patient X's AP
    with patient Y's LAT and destroy the label semantics.
    """
    if alpha <= 0:
        return ap, lat, targets, targets, 1.0
    lam = float(np.random.beta(alpha, alpha))
    index = torch.randperm(ap.size(0), generator=generator, device=ap.device)
    mixed_ap = lam * ap + (1 - lam) * ap[index]
    mixed_lat = lam * lat + (1 - lam) * lat[index]
    return mixed_ap, mixed_lat, targets, targets[index], lam


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimiser: torch.optim.Optimizer | None = None,
    mixup_alpha: float = 0.0,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """One pass. Training when ``optimiser`` is given, else evaluation.

    Returns mean loss, true labels, predicted labels and predicted probabilities.
    """
    training = optimiser is not None
    model.train(training)
    total_loss, total_n = 0.0, 0
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_prob: list[np.ndarray] = []

    for ap, lat, target, _case_ids in loader:
        ap, lat = ap.to(device, non_blocking=True), lat.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.set_grad_enabled(training):
            if training and mixup_alpha > 0:
                mixed_ap, mixed_lat, t_a, t_b, lam = mixup_batch(ap, lat, target, mixup_alpha)
                logits = model(mixed_ap, mixed_lat)
                loss = lam * criterion(logits, t_a) + (1 - lam) * criterion(logits, t_b)
                # Report predictions on the unmixed batch so the training metric
                # stays interpretable.
                with torch.no_grad():
                    logits = model(ap, lat)
            else:
                logits = model(ap, lat)
                loss = criterion(logits, target)

            if training:
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()

        batch_n = target.size(0)
        total_loss += float(loss.detach()) * batch_n
        total_n += batch_n
        probs = torch.softmax(logits.detach(), dim=1)
        all_true.append(target.detach().cpu().numpy())
        all_pred.append(probs.argmax(dim=1).cpu().numpy())
        all_prob.append(probs.cpu().numpy())

    return (
        total_loss / max(total_n, 1),
        np.concatenate(all_true),
        np.concatenate(all_pred),
        np.concatenate(all_prob),
    )


def _selection_score(report, criterion: str) -> float:
    if criterion == "accuracy":
        return report.overall_accuracy
    if criterion == "macro_recall":
        return report.macro_recall
    if criterion == "kappa":
        return report.quadratic_kappa
    raise ValueError(f"unknown --select-on {criterion!r}")


def train(args: argparse.Namespace) -> dict:
    generator = set_seed(args.seed)
    device = resolve_device(args.device)
    log.info("device: %s | seed: %d | architecture: %s", device, args.seed, args.architecture)

    train_cases = read_manifest(args.manifest, split=args.train_split)
    val_cases = read_manifest(args.manifest, split=args.val_split)
    _warn_on_case_overlap(train_cases, val_cases)
    log.info("train: %d cases %s", len(train_cases), class_counts(train_cases))
    log.info("val:   %d cases %s", len(val_cases), class_counts(val_cases))

    train_loader = build_loader(
        train_cases, batch_size=args.batch_size, train=True, input_size=args.input_size,
        num_workers=args.num_workers, image_root=args.image_root, generator=generator,
    )
    val_loader = build_loader(
        val_cases, batch_size=args.batch_size, train=False, input_size=args.input_size,
        num_workers=args.num_workers, image_root=args.image_root,
    )

    model = build_model(
        architecture=args.architecture,
        num_classes=NUM_CLASSES,
        encoder_checkpoint=args.encoder_checkpoint,
        dropout=args.dropout,
        imagenet_fallback=not args.no_imagenet,
    ).to(device)

    weights = inverse_frequency_weights(train_cases).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=args.label_smoothing)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    history: list[EpochRecord] = []
    best_score = -float("inf")
    best_epoch = -1
    best_path = args.out_dir / "best_model.pth"
    epoch = 0

    # --- stage 1: head only --------------------------------------------------
    model.set_backbone_trainable(False)
    optimiser = torch.optim.Adam(
        model.head_parameters(), lr=args.head_lr, weight_decay=args.weight_decay
    )
    log.info("stage 1: %d epochs, head only, lr=%.2e", args.stage1_epochs, args.head_lr)
    for _ in range(args.stage1_epochs):
        epoch += 1
        best_score, best_epoch = _one_epoch(
            epoch, 1, model, train_loader, val_loader, criterion, optimiser, None,
            device, args, history, best_score, best_epoch, best_path,
        )

    # --- stage 2: everything, discriminative LRs ----------------------------
    model.set_backbone_trainable(True)
    backbone_lr = args.head_lr * args.backbone_lr_scale
    optimiser = torch.optim.Adam(
        [
            {"params": list(model.backbone_parameters()), "lr": backbone_lr},
            {"params": list(model.head_parameters()), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=max(args.stage2_epochs, 1))
    log.info(
        "stage 2: %d epochs, all parameters, head lr=%.2e backbone lr=%.2e, cosine",
        args.stage2_epochs, args.head_lr, backbone_lr,
    )
    for _ in range(args.stage2_epochs):
        epoch += 1
        best_score, best_epoch = _one_epoch(
            epoch, 2, model, train_loader, val_loader, criterion, optimiser, scheduler,
            device, args, history, best_score, best_epoch, best_path,
        )

    if best_epoch < 0:
        raise RuntimeError("no epoch completed; nothing was checkpointed")

    summary = {
        "architecture": args.architecture,
        "seed": args.seed,
        "selection_criterion": args.select_on,
        "best_epoch": best_epoch,
        "best_val_score": best_score,
        "checkpoint": str(best_path),
        "classes": list(GRADES),
        "n_train": len(train_cases),
        "n_val": len(val_cases),
        "train_class_counts": class_counts(train_cases),
        "config": {
            k: (str(v) if isinstance(v, Path) else v)
            for k, v in vars(args).items()
        },
        "history": [vars(record) for record in history],
    }
    (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    _write_history_csv(history, args.out_dir / "training_curve.csv")
    log.info("best epoch %d (%s=%.4f) -> %s", best_epoch, args.select_on, best_score, best_path)
    return summary


def _one_epoch(
    epoch, stage, model, train_loader, val_loader, criterion, optimiser, scheduler,
    device, args, history, best_score, best_epoch, best_path,
) -> tuple[float, int]:
    started = time.time()
    train_loss, *_ = run_epoch(
        model, train_loader, criterion, device, optimiser, mixup_alpha=args.mixup_alpha
    )
    val_loss, y_true, y_pred, _ = run_epoch(model, val_loader, criterion, device)
    if scheduler is not None:
        scheduler.step()

    report = classification_report(y_true, y_pred, n_boot=0)
    lrs = [group["lr"] for group in optimiser.param_groups]
    record = EpochRecord(
        epoch=epoch, stage=stage, train_loss=train_loss, val_loss=val_loss,
        val_accuracy=report.overall_accuracy, val_macro_recall=report.macro_recall,
        val_kappa=report.quadratic_kappa,
        head_lr=lrs[-1], backbone_lr=lrs[0] if len(lrs) > 1 else 0.0,
    )
    history.append(record)
    log.info(
        "epoch %02d (stage %d) train_loss=%.4f val_loss=%.4f val_acc=%.1f%% "
        "macro_recall=%.1f%% kappa=%.3f (%.0fs)",
        epoch, stage, train_loss, val_loss, report.overall_accuracy,
        report.macro_recall, report.quadratic_kappa, time.time() - started,
    )

    score = _selection_score(report, args.select_on)
    if score > best_score:
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "architecture": args.architecture,
                "epoch": epoch,
                "classes": list(GRADES),
                "selection_criterion": args.select_on,
                "selection_score": score,
                "input_size": args.input_size,
                "dropout": args.dropout,
            },
            best_path,
        )
        return score, epoch
    return best_score, best_epoch


def _warn_on_case_overlap(a: list[PairedCase], b: list[PairedCase]) -> None:
    """Fail loudly on patient-level leakage between two splits."""
    overlap = {c.case_id for c in a} & {c.case_id for c in b}
    if overlap:
        raise ValueError(
            f"{len(overlap)} case ID(s) appear in both splits -- patient-level leakage: "
            f"{', '.join(sorted(overlap)[:10])}"
        )


def _write_history_csv(history: list[EpochRecord], path: Path) -> None:
    import csv

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(vars(history[0])))
        writer.writeheader()
        for record in history:
            writer.writerow(vars(record))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the end-to-end 5-class Gartland baseline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    data = parser.add_argument_group("data")
    data.add_argument("--manifest", type=Path, required=True,
                      help="Paired manifest from manifest.py.")
    data.add_argument("--image-root", type=Path,
                      help="Re-root manifest image paths (for a different machine).")
    data.add_argument("--train-split", default="train")
    data.add_argument("--val-split", default="val")
    data.add_argument("--input-size", type=int, default=224)
    data.add_argument("--batch-size", type=int, default=16)
    data.add_argument("--num-workers", type=int, default=2)

    arch = parser.add_argument_group("model")
    arch.add_argument("--architecture", choices=sorted(ARCHITECTURES), default="dual")
    arch.add_argument("--encoder-checkpoint", type=Path,
                      help="Autoencoder-pretrained ResNet-18 encoder, as used by the cascade.")
    arch.add_argument("--no-imagenet", action="store_true",
                      help="Do not fall back to ImageNet weights when no checkpoint is given.")
    arch.add_argument("--dropout", type=float, default=0.3)

    optim = parser.add_argument_group("optimisation")
    optim.add_argument("--stage1-epochs", type=int, default=10)
    optim.add_argument("--stage2-epochs", type=int, default=20)
    optim.add_argument("--head-lr", type=float, default=1e-3)
    optim.add_argument("--backbone-lr-scale", type=float, default=0.1,
                       help="Backbone LR as a fraction of the head LR (eta in equation 5).")
    optim.add_argument("--weight-decay", type=float, default=1e-5)
    optim.add_argument("--label-smoothing", type=float, default=0.0)
    optim.add_argument("--mixup-alpha", type=float, default=0.0)
    optim.add_argument("--select-on", choices=("accuracy", "macro_recall", "kappa"),
                       default="accuracy",
                       help="Validation metric for checkpoint selection "
                            "(default matches the cascade).")

    run = parser.add_argument_group("run")
    run.add_argument("--out-dir", type=Path, required=True)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--device", default="auto")
    run.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )
    train(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
