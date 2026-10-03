"""Paired AP+LAT dataset for the end-to-end multiclass baseline.

Augmentation deliberately mirrors the cascade's supervised recipe (report
section 2.7.1) so that the baseline differs from the cascade in *decision
structure only*, not in preprocessing:

  random rotation (+/-15 deg), affine translate 5%, scale 0.95-1.05,
  random sharpness (factor 2, p=0.5), random autocontrast (p=0.3),
  colour jitter (brightness 0.2, contrast 0.3), ImageNet normalisation.

The two views are augmented independently. They are different acquisitions of
the same elbow, so a shared transform would not correspond to any real
geometric relationship between them.
"""
from __future__ import annotations

import csv
import logging
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .labels import GRADES, NUM_CLASSES
from .manifest import MANIFEST_COLUMNS

log = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class PairedCase:
    case_id: str
    ap_path: Path
    lat_path: Path
    label: int
    split: str


def read_manifest(path: Path, split: str | None = None) -> list[PairedCase]:
    """Load the paired manifest written by manifest.py, optionally one split."""
    rows: list[PairedCase] = []
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = set(MANIFEST_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        for line_no, row in enumerate(reader, start=2):
            if split is not None and row["split"] != split:
                continue
            try:
                label = int(row["label_idx"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{line_no}: bad label_idx {row['label_idx']!r}") from exc
            if not 0 <= label < NUM_CLASSES:
                raise ValueError(
                    f"{path}:{line_no}: label_idx {label} outside 0..{NUM_CLASSES - 1}"
                )
            rows.append(
                PairedCase(
                    case_id=row["case_id"],
                    ap_path=Path(row["ap_path"]),
                    lat_path=Path(row["lat_path"]),
                    label=label,
                    split=row["split"],
                )
            )
    if not rows:
        raise ValueError(f"{path} contains no rows for split={split!r}")
    return rows


def build_transform(input_size: int = 224, train: bool = False) -> T.Compose:
    """Training or evaluation transform, matching the cascade's recipe."""
    steps: list[object] = [T.Resize((input_size, input_size))]
    if train:
        steps += [
            T.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            T.RandomAdjustSharpness(sharpness_factor=2, p=0.5),
            T.RandomAutocontrast(p=0.3),
            T.ColorJitter(brightness=0.2, contrast=0.3),
        ]
    steps += [T.ToTensor(), T.Normalize(list(IMAGENET_MEAN), list(IMAGENET_STD))]
    return T.Compose(steps)


class PairedViewDataset(Dataset):
    """Yields ``(ap_tensor, lat_tensor, label, case_id)`` per patient.

    Images are opened as RGB so the ImageNet-pretrained first convolution and
    the normalisation constants apply unchanged, exactly as in the cascade.
    """

    def __init__(
        self,
        cases: list[PairedCase],
        input_size: int = 224,
        train: bool = False,
        image_root: Path | None = None,
    ) -> None:
        if not cases:
            raise ValueError("PairedViewDataset given no cases")
        self.cases = cases
        self.image_root = Path(image_root) if image_root else None
        self.transform = build_transform(input_size, train=train)

    def __len__(self) -> int:
        return len(self.cases)

    def _resolve(self, path: Path) -> Path:
        """Re-root a manifest path, so manifests written on one machine work on another."""
        if self.image_root is None:
            return path
        if path.is_absolute():
            # Keep the trailing components that exist under the new root.
            parts = path.parts
            for start in range(len(parts)):
                candidate = self.image_root.joinpath(*parts[start:])
                if candidate.exists():
                    return candidate
            return path
        return self.image_root / path

    def _load(self, path: Path) -> torch.Tensor:
        resolved = self._resolve(path)
        try:
            with Image.open(resolved) as img:
                return self.transform(img.convert("RGB"))
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"radiograph not found: {resolved} (manifest path {path}); pass "
                f"--image-root if the manifest was written on another machine"
            ) from exc

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, str]:
        case = self.cases[index]
        return self._load(case.ap_path), self._load(case.lat_path), case.label, case.case_id


def class_counts(cases: list[PairedCase]) -> list[int]:
    counts = Counter(c.label for c in cases)
    return [counts.get(i, 0) for i in range(NUM_CLASSES)]


def inverse_frequency_weights(cases: list[PairedCase]) -> torch.Tensor:
    """Class weights w_c = N / (C * N_c), matching the cascade's equation (6).

    Classes absent from the training split get weight 0: they contribute no
    loss term, which is correct, and avoids a division by zero.
    """
    counts = class_counts(cases)
    total = sum(counts)
    weights = [
        (total / (NUM_CLASSES * n)) if n > 0 else 0.0
        for n in counts
    ]
    for grade, count, weight in zip(GRADES, counts, weights):
        if count == 0:
            log.warning("class %s absent from this split -- loss weight set to 0", grade)
        else:
            log.info("class %-9s n=%-4d weight=%.3f", grade, count, weight)
    return torch.tensor(weights, dtype=torch.float32)


def build_loader(
    cases: list[PairedCase],
    batch_size: int = 16,
    train: bool = False,
    input_size: int = 224,
    num_workers: int = 2,
    image_root: Path | None = None,
    generator: torch.Generator | None = None,
) -> DataLoader:
    dataset = PairedViewDataset(cases, input_size=input_size, train=train, image_root=image_root)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        generator=generator if train else None,
    )
