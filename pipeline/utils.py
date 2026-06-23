"""Logging, image I/O, device resolution."""
from __future__ import annotations
import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image


_LOGGER_NAME = "pipeline"


def get_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if not logger.handlers:
        h = logging.StreamHandler(sys.stdout)
        h.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(h)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    return logger


def resolve_device(spec: str = "auto") -> str:
    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec


def load_image_gray(path: str | Path) -> np.ndarray:
    """Load image as uint8 grayscale numpy array (respects EXIF orientation)."""
    from PIL import ImageOps
    img = ImageOps.exif_transpose(Image.open(str(path))).convert("L")
    return np.array(img, dtype=np.uint8)


def load_image_pil_rgb(path: str | Path) -> Image.Image:
    return Image.open(str(path)).convert("RGB")


def numpy_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    if arr.ndim == 2:
        return Image.fromarray(arr).convert("RGB")
    return Image.fromarray(arr)


def file_sha1(path: str | Path) -> str:
    """SHA1 hash of an image file — used for audit logging.

    Lets the prospective study tie a recorded prediction back to the exact
    bytes that were classified, even if the same filename is reused.
    """
    h = hashlib.sha1()
    with open(str(path), "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def append_jsonl(path: str | Path, record: dict[str, Any]) -> None:
    """Append a single JSON record to a .jsonl audit log."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")
