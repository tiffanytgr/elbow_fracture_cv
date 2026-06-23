"""Bridge to experiments/align_image.align_xray (the alignment that produced
aligned_images/).

The user confirmed aligned_images was produced via:
    preprocess_exp3_from_csv(... output_size=(256,256), zoom_out=0.65,
                              padding_ratio=0.2)
which calls align_xray with those exact settings and remove_markers=False.

This module loads align_image.py at import time and exposes a single
align_canonical() function with the same return shape as my old in-package
align_full so pipeline.py can call either interchangeably.
"""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ALIGN_PATH   = _PROJECT_ROOT / "experiments" / "align_image.py"

_align_xray = None
_load_error: Optional[str] = None


def _load():
    global _align_xray, _load_error
    if _align_xray is not None or _load_error is not None:
        return
    try:
        spec = importlib.util.spec_from_file_location(
            "_kkh_align_image_canonical", _ALIGN_PATH)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        _align_xray = mod.align_xray
    except Exception as e:
        _load_error = f"Could not load {_ALIGN_PATH}: {e}"


def is_available() -> bool:
    _load()
    return _align_xray is not None


def align_canonical(img_gray: np.ndarray,
                     output_size: tuple = (256, 256),
                     zoom_out: float = 0.65,
                     padding_ratio: float = 0.2,
                     remove_markers: bool = False,
                     marker_brightness: int = 180,
                     **_unused) -> dict:
    """Run experiments/align_image.align_xray with the canonical settings.

    Defaults match what was used to produce aligned_images:
      output_size=(256,256), zoom_out=0.65, padding_ratio=0.2, remove_markers=False.

    Marker removal is False by default because the canonical script doesn't do
    it during alignment -- pipeline.py applies it afterwards on the aligned image.
    """
    _load()
    if _align_xray is None:
        raise RuntimeError(_load_error or "align_xray not loaded")

    pil = Image.fromarray(img_gray).convert("RGB")
    aligned_pil = _align_xray(
        pil,
        output_size=output_size,
        zoom_out=zoom_out,
        padding_ratio=padding_ratio,
        remove_markers=remove_markers,
        marker_brightness=marker_brightness,
    )
    aligned = np.array(aligned_pil.convert("L"), dtype=np.uint8)

    return {
        "aligned": aligned,
        "cleaned": img_gray,   # unchanged grayscale; pipeline.py applies marker removal separately
        "flip_lr": None,
        "rotation": None,
        "mask": None,
    }
