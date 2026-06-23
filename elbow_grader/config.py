"""User-facing configuration for ElbowGrader.

Only exposes the knobs that a user would normally want to change.
All other inference parameters (SAM2 probe counts, CLAHE settings, etc.)
are kept internal in pipeline.config.PipelineConfig.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class GraderConfig:
    """Configuration for ElbowGrader.

    Parameters
    ----------
    confidence_threshold : float
        Minimum winning-class probability to accept a CNN prediction.
        Below this the result is withheld and flagged as low-confidence.
        Range [0.5, 1.0). Default 0.5 = always accept argmax.
    run_full_lat_alignment : bool
        True  → run the full 6-step alignment pipeline on LAT inputs.
                 Use this for raw clinical X-rays.
        False → skip alignment; only remove L/R markers.
                 Use this if your images are already pre-aligned.
    run_sam2 : bool
        True  → run SAM2 bone segmentation and the geometric grading track.
        False → CNN predictions only (faster; no geometric explainability).
    device : str
        Compute device. "auto" selects CUDA if available, else CPU.
        Override with "cpu" or "cuda".
    """

    confidence_threshold: float = 0.5
    run_full_lat_alignment: bool = True
    run_sam2: bool = True
    device: str = "auto"
