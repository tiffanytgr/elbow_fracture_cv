"""Centralised configuration: paths, model versions, thresholds.

Edit the defaults here OR pass a custom PipelineConfig() to predict().
Every magic number used at inference time should live in this file so
that the prospective study has a single source of truth for what the
model was running.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# Project root resolves relative to this file
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_EXPERIMENTS  = _PROJECT_ROOT / "experiments"
_CHECKPOINTS  = _PROJECT_ROOT / "checkpoints"


@dataclass
class PipelineConfig:
    # === Model checkpoints ===
    # Set to None for experiments that aren't trained yet -- those steps
    # will be skipped at runtime with a clear log message.
    exp1_ckpt: Optional[Path] = _EXPERIMENTS / "checkpoints" / "exp1" / "best_model_for_analysis.pth"
    exp2_ckpt: Optional[Path] = _EXPERIMENTS / "checkpoints" / "exp2" / "best_model_for_analysis.pth"
    exp3_ckpt: Optional[Path] = _EXPERIMENTS / "checkpoints" / "exp3" / "best_model_for_analysis.pth"
    exp4_ckpt: Optional[Path] = _EXPERIMENTS / "checkpoints" / "exp4" / "best_model_for_analysis.pth"
    cap_regressor_ckpt: Path  = _EXPERIMENTS / "checkpoints" / "cap_regressor" / "best_model.pth"
    sam2_ckpt: Path           = _CHECKPOINTS / "sam2_hiera_large.pt"
    sam2_config: str          = "sam2_hiera_l.yaml"
    yolo_baumann_ckpt: Optional[Path] = _PROJECT_ROOT.parent / "explainability" / "exp.pt"

    # === DRUE OOD filter checkpoints (per experiment) ===
    drue_exp1_ckpt: Optional[Path] = _EXPERIMENTS / "drue_outputs_exp1" / "drue_decoders.pth"
    drue_exp2_ckpt: Optional[Path] = _EXPERIMENTS / "drue_outputs_exp2" / "drue_decoders.pth"
    drue_exp3_ckpt: Optional[Path] = _EXPERIMENTS / "drue_outputs_exp3" / "drue_decoders.pth"
    drue_exp4_ckpt: Optional[Path] = _EXPERIMENTS / "drue_outputs_exp4" / "drue_decoders.pth"

    # === DRUE OOD rejection thresholds ===
    # An image is flagged OOD if its DRUE score > the threshold for that
    # experiment. Set to None to disable rejection (score is still reported).
    #
    # How to fill these in:
    #   1. Run the DRUE training notebook for each experiment.
    #   2. At the end, read drue_scores.json (saved by the notebook).
    #   3. Compute the percentile you want as the cutoff. The notebooks
    #      use P90 by default -- replicate with:
    #          import json, numpy as np
    #          scores = [r["uncertainty"] for r in json.load(open("drue_scores.json"))]
    #          print(f"P90 = {np.percentile(scores, 90):.6f}")
    #   4. Paste that number below for the matching experiment.
    #
    # Per-class thresholds aren't supported here on purpose -- at inference
    # you don't know the class label, so a single global cutoff is what
    # actually applies. If you want stricter behaviour, lower the value;
    # for looser, raise it. Re-derive after every retrain.
    drue_threshold_exp1: Optional[float] = None  # TODO: fill from drue_outputs_exp1/drue_scores.json
    drue_threshold_exp2: Optional[float] = None  # TODO: fill from drue_outputs_exp2/drue_scores.json
    drue_threshold_exp3: Optional[float] = None  # TODO: fill from drue_outputs_exp3/drue_scores.json
    drue_threshold_exp4: Optional[float] = None  # TODO: fill from drue_outputs_exp4/drue_scores.json
    drue_reject_above_threshold: bool = False    # set True once you've filled the thresholds above

    # === Confidence threshold ===
    # If the winning class probability is below this value the prediction is
    # withheld (pred_idx set to -1) and flagged as low-confidence.  The
    # probability bars are still shown in the UI so the user can inspect them.
    # Range [0.5, 1.0).  Default 0.5 = always accept argmax.
    confidence_threshold: float = 0.5

    # === CNN classifier preprocessing ===
    classifier_input_size: int = 224
    classifier_mean: tuple = (0.485, 0.456, 0.406)
    classifier_std:  tuple = (0.229, 0.224, 0.225)

    # === Alignment parameters ===
    align_downsample_size: int = 128
    align_output_size: tuple   = (256, 256)
    align_zoom_out: float      = 0.65
    align_padding_ratio: float = 0.05   # use align_full default (alignment notebook)
    # If True, run the full 6-step alignment pipeline on every input.
    # If False, assume images are already aligned and only do marker removal.
    run_full_alignment: bool = True
    # If True, alignment goes through experiments/align_image.align_xray
    # called with the same settings used by preprocess_exp3_from_csv:
    # output_size=(256,256), zoom_out=0.65, padding_ratio=0.20,
    # remove_markers=False. This is what produced aligned_images/.
    use_canonical_alignment: bool = True
    # The notebook does NOT remove markers during alignment. Marker removal
    # for the CNN classifier input happens separately in pipeline.py.
    canonical_remove_markers: bool = False
    canonical_marker_brightness: int = 180

    # === Marker removal ===
    marker_max_area_frac: float = 0.005
    marker_min_brightness: int  = 220

    # === CLAHE (capitellum regressor + SAM2) ===
    clahe_clip: float        = 3.0
    clahe_grid: tuple        = (8, 8)
    cap_regressor_size: int  = 256

    # === SAM2 bone segmentation ===
    sam_k_positive: int    = 15
    sam_n_negative: int    = 5
    sam_min_dist: int      = 15
    sam_box_pad: int       = 20    # was 10; widened for drift tolerance
    sam_box_x_percentile: int = 60 # was hardcoded 40; raise to include more probes
    sam_n_zones: int       = 3

    # === SAM2 capitellum ===
    cap_box_r_x: int           = 25
    cap_box_r_up: int          = 15
    cap_box_r_dn: int          = 30
    cap_shaft_neg_offset: int  = 35
    cap_hum_erosion_iters: int = 3
    cap_area_loss_frac: float  = 0.30

    # === Geometric grading thresholds ===
    # These were tuned on retrospective data. Document any change in the
    # prospective study protocol -- they directly affect predictions.
    anterior_edge: str        = "right"
    bisect_leeway: float      = 0.30   # AHL must hit middle band to be Grade 1
    width_cv_thresh: float    = 0.10   # shaft width CV >= this -> Grade 2b
    matched_ratio_thr: float  = 0.80   # min/max width ratio >= this -> Grade 2a
    n_width_samples: int      = 40
    width_smooth_win: int     = 5
    width_height_px: int      = 30     # measure this many px above capitellum
    shaft_pca_frac: float     = 0.70

    # === Runtime ===
    device: str = "auto"  # "auto" | "cuda" | "cpu"

    # === Logging ===
    log_level: str = "INFO"
    log_predictions_to: Optional[Path] = None  # JSONL path for audit log


def default_config() -> PipelineConfig:
    """Return the project-default config. Override fields as needed."""
    return PipelineConfig()
