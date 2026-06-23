"""End-to-end orchestration: predict() entry point.

View routing (per user spec):
    AP only          -> Exp1 (Normal vs Fractured) -> Exp2 (Grade 3 vs G1/2)
    LAT only         -> Exp3 (G1 vs G2) -> Exp4 (G2a vs G2b)  + SAM2 explainability
    AP + LAT (both)  -> AP cascade first; if not Grade 3, run LAT cascade

Stubbed steps (no checkpoint available yet):
    - Exp1 (AP: Normal vs Fractured)         — assumes "Fractured" and proceeds
    - Exp4 (LAT: Grade 2a vs 2b CNN)         — falls back to SAM2 cortical-width grade
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch

from .config import PipelineConfig, default_config
from .utils import (get_logger, resolve_device, load_image_gray,
                     numpy_to_pil_rgb, file_sha1, append_jsonl)
from .preprocessing import align_full, remove_lr_markers, apply_clahe
from .classifiers import load_classifier, make_transform, classify
from .gradcam import compute_gradcam
from .ood import DRUEScorer
from .capitellum import load_regressor, predict_capitellum
from .segmentation import SAM2Segmenter
from .geometry import (fit_ahl, grade_ahl_bisection, cortical_width_profile,
                        grade_2a_vs_2b)
from .baumann import compute_baumann_angle


# ────────────────────────────────────────────────────────────────────
# Result types
# ────────────────────────────────────────────────────────────────────

@dataclass
class ClassifierResult:
    label: str
    pred_idx: int
    probs: list[float]
    labels: list[str]
    ood_score: Optional[float] = None
    ood_flagged: bool = False
    skipped_reason: Optional[str] = None


@dataclass
class BaumannResult:
    status: str                              # 'ok' | 'no_shaft' | 'no_physis' | 'error: ...'
    baumann_angle_deg: Optional[float] = None
    in_normal_range: Optional[bool] = None   # True if 60–84°
    shaft_angle_deg: Optional[float] = None
    physis_angle_deg: Optional[float] = None
    physis_confidence: Optional[float] = None


@dataclass
class GeometricResult:
    grade_1v2: Optional[str] = None
    grade_2ab: Optional[str] = None
    final_grade: Optional[str] = None
    ahl_diagnostic: Optional[dict] = None
    width_profile: Optional[dict] = None
    bone_sam_score: Optional[float] = None
    capitellum_sam_score: Optional[float] = None
    capitellum_xy_regressor: Optional[tuple[int, int]] = None
    capitellum_xy_sam: Optional[tuple[int, int]] = None
    skipped_reason: Optional[str] = None


@dataclass
class PredictionResult:
    # Inputs
    ap_path: Optional[str] = None
    lat_path: Optional[str] = None
    ap_sha1: Optional[str] = None
    lat_sha1: Optional[str] = None

    # Per-experiment CNN results
    exp1: Optional[ClassifierResult] = None  # AP: Normal vs Fractured
    exp2: Optional[ClassifierResult] = None  # AP: Grade 3 vs G1/2
    exp3: Optional[ClassifierResult] = None  # LAT: G1 vs G2
    exp4: Optional[ClassifierResult] = None  # LAT: G2a vs G2b

    # AP explainability
    baumann: Optional[BaumannResult] = None

    # SAM2 explainability track (LAT only)
    geometric: Optional[GeometricResult] = None

    # Final consensus
    cnn_final_grade: Optional[str] = None
    final_grade: Optional[str] = None  # for the demo: best single label to show
    grade_source: Optional[str] = None  # "cnn", "geometric", "consensus"

    # Discordant flag (per user spec #5: show both)
    discordant: bool = False

    # Provenance
    pipeline_version: str = "0.1.0"
    config_snapshot: Optional[dict] = None
    log: list[str] = field(default_factory=list)

    # Non-serialisable artifacts for the demo app (masks, aligned image, etc.).
    # Excluded from to_dict() so the JSON audit log stays clean.
    debug_artifacts: dict = field(default_factory=dict, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("debug_artifacts", None)
        if d.get("geometric") and d["geometric"].get("width_profile"):
            wp = d["geometric"]["width_profile"]
            if "widths" in wp and len(wp["widths"]) > 100:
                wp["widths"] = wp["widths"][:100]
        return d


# ────────────────────────────────────────────────────────────────────
# Internal: lazy resource cache
# ────────────────────────────────────────────────────────────────────

class _Resources:
    """Holds loaded models so repeated predict() calls don't re-load them."""
    def __init__(self, config: PipelineConfig, device: str):
        self.config = config
        self.device = device
        self._classifiers: dict[str, Any] = {}
        self._drue: dict[str, Any] = {}
        self._cap_regressor = None
        self._sam2: Optional[SAM2Segmenter] = None
        self._yolo_baumann = None
        self._classifier_transform = make_transform(
            config.classifier_input_size, config.classifier_mean, config.classifier_std)

    def classifier(self, name: str, ckpt: Path):
        if name not in self._classifiers:
            self._classifiers[name] = load_classifier(ckpt, num_classes=2, device=self.device)
        return self._classifiers[name]

    def drue(self, name: str, decoders_ckpt: Path,
              classifier_ckpt: Optional[Path] = None) -> DRUEScorer:
        if name not in self._drue:
            threshold = getattr(self.config, f"drue_threshold_{name}", None)
            self._drue[name] = DRUEScorer(
                decoders_ckpt=decoders_ckpt,
                classifier_ckpt=classifier_ckpt,
                img_size=self.config.classifier_input_size,
                device=self.device,
                threshold=threshold,
            )
        return self._drue[name]

    def cap_regressor(self):
        if self._cap_regressor is None:
            self._cap_regressor = load_regressor(self.config.cap_regressor_ckpt, self.device)
        return self._cap_regressor

    def sam2(self) -> SAM2Segmenter:
        if self._sam2 is None:
            self._sam2 = SAM2Segmenter(
                self.config.sam2_config, self.config.sam2_ckpt, self.device)
            self._sam2.load()
        return self._sam2

    def yolo_baumann(self):
        """Lazy-load the YOLO humerus segmentation model for the Baumann pipeline."""
        if self._yolo_baumann is None:
            ckpt = getattr(self.config, "yolo_baumann_ckpt", None)
            if ckpt is None or not Path(ckpt).exists():
                return None
            # Patch torchvision metadata lookup that can be corrupted in some envs
            try:
                import torchvision as _tv
                import importlib.metadata as _im
                _orig = _im.version
                _im.version = lambda n: _tv.__version__ if n == "torchvision" else _orig(n)
            except Exception:
                pass
            from ultralytics import YOLO
            self._yolo_baumann = YOLO(str(ckpt))
        return self._yolo_baumann


_RESOURCE_CACHE: dict[int, _Resources] = {}


def _get_resources(config: PipelineConfig) -> _Resources:
    """Return a process-wide cache of loaded models keyed by config identity."""
    key = id(config)
    if key not in _RESOURCE_CACHE:
        _RESOURCE_CACHE[key] = _Resources(config, resolve_device(config.device))
    return _RESOURCE_CACHE[key]


# ────────────────────────────────────────────────────────────────────
# Sub-pipelines: AP cascade and LAT cascade
# ────────────────────────────────────────────────────────────────────

def _run_ap_cascade(ap_path: str, res: _Resources, log: list[str],
                     artifacts: dict | None = None) -> tuple[ClassifierResult, ClassifierResult]:
    cfg = res.config
    ap_gray = load_image_gray(ap_path)
    ap_cleaned, _ = remove_lr_markers(ap_gray, cfg.marker_max_area_frac, cfg.marker_min_brightness)
    ap_pil = numpy_to_pil_rgb(ap_cleaned)
    if artifacts is not None:
        artifacts["ap_pil"] = ap_pil
        artifacts["ap_cleaned_gray"] = ap_cleaned

    # Exp 1 — STUBBED (no checkpoint)
    if cfg.exp1_ckpt is None or not Path(cfg.exp1_ckpt).exists():
        log.append("Exp1 skipped: no checkpoint — assuming Fractured.")
        exp1 = ClassifierResult(
            label="exp1_normal_vs_fractured",
            pred_idx=1, probs=[0.0, 1.0],
            labels=["Normal", "Fractured"],
            skipped_reason="checkpoint_missing")
    else:
        ood_score = None
        ood_flag = False
        if cfg.drue_exp1_ckpt and Path(cfg.drue_exp1_ckpt).exists():
            try:
                scorer = res.drue("exp1", cfg.drue_exp1_ckpt, cfg.exp1_ckpt)
                ood_flag, ood_score, _ = scorer.is_ood(ap_pil)
            except Exception as e:
                log.append(f"DRUE exp1 failed: {e}")
        m = res.classifier("exp1", cfg.exp1_ckpt)
        pred, probs = classify(m, ap_pil, res._classifier_transform, res.device)
        _thr = cfg.confidence_threshold
        _low_conf1 = probs[pred] < _thr
        exp1 = ClassifierResult(
            label="exp1_normal_vs_fractured",
            pred_idx=-1 if _low_conf1 else pred, probs=probs.tolist(),
            labels=["Normal", "Fractured"],
            ood_score=ood_score, ood_flagged=bool(ood_flag and cfg.drue_reject_above_threshold),
            skipped_reason=f"low_confidence: p={probs[pred]:.2f} < {_thr:.2f}" if _low_conf1 else None)
        if _low_conf1:
            log.append(f"Exp1: low confidence (p={probs[pred]:.2f} < {_thr:.2f}) — abstaining")
        else:
            log.append(f"Exp1: {exp1.labels[pred]} (p={probs[pred]:.3f})")
            if exp1.labels[pred] == "Normal":
                return exp1, None

    # Exp 2
    ood_score = None
    ood_flag = False
    if cfg.drue_exp2_ckpt and Path(cfg.drue_exp2_ckpt).exists():
        try:
            scorer = res.drue("exp2", cfg.drue_exp2_ckpt, cfg.exp2_ckpt)
            ood_flag, ood_score, _ = scorer.is_ood(ap_pil)
        except Exception as e:
            log.append(f"DRUE exp2 failed: {e}")
    if cfg.exp2_ckpt is None or not Path(cfg.exp2_ckpt).exists():
        log.append("Exp2 skipped: no checkpoint.")
        return exp1, ClassifierResult(
            label="exp2_g3_vs_g12", pred_idx=-1, probs=[],
            labels=["Grade 1/2", "Grade 3"], skipped_reason="checkpoint_missing")

    m = res.classifier("exp2", cfg.exp2_ckpt)
    pred, probs = classify(m, ap_pil, res._classifier_transform, res.device)
    _thr = cfg.confidence_threshold
    _low_conf2 = probs[pred] < _thr
    exp2 = ClassifierResult(
        label="exp2_g3_vs_g12",
        pred_idx=-1 if _low_conf2 else pred, probs=probs.tolist(),
        labels=["Grade 1/2", "Grade 3"],
        ood_score=ood_score, ood_flagged=bool(ood_flag and cfg.drue_reject_above_threshold),
        skipped_reason=f"low_confidence: p={probs[pred]:.2f} < {_thr:.2f}" if _low_conf2 else None)
    if _low_conf2:
        log.append(f"Exp2: low confidence (p={probs[pred]:.2f} < {_thr:.2f}) — abstaining")
    else:
        log.append(f"Exp2: {exp2.labels[pred]} (p={probs[pred]:.3f})")
    return exp1, exp2


def _run_lat_cnn_cascade(lat_pil, res: _Resources, log: list[str]) -> tuple[Optional[ClassifierResult], Optional[ClassifierResult]]:
    cfg = res.config
    exp3 = exp4 = None

    if cfg.exp3_ckpt is None or not Path(cfg.exp3_ckpt).exists():
        log.append("Exp3 skipped: no checkpoint.")
        return None, None

    ood_score = None
    ood_flag = False
    if cfg.drue_exp3_ckpt and Path(cfg.drue_exp3_ckpt).exists():
        try:
            scorer = res.drue("exp3", cfg.drue_exp3_ckpt, cfg.exp3_ckpt)
            ood_flag, ood_score, _ = scorer.is_ood(lat_pil)
        except Exception as e:
            log.append(f"DRUE exp3 failed: {e}")

    m = res.classifier("exp3", cfg.exp3_ckpt)
    pred, probs = classify(m, lat_pil, res._classifier_transform, res.device)
    _thr = cfg.confidence_threshold
    _low_conf3 = probs[pred] < _thr
    exp3 = ClassifierResult(
        label="exp3_g1_vs_g2",
        pred_idx=-1 if _low_conf3 else pred, probs=probs.tolist(),
        labels=["Grade 1", "Grade 2"],
        ood_score=ood_score, ood_flagged=bool(ood_flag and cfg.drue_reject_above_threshold),
        skipped_reason=f"low_confidence: p={probs[pred]:.2f} < {_thr:.2f}" if _low_conf3 else None)
    if _low_conf3:
        log.append(f"Exp3: low confidence (p={probs[pred]:.2f} < {_thr:.2f}) — abstaining")
    else:
        log.append(f"Exp3: {exp3.labels[pred]} (p={probs[pred]:.3f})")

    if not _low_conf3 and pred == 1:  # Grade 2 -> exp4
        if cfg.exp4_ckpt is None or not Path(cfg.exp4_ckpt).exists():
            log.append("Exp4 skipped: no checkpoint — will rely on geometric track for 2a/2b.")
            exp4 = ClassifierResult(
                label="exp4_g2a_vs_g2b", pred_idx=-1, probs=[],
                labels=["Grade 2a", "Grade 2b"], skipped_reason="checkpoint_missing")
        else:
            ood_score = None; ood_flag = False
            if cfg.drue_exp4_ckpt and Path(cfg.drue_exp4_ckpt).exists():
                try:
                    scorer = res.drue("exp4", cfg.drue_exp4_ckpt, cfg.exp4_ckpt)
                    ood_flag, ood_score, _ = scorer.is_ood(lat_pil)
                except Exception as e:
                    log.append(f"DRUE exp4 failed: {e}")
            m = res.classifier("exp4", cfg.exp4_ckpt)
            pred4, probs4 = classify(m, lat_pil, res._classifier_transform, res.device)
            _thr4 = cfg.confidence_threshold
            _low_conf4 = probs4[pred4] < _thr4
            exp4 = ClassifierResult(
                label="exp4_g2a_vs_g2b",
                pred_idx=-1 if _low_conf4 else pred4, probs=probs4.tolist(),
                labels=["Grade 2a", "Grade 2b"],
                ood_score=ood_score, ood_flagged=bool(ood_flag and cfg.drue_reject_above_threshold),
                skipped_reason=f"low_confidence: p={probs4[pred4]:.2f} < {_thr4:.2f}" if _low_conf4 else None)
            if _low_conf4:
                log.append(f"Exp4: low confidence (p={probs4[pred4]:.2f} < {_thr4:.2f}) — abstaining")
            else:
                log.append(f"Exp4: {exp4.labels[pred4]} (p={probs4[pred4]:.3f})")
    return exp3, exp4


def _run_geometric_track(lat_gray_aligned: np.ndarray, res: _Resources, log: list[str],
                          artifacts: Optional[dict] = None) -> GeometricResult:
    cfg = res.config
    geo = GeometricResult()
    if artifacts is None:
        artifacts = {}
    artifacts["lat_aligned"] = lat_gray_aligned

    try:
        gray_clahe = apply_clahe(lat_gray_aligned, cfg.clahe_clip, cfg.clahe_grid)
        artifacts["lat_clahe"] = gray_clahe
    except Exception as e:
        geo.skipped_reason = f"clahe_failed: {e}"
        return geo

    # Capitellum regressor
    cap_xy = None
    if Path(cfg.cap_regressor_ckpt).exists():
        try:
            reg = res.cap_regressor()
            cap_xy = predict_capitellum(reg, lat_gray_aligned,
                                         img_size=cfg.cap_regressor_size,
                                         clahe_clip=cfg.clahe_clip,
                                         clahe_grid=cfg.clahe_grid,
                                         device=res.device)
            geo.capitellum_xy_regressor = cap_xy
            log.append(f"Capitellum regressor: {cap_xy}")
        except Exception as e:
            log.append(f"Capitellum regressor failed: {e}")

    # SAM2 bone segmentation
    if not Path(cfg.sam2_ckpt).exists():
        geo.skipped_reason = "sam2_ckpt_missing"
        return geo
    try:
        sam = res.sam2()
        bone = sam.segment_bone(
            gray_clahe, capitellum_xy=cap_xy,
            k_positive=cfg.sam_k_positive,
            n_negative=cfg.sam_n_negative,
            min_dist=cfg.sam_min_dist,
            box_pad=cfg.sam_box_pad,
            n_zones=cfg.sam_n_zones,
            x_percentile=cfg.sam_box_x_percentile,
        )
        geo.bone_sam_score = bone["sam_score"]
        humerus_mask = bone["humerus_mask"]
        artifacts["humerus_mask"] = humerus_mask
        artifacts["forearm_mask"] = bone.get("forearm_mask")
        artifacts["bone_mask"]    = bone.get("bone_mask")
    except Exception as e:
        geo.skipped_reason = f"sam_bone_failed: {e}"
        return geo

    # SAM2 capitellum
    probe = cap_xy if cap_xy is not None else (humerus_mask.shape[1] // 2, humerus_mask.shape[0] // 2)
    try:
        cap = sam.segment_capitellum(
            gray_clahe, humerus_mask, probe,
            box_r_x=cfg.cap_box_r_x, box_r_up=cfg.cap_box_r_up,
            box_r_dn=cfg.cap_box_r_dn,
            shaft_neg_offset=cfg.cap_shaft_neg_offset,
            hum_erosion_iters=cfg.cap_hum_erosion_iters,
            area_loss_frac=cfg.cap_area_loss_frac,
        )
        capitellum_mask = cap["capitellum_mask"]
        geo.capitellum_sam_score = cap["sam_score"]
        geo.capitellum_xy_sam = cap["centroid"]
        artifacts["capitellum_mask"]   = capitellum_mask
        artifacts["capitellum_radius"] = cap["radius"]
    except Exception as e:
        geo.skipped_reason = f"sam_capitellum_failed: {e}"
        return geo

    # AHL fitting + Grade 1 vs 2
    ahl = fit_ahl(humerus_mask, anterior_edge=cfg.anterior_edge)
    if ahl is None:
        geo.skipped_reason = "ahl_fit_failed"
        return geo
    ahl_diag = grade_ahl_bisection(
        ahl[0], ahl[1], cap["centroid"], cap["radius"], capitellum_mask,
        anterior_edge=cfg.anterior_edge, bisect_leeway=cfg.bisect_leeway)
    geo.ahl_diagnostic = ahl_diag
    geo.grade_1v2 = ahl_diag["grade_1v2"]
    log.append(f"Geometric Grade 1 vs 2: {geo.grade_1v2}")

    if geo.grade_1v2 == "Grade 1":
        geo.final_grade = "Grade 1"
        return geo

    # Grade 2: cortical width to discriminate 2a vs 2b
    # Use the regressor point (top of capitellum, near shaft) rather than
    # the SAM centroid (middle of capitellum) so sampling starts right
    # above the capitellum with no gap.
    cap_for_width = cap_xy if cap_xy is not None else cap["centroid"]
    wp = cortical_width_profile(humerus_mask, cap_for_width,
                                  height_px=cfg.width_height_px,
                                  n_samples=cfg.n_width_samples,
                                  smooth_win=cfg.width_smooth_win,
                                  shaft_pca_frac=cfg.shaft_pca_frac)
    geo.width_profile = wp
    geo.grade_2ab = grade_2a_vs_2b(wp, matched_ratio_thr=cfg.matched_ratio_thr)
    geo.final_grade = geo.grade_2ab
    log.append(f"Geometric Grade 2a vs 2b: {geo.grade_2ab} (match_ratio={wp.get('match_ratio', 0):.3f})")
    return geo


def _generate_gradcams(result: PredictionResult, res: _Resources,
                        cfg: PipelineConfig) -> None:
    """Generate Grad-CAM heatmaps for each experiment that produced a prediction."""
    arts = result.debug_artifacts
    tf = res._classifier_transform

    _exp_specs = [
        ("exp1", result.exp1, cfg.exp1_ckpt, "ap_pil"),
        ("exp2", result.exp2, cfg.exp2_ckpt, "ap_pil"),
        ("exp3", result.exp3, cfg.exp3_ckpt, "lat_pil"),
        ("exp4", result.exp4, cfg.exp4_ckpt, "lat_pil"),
    ]
    for name, cr, ckpt, img_key in _exp_specs:
        if cr is None or cr.pred_idx < 0 or img_key not in arts:
            continue
        if ckpt is None or not Path(ckpt).exists():
            continue
        try:
            m = res.classifier(name, ckpt)
            gc = compute_gradcam(m, arts[img_key], tf, res.device,
                                  class_idx=cr.pred_idx,
                                  input_size=cfg.classifier_input_size)
            arts[f"gradcam_{name}"] = gc
            result.log.append(f"Grad-CAM generated for {name}")
        except Exception as e:
            result.log.append(f"Grad-CAM {name} failed: {e}")


# ────────────────────────────────────────────────────────────────────
# Public entry point
# ────────────────────────────────────────────────────────────────────

def predict(ap_path: Optional[str | Path] = None,
             lat_path: Optional[str | Path] = None,
             *,
             config: Optional[PipelineConfig] = None) -> PredictionResult:
    """Run the full pipeline. Pass at least one of ap_path or lat_path."""
    cfg = config or default_config()
    logger = get_logger(cfg.log_level)

    if ap_path is None and lat_path is None:
        raise ValueError("Pass at least one of ap_path or lat_path")

    result = PredictionResult(
        ap_path=str(ap_path) if ap_path else None,
        lat_path=str(lat_path) if lat_path else None,
        config_snapshot={
            "exp1_ckpt": str(cfg.exp1_ckpt) if cfg.exp1_ckpt else None,
            "exp2_ckpt": str(cfg.exp2_ckpt) if cfg.exp2_ckpt else None,
            "exp3_ckpt": str(cfg.exp3_ckpt) if cfg.exp3_ckpt else None,
            "exp4_ckpt": str(cfg.exp4_ckpt) if cfg.exp4_ckpt else None,
            "sam2_ckpt": str(cfg.sam2_ckpt),
            "run_full_alignment": cfg.run_full_alignment,
            "width_cv_thresh": cfg.width_cv_thresh,
            "matched_ratio_thr": cfg.matched_ratio_thr,
        },
    )
    if ap_path:
        try: result.ap_sha1 = file_sha1(ap_path)
        except Exception: pass
    if lat_path:
        try: result.lat_sha1 = file_sha1(lat_path)
        except Exception: pass

    res = _get_resources(cfg)

    # ── AP cascade ────────────────────────────────────────────────
    if ap_path:
        try:
            ap_gray_raw = load_image_gray(str(ap_path))
            result.debug_artifacts["ap_raw"] = ap_gray_raw
            exp1, exp2 = _run_ap_cascade(str(ap_path), res, result.log,
                                          artifacts=result.debug_artifacts)
            result.exp1, result.exp2 = exp1, exp2
        except Exception as e:
            logger.exception("AP cascade failed")
            result.log.append(f"AP cascade failed: {e}")

        # ── Baumann angle (AP explainability, YOLO humerus mask) ───
        ap_gray_clean = result.debug_artifacts.get("ap_cleaned_gray")
        if ap_gray_clean is not None:
            try:
                yolo_m = res.yolo_baumann()
                if yolo_m is None:
                    result.log.append("Baumann angle skipped: YOLO checkpoint not found.")
                    result.baumann = BaumannResult(status="no_yolo_ckpt")
                else:
                    baumann_raw = compute_baumann_angle(ap_gray_clean, yolo_m)
                    result.debug_artifacts["baumann_raw"] = baumann_raw
                    shaft = baumann_raw.get("shaft")
                    physis = baumann_raw.get("physis")
                    result.baumann = BaumannResult(
                        status=baumann_raw["status"],
                        baumann_angle_deg=baumann_raw.get("baumann_angle_deg"),
                        in_normal_range=baumann_raw.get("in_normal_range"),
                        shaft_angle_deg=shaft["angle_deg"] if shaft else None,
                        physis_angle_deg=physis["angle_deg"] if physis else None,
                        physis_confidence=physis.get("confidence") if physis else None,
                    )
                    angle = baumann_raw.get("baumann_angle_deg")
                    in_range = baumann_raw.get("in_normal_range")
                    range_str = "normal" if in_range else "abnormal" if in_range is not None else "n/a"
                    result.log.append(
                        f"Baumann angle: {angle}° ({range_str}, status={baumann_raw['status']})"
                    )
            except Exception as e:
                result.log.append(f"Baumann angle failed: {e}")
                if result.baumann is None:
                    result.baumann = BaumannResult(status=f"error: {e}")

    # ── LAT cascade (CNN + geometric) ─────────────────────────────
    if lat_path:
        try:
            lat_gray_raw = load_image_gray(str(lat_path))
            result.debug_artifacts["lat_raw"] = lat_gray_raw
            if cfg.run_full_alignment:
                aligned = None
                if cfg.use_canonical_alignment:
                    try:
                        from ._canonical_align import align_canonical, is_available
                        if is_available():
                            aligned = align_canonical(
                                lat_gray_raw,
                                output_size=cfg.align_output_size,
                                zoom_out=cfg.align_zoom_out,
                                padding_ratio=cfg.align_padding_ratio,
                                remove_markers=cfg.canonical_remove_markers,
                                marker_brightness=cfg.canonical_marker_brightness,
                            )
                        else:
                            result.log.append("Canonical aligner not available; falling back to align_full")
                    except Exception as e:
                        result.log.append(f"Canonical alignment failed: {e} -- falling back to align_full")
                if aligned is None:
                    aligned = align_full(
                        lat_gray_raw,
                        downsample_size=cfg.align_downsample_size,
                        output_size=cfg.align_output_size,
                        zoom_out=cfg.align_zoom_out,
                        padding_ratio=cfg.align_padding_ratio,
                    )
                # inference_notebook flow: align -> marker-remove on aligned -> CLAHE -> SAM2.
                # The canonical aligner does NOT remove markers (matches the notebook's
                # _step1_to_6_debug_views), so we apply marker removal here on the aligned
                # 256x256 image, then feed that to BOTH the CNN and the geometric track.
                lat_aligned_raw = aligned["aligned"]
                lat_gray, _ = remove_lr_markers(
                    lat_aligned_raw,
                    cfg.marker_max_area_frac,
                    cfg.marker_min_brightness,
                )
                lat_for_cnn = lat_gray
            else:
                lat_for_cnn, _ = remove_lr_markers(lat_gray_raw,
                                                    cfg.marker_max_area_frac,
                                                    cfg.marker_min_brightness)
                lat_gray = lat_for_cnn

            lat_pil = numpy_to_pil_rgb(lat_for_cnn)
            result.debug_artifacts["lat_pil"] = lat_pil

            # Per user spec #5: when both AP+LAT supplied, run AP first, then LAT
            # only if AP did NOT predict Grade 3.
            run_lat_cascade = True
            if ap_path and result.exp2 and result.exp2.pred_idx == 1:
                run_lat_cascade = False
                result.log.append("Skipping LAT cascade: AP predicted Grade 3.")

            if run_lat_cascade:
                exp3, exp4 = _run_lat_cnn_cascade(lat_pil, res, result.log)
                result.exp3, result.exp4 = exp3, exp4

                # Geometric track always runs on LAT (even if no exp3 ckpt) —
                # it's the main explainability output.
                result.geometric = _run_geometric_track(
                    lat_gray, res, result.log, artifacts=result.debug_artifacts)
        except Exception as e:
            logger.exception("LAT cascade failed")
            result.log.append(f"LAT cascade failed: {e}")

    # ── Grad-CAM explainability ─────────────────────────────────────
    _generate_gradcams(result, res, cfg)

    # ── Compose final grade ───────────────────────────────────────
    _resolve_final_grade(result)

    # ── Audit log ─────────────────────────────────────────────────
    if cfg.log_predictions_to:
        try:
            append_jsonl(cfg.log_predictions_to, result.to_dict())
        except Exception as e:
            logger.warning(f"Audit log write failed: {e}")

    return result


def _resolve_final_grade(r: PredictionResult) -> None:
    """Combine CNN + geometric tracks into a single final_grade."""
    cnn_grade = None

    # Exp1: Normal vs Fractured — if Normal, short-circuit
    if r.exp1 and not r.exp1.skipped_reason and r.exp1.pred_idx == 0:
        cnn_grade = "Normal"
        r.cnn_final_grade = cnn_grade
        r.final_grade = cnn_grade
        r.grade_source = "cnn"
        return

    if r.exp2 and r.exp2.pred_idx == 1:
        cnn_grade = "Grade 3"
    elif r.exp3 and r.exp3.pred_idx == 0:
        cnn_grade = "Grade 1"
    elif r.exp3 and r.exp3.pred_idx == 1:
        if r.exp4 and r.exp4.pred_idx >= 0:
            cnn_grade = ["Grade 2a", "Grade 2b"][r.exp4.pred_idx]
        else:
            cnn_grade = (r.geometric.grade_2ab if r.geometric and r.geometric.grade_2ab
                         else "Grade 2 (subgrade unknown)")
    elif r.exp2 and r.exp2.pred_idx == 0 and r.exp3 is None:
        # AP confirmed Grade 1 or 2 but LAT not provided for sub-grading
        cnn_grade = "Grade 1 or 2"
    r.cnn_final_grade = cnn_grade

    geo_grade = r.geometric.final_grade if r.geometric else None

    if cnn_grade and geo_grade:
        if cnn_grade == geo_grade:
            r.final_grade = cnn_grade
            r.grade_source = "consensus"
        else:
            r.final_grade = cnn_grade
            r.grade_source = "cnn"
            r.discordant = True
    elif cnn_grade:
        r.final_grade = cnn_grade
        r.grade_source = "cnn"
    elif geo_grade:
        r.final_grade = geo_grade
        r.grade_source = "geometric"
