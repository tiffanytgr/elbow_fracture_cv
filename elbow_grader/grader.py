"""ElbowGrader: main entry point for the grading pipeline."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from typing import Optional, Union

import numpy as np

# Ensure the project root (KKH_Elbow/) is on sys.path so `pipeline` is importable
# both when running directly and after `pip install -e .`.
_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from pipeline.config import PipelineConfig
from pipeline.pipeline import predict as _pipeline_predict

from .config import GraderConfig
from .results import GraderResult

# Sentinel used when run_sam2=False — the pipeline gracefully skips SAM2
# when the checkpoint path does not exist.
_SAM2_DISABLED = Path("__sam2_disabled__")


def _ckpt_label(path: Optional[Path]) -> str:
    if path is None:
        return "STUBBED"
    return "OK" if Path(path).exists() else "MISSING"


class ElbowGrader:
    """End-to-end Gartland grading for paediatric elbow X-rays.

    Typical usage
    -------------
    >>> grader = ElbowGrader.from_checkpoints("experiments/checkpoints")
    >>> result = grader.predict(ap_image="A102_AP.png", lat_image="A102_LAT.png")
    >>> print(result.final_grade)
    'Grade 2a'
    >>> result.plot_gradcam().savefig("gradcam.png")

    The grader caches loaded model weights internally, so repeated calls to
    predict() do not reload weights from disk.
    """

    def __init__(
        self,
        config: Optional[GraderConfig] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None,
        sam2_ckpt: Optional[Union[str, Path]] = None,
    ):
        """
        Parameters
        ----------
        config : GraderConfig, optional
            User-facing settings. Defaults to GraderConfig() if not provided.
        checkpoint_dir : path, optional
            Root of the experiment checkpoints directory. Expects:
                checkpoint_dir/exp{1-4}/best_model_for_analysis.pth
                checkpoint_dir/cap_regressor/best_model.pth
                checkpoint_dir/drue_outputs_exp{1-4}/drue_decoders.pth  (optional)
            If None, uses the default paths baked into pipeline/config.py.
        sam2_ckpt : path, optional
            Explicit path to sam2_hiera_large.pt. If None, uses the default
            from pipeline/config.py (experiments/checkpoints/sam2/sam2_hiera_large.pt).
        """
        self._grader_config = config or GraderConfig()
        self._pipeline_cfg = self._build_pipeline_config(checkpoint_dir, sam2_ckpt)

    @classmethod
    def from_checkpoints(
        cls,
        checkpoint_dir: Union[str, Path],
        config: Optional[GraderConfig] = None,
        sam2_ckpt: Optional[Union[str, Path]] = None,
    ) -> "ElbowGrader":
        """Create a grader with an explicit checkpoint directory.

        Parameters
        ----------
        checkpoint_dir : path
            Directory containing exp1/, exp2/, exp3/, exp4/, and cap_regressor/
            subdirectories with the model weights.
        config : GraderConfig, optional
        sam2_ckpt : path, optional
            Path to sam2_hiera_large.pt. If None, uses the pipeline default.
        """
        return cls(config=config, checkpoint_dir=checkpoint_dir, sam2_ckpt=sam2_ckpt)

    # ── Config management ───────────────────────────────────────────────

    @property
    def config(self) -> GraderConfig:
        return self._grader_config

    def configure(self, config: GraderConfig) -> None:
        """Update threshold/toggle settings without reloading model weights.

        This mutates the internal PipelineConfig in-place so the model cache
        (keyed by config object identity) is preserved.
        """
        self._grader_config = config
        self._pipeline_cfg.confidence_threshold = config.confidence_threshold
        self._pipeline_cfg.run_full_alignment = config.run_full_lat_alignment
        if not config.run_sam2:
            self._pipeline_cfg.sam2_ckpt = _SAM2_DISABLED
        # Note: device and checkpoint paths are fixed at construction time.

    # ── Status ──────────────────────────────────────────────────────────

    @property
    def model_status(self) -> dict[str, str]:
        """Checkpoint status for each model component: 'OK', 'MISSING', or 'STUBBED'."""
        cfg = self._pipeline_cfg
        return {
            "Exp 1 (AP Normal/Fractured)": _ckpt_label(cfg.exp1_ckpt),
            "Exp 2 (AP Grade 3)":          _ckpt_label(cfg.exp2_ckpt),
            "Exp 3 (LAT Grade 1/2)":       _ckpt_label(cfg.exp3_ckpt),
            "Exp 4 (LAT Grade 2a/2b)":     _ckpt_label(cfg.exp4_ckpt),
            "Capitellum regressor":         _ckpt_label(cfg.cap_regressor_ckpt),
            "SAM2":                         _ckpt_label(cfg.sam2_ckpt),
            "YOLO (Baumann AP)":            _ckpt_label(getattr(cfg, "yolo_baumann_ckpt", None)),
        }

    # ── Inference ───────────────────────────────────────────────────────

    def predict(
        self,
        ap_image: Optional[Union[str, Path, "np.ndarray"]] = None,
        lat_image: Optional[Union[str, Path, "np.ndarray"]] = None,
    ) -> GraderResult:
        """Run the grading pipeline on AP and/or LAT X-ray image(s).

        Parameters
        ----------
        ap_image : path, numpy array, or PIL Image, optional
            Anterior-posterior view. At least one of ap_image / lat_image required.
        lat_image : path, numpy array, or PIL Image, optional
            Lateral view.

        Returns
        -------
        GraderResult
            Structured result with grade, confidence, explainability data,
            and built-in plot methods.
        """
        if ap_image is None and lat_image is None:
            raise ValueError("Provide at least one of ap_image or lat_image.")

        temp_files: list[str] = []
        try:
            ap_path = _to_path(ap_image, temp_files)
            lat_path = _to_path(lat_image, temp_files)
            raw = _pipeline_predict(ap_path=ap_path, lat_path=lat_path,
                                    config=self._pipeline_cfg)
        finally:
            for t in temp_files:
                try:
                    Path(t).unlink(missing_ok=True)
                except Exception:
                    pass

        return GraderResult(raw)

    # ── Internals ───────────────────────────────────────────────────────

    def _build_pipeline_config(
        self,
        checkpoint_dir: Optional[Union[str, Path]],
        sam2_ckpt: Optional[Union[str, Path]],
    ) -> PipelineConfig:
        gc = self._grader_config
        kwargs: dict = dict(
            confidence_threshold=gc.confidence_threshold,
            run_full_alignment=gc.run_full_lat_alignment,
            device=gc.device,
        )

        if checkpoint_dir is not None:
            ckpt = Path(checkpoint_dir)
            kwargs.update(
                exp1_ckpt=ckpt / "exp1" / "best_model_for_analysis.pth",
                exp2_ckpt=ckpt / "exp2" / "best_model_for_analysis.pth",
                exp3_ckpt=ckpt / "exp3" / "best_model_for_analysis.pth",
                exp4_ckpt=ckpt / "exp4" / "best_model_for_analysis.pth",
                cap_regressor_ckpt=ckpt / "cap_regressor" / "best_model.pth",
                drue_exp1_ckpt=ckpt / "drue_outputs_exp1" / "drue_decoders.pth",
                drue_exp2_ckpt=ckpt / "drue_outputs_exp2" / "drue_decoders.pth",
                drue_exp3_ckpt=ckpt / "drue_outputs_exp3" / "drue_decoders.pth",
                drue_exp4_ckpt=ckpt / "drue_outputs_exp4" / "drue_decoders.pth",
            )

        if sam2_ckpt is not None:
            kwargs["sam2_ckpt"] = Path(sam2_ckpt)

        if not gc.run_sam2:
            kwargs["sam2_ckpt"] = _SAM2_DISABLED

        return PipelineConfig(**kwargs)


# ── Helpers ─────────────────────────────────────────────────────────────

def _to_path(img, temp_files: list) -> Optional[str]:
    """Normalise image input to a file path, saving arrays to a temp file if needed."""
    if img is None:
        return None
    if isinstance(img, (str, Path)):
        return str(img)

    # numpy array → save as PNG
    if isinstance(img, np.ndarray):
        from PIL import Image as _PILImage
        pil = _PILImage.fromarray(img.astype(np.uint8))
        f = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        pil.save(f.name)
        f.close()
        temp_files.append(f.name)
        return f.name

    # PIL Image
    try:
        from PIL import Image as _PILImage
        if isinstance(img, _PILImage.Image):
            f = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            img.save(f.name)
            f.close()
            temp_files.append(f.name)
            return f.name
    except ImportError:
        pass

    raise TypeError(f"Unsupported image type: {type(img)}. "
                    "Pass a file path (str/Path), numpy array, or PIL Image.")
