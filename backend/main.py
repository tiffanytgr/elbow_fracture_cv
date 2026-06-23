"""FastAPI backend for the KKH paediatric elbow fracture grading pipeline.

Run from KKH_Elbow/:
    uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import base64
import io
import sys
import tempfile
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend before any plt import
import matplotlib.pyplot as plt

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from elbow_grader import ElbowGrader, GraderConfig  # noqa: E402

app = FastAPI(title="Elbow Grader API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Singleton grader (loaded once, weights stay warm) ──────────────────
_grader: Optional[ElbowGrader] = None


def _get_grader() -> ElbowGrader:
    global _grader
    if _grader is None:
        _grader = ElbowGrader()
    return _grader


# ── Helpers ────────────────────────────────────────────────────────────

def _fig_to_b64(fig: plt.Figure, dpi: int = 120) -> str:
    """Render a matplotlib Figure to a base64-encoded PNG string."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi)
    buf.seek(0)
    plt.close(fig)
    return base64.b64encode(buf.read()).decode()


def _exp_to_dict(exp) -> Optional[dict]:
    if exp is None:
        return None
    return {
        "labels": exp.labels,
        "probs": exp.probs,
        "pred_idx": exp.pred_idx,
        "skipped_reason": exp.skipped_reason,
        "ood_score": exp.ood_score,
        "ood_flagged": exp.ood_flagged,
    }


# ── Routes ─────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    """Return server liveness + checkpoint status for all model components."""
    grader = _get_grader()
    return {
        "status": "ok",
        "model_status": grader.model_status,
    }


@app.post("/predict")
async def predict(
    ap_file: Optional[UploadFile] = File(None),
    lat_file: Optional[UploadFile] = File(None),
    confidence_threshold: float = Form(0.70),
    run_full_lat_alignment: bool = Form(False),
    run_sam2: bool = Form(True),
):
    """Run the grading pipeline and return grades + base64 plots.

    Form fields
    -----------
    ap_file  : AP view image (PNG/JPEG)
    lat_file : LAT view image (PNG/JPEG) — optional for initial screening
    confidence_threshold  : 0.50–0.95, predictions below this are withheld
    run_full_lat_alignment: enable full 6-step LAT alignment
    run_sam2              : enable SAM2 bone segmentation
    """
    if ap_file is None and lat_file is None:
        raise HTTPException(status_code=400,
                            detail="Provide at least one of ap_file or lat_file.")

    grader = _get_grader()
    grader.configure(GraderConfig(
        confidence_threshold=confidence_threshold,
        run_full_lat_alignment=run_full_lat_alignment,
        run_sam2=run_sam2,
    ))

    temp_files: list[str] = []
    try:
        ap_path: Optional[str] = None
        lat_path: Optional[str] = None

        if ap_file is not None:
            suffix = Path(ap_file.filename or "img.png").suffix or ".png"
            f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            f.write(await ap_file.read())
            f.close()
            temp_files.append(f.name)
            ap_path = f.name

        if lat_file is not None:
            suffix = Path(lat_file.filename or "img.png").suffix or ".png"
            f = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            f.write(await lat_file.read())
            f.close()
            temp_files.append(f.name)
            lat_path = f.name

        result = grader.predict(ap_image=ap_path, lat_image=lat_path)

    finally:
        for t in temp_files:
            try:
                Path(t).unlink(missing_ok=True)
            except Exception:
                pass

    r = result._r

    # ── Baumann structured data ──────────────────────────────────────
    baumann_data: Optional[dict] = None
    if r.baumann:
        b = r.baumann
        baumann_data = {
            "baumann_angle_deg": b.baumann_angle_deg,
            "in_normal_range": b.in_normal_range,
            "status": b.status,
            "shaft_angle_deg": b.shaft_angle_deg,
            "physis_angle_deg": getattr(b, "physis_angle_deg", None),
            "physis_confidence": getattr(b, "physis_confidence", None),
        }

    # ── Geometric structured data ────────────────────────────────────
    geometric_data: Optional[dict] = None
    if r.geometric:
        g = r.geometric
        ahl = g.ahl_diagnostic or {}
        wp = g.width_profile
        geometric_data = {
            "grade_1v2": g.grade_1v2,
            "grade_2ab": g.grade_2ab,
            "final_grade": g.final_grade,
            "bone_sam_score": g.bone_sam_score,
            "skipped_reason": g.skipped_reason,
            "ahl_diagnostic": {k: ahl[k] for k in ahl if k in (
                "zone", "method", "ahl_x_at_cap", "dist_to_ahl_px",
                "cap_radius_px", "split_pct_pos", "split_pct_neg",
                "bisection_quality_pct",
            )},
            "width_profile": {
                "height_px": wp.get("height_px"),
                "match_ratio": wp.get("match_ratio"),
            } if wp else None,
        }

    # ── Matplotlib figures → base64 PNG ──────────────────────────────
    plots: dict[str, str] = {}
    arts = r.debug_artifacts

    for k in (1, 2, 3, 4):
        exp_obj = getattr(r, f"exp{k}", None)
        if (f"gradcam_exp{k}" in arts
                and exp_obj is not None
                and exp_obj.pred_idx >= 0):
            try:
                plots[f"gradcam_{k}"] = _fig_to_b64(result.plot_gradcam(experiment=k))
            except Exception:
                pass

    if r.geometric and not r.geometric.skipped_reason:
        try:
            plots["geometric"] = _fig_to_b64(result.plot_geometric())
        except Exception:
            pass

    if r.geometric and r.geometric.width_profile:
        try:
            plots["cortical_width"] = _fig_to_b64(result.plot_cortical_width())
        except Exception:
            pass

    if r.baumann:
        try:
            plots["baumann"] = _fig_to_b64(result.plot_baumann())
        except Exception:
            pass

    return {
        "final_grade": result.final_grade,
        "cnn_grade": result.cnn_grade,
        "geometric_grade": result.geometric_grade,
        "grade_source": result.grade_source,
        "discordant": result.discordant,
        "confidence": result.confidence,
        "is_ood": result.is_ood,
        "baumann_angle": result.baumann_angle,
        "baumann_normal": result.baumann_normal,
        "log": result.log,
        "ap_sha1": r.ap_sha1,
        "lat_sha1": r.lat_sha1,
        "config_snapshot": r.config_snapshot,
        "result_json": result.to_dict(),
        "experiments": {
            "exp1": _exp_to_dict(r.exp1),
            "exp2": _exp_to_dict(r.exp2),
            "exp3": _exp_to_dict(r.exp3),
            "exp4": _exp_to_dict(r.exp4),
        },
        "baumann": baumann_data,
        "geometric": geometric_data,
        "plots": plots,
        "model_status": grader.model_status,
    }
