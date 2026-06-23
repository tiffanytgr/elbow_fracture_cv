"""GraderResult: structured output with built-in visualisation methods."""
from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import matplotlib.pyplot as plt
    from pipeline.pipeline import PredictionResult


class GraderResult:
    """Wraps the internal PredictionResult and exposes a clean, stable API.

    Grade properties
    ----------------
    final_grade, cnn_grade, geometric_grade, grade_source, discordant, confidence, is_ood

    Visualisation
    -------------
    plot_summary()       — probability bars for all active experiments
    plot_gradcam()       — Grad-CAM heatmaps (all or a specific experiment)
    plot_geometric()     — SAM2 overlay with AHL and capitellum annotation
    plot_cortical_width()— cortical width profile + overlay on X-ray

    Export
    ------
    to_dict(), to_json(path)
    """

    def __init__(self, result: "PredictionResult"):
        self._r = result

    # ── Grade properties ─────────────────────────────────────────────

    @property
    def final_grade(self) -> Optional[str]:
        """The consensus final grade, e.g. 'Grade 2a', 'Normal'."""
        return self._r.final_grade

    @property
    def cnn_grade(self) -> Optional[str]:
        """Grade from the CNN cascade alone."""
        return self._r.cnn_final_grade

    @property
    def geometric_grade(self) -> Optional[str]:
        """Grade from the geometric (SAM2 + AHL) track."""
        if self._r.geometric:
            return self._r.geometric.final_grade
        return None

    @property
    def grade_source(self) -> Optional[str]:
        """How the final grade was determined: 'cnn', 'geometric', or 'consensus'."""
        return self._r.grade_source

    @property
    def discordant(self) -> bool:
        """True when CNN and geometric tracks disagree."""
        return self._r.discordant

    @property
    def confidence(self) -> Optional[float]:
        """Confidence of the decisive CNN classifier (highest-level experiment that fired)."""
        for exp in (self._r.exp4, self._r.exp3, self._r.exp2, self._r.exp1):
            if exp and exp.pred_idx >= 0 and exp.probs:
                return exp.probs[exp.pred_idx]
        return None

    @property
    def is_ood(self) -> bool:
        """True if any experiment flagged the image as out-of-distribution."""
        for exp in (self._r.exp1, self._r.exp2, self._r.exp3, self._r.exp4):
            if exp and exp.ood_flagged:
                return True
        return False

    @property
    def ahl_diagnostic(self) -> Optional[dict]:
        """AHL bisection diagnostics from the geometric track (or None)."""
        if self._r.geometric:
            return self._r.geometric.ahl_diagnostic
        return None

    @property
    def baumann_angle(self) -> Optional[float]:
        """Baumann angle in degrees (AP view, normal range 60–84°)."""
        if self._r.baumann:
            return self._r.baumann.baumann_angle_deg
        return None

    @property
    def baumann_normal(self) -> Optional[bool]:
        """True if Baumann angle is within the normal range (60–84°)."""
        if self._r.baumann:
            return self._r.baumann.in_normal_range
        return None

    @property
    def log(self) -> list[str]:
        """Pipeline execution log."""
        return self._r.log

    # ── Visualisations ───────────────────────────────────────────────

    def plot_summary(self) -> "plt.Figure":
        """Probability bar charts for every active experiment.

        Returns a single Figure with one panel per experiment that produced
        a prediction.
        """
        import matplotlib.pyplot as plt

        exps = [
            ("Exp 1 — Normal vs Fractured",   self._r.exp1),
            ("Exp 2 — Grade 3 vs Grade 1/2",  self._r.exp2),
            ("Exp 3 — Grade 1 vs Grade 2",    self._r.exp3),
            ("Exp 4 — Grade 2a vs Grade 2b",  self._r.exp4),
        ]
        active = [(name, e) for name, e in exps if e is not None and e.probs]

        if not active:
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.text(0.5, 0.5, "No predictions available", ha="center", va="center")
            ax.axis("off")
            return fig

        fig, axes = plt.subplots(1, len(active), figsize=(3 * len(active), 2))
        if len(active) == 1:
            axes = [axes]

        for ax, (name, exp) in zip(axes, active):
            _draw_prob_bar(ax, exp.labels, exp.probs, exp.pred_idx, name)

        grade = self._r.final_grade or "UNKNOWN"
        src = self._r.grade_source or "n/a"
        fig.suptitle(f"Final grade: {grade}  (source: {src})", fontsize=10, fontweight="bold")
        fig.tight_layout()
        return fig

    def plot_gradcam(self, experiment: Optional[int] = None) -> "plt.Figure":
        """Grad-CAM heatmaps.

        Parameters
        ----------
        experiment : 1 | 2 | 3 | 4, optional
            Show only this experiment. If None (default), shows all available
            experiments as stacked rows of (Original | Overlay | Heatmap).
        """
        import matplotlib.pyplot as plt

        arts = self._r.debug_artifacts
        exp_map = {
            1: (self._r.exp1, "Normal vs Fractured"),
            2: (self._r.exp2, "Grade 3 vs Grade 1/2"),
            3: (self._r.exp3, "Grade 1 vs Grade 2"),
            4: (self._r.exp4, "Grade 2a vs Grade 2b"),
        }

        candidates = [experiment] if experiment else [1, 2, 3, 4]
        rows = [
            (k, exp_map[k][0], exp_map[k][1])
            for k in candidates
            if k in exp_map
            and f"gradcam_exp{k}" in arts
            and exp_map[k][0] is not None
            and exp_map[k][0].pred_idx >= 0
        ]

        if not rows:
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.text(0.5, 0.5, "No Grad-CAM data available\n(run prediction first)",
                    ha="center", va="center")
            ax.axis("off")
            return fig

        n = len(rows)
        fig, axes = plt.subplots(n, 3, figsize=(7, 2.5 * n), squeeze=False)

        for i, (k, exp, label) in enumerate(rows):
            gc = arts[f"gradcam_exp{k}"]
            pred_label = exp.labels[exp.pred_idx]
            conf = exp.probs[exp.pred_idx] if exp.probs else None

            axes[i, 0].imshow(gc["original"], cmap="gray")
            axes[i, 0].set_title("Original", fontsize=7)
            axes[i, 0].axis("off")

            overlay_title = f"Pred: {pred_label}"
            if conf is not None:
                overlay_title += f" ({conf:.1%})"
            axes[i, 1].imshow(gc["overlay"])
            axes[i, 1].set_title(overlay_title, fontsize=7)
            axes[i, 1].axis("off")

            axes[i, 2].imshow(gc["heatmap"], cmap="jet", vmin=0, vmax=1)
            axes[i, 2].set_title("Heatmap", fontsize=7)
            axes[i, 2].axis("off")

            # Row label on the left margin
            fig.text(0.01, 1 - (i + 0.5) / n, label,
                     va="center", fontsize=7, style="italic",
                     rotation=90, transform=fig.transFigure)

        fig.suptitle(f"Grad-CAM  —  {self._r.final_grade or 'UNKNOWN'}",
                     fontsize=9, fontweight="bold")
        fig.tight_layout(rect=[0.04, 0, 1, 0.97])
        return fig

    def plot_geometric(self) -> "plt.Figure":
        """SAM2 segmentation overlays with AHL line and capitellum annotation."""
        import matplotlib.pyplot as plt
        from elbow_grader._viz import render_overlays

        g = self._r.geometric
        arts = self._r.debug_artifacts

        if not g or g.skipped_reason:
            reason = g.skipped_reason if g else "no LAT input provided"
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.text(0.5, 0.5, f"Geometric track skipped:\n{reason}",
                    ha="center", va="center", fontsize=9)
            ax.axis("off")
            return fig

        lat_img = arts.get("lat_clahe", arts.get("lat_aligned"))
        if lat_img is None:
            fig, ax = plt.subplots(figsize=(4, 3))
            ax.text(0.5, 0.5, "LAT image not available in artifacts",
                    ha="center", va="center")
            ax.axis("off")
            return fig

        ahl = g.ahl_diagnostic or {}
        cap_centroid = g.capitellum_xy_sam or g.capitellum_xy_regressor
        cap_radius = arts.get("capitellum_radius")

        return render_overlays(
            lat_img,
            humerus_mask=arts.get("humerus_mask"),
            forearm_mask=arts.get("forearm_mask"),
            capitellum_mask=arts.get("capitellum_mask"),
            ahl_a=ahl.get("ahl_a"),
            ahl_b=ahl.get("ahl_b"),
            cap_centroid=cap_centroid,
            cap_radius=cap_radius,
            title=f"Geometric grade: {g.final_grade or '?'}",
        )

    def plot_cortical_width(self) -> "plt.Figure":
        """Two-panel figure: width-on-X-ray overlay (left) + profile line chart (right).

        Only available when the geometric track reached the Grade 2a/2b decision.
        """
        import matplotlib.pyplot as plt
        import numpy as np
        from PIL import Image as _PILImage
        from elbow_grader._viz import plot_width_profile, render_width_on_xray

        g = self._r.geometric
        arts = self._r.debug_artifacts

        if not g or not g.width_profile:
            fig, ax = plt.subplots(figsize=(4, 2))
            ax.text(0.5, 0.5,
                    "No cortical width data\n(LAT geometric track did not reach Grade 2)",
                    ha="center", va="center")
            ax.axis("off")
            return fig

        lat_img = arts.get("lat_clahe", arts.get("lat_aligned"))
        cap_xy = g.capitellum_xy_sam or g.capitellum_xy_regressor

        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7))

        if lat_img is not None:
            fig_xray = render_width_on_xray(
                lat_img, g.width_profile,
                humerus_mask=arts.get("humerus_mask"),
                cap_centroid=cap_xy,
            )
            ax_left.imshow(_fig_to_array(fig_xray))
            ax_left.axis("off")
            ax_left.set_title("Width measurements on X-ray", fontsize=8)
            plt.close(fig_xray)
        else:
            ax_left.text(0.5, 0.5, "Image not available", ha="center", va="center")
            ax_left.axis("off")

        fig_profile = plot_width_profile(g.width_profile)
        ax_right.imshow(_fig_to_array(fig_profile))
        ax_right.axis("off")
        ax_right.set_title("Cortical width profile", fontsize=8)
        plt.close(fig_profile)

        mr = g.width_profile.get("match_ratio", 0)
        grade = g.grade_2ab or "n/a"
        fig.suptitle(f"Grade 2a vs 2b: {grade}  (match ratio = {mr:.2f})",
                     fontsize=10, fontweight="bold")
        fig.tight_layout()
        return fig

    def plot_baumann(self) -> "plt.Figure":
        """6-panel step-by-step figure for the AP Baumann angle pipeline.

        Panels: original AP | bone mask | humerus/forearm seg |
                shaft PCA | physis line | Baumann overlay with angle.

        Only available when an AP image was provided.
        """
        import matplotlib.pyplot as plt
        import numpy as np

        arts = self._r.debug_artifacts
        baumann_raw = arts.get("baumann_raw")
        # Prefer the upright-rotated gray stored inside baumann_raw (YOLO path);
        # fall back to the original cleaned AP image if not present.
        # NOTE: cannot use `or` here — numpy arrays raise ValueError on boolean evaluation.
        _rot_gray = baumann_raw.get("gray") if baumann_raw else None
        gray = _rot_gray if _rot_gray is not None else arts.get("ap_cleaned_gray", arts.get("ap_raw"))

        if baumann_raw is None or gray is None:
            fig, ax = plt.subplots(figsize=(5, 2))
            ax.text(0.5, 0.5,
                    "Baumann angle not available\n(requires AP image)",
                    ha="center", va="center", fontsize=9)
            ax.axis("off")
            return fig

        from pipeline.baumann import draw_step_panels
        arr = draw_step_panels(gray, baumann_raw)

        fig, ax = plt.subplots(figsize=(15, 9))
        ax.imshow(arr)
        ax.axis("off")
        fig.tight_layout(pad=0)
        return fig

    # ── Export ───────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise to a plain dict (audit-safe: no numpy arrays)."""
        return self._r.to_dict()

    def to_json(self, path: Union[str, Path], indent: int = 2) -> None:
        """Write the audit JSON to *path*."""
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, default=str, indent=indent)

    def __repr__(self) -> str:
        conf = f"{self.confidence:.3f}" if self.confidence is not None else "n/a"
        return (f"GraderResult(final_grade={self.final_grade!r}, "
                f"confidence={conf}, discordant={self.discordant})")


# ── Helpers ──────────────────────────────────────────────────────────────

def _fig_to_array(fig: "plt.Figure") -> "np.ndarray":
    """Render a matplotlib figure to an RGBA numpy array via an in-memory buffer."""
    import numpy as np
    from PIL import Image as _PILImage

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
    buf.seek(0)
    return np.array(_PILImage.open(buf))


def _draw_prob_bar(ax, labels: list, probs: list, pred_idx: int, title: str) -> None:
    """Draw probability bars onto an existing Axes."""
    colors = ["#2ecc71" if i == pred_idx else "#bdc3c7" for i in range(len(labels))]
    ax.barh(labels, probs, color=colors)
    ax.set_xlim(0, 1)
    for i, p in enumerate(probs):
        ax.text(p + 0.01, i, f"{p:.3f}", va="center", fontsize=5)
    ax.set_title(title, fontsize=6, fontweight="bold")
    ax.tick_params(labelsize=5)
    ax.invert_yaxis()


# Allow `from elbow_grader.results import Union` for type hints in grader.py
from typing import Union  # noqa: E402, F401 (re-export for type checking)
