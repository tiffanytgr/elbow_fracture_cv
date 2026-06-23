"""Matplotlib rendering helpers. All functions return Figure objects; none call plt.show()."""
from __future__ import annotations
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def _to_rgb(gray: np.ndarray) -> np.ndarray:
    if gray is None:
        return None
    if gray.ndim == 3:
        return gray
    return np.stack([gray] * 3, axis=-1)


def render_overlays(aligned_lat: np.ndarray,
                    humerus_mask: Optional[np.ndarray] = None,
                    forearm_mask: Optional[np.ndarray] = None,
                    capitellum_mask: Optional[np.ndarray] = None,
                    ahl_a: Optional[float] = None,
                    ahl_b: Optional[float] = None,
                    cap_centroid: Optional[tuple] = None,
                    cap_radius: Optional[int] = None,
                    title: str = "SAM2 explainability") -> plt.Figure:
    """Aligned LAT image with humerus/forearm/capitellum masks + AHL line."""
    fig, ax = plt.subplots(1, 1, figsize=(1.8, 1.8))
    ax.imshow(_to_rgb(aligned_lat))
    H, W = aligned_lat.shape[:2]

    if humerus_mask is not None:
        ov = np.zeros((*humerus_mask.shape, 4))
        ov[humerus_mask > 0] = [0.2, 0.9, 0.2, 0.30]
        ax.imshow(ov)
    if forearm_mask is not None:
        ov = np.zeros((*forearm_mask.shape, 4))
        ov[forearm_mask > 0] = [0.9, 0.4, 0.1, 0.30]
        ax.imshow(ov)
    if capitellum_mask is not None:
        ov = np.zeros((*capitellum_mask.shape, 4))
        ov[capitellum_mask > 0] = [1.0, 0.9, 0.0, 0.55]
        ax.imshow(ov)

    if ahl_a is not None and ahl_b is not None:
        ys = np.array([0.0, float(H - 1)])
        xs = ahl_a * ys + ahl_b
        ax.plot(xs, ys, color="dodgerblue", linewidth=2.0, label="AHL")

    if cap_centroid is not None and cap_radius is not None:
        cx, cy = cap_centroid
        ax.add_patch(Circle((cx, cy), cap_radius, edgecolor="red",
                            facecolor="none", linewidth=2))
        ax.plot(cx, cy, "r+", markersize=12, markeredgewidth=2)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_title(title, fontsize=7)
    ax.axis("off")
    if ahl_a is not None:
        ax.legend(loc="upper right", fontsize=5)
    return fig


def plot_width_profile(width_profile: dict, title: str = "Cortical width profile") -> plt.Figure:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    widths = np.array(width_profile.get("widths", []), dtype=float)
    if len(widths) == 0:
        ax.text(0.5, 0.5, "No width data", ha="center", va="center")
        ax.axis("off")
        return fig

    height_px = width_profile.get("height_px", 30)
    xs = np.linspace(0, height_px, len(widths))
    ax.plot(xs, widths, "b-", linewidth=2.0, label="smoothed width")
    waist_idx = width_profile.get("waist_idx")
    if waist_idx is not None and waist_idx < len(xs):
        ax.axvline(xs[waist_idx], color="red", linestyle="--", linewidth=1.5,
                   label=f"waist ({xs[waist_idx]:.0f}px)")

    lo = width_profile.get("lower_mean", 0.0)
    hi = width_profile.get("upper_mean", 0.0)
    if lo > 0:
        ax.axhline(lo, color="orange", linestyle=":", linewidth=1.2,
                   label=f"lower mean ({lo:.1f})")
    if hi > 0:
        ax.axhline(hi, color="green", linestyle=":", linewidth=1.2,
                   label=f"upper mean ({hi:.1f})")
    mr = width_profile.get("match_ratio", 0.0)
    ax.set_xlabel("Distance above capitellum (px)", fontsize=11)
    ax.set_ylabel("Perpendicular width (px)", fontsize=11)
    ax.set_title(f"{title}  (match ratio = {mr:.2f})", fontsize=12)
    ax.tick_params(labelsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc="best")
    return fig


def render_width_on_xray(image: np.ndarray,
                          width_profile: dict,
                          humerus_mask: Optional[np.ndarray] = None,
                          cap_centroid: Optional[tuple] = None,
                          title: str = "Cortical width — 30px above capitellum") -> plt.Figure:
    """Overlay width measurement lines on the LAT X-ray (hot colormap: dark=thin, bright=thick)."""
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    ax.imshow(image, cmap="gray")
    H, W = image.shape[:2]

    if humerus_mask is not None:
        ax.contour(humerus_mask, levels=[0.5], colors="cyan", linewidths=0.6)

    sample_lines = width_profile.get("sample_lines", [])
    widths = width_profile.get("widths", [])
    n = len(sample_lines)

    valid_widths = [widths[i] for i in range(n)
                   if i < len(widths) and sample_lines[i] is not None and widths[i] > 0]
    w_min, w_max = (min(valid_widths), max(valid_widths)) if valid_widths else (0, 1)

    cmap = plt.cm.hot
    for i, line in enumerate(sample_lines):
        if line is None:
            continue
        x1, y1, x2, y2 = line
        w_val = widths[i] if i < len(widths) else 0
        norm_w = (w_val - w_min) / (w_max - w_min + 1e-6)
        ax.plot([x1, x2], [y1, y2], color=cmap(norm_w), linewidth=1.5, alpha=0.8,
                solid_capstyle="round")

    if cap_centroid is not None:
        ax.plot(cap_centroid[0], cap_centroid[1], "g+", markersize=10, markeredgewidth=2)

    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    sm = ScalarMappable(cmap=cmap, norm=Normalize(vmin=w_min, vmax=w_max))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Width (px)", fontsize=10)
    cbar.ax.tick_params(labelsize=9)

    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    height_px = width_profile.get("height_px", 30)
    ax.set_title(f"Width profile — {height_px}px above capitellum", fontsize=11)
    ax.axis("off")
    return fig


def prob_bar(labels: list, probs: list, pred_idx: int, title: str) -> plt.Figure:
    """Horizontal probability bars for a single classifier."""
    fig, ax = plt.subplots(1, 1, figsize=(1.6, 0.4 + 0.15 * len(labels)))
    colors = ["#2ecc71" if i == pred_idx else "#bdc3c7" for i in range(len(labels))]
    ax.barh(labels, probs, color=colors)
    ax.set_xlim(0, 1)
    for i, p in enumerate(probs):
        ax.text(p + 0.01, i, f"{p:.3f}", va="center", fontsize=5)
    ax.set_title(title, fontsize=6, fontweight="bold")
    ax.tick_params(labelsize=5)
    ax.invert_yaxis()
    return fig


def render_gradcam(gradcam_dict: dict,
                   pred_label: str = "",
                   conf: Optional[float] = None,
                   title: str = "Grad-CAM") -> plt.Figure:
    """Side-by-side original / overlay / heatmap from a gradcam dict."""
    orig = gradcam_dict["original"]
    overlay = gradcam_dict["overlay"]
    heatmap = gradcam_dict["heatmap"]

    fig, axes = plt.subplots(1, 3, figsize=(7, 2.5))
    axes[0].imshow(orig, cmap="gray")
    axes[0].set_title("Original", fontsize=7)
    axes[0].axis("off")

    overlay_title = f"Pred: {pred_label}"
    if conf is not None:
        overlay_title += f" ({conf:.1%})"
    axes[1].imshow(overlay)
    axes[1].set_title(overlay_title, fontsize=7)
    axes[1].axis("off")

    axes[2].imshow(heatmap, cmap="jet", vmin=0, vmax=1)
    axes[2].set_title("Heatmap", fontsize=7)
    axes[2].axis("off")

    fig.suptitle(title, fontsize=8, fontweight="bold")
    fig.tight_layout()
    return fig
