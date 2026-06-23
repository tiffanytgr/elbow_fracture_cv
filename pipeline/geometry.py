"""Geometric grading: AHL fitting, capitellum bisection, cortical width CV.

Ported from inference_notebook.ipynb cells 37, 38, 39.
"""
from __future__ import annotations
from typing import Optional

import numpy as np


def _anterior_edge_points(mask: np.ndarray, side: str = "right") -> tuple[np.ndarray, np.ndarray]:
    rows, cols = [], []
    for r in range(mask.shape[0]):
        xs = np.where(mask[r] > 0)[0]
        if len(xs) < 4:
            continue
        cols.append(int(xs[0] if side == "left" else xs[-1]))
        rows.append(r)
    return np.array(rows, dtype=float), np.array(cols, dtype=float)


def fit_ahl(humerus_mask: np.ndarray,
             anterior_edge: str = "right") -> Optional[tuple[float, float]]:
    """Fit Anterior Humeral Line as x = a*y + b on the anterior cortex.

    Returns (a, b) or None if too few edge pixels (<20).
    """
    rows, cols = _anterior_edge_points(humerus_mask, anterior_edge)
    if len(rows) < 20:
        return None
    n = len(rows)
    sr, sc = rows[int(n * 0.15):int(n * 0.70)], cols[int(n * 0.15):int(n * 0.70)]
    a, b = 0.0, 0.0
    for _ in range(3):
        if len(sr) < 5:
            break
        a, b = np.polyfit(sr, sc, 1)
        residuals = sc - (a * sr + b)
        std = residuals.std()
        if std < 1e-8:
            break
        keep = np.abs(residuals) < 2.5 * std
        sr, sc = sr[keep], sc[keep]
    return float(a), float(b)


def grade_ahl_bisection(ahl_a: float, ahl_b: float,
                         cap_centroid: tuple[int, int],
                         cap_radius: int,
                         capitellum_mask: np.ndarray,
                         anterior_edge: str = "right",
                         bisect_leeway: float = 0.30) -> dict:
    """Decide Grade 1 vs Grade 2 by AHL-vs-capitellum geometry.

    Two checks computed:
      - bisect_via_xband: x_AHL(cap_y) within (1-2*leeway) * cap_width centred
        on the capitellum (legacy method from notebook cell 37).
      - sam_via_dist: perpendicular distance from SAM centroid to AHL <= radius
        (cell 38; tighter, uses SAM mask shape).

    The 'grade_1v2' field uses the SAM-distance method, falling back to xband
    if the capitellum mask is empty.
    """
    cap_cx, cap_cy = cap_centroid
    ahl_x_at_cap = ahl_a * cap_cy + ahl_b

    # x-band check (centroid-relative)
    cap_left  = cap_cx - cap_radius
    cap_right = cap_cx + cap_radius
    cap_width = max(1, cap_right - cap_left)
    cap_frac  = (ahl_x_at_cap - cap_left) / cap_width
    bisects_xband = bisect_leeway <= cap_frac <= (1.0 - bisect_leeway)
    if bisects_xband:
        zone = "middle"
    elif (anterior_edge == "right" and cap_frac > (1.0 - bisect_leeway)) or \
         (anterior_edge == "left"  and cap_frac < bisect_leeway):
        zone = "anterior"
    else:
        zone = "posterior"

    # SAM-mask perpendicular distance check
    cap_ys, cap_xs = np.where(capitellum_mask > 0)
    if len(cap_xs) == 0:
        grade_1v2 = "Grade 1" if zone == "middle" else "Grade 2"
        return {
            "ahl_a": ahl_a, "ahl_b": ahl_b,
            "ahl_x_at_cap": float(ahl_x_at_cap),
            "cap_frac": float(cap_frac),
            "zone": zone,
            "method": "xband",
            "grade_1v2": grade_1v2,
        }

    y0, y1 = float(cap_cy) - 1.0, float(cap_cy) + 1.0
    x0, x1 = ahl_a * y0 + ahl_b, ahl_a * y1 + ahl_b
    dx, dy = x1 - x0, y1 - y0
    norm = float(np.hypot(dx, dy))
    dist_to_ahl = abs(dy * (cap_cx - x0) - dx * (cap_cy - y0)) / norm

    # Bisection symmetry diagnostic (computed before grade decision)
    signed = (dy * (cap_xs - x0) - dx * (cap_ys - y0)) / norm
    n_pos = int(np.sum(signed > 0))
    n_neg = int(np.sum(signed < 0))
    cap_area = len(cap_xs)
    pct_pos = 100.0 * n_pos / cap_area
    pct_neg = 100.0 * n_neg / cap_area
    bisection_quality = 2 * min(pct_pos, pct_neg)

    # Grade 1 requires AHL to pass through the capitellum (dist ≤ radius)
    # AND to bisect it fairly: neither side < 30% (i.e. split is 30/70 to 70/30).
    _bisects_cap = dist_to_ahl <= cap_radius
    _fair_split  = 30.0 <= pct_pos <= 70.0
    grade_1v2 = "Grade 1" if (_bisects_cap and _fair_split) else "Grade 2"

    return {
        "ahl_a": ahl_a, "ahl_b": ahl_b,
        "ahl_x_at_cap": float(ahl_x_at_cap),
        "cap_frac": float(cap_frac),
        "zone": zone,
        "method": "sam_distance",
        "grade_1v2": grade_1v2,
        "dist_to_ahl_px": float(dist_to_ahl),
        "cap_radius_px": int(cap_radius),
        "bisection_quality_pct": float(bisection_quality),
        "split_pct_pos": float(pct_pos),
        "split_pct_neg": float(pct_neg),
    }


def cortical_width_profile(humerus_mask: np.ndarray,
                            cap_centroid: tuple[int, int],
                            height_px: int = 30,
                            n_samples: int = 40,
                            smooth_win: int = 5,
                            shaft_pca_frac: float = 0.70) -> dict:
    """Cortical width profile measured from the capitellum upward.

    Samples *n_samples* perpendicular widths starting at the capitellum
    (index 0) and going *height_px* pixels proximally along the PCA shaft
    axis.  The match ratio compares the mean width of the lower half
    (near-capitellum) to the upper half (further up the shaft).
    """
    hum_ys, hum_xs = np.where(humerus_mask > 0)
    if len(hum_xs) < 50:
        return {"match_ratio": 0.0, "matched": False, "grade_2ab": None,
                "reason": "humerus_mask_too_small"}

    pts = np.stack([hum_xs, hum_ys], axis=1).astype(np.float32)

    # PCA on upper portion of shaft for a stable axis direction
    y_cutoff = np.percentile(hum_ys, shaft_pca_frac * 100)
    shaft_pts = pts[hum_ys <= y_cutoff]
    if len(shaft_pts) < 30:
        shaft_pts = pts

    centroid = shaft_pts.mean(axis=0)
    centered = shaft_pts - centroid
    cov = np.cov(centered.T)
    _, eigvecs = np.linalg.eigh(cov)
    axis_dir = eigvecs[:, -1]          # principal axis (along shaft)
    perp_dir = np.array([-axis_dir[1], axis_dir[0]], dtype=np.float32)

    # Orient axis so "positive" points AWAY from capitellum (proximal)
    cap_cx, cap_cy = cap_centroid
    cap_vec = np.array([cap_cx - centroid[0], cap_cy - centroid[1]],
                       dtype=np.float32)
    if cap_vec @ axis_dir > 0:
        axis_dir = -axis_dir
        perp_dir = -perp_dir

    # Project all mask pixels relative to capitellum
    cap_pt = np.array([cap_cx, cap_cy], dtype=np.float32)
    rel = pts - cap_pt
    proj = rel @ axis_dir  # positive = proximal (above cap)

    # Sample from capitellum (t=0) to height_px above (t=height_px)
    t_samples = np.linspace(0, height_px, n_samples)
    bin_hw = t_samples[1] / 2.0 if n_samples > 1 else 1.0

    widths = np.zeros(n_samples, dtype=np.float32)
    sample_lines = []  # [(x1,y1,x2,y2)] for visualisation
    for i, t in enumerate(t_samples):
        band = np.abs(proj - t) <= bin_hw
        if band.sum() < 3:
            sample_lines.append(None)
            continue
        perp_vals = rel[band] @ perp_dir
        w_min, w_max = float(perp_vals.min()), float(perp_vals.max())
        widths[i] = w_max - w_min
        mid = cap_pt + t * axis_dir
        p1 = mid + w_min * perp_dir
        p2 = mid + w_max * perp_dir
        sample_lines.append((float(p1[0]), float(p1[1]),
                             float(p2[0]), float(p2[1])))

    # Smooth
    kernel = np.ones(smooth_win, dtype=np.float32) / smooth_win
    widths_smooth = np.convolve(widths, kernel, mode="same")

    # Waist (narrowest point, excluding edges)
    edge = max(2, n_samples // 10)
    search = widths_smooth.copy()
    search[:edge] = np.inf
    search[-edge:] = np.inf
    waist_idx = int(np.argmin(search))
    waist_width = float(widths_smooth[waist_idx])

    # Match ratio: lower half (near capitellum) vs upper half (proximal)
    mid = n_samples // 2
    lower_half = widths_smooth[:mid]
    upper_half = widths_smooth[mid:]
    lower_mean = float(lower_half[lower_half > 0].mean()) if (lower_half > 0).any() else 0.0
    upper_mean = float(upper_half[upper_half > 0].mean()) if (upper_half > 0).any() else 0.0
    if max(lower_mean, upper_mean) > 0:
        match_ratio = min(lower_mean, upper_mean) / max(lower_mean, upper_mean)
    else:
        match_ratio = 0.0

    return {
        "widths": widths_smooth.tolist(),
        "waist_idx": waist_idx,
        "waist_width_px": waist_width,
        "lower_mean": lower_mean,
        "upper_mean": upper_mean,
        "match_ratio": float(match_ratio),
        "sample_lines": sample_lines,
        "height_px": height_px,
    }


def grade_2a_vs_2b(width_profile: dict, matched_ratio_thr: float = 0.80) -> str:
    if width_profile.get("match_ratio", 0.0) >= matched_ratio_thr:
        return "Grade 2a"
    return "Grade 2b"


def shaft_width_cv(hum_mask: np.ndarray, row_indices: np.ndarray,
                    min_bone_px: int = 10) -> float:
    """Legacy CV-based 2a/2b discriminator (cell 37). Kept for parity tests."""
    widths = []
    for r in row_indices:
        xs = np.where(hum_mask[r] > 0)[0]
        if len(xs) < min_bone_px:
            continue
        widths.append(xs[-1] - xs[0])
    if len(widths) < 5:
        return 0.0
    widths = np.array(widths, dtype=float)
    return float(widths.std() / (widths.mean() + 1e-8))
