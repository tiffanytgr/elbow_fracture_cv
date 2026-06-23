"""SAM2 wrappers: bone segmentation + capitellum segmentation.

Ported from inference_notebook.ipynb cells 26, 28, 30, 35. Two entry
points:

    SAM2Segmenter(config).load() -> ready predictor
    seg.segment_bone(gray_clahe) -> dict(bone_mask, humerus_mask, forearm_mask, ...)
    seg.segment_capitellum(gray_clahe, humerus_mask, regressor_xy) -> dict(...)
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from scipy.ndimage import distance_transform_edt, binary_erosion, label as cc_label
from skimage.filters import threshold_otsu

from .preprocessing import _safe_lcc_mask


# ────────────────────────────────────────────────────────────────────
# Probe selection helpers
# ────────────────────────────────────────────────────────────────────

def _pick_top_k_points(image: np.ndarray, k: int, min_dist: int,
                        brightest: bool = True) -> list[tuple[int, int]]:
    patch = cv2.GaussianBlur(image.copy(), (5, 5), 0)
    points = []
    for _ in range(k):
        if brightest:
            _, val, _, loc = cv2.minMaxLoc(patch)
        else:
            val, _, loc, _ = cv2.minMaxLoc(patch)
        if val == 0 and brightest:
            break
        points.append(loc)
        cv2.circle(patch, loc, min_dist, 0 if brightest else 255, -1)
    return points


def _pick_stratified_probes(image: np.ndarray, mask: np.ndarray,
                             k_per_zone: int, min_dist: int,
                             n_zones: int = 3) -> list[tuple[int, int]]:
    bone_rows = np.where(np.any(mask > 0, axis=1))[0]
    if len(bone_rows) == 0:
        return _pick_top_k_points(image, k_per_zone * n_zones, min_dist)
    zone_bounds = np.linspace(bone_rows[0], bone_rows[-1], n_zones + 1).astype(int)
    out = []
    for z in range(n_zones):
        r_lo, r_hi = zone_bounds[z], zone_bounds[z + 1]
        zi = image.copy()
        zi[:r_lo, :] = 0
        zi[r_hi:, :] = 0
        zi[mask == 0] = 0
        out.extend(_pick_top_k_points(zi, k_per_zone, min_dist, brightest=True))
    return out


def _probes_to_box(pos_probes, pad, W, H, x_percentile=40):
    px = sorted([int(p[0]) for p in pos_probes])
    py = [int(p[1]) for p in pos_probes]
    x_max_cap = int(np.percentile(px, x_percentile))
    px_cap = [x for x in px if x <= x_max_cap]
    return np.array([
        max(0, min(px_cap) - pad),
        max(0, min(py) - pad),
        min(W, max(px_cap) + pad),
        min(H, max(py) + pad),
    ], dtype=np.float32)


# ────────────────────────────────────────────────────────────────────
# Humerus / forearm separation (used by segment_bone)
# ────────────────────────────────────────────────────────────────────

def _find_elbow_concavity(mask: np.ndarray) -> Optional[tuple[int, int]]:
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    if len(cnt) < 5:
        return None
    hull_idx = cv2.convexHull(cnt, returnPoints=False)
    if hull_idx is None or len(hull_idx) < 3:
        return None
    try:
        defects = cv2.convexityDefects(cnt, hull_idx)
    except cv2.error:
        return None
    if defects is None:
        return None
    best = max(defects, key=lambda d: d[0][3])
    s, e, f, depth = best[0]
    if depth < 256:
        return None
    fx, fy = cnt[f][0]
    return int(fx), int(fy)


def _separate_humerus_forearm(mask: np.ndarray,
                                capitellum_xy: Optional[tuple[int, int]] = None,
                                hum_frac: float = 0.20,
                                fore_dist_weight: float = 1.5):
    h, w = mask.shape
    bone_rows = np.where(np.any(mask > 0, axis=1))[0]
    bone_cols = np.where(np.any(mask > 0, axis=0))[0]
    if len(bone_rows) == 0:
        return mask, np.zeros_like(mask)

    hum_row_cut = bone_rows[int(len(bone_rows) * hum_frac)]
    hum_seed = np.zeros((h, w), dtype=np.uint8)
    hum_seed[:hum_row_cut] = (mask[:hum_row_cut] > 0).astype(np.uint8)

    if capitellum_xy is not None:
        ex, ey = capitellum_xy
    else:
        cxy = _find_elbow_concavity(mask)
        if cxy is not None:
            ex, ey = cxy
        else:
            fore_col_cut = bone_cols[int(len(bone_cols) * 0.85)]
            fore_row_cut = bone_rows[int(len(bone_rows) * 0.65)]
            ex, ey = None, None

    if ex is not None:
        margin_y = max(20, int(len(bone_rows) * 0.20))
        fore_row_cut = min(h - 1, ey + margin_y)
    else:
        fore_row_cut = bone_rows[int(len(bone_rows) * 0.65)]

    fore_seed = np.zeros((h, w), dtype=np.uint8)

    # Detect forearm direction: check bone mass below the capitellum
    # to handle both left-facing and right-facing lateral views.
    if ex is not None:
        below_rows = np.where(mask[fore_row_cut:] > 0)
        if len(below_rows[1]) > 0:
            mean_col_below = float(below_rows[1].mean())
            if mean_col_below < ex:
                # Forearm extends LEFT of capitellum
                margin_x = max(20, int(len(bone_cols) * 0.15))
                col_cut = max(0, ex - margin_x)
                fore_seed[fore_row_cut:, :col_cut] = (
                    mask[fore_row_cut:, :col_cut] > 0).astype(np.uint8)
            else:
                # Forearm extends RIGHT of capitellum
                margin_x = max(20, int(len(bone_cols) * 0.15))
                col_cut = min(w - 1, ex + margin_x)
                fore_seed[fore_row_cut:, col_cut:] = (
                    mask[fore_row_cut:, col_cut:] > 0).astype(np.uint8)
        else:
            # No bone below capitellum — seed all bone below
            fore_seed[fore_row_cut:] = (
                mask[fore_row_cut:] > 0).astype(np.uint8)
    else:
        fore_col_cut = bone_cols[int(len(bone_cols) * 0.85)]
        fore_seed[fore_row_cut:, fore_col_cut:] = (
            mask[fore_row_cut:, fore_col_cut:] > 0).astype(np.uint8)

    fore_seed[hum_seed > 0] = 0

    dist_hum  = distance_transform_edt(1 - hum_seed)
    dist_fore = distance_transform_edt(1 - fore_seed)
    in_mask   = mask > 0
    hum_mask  = (in_mask & (dist_hum <= dist_fore * fore_dist_weight)).astype(np.uint8)
    fore_mask = (in_mask & ~hum_mask.astype(bool)).astype(np.uint8)
    return hum_mask, fore_mask


# ────────────────────────────────────────────────────────────────────
# Main SAM2 segmenter class
# ────────────────────────────────────────────────────────────────────

class SAM2Segmenter:
    """Lazy-loaded SAM2 wrapper. SAM2 is heavy — instantiate once per process."""

    def __init__(self, sam2_config: str, sam2_ckpt: Path, device: str = "cpu"):
        self.sam2_config = sam2_config
        self.sam2_ckpt = sam2_ckpt
        self.device = device
        self._predictor = None

    def load(self):
        if self._predictor is not None:
            return
        # Imported lazily so the rest of the package can be used without sam2.
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        sam2_model = build_sam2(self.sam2_config, str(self.sam2_ckpt), device=self.device)
        self._predictor = SAM2ImagePredictor(sam2_model)

    def segment_bone(self,
                      gray_clahe: np.ndarray,
                      capitellum_xy: Optional[tuple[int, int]] = None,
                      k_positive: int = 15,
                      n_negative: int = 5,
                      min_dist: int = 15,
                      box_pad: int = 20,
                      n_zones: int = 3,
                      x_percentile: int = 60,
                      min_component_frac: float = 0.15) -> dict:
        """Segment the elbow bone, then split into humerus + forearm.

        Returns dict(bone_mask, humerus_mask, forearm_mask, score, probes).
        """
        self.load()
        H, W = gray_clahe.shape[:2]
        sam_image = np.stack([gray_clahe] * 3, axis=-1)

        clahe_float = gray_clahe.astype(np.float32) / 255.0
        otsu_t = threshold_otsu(clahe_float)
        otsu_mask = _safe_lcc_mask((clahe_float > otsu_t).astype(np.uint8))
        masked = clahe_float * otsu_mask

        k_per_zone = max(1, k_positive // n_zones)
        pos = _pick_stratified_probes(masked, otsu_mask, k_per_zone, min_dist, n_zones)
        neg = _pick_top_k_points(masked, n_negative, min_dist, brightest=False)
        if not pos:
            raise RuntimeError("No positive probes generated for SAM2 — image likely empty after Otsu")
        probe_pts = np.array(pos + neg, dtype=np.float32)
        probe_lbl = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
        probe_box = _probes_to_box(pos, box_pad, W, H, x_percentile=x_percentile)

        self._predictor.set_image(sam_image)
        with torch.inference_mode():
            masks, scores, _ = self._predictor.predict(
                point_coords=probe_pts, point_labels=probe_lbl,
                box=probe_box, multimask_output=True,
            )

        # Mask selection: max coverage, then smallest area
        probe_yx = [(int(p[1]), int(p[0])) for p in pos
                    if 0 <= int(p[1]) < H and 0 <= int(p[0]) < W]
        cands = []
        for i, m in enumerate(masks):
            hits = sum(m[r, c] > 0 for r, c in probe_yx)
            cov = hits / len(probe_yx) if probe_yx else 0
            cands.append((i, cov, int(m.sum())))
        cands.sort(key=lambda x: (-x[1], x[2]))
        best_cov = cands[0][1]
        viable = sorted([c for c in cands if c[1] >= best_cov - 0.05], key=lambda x: x[2])
        best_idx = viable[0][0]
        bone_mask = masks[best_idx].astype(np.uint8)

        # Morphology cleanup
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        bone_mask = cv2.morphologyEx(bone_mask, cv2.MORPH_CLOSE, kernel, iterations=2)

        # Component analysis: 2 large blobs (fracture gap) → assign by row
        n_cc, cc_lab = cv2.connectedComponents(bone_mask)
        comps = sorted(
            [(lbl, int((cc_lab == lbl).sum())) for lbl in range(1, n_cc)],
            key=lambda x: -x[1])

        if len(comps) >= 2 and comps[1][1] >= comps[0][1] * min_component_frac:
            def _row(lbl): return float(np.where(cc_lab == lbl)[0].mean())
            top2 = sorted(comps[:2], key=lambda x: _row(x[0]))
            humerus_mask = (cc_lab == top2[0][0]).astype(np.uint8)
            forearm_mask = (cc_lab == top2[1][0]).astype(np.uint8)
        else:
            if comps:
                bone_mask = (cc_lab == comps[0][0]).astype(np.uint8)
            humerus_mask, forearm_mask = _separate_humerus_forearm(
                bone_mask, capitellum_xy=capitellum_xy)

        # Clean humerus: keep only largest CC
        n_hum, hum_lab = cv2.connectedComponents(humerus_mask)
        if n_hum > 2:
            largest = max(range(1, n_hum), key=lambda l: (hum_lab == l).sum())
            humerus_mask = (hum_lab == largest).astype(np.uint8)

        return {
            "bone_mask": bone_mask,
            "humerus_mask": humerus_mask,
            "forearm_mask": forearm_mask,
            "sam_score": float(scores[best_idx]),
            "probes_pos": pos,
            "probes_neg": neg,
        }

    def segment_capitellum(self,
                            gray_clahe: np.ndarray,
                            humerus_mask: np.ndarray,
                            probe_xy: tuple[int, int],
                            box_r_x: int = 25,
                            box_r_up: int = 15,
                            box_r_dn: int = 30,
                            shaft_neg_offset: int = 35,
                            hum_erosion_iters: int = 3,
                            area_loss_frac: float = 0.30) -> dict:
        """SAM2 capitellum segmentation with humerus subtraction cleanup."""
        self.load()
        H, W = gray_clahe.shape[:2]
        probe_x, probe_y = int(probe_xy[0]), int(probe_xy[1])

        # Negative probes: corners + one along shaft
        margin = 10
        neg = [(margin, margin), (W - margin, margin),
               (margin, H - margin), (W - margin, H - margin)]

        hum_ys, hum_xs = np.where(humerus_mask > 0)
        if len(hum_xs) > 20:
            pts = np.stack([hum_xs, hum_ys], axis=1).astype(np.float32)
            centered = pts - pts.mean(axis=0, keepdims=True)
            cov = np.cov(centered.T)
            _, eigvecs = np.linalg.eigh(cov)
            axis = eigvecs[:, -1]
            cx, cy = float(hum_xs.mean()), float(hum_ys.mean())
            to_shaft = np.array([cx - probe_x, cy - probe_y], dtype=np.float32)
            if np.dot(axis, to_shaft) < 0:
                axis = -axis
            sx = int(round(probe_x + axis[0] * shaft_neg_offset))
            sy = int(round(probe_y + axis[1] * shaft_neg_offset))
            shaft_neg = (max(0, min(W - 1, sx)), max(0, min(H - 1, sy)))
        else:
            shaft_neg = (probe_x, max(0, probe_y - shaft_neg_offset))
        neg.append(shaft_neg)

        cap_box = np.array([
            max(0, probe_x - box_r_x),
            max(0, probe_y - box_r_up),
            min(W, probe_x + box_r_x),
            min(H, probe_y + box_r_dn),
        ], dtype=np.float32)

        probe_pts = np.array([[probe_x, probe_y]] + [list(p) for p in neg], dtype=np.float32)
        probe_lbl = np.array([1] + [0] * len(neg), dtype=np.int32)

        sam_image = np.stack([gray_clahe] * 3, axis=-1)
        self._predictor.set_image(sam_image)
        with torch.inference_mode():
            masks, scores, _ = self._predictor.predict(
                point_coords=probe_pts, point_labels=probe_lbl,
                box=cap_box, multimask_output=True,
            )

        # Smallest mask containing the probe
        cands = [(i, int(m.sum())) for i, m in enumerate(masks) if m[probe_y, probe_x]]
        cands.sort(key=lambda x: x[1])
        best_idx = cands[0][0] if cands else int(np.argmax(scores))
        cap_mask = masks[best_idx].astype(np.uint8)
        raw_area = int(cap_mask.sum())

        # Subtract humerus unless probe is inside it (humerus leakage case)
        probe_in_hum = bool(humerus_mask[probe_y, probe_x] > 0)
        if not probe_in_hum:
            hum_eroded = binary_erosion(humerus_mask > 0, iterations=hum_erosion_iters)
            candidate = (cap_mask.astype(bool) & ~hum_eroded).astype(np.uint8)
            if raw_area > 0 and int(candidate.sum()) < area_loss_frac * raw_area:
                pass  # subtraction destroyed too much -- keep raw
            else:
                cap_mask = candidate

        # Keep CC containing probe (or largest)
        lbl, n_cc = cc_label(cap_mask)
        if n_cc > 1:
            probe_lbl_id = lbl[probe_y, probe_x] if (0 <= probe_y < H and 0 <= probe_x < W) else 0
            if probe_lbl_id > 0:
                cap_mask = (lbl == probe_lbl_id).astype(np.uint8)
            else:
                sizes = [(i, int((lbl == i).sum())) for i in range(1, n_cc + 1)]
                biggest = max(sizes, key=lambda x: x[1])[0]
                cap_mask = (lbl == biggest).astype(np.uint8)
        elif n_cc == 0:
            cap_mask = masks[best_idx].astype(np.uint8)

        ys, xs = np.where(cap_mask > 0)
        if len(xs):
            cx_sam, cy_sam = int(xs.mean()), int(ys.mean())
            r_sam = max(int(max(xs.max() - xs.min(), ys.max() - ys.min()) / 2), 3)
        else:
            cx_sam, cy_sam, r_sam = probe_x, probe_y, 10

        return {
            "capitellum_mask": cap_mask,
            "centroid": (cx_sam, cy_sam),
            "radius": r_sam,
            "sam_score": float(scores[best_idx]),
        }
