"""Baumann angle computation for AP-view elbow X-rays.

Pipeline (YOLO + PCA-alignment):
  1. YOLO instance segmentation → humerus mask (highest-confidence detection)
  2. PCA alignment: rotate image + mask so shaft becomes vertical, condyle at bottom
  3. Find condyle level (peak bone-width row of humerus mask)
  4. Shaft axis via PCA on proximal shaft midpoints
  5. Physeal line: sigma-clip polyfit to distal boundary of humerus mask
     (over the entire distal half, not just a band around the condyle)
  6. Baumann angle = angle between shaft direction and physeal line direction

Normal range: 60–84° (mean 72°)

Usage
-----
    from pipeline.baumann import compute_baumann_angle, draw_step_panels

    result = compute_baumann_angle(gray_img, yolo_model)
    print(result["baumann_angle_deg"])
    fig = draw_step_panels(result["gray"], result)
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks
from typing import Optional

# ── Constants ────────────────────────────────────────────────────────────────
BAUMANN_NORMAL_MEAN = 72.0
BAUMANN_NORMAL_SD   = 12.0
BAUMANN_NORMAL_LO   = BAUMANN_NORMAL_MEAN - BAUMANN_NORMAL_SD   # 60°
BAUMANN_NORMAL_HI   = BAUMANN_NORMAL_MEAN + BAUMANN_NORMAL_SD   # 84°

_SHAFT_END_FRAC    = 0.70
_SIGMA_CLIP_PASSES = 3
_SIGMA_CLIP_THRESH = 2.5

YOLO_CONF = 0.25


# ── Step 0: YOLO humerus mask ─────────────────────────────────────────────────

def yolo_humerus_mask(source, yolo_model, conf: float = YOLO_CONF) -> Optional[np.ndarray]:
    """Run YOLO on an image (path or numpy array) and return the highest-confidence humerus mask."""
    # YOLO backbone expects 3-channel input; convert grayscale arrays before inference
    if isinstance(source, np.ndarray):
        if source.ndim == 2:
            source = np.stack([source] * 3, axis=-1)
        elif source.ndim == 3 and source.shape[2] == 1:
            source = np.concatenate([source] * 3, axis=-1)
    res = yolo_model(source, conf=conf, retina_masks=True, verbose=False)[0]
    if res.masks is None or len(res.masks.data) == 0:
        return None
    i = int(res.boxes.conf.argmax())
    m = (res.masks.data[i].cpu().numpy() > 0.5).astype(np.uint8)
    orig_h, orig_w = res.orig_shape
    if m.shape[:2] != (orig_h, orig_w):
        m = cv2.resize(m, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
    return m


# ── Orientation helper ────────────────────────────────────────────────────────

def _distal_is_below(m: np.ndarray) -> bool:
    """True when the condyle (wider end) is at the bottom of the mask."""
    ys = np.where(m.any(axis=1))[0]
    y0, y1 = int(ys.min()), int(ys.max())
    band = max(3, int(0.2 * (y1 - y0)))

    def meanw(rows):
        w = []
        for y in rows:
            b = np.where(m[y] > 0)[0]
            if len(b) >= 2:
                w.append(b[-1] - b[0])
        return float(np.mean(w)) if w else 0.0

    return meanw(range(y1 - band, y1 + 1)) > meanw(range(y0, y0 + band))


# ── Step 1: Upright alignment (PCA on mask) ───────────────────────────────────

def _align_upright(gray: np.ndarray, mask: np.ndarray):
    """Rotate image and mask so the humerus shaft is vertical, condyle at bottom.

    Returns (gray_rot, mask_rot, rotation_angle_deg).
    """
    ys, xs = np.where(mask > 0)
    pts = np.column_stack([xs.astype(float), ys.astype(float)])
    mean_pt = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - mean_pt, full_matrices=False)
    principal = Vt[0]
    if principal[1] < 0:
        principal = -principal
    angle_deg = float(np.degrees(np.arctan2(principal[0], principal[1])))

    H, W = gray.shape[:2]
    cx, cy = W / 2.0, H / 2.0
    M = cv2.getRotationMatrix2D((cx, cy), -angle_deg, 1.0)
    cos_a, sin_a = abs(M[0, 0]), abs(M[0, 1])
    new_W = int(H * sin_a + W * cos_a)
    new_H = int(H * cos_a + W * sin_a)
    M[0, 2] += new_W / 2.0 - cx
    M[1, 2] += new_H / 2.0 - cy

    gray_rot = cv2.warpAffine(gray, M, (new_W, new_H), flags=cv2.INTER_LINEAR)
    mask_rot = cv2.warpAffine(mask, M, (new_W, new_H), flags=cv2.INTER_NEAREST)

    if not _distal_is_below(mask_rot):
        cx2, cy2 = new_W / 2.0, new_H / 2.0
        M2 = cv2.getRotationMatrix2D((cx2, cy2), 180.0, 1.0)
        gray_rot = cv2.warpAffine(gray_rot, M2, (new_W, new_H), flags=cv2.INTER_LINEAR)
        mask_rot = cv2.warpAffine(mask_rot, M2, (new_W, new_H), flags=cv2.INTER_NEAREST)
        angle_deg += 180.0

    return gray_rot, mask_rot, angle_deg


# ── Step 2: Condyle level ─────────────────────────────────────────────────────

def _find_condyle_level(mask: np.ndarray, smooth_win: int = 20) -> dict:
    """Locate the row with peak bone width (= condyle level)."""
    ys, xs = np.where(mask > 0)
    y0, y1 = int(ys.min()), int(ys.max())
    widths = []
    for y in range(y0, y1 + 1):
        b = np.where(mask[y] > 0)[0]
        widths.append(int(b[-1] - b[0]) if len(b) >= 2 else 0)
    ws = uniform_filter1d(np.array(widths, dtype=float), size=smooth_win)
    N = len(ws)
    margin = max(5, int(0.05 * N))
    peaks, props = find_peaks(ws, prominence=3)
    interior = peaks[(peaks >= margin) & (peaks < N - margin)]
    if len(interior) > 0:
        prom = props["prominences"][(peaks >= margin) & (peaks < N - margin)]
        off  = int(interior[prom.argmax()])
    else:
        off = int(np.argmax(ws[margin: N - margin])) + margin
    y_cond = y0 + off
    return dict(y_condyle=y_cond, y_arm_top=y0, y_arm_bot=y1, width_profile=ws)


# ── Step 3: Shaft axis ────────────────────────────────────────────────────────

def _fit_shaft_axis(mask: np.ndarray, condyle: dict) -> Optional[dict]:
    """PCA on bone midpoints in the proximal shaft region."""
    y0 = condyle["y_arm_top"];  y1 = condyle["y_arm_bot"]
    y_c = condyle["y_condyle"]; above = condyle["shaft_above"]
    if above:
        r_start = y0
        r_end   = int(y0 + _SHAFT_END_FRAC * (y_c - y0))
    else:
        r_start = int(y_c + (1 - _SHAFT_END_FRAC) * (y1 - y_c))
        r_end   = y1
    pts = []
    for y in range(r_start, r_end + 1):
        b = np.where(mask[y] > 0)[0]
        if len(b) >= 5:
            pts.append([(b[0] + b[-1]) / 2.0, float(y)])
    if len(pts) < 15:
        return None
    pts = np.array(pts)
    mean_pt = pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts - mean_pt, full_matrices=False)
    d = Vt[0]
    if d[1] < 0:
        d = -d
    return dict(
        direction=d,
        centroid=mean_pt,
        angle_deg=float(np.degrees(np.arctan2(d[0], d[1]))),
        shaft_above=above,
        shaft_rows=(r_start, r_end),
    )


# ── Step 4: Physeal line ──────────────────────────────────────────────────────

def _detect_physis_line(mask: np.ndarray, condyle: dict,
                         seg_mask: Optional[np.ndarray] = None) -> Optional[dict]:
    """Fit a line to the distal boundary of the humerus mask over the entire distal half.

    Uses column-wise bottommost (shaft above) or topmost (shaft below) mask pixels,
    sigma-clipped to a robust linear fit.
    """
    y_c   = condyle["y_condyle"]
    y_top = condyle["y_arm_top"]
    y_bot = condyle["y_arm_bot"]
    above = condyle.get("shaft_above", True)

    # Entire distal half: condyle-to-bottom when shaft is above, top-to-condyle otherwise
    if above:
        win_y0, win_y1 = y_c, y_bot + 1
    else:
        win_y0, win_y1 = y_top, y_c

    if win_y1 - win_y0 < 5:
        return None

    ref = seg_mask if seg_mask is not None else mask
    col_xs = np.where(ref[win_y0:win_y1].any(axis=0))[0]
    if len(col_xs) < 10:
        return None

    bp = []
    for x in col_xs:
        bone_rows = np.where(mask[win_y0:win_y1, x] > 0)[0]
        if len(bone_rows) >= 1:
            row_idx = bone_rows[-1] if above else bone_rows[0]
            bp.append([float(x), float(win_y0 + row_idx)])
    if len(bp) < 10:
        return None
    bp = np.array(bp)

    a, b = np.polyfit(bp[:, 0], bp[:, 1], 1)
    for _ in range(_SIGMA_CLIP_PASSES):
        res  = bp[:, 1] - (a * bp[:, 0] + b)
        keep = np.abs(res) < _SIGMA_CLIP_THRESH * res.std()
        if keep.sum() < 10:
            break
        a, b = np.polyfit(bp[keep, 0], bp[keep, 1], 1)

    direction = np.array([1.0, a]) / np.sqrt(1.0 + a ** 2)
    x_min, x_max = float(bp[:, 0].min()), float(bp[:, 0].max())
    return dict(
        slope=a, intercept=b, direction=direction,
        angle_deg=float(np.degrees(np.arctan(a))),
        x_range=(x_min, x_max), boundary_pts=bp,
    )


# ── Step 5: Baumann angle ─────────────────────────────────────────────────────

def _compute_baumann(shaft: Optional[dict], physis: Optional[dict]) -> Optional[float]:
    """Unsigned acute angle between shaft axis and physeal line."""
    if shaft is None or physis is None:
        return None
    dot   = float(abs(np.dot(shaft["direction"], physis["direction"])))
    angle = float(np.degrees(np.arccos(np.clip(dot, 0.0, 1.0))))
    return round(180.0 - angle if angle > 90 else angle, 1)


# ── Core: mask → Baumann result ───────────────────────────────────────────────

def baumann_from_humerus_mask(humerus_mask: np.ndarray,
                               gray: Optional[np.ndarray] = None,
                               force_shaft_above: Optional[bool] = None) -> dict:
    """Compute Baumann angle from an already-obtained humerus segmentation mask."""
    m = (humerus_mask > 0).astype(np.uint8)
    if m.sum() < 500:
        return dict(status="no_mask", baumann_angle_deg=None, in_normal_range=None,
                    shaft=None, physis=None, condyle=None, mask=m,
                    humerus_mask=m, forearm_mask=None)

    condyle = _find_condyle_level(m)
    condyle["shaft_above"] = (force_shaft_above if force_shaft_above is not None
                              else _distal_is_below(m))

    shaft  = _fit_shaft_axis(m, condyle)
    physis = _detect_physis_line(m, condyle) if shaft is not None else None
    angle  = _compute_baumann(shaft, physis)

    status = "ok"
    if   shaft  is None: status = "no_shaft"
    elif physis is None: status = "no_physis"
    in_range = (BAUMANN_NORMAL_LO <= angle <= BAUMANN_NORMAL_HI) if angle is not None else None

    return dict(status=status, baumann_angle_deg=angle, in_normal_range=in_range,
                shaft=shaft, physis=physis, condyle=condyle,
                mask=m, humerus_mask=m, forearm_mask=None)


# ── Public entry point ────────────────────────────────────────────────────────

def compute_baumann_angle(gray: np.ndarray, yolo_model) -> dict:
    """Full Baumann angle pipeline for a single AP elbow image.

    Parameters
    ----------
    gray : np.ndarray
        uint8 grayscale AP image (markers already removed).
    yolo_model : ultralytics.YOLO
        Trained YOLO segmentation model for humerus instance segmentation.

    Returns
    -------
    dict with keys:
        status             : 'ok' | 'no_mask' | 'no_shaft' | 'no_physis'
        baumann_angle_deg  : float or None
        in_normal_range    : bool or None
        shaft              : shaft dict (direction, centroid, angle_deg, ...)
        physis             : physis dict (slope, intercept, direction, ...)
        condyle            : condyle dict
        mask               : humerus binary mask (same as humerus_mask)
        humerus_mask       : humerus binary mask
        forearm_mask       : None (unused in YOLO path)
        gray               : rotated/aligned grayscale image used for all calculations
        rotation_angle_deg : rotation applied to align shaft vertically
    """
    if gray is None:
        return dict(status="no_image", baumann_angle_deg=None, in_normal_range=None,
                    shaft=None, physis=None, condyle=None, mask=None,
                    humerus_mask=None, forearm_mask=None, gray=None,
                    rotation_angle_deg=None)

    hmask = yolo_humerus_mask(gray, yolo_model)
    if hmask is None:
        return dict(status="no_mask", baumann_angle_deg=None, in_normal_range=None,
                    shaft=None, physis=None, condyle=None, mask=None,
                    humerus_mask=None, forearm_mask=None, gray=gray,
                    rotation_angle_deg=None)

    gray_rot, mask_rot, rot_angle = _align_upright(gray, hmask)
    result = baumann_from_humerus_mask(mask_rot, gray_rot, force_shaft_above=True)
    result["gray"] = gray_rot
    result["rotation_angle_deg"] = rot_angle
    return result


# ── Visualisation ─────────────────────────────────────────────────────────────

def draw_step_panels(gray: np.ndarray, result: dict,
                      figsize: tuple = (16, 7)) -> np.ndarray:
    """Return a 1×4 panel figure as a numpy RGBA array.

    Panels:
      1. Original AP (upright-aligned)
      2. YOLO humerus mask overlay
      3. Shaft PCA axis + condyle level
      4. Physeal line + Baumann angle annotation
    """
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image as PILImage

    # Prefer the rotated/aligned gray stored inside the result
    display_gray = result.get("gray") if result.get("gray") is not None else gray
    if display_gray is None:
        display_gray = gray
    if display_gray.ndim == 3:
        display_gray = display_gray[:, :, 0]

    mask     = result.get("mask")
    if mask is None:
        mask = result.get("humerus_mask")
    shaft    = result.get("shaft")
    physis   = result.get("physis")
    condyle  = result.get("condyle") or {}
    angle    = result.get("baumann_angle_deg")
    in_range = result.get("in_normal_range")

    if mask is None:
        fig, ax = plt.subplots(figsize=(5, 2))
        ax.text(0.5, 0.5, f"No mask — status: {result.get('status')}",
                ha="center", va="center")
        ax.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        buf.seek(0)
        return np.array(PILImage.open(buf))

    ys_m, xs_m = np.where(mask > 0)
    PAD = 30
    bx0 = max(0, int(xs_m.min()) - PAD);  bx1 = min(display_gray.shape[1], int(xs_m.max()) + PAD)
    by0 = max(0, int(ys_m.min()) - PAD);  by1 = min(display_gray.shape[0], int(ys_m.max()) + PAD)

    def _crop(im):
        return im[by0:by1, bx0:bx1]

    t_max = max(display_gray.shape)
    crop_h, crop_w = by1 - by0, bx1 - bx0

    fig, axs = plt.subplots(1, 4, figsize=figsize)
    fig.patch.set_facecolor("#111")
    for ax in axs:
        ax.set_facecolor("#111")
        ax.axis("off")

    # Panel 1: Original AP (upright-aligned)
    rot_label = ""
    if result.get("rotation_angle_deg") is not None:
        rot_label = f"  (rotated {result['rotation_angle_deg']:.1f}°)"
    axs[0].imshow(_crop(display_gray), cmap="gray")
    axs[0].set_title(f"1. Upright AP{rot_label}", color="white", fontsize=9)

    # Panel 2: YOLO humerus mask overlay
    overlay = np.zeros((*display_gray.shape, 3), dtype=np.uint8)
    overlay[mask > 0] = [60, 200, 230]
    blend = (0.6 * np.stack([_crop(display_gray)] * 3, -1)
             + 0.4 * _crop(overlay)).astype(np.uint8)
    axs[1].imshow(blend)
    axs[1].set_title(f"2. YOLO humerus ({mask.mean():.1%} area)", color="cyan", fontsize=9)

    # Panel 3: Shaft PCA + condyle level
    axs[2].imshow(_crop(display_gray), cmap="gray")
    if shaft is not None:
        cx, cy = shaft["centroid"];  dx, dy = shaft["direction"]
        cxc, cyc = cx - bx0, cy - by0
        axs[2].plot([cxc - dx * t_max, cxc + dx * t_max],
                    [cyc - dy * t_max, cyc + dy * t_max],
                    "c-", lw=2, label=f"Shaft ({shaft['angle_deg']:.1f}°)")
        rs, re = shaft["shaft_rows"]
        axs[2].axhspan(rs - by0, re - by0, alpha=0.12, color="cyan")
    y_cond = condyle.get("y_condyle")
    if y_cond is not None:
        axs[2].axhline(y_cond - by0, color="orange", lw=1.5, ls="--",
                       label=f"Condyle y={y_cond}")
    axs[2].set_xlim(0, crop_w);  axs[2].set_ylim(crop_h, 0)
    axs[2].legend(fontsize=6, loc="upper right")
    axs[2].set_title("3. Shaft PCA + condyle", color="white", fontsize=9)

    # Panel 4: Physeal line + Baumann angle
    axs[3].imshow(_crop(display_gray), cmap="gray")
    if shaft is not None:
        cx, cy = shaft["centroid"];  dx, dy = shaft["direction"]
        cxc, cyc = cx - bx0, cy - by0
        axs[3].plot([cxc - dx * t_max, cxc + dx * t_max],
                    [cyc - dy * t_max, cyc + dy * t_max],
                    "c-", lw=1.5, alpha=0.7, label="Shaft")
    if physis is not None:
        x0_, x1_ = physis["x_range"]
        y0_ = physis["slope"] * x0_ + physis["intercept"]
        y1_ = physis["slope"] * x1_ + physis["intercept"]
        axs[3].plot([x0_ - bx0, x1_ - bx0], [y0_ - by0, y1_ - by0],
                    color="lime", lw=2.5,
                    label=f"Physis ({physis['angle_deg']:.1f}°)")
        bp = physis["boundary_pts"]
        axs[3].scatter(bp[::5, 0] - bx0, bp[::5, 1] - by0, c="yellow", s=6, zorder=5)
    axs[3].set_xlim(0, crop_w);  axs[3].set_ylim(crop_h, 0)
    axs[3].legend(fontsize=6, loc="upper right")
    colour = "#50c864" if in_range else ("#e05252" if in_range is not None else "white")
    tag    = "(normal)" if in_range else ("(abnormal)" if in_range is not None else "")
    axs[3].set_title(f"4. Baumann = {angle}°  {tag}" if angle else "4. Baumann: n/a",
                     color=colour, fontsize=9)

    plt.tight_layout(pad=0.3)
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    return np.array(PILImage.open(buf))
