"""Image preprocessing: L/R marker removal + 6-step alignment.

Ported verbatim from inference_notebook.ipynb (cells 7, 9) and
alignment_pipeline_allinone.ipynb (cell 3). The full alignment is in
align_full() and matches the function _step1_to_6_debug_views from the
batch alignment notebook — but stripped of plotting and metadata
diagnostics that aren't needed at inference time.
"""
from __future__ import annotations
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops


# ────────────────────────────────────────────────────────────────────
# Marker removal
# ────────────────────────────────────────────────────────────────────

def remove_lr_markers(img_gray: np.ndarray,
                      max_area_frac: float = 0.005,
                      min_brightness: int = 220) -> tuple[np.ndarray, np.ndarray]:
    """Detect bright L/R text markers and inpaint them out.

    Markers are near-white (>=min_brightness), small (<max_area_frac of image),
    and compact. Bone is dimmer and elongated, so it won't be caught.

    Returns (cleaned_image, marker_mask).
    """
    _, bright_mask = cv2.threshold(img_gray, min_brightness, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    img_area = img_gray.shape[0] * img_gray.shape[1]
    marker_mask = np.zeros_like(img_gray)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 20 or area > img_area * max_area_frac:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = max(w, h) / (min(w, h) + 1e-6)
        if aspect > 5:
            continue
        cv2.drawContours(marker_mask, [c], -1, 255, thickness=-1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    marker_mask = cv2.dilate(marker_mask, kernel, iterations=2)
    cleaned = cv2.inpaint(img_gray, marker_mask, inpaintRadius=10, flags=cv2.INPAINT_TELEA)
    return cleaned, marker_mask


# ────────────────────────────────────────────────────────────────────
# Alignment helpers (shared with SAM2 step)
# ────────────────────────────────────────────────────────────────────

def _safe_lcc_mask(binary_mask: np.ndarray) -> np.ndarray:
    """Largest connected component of a binary mask."""
    labeled = label(binary_mask)
    if labeled.max() == 0:
        return binary_mask.astype(np.uint8)
    largest = max(regionprops(labeled), key=lambda r: r.area)
    return (labeled == largest.label).astype(np.uint8)


def _forearm_pca_angle(mask_oriented: np.ndarray) -> float:
    """PCA angle of the forearm portion (bottom-right) of the mask."""
    h, w = mask_oriented.shape
    right_half = mask_oriented[:, w // 2:]
    row_sums = np.sum(right_half > 0, axis=1)
    active_rows = np.where(row_sums > 0)[0]
    if len(active_rows) < 2:
        ys, xs = np.where(mask_oriented > 0)
        if len(xs) < 2:
            return 0.0
    else:
        mid_row = active_rows[0] + int(0.6 * (active_rows[-1] - active_rows[0]))
        forearm_mask = np.zeros_like(mask_oriented)
        forearm_mask[mid_row:, w // 2:] = mask_oriented[mid_row:, w // 2:]
        ys, xs = np.where(forearm_mask > 0)
        if len(xs) < 2:
            ys, xs = np.where(mask_oriented > 0)
            if len(xs) < 2:
                return 0.0
    coords = np.column_stack([xs, ys]).astype(np.float64)
    centered = coords - coords.mean(axis=0)
    cov = np.cov(centered, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    principal = eigvecs[:, np.argmax(eigvals)]
    return float(np.degrees(np.arctan2(principal[1], principal[0])))


def _rotate_with_expanded_canvas(img: np.ndarray, mask: np.ndarray,
                                  angle_deg: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = img.shape[:2]
    center = (w / 2.0, h / 2.0)
    m = cv2.getRotationMatrix2D(center, float(angle_deg), 1.0)
    cos_a, sin_a = abs(m[0, 0]), abs(m[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    m[0, 2] += (new_w - w) / 2.0
    m[1, 2] += (new_h - h) / 2.0
    rot_img = cv2.warpAffine(img, m, (new_w, new_h),
                             flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    rot_mask = cv2.warpAffine((mask > 0).astype(np.uint8) * 255, m, (new_w, new_h),
                              flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rot_img, (rot_mask > 0).astype(np.uint8)


def _zoom_and_crop(image_arr: np.ndarray, mask_arr: np.ndarray,
                    zoom_out: float, padding_ratio: float) -> np.ndarray:
    if 0.0 < zoom_out < 1.0:
        h, w = image_arr.shape[:2]
        nw, nh = max(1, int(w * zoom_out)), max(1, int(h * zoom_out))
        small_img = cv2.resize(image_arr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        small_mask = cv2.resize(mask_arr, (nw, nh), interpolation=cv2.INTER_NEAREST)
        canvas_img = np.zeros((h, w), dtype=np.uint8)
        canvas_mask = np.zeros((h, w), dtype=np.uint8)
        x0, y0 = (w - nw) // 2, (h - nh) // 2
        canvas_img[y0:y0 + nh, x0:x0 + nw] = small_img
        canvas_mask[y0:y0 + nh, x0:x0 + nw] = (small_mask > 0).astype(np.uint8)
        image_arr, mask_arr = canvas_img, canvas_mask

    ys, xs = np.where(mask_arr > 0)
    if len(xs) == 0:
        return image_arr
    y1, y2 = ys.min(), ys.max() + 1
    x1, x2 = xs.min(), xs.max() + 1
    bh, bw = y2 - y1, x2 - x1
    pad_y, pad_x = int(bh * padding_ratio), int(bw * padding_ratio)
    y1 = max(0, y1 - pad_y)
    y2 = min(image_arr.shape[0], y2 + pad_y)
    x1 = max(0, x1 - pad_x)
    x2 = min(image_arr.shape[1], x2 + pad_x)
    return image_arr[y1:y2, x1:x2]


# ────────────────────────────────────────────────────────────────────
# Full 6-step alignment (LAT view)
# ────────────────────────────────────────────────────────────────────

def align_full(img_gray: np.ndarray,
                downsample_size: int = 128,
                output_size: tuple = (256, 256),
                zoom_out: float = 0.65,
                padding_ratio: float = 0.05) -> dict:
    """Full LAT alignment: marker removal → content crop → orientation → PCA rotation → zoom.

    Input: grayscale uint8 image of arbitrary size.
    Returns dict with keys:
        aligned (np.ndarray HxW uint8) — final aligned image at output_size
        cleaned (np.ndarray) — after marker removal (full res)
        flip_lr (bool), rotation (float, degrees), mask (np.ndarray)

    Mirrors _step1_to_6_debug_views in alignment_pipeline_allinone.ipynb,
    minus the visualization side outputs.
    """
    # Step 1-2: marker removal + content crop
    cleaned, _ = remove_lr_markers(img_gray)
    img_float = cleaned.astype(np.float32) / 255.0
    img_blur  = cv2.GaussianBlur(img_float, (5, 5), 0)
    t0 = threshold_otsu(img_blur)
    bin0 = (img_blur > t0).astype(np.uint8)

    h0, w0 = cleaned.shape
    lab0 = label(bin0)
    if lab0.max() > 0:
        reg0 = max(regionprops(lab0), key=lambda r: r.area)
        cy1, cx1, cy2, cx2 = reg0.bbox
        my = max(2, int(0.03 * (cy2 - cy1)))
        mx = max(2, int(0.03 * (cx2 - cx1)))
        cy1 = max(0, cy1 - my); cy2 = min(h0, cy2 + my)
        cx1 = max(0, cx1 - mx); cx2 = min(w0, cx2 + mx)
    else:
        cy1, cy2, cx1, cx2 = 0, h0, 0, w0
    img_crop = cleaned[cy1:cy2, cx1:cx2]
    h_crop, w_crop = img_crop.shape

    # Letterbox downsample
    scale_ds = min(downsample_size / max(1, w_crop), downsample_size / max(1, h_crop))
    nw_ds, nh_ds = max(1, int(w_crop * scale_ds)), max(1, int(h_crop * scale_ds))
    small = np.zeros((downsample_size, downsample_size), dtype=np.uint8)
    small_resized = cv2.resize(img_crop, (nw_ds, nh_ds), interpolation=cv2.INTER_AREA)
    dl, dt = (downsample_size - nw_ds) // 2, (downsample_size - nh_ds) // 2
    small[dt:dt + nh_ds, dl:dl + nw_ds] = small_resized

    # Step 3: Otsu LCC mask on downsampled
    sf = small.astype(np.float32) / 255.0
    th = threshold_otsu(sf)
    binary = (sf > th).astype(np.uint8)
    mask = _safe_lcc_mask(binary)

    # Step 4: orientation by quadrant brightness
    masked_small = sf * mask
    h_d, w_d = mask.shape
    mid_y, mid_x = h_d // 2, w_d // 2
    q_tl = masked_small[:mid_y, :mid_x].sum()
    q_tr = masked_small[:mid_y, mid_x:].sum()
    q_bl = masked_small[mid_y:, :mid_x].sum()
    q_br = masked_small[mid_y:, mid_x:].sum()
    quads = np.array([q_tl, q_tr, q_bl, q_br])
    top2 = set(np.argsort(quads)[-2:][::-1].tolist())
    if   top2 == {1, 2}: flip_lr = True
    elif top2 == {0, 3}: flip_lr = False
    elif top2 == {0, 2}: flip_lr = True
    elif top2 == {2, 3}: flip_lr = False
    elif top2 == {1, 3}: flip_lr = False
    else:
        flip_lr = (q_tl + q_bl) > (q_tr + q_br)

    if flip_lr:
        mask_oriented = np.fliplr(mask)
        cleaned_oriented = np.fliplr(cleaned)
    else:
        mask_oriented = mask
        cleaned_oriented = cleaned

    # Step 5: forearm PCA -> rotation angle
    angle = _forearm_pca_angle(mask_oriented)
    rotation = angle
    if rotation > 90: rotation -= 180
    elif rotation < -90: rotation += 180

    # Step 6: rotate full-res, then zoom + crop
    h_full, w_full = cleaned_oriented.shape
    mask_fullres = cv2.resize(mask_oriented.astype(np.uint8),
                               (w_full, h_full), interpolation=cv2.INTER_NEAREST)
    rot_img, rot_mask = _rotate_with_expanded_canvas(cleaned_oriented, mask_fullres, rotation)
    cropped = _zoom_and_crop(rot_img, rot_mask, zoom_out, padding_ratio)

    # Resize to output_size (longer side fits, centre-crop the rest)
    out_w, out_h = output_size
    in_h, in_w = cropped.shape[:2]
    if in_h == 0 or in_w == 0:
        aligned = np.zeros((out_h, out_w), dtype=np.uint8)
    else:
        scale = max(out_w / max(1, in_w), out_h / max(1, in_h))
        nw, nh = max(1, int(in_w * scale)), max(1, int(in_h * scale))
        resized = cv2.resize(cropped, (nw, nh), interpolation=cv2.INTER_LINEAR)
        x0 = max(0, (nw - out_w) // 2)
        y0 = max(0, (nh - out_h) // 2)
        aligned = resized[y0:y0 + out_h, x0:x0 + out_w]
        if aligned.shape != (out_h, out_w):
            aligned = cv2.resize(aligned, (out_w, out_h), interpolation=cv2.INTER_LINEAR)

    return {
        "aligned": aligned,
        "cleaned": cleaned,
        "flip_lr": bool(flip_lr),
        "rotation": float(rotation),
        "mask": mask_oriented,
    }


def apply_clahe(img_gray: np.ndarray,
                 clip: float = 3.0,
                 grid: tuple = (8, 8)) -> np.ndarray:
    op = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    return op.apply(img_gray)
