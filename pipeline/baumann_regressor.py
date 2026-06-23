"""Baumann angle keypoint regressor — ResNet-18 → 4 keypoints → angle.

Predicts four anatomical keypoints on an AP elbow X-ray:
  0: proximal_shaft_mid  — humeral shaft midpoint (proximal)
  1: distal_shaft_mid    — humeral shaft midpoint (distal)
  2: medial_physis       — medial end of the capitellar physis
  3: lateral_physis      — lateral end of the capitellar physis

Baumann angle = angle between shaft vector (distal − proximal)
                and physis vector (lateral − medial).
Normal range: 60–84°.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

TRAIN_SIZE = 256
NORMAL_LO = 60.0
NORMAL_HI = 84.0

_KP_NAMES = ["Prox shaft", "Dist shaft", "Med physis", "Lat physis"]
_KP_COLORS_SHAFT = "#00AAFF"
_KP_COLORS_PHYSIS = "#FF8800"


def _build_model() -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(512, 8)
    return model


def load_regressor(
    ckpt_path: Path,
    device: Optional[str] = None,
) -> Tuple[nn.Module, str]:
    """Load checkpoint and return (model, device_str)."""
    if device is None or device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _build_model()
    state = torch.load(str(ckpt_path), map_location=device)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    return model, device


def predict_keypoints(gray: np.ndarray, model: nn.Module, device: str) -> dict:
    """Predict keypoints and Baumann angle from a uint8 grayscale AP image.

    Returns
    -------
    dict with keys:
        keypoints_norm    : (4, 2) float64 in [0, 1]
        keypoints_px      : (4, 2) float64 in original image pixel coords
        baumann_angle_deg : float
        in_normal_range   : bool
    """
    orig_h, orig_w = gray.shape[:2]
    resized = cv2.resize(gray, (TRAIN_SIZE, TRAIN_SIZE))

    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    t = np.stack([resized, resized, resized], axis=0).astype(np.float32) / 255.0
    t = normalize(torch.from_numpy(t)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(t).cpu().squeeze()

    kps_norm = pred.reshape(4, 2).numpy().astype(np.float64)
    kps_px = kps_norm * np.array([orig_w, orig_h])

    shaft = kps_norm[1] - kps_norm[0]
    physis = kps_norm[3] - kps_norm[2]
    cos_a = float(
        np.dot(shaft, physis)
        / (np.linalg.norm(shaft) * np.linalg.norm(physis) + 1e-8)
    )
    angle = float(np.degrees(np.arccos(np.clip(cos_a, -1.0, 1.0))))
    if angle > 90:
        angle = 180.0 - angle

    return dict(
        keypoints_norm=kps_norm,
        keypoints_px=kps_px,
        baumann_angle_deg=round(angle, 1),
        in_normal_range=(NORMAL_LO <= angle <= NORMAL_HI),
    )


def plot_keypoints(gray: np.ndarray, result: dict):
    """Return a matplotlib Figure overlaying keypoints and lines on the AP image."""
    import matplotlib.pyplot as plt

    kps = result["keypoints_px"]   # (4, 2) pixel coords
    angle = result["baumann_angle_deg"]
    in_range = result["in_normal_range"]

    fig, ax = plt.subplots(figsize=(5, 7))
    fig.patch.set_facecolor("#111")
    ax.set_facecolor("#111")
    ax.imshow(gray, cmap="gray")

    # Shaft axis line
    ax.plot([kps[0, 0], kps[1, 0]], [kps[0, 1], kps[1, 1]],
            color="cyan", lw=2.5, zorder=4, label="Shaft axis")
    # Physis line
    ax.plot([kps[2, 0], kps[3, 0]], [kps[2, 1], kps[3, 1]],
            color="lime", lw=2.5, zorder=4, label="Physis line")

    kp_colors = [_KP_COLORS_SHAFT, _KP_COLORS_SHAFT, _KP_COLORS_PHYSIS, _KP_COLORS_PHYSIS]
    for kp, color, name in zip(kps, kp_colors, _KP_NAMES):
        ax.scatter(kp[0], kp[1], c=color, s=90, zorder=5,
                   edgecolors="white", linewidths=0.8)
        ax.annotate(name, (kp[0], kp[1]),
                    textcoords="offset points", xytext=(8, 3),
                    fontsize=7, color=color,
                    bbox=dict(boxstyle="round,pad=0.2", fc="#111", alpha=0.6, lw=0))

    colour = "#50c864" if in_range else "#e05252"
    tag = "(normal)" if in_range else "(abnormal)"
    ax.set_title(
        f"Regressor: Baumann = {angle}°  {tag}",
        color=colour, fontsize=10, fontweight="bold",
    )
    ax.legend(fontsize=8, loc="upper right",
              labelcolor="white", facecolor="#222", edgecolor="#444")
    ax.axis("off")
    fig.tight_layout(pad=0.5)
    return fig
