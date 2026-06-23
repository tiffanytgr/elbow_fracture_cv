"""Capitellum centre regressor.

ResNet-18 + small head trained to predict (cap_x, cap_y) in normalised
[0,1] coordinates on a 256x256 CLAHE input. See
capitellum_regressor_training.ipynb for training details.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

from .preprocessing import apply_clahe


class CapitellusRegressor(nn.Module):
    """Architecture must match the training notebook exactly."""
    def __init__(self):
        super().__init__()
        bb = models.resnet18(weights=None)
        self.features = nn.Sequential(*list(bb.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.head(self.features(x))


def load_regressor(ckpt: Path, device: str = "cpu") -> CapitellusRegressor:
    model = CapitellusRegressor().to(device)
    state = torch.load(ckpt, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def predict_capitellum(model: CapitellusRegressor,
                        img_gray: np.ndarray,
                        img_size: int = 256,
                        clahe_clip: float = 3.0,
                        clahe_grid: tuple = (8, 8),
                        device: str = "cpu") -> tuple[int, int]:
    """Predict capitellum centre in pixel coords on the input image.

    img_gray must be the same image (size and CLAHE state) that downstream
    SAM2 / geometry steps use, otherwise the prediction will be misaligned.
    """
    # The training notebook applied CLAHE then resized to 256x256 — so we
    # do the same here. We do NOT re-apply CLAHE if it was already applied
    # upstream; check by passing in pre-CLAHE grayscale only.
    gray_clahe = apply_clahe(img_gray, clahe_clip, clahe_grid)
    if gray_clahe.shape != (img_size, img_size):
        # The regressor was trained on 256x256 inputs; resize for inference.
        import cv2
        gray_clahe = cv2.resize(gray_clahe, (img_size, img_size),
                                 interpolation=cv2.INTER_LINEAR)

    f = gray_clahe.astype(np.float32) / 255.0
    t = torch.from_numpy(np.stack([f, f, f], axis=0))
    norm = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    t = norm(t).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_norm = model(t).squeeze(0).cpu()
    cap_x = round(float(pred_norm[0]) * (img_size - 1))
    cap_y = round(float(pred_norm[1]) * (img_size - 1))
    return cap_x, cap_y
