"""Grad-CAM heatmap generation for ResNet-18 classifiers.

Provides a clean API for the pipeline to generate Grad-CAM overlays
without accumulating hooks on cached models.
"""
from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image


class _GradCAM:
    """Lightweight Grad-CAM that registers removable hooks."""

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.gradients = None
        self.activations = None
        self._fwd = target_layer.register_forward_hook(self._save_act)
        self._bwd = target_layer.register_full_backward_hook(self._save_grad)

    def _save_act(self, m, inp, out):
        self.activations = out.detach()

    def _save_grad(self, m, gi, go):
        self.gradients = go[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: int | None = None) -> np.ndarray:
        self.model.eval()
        with torch.enable_grad():
            x = x.clone().requires_grad_(True)
            out = self.model(x)
            if class_idx is None:
                class_idx = out.argmax(1).item()
            self.model.zero_grad()
            out[0, class_idx].backward()
        w = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.activations).sum(1, keepdim=True))
        cam = F.interpolate(cam, x.shape[2:], mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def remove(self):
        self._fwd.remove()
        self._bwd.remove()


def compute_gradcam(
    model: torch.nn.Module,
    pil_image: Image.Image,
    transform: T.Compose,
    device: str = "cpu",
    class_idx: int | None = None,
    input_size: int = 224,
) -> dict:
    """Generate a Grad-CAM heatmap for a single PIL image.

    Returns a dict with:
        heatmap  – (H, W) float32 in [0, 1], resized to input_size×input_size
        overlay  – (H, W, 3) uint8 blended JET colour-map on image
    """
    backbone = model.model if hasattr(model, "model") else model
    target_layer = backbone.layer4[-1]

    cam = _GradCAM(model, target_layer)
    try:
        tensor = transform(pil_image).unsqueeze(0).to(device)
        heatmap = cam(tensor, class_idx=class_idx)
    finally:
        cam.remove()

    # Build overlay at original image resolution
    img_np = np.array(pil_image)
    if img_np.ndim == 2:
        img_np = np.stack([img_np] * 3, axis=-1)
    hm = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
    hm_color = cv2.applyColorMap((hm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB)
    overlay = (0.45 * hm_color + 0.55 * img_np).astype(np.uint8)

    return {"heatmap": hm, "overlay": overlay, "original": img_np}
