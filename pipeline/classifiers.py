"""ResNet-18 binary classifiers for Exp 1/2/3/4.

Loads checkpoints saved either as plain state_dicts or wrapped in the
TransferResNet wrapper from training (`model.fc.1.weight` etc.) and
returns a uniform classify(image_pil) -> (pred_idx, probs).
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


def build_model(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_classifier(checkpoint_path: Path,
                     num_classes: int = 2,
                     device: str = "cpu") -> nn.Module:
    """Load a ResNet-18 binary classifier from a checkpoint.

    Handles two checkpoint formats:
      1. Plain state_dict for a bare ResNet-18 (`fc.weight`, `fc.bias`)
      2. TransferResNet-wrapped (`model.layer1...`, `model.fc.1.weight`)
    """
    model = build_model(num_classes)
    state = torch.load(checkpoint_path, map_location=device, weights_only=True)

    first_key = next(iter(state))
    if first_key.startswith("model."):
        new_state = {}
        for k, v in state.items():
            new_key = k[len("model."):] if k.startswith("model.") else k
            new_key = new_key.replace("fc.1.", "fc.")
            new_state[new_key] = v
        state = new_state

    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def make_transform(input_size: int = 224,
                    mean: tuple = (0.485, 0.456, 0.406),
                    std:  tuple = (0.229, 0.224, 0.225)) -> T.Compose:
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(list(mean), list(std)),
    ])


def classify(model: nn.Module,
              image_pil: Image.Image,
              transform: T.Compose,
              device: str = "cpu") -> tuple[int, np.ndarray]:
    """Run a single image through the classifier."""
    tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return int(probs.argmax()), probs
