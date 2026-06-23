"""DRUE OOD inference.

The DRUE training notebooks (drue_ood_filter_resnet_exp*.ipynb) train two
reconstruction decoders attached to a frozen ResNet-18 backbone. The
saved file `drue_decoders.pth` contains:
    - decoder_shallow: layer1 (64x56x56 -> 224x224) reconstruction
    - decoder_deep   : layer4 (512x7x7  -> 224x224) reconstruction
    - loss_history   : training loss curve (not needed for inference)

Uncertainty score = mean |shallow_recon - deep_recon|.
Higher = more out-of-distribution.

Thresholds are hardcoded in PipelineConfig (one float per experiment).
At inference, an image is OOD if its DRUE score exceeds that threshold.
If the threshold is None, scores are reported but no rejection happens.
See PipelineConfig docstring for how to derive the threshold from
drue_scores.json after training.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image


# ── Architectures (copied verbatim from drue_ood_filter_resnet_exp1_ap.ipynb) ──

class _FeatureExtractor:
    def __init__(self, resnet_model: nn.Module):
        self.features = {}
        self._hooks = []
        for name, layer in [("shallow", resnet_model.layer1),
                            ("deep",    resnet_model.layer4)]:
            self._hooks.append(
                layer.register_forward_hook(
                    lambda mod, inp, out, n=name: self.features.__setitem__(n, out)
                )
            )

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()


class _ReconstructionDecoder(nn.Module):
    def __init__(self, in_channels: int, in_spatial: int, target_size: int = 224):
        super().__init__()
        layers, ch, s, n_up = [], in_channels, in_spatial, 0
        while s < target_size:
            s *= 2; n_up += 1
        for i in range(n_up):
            out_ch = max(ch // 2, 32)
            if i == n_up - 1:
                layers += [nn.ConvTranspose2d(ch, 1, 4, 2, 1), nn.Sigmoid()]
            else:
                layers += [nn.ConvTranspose2d(ch, out_ch, 4, 2, 1),
                           nn.BatchNorm2d(out_ch), nn.ReLU(True)]
            ch = out_ch
        self.decoder = nn.Sequential(*layers)
        self.target_size = target_size

    def forward(self, x):
        out = self.decoder(x)
        if out.shape[-1] != self.target_size:
            out = F.interpolate(out, (self.target_size, self.target_size),
                                mode="bilinear", align_corners=False)
        return out


class DRUEModel(nn.Module):
    def __init__(self, img_size: int = 224):
        super().__init__()
        # ImageNet-pretrained ResNet-18 — at inference time we don't need
        # the experiment-specific classifier weights because the decoders
        # were trained against whatever backbone the training notebook used.
        # However, for maximum fidelity, the caller should pass an explicit
        # classifier checkpoint via load_classifier_into_backbone().
        bb = models.resnet18(weights="IMAGENET1K_V1")
        bb.fc = nn.Linear(bb.fc.in_features, 2)
        for p in bb.parameters():
            p.requires_grad = False
        bb.eval()
        self.backbone = bb
        self.feat_extractor = _FeatureExtractor(self.backbone)
        self.decoder_shallow = _ReconstructionDecoder(64, 56, img_size)
        self.decoder_deep    = _ReconstructionDecoder(512, 7, img_size)
        self.img_size = img_size

    def load_classifier_into_backbone(self, classifier_ckpt: Path, device: str = "cpu"):
        """Load the experiment-specific classifier weights into the backbone.

        Match what the training notebook did so reconstruction quality is
        consistent. If the file doesn't exist, fall back to ImageNet (with a
        printed warning).
        """
        if not Path(classifier_ckpt).exists():
            print(f"[DRUE] WARN: classifier ckpt {classifier_ckpt} missing — "
                  "using ImageNet backbone (OOD scores will differ slightly).")
            return
        state = torch.load(classifier_ckpt, map_location=device, weights_only=False)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        # The training script wraps in TransferResNet, so keys may be `model.layer1...`
        new_state = {}
        for k, v in state.items():
            nk = k[len("model."):] if k.startswith("model.") else k
            nk = nk.replace("fc.1.", "fc.")
            new_state[nk] = v
        self.backbone.load_state_dict(new_state, strict=False)
        self.backbone.eval()

    def load_decoders(self, decoders_ckpt: Path, device: str = "cpu"):
        ckpt = torch.load(decoders_ckpt, map_location=device, weights_only=True)
        self.decoder_shallow.load_state_dict(ckpt["decoder_shallow"])
        self.decoder_deep.load_state_dict(ckpt["decoder_deep"])
        self.decoder_shallow.eval()
        self.decoder_deep.eval()

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            _ = self.backbone(x)
        feat_s = self.feat_extractor.features["shallow"]
        feat_d = self.feat_extractor.features["deep"]
        return self.decoder_shallow(feat_s), self.decoder_deep(feat_d)

    def compute_uncertainty(self, x: torch.Tensor) -> float:
        """Return scalar uncertainty score for a single-image batch."""
        rs, rd = self.forward(x)
        return float(torch.abs(rs - rd).mean(dim=(1, 2, 3)).item())


# ── Public API ───────────────────────────────────────────────────────

def make_transform(img_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class DRUEScorer:
    """Loaded once per experiment, scores any number of images.

    Pass a float `threshold` to enable rejection; pass None to score-only.
    """

    def __init__(self, decoders_ckpt: Path,
                 classifier_ckpt: Optional[Path] = None,
                 img_size: int = 224,
                 device: str = "cpu",
                 threshold: Optional[float] = None):
        self.device = device
        self.transform = make_transform(img_size)
        self.model = DRUEModel(img_size).to(device)
        if classifier_ckpt is not None:
            self.model.load_classifier_into_backbone(classifier_ckpt, device)
        self.model.load_decoders(decoders_ckpt, device)
        self.threshold = float(threshold) if threshold is not None else None

    def score(self, image_pil: Image.Image) -> float:
        x = self.transform(image_pil).unsqueeze(0).to(self.device)
        return self.model.compute_uncertainty(x)

    def is_ood(self, image_pil: Image.Image) -> tuple[bool, float, Optional[float]]:
        """Returns (is_ood, score, threshold_used).

        If threshold is None, returns (False, score, None) — caller can
        still log the score for offline analysis.
        """
        s = self.score(image_pil)
        if self.threshold is None:
            return False, s, None
        return s > self.threshold, s, self.threshold
