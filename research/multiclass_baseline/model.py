"""Dual-view ResNet-18 for end-to-end 5-class Gartland classification.

Architecture (the comparison baseline the reviewer asked for):

    AP  -> ResNet-18 encoder -> GAP -> 512
                                        \
                                         concat 1024 -> dropout -> Linear -> 5 logits
                                        /
    LAT -> ResNet-18 encoder -> GAP -> 512

Two separate encoders rather than one shared encoder: AP and LAT are different
projections with different diagnostic content (the cascade itself routes Nodes
1-2 to AP and Nodes 3-4 to LAT), so weight sharing would force one filter bank
to serve both. Both encoders start from the same autoencoder-pretrained
weights, so the baseline gets exactly the initialisation advantage the cascade
nodes get -- otherwise a weaker baseline would flatter the cascade.

A two-channel single-encoder variant is included for the appendix; it is
cheaper but mixes the views before any view-specific feature extraction.
"""
from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn
from torchvision import models

from .labels import NUM_CLASSES

log = logging.getLogger(__name__)

FEATURE_DIM = 512  # ResNet-18 final stage


def _strip_encoder_state(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Normalise the several checkpoint layouts this project has produced.

    Handles: plain ResNet-18 state dicts; the TransferResNet wrapper used by the
    cascade training notebooks (``model.layer1...``, ``model.fc.1.weight``);
    autoencoder checkpoints that namespace the encoder (``encoder.``); and
    checkpoints saved as ``{"state_dict": ...}`` or ``{"model_state_dict": ...}``.
    """
    for key in ("state_dict", "model_state_dict", "encoder_state_dict"):
        if key in state and isinstance(state[key], dict):
            state = state[key]
            break

    cleaned: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        new_key = key
        for prefix in ("module.", "model.", "encoder.", "backbone."):
            new_key = new_key.removeprefix(prefix)
        # The wrapper stores the head as fc.1 (after a dropout at fc.0).
        new_key = new_key.replace("fc.1.", "fc.")
        cleaned[new_key] = value
    return cleaned


def load_pretrained_encoder(
    checkpoint: Path | None,
    imagenet_fallback: bool = True,
) -> models.ResNet:
    """Build a ResNet-18 trunk, initialised from the project's AE encoder.

    The final ``fc`` is replaced with Identity: this object is a feature
    extractor, and the classification head lives in DualViewResNet18.
    Classifier-head weights in the checkpoint are reported and discarded.
    """
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if imagenet_fallback else None
    net = models.resnet18(weights=weights)

    if checkpoint is not None:
        path = Path(checkpoint)
        if not path.exists():
            raise FileNotFoundError(f"encoder checkpoint not found: {path}")
        raw = torch.load(path, map_location="cpu", weights_only=True)
        state = _strip_encoder_state(raw)
        # Drop head weights; their shape belongs to a different task.
        state = {k: v for k, v in state.items() if not k.startswith("fc.")}
        result = net.load_state_dict(state, strict=False)
        loaded = len(state)
        unexpected = [k for k in result.unexpected_keys]
        missing = [k for k in result.missing_keys if not k.startswith("fc.")]
        log.info("loaded %d encoder tensors from %s", loaded, path.name)
        if missing:
            log.warning("encoder tensors left at initialisation (%d): %s",
                        len(missing), ", ".join(missing[:8]))
        if unexpected:
            log.warning("checkpoint tensors ignored (%d): %s",
                        len(unexpected), ", ".join(unexpected[:8]))
        if not loaded or len(missing) > 20:
            raise ValueError(
                f"{path} does not look like a ResNet-18 encoder checkpoint "
                f"({loaded} tensors matched, {len(missing)} missing)"
            )
    elif imagenet_fallback:
        log.info("no encoder checkpoint given -- starting from ImageNet weights")

    net.fc = nn.Identity()
    return net


class DualViewResNet18(nn.Module):
    """Two ResNet-18 encoders (AP, LAT) with a shared 5-way head."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        encoder_checkpoint: Path | None = None,
        dropout: float = 0.3,
        imagenet_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.ap_encoder = load_pretrained_encoder(encoder_checkpoint, imagenet_fallback)
        self.lat_encoder = load_pretrained_encoder(encoder_checkpoint, imagenet_fallback)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(FEATURE_DIM * 2, num_classes)

    def features(self, ap: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.ap_encoder(ap), self.lat_encoder(lat)], dim=1)

    def forward(self, ap: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.dropout(self.features(ap, lat)))

    # --- parameter groups for the two-stage schedule -------------------------

    def backbone_parameters(self):
        yield from self.ap_encoder.parameters()
        yield from self.lat_encoder.parameters()

    def head_parameters(self):
        yield from self.classifier.parameters()

    def set_backbone_trainable(self, trainable: bool) -> None:
        """Freeze or unfreeze both encoders.

        When frozen, batch-norm layers are also pinned to eval mode by
        ``train()`` below, so the running statistics inherited from pretraining
        are not overwritten by small-batch updates during stage 1.
        """
        for param in self.backbone_parameters():
            param.requires_grad = trainable
        self._backbone_frozen = not trainable

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and getattr(self, "_backbone_frozen", False):
            for module in (self.ap_encoder, self.lat_encoder):
                for sub in module.modules():
                    if isinstance(sub, nn.modules.batchnorm._BatchNorm):
                        sub.eval()
        return self

    def gradcam_target_layers(self) -> dict[str, nn.Module]:
        """Last conv stage of each encoder, for the appendix Grad-CAM panel."""
        return {"AP": self.ap_encoder.layer4, "LAT": self.lat_encoder.layer4}


class TwoChannelResNet18(nn.Module):
    """Appendix variant: AP and LAT stacked as channels into one encoder.

    The first convolution is re-created with 2 input channels, seeded by
    averaging the pretrained RGB filters so the pretrained features are not
    discarded outright.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        encoder_checkpoint: Path | None = None,
        dropout: float = 0.3,
        imagenet_fallback: bool = True,
    ) -> None:
        super().__init__()
        self.encoder = load_pretrained_encoder(encoder_checkpoint, imagenet_fallback)
        old = self.encoder.conv1
        new = nn.Conv2d(2, old.out_channels, kernel_size=old.kernel_size,
                        stride=old.stride, padding=old.padding, bias=old.bias is not None)
        with torch.no_grad():
            seed = old.weight.mean(dim=1, keepdim=True)
            new.weight.copy_(seed.repeat(1, 2, 1, 1))
        self.encoder.conv1 = new
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(FEATURE_DIM, num_classes)

    def forward(self, ap: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        # Collapse each RGB-replicated view back to one grey channel.
        stacked = torch.cat([ap.mean(dim=1, keepdim=True), lat.mean(dim=1, keepdim=True)], dim=1)
        return self.classifier(self.dropout(self.encoder(stacked)))

    def backbone_parameters(self):
        for name, param in self.encoder.named_parameters():
            if not name.startswith("conv1"):  # conv1 is newly initialised
                yield param

    def head_parameters(self):
        yield from self.classifier.parameters()
        yield from self.encoder.conv1.parameters()

    def set_backbone_trainable(self, trainable: bool) -> None:
        for param in self.backbone_parameters():
            param.requires_grad = trainable
        self._backbone_frozen = not trainable

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and getattr(self, "_backbone_frozen", False):
            for sub in self.encoder.modules():
                if isinstance(sub, nn.modules.batchnorm._BatchNorm):
                    sub.eval()
        return self

    def gradcam_target_layers(self) -> dict[str, nn.Module]:
        return {"stacked": self.encoder.layer4}


ARCHITECTURES = {"dual": DualViewResNet18, "two-channel": TwoChannelResNet18}


def build_model(
    architecture: str = "dual",
    num_classes: int = NUM_CLASSES,
    encoder_checkpoint: Path | None = None,
    dropout: float = 0.3,
    imagenet_fallback: bool = True,
) -> nn.Module:
    try:
        cls = ARCHITECTURES[architecture]
    except KeyError as exc:
        raise ValueError(
            f"unknown architecture {architecture!r}; choose from {sorted(ARCHITECTURES)}"
        ) from exc
    model = cls(
        num_classes=num_classes,
        encoder_checkpoint=encoder_checkpoint,
        dropout=dropout,
        imagenet_fallback=imagenet_fallback,
    )
    model.set_backbone_trainable(True)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("built %s model: %.1fM parameters", architecture, n_params / 1e6)
    return model
