"""End-to-end 5-class Gartland baseline: the comparison arm for the cascade.

Given a paired AP and LAT radiograph, predict Normal / Grade 1 / Grade 2A /
Grade 2B / Grade 3 directly, with no hierarchical routing, and compare against
the cascade on the same held-out patients.

Modules
-------
labels      canonical 5-class vocabulary and ordinal helpers
manifest    build a paired AP+LAT manifest, reusing the cascade's splits
dataset     paired dataset with the cascade's augmentation recipe
model       dual-encoder ResNet-18 (and a two-channel appendix variant)
train       two-stage transfer learning, matching report section 2.7.1
evaluate    5x5 confusion matrix, per-grade CIs, QWK, boundary errors
compare     paired cascade-vs-multiclass tests (McNemar, kappa difference)
metrics     shared metric implementations; imports without torch

``metrics`` and ``labels`` are torch-free so the figure scripts can use them on
a machine with no GPU stack.
"""
from __future__ import annotations

from .labels import GRADES, NUM_CLASSES, SHORT_GRADES

__all__ = ["GRADES", "NUM_CLASSES", "SHORT_GRADES"]
