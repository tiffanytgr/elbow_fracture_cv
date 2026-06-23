"""elbow_grader — Gartland classification for paediatric elbow fractures.

Public API
----------
ElbowGrader
    Main entry point. Load once, call predict() repeatedly.

GraderConfig
    User-facing settings (confidence threshold, alignment toggle, etc.).

GraderResult
    Structured output with grade properties and built-in plot methods.

Quick start
-----------
>>> from elbow_grader import ElbowGrader, GraderConfig
>>> grader = ElbowGrader.from_checkpoints("experiments/checkpoints")
>>> result = grader.predict(ap_image="A102_AP.png", lat_image="A102_LAT.png")
>>> print(result.final_grade)          # e.g. 'Grade 2a'
>>> result.plot_gradcam().savefig("gradcam.png")
>>> result.to_json("audit.json")

CLI
---
After `pip install -e .`:
    elbow-grader predict --ap AP.png --lat LAT.png --save-plots --output ./results
"""

from .config import GraderConfig
from .grader import ElbowGrader
from .results import GraderResult

__all__ = ["ElbowGrader", "GraderConfig", "GraderResult"]
__version__ = "0.1.0"
