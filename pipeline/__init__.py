"""KKH pediatric elbow fracture grading pipeline.

Public API
----------
predict(ap_path=None, lat_path=None, *, config=None) -> PredictionResult
    Run the full grading cascade on an AP and/or LAT image and return a
    structured result with per-experiment predictions, probabilities, OOD
    flags, and (LAT only) SAM2 explainability outputs.

Example
-------
>>> from pipeline import predict
>>> result = predict(ap_path="A102 AP.png", lat_path="A102 LAT.png")
>>> print(result.final_grade)
'Grade 2a'
>>> print(result.cnn_track)
{'exp1': ..., 'exp2': ..., 'exp3': ..., 'exp4': ...}

The module is designed to be imported by:
- A Streamlit/Gradio demo (each call returns one PredictionResult)
- A batch evaluation script (for prospective study analysis)
- A FastAPI service (PredictionResult is JSON-serialisable via .to_dict())
"""

from .config import PipelineConfig, default_config
from .pipeline import predict, PredictionResult, BaumannResult

__all__ = ["predict", "PredictionResult", "BaumannResult", "PipelineConfig", "default_config"]
