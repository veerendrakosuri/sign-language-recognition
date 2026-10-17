"""Export supported scikit-learn baselines without pickle in the output."""
import json
from pathlib import Path

import numpy as np

from .api import SignRecognizer, _dependency


def export_baseline(checkpoint, output, *, trusted=False):
    """Export a trusted StandardScaler + LogisticRegression checkpoint as JSON.

    Supports binary logistic regression and multinomial models. Refuses other
    pipelines so the exported numerical model cannot silently change behavior.
    The destination must not already exist.
    """
    recognizer = SignRecognizer.from_model(checkpoint, kind="baseline", trusted=trusted)
    linear = _dependency("sklearn.linear_model", "baseline")
    preprocessing = _dependency("sklearn.preprocessing", "baseline")
    pipeline = recognizer._model
    steps = getattr(pipeline, "steps", [])
    if (len(steps) != 2 or not isinstance(steps[0][1], preprocessing.StandardScaler)
            or not isinstance(steps[1][1], linear.LogisticRegression)):
        raise ValueError("Export requires StandardScaler followed by LogisticRegression")
    scaler, classifier = steps[0][1], steps[1][1]
    if not scaler.with_mean or not scaler.with_std:
        raise ValueError("Export requires both StandardScaler centering and scaling")
    if not np.array_equal(classifier.classes_, np.arange(len(recognizer.labels))):
        raise ValueError("Export requires every label in checkpoint order")
    if len(recognizer.labels) > 2 and (classifier.solver == "liblinear" or getattr(classifier, "multi_class", None) == "ovr"):
        raise ValueError("Multiclass export requires multinomial logistic regression")
    weights, bias = classifier.coef_, classifier.intercept_
    if (weights.shape[1:] != (225,) or scaler.mean_.shape != (225,)
            or not all(np.isfinite(x).all() for x in (weights, bias, scaler.mean_, scaler.scale_))):
        raise ValueError("Expected 225 finite landmark features")
    payload = dict(schema=1, name="Exported landmark logistic baseline",
                   checkpoint_sha256=recognizer._sha256, frames=recognizer.frames,
                   summary=recognizer.summary,
                   labels=list(recognizer.labels), mean=scaler.mean_.tolist(),
                   scale=scaler.scale_.tolist(), weights=weights.tolist(), bias=bias.tolist())
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as destination:
        destination.write(json.dumps(payload, allow_nan=False, separators=(",", ":")) + "\n")
    return output
