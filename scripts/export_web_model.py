"""Export the locally trained baseline as browser-readable numbers.

Only load a checkpoint you created or trust: joblib uses pickle internally.
The exported JSON contains model weights, never training videos or landmarks.
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np

from signlang.data import digest


def export(checkpoint: Path, output: Path):
    pack = joblib.load(checkpoint)
    pipeline = pack["model"]
    scaler = pipeline.named_steps["standardscaler"]
    classifier = pipeline.named_steps["logisticregression"]
    labels = pack["labels"]
    if pack["frames"] != 32 or len(labels) != 100:
        raise ValueError("This web app expects the evaluated 32-frame WLASL100 baseline")
    if not np.array_equal(classifier.classes_, np.arange(len(labels))):
        raise ValueError("Classifier classes must match the WLASL label order")
    if classifier.coef_.shape != (len(labels), 225):
        raise ValueError("Unexpected baseline feature dimensions")
    result = {
        "schema": 1,
        "name": "WLASL100 available-mirror baseline",
        "checkpoint_sha256": digest(checkpoint),
        "frames": pack["frames"],
        "labels": labels,
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "weights": classifier.coef_.tolist(),
        "bias": classifier.intercept_.tolist(),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, separators=(",", ":")), encoding="utf-8")
    print(f"Exported {len(labels)} classes to {output} from {result['checkpoint_sha256']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=Path("runs/baseline-v1/baseline.joblib"))
    parser.add_argument("--output", type=Path, default=Path("web/public/model.json"))
    args = parser.parse_args()
    export(args.checkpoint, args.output)
