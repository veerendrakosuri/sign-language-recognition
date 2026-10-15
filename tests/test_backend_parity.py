"""Numerical compatibility of synthetic sklearn and JSON backends."""
import json

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from signlang import SignRecognizer
from signlang.features import prepare
from test_api import frames
from test_cli import run_cli


def test_multiclass_export_cli_preserves_probabilities(tmp_path):
    raw, features, y = [], [], []
    for label in range(3):
        for shift in (-.02, 0, .02):
            points, valid = frames()
            points[:, 33:54, 0] += label * .15 + shift
            raw.append((points, valid))
            features.append(prepare(points, valid).mean(axis=0))
            y.append(label)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=1, max_iter=500))
    model.fit(features, y)
    checkpoint = tmp_path / "baseline.joblib"
    joblib.dump(dict(model=model, labels=["fabricated_a", "fabricated_b", "fabricated_c"], frames=32), checkpoint)
    exported = tmp_path / "model.json"
    run = run_cli("export-model", "--checkpoint", checkpoint, "--output", exported, "--trust-checkpoint")
    assert run.returncode == 0, run.stderr
    repeat = run_cli("export-model", "--checkpoint", checkpoint, "--output", exported, "--trust-checkpoint")
    assert repeat.returncode == 1 and "Traceback" not in repeat.stderr
    assert json.loads(exported.read_text())["checkpoint_sha256"]
    recognizer = SignRecognizer.from_model(exported)
    for (points, valid), feature in zip(raw, features):
        result = recognizer.predict_landmarks(points, valid, top_k=3)
        scores = {p.label: p.score for p in result.predictions}
        expected = model.predict_proba(feature[None])[0]
        assert [scores[label] for label in recognizer.labels] == pytest.approx(expected, abs=1e-6)


def test_corrupt_gru_checkpoint_has_a_clear_cli_error(tmp_path):
    source = tmp_path / "clip.npz"
    points, valid = frames()
    np.savez(source, points=points, valid=valid)
    checkpoint = tmp_path / "broken.pt"
    checkpoint.write_bytes(b"not a PyTorch checkpoint")
    run = run_cli("predict", source, "--input-type", "landmarks", "--checkpoint", checkpoint)
    assert run.returncode == 1 and "Invalid GRU" in run.stderr and "Traceback" not in run.stderr
