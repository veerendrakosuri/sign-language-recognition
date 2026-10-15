"""Synthetic contract tests; these do not measure recognition accuracy."""
import json

import numpy as np
import pytest

from signlang import SignRecognizer


def frames():
    points = np.zeros((8, 75, 3), np.float32)
    valid = np.zeros((8, 75), bool)
    points[:, 11], points[:, 12] = [.3, .4, .1], [.7, .4, .1]
    points[:, 33:54] = [.2, .3, .01]
    valid[:, [11, 12]], valid[:, 33:54] = True, True
    return points, valid


def write_model(path, **updates):
    weights = np.zeros((2, 225))
    weights[0, 99] = 1
    value = dict(schema=1, frames=32, labels=["synthetic_left", "synthetic_right"],
                 mean=[0.] * 225, scale=[1.] * 225,
                 weights=weights.tolist(), bias=[0., 0.])
    value.update(updates)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_json_prediction_is_ranked_serializable_and_identifies_preprocessing(tmp_path):
    recognizer = SignRecognizer.from_model(write_model(tmp_path / "model.json"))
    result = recognizer.predict_landmarks(*frames(), top_k=2)
    payload = result.to_dict()
    assert payload["schema_version"] == 1
    assert payload["predictions"][0]["label"] == "synthetic_right"
    assert payload["predictions"][0]["score"] == pytest.approx(1 / (1 + np.exp(-.75)))
    assert sum(p["score"] for p in payload["predictions"]) == pytest.approx(1)
    assert payload["model"]["kind"] == "baseline-json"
    assert len(payload["model"]["sha256"]) == 64
    assert payload["model"]["preprocessing"] == "pose33-hands42-shoulders-v1"
    assert payload["input"]["frames"] == 8
    assert payload["tracking"]["hand_frame_fraction"] == 1
    json.dumps(payload, allow_nan=False)


@pytest.mark.parametrize("updates", [
    {"schema": 2}, {"frames": 1}, {"labels": ["same", "same"]},
    {"scale": [0.] * 225}, {"weights": [[0.]]},
    {"mean": [float("nan")] * 225},
])
def test_invalid_json_models_are_rejected(tmp_path, updates):
    with pytest.raises(ValueError):
        SignRecognizer.from_model(write_model(tmp_path / "model.json", **updates))


def test_binary_export_uses_sigmoid_not_one_class_softmax(tmp_path):
    path = write_model(tmp_path / "model.json", weights=[[0.] * 225], bias=[2.])
    result = SignRecognizer.from_model(path).predict_landmarks(*frames())
    assert result.predictions[0].label == "synthetic_right"
    assert result.predictions[0].score == pytest.approx(1 / (1 + np.exp(-2)))


def test_input_validation_and_weak_tracking_warning(tmp_path):
    recognizer = SignRecognizer.from_model(write_model(tmp_path / "model.json"))
    points, valid = frames()
    with pytest.raises(ValueError, match="top_k"):
        recognizer.predict_landmarks(points, valid, top_k=0)
    with pytest.raises(ValueError, match="mask"):
        recognizer.predict_landmarks(points, valid[:, :74])
    with pytest.raises(ValueError, match="finite"):
        recognizer.predict_landmarks(points * np.nan, valid)
    valid[1:, 33:] = False
    result = recognizer.predict_landmarks(points, valid)
    assert result.tracking["hand_frame_fraction"] == .125
    assert result.warnings
    valid[:, 33:] = False
    with pytest.raises(ValueError, match="hands"):
        recognizer.predict_landmarks(points, valid)


def test_pickle_checkpoint_requires_explicit_trust(tmp_path):
    path = tmp_path / "untrusted.joblib"
    path.write_bytes(b"not a checkpoint")
    with pytest.raises(ValueError, match="trust"):
        SignRecognizer.from_model(path)


def test_missing_video_fails_before_loading_optional_detectors(tmp_path):
    recognizer = SignRecognizer.from_model(write_model(tmp_path / "model.json"))
    with pytest.raises(FileNotFoundError):
        recognizer.predict_video(tmp_path / "missing.mp4")
