"""CLI stdout is a machine-readable result; errors go to stderr."""
import json
import subprocess
import sys

import numpy as np

from test_api import frames, write_model


def run_cli(*args):
    return subprocess.run([sys.executable, "-m", "signlang", *map(str, args)],
                          capture_output=True, text=True, timeout=30)


def test_version_and_help_do_not_import_optional_dependencies():
    result = run_cli("--version")
    assert result.returncode == 0 and "0.2.1" in result.stdout
    assert run_cli("predict", "--help").returncode == 0
    code = "import signlang, sys; assert not {'torch','cv2','mediapipe','sklearn'} & sys.modules.keys()"
    subprocess.run([sys.executable, "-c", code], check=True, timeout=30)


def test_predict_landmarks_json_stdout_and_file(tmp_path):
    model = write_model(tmp_path / "model.json")
    source = tmp_path / "clip.npz"
    points, valid = frames()
    np.savez(source, points=points, valid=valid)
    arguments = ("predict", source, "--checkpoint", model, "--input-type", "landmarks")
    result = run_cli(*arguments)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["predictions"][0]["label"] == "synthetic_right"
    output = tmp_path / "nested" / "result.json"
    result = run_cli(*arguments, "--output", output)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
    assert json.loads(output.read_text())["input"]["name"] == "clip.npz"


def test_failures_use_nonzero_status_and_stderr(tmp_path):
    result = run_cli("predict")
    assert result.returncode == 2
    model = write_model(tmp_path / "model.json")
    result = run_cli("predict", tmp_path / "missing.mp4", "--checkpoint", model)
    assert result.returncode == 1
    assert result.stdout == "" and "error:" in result.stderr and "Traceback" not in result.stderr


def test_bad_landmark_container_and_existing_output_are_preserved(tmp_path):
    model = write_model(tmp_path / "model.json")
    source = tmp_path / "wrong.npy"
    np.save(source, np.zeros((2, 75, 3)))
    result = run_cli("predict", source, "--input-type", "landmarks", "--checkpoint", model)
    assert result.returncode == 1 and "NPZ" in result.stderr and "Traceback" not in result.stderr
    output = tmp_path / "existing.json"
    output.write_text("keep this")
    result = run_cli("predict", source, "--checkpoint", model, "--output", output)
    assert result.returncode == 1 and output.read_text() == "keep this"


def test_legacy_baseline_commands_require_trust_before_loading_backends():
    for arguments in [("evaluate", "--kind", "baseline", "--checkpoint", "untrusted.joblib"),
                      ("webcam", "--kind", "baseline", "--checkpoint", "untrusted.joblib")]:
        result = run_cli(*arguments)
        assert result.returncode == 1 and "--trust-checkpoint" in result.stderr
