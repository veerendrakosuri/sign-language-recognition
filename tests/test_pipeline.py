"""Train/evaluate on fabricated landmarks: plumbing check, never an ASL benchmark."""
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


def test_synthetic_end_to_end(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    labels = tmp_path / "labels.json"
    labels.write_text(json.dumps(["synthetic_left", "synthetic_right"]))
    rows = []
    rng = np.random.default_rng(7)
    for split, count in [("train", 6), ("val", 2), ("test", 2)]:
        for label in range(2):
            for j in range(count):
                vid = f"{split}_{label}_{j}"
                p = np.zeros((10, 75, 3), np.float32)
                m = np.zeros((10, 75), bool)
                p[:, 11], p[:, 12] = [.3, .4, 0], [.7, .4, 0]
                p[:, 33:54] = [.2 + label * .5, .3, .01]
                p[:, 33:54] += rng.normal(0, .005, (10, 21, 3))
                m[:, [11, 12]], m[:, 33:54] = True, True
                np.savez(cache / f"{vid}.npz", points=p, valid=m)
                rows.append(dict(video_id=vid, label=label, split=split))
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    common = ["--manifest", str(manifest), "--cache", str(cache), "--labels", str(labels)]
    def run(*args):
        subprocess.run([sys.executable, "-m", "signlang.cli", *args], check=True, capture_output=True, text=True, timeout=120)
    run("baseline", *common, "--output", str(tmp_path / "baseline"))
    run("train", *common, "--output", str(tmp_path / "gru"), "--epochs", "2", "--hidden", "8", "--batch-size", "4")
    for kind, name in [("baseline", "baseline/baseline.joblib"), ("gru", "gru/best.pt")]:
        out = tmp_path / f"eval_{kind}"
        trust = ["--trust-checkpoint"] if kind == "baseline" else []
        run("evaluate", *common, "--kind", kind, "--checkpoint", str(tmp_path / name), "--output", str(out), *trust)
        result = json.loads((out / "metrics.json").read_text())
        assert result["samples"] == 4 and result["coverage"] == 1
        assert (out / "confusion_matrix.png").stat().st_size > 100
        from signlang.demo import load_demo_model, predict_window
        demo_model, pack = load_demo_model(kind, tmp_path / name)
        with np.load(cache / "test_0_0.npz", allow_pickle=False) as z:
            demo_label, demo_confidence = predict_window(kind, demo_model, pack,
                                                         z["points"], z["valid"])
        with (out / "predictions.csv").open(newline="", encoding="utf-8") as f:
            expected = next(row for row in csv.DictReader(f) if row["video_id"] == "test_0_0")
        assert demo_label == expected["predicted_gloss"]
        assert abs(demo_confidence - float(expected["confidence"])) < 1e-6
        from signlang import SignRecognizer, export_baseline
        recognizer = SignRecognizer.from_model(tmp_path / name, trusted=(kind == "baseline"))
        with np.load(cache / "test_0_0.npz", allow_pickle=False) as z:
            result = recognizer.predict_landmarks(z["points"], z["valid"])
        assert result.predictions[0].label == expected["predicted_gloss"]
        assert abs(result.predictions[0].score - float(expected["confidence"])) < 1e-6
        if kind == "baseline":
            exported = export_baseline(tmp_path / name, tmp_path / "binary-model.json", trusted=True)
            numeric = SignRecognizer.from_model(exported)
            with np.load(cache / "test_0_0.npz", allow_pickle=False) as z:
                exported_result = numeric.predict_landmarks(z["points"], z["valid"])
            assert exported_result.predictions[0].label == result.predictions[0].label
            assert abs(exported_result.predictions[0].score - result.predictions[0].score) < 1e-6
