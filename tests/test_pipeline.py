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
        run("evaluate", *common, "--kind", kind, "--checkpoint", str(tmp_path / name), "--output", str(out))
        result = json.loads((out / "metrics.json").read_text())
        assert result["samples"] == 4 and result["coverage"] == 1
        assert (out / "confusion_matrix.png").stat().st_size > 100
