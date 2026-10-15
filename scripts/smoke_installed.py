"""Run outside the source checkout with the freshly installed wheel's Python."""
import importlib.metadata
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import sysconfig
import tempfile

import numpy as np

import signlang
from signlang import SignRecognizer


def main():
    assert importlib.metadata.version("sign-language-recognition") == signlang.__version__
    assert "site-packages" in str(Path(signlang.__file__).resolve()), signlang.__file__
    for optional in ("torch", "sklearn", "cv2", "mediapipe"):
        assert importlib.util.find_spec(optional) is None, f"Core install unexpectedly includes {optional}"
    console = Path(sysconfig.get_path("scripts")) / ("signlang.exe" if sys.platform == "win32" else "signlang")
    subprocess.run([str(console), "--version"], check=True)
    with tempfile.TemporaryDirectory(dir=Path.cwd()) as folder:
        root = Path(folder)
        model = root / "model.json"
        model.write_text(json.dumps(dict(schema=1, labels=["fabricated_a", "fabricated_b"], frames=32,
            mean=[0.] * 225, scale=[1.] * 225, weights=[[0.] * 225] * 2, bias=[0., 1.])))
        points, valid = np.zeros((4, 75, 3), np.float32), np.zeros((4, 75), bool)
        points[:, 11], points[:, 12] = [.3, .4, .1], [.7, .4, .1]
        points[:, 33:54] = [.2, .3, .01]
        valid[:, [11, 12]], valid[:, 33:54] = True, True
        cache = root / "synthetic.npz"
        np.savez(cache, points=points, valid=valid)
        expected = SignRecognizer.from_model(model).predict_landmarks(points, valid).to_dict()
        run = subprocess.run([str(console), "predict", str(cache), "--input-type", "landmarks",
                    "--checkpoint", str(model)], check=True, capture_output=True, text=True)
        assert json.loads(run.stdout)["predictions"] == expected["predictions"]
        missing = subprocess.run([str(console), "predict", str(root / "missing.mp4"),
                    "--checkpoint", str(model)], capture_output=True, text=True)
        assert missing.returncode == 1 and missing.stdout == "" and "Traceback" not in missing.stderr
        video = root / "synthetic-empty.mp4"
        video.touch()
        dependency = subprocess.run([str(console), "predict", str(video), "--checkpoint", str(model)],
                                   capture_output=True, text=True)
        assert dependency.returncode == 1 and "[vision]" in dependency.stderr
    print(f"Installed core API, CLI, and optional-dependency errors passed using synthetic inputs: {signlang.__file__}")


if __name__ == "__main__":
    main()
