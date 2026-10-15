# Python package and CLI guide

Version 0.2.1 is an alpha developer toolkit for isolated ASL words. It exposes saved-video prediction, landmark extraction and audit, model export, and the existing training/evaluation workflows. Recognition remains experimental; package checks are not accuracy measurements.

## Install

Use Python 3.10–3.12 in a virtual environment. The release wheel has no bundled models or datasets. Install it from GitHub Releases:

```sh
python -m pip install "sign-language-recognition[vision] @ https://github.com/veerendrakosuri/sign-language-recognition/releases/download/v0.2.1/sign_language_recognition-0.2.1-py3-none-any.whl"
signlang --version
signlang --help
```

Alternatively, clone the repository, check out `v0.2.1`, and run `python -m pip install ".[vision]"` from its root. Package-registry publication is not claimed; do not assume a same-named PyPI project belongs to this repository.

| Install choice | Purpose |
|---|---|
| Core wheel or `pip install .` | NumPy-based JSON-model inference on cached landmarks |
| `.[vision]` | Add video decoding, MediaPipe extraction, and local camera dependencies |
| `.[baseline]` | Add scikit-learn/joblib for trusted baseline prediction or JSON export |
| `.[train]` | Add PyTorch, scikit-learn, and plotting for training and evaluation |
| `.[vision,train,dev]` | Complete development setup, tests, and distribution tools |

Replace `.` in these examples with the installed wheel URL/name when installing from a release. `python -m signlang` also works when the console command is not on PATH. Linux MediaPipe may need `libegl1` and `libgles2`; use a compatible PyTorch installation for your hardware. This release's CI matrix records platform verification; it does not claim universal hardware support.

## Supply a model

Model paths are explicit. Supported formats are a numeric baseline `.json`, trusted scikit-learn `.joblib`, and a GRU `.pt` checkpoint from this project. `--kind` can override suffix detection. GRU needs the `train` extra. Joblib needs the `baseline` extra and explicit trust because it can execute code.

Version 0.2.0 also requires `--trust-checkpoint` for baseline `evaluate` and `webcam` commands. This is a deliberate change from 0.1.0; pass it only for checkpoints you created or otherwise trust.

Export a baseline you trained yourself:

```sh
signlang export-model --checkpoint runs/baseline-v1/baseline.joblib --output my-model.json --trust-checkpoint
```

Export supports StandardScaler followed by binary or multinomial LogisticRegression. It refuses unsupported pipelines, existing output files, and incomplete class mappings. JSON schema 1 carries labels, frame count, scaler mean/scale, weights, bias, and the source checkpoint hash. JSON inference uses NumPy and does not deserialize Python objects.

The repository's existing `web/public/model.json` can be used for local research experiments subject to its separate terms. It is not bundled in the package. Its baseline scored 32/100 on available offline test clips from an incomplete WLASL100 mirror; it is not a complete benchmark or a live accuracy measurement.

## Predict a saved video

Install the `vision` extra and download the detectors:

```sh
signlang download-models --models models
signlang predict sign.mp4 --checkpoint my-model.json --models models --top-k 5
signlang predict sign.mp4 --checkpoint my-model.json --output result.json
```

Each video must contain one complete, pre-trimmed sign. The tool decodes all frames, normalizes body/hand landmarks, and resamples to the model's sequence length, matching offline evaluation. It does not apply WLASL frame ranges again. Video and extracted landmarks remain local.

Default output is JSON on stdout. `--output` creates a new UTF-8 file and refuses to overwrite an existing file. Diagnostics and errors go to stderr; detector libraries can also emit native stderr messages. Shell redirection with `>` works, but its encoding follows your shell; `--output` consistently writes UTF-8.

MediaPipe can collect API performance/usage metrics under its [privacy notice](https://developers.google.com/edge/mediapipe/solutions/tasks#mediapipe_tasks_privacy_notice). Local video processing does not imply that every dependency is network-silent. Failed telemetry connections can appear on stderr even when prediction succeeds; use the exit code and result to assess the command.

Result schema 1 includes:

- `predictions`: up to `top_k` labels and uncalibrated probabilities, sorted by score. Scores in a truncated list need not sum to one.
- `model`: model-file SHA-256, backend kind, vocabulary size, sequence length, and preprocessing identifier.
- `input`: decoded frame count and input basename; no absolute source path is emitted in a successful result.
- `tracking`: fractions of frames with assigned hands and usable shoulders.
- `warnings`: weak-tracking warnings; these are heuristics, not correctness estimates.

The SHA-256 identifies a model file; it does not establish its authenticity, rights, or accuracy. A model still predicts vocabulary words for unknown gestures. No hands or usable shoulders causes an error; low hand coverage causes a warning.

## Python API

```python
from signlang import SignRecognizer, extract_landmarks

recognizer = SignRecognizer.from_model("my-model.json", models="models")
result = recognizer.predict_video("sign.mp4", top_k=3)
print(result.predictions[0].label)
print(result.to_dict())

# Reuse extracted landmarks, for example when comparing models.
points, valid = extract_landmarks("sign.mp4", models="models")
result = recognizer.predict_landmarks(points, valid, top_k=5)
```

`points` is a finite numeric array with shape `(frames, 75, 3)`, with at least two frames. `valid` is a boolean array of shape `(frames, 75)`. Landmark order is pose 0–32, anatomical left hand 33–53, right hand 54–74. The preprocessing identifier is `pose33-hands42-shoulders-v1`. Detector assignment and normalization live in the package; arbitrary external landmark layouts are not compatible.

For a checkpoint you created or otherwise trust:

```python
recognizer = SignRecognizer.from_model("baseline.joblib", trusted=True)
recognizer = SignRecognizer.from_model("best.pt", device="cpu")
```

API errors use `ValueError` for invalid models/landmarks, `FileNotFoundError` for missing files, and `ImportError` for unavailable optional dependencies. Detector/backend failures can raise `RuntimeError` or `OSError`.

## Cached inputs and exit codes

```sh
signlang predict clip.npz --input-type landmarks --checkpoint my-model.json --output result.json
```

The NPZ must contain numeric `points` and boolean `valid` arrays in the documented layout. Pickled arrays are rejected. This mode requires only the core package for a JSON baseline and is useful for reproducible pipelines without cameras.

| Exit code | Meaning |
|---|---|
| 0 | Success |
| 1 | File, model, data, dependency, or runtime error |
| 2 | Invalid command-line usage |
| 130 | User interruption |

Use `signlang COMMAND --help` for the existing manifest, extract, baseline, train, evaluate, and webcam options. Baseline training and evaluation currently use the `train` extra; webcam needs `vision` plus the checkpoint backend. The owner has deferred the live-webcam trial.

Before training, run `signlang audit --output runs/landmark-audit-v1.json`. It validates cache structure and manifest provenance, reports missing caches, and lists clips with weak hand or shoulder tracking for review. It never replaces an existing report.

## Build and verify

```sh
python -m pip install ".[vision,train,dev]"
python -m pytest -q
python -m build
python -m twine check dist/*
python scripts/check_distribution.py
```

Install the built wheel into a separate virtual environment without system packages. From a directory outside the source tree, run `scripts/smoke_installed.py` with that environment's Python. It verifies the installed package/console entry point using fabricated landmarks and checks that the core installation excludes the optional ML/video stacks. CI performs these checks on Windows and Linux.

Original source code is MIT licensed. Read [THIRD_PARTY_NOTICES.md](../THIRD_PARTY_NOTICES.md) before using models or data. The wheel and source archive exclude recordings, participant landmarks, detector assets, and trained weights.
