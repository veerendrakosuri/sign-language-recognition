"""Public inference API. Optional backends are imported only when requested."""
from dataclasses import asdict, dataclass, replace
import hashlib
import importlib
import json
from pathlib import Path
import pickle

import numpy as np

from .features import baseline_summary, prepare

PREPROCESSING = "pose33-hands42-shoulders-v1"


def _sha256(path):
    result = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _check_top_k(top_k):
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1:
        raise ValueError("top_k must be a positive integer")


def _dependency(name, extra):
    try:
        return importlib.import_module(name)
    except ImportError as error:
        raise ImportError(
            f"{name} is unavailable. Install: pip install 'sign-language-recognition[{extra}]'"
        ) from error


def _metadata(pack):
    labels, frames = pack.get("labels"), pack.get("frames")
    if (not isinstance(labels, list) or len(labels) < 2
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("Model labels must be a unique list of at least two non-empty strings")
    if isinstance(frames, bool) or not isinstance(frames, int) or not 2 <= frames <= 512:
        raise ValueError("Model frames must be an integer between 2 and 512")
    return tuple(labels), frames


@dataclass(frozen=True)
class Prediction:
    """A vocabulary label and an uncalibrated model probability."""
    label: str
    score: float


@dataclass(frozen=True)
class RecognitionResult:
    """Versioned result; ``to_dict()`` contains only JSON-compatible values."""
    predictions: tuple[Prediction, ...]
    model: dict
    input: dict
    tracking: dict
    warnings: tuple[str, ...]
    schema_version: int = 1
    score_type: str = "uncalibrated_model_probability"

    def to_dict(self):
        result = asdict(self)
        result["predictions"] = list(result["predictions"])
        result["warnings"] = list(result["warnings"])
        return result


def extract_landmarks(video, models="models"):
    """Decode one trimmed video into ``(points, valid)`` arrays.

    Points have shape (frames, 75, 3), and the boolean mask is (frames, 75).
    Requires the ``vision`` extra and downloaded MediaPipe detector assets.
    No frames or landmarks are sent to a remote service.
    """
    video = Path(video)
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    _dependency("cv2", "vision")
    _dependency("mediapipe", "vision")
    from .landmarks import video_landmarks
    return video_landmarks(video, models)


class SignRecognizer:
    """Load once and predict trimmed isolated-sign videos or landmark sequences.

    Use ``from_model()`` to load a JSON baseline, trusted joblib baseline, or
    GRU checkpoint. Model files are supplied explicitly and are not bundled.
    """

    @classmethod
    def from_model(cls, checkpoint, *, kind="auto", device="cpu", models="models", trusted=False):
        """Load model metadata and weights without initializing a camera.

        ``auto`` selects JSON for .json, joblib for .joblib, and GRU for .pt.
        Joblib can execute code: it requires ``trusted=True``. GRU uses
        PyTorch's restricted ``weights_only=True`` loader.
        """
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Model not found: {checkpoint}")
        if kind == "auto":
            kind = {".json": "baseline-json", ".joblib": "baseline", ".pt": "gru"}.get(checkpoint.suffix.lower())
        if kind not in {"baseline-json", "baseline", "gru"}:
            raise ValueError("Model kind must be auto, baseline-json, baseline, or gru")
        self = cls()
        self._kind, self._device, self._models = kind, device, Path(models)
        if kind == "baseline-json" and checkpoint.stat().st_size > 64 * 1024 * 1024:
            raise ValueError("JSON model exceeds the 64 MiB limit")
        self._sha256 = _sha256(checkpoint)
        if kind == "baseline-json":
            pack = json.loads(checkpoint.read_text(encoding="utf-8"))
            if not isinstance(pack, dict) or type(pack.get("schema")) is not int or pack["schema"] != 1:
                raise ValueError("Expected JSON baseline model schema 1")
            self.labels, self.frames = _metadata(pack)
            self.summary = pack.get("summary", "zero_mean")
            if self.summary not in {"zero_mean", "observed_mean"}:
                raise ValueError("JSON model summary must be zero_mean or observed_mean")
            try:
                self._mean = np.asarray(pack["mean"], dtype=np.float64)
                self._scale = np.asarray(pack["scale"], dtype=np.float64)
                self._weights = np.asarray(pack["weights"], dtype=np.float64)
                self._bias = np.asarray(pack["bias"], dtype=np.float64)
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError("JSON model must contain numeric mean, scale, weights, and bias") from error
            classes = len(self.labels)
            rows = self._weights.shape[0] if self._weights.ndim == 2 else 0
            if (self._mean.shape != (225,) or self._scale.shape != (225,)
                    or self._weights.shape != (rows, 225) or self._bias.shape != (rows,)
                    or rows not in ({1, 2} if classes == 2 else {classes})
                    or not all(np.isfinite(x).all() for x in (self._mean, self._scale, self._weights, self._bias))
                    or (self._scale <= 0).any()):
                raise ValueError("Invalid JSON model dimensions, non-finite weights, or non-positive scale")
        elif kind == "baseline":
            if not trusted:
                raise ValueError("Joblib checkpoints require explicit trust: use trusted=True or --trust-checkpoint")
            joblib = _dependency("joblib", "baseline")
            _dependency("sklearn", "baseline")
            try:
                pack = joblib.load(checkpoint)
            except (ValueError, EOFError, pickle.UnpicklingError, KeyError, IndexError) as error:
                raise ValueError(f"Invalid baseline checkpoint: {error}") from error
            if not isinstance(pack, dict):
                raise ValueError("Expected a baseline checkpoint dictionary")
            self.labels, self.frames = _metadata(pack)
            self.summary = pack.get("summary", "zero_mean")
            if self.summary not in {"zero_mean", "observed_mean"}:
                raise ValueError("Baseline summary must be zero_mean or observed_mean")
            self._model = pack.get("model")
            if not hasattr(self._model, "predict_proba") or not hasattr(self._model, "classes_"):
                raise ValueError("Baseline checkpoint must provide predict_proba and classes_")
            classes = np.asarray(self._model.classes_)
            if (classes.ndim != 1 or not np.issubdtype(classes.dtype, np.integer)
                    or len(set(classes.tolist())) != len(classes)
                    or (classes < 0).any() or (classes >= len(self.labels)).any()):
                raise ValueError("Baseline class IDs do not match model labels")
        else:
            torch = _dependency("torch", "train")
            from .model import SignGRU
            try:
                pack = torch.load(checkpoint, map_location="cpu", weights_only=True)
            except (ValueError, RuntimeError, EOFError, pickle.UnpicklingError) as error:
                raise ValueError(f"Invalid GRU checkpoint: {error}") from error
            if not isinstance(pack, dict):
                raise ValueError("Expected a GRU checkpoint dictionary")
            self.labels, self.frames = _metadata(pack)
            spec = pack.get("model")
            if (not isinstance(spec, dict) or spec.get("classes") != len(self.labels)
                    or set(spec) - {"classes", "hidden", "layers", "dropout"}
                    or any(type(spec.get(key, default)) is not int or spec.get(key, default) < 1
                           for key, default in (("hidden", 256), ("layers", 2)))
                    or not isinstance(spec.get("dropout", .3), (int, float))
                    or not 0 <= spec.get("dropout", .3) <= 1
                    or "state_dict" not in pack):
                raise ValueError("GRU model specification does not match labels")
            try:
                self._model = SignGRU(**spec)
                self._model.load_state_dict(pack["state_dict"])
            except (TypeError, ValueError, RuntimeError) as error:
                raise ValueError(f"Invalid GRU weights or specification: {error}") from error
            self._model = self._model.to(device).eval()
        return self

    def predict_landmarks(self, points, valid, *, top_k=5):
        """Rank vocabulary words using offline evaluation's preprocessing.

        ``points`` must contain at least two frames, 75 xyz points per frame;
        ``valid`` is a matching boolean mask. Missing points stay zero after
        normalization. No usable hands or shoulders raises ValueError.
        """
        _check_top_k(top_k)
        points, valid = np.asarray(points, dtype=np.float32), np.asarray(valid)
        if points.ndim != 3 or points.shape[1:] != (75, 3) or len(points) < 2:
            raise ValueError("Expected at least two frames with points shape (frames, 75, 3)")
        if valid.shape != points.shape[:2] or valid.dtype != np.bool_:
            raise ValueError("Expected a boolean mask with shape (frames, 75)")
        if not np.isfinite(points).all():
            raise ValueError("Landmark coordinates must be finite")
        if not valid[:, 33:].any():
            raise ValueError("No usable hands detected")
        x = prepare(points, valid, self.frames)
        if self._kind == "baseline-json":
            # Match StandardScaler's two float32 in-place operations.
            scaled = baseline_summary(points, valid, self.frames, self.summary)
            scaled -= self._mean
            scaled /= self._scale
            logits = self._weights @ scaled + self._bias
            if len(logits) == 1:
                positive = float(np.exp(-np.logaddexp(0., -logits[0])))
                probabilities = np.array([1 - positive, positive])
            else:
                probabilities = np.exp(logits - logits.max())
                probabilities /= probabilities.sum()
        elif self._kind == "baseline":
            probabilities = np.zeros(len(self.labels))
            feature = baseline_summary(points, valid, self.frames, self.summary)
            probabilities[np.asarray(self._model.classes_, dtype=int)] = self._model.predict_proba(feature[None])[0]
        else:
            torch = _dependency("torch", "train")
            with torch.inference_mode():
                probabilities = torch.softmax(self._model(torch.from_numpy(x[None]).to(self._device)), 1)[0].cpu().numpy()
        if (probabilities.shape != (len(self.labels),) or not np.isfinite(probabilities).all()
                or (probabilities < 0).any() or not np.isclose(probabilities.sum(), 1, atol=1e-5)):
            raise ValueError("Model returned invalid probabilities")
        order = np.argsort(-probabilities, kind="stable")[:min(top_k, len(self.labels))]
        hand_fraction = float(valid[:, 33:].any(axis=1).mean())
        shoulders = valid[:, 11] & valid[:, 12]
        warnings = ("Assigned hands were present in fewer than half the frames; prediction may be unreliable.",) if hand_fraction < .5 else ()
        return RecognitionResult(
            predictions=tuple(Prediction(self.labels[i], float(probabilities[i])) for i in order),
            model=dict(kind=self._kind, sha256=self._sha256, classes=len(self.labels),
                       sequence_frames=self.frames,
                       preprocessing=(f"{PREPROCESSING}:{self.summary}"
                                      if self._kind.startswith("baseline") else PREPROCESSING)),
            input=dict(frames=len(points)),
            tracking=dict(hand_frame_fraction=hand_fraction, shoulder_frame_fraction=float(shoulders.mean())),
            warnings=warnings,
        )

    def predict_video(self, video, *, top_k=5):
        """Extract a complete trimmed clip and return ranked suggestions."""
        _check_top_k(top_k)
        points, valid = extract_landmarks(video, self._models)
        result = self.predict_landmarks(points, valid, top_k=top_k)
        return replace(result, input={**result.input, "name": Path(video).name})
