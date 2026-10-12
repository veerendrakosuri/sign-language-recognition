"""MediaPipe Tasks extraction; pose wrists assign anatomical hand slots."""
from contextlib import ExitStack
from itertools import permutations
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


class Extractor:
    def __init__(self, models="models"):
        root = Path(models)
        for name in ("pose_landmarker_lite.task", "hand_landmarker.task"):
            if not (root / name).is_file():
                raise FileNotFoundError(f"Missing {root / name}. Run: signlang download-models")
        self.stack = ExitStack()
        try:
            self.pose = self.stack.enter_context(vision.PoseLandmarker.create_from_options(
                vision.PoseLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(root / "pose_landmarker_lite.task")),
                    running_mode=vision.RunningMode.VIDEO, num_poses=1)))
            self.hands = self.stack.enter_context(vision.HandLandmarker.create_from_options(
                vision.HandLandmarkerOptions(
                    base_options=python.BaseOptions(model_asset_path=str(root / "hand_landmarker.task")),
                    running_mode=vision.RunningMode.VIDEO, num_hands=2)))
        except Exception:
            self.stack.close()
            raise
        self.last_timestamp = -1

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.stack.close()

    def frame(self, bgr, timestamp_ms):
        timestamp_ms = max(int(timestamp_ms), self.last_timestamp + 1)
        self.last_timestamp = timestamp_ms
        image = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
        pose = self.pose.detect_for_video(image, timestamp_ms)
        hands = self.hands.detect_for_video(image, timestamp_ms)
        p, mask = np.zeros((75, 3), np.float32), np.zeros(75, bool)
        if pose.pose_landmarks:
            for i, landmark in enumerate(pose.pose_landmarks[0]):
                if landmark.visibility >= 0.5 and landmark.presence >= 0.5:
                    p[i] = [landmark.x, landmark.y, landmark.z]
                    mask[i] = True
        # Anatomical left/right from pose avoids a mirrored-selfie handedness assumption.
        wrists = [i for i in (15, 16) if mask[i]]
        detected = hands.hand_landmarks
        candidates = []
        for n in range(min(len(wrists), len(detected)), 0, -1):
            for hand_ids in permutations(range(len(detected)), n):
                for wrist_ids in permutations(wrists, n):
                    distances = [np.linalg.norm(np.array([detected[h][0].x, detected[h][0].y]) - p[w, :2])
                                 for h, w in zip(hand_ids, wrist_ids)]
                    if max(distances) <= 0.25:
                        candidates.append((sum(distances), list(zip(hand_ids, wrist_ids))))
            if candidates:
                break
        if candidates:
            for h, wrist in min(candidates, key=lambda x: x[0])[1]:
                start = 33 if wrist == 15 else 54
                p[start:start + 21] = [[v.x, v.y, v.z] for v in detected[h]]
                mask[start:start + 21] = True
        p[~mask] = 0
        return p, mask


def video_landmarks(path, models):
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        cap.release()
        raise ValueError(f"Missing/invalid frame rate: {path}")
    points, valid = [], []
    try:
        # New trackers for each clip: no temporal state crosses split boundaries.
        with Extractor(models) as extractor:
            index = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                p, m = extractor.frame(frame, index * 1000 / fps)
                points.append(p)
                valid.append(m)
                index += 1
    finally:
        cap.release()
    if not points:
        raise ValueError(f"Video has no decodable frames: {path}")
    return np.stack(points), np.stack(valid)


def extract_manifest(manifest, cache, models):
    from .data import read_manifest, write_json, digest
    from .features import normalize
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    model_hashes = {p.name: digest(p) for p in Path(models).glob("*.task")}
    # Refuse stale caches rather than mixing different extraction runs.
    provenance = dict(manifest_sha256=digest(manifest), model_sha256=model_hashes,
                      mediapipe=mp.__version__, schema=1)
    import json
    meta = cache / "provenance.json"
    if meta.exists() and json.loads(meta.read_text()) != provenance:
        raise ValueError("Cache provenance changed. Use a new --cache directory.")
    write_json(meta, provenance)
    report = []
    for row in read_manifest(manifest):
        vid = row["video_id"]
        target = cache / f"{vid}.npz"
        status = dict(video_id=vid, split=row["split"], gloss=row["gloss"])
        if target.exists():
            status.update(status="cached")
        elif not Path(row["video_path"]).is_file():
            status.update(status="missing_video")
        else:
            try:
                p, m = video_landmarks(row["video_path"], models)
                normalize(p, m)
                if not m[:, 33:].any():
                    raise ValueError("No usable hands detected")
                np.savez_compressed(target, points=p, valid=m)
                status.update(status="extracted", frames=len(p), hand_presence=float(m[:, 33:].mean()))
            except (ValueError, RuntimeError, OSError) as e:
                status.update(status="failed", error=str(e))
        report.append(status)
        print(f"{vid}: {status['status']}")
    write_json(cache / "extraction_report.json", report)
