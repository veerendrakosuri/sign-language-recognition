"""Compare raw MediaPipe hand detections with previously assigned hand slots."""
import csv
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

ids = ["24636", "65506", "63664", "14685", "32157", "32246"]
with Path("data/processed/manifest.csv").open(newline="", encoding="utf-8") as f:
    rows = {r["video_id"]: r for r in csv.DictReader(f)}
report = []
for video_id in ids:
    row = rows[video_id]
    capture = cv2.VideoCapture(row["video_path"])
    fps = capture.get(cv2.CAP_PROP_FPS)
    if not capture.isOpened() or not np.isfinite(fps) or fps <= 0:
        capture.release()
        raise ValueError(f"Cannot read video or frame rate: {video_id}")
    raw = []
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path="models/hand_landmarker.task"),
        running_mode=vision.RunningMode.VIDEO, num_hands=2)
    with vision.HandLandmarker.create_from_options(options) as detector:
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            image = mp.Image(image_format=mp.ImageFormat.SRGB,
                             data=np.ascontiguousarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            detected = detector.detect_for_video(image, max(int(index * 1000 / fps), index))
            raw.append(bool(detected.hand_landmarks))
            index += 1
    capture.release()
    with np.load(f"data/landmarks/{video_id}.npz", allow_pickle=False) as z:
        valid = z["valid"]
    raw = np.asarray(raw, dtype=bool)
    assigned = valid[:, 33:].any(axis=1)
    wrists = valid[:, [15, 16]].any(axis=1)
    if len(raw) != len(assigned):
        raise ValueError(f"Frame count mismatch for {video_id}: {len(raw)} != {len(assigned)}")
    record = dict(video_id=video_id, gloss=row["gloss"], frames=len(raw),
                  raw_hand_fraction=round(float(raw.mean()), 3),
                  assigned_hand_fraction=round(float(assigned.mean()), 3),
                  raw_detected_but_unassigned_fraction=round(float((raw & ~assigned).mean()), 3),
                  any_wrist_fraction=round(float(wrists.mean()), 3))
    report.append(record)
    print(record, flush=True)
Path("runs/hand-assignment-diagnostic.json").write_text(
    json.dumps(report, indent=2) + "\n", encoding="utf-8")
