"""Validate cached landmarks and report tracking coverage before training."""

import json
from collections import Counter
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from .data import digest, read_manifest, write_json


def _summary(rows):
    if not rows:
        return {"clips": 0}
    fields = ("frames", "shoulder_frame_fraction", "any_hand_frame_fraction",
              "both_hands_frame_fraction")
    result = {"clips": len(rows)}
    for field in fields:
        values = np.asarray([row[field] for row in rows], dtype=float)
        result[field] = {
            name: round(float(np.percentile(values, percentile)), 4)
            for name, percentile in (("min", 0), ("p10", 10), ("median", 50),
                                     ("p90", 90), ("max", 100))
        }
    result["review_candidates"] = dict(Counter(
        flag for row in rows for flag in row["review_flags"]))
    return result


def audit_landmarks(manifest, cache, output):
    """Write a new JSON audit of cache integrity and tracking quality."""
    manifest, cache, output = Path(manifest), Path(cache), Path(output)
    if output.exists():
        raise FileExistsError(f"Output already exists; choose a new path: {output}")
    provenance = cache / "provenance.json"
    if not provenance.is_file():
        raise ValueError(f"Missing cache provenance: {provenance}")
    manifest_hash = digest(manifest)
    try:
        recorded_hash = json.loads(provenance.read_text(encoding="utf-8"))["manifest_sha256"]
    except (OSError, json.JSONDecodeError, KeyError, TypeError) as error:
        raise ValueError(f"Invalid cache provenance: {provenance}") from error
    if recorded_hash != manifest_hash:
        raise ValueError("Cache provenance does not match this manifest")

    records, errors, missing = [], [], []
    for row in read_manifest(manifest):
        path = cache / f'{row["video_id"]}.npz'
        if not path.is_file():
            missing.append(row["video_id"])
            continue
        try:
            with np.load(path, allow_pickle=False) as archive:
                points, valid = archive["points"], archive["valid"]
            if points.ndim != 3 or points.shape[1:] != (75, 3) or len(points) == 0:
                raise ValueError(f"unexpected points shape {points.shape}")
            if valid.shape != points.shape[:2] or valid.dtype != np.bool_:
                raise ValueError(f"unexpected mask shape/type {valid.shape}/{valid.dtype}")
            if not np.isfinite(points).all() or np.any(points[~valid] != 0):
                raise ValueError("non-finite coordinates or nonzero masked points")
            shoulders = valid[:, 11] & valid[:, 12]
            left = valid[:, 33:54].any(axis=1)
            right = valid[:, 54:75].any(axis=1)
            any_hand = left | right
            if not shoulders.any() or not any_hand.any():
                raise ValueError("no usable shoulder pair or hands")
            shoulder_fraction = float(shoulders.mean())
            hand_fraction = float(any_hand.mean())
            flags = []
            if len(points) < 12:
                flags.append("under_12_frames")
            if shoulder_fraction < 0.5:
                flags.append("shoulders_under_half_frames")
            if hand_fraction < 0.5:
                flags.append("hands_under_half_frames")
            records.append({
                "video_id": row["video_id"], "split": row["split"], "gloss": row["gloss"],
                "frames": len(points), "shoulder_frame_fraction": shoulder_fraction,
                "any_hand_frame_fraction": hand_fraction,
                "both_hands_frame_fraction": float((left & right).mean()),
                "review_flags": flags,
            })
        except (OSError, KeyError, ValueError, BadZipFile, EOFError) as error:
            errors.append({"video_id": row["video_id"], "error": str(error)})

    report = {
        "manifest_sha256": manifest_hash,
        "cached_clips": len(records),
        "missing_cache_count": len(missing),
        "malformed_cache_count": len(errors),
        "missing_cache_ids": missing,
        "malformed_caches": errors,
        "all": _summary(records),
        "by_split": {
            split: _summary([record for record in records if record["split"] == split])
            for split in ("train", "val", "test")
        },
        "review_candidates": [record for record in records if record["review_flags"]],
    }
    write_json(output, report)
    return report
