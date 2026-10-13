"""Audit cached landmark structure and tracking coverage without model inference."""
import argparse
import json
from collections import Counter
from pathlib import Path
from zipfile import BadZipFile

import numpy as np

from signlang.data import digest, read_manifest, write_json


def summarize(rows):
    if not rows:
        return {"clips": 0}
    fields = ("frames", "shoulder_frame_fraction", "any_hand_frame_fraction",
              "both_hands_frame_fraction")
    result = {"clips": len(rows)}
    for field in fields:
        values = np.array([row[field] for row in rows], dtype=float)
        result[field] = {name: round(float(np.percentile(values, pct)), 4) for name, pct in
                         (("min", 0), ("p10", 10), ("median", 50), ("p90", 90), ("max", 100))}
    result["review_candidates"] = dict(Counter(
        flag for row in rows for flag in row["review_flags"]))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--cache", default="data/landmarks")
    parser.add_argument("--output", default="runs/landmark-audit.json")
    args = parser.parse_args()
    cache = Path(args.cache)
    provenance = cache / "provenance.json"
    if not provenance.is_file():
        raise ValueError(f"Missing cache provenance: {provenance}")
    manifest_hash = digest(args.manifest)
    if json.loads(provenance.read_text(encoding="utf-8"))["manifest_sha256"] != manifest_hash:
        raise ValueError("Cache provenance does not match this manifest")
    records, errors, missing = [], [], []
    for row in read_manifest(args.manifest):
        path = cache / f'{row["video_id"]}.npz'
        if not path.is_file():
            missing.append(row["video_id"])
            continue
        try:
            with np.load(path, allow_pickle=False) as z:
                points, valid = z["points"], z["valid"]
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
            records.append(dict(video_id=row["video_id"], split=row["split"], gloss=row["gloss"],
                                frames=len(points), shoulder_frame_fraction=shoulder_fraction,
                                any_hand_frame_fraction=hand_fraction,
                                both_hands_frame_fraction=float((left & right).mean()),
                                review_flags=flags))
        except (OSError, KeyError, ValueError, BadZipFile, EOFError) as exc:
            errors.append(dict(video_id=row["video_id"], error=str(exc)))
    report = dict(manifest_sha256=manifest_hash, cached_clips=len(records), missing_cache_count=len(missing),
                  malformed_cache_count=len(errors), missing_cache_ids=missing, malformed_caches=errors,
                  all=summarize(records),
                  by_split={split: summarize([r for r in records if r["split"] == split])
                            for split in ("train", "val", "test")},
                  review_candidates=[r for r in records if r["review_flags"]])
    write_json(args.output, report)
    print(json.dumps({k: report[k] for k in ("cached_clips", "missing_cache_count",
                                             "malformed_cache_count", "all")}, indent=2))
    print(f"Full report: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
