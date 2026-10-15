import csv
import json

import numpy as np

from signlang.audit import audit_landmarks
from signlang.data import digest


def _fixture(tmp_path):
    manifest = tmp_path / "manifest.csv"
    rows = [
        {"video_id": "good", "gloss": "hello", "split": "train"},
        {"video_id": "weak", "gloss": "hello", "split": "val"},
        {"video_id": "missing", "gloss": "hello", "split": "test"},
    ]
    with manifest.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "provenance.json").write_text(
        json.dumps({"manifest_sha256": digest(manifest)}), encoding="utf-8")
    for name, hand_frames in (("good", 12), ("weak", 2)):
        points = np.zeros((12, 75, 3), dtype=np.float32)
        valid = np.zeros((12, 75), dtype=bool)
        valid[:, [11, 12]] = True
        points[:, 11] = [0.3, 0.4, 0.0]
        points[:, 12] = [0.7, 0.4, 0.0]
        valid[:hand_frames, 33:54] = True
        np.savez(cache / f"{name}.npz", points=points, valid=valid)
    return manifest, cache


def test_audit_reports_missing_and_weak_tracking(tmp_path):
    manifest, cache = _fixture(tmp_path)
    output = tmp_path / "audit.json"
    report = audit_landmarks(manifest, cache, output)
    assert report["cached_clips"] == 2
    assert report["missing_cache_ids"] == ["missing"]
    assert report["malformed_cache_count"] == 0
    assert report["all"]["review_candidates"] == {"hands_under_half_frames": 1}
    assert report["review_candidates"][0]["video_id"] == "weak"
    assert json.loads(output.read_text(encoding="utf-8")) == report


def test_audit_preserves_existing_output_and_checks_provenance(tmp_path):
    manifest, cache = _fixture(tmp_path)
    output = tmp_path / "existing.json"
    output.write_text("keep", encoding="utf-8")
    try:
        audit_landmarks(manifest, cache, output)
        raise AssertionError("existing output should be rejected")
    except FileExistsError:
        pass
    assert output.read_text(encoding="utf-8") == "keep"
    output.unlink()
    (cache / "provenance.json").write_text(json.dumps({"manifest_sha256": "wrong"}))
    try:
        audit_landmarks(manifest, cache, output)
        raise AssertionError("mismatched provenance should be rejected")
    except ValueError as error:
        assert "does not match" in str(error)
