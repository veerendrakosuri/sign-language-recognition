"""Preserve upstream WLASL class order and official instance splits."""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def make_manifest(metadata, videos, output, classes=100):
    entries = json.loads(Path(metadata).read_text(encoding="utf-8"))
    if not 2 <= classes <= len(entries):
        raise ValueError("classes must be between 2 and the metadata class count")
    selected = entries[:classes]
    labels = [item["gloss"] for item in selected]
    if len(set(labels)) != len(labels):
        raise ValueError("Duplicate glosses in metadata")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    rows, seen = [], set()
    for label, item in enumerate(selected):
        for instance in item["instances"]:
            vid = str(instance["video_id"])
            if not vid.isdigit() or vid in seen:
                raise ValueError(f"Unsafe or duplicate video ID: {vid}")
            seen.add(vid)
            split = instance["split"]
            if split not in {"train", "val", "test"}:
                raise ValueError(f"Unknown official split: {split}")
            path = (Path(videos) / f"{vid}.mp4").resolve()
            rows.append(dict(video_id=vid, label=label, gloss=item["gloss"], split=split,
                             video_path=str(path), available=int(path.is_file()),
                             signer_id=instance.get("signer_id", "")))
    with (out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    counts = {s: dict(expected=sum(r["split"] == s for r in rows),
                       available=sum(r["split"] == s and r["available"] for r in rows))
              for s in ("train", "val", "test")}
    per_class = {g: {s: sum(r["gloss"] == g and r["split"] == s and r["available"] for r in rows)
                     for s in ("train", "val", "test")} for g in labels}
    write_json(out / "labels.json", labels)
    write_json(out / "coverage.json", dict(classes=classes, counts=counts, per_class=per_class,
               metadata_sha256=digest(metadata), missing_ids=[r["video_id"] for r in rows if not r["available"]]))
    print(json.dumps(counts, indent=2))
    return rows


def read_manifest(path):
    with Path(path).open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    seen = set()
    for r in rows:
        if r["video_id"] in seen or r["split"] not in {"train", "val", "test"}:
            raise ValueError("Duplicate video or invalid split in manifest")
        seen.add(r["video_id"])
    return rows


def load_split(manifest, cache, split, frames=32):
    from .features import prepare
    rows = [r for r in read_manifest(manifest) if r["split"] == split]
    x, y, ids, raw = [], [], [], []
    for row in rows:
        file = Path(cache) / f'{row["video_id"]}.npz'
        if not file.exists():
            continue
        with np.load(file, allow_pickle=False) as z:
            p, m = z["points"], z["valid"]
        x.append(prepare(p, m, frames))
        y.append(int(row["label"]))
        ids.append(row["video_id"])
        raw.append((p, m))
    if not x:
        raise ValueError(f"No cached {split} samples. Run manifest and extract first.")
    return np.stack(x), np.array(y), ids, raw


def require_training_classes(y, labels):
    missing = sorted(set(range(len(labels))) - set(y.tolist()))
    if missing:
        raise ValueError("No training samples for: " + ", ".join(labels[i] for i in missing)
                         + ". Recover videos or explicitly choose a smaller class subset.")
