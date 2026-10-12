import json
import numpy as np
import pytest
from signlang.data import make_manifest, require_training_classes
from signlang.evaluation import metrics


def test_official_splits_and_original_class_order(tmp_path):
    entries = [dict(gloss=g, instances=[dict(video_id=str(i*3+j), split=s)
               for j, s in enumerate(("train", "val", "test"))]) for i, g in enumerate(("zebra", "apple"))]
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps(entries))
    rows = make_manifest(meta, tmp_path / "videos", tmp_path / "out", 2)
    assert rows[0]["gloss"] == "zebra" and rows[0]["label"] == 0
    assert [r["split"] for r in rows[:3]] == ["train", "val", "test"]
    assert len({r["video_id"] for r in rows}) == 6


def test_missing_training_class_is_not_silently_dropped():
    with pytest.raises(ValueError, match="apple"):
        require_training_classes(np.array([0, 0]), ["book", "apple"])


def test_topk_handles_binary_and_full_class_scores():
    result, pred = metrics(np.array([0, 1]), np.array([[.8, .2], [.7, .3]]), ["a", "b"])
    assert result["top1"] == .5 and result["top5"] == 1 and result["top_k"] == 2
    prob = np.array([[.01, .02, .03, .04, .1, .8]])
    result, _ = metrics(np.array([0]), prob, list("abcdef"))
    assert result["top5"] == 0
