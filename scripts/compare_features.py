"""Local validation-only feature comparison; never loads test examples."""
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from signlang.data import load_split

manifest = "data/processed/manifest.csv"
cache = "data/landmarks"
train, train_labels, _, _ = load_split(manifest, cache, "train", 32)
valid, targets, _, _ = load_split(manifest, cache, "val", 32)


def features(x, kind):
    mean = x.mean(axis=1)
    if kind == "mean":
        return mean
    if kind == "mean_std":
        return np.concatenate([mean, x.std(axis=1)], axis=1)
    if kind == "mean_std_delta":
        return np.concatenate([mean, x.std(axis=1), x[:, -1] - x[:, 0]], axis=1)
    if kind == "four_time_bins":
        return x.reshape(len(x), 4, 8, 225).mean(axis=2).reshape(len(x), -1)
    raise ValueError(kind)


results = []
with threadpool_limits(limits=4):
    for kind in ("mean", "mean_std", "mean_std_delta", "four_time_bins"):
        for c in (1.0, 10.0):
            model = make_pipeline(StandardScaler(), LogisticRegression(
                C=c, max_iter=2000, class_weight="balanced", random_state=42))
            model.fit(features(train, kind), train_labels)
            score = float(model.score(features(valid, kind), targets))
            row = dict(features=kind, C=c, validation_top1=score)
            results.append(row)
            print(row, flush=True)
Path("runs/validation-feature-experiment.json").write_text(
    json.dumps(results, indent=2) + "\n", encoding="utf-8")
