# WLASL100 mirror run — 2026-10-13

This is a measured **partial-data experiment**, not a WLASL100 benchmark result. The user supplied pre-trimmed clips from a Kaggle WLASL mirror; clip provenance, labels, and trimming have not been independently audited. The official WLASL metadata defined the class order and train/validation/test splits.

| Split | Cached clips | Official instances | Covered classes |
|---|---:|---:|---:|
| Train | 748 | 1,442 | 100/100 |
| Validation | 165 | 338 | 89/100 |
| Test | 100 | 258 | 72/100 |

All 1,013 available clips produced landmark cache files. The other 1,025 expected videos were absent from this mirror. No failed extraction was reported. Missing test clips include every instance for 28 classes. Both models were evaluated on the **same 100 test video IDs**; the IDs and checkpoint/manifest hashes are recorded in each `metrics.json`. Macro F1 averages over all 100 classes, counting absent test classes as zero.

| Model | Validation top-1 | Test top-1 | Test top-5 | Test macro F1 |
|---|---:|---:|---:|---:|
| Mean-landmark logistic regression, C=10 | 0.4121 | 0.3200 | 0.6300 | 0.2115 |
| Two-layer GRU, 256 hidden units, best epoch 57 | 0.3576 | 0.3000 | 0.6200 | 0.1993 |

Both models used 32 frames, seed 42, and the cached MediaPipe pose/hand landmarks. GRU training used CPU, batch size 32, learning rate 0.001, and no mirroring. The baseline performed better on validation and on this limited test subset. The GRU's low training loss alongside lower validation accuracy suggests overfitting; a larger and verified dataset is more valuable than claiming this as a successful GRU improvement.

The `baseline/` and `gru/` folders contain the exact evaluation outputs: `metrics.json`, `classification_report.json`, `predictions.csv`, `confusion_matrix.csv`, `confusion_matrix.png`, and `results.md`. Baseline selection scores and both training configurations are included; `gru/history.csv` records all 60 epochs. Raw videos, landmark caches, and serialized checkpoints are kept out of Git. The local checkpoints are `runs/baseline-v1/baseline.joblib` and `runs/gru-v1/best.pt`.

The [landmark audit](landmark_audit.json) checked all 1,013 cached arrays for expected shape, finite values, zeroed missing points, and usable shoulders/hands. It found **zero malformed caches**; median length was 65 frames and median assigned-hand coverage was 60% of frames. **325 clips** (241 train, 50 validation, 34 test) had an assigned hand in fewer than half their frames. This 50% threshold is a review heuristic, not an extraction failure or an accuracy measure. Twelve clips selected from low tracking, confident GRU errors, correct GRU predictions, and training examples were inspected at three sampled frames each. The sampled frames showed one visible signer and gesture motion, but they cannot verify full clip boundaries or ASL label correctness. Hands were visible at gesture peaks in some low-coverage clips; the cause of missing assigned-hand landmarks has not been isolated.

The automated tests passed (9/9), the MediaPipe detector assets loaded, and the local GRU checkpoint ran inference on a cached clip. These checks do not measure live-camera performance. The camera has not been opened for this report, so webcam FPS, unknown-gesture behavior, and signer diversity remain unmeasured.

The webcam command can now load either the GRU or the baseline checkpoint with `--kind baseline`. Both paths use the same normalized 32-frame landmarks as evaluation. The baseline option is available because it outperformed the GRU offline on this partial test subset; that does not predict its live-camera ranking.

Before presenting this as a full project result, have an ASL-qualified reviewer check clip labels and complete sign boundaries, recover missing videos where permitted, rerun training and held-out evaluation on the improved coverage, and measure the webcam demo's FPS and failure cases. Do not present these partial-data numbers as the official WLASL100 score.

An [offline follow-up](followup_2026-10-15.md) documents a validation-only feature comparison and an exploratory diagnosis of hand-tracking losses after the first informal webcam feedback. No new test score or live accuracy claim was added.
