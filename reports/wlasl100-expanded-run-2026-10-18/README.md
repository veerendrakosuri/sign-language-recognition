# Expanded partial WLASL100 run — 18 October 2026

This run uses the official WLASL100 metadata and split mapping. It combines the original public processed-video mirror with clips selectively recovered from public raw-video archives. No source videos, landmark caches, or Python checkpoints are published here.

| Split | Available | Official | Coverage |
|---|---:|---:|---:|
| Train | 1,022 | 1,442 | 70.9% |
| Validation | 243 | 338 | 71.9% |
| Test | 201 | 258 | 77.9% |
| Total | 1,466 | 2,038 | 71.9% |

The cache audit covered all 1,466 available clips, found no missing or malformed available caches, and flagged 438 clips where an assigned hand appeared in fewer than half the frames. These flags require visual review; they do not prove the clip or label is wrong.

Validation-only model selection chose an observed-frame logistic-regression baseline with `C=3.0` (validation top-1 0.5432 on 243 clips). The held-out test split was then evaluated once:

| Samples | Top-1 | Top-5 | Macro F1 | Missing test classes |
|---:|---:|---:|---:|---:|
| 201/258 | 0.5323 | 0.7960 | 0.4916 | 2 (`book`, `later`) |

The result is for the available subset, not the complete WLASL100 benchmark. It is also not directly comparable to the earlier 100-test-clip result because the evaluated sample set changed. Browser-upload and live-webcam accuracy remain unmeasured.

- Checkpoint SHA-256: `ee69ed0fbdf1318b88ede131623614af7ac8df6604203fecdc657eef9ddce180`
- Manifest SHA-256: `e70ebd92d1a6e40de42263269eaa605591b5e8549d335d3609b815ef36c15bc7`
- [`baseline/metrics.json`](baseline/metrics.json): denominator, exact evaluated IDs, aggregate metrics, and hashes
- [`baseline/predictions.csv`](baseline/predictions.csv): per-clip predictions
- [`baseline/classification_report.json`](baseline/classification_report.json): per-class metrics
- [`baseline/confusion_matrix.csv`](baseline/confusion_matrix.csv) and [`baseline/confusion_matrix.png`](baseline/confusion_matrix.png): confusion matrix
- [`baseline/validation.json`](baseline/validation.json): validation search
- [`landmark_audit.json`](landmark_audit.json): cache and tracking-quality audit
