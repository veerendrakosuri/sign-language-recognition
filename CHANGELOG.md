# Changelog

## 0.4.0 — 2026-10-18

- Expand the usable official WLASL100 split from 1,013 to 1,466 clips using selectively recovered public raw-video archives; source videos and landmark caches remain excluded from releases.
- Retrain the observed-frame baseline on 1,022 training clips after validation-only selection on 243 clips.
- Improve the measured partial-data result to 0.5323 top-1, 0.7960 top-5, and 0.4916 macro F1 on 201/258 official test clips.
- Publish the expanded run's coverage, cache audit, exact evaluated IDs, hashes, predictions, per-class metrics, and confusion matrix.
- Update the browser research preview to use the expanded baseline weights. Browser-upload and live-webcam accuracy remain unmeasured.

## 0.3.0 — 2026-10-17

- Add versioned `observed_mean` baseline preprocessing so missing landmark frames are excluded from each landmark's mean; old checkpoints continue to use `zero_mean`.
- Expand validation search to six regularization values and make the validated observed-frame summary the default for new baseline training.
- Apply the selected preprocessing consistently in training, evaluation, Python inference, webcam inference, JSON export, and browser inference.
- Improve the measured partial-data result on the same 100 available test clips from 0.32 to 0.38 top-1, 0.63 to 0.73 top-5, and 0.2115 to 0.2769 macro F1.

## 0.2.1 — 2026-10-15

- Add `signlang audit` as a supported cache-integrity and landmark-tracking quality command.
- Add a manually gated PyPI trusted-publishing workflow and maintainer setup guide.

## 0.2.0 — 2026-10-15

- Add `SignRecognizer`, typed prediction results, landmark extraction, and baseline export to the public Python API.
- Add `signlang predict` for trimmed videos and cached landmark NPZs, ranked JSON results, model hashes, tracking coverage, and documented exit codes.
- Add `signlang export-model`, `--version`, and `python -m signlang`.
- Add NumPy-only JSON baseline inference, binary/multiclass export, and explicit trust for API/CLI joblib loading.
- Separate optional video, baseline, training, and development dependencies.
- Add MIT source-code licensing with separate data/model notices and exclude weights/recordings from distributions.
- Add distribution checks, clean-install smoke checks, and package CI.

Recognition accuracy is unchanged. Live-webcam and browser-upload accuracy remain unmeasured.

## 0.1.0

- Initial WLASL manifest/extraction, baseline and GRU training, evaluation, local webcam demo, educational guides, and public browser research preview.
