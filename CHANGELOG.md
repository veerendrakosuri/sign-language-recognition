# Changelog

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
