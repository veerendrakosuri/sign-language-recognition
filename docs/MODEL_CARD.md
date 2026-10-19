# Model card — local WLASL100 mirror experiment

- **Status:** baseline trained locally on an incomplete WLASL100 video collection. The older GRU was not retrained on the expanded data. Original checkpoints remain local; the browser app publishes numeric baseline weights in `web/public/model.json`. [Measured offline results](../reports/wlasl100-expanded-run-2026-10-18/README.md) are published. This is not a full WLASL100 benchmark.
- **Intended use:** educational isolated ASL vocabulary classification.
- **Out of scope:** continuous translation, ISL, medical/legal interpreting, accessibility reliability claims.
- **Training data:** official WLASL100 metadata and split mapping; 1,022/1,442 train, 243/338 validation, and 201/258 test clips available after selective recovery from public archives. Clip labels, provenance, and trimming have not been independently verified.
- **Architecture:** observed-frame mean-landmark logistic regression / two-layer landmark GRU. Missing frames are excluded separately for each landmark in the selected baseline.
- **Inputs:** 32 frames by 225 normalized coordinates; no face mesh.
- **Outputs:** vocabulary scores; optional heuristic confidence/stability gate.
- **Evaluation:** on 201/258 official test IDs, selected baseline top-1 0.5323/top-5 0.7960/macro F1 0.4916. All 100 classes are counted for macro F1, including 2 with no available test clip. The older GRU result used only 100 test clips and is not directly comparable. Exact IDs and checkpoint/manifest hashes are in the metrics files.
- **Browser/live evidence:** browser-upload and live-webcam accuracy are not measured. The browser samples 32 frames directly, while offline evaluation decoded all frames before resampling. Signer diversity, unknown-gesture behavior, hardware FPS, and live accuracy remain untested.
- **Known limitations:** domain shift, missing/incorrect tracking, variable sign boundaries, uncalibrated confidence, finite vocabulary, absent facial grammar. A cache audit found 438/1,466 clips with an assigned hand in fewer than half their frames.
- **Distribution:** numeric baseline weights are included solely for this educational research preview; no training videos or cached landmarks are published. WLASL's academic/computational-use restriction remains relevant. No dataset license is replaced by this project's source code.
