# Model card — local WLASL100 mirror experiment

- **Status:** baseline and GRU trained locally on an incomplete WLASL100 video mirror. Original checkpoints remain local; the browser app publishes numeric baseline weights in `web/public/model.json`. [Measured offline results](../reports/wlasl100-mirror-run-2026-10-13/README.md) are published. This is not a full WLASL100 benchmark.
- **Intended use:** educational isolated ASL vocabulary classification.
- **Out of scope:** continuous translation, ISL, medical/legal interpreting, accessibility reliability claims.
- **Training data:** official WLASL100 metadata and split mapping; 748/1,442 train, 165/338 validation, and 100/258 test clips available in the user's mirror. Clip labels, provenance, and trimming have not been independently verified.
- **Architecture:** observed-frame mean-landmark logistic regression / two-layer landmark GRU. Missing frames are excluded separately for each landmark in the selected baseline.
- **Inputs:** 32 frames by 225 normalized coordinates; no face mesh.
- **Outputs:** vocabulary scores; optional heuristic confidence/stability gate.
- **Evaluation:** on the same 100 available test IDs, selected baseline top-1 0.3800/top-5 0.7300/macro F1 0.2769; original zero-filled baseline top-1 0.3200/top-5 0.6300/macro F1 0.2115; GRU top-1 0.3000/top-5 0.6200/macro F1 0.1993. All 100 classes are counted for macro F1, including 28 with no available test clip. Exact IDs and checkpoint/manifest hashes are in the metrics files.
- **Browser/live evidence:** browser-upload and live-webcam accuracy are not measured. The browser samples 32 frames directly, while offline evaluation decoded all frames before resampling. Signer diversity, unknown-gesture behavior, hardware FPS, and live accuracy remain untested.
- **Known limitations:** domain shift, missing/incorrect tracking, variable sign boundaries, uncalibrated confidence, finite vocabulary, absent facial grammar. A cache audit found 325/1,013 clips with an assigned hand in fewer than half their frames.
- **Distribution:** numeric baseline weights are included solely for this educational research preview; no training videos or cached landmarks are published. WLASL's academic/computational-use restriction remains relevant. No dataset license is replaced by this project's source code.
