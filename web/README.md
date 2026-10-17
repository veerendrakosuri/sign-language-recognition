# SignLab web app

This is a research-preview upload app for **one isolated ASL word at a time**. It is not an interpreter or accessibility service. Browser-upload accuracy has not been measured. The published baseline scored 38/100 correct on the *available* official WLASL100 test clips; 158 other test clips were missing from the mirror.

## Build and test

From `web/`, run `npm ci`, `npm test`, and `npm run build`. For local development run `npm run dev`. Open the Vite URL in a browser. The app downloads MediaPipe WASM and detector models when someone analyzes a video, so the visitor needs an internet connection. Video frames are processed in the browser and are not sent to this app's server. MediaPipe may collect [API usage metrics](https://developers.google.com/edge/mediapipe/solutions/tasks#mediapipe_tasks_privacy_notice).

The web model is exported from the trusted local checkpoint with `python scripts/export_web_model.py`. The JSON in `public/model.json` contains weights and labels, not source videos or extracted landmarks. Its `checkpoint_sha256` matches the [evaluated baseline report](../reports/wlasl100-mirror-run-2026-10-13/baseline/metrics.json). Never run `joblib.load` on an untrusted file.

The browser samples 32 frames from a trimmed clip and uses the same normalization, observed-frame landmark summary, and class order as the Python baseline. Offline evaluation decoded every frame before resampling, so its 38% score does **not** establish a browser-upload accuracy rate. Low hand tracking, different camera conditions, unknown signs, and video codec support can produce wrong suggestions. Scores are uncalibrated model probabilities.

WLASL data are restricted to academic and computational use by the [dataset authors](https://github.com/dxli94/WLASL). No WLASL source video or personal recording is hosted here. Review the dataset terms before reusing the model, especially for any public or commercial application.
