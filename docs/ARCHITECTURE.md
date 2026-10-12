# Architecture and implementation choices

`data.py` preserves upstream class order and official splits. `landmarks.py` extracts frame-wise features with MediaPipe Tasks, using a new tracker per video. `features.py` is the single preprocessing path used by training, evaluation and live inference.

## Feature representation

The representation contains 225 values per frame. Masks prevent zero-filled missing points from becoming fake nonzero observations during centering. Median shoulder width provides one scale for the clip. Missing shoulder references are interpolated; entirely missing shoulders reject a clip. All-zero hand clips are rejected during extraction.

Pose x/y and hand x/y use image coordinates. The z values are approximate and use different reference origins: pose is centered using shoulder z, while hands retain wrist-relative z. Image aspect ratio is not corrected in this first version, so changing camera aspect ratios can affect geometry. This representation is a learning baseline, not a calibrated 3D reconstruction.

Pose wrists assign hands by the lowest total image-distance matching within a gate. This avoids reliance on selfie handedness conventions but can fail with crossed hands, poor pose tracking or multiple people. Hands without a usable matching pose wrist are omitted. Evaluate this explicitly before upgrading the recognizer.

## Training

Logistic regression uses StandardScaler fit on train only and class balancing. GRU uses two layers, hidden size 256 by default, dropout 0.3, and a linear classifier. AdamW optimizes class-weighted cross entropy, with gradient norm clipped to 1.0. Validation top-1 selects the saved model and early stopping. Test clips are not loaded during training.

Small rotations and scaling are applied consistently across a training clip. Optional reflection swaps left/right body points, hand blocks and masks; it is disabled until label semantics have been checked. Validation and test preprocessing are deterministic.

## Live behavior

Frame rate fluctuates, so the demo stores a time-duration window instead of assuming 32 captured frames always mean the same duration. It resamples that window to the checkpoint's sequence length. Predictions occur every N processed frames and must pass a confidence-and-repeat gate. A loss of body/hand tracking clears accumulated context.

Training uses complete annotated clips; live windows may contain partial signs or several signs. This is a deliberate limitation, not solved by confidence smoothing. A useful extension is explicit sign segmentation or a neutral/background class trained from appropriate recordings. Calibrate thresholds on held-out live development recordings; keep a separate final live test.

## Next experiments

Add velocity features; append visibility masks as model inputs; compare wrist-relative hand features; correct image aspect ratio; examine crossed-hand assignment; add temporal convolutions; introduce sign-boundary detection; calibrate rejection on unknown gestures. Change one factor at a time and keep official test data untouched until decisions are final.
