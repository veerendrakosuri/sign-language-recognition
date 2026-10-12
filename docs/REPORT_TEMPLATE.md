# Project report: isolated ASL word recognition

## Problem and scope
Describe the target vocabulary and educational use. State explicitly that this does not translate continuous signed sentences.

## Data
Record metadata version/hash, subset size, source/access date, expected versus available counts per split, extraction failure counts, and excluded/missing classes. Describe the manual clip audit and official split policy.

## Method
Explain the 75 landmarks, mask handling, normalization, 32-frame sampling, baseline, GRU, hyperparameters, augmentation and validation selection. Include a pipeline diagram and exact commands.

## Results
Fill with measured values; leave blank until evaluated.

| Model | Evaluated / expected clips | Top-1 | Top-5 | Macro F1 |
|---|---|---|---|---|
| Logistic regression | | | | |
| GRU | | | | |

Include confusion matrix and five example errors. State whether both models use exactly the same evaluation IDs. Explain differences from any published benchmark before comparing numbers.

## Live testing
Record hardware, window length, threshold, signers, trial counts, conditions, rejection rate, false accepts, accuracy and FPS measurement method. Link a consented demo if available. Distinguish offline and live results.

## Limitations and future work
Discuss incomplete data, extraction quality, unseen signers, missing facial cues, limited vocabulary and unknown-gesture rejection. Name the most useful next experiment and why.

## Reproducibility and references
Include commit, environment versions, commands, random seed, selected checkpoint hash and WLASL/MediaPipe citations. Do not include credentials or personal participant information.
