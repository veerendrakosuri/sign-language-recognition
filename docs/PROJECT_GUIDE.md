# Your complete Project 3 guide

## 1. Understand what you are building

Input: a short clip of a person signing one ASL word. Output: the most likely word among your selected training classes. After offline evaluation, the same trained model reads a rolling webcam window and displays a stable prediction.

Learn these ideas as you build:

| Concept | Meaning in this project |
|---|---|
| Label / gloss | A word used as the class name for a sign |
| Landmark | A tracked body or hand keypoint, represented by x, y, z |
| Sequence | Ordered landmark frames: movement matters |
| Train / validation / test | Learn weights / select settings / measure final performance |
| Baseline | A simple model that tells you whether the GRU adds value |
| GRU | A neural network that maintains a state while reading a sequence |
| Epoch | One pass over all training examples |
| Overfitting | Learning training examples without generalizing to new clips |
| Top-1 / top-5 | Correct class ranked first / anywhere in the five highest scores |

You should be comfortable with Python functions, lists, file paths, NumPy arrays, and basic probability. You can learn PyTorch while following this guide. Start on CPU; use a GPU only if training time requires it.

## 2. Set up your computer

Use a laptop with a webcam, 64-bit Python 3.11 or 3.12, Git, and an editor such as VS Code. About 8 GB RAM is a practical starting point for cached-landmark experiments; reserve disk space according to the clips you download. Video extraction usually takes more time than this small model's training.

Follow the installation commands in the README. Run all commands from the repository root. Throughout this guide, `python` means the virtual environment's interpreter. On Windows you can replace every `python` with `.\.venv\Scripts\python.exe` without activating anything.

```powershell
.\.venv\Scripts\python.exe -m signlang.cli --help
.\.venv\Scripts\python.exe -m pytest -q
```

**Checkpoint:** help lists seven commands and tests pass. No webcam or dataset is required for the tests. Test fixtures are fabricated landmarks, not real signs.

On Ubuntu, the MediaPipe native library also needs `libegl1` and `libgles2`. Install them with `sudo apt-get update` followed by `sudo apt-get install -y libegl1 libgles2`. After downloading detector assets, `python scripts/check_landmarkers.py` verifies they load and process blank frames without opening a camera.

## 3. Obtain the correct data

Read [DATASET.md](DATASET.md). WLASL is American Sign Language. Do not mix ISL, ASL alphabet images, custom gestures, and WLASL labels into the same benchmark.

Start with the first 10 classes to debug the pipeline if needed, then rebuild separate manifests and caches for all 100 classes. Use the original metadata order. Do not alphabetically sort classes or choose new top classes based on whichever downloads worked.

Expected local layout:

```text
data/
  WLASL_v0.3.json
  videos/
    00335.mp4
    ...
```

The example ID is illustrative. Actual names must match `video_id` exactly, including leading zeros. Files must already contain only the annotated sign segment.

```powershell
python -m signlang.cli manifest --metadata data/WLASL_v0.3.json --videos data/videos --classes 100
```

**Checkpoint:** open `data/processed/coverage.json`. Inspect expected and available counts for train, val and test, plus per-class counts. Recover missing data when possible. Do not randomize the official split to make training easier.

## 4. Extract and inspect landmarks

```powershell
python -m signlang.cli download-models
python -m signlang.cli extract
```

The download command retrieves two Google detector assets and records SHA-256 hashes. Extraction processes video frames in timestamp order with separate trackers for every clip. A point mask records which observations are present.

Each cache file contains `points` with shape `(T, 75, 3)` and `valid` with shape `(T, 75)`. Slots 0–32 are pose, 33–53 left hand, 54–74 right hand. Hands are assigned to the anatomical side using pose wrists. If pose wrists are not reliable, those hands are left missing rather than guessing their identity.

Inspect at least five clips per class for your small prototype. Confirm that sign motion is complete, one signer is visible, wrists and shoulders are tracked, and clips are not reversed or double-trimmed. Use the notebook to inspect cached arrays and sample frames. Read `extraction_report.json`; failed samples are counted instead of silently treated as successful data.

**Checkpoint:** each intended class has cached train examples and a useful amount of validation/test coverage. A class with no train samples stops training. Partially missing validation/test classes remain in reporting; inspect them explicitly.

## 5. Understand preprocessing

1. Estimate each frame's shoulder midpoint.
2. Interpolate the body reference through frames missing shoulders.
3. Center image x/y coordinates around that reference.
4. Divide coordinates by median valid shoulder width for the clip.
5. Keep missing points exactly zero using the mask.
6. Select 32 evenly spaced frames and flatten 75 × 3 coordinates into 225 features.

The GRU input is `(batch, 32, 225)`. The baseline input is `(batch, 225)` after averaging over time.

Pose and hand z values have different origins. The implementation centers pose z but retains wrist-relative hand z. They are model features, not a metrically accurate common 3D skeleton. See [architecture details](ARCHITECTURE.md) for limitations.

## 6. Train the baseline first

```powershell
python -m signlang.cli baseline --output runs/baseline-v1
```

The baseline averages each normalized sequence, scales features using training data, and fits logistic regression. It chooses regularization from three values using validation top-1 accuracy. It does not read the test split.

Outputs: `baseline.joblib`, `validation.json`, `config.json`, and `sample_ids.json`. A mean-pose model ignores order: two signs with similar hand positions but different motion may confuse it. This gives the GRU a meaningful comparison.

**Checkpoint:** validation results and class coverage are recorded. Load joblib files only from sources you trust; they use Python pickle serialization.

## 7. Train the GRU

```powershell
python -m signlang.cli train --epochs 60 --batch-size 32 --output runs/gru-v1
```

The model has two GRU layers with 256 hidden units and dropout, followed by a classifier. It uses class-weighted cross entropy, AdamW, gradient clipping, small in-plane rotations and scaling. Early stopping selects the best validation top-1 checkpoint.

Defaults are starting values, not optimized settings. Try one controlled change at a time: hidden size 128 versus 256, learning rate 0.0003 versus 0.001, or sequence length 32 versus 48. Keep all experimental choices on validation. The same seed improves reproducibility but does not guarantee bitwise agreement across operating systems or hardware.

Optional mirroring correctly swaps all body side pairs and hand blocks:

```powershell
python -m signlang.cli train --mirror-probability 0.5 --output runs/gru-mirror-v1
```

Mirroring is off by default because horizontal reflection can change the meaning or naturalness of some signs. Enable it only after verifying that your chosen labels tolerate it; compare validation performance and inspect mirrored examples.

For a GPU environment with a compatible CUDA PyTorch installation, add `--device cuda`. CPU installation commands deliberately select CPU PyTorch. To use Colab/Kaggle, clone this repo there, install requirements, mount/copy your authorized data and cache, and run the same training commands. Use a local machine for the OpenCV webcam window. Save checkpoints before the cloud session ends.

**Checkpoint:** `best.pt` exists, `history.csv` shows learning, and validation score improves over a trivial guess. If training cannot fit even a tiny sample, debug data/labels before increasing model size.

## 8. Evaluate fairly

Finalize settings using validation, then evaluate the selected baseline and GRU once on test:

```powershell
python -m signlang.cli evaluate --kind baseline --checkpoint runs/baseline-v1/baseline.joblib --output runs/baseline-test-v1
python -m signlang.cli evaluate --kind gru --checkpoint runs/gru-v1/best.pt --output runs/gru-test-v1
```

Open `results.md`, `metrics.json`, `classification_report.json`, and `confusion_matrix.png`. Compare top-1 and top-5, macro F1, sample coverage, and the exact evaluated IDs. Compare only runs with identical class mapping and test IDs. Incomplete downloaded data does not produce a directly comparable full published benchmark.

For the five most frequent confusions, watch examples and record a hypothesis: handshape similarity, missing hand tracking, signing speed, view angle, or incorrect trimming. You do not need to beat the baseline to finish honestly; explain what happened and what you would improve.

**Checkpoint:** fill `reports/experiment_log.csv` and the README table using these actual outputs. Never use synthetic test scores or another repository's accuracy as your result.

## 9. Run your live demonstration

```powershell
python -m signlang.cli webcam --checkpoint runs/gru-v1/best.pt --window-seconds 1.5 --threshold 0.75
```

Stand so that shoulders, elbows and hands are visible. Use one signer, a stable camera and even lighting. Perform a vocabulary word clearly, with pauses between words. Press Q or Escape to quit.

The demo keeps a time-based rolling window, resamples to the training frame count, and predicts periodically. The same word must exceed the confidence threshold for three predictions before display. Missing tracking or low confidence clears the prediction. The screen is mirrored for comfort; inference uses the original frame orientation.

The 1.5-second default is only a starting value: tune window duration and threshold on validation/live development recordings, then freeze settings before your live test. Consecutive windows overlap and are not independent evidence. Displayed probabilities are not calibrated guarantees. Unknown gestures may still be misclassified.

## 10. Test with people and document limitations

With consent, test known vocabulary words across several signers, backgrounds, lighting conditions and signing speeds. Use anonymous signer codes. Do not publish their recordings without permission. If you cannot recruit ASL signers, state that limitation; imitating signs yourself does not establish broad usability.

Use `reports/live_test_log.csv` to record intended word, displayed word, accepted/uncertain state, FPS and conditions. Include neutral activity and out-of-vocabulary signs to measure false acceptance. Report number of trials, correct accepted predictions, rejected trials, errors, false accepts and median displayed FPS separately from offline scores. If measuring latency, time from the end of the sign to a stable output and state your method.

Record a brief demo only after you have a real trained model. Show successful and failed cases. A screen-recording GIF is a portfolio artifact, not proof of benchmark accuracy.

## 11. Prepare your submission / portfolio

Use [REPORT_TEMPLATE.md](REPORT_TEMPLATE.md) and [MODEL_CARD.md](MODEL_CARD.md). Include the problem, dataset coverage, method, baseline comparison, confusion analysis, live tests, limitations and next steps. Cite WLASL and MediaPipe. Replace every result placeholder with a measured value or 'not measured'.

Copy selected non-sensitive evaluation reports into a reviewed `reports/final/` directory if you want them on GitHub. Raw videos, caches and checkpoints remain ignored. Check dataset terms before sharing derived artifacts. Save dependency versions with `python -m pip freeze > runs/environment.txt`.

Resume sentence to fill only after evaluation: “Built an isolated ASL word classifier on [subset and available test count], comparing a landmark GRU with logistic regression; measured [top-1], [top-5] and [FPS] on [hardware].”

## Three-week plan

| Days | Work | Evidence to keep |
|---|---|---|
| 1–2 | Python setup; understand ASL scope and array shapes | Passing tests; notes |
| 3–5 | Obtain clips; audit splits; prototype 10 classes | Coverage and inspected clips |
| 6–7 | Extract landmarks; baseline | Cache report and validation scores |
| 8–11 | GRU training and controlled experiments | History, config, sample IDs |
| 12–14 | Expand to 100 classes; freeze settings; test | Baseline/GRU test reports |
| 15–17 | Webcam tuning and held-out live trials | FPS/error/uncertainty logs |
| 18–21 | Analyze mistakes; record demo; write report | Honest results and limitations |

Data access and extraction can extend this schedule. Spend the first session getting setup and coverage correct; do not begin by designing a polished UI.

## Troubleshooting

| Symptom | Action |
|---|---|
| `py` opens Store / is missing | Install Python from python.org and reopen terminal; select Python 3.11/3.12 |
| No module `signlang` | Run `python -m pip install -r requirements.txt` from repo root using the same interpreter |
| No matching MediaPipe wheel | Use 64-bit supported Python and update pip; do not use Python 3.13+ for this setup |
| Missing detector assets | Run `download-models`; verify both `.task` files exist |
| No cached train examples | Check filenames and coverage; run extraction; inspect failures |
| Missing training class | Recover that class's videos, or explicitly rebuild a smaller first-K prototype |
| Cache provenance changed | Use a new `--cache` directory; never combine incompatible feature extractions |
| Camera black / cannot open | Close Teams/Zoom; enable camera permissions; try `--camera 1` |
| OpenCV cannot create window | Use local desktop Python with `opencv-contrib-python`, not a headless package or hosted notebook; install only one OpenCV distribution |
| Accuracy high offline, poor live | Check crop/orientation, lighting, sign duration, tracking and domain shift |
| Always uncertain | Inspect tracking and true vocabulary first; tune threshold on development data, not test |
| Very slow | Use fewer prototype classes, smaller `--hidden 128`, or a suitable GPU; measure before optimizing |
| Output directory not empty | Use a new run name; prior experiments are intentionally protected |
