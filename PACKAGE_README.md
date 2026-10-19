# Isolated ASL Recognition Toolkit

A Python API and `signlang` CLI for experimental isolated ASL word recognition, landmark extraction, training, and reproducible evaluation. Python 3.10–3.12. Version 0.4.0 is an alpha release.

Install from the [GitHub releases](https://github.com/veerendrakosuri/sign-language-recognition/releases) wheel, or from the repository using `python -m pip install .`. Package-registry publication is not claimed.

The core requires NumPy and predicts from cached landmarks with a user-supplied numeric JSON model. Add `[vision]` for video extraction, `[baseline]` for trusted scikit-learn checkpoints, or `[train]` for GRU/training/evaluation workflows. No trained models or datasets are bundled.

```sh
signlang --version
signlang predict sign.mp4 --checkpoint model.json --output result.json
signlang predict clip.npz --input-type landmarks --checkpoint model.json
```

```python
from signlang import SignRecognizer

recognizer = SignRecognizer.from_model("model.json", models="models")
result = recognizer.predict_video("sign.mp4")  # requires vision extra and detector assets
print(result.to_dict())
```

Video input must contain one complete trimmed sign. Download the detector assets first with `signlang download-models`. JSON results include ranked uncalibrated scores, tracking coverage, a model SHA-256 fingerprint, and preprocessing metadata.

The reference baseline scored 107/201 (53.23%) on available offline test clips from an incomplete WLASL100 collection. This is not full-benchmark or live accuracy. Predictions and unknown-gesture behavior remain experimental. ASL and ISL require different datasets and evaluation.

Original project source code is MIT licensed. WLASL, derived model weights, recordings, and third-party assets retain separate terms; see [notices](https://github.com/veerendrakosuri/sign-language-recognition/blob/main/THIRD_PARTY_NOTICES.md).

[API/CLI guide](https://github.com/veerendrakosuri/sign-language-recognition/blob/main/docs/PACKAGE_GUIDE.md) · [Repository](https://github.com/veerendrakosuri/sign-language-recognition) · [Browser research preview](https://signlab-asl-veerendra.heartyspoon1024.chatgpt.site)
