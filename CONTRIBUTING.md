# Contributing

The primary deliverable is a Python developer toolkit and CLI for isolated ASL experimentation. See [the agreed direction](docs/PROJECT_DIRECTION.md) and [package guide](docs/PACKAGE_GUIDE.md).

Use Python 3.10–3.12 and a virtual environment. Install with `python -m pip install -r requirements.txt`. Run `python -m pytest -q` before a pull request. Core-only work can run `tests/test_api.py`, `tests/test_cli.py`, and `tests/test_features.py` using NumPy and pytest. Packaging changes also require `python -m build`, `python -m twine check dist/*`, and `python scripts/check_distribution.py`.

Keep changes focused. Document public API changes, CLI exit/output changes, and model/preprocessing compatibility. Add meaningful regression tests for behavioral changes. Fabricated fixtures validate software behavior and must never be presented as ASL accuracy evidence.

Model experiments must preserve official splits and report exact evaluation IDs, dataset coverage, and missing classes. Do not select models using test scores. Publish only reviewed, non-sensitive reports. Do not commit tokens, recordings, dataset videos, participant landmark caches, or private identifying information.

Original contributions use the repository's MIT source-code license. Third-party assets need their own attribution and verified permissions. WLASL and derived weights retain separate terms; see [notices](THIRD_PARTY_NOTICES.md). File an issue with dependency versions, a minimal reproduction, and sanitized errors. Share participant media only with appropriate permission.
