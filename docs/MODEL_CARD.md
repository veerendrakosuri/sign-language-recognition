# Model card (complete after training)

- **Status:** no WLASL-trained model shipped in the initial repository.
- **Intended use:** educational isolated ASL vocabulary classification.
- **Out of scope:** continuous translation, ISL, medical/legal interpreting, accessibility reliability claims.
- **Training data:** fill subset, version, coverage, exclusions and terms.
- **Architecture:** mean-landmark logistic regression / two-layer landmark GRU.
- **Inputs:** 32 frames by 225 normalized coordinates; no face mesh.
- **Outputs:** vocabulary scores; optional heuristic confidence/stability gate.
- **Evaluation:** fill official test denominator, top-1/top-5, macro F1, missing classes, checkpoint and metadata hashes.
- **Live evidence:** fill signer diversity, conditions, unknown-gesture tests, hardware and measured FPS.
- **Known limitations:** domain shift, missing/incorrect tracking, variable sign boundaries, uncalibrated confidence, finite vocabulary, absent facial grammar.
- **Distribution:** review data/model terms before sharing weights or derived data. No dataset license is replaced by this project's source code.
