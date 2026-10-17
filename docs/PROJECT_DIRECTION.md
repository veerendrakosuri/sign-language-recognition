# Agreed project direction

Accepted by the project owner on 15 October 2026.

## Product

Build a developer-focused, open-source Python package and CLI for **isolated ASL word recognition and reproducible experimentation**. Keep the existing website as the demonstration interface. Dataset preparation, landmark extraction, tracking-quality audits, training, and evaluation are useful deliverables alongside experimental recognition.

Primary users are developers, students, and researchers who want to process their own authorized videos, reproduce experiments, or integrate recognition into an application.

## Current foundation

- Installable Python project with the `signlang` CLI entry point.
- Manifest preparation, landmark extraction, baseline/GRU training, evaluation, and local webcam commands.
- Evaluation reports, automated tests, tracking diagnostics, and a public browser research preview.
- Selected baseline accuracy of 38/100 on the available offline test clips; incomplete WLASL100 data coverage.

## First package release

Complete these in order, with tested behavior and documentation at each step:

1. Add saved-video prediction with ranked vocabulary suggestions and structured JSON output.
2. Expose a documented Python API for loading a model, extracting landmarks, and predicting a video.
3. Define input validation, useful errors, exit codes, and model/preprocessing metadata.
4. Separate optional training dependencies where practical; document supported Python/platform combinations.
5. Build wheel and source distributions, verify installation in a clean environment, and add package/CLI checks to CI.
6. Add an explicit source-code license, contribution guidance, changelog, and separate dataset/model notices. Verify model distribution permissions and package-name availability before a registry release.
7. Publish a versioned package release and update GitHub installation examples and the website's links.

Version 0.2.0 implements `predict`, structured JSON output, the public Python API, numeric baseline export, optional dependency groups, distribution validation, and clean-install checks. It adds an MIT license for original source code and separate dataset/model notices. See the [package guide](PACKAGE_GUIDE.md) and [changelog](../CHANGELOG.md).

Distribute the alpha wheel and source archive through GitHub Releases. Package-registry publication remains a separate step requiring verified package-name ownership and publisher setup; no PyPI name is reserved or PyPI release claimed. Keep trained weights and datasets out of package distributions while their rights and quality remain unresolved.

## Evidence and boundaries

- Preserve official dataset splits, exact evaluated IDs, model provenance, and honest denominators.
- Packaging does not establish improved recognition accuracy. Measure model changes separately.
- Browser-upload and live-webcam accuracy remain unmeasured. The owner has deferred the webcam check.
- Continuous sentence translation, ISL recognition, and dependable interpreting are outside this release's scope.
- Do not bundle source videos, cached participant landmarks, or private recordings in package releases.
- Document source-code rights separately from WLASL and model terms; a source-code license does not replace upstream terms.

## Later opportunities

Prioritize dataset-quality tooling and human-reviewed annotation suggestions. Consider ASL practice feedback after evaluation with ASL-qualified reviewers and signers. Treat custom gesture controls as a separately defined vocabulary and evaluation task.
