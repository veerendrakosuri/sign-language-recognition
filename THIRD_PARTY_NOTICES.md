# Source code, data, and model terms

The MIT license applies to original project source code and documentation. It does not grant rights to third-party datasets, model weights, recordings, or dependency code.

The wheel and source distribution contain no WLASL videos, metadata, extracted participant landmarks, trained checkpoints, or MediaPipe detector assets. Users supply their own authorized model files and videos.

## WLASL and experimental weights

WLASL is distributed under the authors' C-UDA terms. The authors' repository states that WLASL data are intended for academic and computational use and prohibits commercial usage. Read the actual agreement and applicable source-video terms:

- https://github.com/dxli94/WLASL
- https://github.com/dxli94/WLASL/blob/master/start_kit/C-UDA-1.0.pdf

The GitHub repository's `web/public/model.json` is an experimental baseline derived from an incomplete WLASL mirror. It is excluded from package distributions and is not covered by the project's MIT source-code license. Its presence in GitHub does not establish unrestricted model rights or validated accuracy. Verify applicable permissions before redistributing or deploying a derived model, particularly commercially. Mirror provenance and clip labels have not been independently verified.

## MediaPipe and dependencies

`signlang download-models` downloads detector assets from Google's model hosting. Those assets locate hands/body points and do not classify ASL words. Their terms are separate from this project. Review MediaPipe's official documentation and notices:

- https://github.com/google-ai-edge/mediapipe
- https://developers.google.com/edge/mediapipe/solutions/tasks

Dependencies retain their respective licenses. Package installation does not bundle dependency source code into this project's distribution archive.
