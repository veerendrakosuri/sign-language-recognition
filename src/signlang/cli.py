import argparse
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from urllib.request import urlopen


def download_models(folder):
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    urls = {
        "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
    }
    checksums = {}
    for name, url in urls.items():
        path = root / name
        if not path.exists():
            print(f"Downloading {name} from Google...")
            with urlopen(url, timeout=120) as response:
                content = response.read()
            if len(content) < 100_000:
                raise ValueError(f"Unexpectedly small model: {name}")
            temporary = path.with_suffix(".download")
            temporary.write_bytes(content)
            temporary.replace(path)
        checksums[name] = dict(url=url, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    from .data import write_json
    write_json(root / "model_sources.json", checksums)
    print(f"Models ready in {root}")


def parser():
    from . import __version__
    p = argparse.ArgumentParser(description="Developer toolkit for experimental isolated ASL word recognition")
    p.add_argument("--version", action="version", version=f"signlang {__version__}")
    commands = p.add_subparsers(dest="command", required=True)
    predict = commands.add_parser("predict", help="Predict one trimmed video or saved landmark sequence; emit JSON")
    predict.add_argument("input", help="Trimmed video, or an NPZ with points/valid when --input-type landmarks")
    predict.add_argument("--checkpoint", required=True, help="JSON baseline, trusted .joblib baseline, or .pt GRU")
    predict.add_argument("--kind", choices=["auto", "baseline-json", "baseline", "gru"], default="auto")
    predict.add_argument("--input-type", choices=["video", "landmarks"], default="video")
    predict.add_argument("--models", default="models", help="Directory containing MediaPipe detector assets")
    predict.add_argument("--device", default="cpu")
    predict.add_argument("--top-k", type=int, default=5)
    predict.add_argument("--output", default="-", help="JSON file (must not exist), or - for stdout")
    predict.add_argument("--trust-checkpoint", action="store_true", help="Allow joblib loading only for a checkpoint you trust")
    export = commands.add_parser("export-model", help="Export a trusted logistic baseline as numeric JSON")
    export.add_argument("--checkpoint", required=True)
    export.add_argument("--output", required=True, help="Destination JSON file; must not exist")
    export.add_argument("--trust-checkpoint", action="store_true")
    download = commands.add_parser("download-models", help="Download Google's hand/pose detector assets")
    download.add_argument("--models", default="models")
    manifest = commands.add_parser("manifest", help="Index pre-trimmed WLASL clips with official splits")
    manifest.add_argument("--metadata", required=True)
    manifest.add_argument("--videos", default="data/videos")
    manifest.add_argument("--output", default="data/processed")
    manifest.add_argument("--classes", type=int, default=100)
    extract = commands.add_parser("extract", help="Cache landmarks for available clips")
    extract.add_argument("--manifest", default="data/processed/manifest.csv")
    extract.add_argument("--cache", default="data/landmarks")
    extract.add_argument("--models", default="models")
    for name in ("baseline", "train", "evaluate"):
        cmd = commands.add_parser(name)
        cmd.add_argument("--manifest", default="data/processed/manifest.csv")
        cmd.add_argument("--labels", default="data/processed/labels.json")
        cmd.add_argument("--cache", default="data/landmarks")
        cmd.add_argument("--output", default=f"runs/{name}")
        if name != "evaluate":
            cmd.add_argument("--frames", type=int, default=32)
            cmd.add_argument("--seed", type=int, default=42)
        if name == "train":
            cmd.add_argument("--epochs", type=int, default=60)
            cmd.add_argument("--batch-size", type=int, default=32)
            cmd.add_argument("--hidden", type=int, default=256)
            cmd.add_argument("--lr", type=float, default=1e-3)
            cmd.add_argument("--patience", type=int, default=10)
            cmd.add_argument("--device", default="cpu")
            cmd.add_argument("--mirror-probability", type=float, default=0.0)
        if name == "evaluate":
            cmd.add_argument("--kind", choices=["baseline", "gru"], required=True)
            cmd.add_argument("--checkpoint", required=True)
            cmd.add_argument("--trust-checkpoint", action="store_true", help="Required for trusted joblib baseline files")
            cmd.add_argument("--split", choices=["val", "test"], default="test")
            cmd.add_argument("--device", default="cpu")
    demo = commands.add_parser("webcam", help="Local camera demo; requires your trained checkpoint")
    demo.add_argument("--kind", choices=["gru", "baseline"], default="gru")
    demo.add_argument("--checkpoint", default="runs/train/best.pt")
    demo.add_argument("--trust-checkpoint", action="store_true", help="Required for trusted joblib baseline files")
    demo.add_argument("--models", default="models")
    demo.add_argument("--camera", type=int, default=0)
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--threshold", type=float, default=.75)
    demo.add_argument("--repeats", type=int, default=3)
    demo.add_argument("--predict-every", type=int, default=3)
    demo.add_argument("--window-seconds", type=float, default=1.5)
    return p


def _run(args):
    if args.command in {"evaluate", "webcam"} and args.kind == "baseline" and not args.trust_checkpoint:
        raise ValueError("Joblib checkpoints require explicit trust: use --trust-checkpoint only for a file you trust")
    if args.command == "predict":
        from .api import SignRecognizer
        if args.top_k < 1:
            raise ValueError("top_k must be a positive integer")
        source = Path(args.input)
        if not source.is_file():
            raise FileNotFoundError(f"Input not found: {source}")
        if args.output != "-" and Path(args.output).exists():
            raise FileExistsError(f"Output already exists; choose a new path: {args.output}")
        recognizer = SignRecognizer.from_model(args.checkpoint, kind=args.kind,
                    device=args.device, models=args.models, trusted=args.trust_checkpoint)
        if args.input_type == "landmarks":
            import numpy as np
            sequence = np.load(source, allow_pickle=False)
            if not hasattr(sequence, "files"):
                raise ValueError("Landmark input must be an NPZ archive containing points and valid")
            with sequence:
                if not {"points", "valid"} <= set(sequence.files):
                    raise ValueError("Landmark NPZ must contain points and valid arrays")
                result = recognizer.predict_landmarks(sequence["points"], sequence["valid"], top_k=args.top_k)
            result = replace(result, input={**result.input, "name": source.name})
        else:
            result = recognizer.predict_video(source, top_k=args.top_k)
        payload = json.dumps(result.to_dict(), indent=2, allow_nan=False) + "\n"
        if args.output == "-":
            sys.stdout.write(payload)
        else:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8") as destination:
                destination.write(payload)
    elif args.command == "export-model":
        from .serialization import export_baseline
        destination = export_baseline(args.checkpoint, args.output, trusted=args.trust_checkpoint)
        sys.stderr.write(f"Exported model to {destination}\n")
    elif args.command == "download-models":
        download_models(args.models)
    elif args.command == "manifest":
        from .data import make_manifest
        make_manifest(args.metadata, args.videos, args.output, args.classes)
    elif args.command == "extract":
        from .landmarks import extract_manifest
        extract_manifest(args.manifest, args.cache, args.models)
    elif args.command == "baseline":
        from .training import train_baseline
        train_baseline(args)
    elif args.command == "train":
        from .training import train_gru
        train_gru(args)
    elif args.command == "evaluate":
        from .evaluation import evaluate
        evaluate(args)
    elif args.command == "webcam":
        from .demo import webcam
        webcam(args)


def main(argv=None):
    """CLI entry point: 0 success, 1 runtime/input error, 2 usage error."""
    args = parser().parse_args(argv)
    try:
        _run(args)
    except (ValueError, OSError, RuntimeError, ImportError, EOFError) as error:
        sys.stderr.write(f"signlang: error: {error}\n")
        if isinstance(error, ImportError):
            sys.stderr.write('Optional workflows may require: pip install "sign-language-recognition[train,vision]"\n')
        return 1
    except KeyboardInterrupt:
        sys.stderr.write("signlang: interrupted\n")
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
