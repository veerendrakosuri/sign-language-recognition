import argparse
import hashlib
import json
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
    p = argparse.ArgumentParser(description="Project 3: isolated ASL word recognition")
    commands = p.add_subparsers(dest="command", required=True)
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
            cmd.add_argument("--split", choices=["val", "test"], default="test")
            cmd.add_argument("--device", default="cpu")
    demo = commands.add_parser("webcam", help="Local camera demo; requires your trained checkpoint")
    demo.add_argument("--kind", choices=["gru", "baseline"], default="gru")
    demo.add_argument("--checkpoint", default="runs/train/best.pt")
    demo.add_argument("--models", default="models")
    demo.add_argument("--camera", type=int, default=0)
    demo.add_argument("--device", default="cpu")
    demo.add_argument("--threshold", type=float, default=.75)
    demo.add_argument("--repeats", type=int, default=3)
    demo.add_argument("--predict-every", type=int, default=3)
    demo.add_argument("--window-seconds", type=float, default=1.5)
    return p


def main():
    args = parser().parse_args()
    if args.command == "download-models":
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


if __name__ == "__main__":
    main()
