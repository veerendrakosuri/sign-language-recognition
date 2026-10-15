"""Package a buildless Sites checkout on Windows without Bash/WSL."""

import argparse
import json
import tarfile
import tempfile
from pathlib import Path


def package(project: Path, archive: Path):
    project = project.resolve()
    source_manifest = project / ".openai" / "hosting.json"
    source_dist = project / "dist"
    manifest = json.loads(source_manifest.read_text(encoding="utf-8"))
    if manifest.get("static", {}).get("directory") != "dist" or not (source_dist / "index.html").is_file():
        raise ValueError("Expected a static Site with dist/index.html")
    for path in source_dist.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Refusing symlink in deployment: {path}")
    archive.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=project) as temporary:
        staged = Path(temporary) / "dist"
        import shutil
        shutil.copytree(source_dist, staged)
        (staged / ".openai").mkdir(exist_ok=True)
        (staged / ".openai" / "hosting.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(staged, arcname="dist")
    print(f"Packaged {archive} for project {manifest['project_id']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    parser.add_argument("archive", type=Path)
    arguments = parser.parse_args()
    package(arguments.project, arguments.archive)
