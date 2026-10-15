"""Verify distribution contents and dependency boundaries before release."""
from email.parser import Parser
from pathlib import Path
import tarfile
import zipfile


def check(root="dist"):
    wheels = list(Path(root).glob("*.whl"))
    sources = list(Path(root).glob("*.tar.gz"))
    if len(wheels) != 1 or len(sources) != 1:
        raise ValueError("Expected exactly one wheel and one source distribution")
    banned = {"data", "models", "runs", "web", "reports", ".venv", ".git", ".site-deploy"}
    extensions = {".mp4", ".mov", ".avi", ".npz", ".npy", ".pt", ".joblib", ".task"}
    for archive in [*wheels, *sources]:
        if archive.suffix == ".whl":
            with zipfile.ZipFile(archive) as file:
                names = file.namelist()
                metadata = Parser().parsestr(file.read(next(n for n in names if n.endswith(".dist-info/METADATA"))).decode())
                base = [line for line in metadata.get_all("Requires-Dist", []) if "extra ==" not in line]
                assert len(base) == 1 and base[0].startswith("numpy"), base
                assert metadata["License-Expression"] == "MIT"
        else:
            with tarfile.open(archive) as file:
                names = file.getnames()
        for name in names:
            path = Path(name)
            assert not set(path.parts) & banned, name
            assert path.suffix.lower() not in extensions, name
            assert path.name != ".env", name
        print(f"Distribution content check passed: {archive.name} ({len(names)} entries)")


if __name__ == "__main__":
    check()
