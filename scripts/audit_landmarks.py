"""Compatibility wrapper for the supported ``signlang audit`` command."""

import argparse
import json
from pathlib import Path

from signlang.audit import audit_landmarks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--cache", default="data/landmarks")
    parser.add_argument("--output", default="runs/landmark-audit.json")
    args = parser.parse_args()
    report = audit_landmarks(args.manifest, args.cache, args.output)
    print(json.dumps({
        key: report[key]
        for key in ("cached_clips", "missing_cache_count", "malformed_cache_count", "all")
    }, indent=2))
    print(f"Full report: {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
