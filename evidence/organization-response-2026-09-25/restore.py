"""Explicitly restore only this Git-backed incremental snapshot; no network/science.

Reuse the earlier snapshot's unchanged path and checksum helpers, not its CLI
or manifest. Python 3.11+. Never downloads inherited evidence.
"""

import argparse
import importlib.util
import json
import re
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "completed_evidence_paths", HERE.parent / "completed-2026-09-25/restore.py"
)
paths = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(paths)
PREFIX = "work/mediated_patterns/organization_response/"


def validate_manifest(manifest):
    seen = set()
    for row in manifest["files"]:
        name = row["path"]
        paths.safe_name(name)
        if not name.startswith(PREFIX) or name in seen:
            raise ValueError(f"Unexpected or duplicate incremental path: {name}")
        if not isinstance(row["bytes"], int) or row["bytes"] < 0:
            raise ValueError("Invalid member size")
        if not re.fullmatch(r"[0-9a-f]{64}", row["sha256"]):
            raise ValueError("Invalid SHA-256")
        seen.add(name)
    if not seen:
        raise ValueError("Empty snapshot")


def verify(manifest, root):
    validate_manifest(manifest)
    if root.is_symlink():
        raise ValueError("Root must not be a symlink")
    return paths.verify_files(manifest, root.absolute())


def restore(manifest, source, destination):
    # Validate every source and target before copying any file.
    verify(manifest, source)
    if destination.is_symlink():
        raise ValueError("Destination must not be a symlink")
    destination = destination.absolute()
    for row in manifest["files"]:
        if paths.target(destination, row["path"]).exists():
            raise FileExistsError(row["path"])
    for row in manifest["files"]:
        src = paths.target(source.absolute(), row["path"])
        dst = paths.target(destination, row["path"])
        dst.parent.mkdir(parents=True, exist_ok=True)
        with src.open("rb") as original, dst.open("xb") as copy:
            shutil.copyfileobj(original, copy)
        if paths.digest(dst) != row["sha256"]:
            raise ValueError(f"Copy mismatch; partial restore retained: {row['path']}")
    return verify(manifest, destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    manifest = json.loads((HERE / "artifacts.json").read_text())
    source = HERE / "data"
    if args.verify_only:
        count = verify(manifest, args.destination or source)
    elif args.destination:
        count = restore(manifest, source, args.destination)
    else:
        p.error("Choose --destination for explicit copying, or --verify-only")
    print(json.dumps({"verified_members": count, "experiments_executed": False}))


if __name__ == "__main__":
    main()
