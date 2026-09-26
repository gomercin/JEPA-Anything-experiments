"""Verify or exclusively restore this local compressed incremental evidence only."""

import argparse
import importlib.util
import json
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "repeated_intervention_copy_helpers",
    HERE.parent / "organization-response-2026-09-25/restore.py",
)
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)
helpers.PREFIX = "work/mediated_patterns/repeated_intervention_state/"


def unpack(destination):
    manifest = json.loads((HERE / "artifacts.json").read_text())
    helpers.validate_manifest(manifest)
    destination = destination.absolute()
    if any(p.is_symlink() for p in [destination, *destination.parents]):
        raise ValueError("Symlink destination forbidden")
    for archive in manifest["archives"]:
        if Path(archive["name"]).name != archive["name"]:
            raise ValueError("Archive must be local basename")
    # Existing archive helper preflights hashes, every member, links, extras,
    # traversal and all collisions before creating any member exclusively.
    return helpers.paths.restore(manifest, HERE, destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads((HERE / "artifacts.json").read_text())
    if args.verify_only and args.destination:
        count = helpers.verify(manifest, args.destination)
    elif args.verify_only:
        with tempfile.TemporaryDirectory(
            prefix="verify-repeated-evidence-",
            dir=Path(tempfile.gettempdir()).resolve(),
        ) as tmp:
            count = unpack(Path(tmp))
    elif args.destination:
        count = unpack(args.destination)
    else:
        parser.error("Choose --verify-only or a new --destination")
    print(json.dumps({"verified_members": count, "experiments_executed": False}))


if __name__ == "__main__":
    main()
