"""Restore only this incremental package; --fixtures is local and solver-free."""

import argparse
import importlib.util
import json
import shutil
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "event_operator_restore_helpers",
    HERE.parent / "organization-response-2026-09-25/restore.py",
)
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)
helpers.PREFIX = "work/mediated_patterns/event_age_operator_family/"


def manifest(fixtures=False):
    m = json.loads((HERE / "artifacts.json").read_text())
    helpers.validate_manifest(m)
    for a in m["archives"]:
        if Path(a["name"]).name != a["name"]:
            raise ValueError("Local archive basename required")
        if not a["local"] and not a["url"].startswith(
            "https://github.com/gomercin/JEPA-Anything-experiments/releases/download/event-age-operator-family-"
        ):
            raise ValueError("Unexpected evidence asset host/path")
    if fixtures:
        m["archives"] = [a for a in m["archives"] if a["local"]]
        names = {a["name"] for a in m["archives"]}
        m["files"] = [r for r in m["files"] if r["archive"] in names]
        m["bytes"] = sum(r["bytes"] for r in m["files"])
    return m


def unpack(destination, fixtures=False, archives=None):
    m = manifest(fixtures)
    destination = destination.absolute()
    if any(p.is_symlink() for p in [destination, *destination.parents]):
        raise ValueError("Symlink destination forbidden")
    if archives is not None:
        return helpers.paths.restore(m, archives, destination)
    if fixtures:
        return helpers.paths.restore(m, HERE, destination)
    with tempfile.TemporaryDirectory(
        prefix="operator-assets-", dir=Path(tempfile.gettempdir()).resolve()
    ) as tmp:
        cache = Path(tmp)
        for a in m["archives"]:
            if a["local"]:
                with (
                    (HERE / a["name"]).open("rb") as source,
                    (cache / a["name"]).open("xb") as target,
                ):
                    shutil.copyfileobj(source, target)
            else:
                helpers.paths.download({"archives": [a]}, cache)
        return helpers.paths.restore(m, cache, destination)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path)
    p.add_argument(
        "--archives",
        type=Path,
        help="Optional folder containing both predownloaded archives",
    )
    p.add_argument(
        "--fixtures",
        action="store_true",
        help="Only the local regression subset; no network",
    )
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    if args.verify_only and args.destination:
        count = helpers.verify(manifest(args.fixtures), args.destination)
    elif args.verify_only:
        with tempfile.TemporaryDirectory(
            prefix="verify-operator-", dir=Path(tempfile.gettempdir()).resolve()
        ) as tmp:
            count = unpack(Path(tmp), args.fixtures, args.archives)
    elif args.destination:
        count = unpack(args.destination, args.fixtures, args.archives)
    else:
        p.error("Choose --verify-only or a new --destination")
    print(
        json.dumps(
            {
                "verified_members": count,
                "fixtures_only": args.fixtures,
                "experiments_executed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
