"""Verify/restore only the new present-state data using the existing safe-copy convention."""

import argparse
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "present_state_copy_helpers", HERE.parent / "organization-response-2026-09-25/restore.py"
)
helpers = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(helpers)
helpers.PREFIX = "work/mediated_patterns/present_state_transmission/"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--destination", type=Path)
    p.add_argument("--verify-only", action="store_true")
    args = p.parse_args()
    manifest = json.loads((HERE / "artifacts.json").read_text())
    if args.verify_only:
        count = helpers.verify(manifest, args.destination or HERE / "data")
    elif args.destination:
        count = helpers.restore(manifest, HERE / "data", args.destination)
    else:
        p.error("Choose --verify-only or a new --destination")
    print(json.dumps({"verified_members": count, "experiments_executed": False}))


if __name__ == "__main__":
    main()
