"""Explicit, hash-checked evidence restoration. Does not import experiment code."""

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path, PurePosixPath

HERE = Path(__file__).resolve().parent


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def safe_name(name):
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or str(path) != name
        or not path.parts
        or path.parts[0] != "work"
    ):
        raise ValueError(f"Unsafe archive path: {name!r}")
    return path


def target(root, name):
    path = root.joinpath(*safe_name(name).parts)
    for parent in (path, *path.parents):
        if parent.is_symlink():
            raise ValueError(f"Symlink in destination: {parent}")
        if parent == root:
            break
    return path


def verify_files(manifest, destination):
    for row in manifest["files"]:
        p = target(destination, row["path"])
        if p.stat().st_size != row["bytes"] or digest(p) != row["sha256"]:
            raise ValueError(f"Restored member mismatch: {row['path']}")
    return len(manifest["files"])


def restore(manifest, archives, destination):
    """Preflight all archives and targets; reject extra members, links and overwrites."""
    expected = {r["member"]: r for r in manifest["files"]}
    if len(expected) != len(manifest["files"]):
        raise ValueError("Duplicate manifest members")
    for row in expected.values():
        if row["path"] != row["member"]:
            raise ValueError("This snapshot restores only original relative paths")
        if target(destination, row["path"]).exists():
            raise FileExistsError(row["path"])
    seen = set()
    for archive in manifest["archives"]:
        p = archives / archive["name"]
        if p.stat().st_size != archive["bytes"] or digest(p) != archive["sha256"]:
            raise ValueError(f"Archive checksum/size mismatch: {p.name}")
        with tarfile.open(p, "r:gz") as tar:
            for member in tar:
                safe_name(member.name)
                row = expected.get(member.name)
                if (
                    not member.isfile()
                    or row is None
                    or member.name in seen
                    or row["archive"] != archive["name"]
                    or member.size != row["bytes"]
                ):
                    raise ValueError(f"Unexpected archive member: {member.name}")
                seen.add(member.name)
    if seen != set(expected):
        raise ValueError("Archive member set incomplete")
    for archive in manifest["archives"]:
        with tarfile.open(archives / archive["name"], "r:gz") as tar:
            for member in tar:
                p = target(destination, member.name)
                p.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as source, p.open("xb") as out:
                    shutil.copyfileobj(source, out)
                row = expected[member.name]
                if digest(p) != row["sha256"]:
                    raise ValueError(
                        f"Member hash mismatch: {member.name}; incomplete restore retained"
                    )
    return verify_files(manifest, destination)


def download(manifest, output):
    for archive in manifest["archives"]:
        if Path(archive["name"]).name != archive["name"]:
            raise ValueError("Unsafe asset name")
        request = urllib.request.Request(
            archive["url"], headers={"User-Agent": "jepa-evidence-restore"}
        )
        count = 0
        with urllib.request.urlopen(request, timeout=60) as response:
            with (output / archive["name"]).open("xb") as out:
                while block := response.read(1024 * 1024):
                    count += len(block)
                    if count > archive["bytes"]:
                        raise ValueError("Downloaded asset exceeds declared size")
                    out.write(block)
        if count != archive["bytes"]:
            raise ValueError("Incomplete asset download")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--archives", type=Path, help="Explicit predownloaded assets; still verified"
    )
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.destination.is_symlink():
        raise ValueError("Destination must not be a symlink")
    destination = args.destination.absolute()
    manifest = json.loads((HERE / "artifacts.json").read_text())
    if args.verify_only:
        count = verify_files(manifest, destination)
    elif args.archives:
        count = restore(manifest, args.archives, destination)
    else:
        with tempfile.TemporaryDirectory(prefix="jepa-evidence-download-") as temp:
            folder = Path(temp)
            download(manifest, folder)
            count = restore(manifest, folder, destination)
    print(
        json.dumps(
            {
                "verified_members": count,
                "destination": str(destination),
                "experiments_executed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
