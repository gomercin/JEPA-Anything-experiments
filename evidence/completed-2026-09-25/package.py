"""Package the two completed work trees; no simulator, training, or fitting imports."""

import argparse
import gzip
import hashlib
import json
import tarfile
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TAG = "evidence-completed-2026-09-25"
REPO = "gomercin/JEPA-Anything-experiments"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cache(path):
    return "__pycache__" in path.parts or path.name == ".DS_Store" or path.name.startswith("._")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": 1, "repository": REPO, "tag": TAG, "archives": [], "files": []}
    duplicates = defaultdict(list)
    excluded = []
    for family in ("coupled_oscillators", "mediated_patterns"):
        base = ROOT / "work" / family
        paths = sorted(p for p in base.rglob("*") if p.is_file())
        name = family + ".tar.gz"
        archive = args.output / name
        with archive.open("xb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0, compresslevel=6) as gz:
                with tarfile.open(fileobj=gz, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                    for path in paths:
                        if path.is_symlink():
                            raise ValueError(f"Symlink not allowed: {path}")
                        rel = path.relative_to(ROOT).as_posix()
                        size, sha = path.stat().st_size, digest(path)
                        if cache(path):
                            excluded.append(
                                {"path": rel, "bytes": size, "classification": "replaceable_cache"}
                            )
                            continue
                        context = []
                        parent = path.parent
                        while parent != base.parent:
                            for candidate in (
                                "protocol.json",
                                "freeze.json",
                                "COMMANDS.md",
                                "commands.txt",
                                "session_manifest.json",
                            ):
                                p = parent / candidate
                                if p.is_file() and p != path:
                                    context.append(p.relative_to(ROOT).as_posix())
                            parent = parent.parent
                        manifest["files"].append(
                            {
                                "path": rel,
                                "bytes": size,
                                "sha256": sha,
                                "archive": name,
                                "member": rel,
                                "provenance_context": context,
                            }
                        )
                        duplicates[sha].append(rel)
                        info = tarfile.TarInfo(rel)
                        info.size, info.mode, info.mtime = size, 0o644, 0
                        with path.open("rb") as stream:
                            tar.addfile(info, stream)
        manifest["archives"].append(
            {
                "name": name,
                "bytes": archive.stat().st_size,
                "sha256": digest(archive),
                "url": f"https://github.com/{REPO}/releases/download/{TAG}/{name}",
            }
        )
        print(name, archive.stat().st_size, flush=True)
    total = sum(a["bytes"] for a in manifest["archives"])
    if total > 500 * 2**20:
        raise RuntimeError(f"Archive cap exceeded: {total}; nothing authorized for upload")
    for name, data in (
        ("artifacts.json", manifest),
        ("excluded-caches.json", excluded),
        ("duplicate-bytes.json", {k: v for k, v in duplicates.items() if len(v) > 1}),
    ):
        with (HERE / name).open("x") as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write("\n")
    print("Total compressed bytes", total, "members", len(manifest["files"]))


if __name__ == "__main__":
    main()
