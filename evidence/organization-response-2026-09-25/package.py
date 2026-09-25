"""Copy only the extant incremental work tree into Git; no archive or science."""

import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
WORK = ROOT / "work/mediated_patterns/organization_response"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    with (HERE / name).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def main():
    data = HERE / "data"
    data.mkdir(exist_ok=False)
    files, excluded = [], []
    for path in sorted(WORK.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"Symlink not allowed: {path}")
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if (
            "__pycache__" in path.parts
            or path.name == ".DS_Store"
            or path.name.startswith("._")
        ):
            excluded.append({"path": rel, "classification": "replaceable_cache"})
            continue
        if path.suffix not in {".py", ".json", ".npz", ".md", ".png", ".txt"}:
            raise ValueError(f"Unexpected evidence file type: {path}")
        target = data / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        with path.open("rb") as src, target.open("xb") as dst:
            shutil.copyfileobj(src, dst)
        sha = digest(path)
        if digest(target) != sha:
            raise ValueError(f"Copy changed bytes: {path}")
        files.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha})
    total = sum(row["bytes"] for row in files)
    if total > 25 * 2**20:
        raise ValueError("Git evidence ceiling exceeded; do not stage or publish")
    write(
        "artifacts.json",
        {
            "schema": 1,
            "snapshot_id": "organization-response-2026-09-25",
            "identifier_source": "Date in the original prospective contract and completed lab record",
            "repository": "gomercin/JEPA-Anything-experiments",
            "storage": "git:data/<original-relative-path>",
            "archives": [],
            "archive_bytes": 0,
            "file_count": len(files),
            "original_bytes": total,
            "files": files,
        },
    )
    write("excluded-caches.json", excluded)
    # Preserve the completed handoff's exact original bytes before publication append.
    before = HERE / "execution-records"
    before.mkdir(exist_ok=False)
    handoff = ROOT / "experiments/mediated_patterns/README.md"
    shutil.copyfile(handoff, before / "lab-README-before-publication.md")
    records = []
    for name in (
        "organization_response.py",
        "ORGANIZATION_RESPONSE.md",
        "tests/test_organization_response.py",
        "README.md",
    ):
        path = ROOT / "experiments/mediated_patterns" / name
        records.append(
            {
                "original_path": path.relative_to(ROOT).as_posix(),
                "preserved_path": "execution-records/lab-README-before-publication.md"
                if name == "README.md"
                else "../../" + path.relative_to(ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
    report = ROOT / "experiments/mediated_patterns/ORGANIZATION_RESPONSE.md"
    pending = report.read_text().split("### Exact pending Atlas paragraph", 1)[1]
    write(
        "source-records.json",
        {
            "execution_base_revision": "1c555d5aab826418c81e0a9005a85200e3f8fd65",
            "execution_state": "Dirty-worktree execution; contemporaneous stage source snapshots are authoritative, not the later publication commit.",
            "files": records,
            "pending_atlas_suffix_sha256": hashlib.sha256(pending.encode()).hexdigest(),
        },
    )
    print(
        json.dumps(
            {
                "copied_files": len(files),
                "copied_bytes": total,
                "excluded_caches": len(excluded),
                "science_executed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
