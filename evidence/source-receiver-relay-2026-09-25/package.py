"""Snapshot this cycle once; no inherited data or local conversation files."""

import hashlib
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = REPO / "work/mediated_patterns/source_receiver_relay"


def main():
    destination = HERE / "data"
    if (
        destination.exists()
        or destination.is_symlink()
        or (HERE / "artifacts.json").exists()
    ):
        raise FileExistsError("Evidence snapshot already exists")
    rows = []
    for p in sorted(SOURCE.rglob("*")):
        if p.is_symlink():
            raise ValueError("Symlink in evidence source")
        if p.is_file():
            if p.suffix not in (".json", ".npz", ".png", ".txt", ".md"):
                raise ValueError("Unexpected source file: " + str(p))
            rows.append(
                {
                    "path": p.relative_to(REPO).as_posix(),
                    "bytes": p.stat().st_size,
                    "sha256": hashlib.sha256(p.read_bytes()).hexdigest(),
                }
            )
    if not rows or sum(r["bytes"] for r in rows) > 25 * 1024**2:
        raise ValueError("Empty package or generated Git data exceeds 25 MiB")
    destination.mkdir()
    for row in rows:
        target = destination / row["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / row["path"], target)
    (HERE / "artifacts.json").write_text(json.dumps({"files": rows}, indent=2) + "\n")
    print(json.dumps({"members": len(rows), "bytes": sum(r["bytes"] for r in rows)}))


if __name__ == "__main__":
    main()
