"""Only file/serialization safety; no scientific model is imported or advanced."""

import hashlib
import importlib.util
import json
import re
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "organization_restore", HERE / "restore.py"
)
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)


def fixture(tmp_path):
    name = restore.PREFIX + "fixture.json"
    source = tmp_path / "source"
    path = source / name
    path.parent.mkdir(parents=True)
    body = b'{"saved":true}\n'
    path.write_bytes(body)
    manifest = {
        "files": [
            {
                "path": name,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
        ]
    }
    return manifest, source, tmp_path / "destination"


def test_copy_all_bytes_and_refuse_existing_destination(tmp_path):
    manifest, source, destination = fixture(tmp_path)
    assert restore.restore(manifest, source, destination) == 1
    assert restore.verify(manifest, destination) == 1
    with pytest.raises(FileExistsError):
        restore.restore(manifest, source, destination)


@pytest.mark.parametrize(
    "name",
    [
        "/tmp/escape",
        "work/../escape",
        "work//a",
        "work/./a",
        "work\\a",
        "work/other/data",
    ],
)
def test_unsafe_or_out_of_scope_path(tmp_path, name):
    manifest, source, destination = fixture(tmp_path)
    manifest["files"][0]["path"] = name
    with pytest.raises(ValueError):
        restore.restore(manifest, source, destination)
    assert not destination.exists()


def test_tampering_duplicate_and_symlink_rejected_before_copy(tmp_path):
    manifest, source, destination = fixture(tmp_path)
    manifest["files"].append(manifest["files"][0].copy())
    with pytest.raises(ValueError, match="duplicate"):
        restore.restore(manifest, source, destination)
    manifest["files"].pop()
    path = source / manifest["files"][0]["path"]
    path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        restore.restore(manifest, source, destination)
    assert not destination.exists()
    path.unlink()
    path.symlink_to(tmp_path / "unrelated")
    with pytest.raises(ValueError, match="Symlink"):
        restore.restore(manifest, source, destination)


def test_destination_parent_symlink_rejected(tmp_path):
    manifest, source, destination = fixture(tmp_path)
    destination.mkdir()
    (destination / "work").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="Symlink"):
        restore.restore(manifest, source, destination)


def test_snapshot_member_integrity_and_saved_serialization():
    manifest = json.loads((HERE / "artifacts.json").read_text())
    assert restore.verify(manifest, HERE / "data") == 163
    spec = importlib.util.spec_from_file_location(
        "organization_saved", HERE / "saved_data_smoke.py"
    )
    saved = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(saved)
    result = saved.inspect(HERE / "data")
    assert result["experiments_executed"] is False
    assert result["criteria"]["selectivity_nrmse"] == 0.05
    assert len(result["saved_comparator_table"]) == 9


def test_source_records_and_portable_index_links():
    records = json.loads((HERE / "source-records.json").read_text())
    for row in records["files"]:
        body = (HERE / row["preserved_path"]).read_bytes()
        assert len(body) == row["bytes"]
        assert hashlib.sha256(body).hexdigest() == row["sha256"]
    for link in re.findall(
        r"!?\[[^\]]*\]\(([^)]+)\)", (HERE / "README.md").read_text()
    ):
        if not link.startswith(("https://", "http://", "#")):
            assert (HERE / link.split("#", 1)[0]).is_file(), link
