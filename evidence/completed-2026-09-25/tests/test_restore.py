"""Archive safety and deterministic byte restoration, no scientific execution."""

import hashlib
import importlib.util
import io
import tarfile
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1] / "restore.py"
SPEC = importlib.util.spec_from_file_location("evidence_restore", PATH)
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)


def fixture(tmp_path, name="work/test/result.json", link=False):
    body = b'{"saved": true}\n'
    path = tmp_path / "example.tar.gz"
    with tarfile.open(path, "w:gz") as tar:
        info = tarfile.TarInfo(name)
        info.size = len(body)
        if link:
            info.type, info.linkname = tarfile.SYMTYPE, "/tmp/escape"
            tar.addfile(info)
        else:
            tar.addfile(info, io.BytesIO(body))
    return {
        "archives": [
            {"name": path.name, "bytes": path.stat().st_size, "sha256": restore.digest(path)}
        ],
        "files": [
            {
                "path": name,
                "member": name,
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
                "archive": path.name,
            }
        ],
    }


def test_roundtrip_and_no_overwrite(tmp_path):
    manifest = fixture(tmp_path)
    dest = tmp_path / "restored"
    assert restore.restore(manifest, tmp_path, dest) == 1
    assert restore.verify_files(manifest, dest) == 1
    with pytest.raises(FileExistsError):
        restore.restore(manifest, tmp_path, dest)


@pytest.mark.parametrize(
    "name", ["/tmp/escape", "work/../escape", "work//x", "work/./x", "elsewhere/a", "work\\x"]
)
def test_reject_unsafe_names(tmp_path, name):
    with pytest.raises(ValueError):
        restore.restore(fixture(tmp_path, name), tmp_path, tmp_path / "dest")


def test_reject_links_and_destination_symlink(tmp_path):
    manifest = fixture(tmp_path, link=True)
    with pytest.raises(ValueError):
        restore.restore(manifest, tmp_path, tmp_path / "dest")
    dest = tmp_path / "dest"
    dest.mkdir()
    (dest / "work").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        restore.restore(manifest, tmp_path, dest)


def test_reject_archive_and_member_tampering(tmp_path):
    manifest = fixture(tmp_path)
    manifest["archives"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        restore.restore(manifest, tmp_path, tmp_path / "dest")
    manifest["archives"][0]["sha256"] = restore.digest(tmp_path / "example.tar.gz")
    manifest["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Member hash"):
        restore.restore(manifest, tmp_path, tmp_path / "dest")
