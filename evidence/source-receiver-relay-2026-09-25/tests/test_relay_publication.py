"""Read-only serialization and provenance checks; no scientific execution."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("relay_restore", HERE / "restore.py")
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)
DATA = HERE / "data/work/mediated_patterns/source_receiver_relay"


def test_snapshot_integrity_and_nonoverwriting_restore(tmp_path):
    manifest = json.loads((HERE / "artifacts.json").read_text())
    assert sum(r["bytes"] for r in manifest["files"]) < 25 * 1024**2
    assert restore.helpers.verify(manifest, HERE / "data") == len(manifest["files"])
    # The same helper's traversal/symlink fixtures remain in the earlier tests.
    subset = {"files": manifest["files"][:1]}
    restore.helpers.restore(subset, HERE / "data", tmp_path / "copy")
    with pytest.raises(FileExistsError):
        restore.helpers.restore(subset, HERE / "data", tmp_path / "copy")


def test_saved_fresh_contrasts_and_prospective_predictions():
    freeze = json.loads((DATA / "frozen-01/freeze.json").read_text())
    copied = json.loads((DATA / "fresh-01/freeze-copy.json").read_text())
    assert freeze == copied
    assert set(freeze["development_seeds"]).isdisjoint(freeze["fresh_seeds"])
    assert len(freeze["fresh_seeds"]) == 3
    for seed in freeze["fresh_seeds"]:
        for b in freeze["positions"]:
            name = f"s{seed}-b{b:g}"
            z = np.load(DATA / f"fresh-01/{name}.npz", allow_pickle=False)
            a = z["absolute"]
            np.testing.assert_array_equal(z["full"], a[:, 2::2] - a[:, 0, None])
            np.testing.assert_array_equal(z["cut"], a[:, 3::2] - a[:, 1, None])
            np.testing.assert_array_equal(z["qb"], z["full"] - z["cut"])
            p = np.load(DATA / f"fresh-01/{name}-prediction.npz", allow_pickle=False)
            np.testing.assert_array_equal(z["tangent_full"], p["tangent_full"])
            meta = json.loads((DATA / f"fresh-01/{name}.json").read_text())
            assert meta["amplitudes"] == [0.02, 0.03]
            assert meta["sham_field_max"] < 1e-12
    summary = json.loads((DATA / "analysis-01/summary.json").read_text())
    assert len(summary["organizations"]) == 6
    assert all(r["geometry"]["qualified"] for r in summary["organizations"])
