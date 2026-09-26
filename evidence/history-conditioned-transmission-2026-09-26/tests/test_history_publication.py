"""Read-only serialization and provenance checks; no scientific execution."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("history_restore", HERE / "restore.py")
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)
DATA = HERE / "data/work/mediated_patterns/history_conditioned_transmission"


def test_snapshot_integrity_and_nonoverwriting_restore(tmp_path):
    manifest = json.loads((HERE / "artifacts.json").read_text())
    assert sum(r["bytes"] for r in manifest["files"]) < 25 * 1024**2
    assert restore.helpers.verify(manifest, HERE / "data") == len(manifest["files"])
    # The same helper's traversal/symlink fixtures remain in the earlier tests.
    subset = {"files": manifest["files"][:1]}
    restore.helpers.restore(subset, HERE / "data", tmp_path / "copy")
    with pytest.raises(FileExistsError):
        restore.helpers.restore(subset, HERE / "data", tmp_path / "copy")


def test_saved_matched_histories_and_prediction_order():
    freeze = json.loads((DATA / "frozen-01/freeze.json").read_text())
    copied = json.loads((DATA / "fresh-01/freeze-copy.json").read_text())
    assert freeze == copied
    assert set(freeze["development_seeds"]).isdisjoint(freeze["fresh_seeds"])
    assert len(freeze["fresh_seeds"]) == 3
    budget = json.loads((DATA / "fresh-01/budget.json").read_text())
    order = [s["task"] for s in budget["sections"]]
    for seed in freeze["fresh_seeds"]:
        history = np.load(DATA / f"fresh-01/s{seed}-history.npz", allow_pickle=False)
        preparation = np.load(
            DATA / f"fresh-01/s{seed}-preparation.npz", allow_pickle=False
        )
        np.testing.assert_array_equal(history["initial"], preparation["initial"])
        np.testing.assert_array_equal(history["post_write"][0], preparation["initial"])
        np.testing.assert_array_equal(
            history["post_write"][0, 1], history["post_write"][1, 1]
        )
        for wait in freeze["waits"]:
            for written in (False, True):
                name = f"s{seed}-t{wait:g}-" + ("written" if written else "unwritten")
                assert order.index(name + "-prediction") < order.index(name)
                z = np.load(DATA / f"fresh-01/{name}.npz", allow_pickle=False)
                a = z["absolute"]
                # Integer-time probe shams continue the exact aged history;
                # the other history and a fresh preparation are not substitutes.
                np.testing.assert_allclose(
                    a[::2, 0],
                    history["absolute"][int(wait) : int(wait) + 81, int(written)],
                    atol=1e-12,
                    rtol=0,
                )
                np.testing.assert_array_equal(z["full"], a[:, 2::2] - a[:, 0, None])
                np.testing.assert_array_equal(z["cut"], a[:, 3::2] - a[:, 1, None])
                np.testing.assert_array_equal(z["qb"], z["full"] - z["cut"])
                np.testing.assert_array_equal(z["absolute_time"], z["time"] + wait)
                p = np.load(
                    DATA / f"fresh-01/{name}-prediction.npz", allow_pickle=False
                )
                np.testing.assert_array_equal(z["tangent_full"], p["tangent_full"])
                meta = json.loads((DATA / f"fresh-01/{name}.json").read_text())
                assert meta["amplitudes"] == [0.02]
                assert meta["sham_field_max"] < 1e-12
                # This is a serialization check of the frozen population, not a
                # test demanding that another nonlinear world retain a change.
                assert meta["qualified"]
    summary = json.loads((DATA / "analysis-01/summary.json").read_text())
    assert len(summary["histories"]) == 3
    assert len(summary["comparisons"]) == 9
