"""Read-only saved-evidence checks; never run a field simulation or refit."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.mediated_patterns import present_state_model as m

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data/work/mediated_patterns/present_state_transmission"
SPEC = importlib.util.spec_from_file_location("snapshot_restore", HERE / "restore.py")
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_package_integrity_and_exclusive_restore(tmp_path):
    manifest = load(HERE / "artifacts.json")
    assert sum(r["bytes"] for r in manifest["files"]) < 25 * 1024**2
    assert restore.helpers.verify(manifest, HERE / "data") == len(manifest["files"])
    subset = {"files": manifest["files"][:1]}
    restore.helpers.restore(subset, HERE / "data", tmp_path / "copy")
    with pytest.raises(FileExistsError):
        restore.helpers.restore(subset, HERE / "data", tmp_path / "copy")


def test_frozen_model_and_serialized_descriptor_only_predictions():
    frozen, fresh = DATA / "frozen-01", DATA / "fresh-01"
    contract = load(frozen / "freeze.json")
    seal = load(fresh / "prediction-seal.json")
    assert contract["sources"] == load(fresh / "protocol.json")["sources"]
    assert (
        contract["models_sha256"] == seal["model_sha256"] == sha(frozen / "models.json")
    )
    assert seal["predictions_sha256"] == sha(fresh / "predictions.npz")
    assert seal["descriptors_sha256"] == sha(fresh / "descriptors.npz")
    assert not set(contract["fresh_seeds"]) & {8101, 10101, 10102, 10103}
    with np.load(fresh / "descriptors.npz", allow_pickle=False) as data:
        z = data["z"]
    with np.load(fresh / "predictions.npz", allow_pickle=False) as prediction:
        for name, fitted in load(frozen / "models.json").items():
            np.testing.assert_allclose(
                m.predict(fitted, z[:, fitted["columns"]]),
                prediction[name],
                rtol=1e-13,
                atol=1e-20,
            )
    rows = load(fresh / "rows.json")
    for i, row in enumerate(rows):
        order = list(
            dict.fromkeys(r["history"] for r in rows if r["seed"] == row["seed"])
        )
        current = m.load_checkpoint(
            fresh / f"s{row['seed']}-history.npz",
            row["wait"],
            order.index(row["history"]),
        )
        # Floating reductions can differ by a few ulps across CPU/NumPy builds.
        # Serialized current-state and model identity remain exact hash checks.
        np.testing.assert_allclose(m.extract(current), z[i], rtol=1e-13, atol=1e-15)
        assert (
            hashlib.sha256(current.tobytes()).hexdigest()
            == load(fresh / f"reference-{i}.json")["current_sha256"]
        )


def test_saved_response_subtraction_scores_and_shared_floors():
    fresh = DATA / "fresh-01"
    rows = load(fresh / "rows.json")
    with np.load(fresh / "targets.npz", allow_pickle=False) as data:
        y = data["y"]
    for i in range(len(rows)):
        with np.load(fresh / f"reference-{i}.npz", allow_pickle=False) as data:
            np.testing.assert_array_equal(
                y[i], data["absolute"][:, 1, 2, :2] - data["absolute"][:, 0, 2, :2]
            )
    floors = np.full((2, 2), 1e-12)
    for name in ["dev-refine-01", "fresh-refine-01"]:
        ref = load(DATA / name / "refinement.json")
        for check in ref["checks"]:
            branches = np.array(check["branch_errors"])
            np.testing.assert_array_equal(branches.sum(axis=0), check["sum_errors"])
            floors = np.maximum(floors, 5 * branches.sum(axis=0))
    summary = load(DATA / "analysis-final/summary.json")
    np.testing.assert_array_equal(summary["floors"], floors)
    with np.load(fresh / "predictions.npz", allow_pickle=False) as data:
        for name, scores in summary["scores"].items():
            for score in scores:
                j = next(
                    i
                    for i, r in enumerate(rows)
                    if all(r[k] == score[k] for k in ["seed", "wait", "history"])
                )
                actual, predicted = y[j].copy(), data[name][j].copy()
                if score["kind"] == "Delta_R":
                    i = next(
                        i
                        for i, r in enumerate(rows)
                        if r["seed"] == score["seed"]
                        and r["wait"] == score["wait"]
                        and r["history"] == "none"
                    )
                    actual -= y[i]
                    predicted -= data[name][i]
                o = ["mass", "moment"].index(score["output"])
                mask = m.TIMES >= (40 if score["window"] == "late" else 0)
                error = np.sqrt(np.mean((predicted[mask, o] - actual[mask, o]) ** 2))
                denominator = np.sqrt(np.mean(actual[mask, o] ** 2))
                assert score["residual_rms"] == pytest.approx(error, rel=1e-12)
                assert score["relative_rms"] == pytest.approx(
                    error / denominator, rel=1e-12
                )
                gate = 0.02 if score["kind"] == "R" else 0.10
                assert score["passed"] == bool(
                    denominator > score["floor"] and error / denominator <= gate
                )
