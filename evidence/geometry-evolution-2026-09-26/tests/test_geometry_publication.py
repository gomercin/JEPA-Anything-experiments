"""Fast read-only evidence arithmetic and descriptor-only replay; no scientific runs."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.mediated_patterns import geometry_model as gm
from experiments.mediated_patterns import present_state_model as pm

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data/work/mediated_patterns/geometry_evolution"
SPEC = importlib.util.spec_from_file_location("geometry_restore", HERE / "restore.py")
restore = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(restore)


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_new_package_integrity_and_exclusive_restore(tmp_path):
    manifest = load(HERE / "artifacts.json")
    assert sum(r["bytes"] for r in manifest["files"]) < 25 * 1024**2
    assert restore.helpers.verify(manifest, HERE / "data") == len(manifest["files"])
    subset = {"files": manifest["files"][:1]}
    restore.helpers.restore(subset, HERE / "data", tmp_path / "restored")
    with pytest.raises(FileExistsError):
        restore.helpers.restore(subset, HERE / "data", tmp_path / "restored")


def test_frozen_identity_and_descriptor_only_composed_forecast():
    frozen, fresh = DATA / "frozen-01", DATA / "fresh-01"
    contract = load(frozen / "freeze.json")
    assert contract["sources"] == load(fresh / "protocol.json")["sources"]
    assert not set(contract["fresh_seeds"]) & {
        8101,
        10101,
        10102,
        10103,
        12101,
        12102,
        12103,
    }
    fpath = HERE.parents[1] / contract["F_path"]
    assert sha(fpath) == contract["F_sha256"]
    assert sha(frozen / "models.json") == contract["models_sha256"]
    F = load(fpath)["geometry"]
    for origin in [50, 60]:
        seal = load(fresh / f"seal-{origin}.json")
        assert seal["predictions_sha256"] == sha(fresh / f"predictions-{origin}.npz")
        assert seal["geometry_sha256"] == sha(fresh / f"geometry-{origin}.npz")
        assert seal["initial_descriptors_sha256"] == sha(
            fresh / f"initial-descriptors-{origin}.npz"
        )
        assert seal["max_resume_discrepancy"] <= 1e-14
        with np.load(
            fresh / f"initial-descriptors-{origin}.npz", allow_pickle=False
        ) as a:
            assert a.files == ["z"]
            z = a["z"]
        with np.load(fresh / f"boundary-{origin}.npz", allow_pickle=False) as a:
            measured = np.array(
                [pm.extract(s, feature_set="geometry") for s in a["states"]]
            )
        np.testing.assert_allclose(z, measured, atol=1e-15, rtol=1e-13)
        for name, g in load(frozen / "models.json").items():
            assert gm.identity(g) == seal["updaters_sha256"][name]
            delays = np.array(contract["times"][str(origin)]) - origin
            coords = np.array([gm.rollout(g, start, delays) for start in z])
            response = pm.predict(F, coords.reshape(-1, 3)).reshape(
                len(z), len(delays), 161, 2
            )
            with np.load(fresh / f"geometry-{origin}.npz", allow_pickle=False) as a:
                np.testing.assert_allclose(coords, a[name], rtol=1e-12, atol=1e-15)
            with np.load(fresh / f"predictions-{origin}.npz", allow_pickle=False) as a:
                np.testing.assert_allclose(response, a[name], rtol=1e-12, atol=1e-20)


def test_targets_are_independent_same_origin_probe_subtractions():
    fresh = DATA / "fresh-01"
    with np.load(fresh / "targets.npz", allow_pickle=False) as a:
        truth, times = a["y"], a["time"]
    for i in range(len(load(fresh / "rows.json"))):
        for j, t in enumerate(times):
            with np.load(fresh / f"reference-{i}-{t:g}.npz", allow_pickle=False) as a:
                np.testing.assert_array_equal(
                    truth[i, j], a["absolute"][:, 1, 2, :2] - a["absolute"][:, 0, 2, :2]
                )


def test_scores_floors_and_signed_error_accounting():
    summary = load(DATA / "analysis-final/summary.json")
    rows = load(DATA / "analysis-final/rows.json")
    floors = np.full((2, 2), 1e-12)
    coordinate = np.full(3, 1e-10)
    for stage in ["dev-refine-01", "fresh-refine-01", "fresh-motion-refine-01"]:
        r = load(DATA / stage / "refinement.json")
        floors = np.maximum(
            floors, np.max([s["response_floor"] for s in r["records"]], axis=0)
        )
        coordinate = np.maximum(
            coordinate, np.max([s["coordinate_floor"] for s in r["records"]], axis=0)
        )
    np.testing.assert_array_equal(summary["response_floors"], floors)
    np.testing.assert_array_equal(summary["coordinate_floors"], coordinate)
    with np.load(
        DATA / "analysis-final/signed-predictions.npz", allow_pickle=False
    ) as arrays:
        for model, scores in summary["scores"].items():
            for score in scores:
                i = next(
                    i for i, r in enumerate(rows) if all(r[k] == score[k] for k in r)
                )
                truth, pred = arrays["truth"][i].copy(), arrays[model][i].copy()
                if score["kind"] == "Delta_R":
                    j = next(
                        j
                        for j, r in enumerate(rows)
                        if r["history"] == "none"
                        and all(r[k] == score[k] for k in ["seed", "origin", "wait"])
                    )
                    truth -= arrays["truth"][j]
                    pred -= arrays[model][j]
                o = ["mass", "moment"].index(score["output"])
                mask = pm.TIMES >= (40 if score["window"] == "late" else 0)
                rms = np.sqrt(np.mean(truth[mask, o] ** 2))
                error = np.sqrt(np.mean((pred[mask, o] - truth[mask, o]) ** 2))
                assert score["residual_rms"] == pytest.approx(error, rel=1e-12)
                assert score["relative_rms"] == pytest.approx(error / rms, rel=1e-12)
                assert score["passed"] == bool(
                    rms > score["floor"]
                    and error / rms <= (0.02 if score["kind"] == "R" else 0.1)
                )
        for model in load(DATA / "frozen-01/models.json"):
            np.testing.assert_allclose(
                arrays[model] - arrays["truth"],
                arrays["update_" + model] + arrays["readout_" + model],
                atol=1e-22,
            )
    for term in summary["decomposition"]:
        assert term["total_mse"] == pytest.approx(
            term["update_mse"] + term["readout_mse"] + term["cross_term"],
            rel=1e-12,
            abs=1e-34,
        )
    for row in summary["geometry"]:
        expected = (
            (
                np.array(row["max_error"])
                <= load(DATA / "frozen-01/freeze.json")["coordinate_tolerances"]
            )
            & (np.array(row["relative_movement_rms"]) <= 0.1)
            & np.array(row["movement_resolved"])
        )
        assert row["passed"] == expected.tolist()


def test_probe_initial_readouts_match_actual_later_field():
    from experiments.mediated_patterns.simulator import Field
    from experiments.mediated_patterns.source_receiver_relay import (
        CFG,
        read,
        source_profile,
    )

    field = Field(CFG)
    fresh = DATA / "fresh-01"
    with np.load(fresh / "evaluator-states.npz", allow_pickle=False) as a:
        states, times = a["states"], a["time"]
    for i, trajectory in enumerate(states):
        for state, t in zip(trajectory, times, strict=True):
            stimulated = state.copy()
            stimulated[0] += 0.02 * source_profile(field)
            initial = np.fft.irfft(np.fft.rfft([state, stimulated]), n=CFG.n)
            with np.load(fresh / f"reference-{i}-{t:g}.npz", allow_pickle=False) as a:
                np.testing.assert_allclose(
                    a["absolute"][0], read(field, initial, -4.0), rtol=1e-13, atol=1e-14
                )
