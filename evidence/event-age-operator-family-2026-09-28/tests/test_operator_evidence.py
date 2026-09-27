"""Portable fixed-age success/shared-family failure replay; no field simulation."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.mediated_patterns import event_operator_model as om
from experiments.mediated_patterns.event_age_operator_family import scores
from experiments.mediated_patterns.event_age_response import (
    WINDOWS,
    arrays,
    paired,
    rms,
)
from experiments.mediated_patterns.event_state_analysis import summary

HERE = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def restorer():
    spec = importlib.util.spec_from_file_location(
        "operator_restore", HERE / "restore.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("operator-evidence")
    restorer().unpack(root, fixtures=True)
    return root / "work/mediated_patterns/event_age_operator_family"


def test_archive_exclusive_complete_and_bounded(data, tmp_path):
    m = restorer().manifest(fixtures=True)
    assert sum(a["bytes"] for a in m["archives"]) < 25 * 1024**2
    assert restorer().helpers.verify(m, data.parents[2]) == len(m["files"])
    with pytest.raises(FileExistsError):
        restorer().unpack(data.parents[2], fixtures=True)
    link = tmp_path / "symlink"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError):
        restorer().unpack(link, fixtures=True)


def test_all_descendants_and_entire_ages_are_excluded(data):
    x = arrays(data / "develop-01/states.npz")["x"]
    rows = load(data / "develop-01/rows.json")
    groups = np.array([r["seed"] for r in rows])
    ages = np.array([r["age"] for r in rows])
    assert len(set(groups)) == 12
    assert sorted(set(ages)) == [10, 15, 18, 20, 30]
    for fit in ["fit-01", "fit-spline-01"]:
        decision = load(data / fit / "decision.json")
        for candidate in decision["candidates"]:
            folder = data / fit / Path(candidate["path"]).name
            for f in folder.glob("folds-*.json"):
                for fold in load(f):
                    tr = np.asarray(fold["training_indices"])
                    test = np.asarray(fold["test_indices"])
                    assert not set(tr) & set(test)
                    assert not (groups[tr] == fold["held_group"]).any()
                    if fold["held_age"] is not None:
                        assert not (ages[tr] == fold["held_age"]).any()
                        assert (ages[test] == fold["held_age"]).all()
                    if not (folder / fold["model"]).exists():
                        continue  # Full coefficients are in the checksum-recorded remote supplement.
                    m = load(folder / fold["model"])
                    np.testing.assert_allclose(
                        m["mean"], x[tr].mean(0), rtol=1e-13, atol=1e-15
                    )
                    np.testing.assert_allclose(
                        m["scale"], x[tr].std(0), rtol=1e-13, atol=1e-15
                    )
                    assert "age_ids" not in m
        for f in (data / fit / "fixed-0.001").glob("fixed-fold-*.json"):
            models = list(load(f).values())
            assert all(m["bases"] == models[0]["bases"] for m in models)
            assert all(m["mean"] == models[0]["mean"] for m in models)


def test_scores_reconstruct_the_negative_result_and_fresh_stop(data):
    y = arrays(data / "develop-01/targets.npz")["y"]
    rows = load(data / "develop-01/rows.json")
    floors = load(data / "refine-01/refinement.json")["floors"]
    expected = {
        "fit-01": {"15": 126, "18": 10, "20": 94},
        "fit-spline-01": {"15": 10, "18": 4, "20": 24},
    }
    for fit, fails in expected.items():
        decision = load(data / fit / "decision.json")
        assert decision["status"] == "INTERPOLATION_LIMIT"
        assert decision["selected"] is None
        assert all(v["failed"] == 0 for v in decision["fixed_summaries"].values())
        family = "polynomial" if fit == "fit-01" else "spline"
        folder = data / fit / (family + "-1e-06")
        for age, count in fails.items():
            saved = arrays(folder / f"predictions-{age}.npz")
            ii = saved["indices"]
            rebuilt = scores(y[ii], saved["y"], [rows[i] for i in ii], floors)
            original = load(folder / f"scores-{age}.json")
            assert summary(rebuilt)["failed"] == count
            assert summary(rebuilt)["unresolved"] == 0
            assert all(
                a["passed"] == b["passed"]
                for a, b in zip(rebuilt, original, strict=True)
            )
            for key in ["relative", "error_rms", "error_max", "magnitude"]:
                np.testing.assert_allclose(
                    [r[key] for r in rebuilt],
                    [r[key] for r in original],
                    rtol=1e-12,
                    atol=1e-24,
                )
    assert not (data / "frozen-01").exists()
    assert not (data / "fresh-01").exists()
    assert not (data / "repeated-01").exists()
    assert load(data / "diagnosis-01/budget.json")["status"] == "FAILED"
    assert 30 + sum(load(p)["cpu_seconds"] for p in data.glob("*/budget.json")) < 1800


def test_current_center_query_replay_without_age_lookup(data):
    x = arrays(data / "develop-01/states.npz")["x"]
    rows = load(data / "develop-01/rows.json")
    folder = data / "fit-spline-01/spline-1e-06"
    fold = next(f for f in load(folder / "folds-18.json") if f["held_group"] == 20101)
    m = load(folder / fold["model"])
    saved = arrays(folder / "predictions-18.npz")
    for i in fold["test_indices"]:
        local = int(np.flatnonzero(saved["indices"] == i)[0])
        rebuilt = np.array(
            [om.predict(m, x[i], a, rows[i]["age"]) for a in [0.0, -0.02, 0.02]]
        )
        np.testing.assert_allclose(rebuilt, saved["y"][local], rtol=1e-12, atol=1e-22)
    assert np.asarray(m["coefficients"]).shape == (40, 24)


def test_worst_gate_is_exactly_the_numerically_refined_pilot(data):
    worst = load(data / "diagnosis-02/adverse.json")
    i = worst["development_index"]
    assert (worst["seed"], worst["history"], worst["age"]) == (20101, "odd04", 20)
    np.testing.assert_array_equal(
        arrays(data / "pilot-01/states.npz")["states"][0],
        arrays(data / "develop-01/states.npz")["states"][i],
    )
    np.testing.assert_array_equal(
        arrays(data / "pilot-01/targets.npz")["y"][0],
        arrays(data / "develop-01/targets.npz")["y"][i],
    )
    y = arrays(data / "pilot-01/targets.npz")["y"][0]
    absolute = np.array(
        [
            paired(data / "pilot-01" / f"reference-0-{a:g}.npz")["absolute"]
            for a in [0.0, -0.02, 0.02]
        ]
    )
    floors = {}
    for label in ["halfdt", "doubleN"]:
        r = np.array(
            [
                paired(data / "refine-01" / label / f"reference-{a:g}.npz")["response"]
                for a in [0.0, -0.02, 0.02]
            ]
        )
        for win, sl in WINDOWS.items():
            err = np.maximum(
                5 * rms((r - y)[:, sl], axis=1),
                64
                * np.finfo(float).eps
                * abs(absolute[:, :, :, 2, :2]).max(axis=(0, 1, 2)),
            )
            for j in [1, 2]:
                key = "D1/" + win
                floors[key] = np.maximum(floors.get(key, np.zeros(2)), err[0] + err[j])
    published = load(data / "refine-01/refinement.json")["floors"]
    for key, v in floors.items():
        np.testing.assert_allclose(v, published[key], rtol=1e-12, atol=1e-24)
    assert worst["error_rms"] > 16 * worst["floor"]
