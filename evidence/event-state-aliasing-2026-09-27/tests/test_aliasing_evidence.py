"""Portable new evidence, paired arithmetic and seals; no scientific simulation."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.mediated_patterns import event_state_model as sm
from experiments.mediated_patterns import geometry_model as gm
from experiments.mediated_patterns.event_age_response import arrays, paired
from experiments.mediated_patterns.event_state_analysis import (
    gates,
    prediction,
    summary,
)
from experiments.mediated_patterns.present_state_transmission import digest

HERE = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def restore():
    spec = importlib.util.spec_from_file_location("alias_restore", HERE / "restore.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("aliasing-evidence")
    restore().unpack(root)
    return root / "work/mediated_patterns/event_state_aliasing"


def test_exclusive_restoration(data, tmp_path):
    m = load(HERE / "artifacts.json")
    module = restore()
    assert sum(x["bytes"] for x in m["archives"]) < 25 * 1024**2
    assert module.helpers.verify(m, data.parents[2]) == len(m["files"])
    with pytest.raises(FileExistsError):
        module.unpack(data.parents[2])
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError):
        module.unpack(link)


def test_response_blind_selection_and_grouped_preprocessing(data):
    b = data / "boundaries-01"
    s = load(b / "selection.json")
    x = arrays(b / "descriptors.npz")
    rows = load(b / "rows.json")
    assert digest(b / "descriptors.npz") == s["descriptors_sha256"]
    expected = sm.choose_pairs(
        x["x"], [r["seed"] for r in rows], x["extraction_uncertainty"].max(0)[:3]
    )
    assert expected["all_pairs"] == s["all_pairs"]
    assert expected["scale"] == s["scale"]
    assert load(data / "responses-01/selection-copy.json") == s
    for folder in data.glob("fit-*"):
        selected = load(folder / "selection.json")
        cols = sm.SETS[selected["feature_set"]]
        for file in folder.glob("folds-*.json"):
            for fold in load(file):
                held = fold["held_group"]
                train = [r["seed"] != held for r in rows]
                assert held not in fold["training_groups"]
                np.testing.assert_allclose(
                    fold["mean"], x["x"][train][:, cols].mean(0), rtol=1e-14, atol=1e-15
                )


def test_snapshot_freeze_prediction_and_scores(data):
    f = load(data / "frozen-01/freeze.json")
    fresh = data / "fresh-01"
    seal = load(fresh / "seal.json")
    m = load(data / "frozen-01/model.json")
    assert f["sources"] == load(fresh / "protocol.json")["sources"]
    assert gm.identity(m) == f["model_sha256"] == seal["model_sha256"]
    assert digest(fresh / "predictions.npz") == seal["prediction_sha256"]
    assert digest(fresh / "descriptors.npz") == seal["descriptor_sha256"]
    assert digest(data / "frozen-01/freeze.json") == seal["freeze_sha256"]
    rows = load(fresh / "rows.json")
    assert len({r["seed"] for r in rows}) >= 3
    assert not {r["seed"] for r in rows} & {
        r["seed"] for r in load(data / "boundaries-01/rows.json")
    }
    x = arrays(fresh / "descriptors.npz")["x"]
    p = arrays(fresh / "predictions.npz")["y"]
    np.testing.assert_array_equal(prediction(m, x), p)
    y = arrays(fresh / "targets.npz")["y"]
    for i in range(len(rows)):
        for j, a in enumerate([0.0, -0.02, 0.02]):
            np.testing.assert_array_equal(
                paired(fresh / f"reference-{i}-{a:g}.npz")["response"], y[i, j]
            )
    actual = gates(y, p, rows, f["floors"])
    assert actual == load(data / "fresh-analysis-01/scores.json")
    assert summary(actual) == load(data / "fresh-analysis-01/summary.json")
    body = Path("experiments/mediated_patterns/event_state_analysis.py").read_text()
    body = body[body.index("def fresh(") : body.index("def analyze(")]
    assert body.index('"seal.json"') < body.rindex("response_pair(")


def test_new_numerical_qualification_and_cost(data):
    refinement = load(data / "refinement-01/refinement.json")
    for row in refinement["records"]:
        np.testing.assert_array_equal(
            np.sum(row["pair_uncertainty"], axis=0), row["D1_bound"]
        )
        assert all(v >= 0 for v in row["direct_D1"])
    costs = [load(p)["cpu_seconds"] for p in data.glob("*/budget.json")]
    assert 30 + sum(costs) < 1800
    for p in data.rglob("reference-*.json"):
        row = load(p)
        if "overlap_error" in row:
            assert row["overlap_error"] < 1e-12
            assert row["qualified"]
