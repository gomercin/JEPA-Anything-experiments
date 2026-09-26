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
    for row, expected_row in zip(s["all_pairs"], expected["all_pairs"], strict=True):
        assert row["indices"] == expected_row["indices"]
        assert row["near"] == expected_row["near"]
        for key in [
            "distance",
            "normalized_max",
            "uncertainty_distance",
            "overlap_difference",
        ]:
            np.testing.assert_allclose(
                row[key], expected_row[key], rtol=1e-14, atol=1e-24
            )
    assert [r["indices"] for r in expected["candidate_pairs"]] == [
        r["indices"] for r in s["candidate_pairs"]
    ]
    for pair in s["candidate_pairs"]:
        i, j = pair["indices"]
        np.testing.assert_array_equal(
            pair["descriptor_difference"], x["x"][i] - x["x"][j]
        )
    np.testing.assert_allclose(expected["scale"], s["scale"], rtol=1e-14, atol=1e-24)
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
    np.testing.assert_allclose(prediction(m, x), p, rtol=1e-12, atol=1e-22)
    y = arrays(fresh / "targets.npz")["y"]
    for i in range(len(rows)):
        for j, a in enumerate([0.0, -0.02, 0.02]):
            np.testing.assert_array_equal(
                paired(fresh / f"reference-{i}-{a:g}.npz")["response"], y[i, j]
            )
    actual = gates(y, p, rows, f["floors"])
    numeric_records_equal(actual, load(data / "fresh-analysis-01/scores.json"))
    numeric_records_equal(
        summary(actual), load(data / "fresh-analysis-01/summary.json")
    )
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


def test_retained_fresh_seals_and_replay(data, tmp_path):
    import shutil
    import subprocess
    import sys

    frozen = data / "retained-frozen-01"
    fresh = data / "retained-fresh-01"
    f = load(frozen / "freeze.json")
    seal = load(fresh / "seal.json")
    m = load(frozen / "model.json")
    g = load(frozen / "G.json")
    assert f["sources"] == load(fresh / "protocol.json")["sources"]
    assert (
        gm.identity(m)
        == seal["model_sha256"]
        == load(data / "frozen-01/freeze.json")["model_sha256"]
    )
    assert gm.identity(g) == seal["G_sha256"] == f["G_sha256"]
    assert digest(fresh / "predictions.npz") == seal["prediction_sha256"]
    assert (
        digest(fresh / "initial-descriptors.npz") == seal["initial_descriptor_sha256"]
    )
    rows = load(fresh / "rows.json")
    assert not {r["seed"] for r in rows} & {
        r["seed"] for r in load(data / "fresh-01/rows.json")
    }
    p = arrays(fresh / "predictions.npz")["y"]
    y = arrays(fresh / "targets.npz")["y"]
    records = gates(y, p, rows, f["floors"])
    numeric_records_equal(records, load(data / "retained-analysis-01/scores.json"))
    for i in range(len(rows)):
        for j, a in enumerate([0.0, -0.02, 0.02]):
            np.testing.assert_array_equal(
                paired(fresh / f"reference-{i}-{a:g}.npz")["response"], y[i, j]
            )
    package = tmp_path / "runtime"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name in [
        "event_state_model",
        "geometry_model",
        "present_state_model",
        "measurements",
    ]:
        shutil.copyfile(
            Path(sm.__file__).with_name(name + ".py"), package / (name + ".py")
        )
    (tmp_path / "model.json").write_text(json.dumps(m))
    (tmp_path / "G.json").write_text(json.dumps(g))
    script = """
import builtins,json,sys
sys.path.insert(0,'.')
original=builtins.__import__
def guard(name,*args,**kwargs):
    if name.startswith(('scipy','experiments')) or 'simulator' in name:raise RuntimeError('solver unavailable')
    return original(name,*args,**kwargs)
builtins.__import__=guard
from runtime.geometry_model import Continuation
from runtime.event_state_model import predict
s=Continuation.restore(json.load(open('G.json')),json.load(open('checkpoint.json')))
z=s.advance(12);m=json.load(open('model.json'))
import numpy as np
g=json.load(open('G.json'));initial=json.load(open('initial.json'))
u=Continuation(g,initial);u.advance(10)
path='local-checkpoint-'+sys.argv[1]+'.json';u.checkpoint(path)
v=Continuation.restore(g,json.load(open(path)))
full=Continuation(g,initial).advance(22)
np.testing.assert_array_equal(v.advance(12),full)
np.testing.assert_allclose(z,full,rtol=1e-14,atol=1e-20)
print(json.dumps([predict(m,z,a).tolist() for a in [0.,-.02,.02]]))
"""
    (tmp_path / "resume.py").write_text(script)
    initial = arrays(fresh / "initial-descriptors.npz")["x"]
    for i in range(len(rows)):
        (tmp_path / "initial.json").write_text(json.dumps(initial[i].tolist()))
        cp = fresh / f"checkpoint-{i}.json"
        assert digest(cp) == seal["checkpoints"][cp.name]
        (tmp_path / "checkpoint.json").write_bytes(cp.read_bytes())
        result = subprocess.check_output(
            [sys.executable, "-I", "resume.py", str(i)], cwd=tmp_path, text=True
        )
        np.testing.assert_allclose(json.loads(result), p[i], rtol=1e-12, atol=1e-22)


def numeric_records_equal(actual, expected):
    """Decision/identity exact; arithmetic portable far below scientific floors."""
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            numeric_records_equal(actual[key], expected[key])
    elif isinstance(expected, list):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            numeric_records_equal(a, b)
    elif isinstance(expected, float):
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-24)
    else:
        assert actual == expected
