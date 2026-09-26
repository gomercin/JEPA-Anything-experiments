"""Fast restoration, frozen-source, arithmetic and replay checks; no PDE panel."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.mediated_patterns import geometry_model as gm
from experiments.mediated_patterns.event_age_response import (
    arrays,
    paired,
    predict,
    score,
)
from experiments.mediated_patterns.present_state_transmission import digest

HERE = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def restore():
    spec = importlib.util.spec_from_file_location("age_restore", HERE / "restore.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("event-age-evidence")
    restore().unpack(root)
    return root / "work/mediated_patterns/event_age_response"


def test_portable_safe_restore(data, tmp_path):
    m = load(HERE / "artifacts.json")
    module = restore()
    assert m["bytes"] == sum(x["bytes"] for x in m["files"])
    assert sum(x["bytes"] for x in m["archives"]) < 25 * 1024**2
    assert module.helpers.verify(m, data.parents[2]) == len(m["files"])
    with pytest.raises(FileExistsError):
        module.unpack(data.parents[2])
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError):
        module.unpack(link)


def test_frozen_seals_and_exposure(data):
    f = load(data / "frozen-01/freeze.json")
    fresh = data / "fresh-01"
    seal = load(fresh / "seal.json")
    assert f["sources"] == load(fresh / "protocol.json")["sources"]
    assert (
        gm.identity(load(data / "frozen-01/model.json"))
        == f["model_sha256"]
        == seal["model_sha256"]
    )
    assert (
        gm.identity(load(data / "frozen-01/age-diagnostic.json"))
        == f["diagnostic_sha256"]
    )
    assert not set(f["development_seeds"]) & set(f["fresh_seeds"])
    assert f["withheld_age"] not in f["development_ages"]
    assert len({x["seed"] for x in load(fresh / "rows.json")}) >= 3
    for name, sha in {**seal["predictions"], **seal["checkpoints"]}.items():
        assert digest(fresh / name) == sha
    assert digest(fresh / "initial-descriptors.npz") == seal["initialization_sha256"]
    for stage in ["fit-two-01", "fit-conditioned-01", "fit-anchored-01"]:
        for seed in f["development_seeds"]:
            fit = load(data / stage / f"fold-{seed}-fit.json")
            assert (
                seed not in fit["training_seeds"]
                and f["withheld_age"] not in fit["training_ages"]
            )
    source = (
        Path(__file__).resolve().parents[3]
        / "experiments/mediated_patterns/event_age_response.py"
    )
    text = source.read_text()
    body = text[text.index("def fresh(") : text.index("def main(")]
    assert body.index('"seal.json"') < body.rindex("prefix_reference(")


def test_all_fresh_scores_and_retained_rollouts(data):
    fresh = data / "fresh-01"
    f = load(data / "frozen-01/freeze.json")
    m = load(data / "frozen-01/model.json")
    all_scores = load(data / "fresh-analysis-01/scores.json")
    z0 = arrays(fresh / "initial-descriptors.npz")["z"]
    for c in load(fresh / "cases.json"):
        y = np.array(
            [paired(fresh / (name + ".npz"))["response"] for name in c["prefixes"]]
        )
        saved = arrays(fresh / (c["prediction"] + ".npz"))
        pred, tr, _state = predict(m, z0[c["index"]], c["age"], c["a"])
        np.testing.assert_allclose(pred, saved["y"], rtol=1e-12, atol=1e-22)
        np.testing.assert_allclose(tr, saved["trajectory"], rtol=1e-12, atol=1e-16)
        for name, p in [
            ("causal", saved["y"]),
            ("original9", saved["original"]),
            ("original11", saved["original"]),
            ("explicit_age", saved["privileged"]),
        ]:
            expected = score(y, p, f["floors"]["floors"])
            actual = [
                r
                for r in all_scores
                if r["model"] == name
                and all(r[k] == c[k] for k in ["seed", "history", "age", "a"])
            ]
            assert len(expected) == len(actual)
            for e, a in zip(expected, actual, strict=True):
                for key, value in e.items():
                    if isinstance(value, float):
                        np.testing.assert_allclose(
                            a[key], value, rtol=1e-12, atol=1e-25
                        )
                    else:
                        assert a[key] == value
    decision = load(data / "fresh-analysis-01/decision.json")
    assert decision["single_event_pass"] == all(
        r["passed"] is True for r in all_scores if r["model"] == "causal"
    )
    assert decision["conditional_repeated_authorized"] == decision["single_event_pass"]


def test_instrument_regimes_and_numerical_propagation(data):
    for stage in ["pilot-01", "develop-01", "fresh-01"]:
        for path in (data / stage).glob("reference-*.json"):
            row = load(path)
            assert row["qualified"]
            for e in row["events"]:
                assert e["identity_max"] < 1e-13
                np.testing.assert_array_equal(e["before"][1:], e["after"][1:])
    f = load(data / "refine-01/refinement.json")
    for r in f["records"]:
        pair = np.array(r["pair"])
        want = (
            pair.sum(axis=0)
            if r["kind"] == "D1"
            else pair[0 if r["kind"] == "R_without" else 1]
        )
        np.testing.assert_array_equal(want, r["floor"])
    audit = load(data / "runtime-audit.json")
    assert audit["checkpoints"] == 72 and audit["exact_equal"]
    assert not audit["fields_available"] and not audit["scientific_solvers_available"]
