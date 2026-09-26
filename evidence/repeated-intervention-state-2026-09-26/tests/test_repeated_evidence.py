"""Fast immutable-data arithmetic and provenance checks; no field simulation."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.mediated_patterns import geometry_model as gm
from experiments.mediated_patterns import repeated_intervention_model as rm
from experiments.mediated_patterns.repeated_intervention_analysis import score, truth

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data/work/mediated_patterns/repeated_intervention_state"


def load(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_portable_hashes_and_safe_restore(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "repeated_restore", HERE / "restore.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = load(HERE / "artifacts.json")
    assert manifest["bytes"] == sum(r["bytes"] for r in manifest["files"])
    assert manifest["bytes"] < 25 * 1024**2
    assert module.helpers.verify(manifest, HERE / "data") == len(manifest["files"])
    subset = {"files": manifest["files"][:1]}
    module.helpers.restore(subset, HERE / "data", tmp_path / "restored")
    with pytest.raises(FileExistsError):
        module.helpers.restore(subset, HERE / "data", tmp_path / "restored")


def test_sealed_sources_and_preparation_groups():
    freeze = load(DATA / "frozen-01/freeze.json")
    fresh = DATA / "fresh-01"
    seal = load(fresh / "seal.json")
    assert freeze["sources"] == load(fresh / "protocol.json")["sources"]
    assert (
        gm.identity(load(DATA / "frozen-01/model.json"))
        == freeze["selected_sha256"]
        == seal["model_sha256"]
    )
    assert not set(freeze["fresh_seeds"]) & set(freeze["development_seeds"])
    assert len({r["seed"] for r in load(fresh / "rows.json")}) == 3
    for name, digest in {**seal["predictions"], **seal["checkpoints"]}.items():
        assert sha(fresh / name) == digest
    assert sha(fresh / "initial-descriptors.npz") == seal["initialization_sha256"]
    checks = load(DATA / "runtime-audit.json")["checks"]
    assert len(checks) == 192
    assert all(c["state_max"] == c["response_max"] == 0 for c in checks)


def test_all_scores_targets_and_no_reset():
    fresh = DATA / "fresh-01"
    frozen = load(DATA / "frozen-01/freeze.json")
    scores = load(DATA / "fresh-analysis-01/scores.json")
    z0 = np.load(fresh / "initial-descriptors.npz")["z"]
    model = load(DATA / "frozen-01/model.json")
    for case in load(fresh / "cases.json"):
        y, _, _ = truth(fresh, case)
        schedule = case["schedule"]
        stored = np.load(fresh / f"prediction-{case['index']}-{schedule['name']}.npz")
        for j, flags in enumerate(rm.PREFIXES):
            aa = [a * b for a, b in zip(schedule["amplitudes"], flags, strict=True)]
            tr, pred = rm.forecast(model, z0[case["index"]], schedule["times"], aa)
            np.testing.assert_allclose(
                tr, stored["trajectory"][j], rtol=1e-11, atol=1e-14
            )
            np.testing.assert_allclose(pred, stored["y"][j], rtol=1e-11, atol=1e-20)
        for name, pred in rm.comparators(stored["y"]).items():
            computed = score(y, pred, frozen["floors"]["floors"])
            matching = [
                s
                for s in scores
                if s["seed"] == case["seed"]
                and s["history"] == case["history"]
                and s["schedule"] == schedule["name"]
                and s["model"] == name
            ]
            assert len(matching) == len(computed)
            for actual, expected in zip(matching, computed, strict=True):
                for key, value in expected.items():
                    if isinstance(value, float):
                        np.testing.assert_allclose(
                            actual[key], value, rtol=1e-12, atol=1e-25
                        )
                    else:
                        assert actual[key] == value


def test_all_reference_regimes_and_local_jumps():
    for stage in ["pilot-01", "develop-01", "fresh-01"]:
        for path in (DATA / stage).glob("reference-*.json"):
            row = load(path)
            assert row["qualified"]
            for event in row["events"]:
                assert event["identity_max"] < 1e-13
                np.testing.assert_array_equal(event["before"][1:], event["after"][1:])
