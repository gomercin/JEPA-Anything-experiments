"""Synthetic/instrument checks, not assertions about field predictability."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.mediated_patterns import event_operator_model as om
from experiments.mediated_patterns import geometry_model as gm
from experiments.mediated_patterns import present_state_model as pm
from experiments.mediated_patterns.event_age_operator_family import safe_output, scores


def fixture():
    rng = np.random.default_rng(701)
    x = rng.normal(size=(64, 3))
    ages = np.tile([10.0, 15.0, 20.0, 30.0], 16)
    h = np.arange(161) / 2
    shapes = np.array([np.exp(-h / 10), np.exp(-h / 35), h / 80, np.sin(h / 30)])

    def response(z, a, t):
        p = (t - 20) / 10
        b = a / 0.02
        common = (
            np.array([1 + z[0] + 0.05 * z[1] ** 2, 0.7 + z[2], 0.2 * p, 0.1 * p * p])
            @ shapes
        )
        odd = (
            np.array([0.1 + 0.02 * z[0] * p, 0.03 * p * p, 0.02 * z[1], 0.01]) @ shapes
        )
        even = np.array([0.02 * p, 0.01 * z[2], 0.03, 0.01 * z[0] * p * p]) @ shapes
        v = common + b * odd + b * b * even
        return np.column_stack([v, 0.3 * v])

    y = np.array(
        [
            [response(z, a, t) for a in [0.0, -0.02, 0.02]]
            for z, t in zip(x, ages, strict=True)
        ]
    )
    return x, ages, y, response


def test_smooth_contextual_operator_interpolates_unseen_age():
    x, t, y, truth = fixture()
    model = om.fit(x, t, y, ridge=1e-12)
    z = np.array([0.2, -0.5, 0.3])
    for a in [0.0, -0.02, 0.02]:
        np.testing.assert_allclose(
            om.predict(model, z, a, 18), truth(z, a, 18), rtol=1e-10, atol=1e-10
        )
    assert np.asarray(model["coefficients"]).shape == (30, 24)
    assert not any(
        k in model for k in ["ages", "age_ids", "training_examples", "responses"]
    )


def test_full_age_and_preparation_exclusion_train_only_scaling():
    x, t, y, _ = fixture()
    groups = np.repeat(np.arange(16), 4)
    mask = om.training_mask(groups, t, 3, 15)
    assert not mask[groups == 3].any()
    assert not mask[t == 15].any()
    m = om.fit(x[mask], t[mask], y[mask])
    np.testing.assert_array_equal(m["mean"], x[mask].mean(0))
    changed = y.copy()
    changed[~mask] = 1e100
    assert om.fit(x[mask], t[mask], changed[mask]) == m


def test_shared_representation_and_independent_amplitude_readouts():
    x, t, y, _ = fixture()
    rep = om.representation(x, y)
    models = [
        om.fit(x[t == a], t[t == a], y[t == a], family="blind", rep=rep)
        for a in [10, 15]
    ]
    assert models[0]["bases"] == models[1]["bases"]
    assert models[0]["mean"] == models[1]["mean"]
    np.testing.assert_array_equal(
        om.predict(models[0], x[0], 0.02, 10), om.predict(models[0], x[0], 0.02, 30)
    )
    m = om.fit(x, t, y)
    z = x[0]
    zero = om.predict(m, z, 0, 18)
    plus = om.predict(m, z, 0.02, 18) - zero
    minus = om.predict(m, z, -0.02, 18) - zero
    half = om.predict(m, z, 0.01, 18) - zero
    np.testing.assert_allclose(
        half, 0.5 * (plus - minus) / 2 + 0.25 * (plus + minus) / 2, atol=1e-14
    )


def test_context_does_not_enter_autonomous_geometry_and_checkpoint(tmp_path):
    g = gm.fit(
        np.arange(30).reshape(10, 3) / 100, np.ones((10, 3)) * 0.001, kind="drift"
    )
    flow = gm.Continuation(g, [0.0, 0.0, 0.0])
    flow.advance(10)
    cp = tmp_path / "checkpoint.json"
    flow.checkpoint(cp)
    restored = gm.Continuation.restore(g, json.loads(cp.read_text()))
    np.testing.assert_array_equal(restored.advance(12), flow.advance(12))
    with pytest.raises(FileExistsError):
        flow.checkpoint(cp)
    # Age is supplied to a query, never injected into G or retained by the map.
    x, t, y, _ = fixture()
    m = om.fit(x, t, y)
    snapshot = flow.z.copy()
    om.predict(m, flow.z, 0.02, 18)
    om.predict(m, flow.z, 0.02, 24)
    np.testing.assert_array_equal(flow.z, snapshot)


def test_serialization_and_age_guard():
    x, t, y, _ = fixture()
    m = om.fit(x, t, y)
    m2 = json.loads(json.dumps(m, sort_keys=True, allow_nan=False))
    np.testing.assert_array_equal(
        om.predict(m, x[0], -0.02, 24), om.predict(m2, x[0], -0.02, 24)
    )
    for age in [9, 31, np.nan]:
        with pytest.raises(ValueError):
            om.predict(m, x[0], 0.02, age)
    for age in [10, 18, 24, 30]:
        assert om.age_features([age], "spline").shape == (1, 4)
    with pytest.raises(ValueError):
        om.predict(m, x[0], 0.03, 18)


def test_score_matches_independent_four_branch_arithmetic():
    h = np.arange(161)[:, None]
    common = np.ones((161, 2)) * 3
    change = (h + 1) * np.array([0.001, -0.002])
    y = np.array([[common, common - change, common + change]])
    p = y.copy()
    p[0, 2] += change * 0.2
    rows = [{"seed": 1, "history": "fixture", "age": 18}]
    result = scores(y, p, rows, {})
    d = [r for r in result if r["kind"] == "D1" and r["a"] > 0]
    np.testing.assert_allclose([r["relative"] for r in d], 0.2)
    assert all(r["passed"] is False for r in d)
    # Lingering first-event output cancels before the response target.
    lingering = np.ones((161, 2)) * 100
    np.testing.assert_allclose((lingering + common + change) - lingering, y[0, 2])


def test_non_overwriting_output_and_symlink(tmp_path, monkeypatch):
    import experiments.mediated_patterns.event_age_operator_family as runner

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = tmp_path / "new"
    safe_output(path)
    with pytest.raises(FileExistsError):
        safe_output(path)
    link = tmp_path / "link"
    link.symlink_to(path, target_is_directory=True)
    with pytest.raises(ValueError):
        safe_output(link)
    p = path / "model.json"
    pm.save_json(p, {"value": 1})
    with pytest.raises(FileExistsError):
        pm.save_json(p, {"value": 2})


def test_runtime_has_no_field_or_reference_import():
    source = Path(om.__file__).read_text()
    assert "simulator" not in source
    assert "load_checkpoint" not in source
    assert "load(" not in source
    assert "reference(" not in source


def test_both_forecast_seals_precede_their_reference_access(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from experiments.mediated_patterns import event_age_operator_family as runner
    from experiments.mediated_patterns import event_operator_analysis as analysis
    from experiments.mediated_patterns import geometry_assay, geometry_evolution
    from experiments.mediated_patterns import organization_response as org
    from experiments.mediated_patterns import repeated_intervention_state as repeated
    from experiments.mediated_patterns import source_receiver_relay as relay

    x, t, y, _ = fixture()
    model = om.fit(x, t, y)
    g = gm.fit(x, np.zeros_like(x), kind="drift")
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    development = tmp_path / "development"
    development.mkdir()
    pm.save_json(development / "rows.json", [{"seed": 1}])
    for name in ["selected", "blind"]:
        pm.save_json(frozen / (name + ".json"), model)
    pm.save_json(frozen / "G.json", g)
    pm.save_json(
        frozen / "freeze.json",
        {"sources": {}, "models": {}, "development": str(development)},
    )
    monkeypatch.setattr(runner, "FRESH", [999])
    monkeypatch.setattr(repeated, "source_hashes", dict)
    monkeypatch.setattr(org, "isolated_components", lambda *a: None)
    monkeypatch.setattr(relay, "prepare", lambda *a: np.zeros((2, 768)))
    monkeypatch.setattr(geometry_assay, "initialize_written", lambda state, *a: state)
    monkeypatch.setattr(geometry_evolution, "unforced", lambda state, *a: ([state], []))
    monkeypatch.setattr(pm, "extract", lambda *a, **k: x[0])
    out = tmp_path / "out"
    out.mkdir()
    calls = []

    def later_field(state, *a):
        assert (out / "retained-seal.json").is_file()
        assert not (out / "snapshot-seal.json").exists()
        calls.append("current")
        return state

    def future_response(*a):
        assert (out / "retained-seal.json").is_file()
        assert (out / "snapshot-seal.json").is_file()
        calls.append("future")
        return np.zeros((161, 2))

    monkeypatch.setattr(runner, "boundary", later_field)
    monkeypatch.setattr(runner, "response_pair", future_response)
    budget = SimpleNamespace(begin=lambda *a: None, finish=lambda: None)
    analysis.fresh(out, SimpleNamespace(data=frozen), budget)
    assert calls == ["current"] * 4 + ["future"] * 12
    assert len(list(out.glob("checkpoint-*.json"))) == 2
    with pytest.raises(FileExistsError):
        analysis.fresh(out, SimpleNamespace(data=frozen), budget)
