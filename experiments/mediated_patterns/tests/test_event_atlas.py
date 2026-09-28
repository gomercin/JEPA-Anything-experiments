"""Synthetic instrument checks; none assert the SH35 scientific outcome."""

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.mediated_patterns import event_age_operator_atlas as runner
from experiments.mediated_patterns import event_atlas_model as am
from experiments.mediated_patterns import event_operator_model as om
from experiments.mediated_patterns import present_state_model as pm
from experiments.mediated_patterns.event_age_operator_family import scores
from experiments.mediated_patterns.hybrid_pair import save_npz_exclusive
from experiments.mediated_patterns.present_state_transmission import digest


def fixture():
    rng = np.random.default_rng(817)
    z = rng.normal(size=(24, 3))
    nodes = [10, 14, 17, 20, 25, 30]
    x = np.tile(z, (len(nodes), 1))
    ages = np.repeat(nodes, len(z))
    groups = np.tile(np.repeat(np.arange(12), 2), len(nodes))
    h = np.arange(161) / 80
    shapes = np.array([np.ones(161), h, h * h, np.sin(h)])

    def truth(z, age, a):
        t = (age - 20) / 10
        b = a / 0.02
        r = np.array([1 + z[0], z[1], z[2] ** 2, 0.2]) @ shapes
        d = np.array([0.1 * t * t, 0.02 * z[0] * t, 0.03, 0.01 * z[2]]) @ shapes
        v = r + b * d + b * b * 0.02 * shapes[1]
        return np.column_stack([v, 0.3 * v])

    y = np.array(
        [
            [truth(z, t, a) for a in (0.0, -0.02, 0.02)]
            for z, t in zip(x, ages, strict=True)
        ]
    )
    return x, ages, groups, y, truth


def test_linear_context_is_exact_at_unseen_age():
    nodes = [10, 14, 20, 30]
    values = np.asarray(nodes)[:, None, None] * np.arange(6).reshape(2, 3) + 1
    for family in ("linear", "quadratic", "pchip"):
        np.testing.assert_allclose(
            am.interpolate(nodes, values, 18, family),
            18 * np.arange(6).reshape(2, 3) + 1,
        )


def test_refinement_improves_strong_smooth_curvature():
    target = 15
    coarse = np.array([10, 18, 20, 30])
    refined = np.array([10, 14, 17, 18, 20, 25, 30])
    truth = np.exp(-target / 3)
    for family in ("linear", "quadratic", "pchip"):
        errors = [
            abs(am.interpolate(a, np.exp(-a / 3), target, family) - truth)
            for a in (coarse, refined)
        ]
        assert errors[1] < errors[0]


def test_neighbor_rule_and_exact_nodes():
    nodes = [10, 14, 17, 18, 20, 25, 30]
    assert np.asarray(nodes)[am.neighbors(nodes, 15, "linear")].tolist() == [14, 17]
    assert np.asarray(nodes)[am.neighbors(nodes, 15, "quadratic")].tolist() == [
        14,
        17,
        18,
    ]
    assert np.asarray(nodes)[am.neighbors(nodes, 15, "pchip")].tolist() == [
        10,
        14,
        17,
        18,
    ]
    coarse = [10, 15, 18, 30]
    assert np.asarray(coarse)[am.neighbors(coarse, 20, "quadratic")].tolist() == [
        15,
        18,
        30,
    ]
    for f in ("linear", "quadratic", "pchip"):
        np.testing.assert_array_equal(am.interpolate(nodes, np.arange(7), 17, f), 2)
    for bad in ([10, 10, 20], [20, 10, 30], [10, np.nan, 30]):
        with pytest.raises(ValueError):
            am.interpolate(bad, np.zeros(3), 15, "linear")
    with pytest.raises(ValueError):
        am.interpolate(nodes, np.zeros(7), 31, "linear")


def test_full_age_group_exclusion_and_train_only_coordinates():
    x, ages, groups, y, _ = fixture()
    mask = om.training_mask(groups, ages, 3, 17)
    nodes = sorted(set(ages[mask]))
    assert not mask[groups == 3].any()
    assert not mask[ages == 17].any()
    m = am.fit(x[mask], ages[mask], y[mask], nodes)
    np.testing.assert_array_equal(m["mean"], x[mask].mean(0))
    polluted_x, polluted_y = x.copy(), y.copy()
    polluted_x[~mask], polluted_y[~mask] = 1e100, -1e100
    assert am.fit(polluted_x[mask], ages[mask], polluted_y[mask], nodes) == m
    assert np.asarray(m["coefficients"]).shape == (5, 10, 24)
    assert np.asarray(m["bases"]).shape == (2, 4, 161)
    with pytest.raises(ValueError):
        am.fit(x[mask], ages[mask], y[mask], [10, 14, 17, 20, 25, 30])


def test_quadratic_operator_and_direct_prediction_interpolation():
    x, ages, _groups, y, truth = fixture()
    model = am.fit(x, ages, y, sorted(set(ages)), ridge=1e-12)
    z = np.array([0.2, -0.3, 0.7])
    for a in (0.0, -0.02, 0.02):
        pred = am.predict(model, z, a, 18, "quadratic")
        np.testing.assert_allclose(pred, truth(z, 18, a), atol=2e-12, rtol=1e-10)
        for family in ("linear", "quadratic"):
            np.testing.assert_allclose(
                am.predict(model, z, a, 18, family),
                am.predict(model, z, a, 18, family, True),
                atol=1e-14,
            )


def test_context_query_does_not_mutate_model_or_input_and_serializes():
    x, ages, _groups, y, _truth = fixture()
    model = am.fit(x, ages, y, sorted(set(ages)))
    frozen = json.dumps(model, sort_keys=True, allow_nan=False)
    z = x[0].copy()
    pred = am.predict(model, z, -0.02, 24, "pchip")
    am.predict(model, z, 0.02, 15, "pchip")
    assert json.dumps(model, sort_keys=True, allow_nan=False) == frozen
    np.testing.assert_array_equal(z, x[0])
    np.testing.assert_array_equal(
        pred, am.predict(json.loads(frozen), z, -0.02, 24, "pchip")
    )
    for amplitude in (0.03, np.nan):
        with pytest.raises(ValueError):
            am.predict(model, z, amplitude, 24)


def test_no_runtime_reference_access():
    source = Path(am.__file__).read_text()
    for forbidden in ("simulator", "reference(", "load(", "seed", "history_id"):
        assert forbidden not in source


def test_prediction_seal_is_immutable_before_reference(tmp_path):
    # Same exclusive artifact helpers used by the scientific panels.
    pred = tmp_path / "predictions.npz"
    save_npz_exclusive(pred, y=np.zeros((3, 161, 2)))
    seal = tmp_path / "seal.json"
    pm.save_json(seal, {"prediction_sha256": digest(pred)})

    def future_reference():
        assert json.loads(seal.read_text())["prediction_sha256"] == digest(pred)
        return np.ones((3, 161, 2))

    future_reference()
    with pytest.raises(FileExistsError):
        save_npz_exclusive(pred, y=np.ones((3, 161, 2)))
    with pytest.raises(FileExistsError):
        pm.save_json(seal, {})


def test_safe_output_and_symlink(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    path = tmp_path / "new"
    runner.safe_output(path)
    with pytest.raises(FileExistsError):
        runner.safe_output(path)
    link = tmp_path / "link"
    link.symlink_to(path, target_is_directory=True)
    with pytest.raises(ValueError):
        runner.safe_output(link)


def test_score_reconstruction_and_unresolved_is_not_pass():
    h = np.arange(161)[:, None]
    common = np.ones((161, 2))
    contrast = (h + 1) * np.array([1e-7, -2e-7])
    y = np.array([[common, common - contrast, common + contrast]])
    p = y.copy()
    p[0, 2] += contrast * 0.12
    rows = [{"seed": 1, "history": "synthetic", "age": 18}]
    gates = scores(y, p, rows, {})
    d = [g for g in gates if g["kind"] == "D1" and g["a"] > 0]
    np.testing.assert_allclose([g["relative"] for g in d], 0.12, rtol=1e-8)
    assert all(g["passed"] is False for g in d)
    gates = scores(y, p, rows, {"D1/whole": [1.0, 1.0], "D1/late": [1.0, 1.0]})
    assert all(g["passed"] is None for g in gates if g["kind"] == "D1")


def test_two_atlas_densities_share_exact_coordinates():
    x, ages, _groups, y, _truth = fixture()
    coarse = np.isin(ages, [10, 20, 30])
    rep = om.representation(x[coarse], y[coarse])
    a = am.fit(x[coarse], ages[coarse], y[coarse], [10, 20, 30], rep=rep)
    b = am.fit(x, ages, y, sorted(set(ages)), rep=rep)
    for key in ("mean", "scale", "bases", "scales"):
        assert a[key] == b[key]
    for i, node in enumerate(a["nodes"]):
        j = b["nodes"].index(node)
        np.testing.assert_array_equal(a["coefficients"][i], b["coefficients"][j])


def test_actual_fresh_runner_seals_before_every_future(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from experiments.mediated_patterns import event_atlas_validation as v
    from experiments.mediated_patterns import geometry_assay, geometry_evolution
    from experiments.mediated_patterns import organization_response as org
    from experiments.mediated_patterns import source_receiver_relay as relay

    x, ages, _groups, y, _truth = fixture()
    model = am.fit(x, ages, y, sorted(set(ages)))
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    pm.save_json(frozen / "model.json", model)
    pm.save_json(
        frozen / "freeze.json",
        {
            "sources": {},
            "models": {},
            "development_groups": [1],
            "ages": [24, 25],
            "family": "quadratic",
        },
    )
    monkeypatch.setattr(v, "FRESH", (999,))
    monkeypatch.setattr(v, "source_hashes", dict)
    monkeypatch.setattr(org, "isolated_components", lambda *a: None)
    monkeypatch.setattr(relay, "prepare", lambda *a: np.zeros((2, 768)))
    monkeypatch.setattr(geometry_assay, "initialize_written", lambda state, *a: state)
    monkeypatch.setattr(geometry_evolution, "unforced", lambda state, *a: ([state], []))
    monkeypatch.setattr(pm, "extract", lambda *a, **k: x[0])
    out = tmp_path / "out"
    out.mkdir()
    calls = []

    def current(state, *a):
        assert not (out / "seal.json").exists()
        calls.append("current")
        return state

    def future(*args):
        seal = json.loads((out / "seal.json").read_text())
        assert all(
            digest(out / name) == sha for name, sha in seal["predictions"].items()
        )
        calls.append("future")
        return np.zeros((161, 2))

    monkeypatch.setattr(v, "boundary", current)
    monkeypatch.setattr(v, "response_pair", future)
    v.fresh(
        out,
        SimpleNamespace(data=frozen),
        SimpleNamespace(begin=lambda *a: None, finish=lambda: None),
    )
    assert calls == ["current"] * 4 + ["future"] * 12
    with pytest.raises(FileExistsError):
        v.fresh(
            out,
            SimpleNamespace(data=frozen),
            SimpleNamespace(begin=lambda *a: None, finish=lambda: None),
        )


def test_retained_and_repeated_stages_require_snapshot_pass(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from experiments.mediated_patterns import event_atlas_validation as v

    monkeypatch.setattr(v, "ROOT", tmp_path)
    (tmp_path / "fresh-analysis-01").mkdir()
    pm.save_json(
        tmp_path / "fresh-analysis-01/decision.json", {"snapshot_qualified": False}
    )
    for stage in (v.retained, v.repeated):
        with pytest.raises(ValueError, match="napshot"):
            stage(tmp_path, SimpleNamespace(), None)


def test_retained_query_uses_t50_only_until_prediction_seal(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from experiments.mediated_patterns import event_atlas_validation as v
    from experiments.mediated_patterns import geometry_model as gm

    x, ages, _groups, y, _truth = fixture()
    model = am.fit(x, ages, y, sorted(set(ages)))
    g = gm.fit(x, np.zeros_like(x), kind="drift")
    monkeypatch.setattr(v, "ROOT", tmp_path)
    for folder in ("fresh-analysis-01", "frozen-01", "fresh-01", "retained-01"):
        (tmp_path / folder).mkdir()
    pm.save_json(
        tmp_path / "fresh-analysis-01/decision.json", {"snapshot_qualified": True}
    )
    frozen = tmp_path / "frozen-01"
    pm.save_json(frozen / "model.json", model)
    pm.save_json(frozen / "G.json", g)
    pm.save_json(frozen / "freeze.json", {"family": "quadratic", "floors": {}})
    base, out = tmp_path / "fresh-01", tmp_path / "retained-01"
    save_npz_exclusive(base / "boundary-50.npz", states=np.ones((1, 2, 768)) * 7)
    rows = [
        {"seed": 1, "history": "none", "start_index": 0, "age": a} for a in (24, 25)
    ]
    pm.save_json(base / "rows.json", rows)

    def initial_extract(state, **kwargs):
        np.testing.assert_array_equal(state, np.ones((2, 768)) * 7)
        return x[0]

    monkeypatch.setattr(pm, "extract", initial_extract)

    def evaluator(path):
        assert path == base
        assert (out / "seal.json").exists()
        assert (out / "predictions.npz").exists()
        xx = np.array([x[0], x[0]])
        yy = am.predict_panel(model, xx, [24, 25], "quadratic")
        return xx, yy, rows, np.array([24, 25]), np.ones(2)

    monkeypatch.setattr(v, "dataset", evaluator)
    v.retained(
        out,
        SimpleNamespace(data=base),
        SimpleNamespace(begin=lambda *a: None, finish=lambda: None),
    )
    assert json.loads((out / "summary.json").read_text())["exposed_diagnostic"]
