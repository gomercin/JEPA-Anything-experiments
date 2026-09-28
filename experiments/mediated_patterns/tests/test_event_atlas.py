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
