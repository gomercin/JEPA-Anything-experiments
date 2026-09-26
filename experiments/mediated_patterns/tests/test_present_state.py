"""Fast data-flow invariants; no scientific simulation or model-winning tests."""

import builtins
import json

import numpy as np
import pytest

from experiments.mediated_patterns import present_state_model as m
from experiments.mediated_patterns.hybrid_pair import save_npz_exclusive
from experiments.mediated_patterns.present_state_transmission import paired_indices


def snapshot():
    x = np.arange(768) / 4 - 96
    u = sum(np.cos(x - c) * np.exp(-(((x - c) / 4) ** 8)) for c in m.ANCHORS)
    return np.array([u, 0.2 + x / 1000])


def training():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(12, 15))
    y = (1 + z[:, :1, None] / 20) * np.stack([m.TIMES, m.TIMES**2 / 80], axis=1)
    return z, y


def test_current_checkpoint_loader_ignores_future_and_metadata(tmp_path):
    p = tmp_path / "history.npz"
    s = snapshot()
    save_npz_exclusive(
        p,
        checkpoint_times=[50, 100],
        checkpoints=np.array([[s, s], [s, s]]),
        future=np.full((8, 2, 768), np.nan),
        seed=np.array({"private": 1}, dtype=object),
    )
    np.testing.assert_array_equal(m.load_checkpoint(p, 50, 0), s)
    with pytest.raises(ValueError):
        m.load_checkpoint(p, 51, 0)
    with pytest.raises(ValueError):
        m.extract(np.array([s, s]))
    with pytest.raises((TypeError, ValueError)):
        m.extract({"state": s, "seed": 1})


def test_fixed_anchors_and_single_snapshot():
    s = snapshot()
    before = s.copy()
    z = m.extract(s)
    np.testing.assert_allclose(z[:3], 0, atol=1e-12)
    shifted = np.roll(s, 1, axis=1)
    np.testing.assert_allclose(m.extract(shifted)[:3], 0.25, atol=1e-6)
    np.testing.assert_array_equal(s, before)
    np.testing.assert_array_equal(m.extract(s, feature_set="geometry"), z[:3])
    changed_m = s.copy()
    changed_m[1] *= 100
    np.testing.assert_array_equal(m.extract(changed_m, feature_set="geometry"), z[:3])


def test_grouped_split_keeps_every_descendant():
    groups = np.repeat([5, 8, 20], 6)
    for train, test in m.grouped_folds(groups):
        assert set(groups[train]).isdisjoint(groups[test])
        assert len(test) == 6
    with pytest.raises(ValueError):
        m.grouped_folds([1, 1, 2])


def test_train_only_preprocessing_and_temporal_basis():
    z, y = training()
    fitted = m.fit(z[:8], y[:8], [], "geometry")
    np.testing.assert_allclose(fitted["mean"], z[:8, :3].mean(axis=0))
    z[8:] = 1e10
    y[8:] = -1e10
    assert fitted == m.fit(z[:8], y[:8], [], "geometry")
    assert fitted["retained_training_examples"] == 0


def test_inference_from_serialized_inputs_without_solver(tmp_path, monkeypatch):
    z, y = training()
    fitted = m.fit(z, y, [[0, 1]], "geometry")
    p, q = tmp_path / "model.json", tmp_path / "model-again.json"
    m.save_json(p, fitted)
    m.save_json(q, fitted)
    assert p.read_bytes() == q.read_bytes()
    original = builtins.__import__

    def no_solver(name, *args, **kwargs):
        if "simulator" in name or "transmission" in name or "relay" in name:
            raise AssertionError("Reference access forbidden")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_solver)
    predicted = m.predict(
        json.loads(p.read_text()), json.loads(json.dumps(z[:, :3].tolist()))
    )
    assert predicted.shape == y.shape
    with pytest.raises(ValueError):
        m.predict(fitted, z)  # undeclared feature access rejected
    with pytest.raises(ValueError):
        m.predict(fitted, z[:, :3], probe=0.03)


def test_paired_subtraction_and_blind_failure():
    rows = [{"seed": 3, "wait": 50, "history": h} for h in ["none", "write"]]
    truth = np.ones((2, 161, 2))
    truth[1] *= 1.1
    prediction = np.ones_like(truth)
    pairs = paired_indices(rows)
    assert pairs == [[0, 1]]
    s = m.scores(truth, prediction, pairs, rows)
    assert all(r["relative_rms"] == 1 for r in s if r["kind"] == "Delta_R")
    assert all(not r["passed"] for r in s if r["kind"] == "Delta_R")
    unresolved = m.scores(truth, prediction, pairs, rows, np.ones((2, 2)))
    assert all(
        not r["resolved"] and not r["passed"]
        for r in unresolved
        if r["kind"] == "Delta_R"
    )


def test_exclusive_output(tmp_path):
    p = tmp_path / "model.json"
    m.save_json(p, {"x": 1})
    with pytest.raises(FileExistsError):
        m.save_json(p, {"x": 2})
    q = tmp_path / "fields"
    save_npz_exclusive(q, z=[1])
    with pytest.raises(FileExistsError):
        save_npz_exclusive(q, z=[2])
    link = tmp_path / "dangling.json"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(FileExistsError):
        m.save_json(link, {})
