"""Snapshot instrument and exposure tests, not assertions about SH35 sufficiency."""

import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.mediated_patterns import event_state_model as sm
from experiments.mediated_patterns import present_state_model as pm
from experiments.mediated_patterns.event_state_analysis import prediction
from experiments.mediated_patterns.hybrid_pair import save_npz_exclusive
from experiments.mediated_patterns.measurements import regions


def synthetic():
    rng = np.random.default_rng(7)
    centers = rng.normal(size=(14, 3))
    x = np.c_[
        np.repeat(centers, 2, axis=0), np.tile([-1.0, 1.0], 14), np.zeros((28, 2))
    ]
    h = np.arange(161) / 2
    curve = np.exp(-h / 17)[:, None] * np.array([1.0, -2.0])[None]
    base = np.ones((161, 2)) * [10, -5]
    y = np.array([[base, base - v[3] * curve, base + v[3] * curve] for v in x])
    return x, y


def test_coarse_collision_and_augmented_synthetic_fixture():
    x, y = synthetic()
    c = sm.fit(x, y, "centers")
    a = sm.fit(x, y, "overlap")
    np.testing.assert_array_equal(sm.predict(c, x[0], 0.02), sm.predict(c, x[1], 0.02))
    assert np.max(abs(y[0, 2] - y[1, 2])) > 1
    np.testing.assert_allclose(prediction(a, x), y, atol=1e-6)


def test_extraction_and_signed_source_overlap():
    n, length = 768, 192.0
    xx = np.arange(n) * length / n - length / 2
    u = np.cos(xx) * sum(np.exp(-(((xx - a) / 3) ** 2)) for a in pm.ANCHORS)
    m = 0.3 + 0.1 * np.sin(xx)
    q = np.exp(-(((xx - pm.ANCHORS[0]) / 2) ** 2)) * (abs(xx - pm.ANCHORS[0]) < 8)
    state = np.array([u, m])
    x = sm.extract(state, q)
    w = regions(xx, pm.ANCHORS)[0]
    np.testing.assert_array_equal(x[:3], pm.extract(state, feature_set="geometry"))
    assert x[3] == pytest.approx(np.sum(w * u * q) * length / n)
    assert x[4] == pytest.approx(np.sum(w * m) / np.sum(w))
    assert x[5] == pytest.approx(np.sum(w * u * u) * length / n)
    for a in [-0.02, 0.02]:
        changed = state.copy()
        changed[0] += a * q
        actual = np.sum(w * (changed[0] ** 2 - u * u)) * length / n
        expected = 2 * a * x[3] + a * a * np.sum(w * q * q) * length / n
        assert actual == pytest.approx(expected, abs=1e-15)
        np.testing.assert_array_equal(sm.extract(changed, q)[1:3], x[1:3])
    with pytest.raises(ValueError):
        sm.extract(state, q[:-1])


def test_response_blind_matching_and_tolerance():
    assert list(inspect.signature(sm.choose_pairs).parameters) == [
        "descriptors",
        "groups",
        "uncertainty",
    ]
    x = np.array(
        [[0, 0, 0, 1], [1e-5, 1e-5, 1e-6, 2], [0.2, 0.3, 0.4, 3], [0.21, 0.31, 0.41, 4]]
    )
    s = sm.choose_pairs(x, [1, 2, 1, 3], [1e-8] * 3)
    assert s["candidate_pairs"][0]["indices"] == [0, 1]
    assert s["candidate_pairs"][0]["near"]
    assert all(p["indices"] != [0, 2] for p in s["all_pairs"])
    x[:, 3] *= -7
    t = sm.choose_pairs(x, [1, 2, 1, 3], [1e-8] * 3)
    assert [p["indices"] for p in t["candidate_pairs"][:5]] == [
        p["indices"] for p in s["candidate_pairs"][:5]
    ]
    tiny = sm.choose_pairs(
        np.array([[0, 0, 0, 1], [1e-10, 0, 0, 2]]), [1, 2], [1e-8] * 3
    )
    assert not tiny["candidate_pairs"][0]["near"]
    np.testing.assert_allclose(tiny["scale"], [5e-4, 2e-4, 2e-5], rtol=1e-15)


def test_train_only_scaling_basis_and_deterministic_serialization():
    x, y = synthetic()
    train = np.arange(len(x)) < 20
    m = sm.fit(x[train], y[train], "overlap")
    np.testing.assert_allclose(m["mean"], x[train, :4].mean(0))
    expected = sm.predict(m, x[-1], -0.02)
    x[~train] *= 1e9
    y[~train] *= 1e9
    assert sm.fit(x[train], y[train], "overlap") == m
    np.testing.assert_array_equal(
        sm.predict(
            json.loads(json.dumps(m, sort_keys=True)), synthetic()[0][-1], -0.02
        ),
        expected,
    )
    assert list(inspect.signature(sm.predict).parameters) == [
        "model",
        "descriptor",
        "amplitude",
    ]
    for a in [float("nan"), 0.021]:
        with pytest.raises(ValueError):
            sm.predict(m, x[0], a)


def test_runtime_has_no_solver_or_response_files(tmp_path):
    package = tmp_path / "runtime"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name in ["event_state_model", "present_state_model", "measurements"]:
        shutil.copyfile(
            Path(sm.__file__).with_name(name + ".py"), package / (name + ".py")
        )
    x, y = synthetic()
    m = sm.fit(x, y, "overlap")
    (tmp_path / "state.json").write_text(
        json.dumps({"model": m, "descriptor": x[0].tolist()})
    )
    (tmp_path / "run.py").write_text("""
import builtins,json,sys
sys.path.insert(0,'.')
original=builtins.__import__
def guard(name,*args,**kwargs):
    if name.startswith(('scipy','experiments')) or 'simulator' in name:raise RuntimeError('unavailable')
    return original(name,*args,**kwargs)
builtins.__import__=guard
from runtime.event_state_model import predict
s=json.load(open('state.json'));print(json.dumps(predict(s['model'],s['descriptor'],.02).tolist()))
""")
    result = subprocess.check_output(
        [sys.executable, "-I", "run.py"], cwd=tmp_path, text=True
    )
    np.testing.assert_array_equal(json.loads(result), sm.predict(m, x[0], 0.02))


def test_fresh_seal_precedes_future_reference(monkeypatch, tmp_path):
    from experiments.mediated_patterns import event_state_analysis as sa
    from experiments.mediated_patterns import organization_response as org
    from experiments.mediated_patterns import source_receiver_relay as relay
    from experiments.mediated_patterns.geometry_model import identity

    x, y = synthetic()
    m = sm.fit(x, y, "overlap")
    frozen = tmp_path / "frozen"
    frozen.mkdir()
    pm.save_json(frozen / "model.json", m)
    pm.save_json(frozen / "freeze.json", {"sources": {}, "model_sha256": identity(m)})
    monkeypatch.setattr(sa, "source_hashes", dict)
    monkeypatch.setattr(sa, "development", lambda: (None, None, [{"seed": 1}]))
    monkeypatch.setattr(sa, "FRESH", [99])
    monkeypatch.setattr(org, "isolated_components", lambda *a: None)
    monkeypatch.setattr(relay, "prepare", lambda *a: np.zeros((2, 768)))
    monkeypatch.setattr(sa, "initialize_written", lambda s, *a: s)
    monkeypatch.setattr(sa, "unforced", lambda s, *a: ([s], []))
    monkeypatch.setattr(sa, "reach_boundary", lambda s, *a: s)
    monkeypatch.setattr(sa, "descriptor", lambda s: x[0])
    out = tmp_path / "fresh"
    out.mkdir()
    calls = []

    def reference(*a):
        assert (out / "seal.json").is_file()
        assert (out / "predictions.npz").is_file()
        calls.append(a[3])
        return np.zeros((161, 2))

    monkeypatch.setattr(sa, "response_pair", reference)
    budget = SimpleNamespace(begin=lambda *a: None, finish=lambda: None)
    sa.fresh(out, SimpleNamespace(freeze=frozen), budget)
    assert calls == [0.0, -0.02, 0.02] * 2
    with pytest.raises(FileExistsError):
        sa.fresh(out, SimpleNamespace(freeze=frozen), budget)


def test_safe_snapshot_outputs(tmp_path):
    p = tmp_path / "one.npz"
    save_npz_exclusive(p, x=[1])
    with pytest.raises(FileExistsError):
        save_npz_exclusive(p, x=[2])
    link = tmp_path / "link.npz"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        save_npz_exclusive(link, x=[2])
