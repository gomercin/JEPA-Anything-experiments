"""Access and composition contracts, not assertions that science must pass."""

import inspect
import json
import os
import subprocess
import sys

import numpy as np
import pytest

from experiments.mediated_patterns import geometry_model as gm
from experiments.mediated_patterns import present_state_model as pm
from experiments.mediated_patterns.geometry_evolution import (
    center_rates,
    response_pairs,
)
from experiments.mediated_patterns.measurements import regions
from experiments.mediated_patterns.simulator import Field
from experiments.mediated_patterns.source_receiver_relay import CFG


def fitted(kind="affine"):
    rng = np.random.default_rng(41)
    z = rng.normal(size=(30, 3)) * 0.02
    rates = 1e-4 + z @ np.diag([-0.01, -0.02, -0.03])
    return gm.fit(z, rates, kind)


def test_shared_composition_and_analytic_affine():
    g = fitted()
    z = [0.02, -0.01, 0.03]
    whole = gm.rollout(g, z, [0, 10, 25, 50])
    state = gm.Continuation(g, z)
    assert np.array_equal(state.advance(10), whole[1])
    state.advance(15)
    assert np.array_equal(state.advance(25), whole[-1])
    # Independent exact solution for the imposed diagonal affine law.
    lam = np.array([0.01, 0.02, 0.03])
    exact = 1e-4 / lam + (np.array(z) - 1e-4 / lam) * np.exp(-lam * 50)
    np.testing.assert_allclose(whole[-1], exact, rtol=1e-6, atol=1e-10)


def test_persistence_and_shared_drift():
    z = np.array([0.1, 0.2, 0.3])
    assert np.array_equal(gm.rollout(fitted("persistence"), z, [10, 50]), [z, z])
    g = fitted("drift")
    np.testing.assert_allclose(
        gm.rollout(g, z, [50])[0], z + 50 * np.array(g["coefficients"][0])
    )


@pytest.mark.parametrize("kind", ["affine", "quadratic"])
def test_checkpoint_exclusive_and_solver_free_resume(tmp_path, kind):
    g = fitted(kind)
    s = gm.Continuation(g, [0.1, 0.2, 0.3])
    s.advance(25)
    checkpoint = tmp_path / "state.json"
    s.checkpoint(checkpoint)
    with pytest.raises(FileExistsError):
        s.checkpoint(checkpoint)
    alias = tmp_path / "alias.json"
    alias.symlink_to(tmp_path / "absent.json")
    with pytest.raises(FileExistsError):
        s.checkpoint(alias)
    assert not (tmp_path / "absent.json").exists()
    pm.save_json(tmp_path / "model.json", g)
    from experiments.mediated_patterns.geometry_evolution import frozen_response

    response_model = frozen_response()
    pm.save_json(tmp_path / "response-model.json", response_model)
    code = """
import importlib.abc,json,sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.endswith(('simulator','geometry_evolution','history_conditioned_transmission')):
            raise RuntimeError('Reference solver unavailable')
sys.meta_path.insert(0,Block())
from experiments.mediated_patterns.geometry_model import Continuation
from experiments.mediated_patterns.present_state_model import predict
from pathlib import Path
p=Path(sys.argv[1])
g=json.loads((p/'model.json').read_text())
s=Continuation.restore(g,json.loads((p/'state.json').read_text()))
z=s.advance(25)
(p/'resumed.json').write_text(json.dumps(z.tolist()))
f=json.loads((p/'response-model.json').read_text())
(p/'response.json').write_text(json.dumps(predict(f,z[None]).tolist()))
"""
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        check=True,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
    )
    assert np.array_equal(
        json.loads((tmp_path / "resumed.json").read_text()), s.advance(25)
    )
    np.testing.assert_allclose(
        json.loads((tmp_path / "response.json").read_text()),
        pm.predict(response_model, s.z[None]),
        rtol=1e-13,
        atol=1e-20,
    )
    other = fitted("drift")
    with pytest.raises(ValueError, match="mismatch"):
        gm.Continuation.restore(other, json.loads(checkpoint.read_text()))


def test_no_metadata_or_clock_interface():
    assert list(inspect.signature(gm.velocity).parameters) == ["model", "z"]
    assert list(inspect.signature(gm.Continuation).parameters) == ["model", "z"]
    with pytest.raises((TypeError, ValueError)):
        gm.Continuation(fitted(), {"seed": 1, "z": [1, 2, 3]})
    with pytest.raises(ValueError):
        gm.Continuation(fitted(), np.zeros((2, 768)))
    for delays in ([0, 1.5], [1, 1], [-1, 1]):
        with pytest.raises(ValueError):
            gm.rollout(fitted(), [0, 0, 0], delays)


@pytest.mark.parametrize("kind", ["affine", "quadratic"])
def test_train_only_processing_and_grouped_descendants(kind):
    z = np.arange(36, dtype=float).reshape(12, 3)
    groups = np.repeat([1, 2, 3], 4)
    for train, test in pm.grouped_folds(groups):
        assert not set(groups[train]) & set(groups[test])
        g = gm.fit(z[train], z[train] * 1e-5, kind)
        assert np.array_equal(g["mean"], z[train].mean(axis=0))
        changed = z.copy()
        changed[test] += 1e9
        assert g == gm.fit(changed[train], changed[train] * 1e-5, kind)


def test_fixed_window_chain_rule_and_single_snapshot():
    f = Field(CFG)
    state = np.array(
        [
            sum(np.exp(-(((f.x - c) / 3) ** 2)) * np.cos(f.x - c) for c in pm.ANCHORS),
            np.zeros(CFG.n),
        ]
    )
    z = pm.extract(state, feature_set="geometry")
    assert np.max(abs(z)) < 1e-8
    v = np.fft.rfft(state)
    rhs = np.fft.irfft(f.linear * v + f.nonlinear(v), n=CFG.n)
    eps = 1e-6
    fd = (
        pm.extract(state + eps * rhs, feature_set="geometry")
        - pm.extract(state - eps * rhs, feature_set="geometry")
    ) / (2 * eps)
    np.testing.assert_allclose(center_rates(state), fd, atol=3e-10)
    # The actual extractor uses the declared unwrapped fixed local coordinate.
    w = regions(f.x, pm.ANCHORS)
    expected = (w * state[0] ** 2 * (f.x[None] - pm.ANCHORS[:, None])).sum(1) / (
        w * state[0] ** 2
    ).sum(1)
    np.testing.assert_allclose(z, expected, atol=1e-15)
    with pytest.raises(ValueError):
        pm.extract(np.stack([state, state]), feature_set="geometry")


def test_history_pairing_never_crosses_origin_or_preparation():
    rows = [
        {"seed": s, "history": h, "origin": o, "wait": 100}
        for s in [1, 2]
        for o in [50, 60]
        for h in ["none", "odd04"]
    ]
    assert response_pairs(rows) == [[0, 1], [2, 3], [4, 5], [6, 7]]
    with pytest.raises(ValueError):
        response_pairs(rows[1:])


def test_deterministic_serialization(tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    pm.save_json(a, fitted())
    pm.save_json(b, fitted())
    assert a.read_bytes() == b.read_bytes()
    assert gm.identity(json.loads(a.read_text())) == gm.identity(
        json.loads(b.read_text())
    )


def test_signed_residual_and_independent_contrast_identity():
    from experiments.mediated_patterns.geometry_results import error_terms

    truth = np.array([[[2.0, 3.0]], [[4.0, 7.0]]])
    snapshot = truth + np.array([[[0.3, -0.2]], [[0.8, -0.1]]])
    prediction = snapshot + np.array([[[-0.1, 0.4]], [[-0.2, 0.6]]])
    u, r = error_terms(prediction, snapshot, truth)
    np.testing.assert_allclose(u + r, prediction - truth, atol=1e-15)
    du, dr = u[1] - u[0], r[1] - r[0]
    residual = (prediction[1] - prediction[0]) - (truth[1] - truth[0])
    np.testing.assert_allclose(du + dr, residual, atol=1e-15)
    np.testing.assert_allclose(
        np.mean(residual**2), np.mean(du**2) + np.mean(dr**2) + 2 * np.mean(du * dr)
    )


def test_fresh_access_order_saves_before_each_future(monkeypatch, tmp_path):
    from experiments.mediated_patterns import geometry_assay as assay
    from experiments.mediated_patterns.geometry_evolution import F_SHA
    from experiments.mediated_patterns.present_state_transmission import digest

    frozen = tmp_path / "frozen"
    frozen.mkdir()
    out = tmp_path / "out"
    out.mkdir()
    models = {"affine": fitted()}
    pm.save_json(frozen / "models.json", models)
    contract = {
        "models_sha256": digest(frozen / "models.json"),
        "F_sha256": F_SHA,
        "fresh_seeds": list(assay.FRESH),
        "families": assay.FAMILIES,
        "times": {"50": [60, 75, 90, 100], "60": [75, 90, 100]},
    }
    pm.save_json(frozen / "freeze.json", contract)
    log = []
    monkeypatch.setattr(assay, "isolated_components", lambda *args: None)
    monkeypatch.setattr(assay, "prepare", lambda *args: np.zeros((2, 768)))
    monkeypatch.setattr(
        assay, "initialize_written", lambda initial, *args: initial.copy()
    )
    monkeypatch.setattr(
        pm, "extract", lambda state, **kwargs: np.array([state[0, 0], 0, 0])
    )

    def unforced(state, config, times, budget):
        log.append(("evolve", float(state[0, 0]), float(times[-1])))
        return np.array([state + t for t in times]), [{"qualified": True}]

    monkeypatch.setattr(assay, "unforced", unforced)

    def forecasts(out, starts, models, origin, times):
        log.append(("seal", origin))
        assert np.all(starts[:, 0] == origin)

    monkeypatch.setattr(assay, "forecasts", forecasts)

    def reference(state, *args):
        log.append(("probe", float(state[0, 0])))
        return {"response": np.zeros((161, 2))}, [{"qualified": True}]

    monkeypatch.setattr(assay, "reference", reference)

    class Budget:
        def begin(self, *args):
            pass

        def finish(self):
            pass

    assay.fresh(out, frozen, Budget())
    first = log.index(("seal", 50))
    second = log.index(("seal", 60))
    assert first == 9
    assert all(
        x[0] == "evolve" and x[1:] == (50.0, 10.0) for x in log[first + 1 : second]
    )
    assert all(x[0] != "probe" for x in log[: second + 1])
    assert all(x[1] >= 60 for x in log if x[0] == "probe")
