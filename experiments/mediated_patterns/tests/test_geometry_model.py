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


def test_checkpoint_exclusive_and_solver_free_resume(tmp_path):
    g = fitted()
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
(p/'resumed.json').write_text(json.dumps(s.advance(25).tolist()))
"""
    subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)],
        check=True,
        env={**os.environ, "OPENBLAS_NUM_THREADS": "1"},
    )
    assert np.array_equal(
        json.loads((tmp_path / "resumed.json").read_text()), s.advance(25)
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


def test_train_only_processing_and_grouped_descendants():
    z = np.arange(36, dtype=float).reshape(12, 3)
    groups = np.repeat([1, 2, 3], 4)
    for train, test in pm.grouped_folds(groups):
        assert not set(groups[train]) & set(groups[test])
        g = gm.fit(z[train], z[train] * 1e-5)
        assert np.array_equal(g["mean"], z[train].mean(axis=0))
        changed = z.copy()
        changed[test] += 1e9
        assert g == gm.fit(changed[train], changed[train] * 1e-5)


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
