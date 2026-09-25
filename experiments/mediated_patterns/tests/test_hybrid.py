"""Hybrid information/ownership/numerical invariants, not outcome assertions."""

import numpy as np
import pytest

from experiments.mediated_patterns import measurements
from experiments.mediated_patterns.explore import start_panel
from experiments.mediated_patterns.hybrid_pair import (
    PairHybrid,
    pair_indices,
    pulse_window,
)
from experiments.mediated_patterns.recursive_response import assert_run_split
from experiments.mediated_patterns.simulator import Config, Field


def fixture(rank=4):
    cfg = Config(length=96, n=128, dt=0.005)
    field = Field(cfg)
    initial = field.seed(centers=(-14, 14, 38))
    idx = pair_indices(field.x)
    rng = np.random.default_rng(8)
    b = np.linalg.qr(rng.normal(size=(len(idx), rank)))[0]
    return cfg, field, initial, b


def test_resolved_trial_space_reproduces_monolithic_update():
    cfg, f, initial, _ = fixture()
    m = PairHybrid.initialize(cfg, np.eye(len(pair_indices(f.x))), initial)
    v = np.fft.rfft(initial, axis=-1)
    for _ in range(5):
        v, _ = f.step(v)
        m.step()
    np.testing.assert_allclose(
        m.fields(), np.fft.irfft(v, n=cfg.n, axis=-1), atol=2e-11
    )


def test_source_owned_once_and_feedback_units():
    cfg, _, initial, b = fixture()
    m = PairHybrid.initialize(cfg, b, initial)
    force, source = m.nonlinear(m.q, m.m)
    u, mediator = m.fields()
    np.testing.assert_allclose(
        np.fft.irfft(source, n=cfg.n), cfg.source * u * u / cfg.tau, atol=1e-15
    )
    shifted = np.fft.rfft(mediator + 0.2)
    changed, new_source = m.nonlinear(m.q, shifted)
    np.testing.assert_allclose(
        changed - force, cfg.feedback * 0.2 * (m.lift.T @ u), atol=1e-13
    )
    np.testing.assert_array_equal(source, new_source)


def test_pair_grid_has_no_independently_evolving_residual():
    cfg, _, initial, b = fixture()
    m = PairHybrid.initialize(cfg, b, initial)
    assert m.q.shape == (len(m.outside) + b.shape[1],)
    for _ in range(4):
        m.step()
        delta = m.fields()[0, m.inside] - m.template
        np.testing.assert_allclose(delta, b @ (b.T @ delta), atol=1e-13)
    ports = m.ports()
    assert all(v.shape == (b.shape[1],) for v in ports.values())


def test_relative_pulse_and_causal_exchange():
    cfg, f, initial, b = fixture()
    a = PairHybrid.initialize(cfg, b, initial)
    other = PairHybrid.initialize(cfg, b, initial)
    for _ in range(3):
        a.step()
        other.step()
    np.testing.assert_array_equal(a.q, other.q)
    prior = other.fields()
    old_m = other.m.copy()
    other.pulse(38, 0.075)
    expected = f.pulse(prior, 38, 0.075, width=8, relative=True, compact=True)
    # Pulse support is wholly outside removed AB; no projected pulse alteration.
    assert np.all(pulse_window(f.x, 38, cfg.length)[other.inside] == 0)
    np.testing.assert_allclose(other.fields(), expected, atol=3e-14)
    np.testing.assert_array_equal(other.m, old_m)
    a.step()
    other.step()
    assert not np.array_equal(a.q, other.q)


def test_complete_checkpoint_without_reference_or_pair_microstate(
    tmp_path, monkeypatch
):
    cfg, _, initial, b = fixture()
    m = PairHybrid.initialize(cfg, b, initial)
    m.pulse(38, 0.075)
    for _ in range(5):
        m.step()
    path = tmp_path / "hybrid.npz"
    m.save(path)
    data = np.load(path, allow_pickle=False)
    assert set(data.files) == {
        "config",
        "basis",
        "template",
        "spectral_coordinates",
        "mediator",
        "clock",
    }
    assert data["template"].shape == (
        len(m.inside),
    )  # static initial template, allowed
    assert data["spectral_coordinates"].size == b.shape[1] + len(m.outside)

    def denied(*args, **kwargs):
        raise AssertionError("Reference access forbidden")

    monkeypatch.setattr(Field, "__init__", denied)
    monkeypatch.setattr(Field, "step", denied)
    monkeypatch.setattr(measurements, "describe", denied)
    resumed = PairHybrid.load(path)
    for _ in range(5):
        m.step()
        resumed.step()
        np.testing.assert_array_equal(m.q, resumed.q)
        np.testing.assert_array_equal(m.m, resumed.m)
    with pytest.raises(FileExistsError):
        m.save(path)


def test_run_split_and_nonoverwrite(tmp_path):
    assert_run_split(["s71-c38"], ["s2011-c39"])
    with pytest.raises(ValueError):
        assert_run_split(["s71-c38"], ["s71-c38"])
    out = tmp_path / "new"
    start_panel(out, {"purpose": "test"})
    with pytest.raises(FileExistsError):
        start_panel(out, {})


def test_sham_replay_preserves_hybrid_and_nonfinite_is_failure():
    cfg, _, initial, b = fixture()
    a = PairHybrid.initialize(cfg, b, initial)
    c = PairHybrid.initialize(cfg, b, initial)
    weight = np.ones(cfg.n)
    stages = a.step(record=True)
    c.step(replay=stages, replay_weight=weight)
    np.testing.assert_array_equal(a.q, c.q)
    np.testing.assert_array_equal(a.m, c.m)
    c.q[:] = np.inf
    with pytest.raises(FloatingPointError):
        c.step()
