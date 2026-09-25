"""Separate local projections, ownership and restart; no superiority assertions."""

import numpy as np
import pytest

from experiments.mediated_patterns import measurements
from experiments.mediated_patterns.dual_pair import DualHybrid, c_indices
from experiments.mediated_patterns.explore import start_panel, write_json
from experiments.mediated_patterns.hybrid_pair import PairHybrid, pair_indices
from experiments.mediated_patterns.hybrid_schedule import (
    controlled_responses,
    event_ticks,
)
from experiments.mediated_patterns.recursive_response import assert_run_split
from experiments.mediated_patterns.simulator import Config, Field


def fixture():
    cfg = Config(length=96, n=128, dt=0.005)
    f = Field(cfg)
    initial = f.seed(centers=(-14, 14, 38))
    rng = np.random.default_rng(184)
    ab = np.linalg.qr(rng.normal(size=(len(pair_indices(f.x)), 4)))[0]
    cb = np.linalg.qr(rng.normal(size=(len(c_indices(f.x, 38)), 3)))[0]
    return cfg, f, initial, ab, cb


def test_resolved_partition_and_unchanged_ab():
    cfg, f, initial, ab, cb = fixture()
    resolved = DualHybrid.initialize(cfg, 38, None, np.eye(len(cb)), initial)
    v = np.fft.rfft(initial, axis=-1)
    old = PairHybrid.initialize(cfg, ab, initial)
    extended = DualHybrid.initialize(cfg, 38, ab, None, initial)
    full_c = DualHybrid.initialize(cfg, 38, ab, np.eye(len(cb)), initial)
    for _ in range(5):
        v, _ = f.step(v)
        resolved.step()
        old.step()
        extended.step()
        full_c.step()
    np.testing.assert_allclose(
        resolved.fields(), np.fft.irfft(v, n=cfg.n, axis=-1), atol=3e-11
    )
    np.testing.assert_allclose(old.fields(), extended.fields(), atol=2e-13)
    np.testing.assert_allclose(old.fields(), full_c.fields(), atol=3e-11)


def test_source_once_no_hidden_evolving_participant_residual():
    cfg, _, initial, ab, cb = fixture()
    m = DualHybrid.initialize(cfg, 38, ab, cb, initial)
    np.testing.assert_allclose(m.fields(), initial, atol=2e-13)
    assert m.q.size == len(m.outside) + 7
    assert not np.intersect1d(m.ab_idx, m.c_idx).size
    _, source = m.nonlinear(m.q, m.m)
    np.testing.assert_allclose(
        np.fft.irfft(source, n=cfg.n),
        cfg.source * m.fields()[0] ** 2 / cfg.tau,
        atol=1e-14,
    )
    for _ in range(3):
        m.step()
    for idx, b in [(m.ab_idx, ab), (m.c_idx, cb)]:
        delta = (m.fields()[0] - m.offset)[idx]
        np.testing.assert_allclose(delta, b @ (b.T @ delta), atol=1e-13)
    assert m.block_ports("C").shape == (6,)
    assert m.block_ports("AB").shape == (8,)


def test_actual_relative_event_is_projected_and_causal():
    cfg, f, initial, ab, cb = fixture()
    m = DualHybrid.initialize(cfg, 38, ab, cb, initial)
    other = DualHybrid.initialize(cfg, 38, ab, cb, initial)
    for _ in range(3):
        m.step()
        other.step()
    np.testing.assert_array_equal(m.q, other.q)
    prior = m.fields()
    exact = f.pulse(prior, 38, 0.075, width=8, relative=True, compact=True)
    q = m.q.copy()
    mediator = m.m.copy()
    diag = m.pulse(38, 0.075)
    np.testing.assert_allclose(m.q, q + m.lift.T @ (exact[0] - prior[0]), atol=1e-14)
    np.testing.assert_array_equal(m.m, mediator)
    np.testing.assert_allclose(
        m.fields()[0, m.outside], prior[0, m.outside], atol=2e-13
    )
    assert diag["increment_l2_error"] >= 0
    assert event_ticks([[0, 0.075], [0.01, -0.065]], cfg.dt) == [
        (0, 0.075),
        (2, -0.065),
    ]
    with pytest.raises(ValueError):
        event_ticks([[0.007, 0.075]], cfg.dt)


def test_own_sham_control_preserves_no_stimulus():
    from experiments.mediated_patterns.two_way_hybrid import replay_weight

    cfg, _, initial, ab, cb = fixture()
    sham = DualHybrid.initialize(cfg, 38, ab, cb, initial)
    cut_sham = DualHybrid.initialize(cfg, 38, ab, cb, initial)
    for _ in range(5):
        stages = sham.step(record=True)
        cut_sham.step(replay=stages, replay_weight=replay_weight(sham.x))
    np.testing.assert_array_equal(sham.q, cut_sham.q)
    np.testing.assert_array_equal(sham.m, cut_sham.m)


def test_restart_without_truth_or_extractor(tmp_path, monkeypatch):
    cfg, _, initial, ab, cb = fixture()
    m = DualHybrid.initialize(cfg, 38, ab, cb, initial)
    m.pulse(38, 0.075)
    for _ in range(3):
        m.step()
    path = tmp_path / "state.npz"
    m.save(path)
    data = np.load(path, allow_pickle=False)
    assert set(data.files) == {
        "config",
        "c_position",
        "ab_basis",
        "c_basis",
        "template",
        "spectral_coordinates",
        "mediator",
        "clock",
    }
    assert data["spectral_coordinates"].size == m.size

    def forbidden(*a, **kw):
        raise AssertionError("Truth access after initialization")

    monkeypatch.setattr(Field, "__init__", forbidden)
    monkeypatch.setattr(Field, "step", forbidden)
    monkeypatch.setattr(measurements, "describe", forbidden)
    r = DualHybrid.load(path)
    for _ in range(3):
        m.step()
        r.step()
    m.pulse(38, -0.065)
    r.pulse(38, -0.065)
    np.testing.assert_array_equal(m.q, r.q)
    np.testing.assert_array_equal(m.m, r.m)
    with pytest.raises(FileExistsError):
        m.save(path)


def test_control_bookkeeping_split_and_nonoverwrite(tmp_path):
    values = np.random.default_rng(23).normal(size=(4, 4, 6))
    full, cut, contrast = controlled_responses(values)
    np.testing.assert_allclose(
        contrast, values[:, 1] - values[:, 0] - values[:, 2] + values[:, 3]
    )
    np.testing.assert_allclose(contrast, full - cut)
    assert_run_split(["s71-c38", "s82-c42"], ["s4011-c39"])
    with pytest.raises(ValueError):
        assert_run_split(["s71-c38"], ["s71-c38"])
    start_panel(tmp_path / "panel", {"test": True})
    with pytest.raises(FileExistsError):
        start_panel(tmp_path / "panel", {"test": True})
    with pytest.raises(ValueError):
        write_json(tmp_path / "nonfinite.json", {"value": float("nan")})
