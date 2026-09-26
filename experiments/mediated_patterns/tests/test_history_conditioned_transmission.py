"""Fast history/event invariants; retention is an empirical outcome, never a test."""

from dataclasses import replace

import numpy as np
import pytest

from experiments.mediated_patterns import history_conditioned_transmission as h
from experiments.mediated_patterns import source_receiver_relay as r
from experiments.mediated_patterns.organization_response import Budget
from experiments.mediated_patterns.simulator import Field


def test_compact_write_event_and_fixed_apparatus():
    f = Field(h.CFG)
    state = f.seed(centers=(h.A, h.B, h.C))
    original = state.copy()
    for kind in ("even", "odd"):
        profile = h.write_profile(f, kind)
        assert np.sum(profile**2) * f.c.length / f.c.n == pytest.approx(1)
        assert not profile[abs(f.x - h.B) >= 8].any()
        assert not (profile * r.source_profile(f)).any()
        assert not profile[abs(f.x - h.C) <= 8].any()
        changed, info = h.event(f, state, profile, h.WRITE)
        np.testing.assert_array_equal(state, original)
        np.testing.assert_array_equal(changed[1], state[1])
        assert info["l2"] == pytest.approx(h.WRITE)
        assert info["energy_jump"] == pytest.approx(
            h.energy(f, changed) - h.energy(f, state)
        )
        assert info["source_jump"] == pytest.approx(
            f.c.source * np.sum(changed[0] ** 2 - state[0] ** 2) * f.c.length / f.c.n
        )
    # Apparatus has no state/center argument: diagnostic drift cannot recenter it.
    np.testing.assert_array_equal(h.write_profile(f, "odd"), h.write_profile(f, "odd"))
    with pytest.raises(ValueError):
        h.write_profile(f, "third")


def test_exact_events_and_common_ancestry():
    cfg = replace(h.CFG, n=384)
    f = Field(cfg)
    state = f.seed(centers=(h.A, h.B, h.C))
    arrays, meta = h.histories(
        cfg, state, [("odd", 0), ("odd", 0.1)], (0.5, 1.0), Budget(), end=1.0
    )
    np.testing.assert_array_equal(arrays["initial"], state)
    np.testing.assert_array_equal(arrays["post_write"][0], arrays["post_write"][1])
    np.testing.assert_array_equal(
        arrays["checkpoints"][:, 0], arrays["checkpoints"][:, 1]
    )
    assert list(arrays["checkpoint_times"]) == [0.5, 1.0]
    assert h.checkpoint(arrays, 0.5).shape == (3, 2, cfg.n)
    with pytest.raises(ValueError):
        h.checkpoint(arrays, 0.51)
    with pytest.raises(ValueError):
        h.grid_steps(0.501, cfg.dt)
    # Uninterrupted no-event history equals the sampled/branched ancestor.
    v = np.fft.rfft(state)
    for _ in range(h.grid_steps(0.5, cfg.dt)):
        v, _ = f.step(v)
    np.testing.assert_array_equal(
        h.checkpoint(arrays, 0.5)[0], np.fft.irfft(v, n=cfg.n)
    )
    assert meta["waits"] == [0.5, 1.0]


def test_four_branch_subtraction_linear_superposition():
    t = np.arange(4.0)
    base, write, probe = t**2, 0.3 * np.exp(-t), 0.02 * np.sin(t)
    y00, y10, y01, y11 = base, base + write, base + probe, base + write + probe
    a = {"time": t, "full": (y01 - y00)[:, None]}
    b = {"time": t, "full": (y11 - y10)[:, None]}
    r0, rw, delta = h.history_contrasts(a, b)
    np.testing.assert_allclose(delta, 0, atol=1e-15)
    np.testing.assert_array_equal(delta, rw - r0)
    with pytest.raises(ValueError):
        h.history_contrasts(a, dict(b, time=t + 1))


def test_each_history_has_own_sham_cut_and_tangent():
    cfg = replace(h.CFG, n=384)
    f = Field(cfg)
    state = f.seed(centers=(h.A, h.B, h.C))
    written, _ = h.event(f, state, h.write_profile(f, "odd"), 0.1)
    outputs = []
    for s in (state, written):
        a, m = r.simulate(
            cfg, s, h.B, [0, 1e-5, -1e-5], Budget(), horizon=0.5, translated=True
        )
        assert m["sham_field_max"] < 2e-14
        np.testing.assert_allclose(a["full"][:, 0], 0, atol=2e-14)
        np.testing.assert_allclose(a["cut"][:, 0], 0, atol=2e-14)
        for key in ("full", "cut", "qb"):
            derivative = (a[key][:, 1] - a[key][:, 2]) / (2e-5) * h.EPS
            np.testing.assert_allclose(derivative, a["tangent_" + key], atol=4e-11)
        np.testing.assert_array_equal(a["mask"], r.cut_mask(f, h.B, True))
        outputs.append(a)
    assert not np.array_equal(
        outputs[0]["absolute"][:, 0], outputs[1]["absolute"][:, 0]
    )
    for key in ("full", "cut", "qb"):
        np.testing.assert_allclose(
            h.history_contrasts(*outputs, key=key)[2], 0, atol=3e-14
        )


def test_splits_and_nonoverwriting_arrays(tmp_path):
    h.validate_split()
    with pytest.raises(ValueError):
        h.validate_split((1,), (1, 2, 3))
    with pytest.raises(ValueError):
        h.validate_split((1,), (2, 3, 3))
    h.save(tmp_path, "once", {"x": np.arange(3)}, {"status": "test"})
    with pytest.raises(FileExistsError):
        h.save(tmp_path, "once", {}, {})
    (tmp_path / "dangling.npz").symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        h.save(tmp_path, "dangling", {}, {})
    assert not (tmp_path / "missing").exists()
    a, m = h.read_saved(tmp_path, "once")
    np.testing.assert_array_equal(a["x"], np.arange(3))
    assert m["status"] == "test"


def test_absolute_history_times_must_match():
    a = {
        "time": np.arange(3.0),
        "absolute_time": np.arange(3.0) + 50,
        "full": np.ones((3, 1)),
    }
    b = dict(a, absolute_time=np.arange(3.0) + 100)
    with pytest.raises(ValueError, match="absolute"):
        h.history_contrasts(a, b)


def test_exact_conditional_mediator_homogeneous_operator():
    f = Field(replace(h.CFG, n=192))
    s = np.zeros((2, f.c.n))
    s[1] = 0.2 + 0.1 * np.cos(2 * np.pi * f.x / f.c.length)
    v = np.fft.rfft(s)
    for _ in range(16):
        v, _ = f.step(v)
    expected = np.fft.irfft(
        np.exp(f.linear[1] * 16 * f.c.dt) * np.fft.rfft(s[1]), n=f.c.n
    )
    np.testing.assert_allclose(np.fft.irfft(v, n=f.c.n)[1], expected, atol=1e-15)


def test_start_output_rejects_existing_and_symlinked_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(h, "ROOT", tmp_path)
    monkeypatch.setattr(h.subprocess, "check_output", lambda *a, **kw: "")
    monkeypatch.setattr(h.subprocess, "run", lambda *a, **kw: None)
    out = tmp_path / "new"
    h.start(out, "test", Budget())
    with pytest.raises(FileExistsError):
        h.start(out, "test", Budget())
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(ValueError, match="nonsymlink"):
        h.start(link, "test", Budget())
