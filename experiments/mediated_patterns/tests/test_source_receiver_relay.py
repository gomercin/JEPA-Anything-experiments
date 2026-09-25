"""Fast invariants for the source contract; no scientific panel in CI."""

from dataclasses import replace

import numpy as np
import pytest

from experiments.mediated_patterns import source_receiver_relay as relay
from experiments.mediated_patterns.organization_response import (
    Budget,
    tangent_nonlinear,
    tangent_step,
)
from experiments.mediated_patterns.simulator import Field


def test_source_event_fixed_anchor_support_and_normalization():
    f = Field(relay.CFG)
    q = relay.source_profile(f)
    assert np.sum(q * q) * f.c.length / f.c.n == pytest.approx(1, abs=4e-16)
    assert not q[abs(f.x - relay.A) >= 8].any()
    for b in relay.POSITIONS:
        w = relay.cut_mask(f, b)
        assert not (w * q).any()
        assert not w[abs(f.x - relay.C) <= 8].any()
        for translated in (False, True):
            m = relay.cut_mask(f, b, translated)
            assert (m[abs(f.x - b) <= 8] == 1).all()
    np.testing.assert_array_equal(relay.cut_mask(f, 0), relay.cut_mask(f, -4))
    np.testing.assert_array_equal(
        relay.weights(f, 0)[0][[0, 2]], relay.weights(f, -4)[0][[0, 2]]
    )


def test_derivative_event_and_source_ownership():
    f = Field(replace(relay.CFG, n=192))
    rng = np.random.default_rng(1)
    s = f.seed(centers=(relay.A, 0, relay.C))
    z = rng.normal(size=(1, 2, f.c.n)) * 0.01
    v, zv = np.fft.rfft(s), np.fft.rfft(z)
    mask = relay.cut_mask(f, 0)
    _, tz, stages = tangent_step(f, v, zv, mask[None])
    h = 1e-5
    plus = f.step(v + h * zv[0], replay=stages, replay_weight=mask)[0]
    minus = f.step(v - h * zv[0], replay=stages, replay_weight=mask)[0]
    np.testing.assert_allclose(tz[0], (plus - minus) / (2 * h), atol=3e-9)
    full = tangent_nonlinear(f, v, zv, np.zeros((1, f.c.n)))
    cut = tangent_nonlinear(f, v, zv, mask[None])
    # Feedback cut does NOT cut the induced source, including B's source.
    np.testing.assert_array_equal(full[:, 1], cut[:, 1])
    np.testing.assert_allclose(
        np.fft.irfft(full[:, 1], n=f.c.n),
        2 * f.c.source * s[0] * z[:, 0] / f.c.tau,
        atol=1e-16,
    )
    fd = (relay.read(f, s + h * z[0], 0) - relay.read(f, s - h * z[0], 0)) / (2 * h)
    np.testing.assert_allclose(fd, relay.read_tangent(f, s, z[0], 0), atol=1e-9)


def test_matched_shams_contrast_and_tangent_short_flow():
    cfg = replace(relay.CFG, n=384)
    f = Field(cfg)
    s = f.seed(centers=(relay.A, 0, relay.C))
    arrays, meta = relay.simulate(cfg, s, 0, [0, 1e-5, -1e-5], Budget(), horizon=0.5)
    assert meta["sham_field_max"] < 2e-14
    for kind in ("full", "cut", "qb"):
        np.testing.assert_allclose(arrays[kind][:, 0], 0, atol=2e-14)
    full, cut, qb = relay.contrasts(arrays["absolute"])
    np.testing.assert_array_equal(qb, full - cut)
    np.testing.assert_array_equal(qb[0], 0)
    for kind in ("full", "cut", "qb"):
        fd = (arrays[kind][:, 1] - arrays[kind][:, 2]) / (2e-5) * relay.EPS
        np.testing.assert_allclose(fd, arrays["tangent_" + kind], atol=4e-11)


def test_preparation_split_and_safe_outputs(tmp_path, monkeypatch):
    relay.validate_split()
    monkeypatch.setattr(relay, "FRESH", relay.DEV + (2, 3))
    with pytest.raises(ValueError, match="disjoint"):
        relay.validate_split()
    monkeypatch.setattr(relay, "ROOT", tmp_path)
    monkeypatch.setattr(relay.subprocess, "check_output", lambda *a, **kw: "")
    out = tmp_path / "unique"
    relay.start_output(out, "test")
    with pytest.raises(FileExistsError):
        relay.start_output(out, "test")
    link = tmp_path / "dangling"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError, match="nonsymlink"):
        relay.start_output(link, "test")
