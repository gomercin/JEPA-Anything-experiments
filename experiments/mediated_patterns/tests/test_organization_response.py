"""Instrument checks on tiny/short fixtures, never a scientific panel."""

from dataclasses import replace

import numpy as np
import pytest

from experiments.mediated_patterns.explore import start_panel
from experiments.mediated_patterns.organization_response import (
    CFG,
    DEV_SEEDS,
    FRESH_SEEDS,
    Budget,
    cut_weight,
    gain_delay,
    metric,
    probes,
    pulse,
    read,
    read_tangent,
    shifted,
    simulate,
    tangent_nonlinear,
    tangent_step,
    validate_split,
)
from experiments.mediated_patterns.simulator import Field


def test_fixed_actuator_l2_support_and_independent_channels():
    field = Field(CFG)
    q = probes(field)
    dx = CFG.length / CFG.n
    np.testing.assert_allclose((q * q).sum(axis=1) * dx, 1, atol=3e-16)
    assert abs(np.dot(*q) * dx) < 1e-15
    assert not q[:, abs(field.x - 39) >= 8].any()
    initial = field.seed(centers=(-10, 14, 39), variation=0.01)
    other = initial * 1.5
    for state in (initial, other):
        after = pulse(state, q[0], 0.02)
        np.testing.assert_allclose(after[0] - state[0], 0.02 * q[0], atol=2e-16)
        np.testing.assert_array_equal(after[1], state[1])


def test_symmetry_requires_complementary_sensor():
    f = Field(CFG)
    state = f.seed(centers=(39,))
    perturbation = np.zeros((2, 2, CFG.n))
    perturbation[:, 0] = probes(f)
    y = read_tangent(f, state, perturbation)
    assert abs(y[0, 0]) > 1
    assert abs(y[1, 1]) > 0.01
    assert abs(y[1, 0]) < 1e-14
    assert abs(y[0, 1]) < 1e-14


def test_readout_derivative_and_historical_event_map():
    f = Field(CFG)
    s = f.seed(centers=(39,))
    q = probes(f)[0]
    z = np.zeros_like(s)
    z[0] = q
    h = 1e-5
    fd = (read(f, s + h * z) - read(f, s - h * z)) / (2 * h)
    np.testing.assert_allclose(fd, read_tangent(f, s, z), atol=1e-10)
    window = (
        f.pulse(np.ones_like(s), 39, 1, width=8, relative=True, compact=True)[0] - 1
    )
    plus = f.pulse(s, 39, h, width=8, relative=True, compact=True)
    minus = f.pulse(s, 39, -h, width=8, relative=True, compact=True)
    np.testing.assert_allclose(
        (plus[0] - minus[0]) / (2 * h), window * s[0], atol=2e-11
    )


def test_stage_derivative_and_same_reference_step():
    f = Field(replace(CFG, n=96))
    rng = np.random.default_rng(7)
    state = rng.normal(size=(2, f.c.n)) * 0.1
    z = rng.normal(size=(1, 2, f.c.n)) * 0.1
    v, zv = np.fft.rfft(state), np.fft.rfft(z)
    weights = np.zeros((1, f.c.n))
    result, prediction, _ = tangent_step(f, v, zv, weights)
    np.testing.assert_array_equal(result, f.step(v)[0])
    h = 1e-5
    fd = (f.step(v + h * zv[0])[0] - f.step(v - h * zv[0])[0]) / (2 * h)
    np.testing.assert_allclose(prediction[0], fd, atol=5e-11)


def test_cut_removes_only_induced_mediator_feedback():
    f = Field(replace(CFG, n=96))
    state = f.seed(centers=(-10, 14, 39))
    rng = np.random.default_rng(21)
    z = rng.normal(size=(1, 2, f.c.n))
    mask = cut_weight(f.x)
    v, zv = np.fft.rfft(state), np.fft.rfft(z)
    full = tangent_nonlinear(f, v, zv, np.zeros((1, f.c.n)))
    cut = tangent_nonlinear(f, v, zv, mask[None])
    diff = np.fft.irfft(full - cut, n=f.c.n)
    np.testing.assert_allclose(
        diff[0, 0], f.c.feedback * state[0] * z[0, 1] * mask, atol=1e-15
    )
    np.testing.assert_array_equal(full[:, 1], cut[:, 1])
    assert not mask[abs(f.x - 39) <= 8].any()


def test_matched_sham_subtraction_zero_stimulus_and_t0_return():
    c = replace(CFG, n=384)
    f = Field(c)
    budget = Budget()
    initial = f.seed(centers=(-10, 14, 39))
    result = simulate(
        c, initial, -10, [("zero", 1, 0, 0), ("even", 1, 0, 0.02)], budget, horizon=0.5
    )
    assert result["sham_field_max"] < 2e-14
    for kind in ("full", "cut", "return"):
        np.testing.assert_allclose(np.array(result[kind])[:, 0], 0, atol=2e-14)
    np.testing.assert_array_equal(np.array(result["return"])[0], 0)
    a = np.asarray(result["absolute"])
    expected = (a[:, 4] - a[:, 0]) - (a[:, 5] - a[:, 1])
    np.testing.assert_array_equal(np.asarray(result["return"])[:, 1], expected)


def test_shared_gain_delay_and_window_not_cropped():
    times = np.arange(0, 20.5, 0.5)
    trace = np.stack([np.exp(-times / 3), np.exp(-times / 8)], axis=-1)[:, None, :]
    target = -1.2 * shifted(trace, times, 1)
    fit = gain_delay([(trace, target)], times, np.ones(2), [-1, 0, 1, 2])
    assert fit["gain"] == pytest.approx(-1.2)
    assert fit["delay"] == 1
    np.testing.assert_array_equal(shifted(trace, times, 1)[:2], 0)
    assert shifted(trace, times, -1).shape == trace.shape


def test_absolute_floor_prevents_tiny_ratio_claim():
    score = metric(
        np.ones((5, 2)) * 1e-14, np.zeros((5, 2)), np.ones(2), np.ones(2) * 1e-10
    )
    assert not score["practical_failure"]
    assert score["resolved"] == [False, False]


def test_preparation_split_and_nonoverwriting_output(tmp_path):
    validate_split({"development_seeds": DEV_SEEDS, "fresh_seeds": FRESH_SEEDS})
    with pytest.raises(ValueError, match="cross"):
        validate_split({"development_seeds": [1], "fresh_seeds": [1, 2, 3]})
    with pytest.raises(ValueError, match="three"):
        validate_split({"development_seeds": [1], "fresh_seeds": [2, 3]})
    out = tmp_path / "unique"
    start_panel(out, {"purpose": "instrument-test"})
    with pytest.raises(FileExistsError):
        start_panel(out, {})


def test_resource_stop():
    budget = Budget(previous=1800)
    with pytest.raises(RuntimeError, match="Aggregate"):
        budget.check()
