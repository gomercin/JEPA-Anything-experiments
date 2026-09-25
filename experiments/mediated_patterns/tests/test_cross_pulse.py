"""Correctness boundaries; no assertions that the candidate wins on fields."""

import json

import numpy as np
import pytest

from experiments.mediated_patterns import measurements
from experiments.mediated_patterns.cross_pulse import assess, cross_identity
from experiments.mediated_patterns.event_response import CrossResponse
from experiments.mediated_patterns.explore import start_panel
from experiments.mediated_patterns.recursive_response import assert_run_split
from experiments.mediated_patterns.response_state import ResponseState
from experiments.mediated_patterns.simulator import Config, Field


def models():
    # Arbitrary stable engineering fixture, not a fitted scientific outcome.
    rng = np.random.default_rng(55)
    a = np.diag([0.8, 0.9])
    b = rng.normal(size=(2, 10)) * 0.01
    c = np.array([1.0, -0.3])
    base = ResponseState(0.25, a, b, c)
    repair = CrossResponse(
        0.25,
        a.copy(),
        b.copy(),
        c.copy(),
        np.array([1.0, 2.0]),
        np.array([-0.2, 0.5, 2.0, -3.0]),
    )
    return base, repair


def test_multiplicative_pulse_composition_is_not_inverse():
    f = Field(Config(n=128, length=64))
    state = f.seed()
    w = f.pulse(np.ones_like(state), 0, 1, width=8, relative=True, compact=True)[0] - 1
    a = 0.1
    b = -0.1
    two = f.pulse(
        f.pulse(state, 0, a, width=8, relative=True, compact=True),
        0,
        b,
        width=8,
        relative=True,
        compact=True,
    )
    np.testing.assert_allclose(
        two[0], (1 + (a + b) * w + a * b * w * w) * state[0], atol=5e-16
    )
    np.testing.assert_array_equal(two[1], state[1])
    assert not np.allclose(two[0], state[0])
    scalar_inverse = f.pulse(
        f.pulse(state, 0, a, width=8, relative=True, compact=True),
        0,
        -a / (1 + a),
        width=8,
        relative=True,
        compact=True,
    )
    assert not np.allclose(scalar_inverse[0], state[0])
    np.testing.assert_array_equal(two[:, abs(f.x) >= 8], state[:, abs(f.x) >= 8])


def test_four_run_identity_and_correlated_errors():
    rng = np.random.default_rng(42)
    values = rng.normal(size=(6, 30))
    result = cross_identity(*values)
    assert result["identity_max_error"] < 2e-15
    np.testing.assert_allclose(
        result["cross"] + result["kernel_fit"] + result["realization"],
        result["total_error"],
        atol=2e-15,
    )
    terms = np.array([result[k] for k in ("cross", "kernel_fit", "realization")])
    assert not np.isclose(
        np.mean(result["total_error"] ** 2), sum(np.mean(v * v) for v in terms)
    )


def test_first_pulse_and_zero_input_evolution_preserved():
    base, repair = models()
    t = np.arange(81) * 0.25
    for a in (-0.1, 0.075):
        np.testing.assert_allclose(
            repair.rollout(28, [(0, a)], t), base.rollout(28, [(0, a)], t), atol=1e-17
        )
    np.testing.assert_array_equal(repair.rollout(28, [], t), np.zeros(len(t)))


def test_future_events_cannot_change_earlier_outputs():
    _, m = models()
    t = np.arange(81) * 0.25
    a = m.rollout(28, [(0, 0.1)], t)
    b = m.rollout(28, [(0, 0.1), (3, -0.1)], t)
    np.testing.assert_array_equal(a[:13], b[:13])  # continuous sensor at event
    assert not np.array_equal(a[13:], b[13:])
    np.testing.assert_array_equal(b, m.rollout(28, [(0, 0.1), (3, -0.1)], t))


@pytest.mark.parametrize("conditioned", [False, True])
def test_checkpoint_no_field_or_future_history_access(
    tmp_path, monkeypatch, conditioned
):
    _, m = models()
    if conditioned:
        m.separation_knots = np.array([24.0, 26.0, 28.0, 30.0, 32.0])
        m.injection_knots = np.array(
            [m.injection[:, :4] * (1 + d / 100) for d in m.separation_knots]
        )
        m.cross_weights = np.vstack([m.cross_weights, np.zeros(4), np.zeros(4)])
    m.initialize(28)
    m.jump(0.1)
    m.advance(2.5)
    m.jump(-0.09)
    path = tmp_path / "checkpoint.json"
    m.save(path)
    data = json.loads(path.read_text())
    assert set(data) == {
        "step",
        "transition",
        "injection",
        "readout",
        "decay_times",
        "cross_weights",
        "separation",
        "state",
        "memory",
        "pending",
        "pending_cross",
        "time",
        "separation_knots",
        "injection_knots",
    }

    def denied(*args, **kwargs):
        raise AssertionError("No field access permitted")

    monkeypatch.setattr(Field, "step", denied)
    monkeypatch.setattr(Field, "__init__", denied)
    monkeypatch.setattr(measurements, "describe", denied)
    resumed = CrossResponse.load(path)
    for i in range(40):
        if i == 20:
            m.jump(0.05)
            resumed.jump(0.05)
        m.advance(0.25)
        resumed.advance(0.25)
        assert m.output() == resumed.output()
        np.testing.assert_array_equal(m.memory, resumed.memory)


def test_grid_pending_input_and_nonfinite_boundaries(tmp_path):
    _, m = models()
    m.initialize(28)
    m.jump(0.1)
    with pytest.raises(ValueError, match="One finite"):
        m.jump(0.05)
    with pytest.raises(ValueError, match="grid"):
        m.advance(0.1)
    with pytest.raises(ValueError, match="chronological"):
        m.rollout(28, [(0.1, 0.1)], np.arange(5) * 0.25)
    m.cross_weights[0] = np.nan
    with pytest.raises(ValueError):
        m.save(tmp_path / "invalid.json")
    m.cross_weights[0] = 0
    m.transition[:] = np.inf
    m.state[:] = 1
    with pytest.raises(FloatingPointError):
        m.advance(0.25)


def test_static_geometry_interpolation_bounds_and_checkpoint_cache(tmp_path):
    _, m = models()
    m.separation_knots = np.array([24.0, 26.0, 28.0, 30.0, 32.0])
    m.injection_knots = np.array(
        [m.injection[:, :4] * (1 + d / 100) for d in m.separation_knots]
    )
    m.initialize(27.0)
    np.testing.assert_allclose(m.injection, m.injection_knots[0] * (1.27 / 1.24))
    m.save(tmp_path / "model.json")
    restored = CrossResponse.load(tmp_path / "model.json")
    np.testing.assert_array_equal(restored.injection, m.injection)
    with pytest.raises(ValueError, match="geometry"):
        m.initialize(23.9)


def test_split_serialization_and_output_preservation(tmp_path):
    assert_run_split(["s11-d28"], ["s741-d27"])
    with pytest.raises(ValueError):
        assert_run_split(["s11-d28"], ["s11-d28"])
    out = tmp_path / "panel"
    start_panel(out, {"purpose": "engineering"})
    old = (out / "protocol.json").read_bytes()
    with pytest.raises(FileExistsError):
        start_panel(out, {})
    assert (out / "protocol.json").read_bytes() == old


def test_original_scientific_gates_unchanged():
    gates = {
        "per_run_trace_nrmse": 0.05,
        "per_run_max_absolute": 2e-6,
        "response_resolution_floor": 1e-7,
        "early_first_0_to_5_max_absolute": 2e-8,
    }
    row = {
        "time": [0, 6, 7],
        "pulses": [[0, 0.1], [6, -0.1]],
        "response": [0, 1e-5, 1e-5],
    }
    score = assess(row, [0, 1.06e-5, 1.06e-5], gates)
    assert score["failed"] == ["all_relative", "after_last_relative"]
