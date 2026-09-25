"""Engineering boundaries, never assertions of scientific model superiority."""

import json

import numpy as np
import pytest

from experiments.mediated_patterns.explore import start_panel
from experiments.mediated_patterns.recursive_response import (
    assert_run_split,
    grid_index,
    trace,
)
from experiments.mediated_patterns.response_state import (
    CausalKernel,
    ResponseState,
    effective_pulses,
)
from experiments.mediated_patterns.simulator import Config, Field


def synthetic_kernel():
    h = np.zeros((81, 10))
    for k in range(1, len(h)):
        h[k] = 0.92 ** (k - 1) * np.arange(1, 11) - 0.8 ** (k - 1) * np.arange(
            10, 0, -1
        )
    return CausalKernel(0.25, h)


def test_kernel_is_causal_and_rejects_unmeasured_tail():
    kernel = synthetic_kernel()
    assert np.all(kernel.impulse(28, 0.1, [-2, -1, 0]) == 0)
    assert np.all(kernel.predict(28, [(10, 0.1)], np.arange(9)) == 0)
    with pytest.raises(ValueError, match="horizon"):
        kernel.impulse(28, 0.1, [21])
    with pytest.raises(ValueError, match="time zero"):
        CausalKernel.fit(
            [
                {
                    "time": [0, 0.25, 0.5],
                    "pulses": [[0.25, 0.1]],
                    "response": [0, 0, 1],
                    "separation": 28,
                }
            ]
        )


def test_realization_matches_known_markov_sequence():
    kernel = synthetic_kernel()
    model = ResponseState.fit(kernel, order=2, block=20)
    t = np.arange(81) * 0.25
    pulses = [(0, 0.1), (2, -0.075), (8, 0.05)]
    np.testing.assert_allclose(
        model.rollout(28, pulses, t), kernel.predict(28, pulses, t), atol=2e-11
    )


def test_checkpoint_has_only_retained_state_and_resumes_without_fields(
    tmp_path, monkeypatch
):
    model = ResponseState.fit(synthetic_kernel(), order=2, block=20).initialize(27)
    model.susceptibility_gain = 1.2
    model.susceptibility_quadratic_gain = -0.4
    model.susceptibility_tau = 2.3
    model.jump(0.1)
    model.advance(2)
    model.jump(-0.075)  # Include a pending event in the serialized boundary.
    path = tmp_path / "checkpoint.json"
    model.save(path)
    saved = json.loads(path.read_text())
    assert set(saved) == {
        "step",
        "transition",
        "injection",
        "readout",
        "degree",
        "separation",
        "state",
        "pending",
        "time",
        "susceptibility",
        "susceptibility_tau",
        "susceptibility_gain",
        "susceptibility_quadratic_gain",
    }

    def denied(*args, **kwargs):
        raise AssertionError("Field access after initialization")

    monkeypatch.setattr(Field, "step", denied)
    monkeypatch.setattr(Field, "__init__", denied)
    from experiments.mediated_patterns import measurements

    monkeypatch.setattr(measurements, "describe", denied)
    resumed = ResponseState.load(path)
    for k in range(40):
        if k == 10:
            model.jump(0.05)
            resumed.jump(0.05)
        model.advance(0.25)
        resumed.advance(0.25)
        assert model.output() == resumed.output()
    np.testing.assert_array_equal(model.state, resumed.state)


def test_susceptibility_correction_has_same_causal_input_semantics():
    k = synthetic_kernel()
    m = ResponseState.fit(k, order=2, block=20)
    m.susceptibility_tau, m.susceptibility_gain, m.susceptibility_quadratic_gain = (
        2,
        1.2,
        -0.4,
    )
    pulses = [(0, 0.1), (2, -0.1), (7, 0.05)]
    times = np.arange(81) * 0.25
    expected = k.predict(28, effective_pulses(pulses, 2, 1.2, -0.4), times)
    np.testing.assert_allclose(m.rollout(28, pulses, times), expected, atol=2e-11)


def test_exact_event_and_relative_support():
    f = Field(Config(n=128, length=64, dt=0.0125))
    initial = f.seed(centers=(-12, 12))
    changed = f.pulse(initial, -12, 0.1, width=8, relative=True, compact=True)
    np.testing.assert_array_equal(
        changed[:, abs(f.x + 12) >= 8], initial[:, abs(f.x + 12) >= 8]
    )
    np.testing.assert_array_equal(changed[1], initial[1])
    np.testing.assert_allclose(
        f.pulse(2 * initial, -12, 0.1, width=8, relative=True, compact=True),
        2 * changed,
    )
    rows, _ = trace(f, initial, 24, [[(0.025, 0.1)], []], horizon=0.05, sample=0.0125)
    np.testing.assert_allclose(
        rows[0]["response"][:3], 0, atol=1e-18
    )  # m continuous; FFT roundoff.
    assert rows[0]["descriptors"]["mass"][2][0] != rows[1]["descriptors"]["mass"][2][0]
    assert rows[1]["response"] == [0] * 5
    with pytest.raises(ValueError, match="grid"):
        grid_index(0.01, 0.0125)


def test_finite_deterministic_trace_and_new_outputs_only(tmp_path):
    f = Field(Config(n=128, length=64))
    state = f.seed(centers=(-12, 12))
    args = (f, state, 24, [[(0, 0.1), (0.025, -0.1)]])
    a, _ = trace(*args, horizon=0.05, sample=0.0125)
    b, _ = trace(*args, horizon=0.05, sample=0.0125)
    assert a == b
    json.dumps(a, allow_nan=False)
    out = tmp_path / "new"
    start_panel(out, {"split": "engineering test"})
    with pytest.raises(FileExistsError):
        start_panel(out, {})


def test_initialization_only_scalar_geometry_and_grid_updates():
    model = ResponseState.fit(synthetic_kernel(), order=2, block=20)
    with pytest.raises(ValueError, match="Initialize"):
        model.advance(0.25)
    with pytest.raises(ValueError, match="Finite"):
        model.initialize(float("nan"))
    model.initialize(28)
    with pytest.raises(ValueError, match="grid"):
        model.advance(0.1)
    with pytest.raises(ValueError, match="off model grid"):
        model.rollout(28, [(0.1, 0.1)], np.arange(5) * 0.25)
    with pytest.raises(ValueError, match="chronological"):
        model.rollout(28, [(-0.25, 0.1)], np.arange(5) * 0.25)


def test_split_is_by_whole_preparation_not_input_or_snapshot():
    assert_run_split(["s11-d28", "s22-d28"], ["s411-d29"])
    with pytest.raises(ValueError, match="leakage"):
        assert_run_split(["s11-d28"], ["s11-d28"])


def test_nonfinite_recurrence_is_a_failure_not_clipped():
    m = ResponseState.fit(synthetic_kernel(), order=2, block=20).initialize(28)
    m.state[:] = 1
    m.transition[:] = np.inf
    with pytest.raises(FloatingPointError, match="diverged"):
        m.advance(0.25)
