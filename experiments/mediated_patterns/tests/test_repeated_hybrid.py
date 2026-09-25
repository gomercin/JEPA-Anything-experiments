"""Repeated-event information, scheduling, control and persistence invariants."""

import json

import numpy as np
import pytest

from experiments.mediated_patterns import measurements
from experiments.mediated_patterns.explore import start_panel, write_json
from experiments.mediated_patterns.hybrid_pair import (
    PairHybrid,
    pair_indices,
    pulse_window,
)
from experiments.mediated_patterns.hybrid_schedule import (
    ScheduledHybrid,
    controlled_responses,
    event_ticks,
)
from experiments.mediated_patterns.recursive_response import assert_run_split
from experiments.mediated_patterns.repeated_hybrid import run_sequence, windows
from experiments.mediated_patterns.simulator import Config, Field


def setup():
    c = Config(length=96, n=128, dt=0.005)
    f = Field(c)
    initial = f.seed(centers=(-14, 14, 38))
    b = np.linalg.qr(
        np.random.default_rng(12).normal(size=(len(pair_indices(f.x)), 4))
    )[0]
    return c, f, initial, b


def test_events_are_exact_bounded_and_causal():
    c, _, initial, b = setup()
    a = ScheduledHybrid(
        PairHybrid.initialize(c, b, initial), 38, [[0, 0.075], [0.05, -0.075]]
    )
    other = ScheduledHybrid(
        PairHybrid.initialize(c, b, initial), 38, [[0, 0.075], [0.05, 0.065]]
    )
    for _ in range(9):
        a.step()
        other.step()
        np.testing.assert_array_equal(a.model.q, other.model.q)
    before = a.model.fields()
    a.step()
    other.step()
    assert not np.array_equal(a.model.q, other.model.q)
    assert a.tick == 10 and a.next_event == 2
    assert np.all(pulse_window(a.model.x, 38, c.length)[a.model.inside] == 0)
    assert np.isfinite(before).all()
    for invalid in [
        [[0.001, 0.05]],
        [[0, 0.11]],
        [[0, 0.05], [0, -0.05]],
        [[float("nan"), 0.05]],
    ]:
        with pytest.raises(ValueError):
            event_ticks(invalid, c.dt)


def test_relative_opposite_jumps_are_not_inverse():
    c, f, state, _ = setup()
    a = 0.075
    w = pulse_window(f.x, 38, c.length)
    first = f.pulse(state, 38, a, width=8, relative=True, compact=True)
    second = f.pulse(first, 38, -a, width=8, relative=True, compact=True)
    np.testing.assert_allclose(second[0], (1 - a * a * w * w) * state[0], atol=1e-15)
    np.testing.assert_array_equal(second[1], state[1])


def test_controlled_contrast_subtracts_each_own_sham():
    rng = np.random.default_rng(2)
    a = rng.normal(size=(5, 4, 6))
    full, cut, contrast = controlled_responses(a)
    np.testing.assert_allclose(contrast, a[:, 1] - a[:, 0] - a[:, 2] + a[:, 3])
    a[:, 2:] += 42
    new = controlled_responses(a)
    np.testing.assert_allclose(full, new[0])
    np.testing.assert_allclose(cut, new[1], atol=1e-14)
    np.testing.assert_allclose(contrast, new[2], atol=1e-14)


def test_zero_events_and_sham_do_not_change_partition():
    c, _, initial, b = setup()
    empty = run_sequence(c, initial, 38, [], 1, basis=b)
    zeros = run_sequence(c, initial, 38, [[0, 0], [0.5, 0]], 1, basis=b)
    np.testing.assert_array_equal(empty["absolute"], zeros["absolute"])
    np.testing.assert_array_equal(zeros["response"], np.zeros((3, 6)))
    assert zeros["sham_replay_max"] == 0


def test_checkpoint_before_second_event_has_no_truth_dependencies(
    tmp_path, monkeypatch
):
    c, _, initial, b = setup()
    a = ScheduledHybrid(
        PairHybrid.initialize(c, b, initial),
        38,
        [[0, 0.075], [0.05, -0.075], [0.08, 0.06]],
    )
    for _ in range(6):
        a.step()
    a.save(tmp_path / "checkpoint")

    def deny(*args, **kwargs):
        raise AssertionError("Truth/extraction forbidden")

    monkeypatch.setattr(Field, "step", deny)
    monkeypatch.setattr(Field, "__init__", deny)
    monkeypatch.setattr(measurements, "describe", deny)
    resumed = ScheduledHybrid.load(tmp_path / "checkpoint")
    for _ in range(14):
        a.step()
        resumed.step()
        np.testing.assert_array_equal(a.model.q, resumed.model.q)
        np.testing.assert_array_equal(a.model.m, resumed.model.m)
    assert a.next_event == 3
    assert set(json.loads((tmp_path / "checkpoint/schedule.json").read_text())) == {
        "center",
        "events",
        "tick",
        "next_event",
        "state_sha256",
        "contract",
    }
    with pytest.raises(FileExistsError):
        a.save(tmp_path / "checkpoint")


def test_split_nonoverwrite_and_strict_serialization(tmp_path):
    assert_run_split(["s2011-c39"], ["s3011-c39"])
    with pytest.raises(ValueError):
        assert_run_split(["s2011-c39"], ["s2011-c39"])
    start_panel(tmp_path / "out", {"test": True})
    with pytest.raises(FileExistsError):
        start_panel(tmp_path / "out", {})
    with pytest.raises(ValueError):
        write_json(tmp_path / "invalid.json", {"value": float("inf")})
    assert windows([[0, 0.075], [8, -0.075]], 80)["last_return"] == (18, 80)
