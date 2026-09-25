"""Instrument boundaries only: no model ranking assertions."""

import json
from dataclasses import replace

import pytest
import torch
from jepa_anything_core.baselines import parameter_count

from experiments.coupled_oscillators.evaluate import evaluate
from experiments.coupled_oscillators.frozen_encoder import PhysicalPredictor, make_model
from experiments.coupled_oscillators.interrogate import orthogonal_draws
from experiments.coupled_oscillators.nonlinear import (
    DuffingSimulator,
    PolynomialPredictor,
    polynomial_features,
)
from experiments.coupled_oscillators.phase_b import IDENTITY, physical_splits, train_frozen
from experiments.coupled_oscillators.phase_b_followup import (
    fold_opf_predictor,
    output_transform_control,
)
from experiments.coupled_oscillators.run_experiment import Config, write_json
from experiments.coupled_oscillators.simulator import Simulator


def test_zero_alpha_is_bitwise_phase_a():
    expected = Simulator().trajectories(8, 32, 11)
    for substeps in (1, 4, 8):
        assert torch.equal(
            DuffingSimulator(alpha=0, substeps=substeps).trajectories(8, 32, 11), expected
        )


@pytest.mark.parametrize("alpha", [0.1, 0.5])
def test_nonlinear_determinism_energy_and_convergence(alpha):
    simulator = DuffingSimulator(alpha=alpha, substeps=8)
    states = simulator.trajectories(8, 32, 11)
    assert torch.equal(states, simulator.trajectories(8, 32, 11))
    initial = 1.5 * states[:, 0]
    shifted = simulator.simulate(initial, 32)
    refined = replace(simulator, substeps=16).simulate(initial, 32)
    assert torch.isfinite(shifted).all()
    assert (shifted - refined).abs().max() < 1e-5
    energy = simulator.energy(shifted)
    assert (energy[:, 1:] - energy[:, :-1]).max() < 1e-9
    assert states.shape == (8, 33, 6)
    with pytest.raises(ValueError):
        DuffingSimulator(alpha=-1)
    with pytest.raises(ValueError, match="state-independent"):
        simulator.transition()


def test_polynomial_features_and_identification_are_deterministic():
    rng = torch.Generator().manual_seed(12)
    inputs = torch.randn(256, 6, generator=rng, dtype=torch.float64)
    features = polynomial_features(inputs)
    assert features.shape == (256, 84)
    assert torch.equal(features, polynomial_features(inputs))
    assert torch.equal(features[:, 0], torch.ones(256, dtype=torch.float64))
    torch.testing.assert_close(features[:, 1:7], inputs)
    weights = torch.randn(84, 6, generator=rng, dtype=torch.float64)
    targets = features @ weights
    first, second = PolynomialPredictor(inputs, targets), PolynomialPredictor(inputs, targets)
    torch.testing.assert_close(first.weights, second.weights)
    unseen = torch.randn(20, 6, generator=rng, dtype=torch.float64)
    torch.testing.assert_close(
        first(unseen), polynomial_features(unseen) @ weights, atol=1e-11, rtol=1e-11
    )


def test_nonlinear_oracle_evaluator_uses_nonlinear_pulse_truth():
    simulator = DuffingSimulator(alpha=0.5, substeps=8)
    physical = simulator.trajectories(4, 32, 11)

    def predict(state):
        return simulator.simulate(state, 1)[..., 1, :]

    report = evaluate(predict, physical, physical, IDENTITY, simulator, Config())
    for name in ("rollout_mse", "pulse_rollout_mse", "pulse_response_mse"):
        assert max(report[name].values()) < 1e-25
    # Phase A's transition is deliberately not a valid oracle for this world.
    wrong = evaluate(
        lambda state: state @ Simulator().transition().T,
        physical,
        physical,
        IDENTITY,
        simulator,
        Config(),
    )
    assert wrong["pulse_response_mse"]["16"] > 1e-8


def test_frozen_nonlinear_training_and_finite_serialization(tmp_path):
    torch.set_num_threads(1)
    config = replace(Config(), training_steps=2, train_count=8, test_count=2, trajectory_steps=24)
    simulator = DuffingSimulator(alpha=0.5, substeps=8)
    data, train, splits = physical_splits(simulator, 11, config)
    assert set(data.train_ids.tolist()).isdisjoint(data.test_ids.tolist())
    torch.testing.assert_close(splits["amplitude_shift"][:, 0], 1.5 * splits["interpolation"][:, 0])
    model = make_model(11, "opf", IDENTITY, IDENTITY, config)
    trace = train_frozen(model, IDENTITY, train, splits, simulator, 11, config)
    assert trace[-1]["encoder_drift"] == trace[-1]["target_encoder_drift"] == 0
    assert trace[-1]["audit"]["round_trip"]["passed"]
    path = tmp_path / "results.json"
    write_json(path, trace)
    decoded = json.loads(path.read_text())
    assert decoded[-1]["metrics"] == trace[-1]["metrics"]
    assert decoded[-1]["state_dict"] == trace[-1]["state_dict"]


def test_extra_transform_matches_initial_function_and_parameter_count():
    config = Config()
    control = output_transform_control(11, config)
    opf = make_model(11, "opf", IDENTITY, IDENTITY, config)
    assert parameter_count(control) == parameter_count(opf) == 458
    states = DuffingSimulator(alpha=0.5).trajectories(4, 4, 11)
    torch.testing.assert_close(
        PhysicalPredictor(control, IDENTITY, IDENTITY)(states),
        PhysicalPredictor(opf, IDENTITY, IDENTITY)(states),
        atol=1e-14,
        rtol=1e-14,
    )


@pytest.mark.parametrize("rotated", [False, True])
def test_folding_preserves_physical_function_with_nonorthogonal_basis(rotated):
    frame = orthogonal_draws(2718, 1)[0] if rotated else IDENTITY
    model = make_model(11, "opf", frame, IDENTITY, Config())
    with torch.no_grad():
        model.opf.raw_basis.mul_(1.15)
        model.opf.raw_basis[0, 0, 2] += 0.03
    folded = fold_opf_predictor(model, frame)
    states = DuffingSimulator(alpha=0.5).trajectories(4, 4, 11)
    torch.testing.assert_close(
        folded(states), PhysicalPredictor(model, frame, IDENTITY)(states), atol=1e-13, rtol=1e-13
    )
    assert parameter_count(folded) == 422
