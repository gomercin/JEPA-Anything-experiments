import json

import pytest
import torch

from experiments.coupled_oscillators.dataset import make_dataset, observation_matrix
from experiments.coupled_oscillators.frozen_encoder import (
    FrozenIsometry,
    PhysicalPredictor,
    latent_equivalence,
    make_model,
    paired_differences,
    train,
)
from experiments.coupled_oscillators.interrogate import orthogonal_draws
from experiments.coupled_oscillators.run_experiment import Config
from experiments.coupled_oscillators.simulator import Simulator


def test_frozen_isometry_and_raw_mixed_equivalence():
    data = Simulator().trajectories(4, 32, 11)
    frame = orthogonal_draws(2718, 1)[0]
    observation = observation_matrix("MIXED")
    report = latent_equivalence(data, frame, observation)
    assert report["passed"]
    torch.testing.assert_close(frame.T @ frame, torch.eye(6, dtype=torch.float64))
    assert not list(FrozenIsometry(frame).parameters())
    assert not latent_equivalence(data, 2 * frame, observation)["passed"]


@pytest.mark.parametrize("family", ["standard", "opf", "opf_fixed"])
def test_initial_physical_functions_match_across_frames_and_sensors(family):
    config = Config()
    identity = torch.eye(6, dtype=torch.float64)
    rotation = orthogonal_draws(2718, 1)[0]
    observation = observation_matrix("MIXED")
    data = Simulator().trajectories(4, 16, 11)
    native = make_model(11, family, identity, identity, config)
    rotated = make_model(11, family, rotation, observation, config)
    expected = PhysicalPredictor(native, identity, identity)(data)
    actual = PhysicalPredictor(rotated, rotation, observation)(data)
    torch.testing.assert_close(expected, actual, atol=1e-14, rtol=1e-14)
    assert all(not p.requires_grad for p in rotated.target_encoder.parameters())
    assert rotated.target_encoder is not rotated.context_encoder


@pytest.mark.parametrize("family", ["standard", "opf", "opf_fixed"])
def test_sgd_paired_equivariance_frozen_buffers_and_json(family):
    torch.set_num_threads(1)
    config = Config(training_steps=3)
    dataset = make_dataset(Simulator(), 11, 4, 2, 32)
    identity = torch.eye(6, dtype=torch.float64)
    rotation = orthogonal_draws(2718, 1)[0]
    traces = []
    for frame in (identity, rotation):
        model = make_model(11, family, frame, identity, config)
        traces.append(
            train(model, dataset, 11, "SGD", frame, identity, config, checkpoints=(0, 1, 3))
        )
        assert torch.equal(model.context_encoder.weight, frame)
        assert torch.equal(model.target_encoder.weight, frame)
    paired = paired_differences(*traces)
    assert max(paired[-1]["physical_prediction_max_abs_by_horizon"]) < 1e-13
    # Strict serialization rejects NaN/Infinity; no performance-order assertion.
    decoded = json.loads(json.dumps(traces, allow_nan=False))
    assert decoded[0][-1]["metrics"] == traces[0][-1]["metrics"]
    assert decoded[1][-1]["state_dict"] == traces[1][-1]["state_dict"]


def test_oracle_inverse_in_rotated_frame():
    simulator = Simulator()
    rotation = orthogonal_draws(2718, 1)[0]
    physical = simulator.trajectories(4, 16, 11)
    latent = physical @ rotation.T
    latent_transition = rotation @ simulator.transition() @ rotation.T
    decoded_next = latent[:, :-1] @ latent_transition.T @ rotation
    torch.testing.assert_close(decoded_next, physical[:, 1:], atol=1e-14, rtol=1e-14)
