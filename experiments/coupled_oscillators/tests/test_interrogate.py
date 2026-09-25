import copy

import pytest
import torch
from jepa_anything_core.baselines import parameter_count

from experiments.coupled_oscillators.dataset import observation_matrix, pairs
from experiments.coupled_oscillators.evaluate import subspace_alignment
from experiments.coupled_oscillators.interrogate import (
    equivariance_probe,
    fit_with_trace,
    make_variant,
    null_report,
    orthogonal_draws,
    overlap_scores,
    physical_rows,
    snapshot,
)
from experiments.coupled_oscillators.run_experiment import Config
from experiments.coupled_oscillators.simulator import Simulator
from experiments.coupled_oscillators.train import train_neural


def test_mixed_pullback_recovers_same_physical_analysis():
    raw, _ = make_variant(11, "opf", Config())
    mixed = copy.deepcopy(raw)
    matrix = observation_matrix("MIXED")
    with torch.no_grad():
        mixed.target_encoder.weight.copy_(mixed.target_encoder.weight @ matrix.T)
    torch.testing.assert_close(
        physical_rows(raw, torch.eye(6, dtype=torch.float64)), physical_rows(mixed, matrix)
    )


def test_null_determinism_and_correct_permutation_statistic():
    draws = orthogonal_draws(314159, 16)
    assert torch.equal(draws, orthogonal_draws(314159, 16))
    torch.testing.assert_close(
        draws @ draws.transpose(-1, -2), torch.eye(6, dtype=torch.float64).expand(16, 6, 6)
    )
    modes = Simulator().modes()[1]
    rows = draws[0].reshape(3, 2, 6)
    scalar = subspace_alignment(rows, modes)["best_permutation_mean_overlap"]
    assert float(overlap_scores(rows, modes)) == pytest.approx(scalar)
    report = null_report(rows, modes, draws)
    assert report == null_report(rows, modes, draws)
    assert len(report["scores"]) == 16
    assert all(0 <= s <= 1 for s in report["scores"])
    with pytest.raises(ValueError, match="full-rank"):
        overlap_scores(torch.zeros_like(rows), modes)


def test_extra_transform_matches_count_and_initial_function():
    config = Config()
    ordinary, _ = make_variant(11, "standard", config)
    extra, _ = make_variant(11, "output_transform", config)
    opf, _ = make_variant(11, "opf", config)
    assert parameter_count(extra) == parameter_count(opf) == 500
    context, target = pairs(Simulator().trajectories(4, 8, 11))
    torch.testing.assert_close(
        ordinary(context, target).prediction, extra(context, target).prediction
    )


def test_trace_does_not_change_training_and_error_decomposition():
    torch.set_num_threads(1)
    config = Config(training_steps=3)
    model, _ = make_variant(11, "opf", config)
    reference = copy.deepcopy(model)
    simulator = Simulator()
    data = simulator.trajectories(4, 32, 11)
    matrix = observation_matrix("RAW")
    train_neural(reference, data, 11, config)
    trace = fit_with_trace(
        model,
        data,
        11,
        config,
        lambda: snapshot(model, data, data, data, matrix, simulator, config),
        checkpoints=(0, 1, 3),
    )
    for key, value in reference.state_dict().items():
        torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
    assert sum(trace[-1]["observable_mse_by_encoder_singular_direction"]) == pytest.approx(
        trace[-1]["metrics"]["one_step_mse"], rel=1e-10
    )


def test_matched_initialization_and_sgd_rotation_equivariance():
    report = equivariance_probe(11, "opf", Config(), steps=3)
    assert report["Adam"][0]["prediction_max_difference"] < 1e-14
    assert report["SGD"][-1]["prediction_max_difference"] < 1e-14
    assert report["SGD"][-1]["physical_encoder_max_difference"] < 1e-14
