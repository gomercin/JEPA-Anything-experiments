import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
from jepa_anything_core.audit import audit_opf_geometry
from jepa_anything_core.opf import validate_factorization

from experiments.coupled_oscillators.dataset import make_dataset, observation_matrix, pairs
from experiments.coupled_oscillators.evaluate import evaluate, subspace_alignment
from experiments.coupled_oscillators.models import build_models
from experiments.coupled_oscillators.run_experiment import Config
from experiments.coupled_oscillators.simulator import Simulator
from experiments.coupled_oscillators.train import training_loss


def test_simulator_deterministic_stable_and_exact_flow():
    simulator = Simulator()
    first = simulator.trajectories(8, 32, seed=11)
    assert torch.equal(first, simulator.trajectories(8, 32, seed=11))
    assert not torch.equal(first, simulator.trajectories(8, 32, seed=22))
    assert first.shape == (8, 33, 6)
    assert torch.linalg.eigvals(simulator.transition()).abs().max() < 1
    expected = first[:, 0] @ torch.matrix_exp(32 * simulator.dt * simulator.generator()).T
    torch.testing.assert_close(first[:, -1], expected, atol=1e-12, rtol=1e-12)


def test_trajectory_split_and_pairs():
    dataset = make_dataset(Simulator(), 11, 8, 4, 32)
    assert set(dataset.train_ids.tolist()).isdisjoint(dataset.test_ids.tolist())
    assert sorted(dataset.train_ids.tolist() + dataset.test_ids.tolist()) == list(range(12))
    train, test = dataset.observed(observation_matrix("RAW"))
    context, target = pairs(train)
    assert context.shape == target.shape == (8 * 32, 6)
    assert test.shape == (4, 33, 6)
    torch.testing.assert_close(context.reshape(8, 32, 6), dataset.physical[dataset.train_ids, :-1])
    torch.testing.assert_close(target.reshape(8, 32, 6), dataset.physical[dataset.train_ids, 1:])


def test_mixed_preserves_information_and_mse():
    matrix = observation_matrix("MIXED")
    assert torch.equal(matrix, observation_matrix("MIXED"))
    assert torch.linalg.matrix_rank(matrix) == 6
    torch.testing.assert_close(matrix.T @ matrix, torch.eye(6, dtype=torch.float64))
    physical = Simulator().trajectories(4, 16, 11)
    mixed = physical @ matrix.T
    torch.testing.assert_close(mixed @ torch.linalg.inv(matrix).T, physical)
    torch.testing.assert_close(mixed.square().mean(), physical.square().mean())
    with pytest.raises(ValueError, match="PARTIAL"):
        observation_matrix("PARTIAL")


def test_shapes_factorization_capacity_and_opf_backward():
    torch.set_num_threads(1)
    config = Config()
    assert config.k * config.r == config.d
    with pytest.raises(ValueError):
        validate_factorization(6, 3, 4)
    models, match = build_models(11, config)
    assert match["matched"] and match["absolute_difference"] == 0
    context, target = pairs(Simulator().trajectories(4, 8, 11))
    for model in models.values():
        output = model(context, target)
        assert output.prediction.shape == output.context_state.shape == (32, 6)
        assert not output.target.requires_grad
    torch.testing.assert_close(
        models["standard_jepa"](context, target).prediction,
        models["unconstrained_multihead"](context, target).prediction,
    )
    model = models["opf"]
    assert model.opf(model.target_encoder(target)).shape == (32, 3, 2)
    loss, _ = training_loss(model, context, target, config)
    loss.backward()
    assert model.opf.raw_basis.grad is not None
    assert model.opf.raw_basis.grad.abs().sum() > 0
    assert model.context_encoder.weight.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())
    torch.optim.Adam(model.parameters(), lr=config.learning_rate).step()
    model.update_target_encoder()
    audit = audit_opf_geometry(model.opf, model.context_encoder(context)).to_dict()
    assert audit["basis"]["rank"] == 6
    assert audit["round_trip"]["passed"]
    assert "max_cross_factor_correlation" in audit["factors"]
    # A learned soft-Gram map is not required to pass exact orthogonality.


@pytest.mark.parametrize("condition", ["RAW", "MIXED"])
def test_oracle_rollout_and_declared_pulse(condition):
    config = Config()
    simulator = Simulator()
    physical = simulator.trajectories(4, 32, 11)
    matrix = observation_matrix(condition)
    observed_transition = matrix @ simulator.transition() @ matrix.T
    report = evaluate(
        lambda state: state @ observed_transition.T,
        physical @ matrix.T,
        physical,
        matrix,
        simulator,
        config,
    )
    assert report["one_step_mse"] < 1e-25
    for family in ("rollout_mse", "pulse_rollout_mse", "pulse_response_mse"):
        assert set(report[family]) == {"1", "4", "8", "16"}
        assert max(report[family].values()) < 1e-25
    # A predictor ignoring the changed dynamics must not receive oracle scores.
    persistence = evaluate(
        lambda state: state, physical @ matrix.T, physical, matrix, simulator, config
    )
    assert persistence["pulse_response_mse"]["16"] > 1e-5


def test_subspace_alignment_ignores_sign_rotation_and_permutation():
    modes = Simulator().modes()[1]
    rotation = torch.tensor([[0.6, -0.8], [0.8, 0.6]], dtype=torch.float64)
    rows = torch.stack([(modes[index] @ rotation).T for index in (2, 0, 1)])
    report = subspace_alignment(rows, modes)
    assert report["best_mode_permutation"] == [2, 0, 1]
    assert report["best_permutation_mean_overlap"] == pytest.approx(1.0)
    rows[0] = 0
    deficient = subspace_alignment(rows, modes)
    assert deficient["factor_ranks"][0] == 0
    assert deficient["principal_angles_degrees_matrix"][0] == [None, None, None]


def test_quick_cli_json_and_finite_metrics(tmp_path):
    root = Path(__file__).resolve().parents[3]
    output = tmp_path / "quick"
    command = [
        sys.executable,
        str(root / "experiments/coupled_oscillators/run_experiment.py"),
        "--quick",
        "--output-dir",
        str(output),
    ]
    completed = subprocess.run(
        command, cwd=root, text=True, capture_output=True, timeout=180, check=False
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads((output / "results.json").read_text())
    assert report["status"] == "complete"
    assert len(report["records"]) == 24
    assert len(report["summary"]) == 8
    assert (output / "protocol.json").is_file()
    assert (output / "summary.md").is_file()
    assert (output / "results.sha256").is_file()
    for record in report["records"]:
        metrics = record["metrics"]
        assert metrics["one_step_mse"] == metrics["rollout_mse"]["1"]
        for family in ("rollout_mse", "pulse_rollout_mse", "pulse_response_mse"):
            assert all(
                torch.isfinite(torch.tensor(value)) and value >= 0
                for value in metrics[family].values()
            )
        if record["model"] == "opf":
            assert "opf_geometry" in record["diagnostics"]
    # Refuse to overwrite a completed run, including its saved protocol.
    repeated = subprocess.run(
        command, cwd=root, text=True, capture_output=True, timeout=10, check=False
    )
    assert repeated.returncode != 0
    assert json.loads((output / "results.json").read_text()) == report
