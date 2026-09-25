"""Phase D information boundaries and numerical mechanics, never model rankings."""

import json
from dataclasses import asdict, replace

import pytest
import torch

from experiments.coupled_oscillators.noise_baselines import CausalPolynomialState
from experiments.coupled_oscillators.noise_data import (
    NoiseConfig,
    add_noise,
    learner_config,
    noisy_data,
)
from experiments.coupled_oscillators.noise_filters import (
    CubicMechanics,
    WindowEKF,
    identify_mechanics,
    oracle_mechanics,
)
from experiments.coupled_oscillators.nonlinear import DuffingSimulator
from experiments.coupled_oscillators.partial_data import observer_data, windows
from experiments.coupled_oscillators.partial_models import HistoryJEPA, train_history_model
from experiments.coupled_oscillators.phase_d_neural import snapshot_noise
from experiments.coupled_oscillators.phase_d_refine import refine_mechanics


def small_config():
    return replace(NoiseConfig(), train_count=8, test_count=3, trajectory_steps=40)


def test_noise_reproduction_and_no_noise_calibration():
    config = small_config()
    first, truth = noisy_data(config, 11, 0.15)
    restored = NoiseConfig(**json.loads(json.dumps(asdict(config))))
    second, _ = noisy_data(restored, 11, 0.15)
    assert torch.equal(first.train, second.train)
    assert torch.equal(first.test, second.test)
    assert first.train.shape[-1] == 3 and first.fields == ("q1", "q2", "q3")
    assert not torch.equal(first.train, truth.observed.train)
    assert set(first.train_ids).isdisjoint(first.test_ids)
    zero, _ = noisy_data(config, 11, 0)
    old, _ = observer_data(config, 11, "POSITIONS")
    assert torch.equal(zero.train, old.train)
    assert torch.equal(zero.test, old.test)
    with pytest.raises(ValueError):
        add_noise(truth.physical.train, 0.1, 123)


def test_noisy_targets_and_learner_settings_exclude_truth():
    config = small_config()
    data, truth = noisy_data(config, 11, 0.15)
    batch = windows(data.train, 8)
    index, time = batch["trajectory"][0], batch["time"][0]
    assert torch.equal(batch["future"][0, 0], data.train[index, time + 1])
    assert not torch.equal(batch["future"][0, 0], truth.observed.train[index, time + 1])
    allowed = vars(learner_config(config))
    assert not {"alpha", "pulse_magnitude", "noise_ratios", "substeps", "sigma"} & allowed.keys()
    assert not {"physical", "clean", "sigma", "noise"} & vars(data).keys()


def test_endpoint_filter_uses_no_future_and_pulse_is_unknown():
    config = small_config()
    data, truth = noisy_data(config, 11, 0.15)
    model = CausalPolynomialState(data.train, 8)
    history = windows(data.test, 8)["history"][0:1]
    changed = data.test.clone()
    changed[:, 16:] += 100000
    torch.testing.assert_close(
        model.encode(history), model.encode(windows(changed, 8)["history"][0:1])
    )
    for branches in truth.observed.pulse_observations.values():
        assert torch.equal(branches["base"][:, 0], branches["pulse"][:, 0])
    assert not any("action" in name for name in vars(data))


def test_filter_tangent_covariance_and_oracle_labels():
    torch.set_num_threads(1)
    data, _ = noisy_data(small_config(), 11, 0.15)
    dynamics = identify_mechanics(data.train)
    assert dynamics.fit_report["oracle"] is False
    state = torch.tensor([0.2, -0.4, 0.1, 0.3, 0.2, -0.3], dtype=torch.float64)
    _, analytic = dynamics.step(state, True)
    automatic = torch.autograd.functional.jacobian(dynamics.step, state)
    torch.testing.assert_close(analytic, automatic, atol=1e-12, rtol=1e-12)
    model = WindowEKF(dynamics)
    history = windows(data.test, 8)["history"][:4]
    _, covariance = model.estimate(history)
    assert torch.linalg.eigvalsh(covariance).min() > 0
    prediction, uncertainty = model.forecast_distribution(history, 4)
    torch.testing.assert_close(prediction, model.forecast(history, 4))
    assert torch.isfinite(uncertainty).all()
    oracle = oracle_mechanics(0.5, 0.1)
    assert oracle.fit_report["oracle"] is True
    expected = DuffingSimulator(alpha=0.5, substeps=8).simulate(state[None], 1)[:, -1]
    torch.testing.assert_close(oracle.step(state[None]), expected, atol=1e-8, rtol=1e-8)
    # The future never reaches this online estimator.
    altered = data.test.clone()
    altered[:, 16:] += 1e5
    torch.testing.assert_close(
        model.encode(windows(data.test, 8)["history"][:1]),
        model.encode(windows(altered, 8)["history"][:1]),
    )


def test_output_error_optimizer_handles_noncontiguous_fit_and_reconstruction():
    torch.set_num_threads(1)
    config = replace(NoiseConfig(), train_count=3, test_count=2)
    data, _ = noisy_data(config, 11, 0.15)
    initial = identify_mechanics(data.train)
    initial.weights = initial.weights.T.contiguous().T
    assert not initial.weights.is_contiguous()
    fitted = refine_mechanics(data.train, initial, iterations=1)
    assert fitted.fit_report["oracle"] is False
    saved = json.loads(json.dumps(fitted.fit_report, allow_nan=False))
    restored = CubicMechanics(torch.tensor(saved["weights"], dtype=torch.float64), saved["dt"])
    state = torch.tensor([[0.2, 0.3, -0.2, 0.1, 0.4, -0.1]], dtype=torch.float64)
    torch.testing.assert_close(restored.rollout(state, 4), fitted.rollout(state, 4))


def test_short_noisy_opf_training_and_strict_result_json(tmp_path):
    torch.set_num_threads(1)
    config = replace(small_config(), training_steps=2)
    data, truth = noisy_data(config, 11, 0.15)
    settings = learner_config(config)
    model = HistoryJEPA(data.train, 8, "opf", 11, torch.eye(6, dtype=torch.float64), settings)
    trace = train_history_model(
        model,
        data.train,
        11,
        settings,
        lambda current, step: snapshot_noise(current, data, truth, step),
    )
    assert trace[-1]["step"] == 2
    path = tmp_path / "result.json"
    path.write_text(json.dumps(trace, allow_nan=False))
    saved = json.loads(path.read_text())
    assert "noisy_target_mse" in saved[-1]["metrics"]["interpolation"]
    assert "clean_truth_mse" in saved[-1]["metrics"]["interpolation"]
    assert all(p.grad is None for p in model.jepa.target_encoder.parameters())
