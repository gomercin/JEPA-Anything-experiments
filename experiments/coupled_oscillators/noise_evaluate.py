"""Only this evaluator consumes clean labels; models receive noisy histories."""

import torch

from .models import ObservablePredictor, fit_affine
from .partial_baselines import finite_float, prediction_metrics
from .partial_data import HORIZONS, PULSE_ASSIMILATION, windows


@torch.no_grad()
def evaluate_noise(model, noisy, truth, history, oracle_context=False):
    report = {"oracle_clean_history_diagnostic": oracle_context}
    for split, observations, clean in (
        ("interpolation", noisy.test, truth.observed.test),
        ("amplitude_shift", noisy.shifted, truth.observed.shifted),
    ):
        sensor = windows(observations, history, max(HORIZONS))
        target = windows(clean, history, max(HORIZONS))
        context = target["history"] if oracle_context else sensor["history"]
        prediction = model.forecast(context, max(HORIZONS))
        start, stop = PULSE_ASSIMILATION - history + 1, PULSE_ASSIMILATION + 1
        if start < 1:
            raise ValueError("Pulse history must be strictly post-pulse")
        pulse_source = truth.observed if oracle_context else noisy
        pulse_prediction = {
            branch: model.forecast(values[:, start:stop], max(HORIZONS))
            for branch, values in pulse_source.pulse_observations[split].items()
        }
        record = {
            "noisy_target_mse": prediction_metrics(prediction, sensor["future"]),
            "clean_truth_mse": prediction_metrics(prediction, target["future"]),
            "pulse_noisy_mse": prediction_metrics(
                pulse_prediction["pulse"], noisy.pulse_observations[split]["pulse"][:, stop:]
            ),
            "pulse_clean_mse": prediction_metrics(
                pulse_prediction["pulse"],
                truth.observed.pulse_observations[split]["pulse"][:, stop:],
            ),
            "pulse_response_clean_mse": prediction_metrics(
                pulse_prediction["pulse"] - pulse_prediction["base"],
                truth.observed.pulse_observations[split]["pulse"][:, stop:]
                - truth.observed.pulse_observations[split]["base"][:, stop:],
            ),
            "max_prediction_norm": finite_float(prediction.norm(dim=-1).max()),
            "nonfinite_prediction_count": int((~torch.isfinite(prediction)).sum()),
        }
        if hasattr(model, "forecast_distribution"):
            mean, covariance = model.forecast_distribution(context, max(HORIZONS))
            variance = covariance[..., :3, :3].diagonal(dim1=-2, dim2=-1).clamp_min(1e-15)
            record["uncertainty"] = {
                str(h): {
                    "clean_95pct_marginal_coverage": float(
                        (
                            (mean[:, h - 1] - target["future"][:, h - 1]).abs()
                            <= 1.96 * variance[:, h - 1].sqrt()
                        )
                        .double()
                        .mean()
                    ),
                    "noisy_95pct_marginal_coverage": float(
                        (
                            (mean[:, h - 1] - sensor["future"][:, h - 1]).abs()
                            <= 1.96 * (variance[:, h - 1] + model.noise_variance).sqrt()
                        )
                        .double()
                        .mean()
                    ),
                    "mean_position_variance": float(variance[:, h - 1].mean()),
                }
                for h in HORIZONS
            }
        report[split] = record
    return report


@torch.no_grad()
def state_probe(encode, noisy, truth, history):
    """Post-fit evaluator-only affine probe; never a main training target."""
    train = windows(noisy.train, history, max(HORIZONS))
    labels = truth.physical.train[train["trajectory"], train["time"]]
    weights, fit = fit_affine(encode(train["history"]), labels)
    probe = ObservablePredictor(weights)
    report = {"evaluator_only": True, "fit": fit}
    for name, observations, physical in (
        ("interpolation", noisy.test, truth.physical.test),
        ("amplitude_shift", noisy.shifted, truth.physical.shifted),
    ):
        batch = windows(observations, history, max(HORIZONS))
        prediction = probe(encode(batch["history"]))
        labels = physical[batch["trajectory"], batch["time"]]
        error = (prediction - labels).square().mean(0)
        report[name] = {
            "position_mse": float(error[:3].mean()),
            "velocity_mse": float(error[3:].mean()),
            "full_state_mse": float(error.mean()),
        }
    return report


@torch.no_grad()
def filter_state_diagnostics(model, noisy, truth, history):
    report = {"evaluator_only": True, "oracle_true_initial_state": {}}
    for name, observations, physical in (
        ("interpolation", noisy.test, truth.physical.test),
        ("amplitude_shift", noisy.shifted, truth.physical.shifted),
    ):
        batch = windows(observations, history, max(HORIZONS))
        target = physical[batch["trajectory"], batch["time"]]
        state, covariance = model.estimate(batch["history"])
        error = (state - target).square().mean(0)
        report[name] = {
            "position_mse": float(error[:3].mean()),
            "velocity_mse": float(error[3:].mean()),
            "minimum_covariance_eigenvalue": float(torch.linalg.eigvalsh(covariance).min()),
        }
        prediction = model.dynamics.rollout(target, max(HORIZONS))
        future = windows(physical[..., :3], history, max(HORIZONS))["future"]
        report["oracle_true_initial_state"][name] = prediction_metrics(prediction, future)
    return report
