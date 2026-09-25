"""Conventional delay prediction and a six-dimensional SVD delay state."""

import torch

from .models import ObservablePredictor, fit_affine
from .nonlinear import PolynomialPredictor
from .partial_data import HORIZONS, PULSE_ASSIMILATION, HistoryNormalizer, windows


class DelayRegression:
    def __init__(self, observations, history, family, dt=0.1):
        self.history, self.family = history, family
        batch = windows(observations, history)
        self.normalizer = HistoryNormalizer(batch["history"], dt)
        features = self.normalizer(batch["history"])
        self.center = features.mean(0)
        _, singular, vh = torch.linalg.svd(features - self.center, full_matrices=False)
        dimension = features.shape[-1] if family == "cubic_full" else min(6, features.shape[-1])
        self.basis = vh[:dimension].T
        self.scale = (singular[: self.basis.shape[-1]] / len(features) ** 0.5).clamp_min(1e-8)
        self.singular_values = singular.tolist()
        target = batch["future"][:, 0]
        if family == "linear_ar":
            weights, fit = fit_affine(features, target)
            self.predictor = ObservablePredictor(weights)
            self.fit = {**fit, "weights": weights.tolist(), "parameter_count": weights.numel()}
        else:
            z = self.encode(batch["history"])
            if family == "svd_cubic_state":
                target = self.encode(batch["next_history"])
                readout, _ = fit_affine(z, batch["history"][:, -1])
                self.readout = ObservablePredictor(readout)
            self.predictor = PolynomialPredictor(z, target)
            self.fit = self.predictor.fit_report

    def encode(self, history):
        return ((self.normalizer(history) - self.center) @ self.basis) / self.scale

    def forecast(self, history, steps):
        result = []
        if self.family == "svd_cubic_state":
            state = self.encode(history)
            for _ in range(steps):
                state = self.predictor(state)
                result.append(self.readout(state))
        else:
            for _ in range(steps):
                inputs = (
                    self.normalizer(history) if self.family == "linear_ar" else self.encode(history)
                )
                prediction = self.predictor(inputs)
                result.append(prediction)
                history = torch.cat((history[:, 1:], prediction[:, None]), 1)
        return torch.stack(result, 1)

    def report(self):
        return {
            "family": self.family,
            "history": self.history,
            "raw_history_dimension": self.normalizer.mean.numel(),
            "svd_dimension": self.basis.shape[-1],
            "normalizer_mean": self.normalizer.mean.tolist(),
            "normalizer_scale": self.normalizer.scale.tolist(),
            "history_singular_values": self.singular_values,
            "pca_center": self.center.tolist(),
            "pca_basis": self.basis.tolist(),
            "pca_scale": self.scale.tolist(),
            "fit": self.fit,
            "readout": self.readout.readout.tolist() if self.family == "svd_cubic_state" else None,
            "autonomous_state_dimension": self.basis.shape[-1]
            if self.family == "svd_cubic_state"
            else self.normalizer.mean.numel(),
        }


class RecentTwoState:
    """Ignore redundant older lags; maintain six position-delay coordinates."""

    def __init__(self, observations, dt=0.1):
        self.model = DelayRegression(observations, 2, "svd_cubic_state", dt)

    def forecast(self, history, steps):
        return self.model.forecast(history[:, -2:], steps)

    def encode(self, history):
        return self.model.encode(history[:, -2:])


def prediction_metrics(predicted, truth):
    """Retain unstable outcomes explicitly; never clip predictions or hide failures."""
    result = {}
    for horizon in HORIZONS:
        estimate = predicted[:, horizon - 1]
        error = (estimate - truth[:, horizon - 1]).square().mean()
        result[str(horizon)] = float(error) if torch.isfinite(error) else None
    return result


def finite_float(value):
    return float(value) if torch.isfinite(value) else None


@torch.no_grad()
def evaluate_observer(model, data, history):
    report = {}
    for split, observed in (("interpolation", data.test), ("amplitude_shift", data.shifted)):
        batch = windows(observed, history, max(HORIZONS))
        predicted = model.forecast(batch["history"], max(HORIZONS))
        pulse = data.pulse_observations[split]
        start, stop = PULSE_ASSIMILATION - history + 1, PULSE_ASSIMILATION + 1
        if start < 1:
            raise ValueError("Pulse forecasts require a history entirely after the unseen pulse")
        pulse_predictions = {
            name: model.forecast(values[:, start:stop], max(HORIZONS))
            for name, values in pulse.items()
        }
        response = pulse_predictions["pulse"] - pulse_predictions["base"]
        truth_response = pulse["pulse"][:, stop:] - pulse["base"][:, stop:]
        report[split] = {
            "rollout_mse": prediction_metrics(predicted, batch["future"]),
            "pulse_rollout_mse": prediction_metrics(
                pulse_predictions["pulse"], pulse["pulse"][:, stop:]
            ),
            "pulse_response_mse": prediction_metrics(response, truth_response),
            "max_predicted_observation_norm": finite_float(predicted.norm(dim=-1).max()),
            "max_true_observation_norm": float(batch["future"].norm(dim=-1).max()),
            "nonfinite_prediction_count": int((~torch.isfinite(predicted)).sum()),
        }
    return report
