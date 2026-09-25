"""Observation-only local polynomial states; chronological filtering contract."""

import torch

from .nonlinear import PolynomialPredictor
from .partial_data import windows


def polynomial_window_matrix(length, degree, dt, location=None):
    """Rows map an observed window to polynomial coefficients at one location."""
    if length <= degree or dt <= 0:
        raise ValueError("Need more samples than polynomial degree and positive dt")
    if location is None:
        location = length - 1
    times = (torch.arange(length, dtype=torch.float64) - location) * dt
    design = torch.stack([times**power for power in range(degree + 1)], -1)
    return torch.linalg.pinv(design)


class CausalPolynomialState:
    """Estimate current q and its slope from past-only positions; fit cubic recurrence.

    This is local polynomial endpoint filtering, not an offline smoother.
    Both transition targets and inputs are derived from noisy observations.
    """

    def __init__(self, observations, history, dt=0.1):
        self.history = history
        self.degree = min(3, history - 1)
        self.matrix = polynomial_window_matrix(history, self.degree, dt)
        batch = windows(observations, history)
        self.predictor = PolynomialPredictor(
            self.encode(batch["history"]), self.encode(batch["next_history"])
        )

    def encode(self, history):
        coefficients = torch.einsum("kh,...hm->...km", self.matrix, history)
        return coefficients[..., :2, :].flatten(-2)

    def forecast(self, history, steps):
        state = self.encode(history)
        result = []
        for _ in range(steps):
            state = self.predictor(state)
            result.append(state[..., :3])
        return torch.stack(result, 1)

    def report(self):
        return {
            "oracle": False,
            "information": "last H noisy samples only",
            "history": self.history,
            "degree": self.degree,
            "filter_matrix": self.matrix.tolist(),
            "fit": self.predictor.fit_report,
        }
