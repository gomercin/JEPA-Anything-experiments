"""Local Duffing extension and ordinary polynomial transition identification."""

from dataclasses import dataclass
from itertools import combinations_with_replacement

import torch

from .simulator import Simulator


@dataclass(frozen=True)
class DuffingSimulator(Simulator):
    alpha: float = 0.0
    substeps: int = 4

    def __post_init__(self):
        if self.alpha < 0 or self.substeps < 1 or self.dt <= 0:
            raise ValueError("Use nonnegative cubic stiffness and positive timestep/substeps")

    def transition(self):
        if self.alpha != 0:
            raise ValueError("Nonlinear flow has no state-independent transition; use simulate")
        return super().transition()

    def simulate(self, initial, steps):
        """RK4 inside each observation interval; alpha=0 uses Phase A exactly."""
        if self.alpha == 0:
            return super().simulate(initial, steps)
        stiffness = self.stiffness()
        dt = self.dt / self.substeps

        def derivative(state):
            q, p = state[..., :3], state[..., 3:]
            return torch.cat((p, -q @ stiffness.T - self.damping * p - self.alpha * q**3), -1)

        state = initial
        states = [state]
        for _ in range(steps):
            for _ in range(self.substeps):
                k1 = derivative(state)
                k2 = derivative(state + dt * k1 / 2)
                k3 = derivative(state + dt * k2 / 2)
                k4 = derivative(state + dt * k3)
                state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            states.append(state)
        return torch.stack(states, dim=-2)

    def energy(self, states):
        q, p = states[..., :3], states[..., 3:]
        return (p.square().sum(-1) + (q * (q @ self.stiffness())).sum(-1)) / 2 + (
            self.alpha * q.pow(4).sum(-1) / 4
        )


def polynomial_features(values, degree=3):
    """All monomials through degree, including constant; no simulator parameters."""
    columns = [torch.ones_like(values[..., 0])]
    for order in range(1, degree + 1):
        for indices in combinations_with_replacement(range(values.shape[-1]), order):
            columns.append(values[..., list(indices)].prod(-1))
    return torch.stack(columns, dim=-1)


class PolynomialPredictor:
    """Train-only column scaling and double-precision SVD least squares.

    Fits the sampled state transition, not the known differential equation.
    A cubic vector field has a flow with higher-order terms: degree three is
    physically motivated but is not an exact oracle for the discrete transition.
    """

    def __init__(self, inputs, targets, degree=3):
        self.degree = degree
        features = polynomial_features(inputs, degree)
        self.scale = features.square().mean(0).sqrt().clamp_min(1e-12)
        fit = torch.linalg.lstsq(features / self.scale, targets, driver="gelsd")
        self.weights = fit.solution
        self.fit_report = {
            "degree": degree,
            "feature_count": features.shape[-1],
            "parameter_count": self.weights.numel(),
            "design_rank": int(fit.rank),
            "scaled_design_singular_values": fit.singular_values.tolist(),
            "scaled_design_condition": float(fit.singular_values[0] / fit.singular_values[-1]),
            "feature_scale": self.scale.tolist(),
            "weights": self.weights.tolist(),
            "train_mse": float((self(inputs) - targets).square().mean()),
        }

    def __call__(self, values):
        return (polynomial_features(values, self.degree) / self.scale) @ self.weights
