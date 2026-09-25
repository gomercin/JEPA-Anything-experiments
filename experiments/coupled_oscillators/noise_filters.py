"""Small polynomial mechanical identifier and windowed EKF, with explicit oracles.

Learned identification sees only noisy position trajectories. The kinematic
prior q'=v is supplied; acceleration is a complete cubic in q plus linear v.
No oscillator stiffness, damping, cubic coefficient, or true noise enters fit.
"""

from itertools import combinations_with_replacement

import torch

from .noise_baselines import polynomial_window_matrix
from .simulator import Simulator


def position_exponents():
    result = [[0, 0, 0]]
    for degree in (1, 2, 3):
        for indices in combinations_with_replacement(range(3), degree):
            result.append([indices.count(i) for i in range(3)])
    return torch.tensor(result, dtype=torch.float64)


class CubicMechanics:
    """q'=v, v'=polynomial(q)+D v. RK4 and its analytic tangent map."""

    def __init__(self, weights, dt=0.1, substeps=4):
        self.weights, self.dt, self.substeps = weights, dt, substeps
        self.exponents = position_exponents()
        self.oracle = False

    def features(self, state):
        q, v = state[..., :3], state[..., 3:]
        monomials = q[..., None, :].pow(self.exponents).prod(-1)
        return torch.cat((monomials, v), -1)

    def rhs(self, state, jacobian=False):
        derivative = torch.cat((state[..., 3:], self.features(state) @ self.weights), -1)
        if not jacobian:
            return derivative
        matrix = torch.zeros(*state.shape[:-1], 6, 6, dtype=state.dtype)
        matrix[..., :3, 3:] = torch.eye(3, dtype=state.dtype)
        matrix[..., 3:, 3:] = self.weights[-3:].T
        for axis in range(3):
            exponents = self.exponents.clone()
            exponents[:, axis] = (exponents[:, axis] - 1).clamp_min(0)
            features = state[..., None, :3].pow(exponents).prod(-1) * self.exponents[:, axis]
            matrix[..., 3:, axis] = features @ self.weights[:20]
        return derivative, matrix

    def step(self, state, jacobian=False):
        dt = self.dt / self.substeps
        identity = torch.eye(6, dtype=state.dtype).expand(*state.shape[:-1], 6, 6)
        total = identity
        for _ in range(self.substeps):
            if jacobian:
                k1, j1 = self.rhs(state, True)
                k2, a2 = self.rhs(state + dt * k1 / 2, True)
                j2 = a2 @ (identity + dt * j1 / 2)
                k3, a3 = self.rhs(state + dt * k2 / 2, True)
                j3 = a3 @ (identity + dt * j2 / 2)
                k4, a4 = self.rhs(state + dt * k3, True)
                j4 = a4 @ (identity + dt * j3)
                total = (identity + dt * (j1 + 2 * j2 + 2 * j3 + j4) / 6) @ total
            else:
                k1 = self.rhs(state)
                k2 = self.rhs(state + dt * k1 / 2)
                k3 = self.rhs(state + dt * k2 / 2)
                k4 = self.rhs(state + dt * k3)
            state = state + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return (state, total) if jacobian else state

    def rollout(self, state, steps):
        result = []
        for _ in range(steps):
            state = self.step(state)
            result.append(state[..., :3])
        return torch.stack(result, 1)


def identify_mechanics(observations, dt=0.1):
    """Offline training-only smoothing; online evaluation never uses future data."""
    length, degree = 11, 5
    matrix = polynomial_window_matrix(length, degree, dt, location=length // 2)
    segments = observations.unfold(1, length, 1).transpose(-1, -2)
    coefficients = torch.einsum("kh,nthm->ntkm", matrix, segments)
    state = coefficients[..., :2, :].flatten(-2).reshape(-1, 6)
    acceleration = (2 * coefficients[..., 2, :]).reshape(-1, 3)
    temporary = CubicMechanics(torch.zeros(23, 3, dtype=observations.dtype), dt)
    features = temporary.features(state)
    scale = features.square().mean(0).sqrt().clamp_min(1e-12)
    solution = torch.linalg.lstsq(features / scale, acceleration, driver="gelsd")
    weights = solution.solution / scale[:, None]
    model = CubicMechanics(weights, dt)
    # White measurement noise contribution to third differences is 20 sigma^2.
    # Smooth signal contributes a small positive bias; no true sigma is supplied.
    variance = observations.diff(n=3, dim=1).square().mean((0, 1)) / 20
    model.noise_variance = variance.clamp_min(1e-12)
    model.fit_report = {
        "oracle": False,
        "kinematic_prior": "q'=v; cubic(q)+linear(v) acceleration",
        "training_processing": "centered degree-five polynomial, 11 noisy training samples",
        "test_information": "past/current noisy observations only",
        "features": model.exponents.tolist(),
        "weights": weights.tolist(),
        "parameter_count": weights.numel(),
        "derivative_fit_mse": float((features @ weights - acceleration).square().mean()),
        "design_singular_values": solution.singular_values.tolist(),
        "estimated_noise_variance": variance.tolist(),
        "dt": dt,
        "rk4_substeps": model.substeps,
        "smoothing_matrix": matrix.tolist(),
    }
    return model


def oracle_mechanics(alpha, sigma, dt=0.1):
    """Explicit upper-reference model: true equations and true measurement variance."""
    simulator = Simulator(dt=dt)
    weights = torch.zeros(23, 3, dtype=torch.float64)
    weights[1:4] = -simulator.stiffness().T
    weights[-3:] = -simulator.damping * torch.eye(3, dtype=torch.float64)
    exponents = position_exponents()
    for axis in range(3):
        index = int(torch.where(exponents[:, axis] == 3)[0][0])
        weights[index, axis] = -alpha
    model = CubicMechanics(weights, dt)
    model.oracle = True
    model.noise_variance = torch.full((3,), max(sigma**2, 1e-12), dtype=torch.float64)
    model.fit_report = {
        "oracle": True,
        "label": "ORACLE_EKF",
        "true_equations_and_noise": True,
        "weights": weights.tolist(),
        "noise_variance": model.noise_variance.tolist(),
        "dt": dt,
        "rk4_substeps": model.substeps,
    }
    return model


class WindowEKF:
    """Reset at each H-sample window; no privileged longer filtering history."""

    def __init__(self, dynamics):
        self.dynamics = dynamics
        self.noise_variance = dynamics.noise_variance
        self.oracle = dynamics.oracle

    def estimate(self, history):
        n = len(history)
        state = torch.cat((history[:, 0], torch.zeros(n, 3, dtype=history.dtype)), -1)
        # Same weak velocity prior for learned and oracle filters; no true states.
        covariance = torch.diag(
            torch.cat((self.noise_variance, torch.ones(3, dtype=history.dtype)))
        )
        covariance = covariance.expand(n, 6, 6).clone()
        noise = torch.diag(self.noise_variance)
        identity = torch.eye(6, dtype=history.dtype)
        for observation in history[:, 1:].unbind(1):
            state, transition = self.dynamics.step(state, True)
            covariance = transition @ covariance @ transition.transpose(-1, -2)
            innovation_cov = covariance[:, :3, :3] + noise
            gain = torch.linalg.solve(innovation_cov, covariance[:, :3, :]).transpose(-1, -2)
            state = state + torch.einsum("bij,bj->bi", gain, observation - state[:, :3])
            update = identity.expand(n, 6, 6).clone()
            update[..., :, :3] -= gain
            covariance = update @ covariance @ update.transpose(
                -1, -2
            ) + gain @ noise @ gain.transpose(-1, -2)
            covariance = (covariance + covariance.transpose(-1, -2)) / 2
        return state, covariance

    def encode(self, history):
        return self.estimate(history)[0]

    def forecast(self, history, steps):
        return self.dynamics.rollout(self.encode(history), steps)

    def forecast_distribution(self, history, steps):
        state, covariance = self.estimate(history)
        means, covariances = [], []
        for _ in range(steps):
            state, transition = self.dynamics.step(state, True)
            covariance = transition @ covariance @ transition.transpose(-1, -2)
            means.append(state[..., :3])
            covariances.append(covariance)
        return torch.stack(means, 1), torch.stack(covariances, 1)

    def report(self):
        return {
            **self.dynamics.fit_report,
            "estimator": "windowed extended Kalman filter",
            "initial_velocity_mean": 0.0,
            "initial_velocity_covariance": [1.0] * 3,
            "process_noise": 0.0,
            "parameter_uncertainty_in_covariance": False,
            "covariance_caveat": (
                "linearized measurement uncertainty, not calibrated epistemic uncertainty"
            ),
        }
