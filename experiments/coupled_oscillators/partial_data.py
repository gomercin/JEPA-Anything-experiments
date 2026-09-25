"""Phase C observation firewall and strictly past-only history windows."""

from dataclasses import dataclass

import torch
from torch import nn

from .dataset import make_dataset
from .nonlinear import DuffingSimulator

OBSERVATIONS = {"FULL": (0, 1, 2, 3, 4, 5), "POSITIONS": (0, 1, 2), "LOCAL": (0, 1)}
FIELD_NAMES = ("q1", "q2", "q3", "p1", "p2", "p3")
HORIZONS = (1, 4, 8, 16)
ORIGIN_START = 15  # Same origins at every H; leaves eight older samples for H<=8.
EXTRA_HISTORY = 8
PULSE_TIME = 24
PULSE_ASSIMILATION = 8


@dataclass(frozen=True)
class PartialConfig:
    seeds: tuple[int, ...] = (11, 22, 33)
    alpha: float = 0.5
    substeps: int = 8
    dt: float = 0.1
    train_count: int = 48
    test_count: int = 16
    trajectory_steps: int = 64
    amplitude_multiplier: float = 1.5
    pulse_magnitude: float = 0.2
    histories: tuple[int, ...] = (1, 2, 4, 8)
    training_steps: int = 1000
    batch_size: int = 128
    learning_rate: float = 0.003
    d: int = 6
    k: int = 3
    r: int = 2
    hidden_dim: int = 32
    ema_momentum: float = 0.99
    gram_weight: float = 0.05
    activity_weight: float = 0.1
    min_std: float = 0.1
    rotation_seed: int = 2718


@dataclass
class ObservedData:
    """The only data object made available to learner-facing code."""

    train: torch.Tensor
    test: torch.Tensor
    shifted: torch.Tensor
    pulse_observations: dict
    train_ids: list[int]
    test_ids: list[int]
    fields: tuple[str, ...]


@dataclass
class EvaluationTruth:
    """Never passed to a main model, loss, normalizer or training function."""

    train: torch.Tensor
    test: torch.Tensor
    shifted: torch.Tensor


def observe(physical, regime):
    return physical[..., list(OBSERVATIONS[regime])].clone()


def observer_data(config, seed, regime):
    simulator = DuffingSimulator(alpha=config.alpha, substeps=config.substeps, dt=config.dt)
    data = make_dataset(
        simulator, seed, config.train_count, config.test_count, config.trajectory_steps
    )
    train, test = data.physical[data.train_ids], data.physical[data.test_ids]
    shifted = simulator.simulate(config.amplitude_multiplier * test[:, 0], config.trajectory_steps)
    pulses = {}
    for name, states in (("interpolation", test), ("amplitude_shift", shifted)):
        initial = states[:, PULSE_TIME].clone()
        pulsed = initial.clone()
        pulsed[:, 3] += config.pulse_magnitude
        pulses[name] = {
            label: observe(simulator.simulate(value, PULSE_ASSIMILATION + max(HORIZONS)), regime)
            for label, value in (("base", initial), ("pulse", pulsed))
        }
    observed = ObservedData(
        observe(train, regime),
        observe(test, regime),
        observe(shifted, regime),
        pulses,
        data.train_ids.tolist(),
        data.test_ids.tolist(),
        tuple(FIELD_NAMES[index] for index in OBSERVATIONS[regime]),
    )
    return observed, EvaluationTruth(train, test, shifted)


def windows(observations, history, future_steps=1, start=ORIGIN_START):
    """Targets start strictly after context; trajectory and time indices stay explicit."""
    if history < 1 or start < history - 1 or future_steps < 1:
        raise ValueError("Need a valid past history and strictly future targets")
    n, length, width = observations.shape
    times = torch.arange(start, length - future_steps)
    if not len(times):
        raise ValueError("No valid windows")
    history_indices = times[:, None] + torch.arange(-history + 1, 1)
    future_indices = times[:, None] + torch.arange(1, future_steps + 1)
    return {
        "history": observations[:, history_indices].reshape(-1, history, width),
        "next_history": observations[:, history_indices + 1].reshape(-1, history, width),
        "future": observations[:, future_indices].reshape(-1, future_steps, width),
        "trajectory": torch.arange(n).repeat_interleave(len(times)),
        "time": times.repeat(n),
    }


def history_coordinates(history, dt=0.1):
    """Invertible allowed-observation transform: latest values and adjacent slopes.

    Slopes are finite differences, not supplied simulator velocities. All lags
    are retained. This avoids asking an MLP to subtract almost identical inputs
    before it can use temporal information.
    """
    return torch.cat((history[..., -1, :], (history.diff(dim=-2) / dt).flatten(-2)), -1)


class HistoryNormalizer(nn.Module):
    def __init__(self, train_history, dt=0.1):
        super().__init__()
        self.dt = dt
        values = history_coordinates(train_history, dt)
        self.register_buffer("mean", values.mean(0))
        self.register_buffer("scale", values.std(0, correction=0).clamp_min(1e-8))

    def forward(self, history):
        return (history_coordinates(history, self.dt) - self.mean) / self.scale


def linear_observability(regime, history, dt=0.1):
    simulator = DuffingSimulator(alpha=0, dt=dt)
    transition = simulator.transition()
    observation = torch.eye(6, dtype=torch.float64)[list(OBSERVATIONS[regime])]
    matrix = torch.cat(
        [observation @ torch.linalg.matrix_power(transition, t) for t in range(history)]
    )
    singular = torch.linalg.svdvals(matrix)
    rank = int(torch.linalg.matrix_rank(matrix))
    return {
        "rank": rank,
        "singular_values": singular.tolist(),
        "full_state_condition_number": float(singular[0] / singular[-1]) if rank == 6 else None,
        "interpretation": "Linear calibration only; rank does not certify nonlinear identification",
    }
