"""Seeded initial-condition splits and information-preserving observations."""

from dataclasses import dataclass

import torch


def observation_matrix(condition, seed=1729):
    if condition == "RAW":
        return torch.eye(6, dtype=torch.float64)
    if condition != "MIXED":
        raise ValueError("Only RAW and MIXED are implemented; PARTIAL is a different task")
    rng = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(6, 6, generator=rng, dtype=torch.float64))
    return q * torch.where(r.diag() < 0, -1.0, 1.0)


@dataclass
class Trajectories:
    physical: torch.Tensor
    train_ids: torch.Tensor
    test_ids: torch.Tensor

    def observed(self, matrix):
        observations = self.physical @ matrix.T
        return observations[self.train_ids], observations[self.test_ids]


def make_dataset(simulator, seed, train_count, test_count, steps):
    physical = simulator.trajectories(train_count + test_count, steps, seed)
    # Split whole trajectory IDs before constructing any time-step pairs.
    rng = torch.Generator().manual_seed(seed + 10000)
    order = torch.randperm(train_count + test_count, generator=rng)
    return Trajectories(physical, order[:train_count], order[train_count:])


def pairs(trajectories):
    return trajectories[:, :-1].reshape(-1, 6), trajectories[:, 1:].reshape(-1, 6)
