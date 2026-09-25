"""Three damped oscillators; exact sampled linear flow, in float64 on CPU."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Simulator:
    dt: float = 0.1
    coupling: float = 0.2
    damping: float = 0.03
    onsite_stiffness: tuple[float, ...] = (1.0, 1.4, 2.0)
    initial_std: float = 0.7

    def stiffness(self):
        laplacian = torch.tensor([[1, -1, 0], [-1, 2, -1], [0, -1, 1]], dtype=torch.float64)
        return torch.diag(torch.tensor(self.onsite_stiffness, dtype=torch.float64)) + (
            self.coupling * laplacian
        )

    def generator(self):
        matrix = torch.zeros(6, 6, dtype=torch.float64)
        matrix[:3, 3:] = torch.eye(3, dtype=torch.float64)
        matrix[3:, :3] = -self.stiffness()
        matrix[3:, 3:] = -self.damping * torch.eye(3, dtype=torch.float64)
        return matrix

    def transition(self):
        return torch.matrix_exp(self.dt * self.generator())

    def simulate(self, initial, steps):
        """Return (..., steps + 1, 6), including the supplied initial state."""
        states = [initial]
        transition = self.transition()
        for _ in range(steps):
            states.append(states[-1] @ transition.T)
        return torch.stack(states, dim=-2)

    def trajectories(self, count, steps, seed):
        rng = torch.Generator().manual_seed(seed)
        initial = self.initial_std * torch.randn(count, 6, generator=rng, dtype=torch.float64)
        return self.simulate(initial, steps)

    def modes(self):
        """Orthonormal columns spanning each (modal position, modal velocity) plane."""
        eigenvalues, vectors = torch.linalg.eigh(self.stiffness())
        planes = torch.zeros(3, 6, 2, dtype=torch.float64)
        for index in range(3):
            planes[index, :3, 0] = vectors[:, index]
            planes[index, 3:, 1] = vectors[:, index]
        return eigenvalues, planes
