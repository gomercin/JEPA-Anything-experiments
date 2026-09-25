"""Periodic SH35 plus a finite-time diffusive mediator. No object tracking here."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Config:
    length: float = 128.0
    n: int = 512
    dt: float = 0.0125
    r: float = -0.67
    cubic: float = 2.0
    feedback: float = 0.05
    source: float = 1.0
    diffusion: float = 64.0
    tau: float = 10.0


class Field:
    """u_t=ru-(1+∂xx)^2u+bu^3-u^5+gmu; τm_t=Dm_xx-m+su².

    Fourier collocation, Cox-Matthews ETDRK4 with contour-evaluated coefficients.
    No clipping, recentering, pinning, labels, pair forces, or stochastic updates.
    Spatial aliasing is controlled by explicit resolution qualification, not by
    modifying the physical nonlinearity. Arrays are real float64 (2,n).
    """

    def __init__(self, config=None):
        config = Config() if config is None else config
        self.c = config
        if config.n % 2 or min(config.length, config.dt, config.tau) <= 0:
            raise ValueError("Positive time/length scales and even n required")
        self.x = np.arange(config.n) * config.length / config.n - config.length / 2
        self.k = 2 * np.pi * np.fft.rfftfreq(config.n, config.length / config.n)
        linear = np.array(
            [
                config.r - (1 - self.k**2) ** 2,
                (-1 - config.diffusion * self.k**2) / config.tau,
            ]
        )
        self.linear = linear
        h = config.dt
        self.e, self.e2 = np.exp(h * linear), np.exp(h * linear / 2)
        roots = np.exp(1j * np.pi * (np.arange(32) + 0.5) / 32)
        z = h * linear[..., None] + roots
        self.q = h * np.mean((np.exp(z / 2) - 1) / z, axis=-1).real
        self.f1 = h * np.mean((-4 - z + np.exp(z) * (4 - 3 * z + z * z)) / z**3, axis=-1).real
        self.f2 = h * np.mean((2 + z + np.exp(z) * (-2 + z)) / z**3, axis=-1).real
        self.f3 = h * np.mean((-4 - 3 * z - z * z + np.exp(z) * (4 - z)) / z**3, axis=-1).real

    def nonlinear(self, v, replay_m=None, replay_weight=None):
        u, m = np.fft.irfft(v, n=self.c.n, axis=-1)
        if replay_m is not None:
            m = (
                replay_m
                if replay_weight is None
                else ((1 - replay_weight) * m + replay_weight * replay_m)
            )
        with np.errstate(over="raise", invalid="raise"):
            return np.fft.rfft(
                np.array(
                    [
                        self.c.cubic * u**3 - u**5 + self.c.feedback * m * u,
                        self.c.source * u * u / self.c.tau,
                    ]
                ),
                axis=-1,
            )

    def step(self, v, replay=None, record=False, replay_weight=None):
        """Replay uses a sham's four RK-stage mediator fields, open loop only."""

        def nl(w, i):
            return self.nonlinear(w, None if replay is None else replay[i], replay_weight)

        nv = nl(v, 0)
        a = self.e2 * v + self.q * nv
        na = nl(a, 1)
        b = self.e2 * v + self.q * na
        nb = nl(b, 2)
        c = self.e2 * a + self.q * (2 * nb - nv)
        nc = nl(c, 3)
        result = self.e * v + self.f1 * nv + 2 * self.f2 * (na + nb) + self.f3 * nc
        if not np.isfinite(result).all():
            raise FloatingPointError("Nonfinite field; trajectory not clipped")
        stages = None
        if record:
            stages = np.array([np.fft.irfft(w[1], n=self.c.n) for w in (v, a, b, c)])
        return result, stages

    def evolve(self, state, duration, sample=1.0, replay=None, record=False, replay_weight=None):
        nsteps, stride = round(duration / self.c.dt), round(sample / self.c.dt)
        if not np.isclose(nsteps * self.c.dt, duration) or stride < 1:
            raise ValueError("Duration/sample must match time grid")
        state = np.array(state, dtype=float, copy=True)
        if state.shape != (2, self.c.n) or not np.isfinite(state).all():
            raise ValueError("Expected finite (2,n) initial field")
        if replay is not None and len(replay) != nsteps:
            raise ValueError("Replay must cover the exact simulation interval")
        if replay_weight is not None:
            replay_weight = np.asarray(replay_weight)
            if replay is None or replay_weight.shape != (self.c.n,):
                raise ValueError("Spatial replay needs a full grid weight and sham stages")
            if (
                not np.isfinite(replay_weight).all()
                or ((replay_weight < 0) | (replay_weight > 1)).any()
            ):
                raise ValueError("Replay weight must be between zero and one")
        v = np.fft.rfft(state, axis=-1)
        snapshots, times, stages = [state], [0.0], []
        for i in range(nsteps):
            v, stage = self.step(v, None if replay is None else replay[i], record, replay_weight)
            if record:
                stages.append(stage)
            if (i + 1) % stride == 0 or i + 1 == nsteps:
                snapshots.append(np.fft.irfft(v, n=self.c.n, axis=-1))
                times.append((i + 1) * self.c.dt)
        return np.array(times), np.array(snapshots), np.array(stages) if record else None

    def seed(self, centers=(0.0,), width=5.0, seed=0, variation=0.0):
        """Initial field only: localized patterned patches plus seeded smooth residuals."""
        rng = np.random.default_rng(seed)
        u = np.zeros(self.c.n)
        for center in centers:
            d = (self.x - center + self.c.length / 2) % self.c.length - self.c.length / 2
            residual = sum(
                rng.normal() * np.cos(j * d / 3 + rng.uniform(0, 2 * np.pi)) for j in range(1, 5)
            )
            u += 1.3 * np.cos(d) * np.exp(-((d / width) ** 8)) * (1 + variation * residual)
        return np.array([u, np.zeros_like(u)])

    def pulse(self, state, center, amplitude, width=2.0, channel=0, relative=False, compact=False):
        """One instantaneous Gaussian field perturbation; no future forcing."""
        if channel not in (0, 1) or width <= 0:
            raise ValueError("Unknown field or invalid width")
        result = np.array(state, copy=True)
        d = (self.x - center + self.c.length / 2) % self.c.length - self.c.length / 2
        if compact:
            envelope = np.zeros_like(d)
            inside = abs(d) < width
            envelope[inside] = amplitude * np.exp(1 - 1 / (1 - (d[inside] / width) ** 2))
        else:
            envelope = amplitude * np.exp(-0.5 * (d / width) ** 2)
        result[channel] += envelope * (state[channel] if relative else 1.0)
        return result
