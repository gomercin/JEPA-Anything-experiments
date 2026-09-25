"""Causal kernel and fixed-size discrete response realization. No field imports.

Both approximate the declared *relative* pulse amplitude; neither receives the
instantaneous microscopic increment. A quadratic input map approximates isolated
pulse nonlinearity, not arbitrary state-dependent susceptibility.
"""

import json
from dataclasses import dataclass, field

import numpy as np

from .reduced import features


def input_features(separation, amplitude, degree=2):
    phi = features([separation, amplitude])
    return phi[:5] if degree == 1 else phi


@dataclass
class CausalKernel:
    step: float
    coefficients: np.ndarray  # time x input features; interacting-pair responses
    degree: int = 2

    @classmethod
    def fit(cls, records, degree=2):
        times = np.array(records[0]["time"])
        if degree not in (1, 2) or len(times) < 2 or times[0] != 0:
            raise ValueError("Use degree1/2 and a time grid starting at zero")
        if times[1] <= 0 or not np.allclose(np.diff(times), times[1]):
            raise ValueError("Use a uniform positive time grid")
        for r in records:
            if (
                not np.array_equal(r["time"], times)
                or len(r["pulses"]) != 1
                or r["pulses"][0][0] != 0
            ):
                raise ValueError(
                    "Kernel fit needs aligned isolated pulses at time zero"
                )
        x = np.array(
            [
                input_features(r["separation"], r["pulses"][0][1], degree)
                for r in records
            ]
        )
        y = np.array([r["response"] for r in records])
        coeff, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
        if rank != x.shape[1]:
            raise ValueError("Rank-deficient kernel design")
        coeff[:, 0] = 0  # u pulse cannot instantaneously change mediator measurement.
        return cls(float(times[1] - times[0]), coeff.T, degree)

    def impulse(self, separation, amplitude, elapsed):
        elapsed = np.asarray(elapsed)
        if np.any(elapsed > self.step * (len(self.coefficients) - 1) + 1e-10):
            raise ValueError(
                "Outside measured kernel horizon; no silent tail truncation"
            )
        values = self.coefficients @ input_features(separation, amplitude, self.degree)
        return np.interp(elapsed, np.arange(len(values)) * self.step, values, left=0)

    def predict(self, separation, pulses, times):
        times = np.asarray(times)
        prediction = np.zeros_like(times, dtype=float)
        for when, amplitude in pulses:
            prediction += self.impulse(separation, amplitude, times - when)
        return prediction

    def save(self, path):
        path.write_text(
            json.dumps(
                {
                    "step": self.step,
                    "degree": self.degree,
                    "coefficients": self.coefficients.tolist(),
                },
                allow_nan=False,
            )
        )

    @classmethod
    def load(cls, path):
        d = json.loads(path.read_text())
        return cls(d["step"], np.array(d["coefficients"]), d["degree"])


@dataclass
class ResponseState:
    """Discrete ERA realization, with a two-scalar one-tick event buffer.

    The buffer enforces continuous exterior output at an instantaneous u pulse.
    Unforced steps advance A; no pulse history, kernel table or field replay.
    Geometry is fixed at initialization. Arbitrary sub-tick times are rejected.
    """

    step: float
    transition: np.ndarray
    injection: np.ndarray
    readout: np.ndarray
    degree: int = 2
    separation: float = 0.0
    state: np.ndarray = field(default_factory=lambda: np.empty(0))
    pending: np.ndarray = field(default_factory=lambda: np.zeros(2))
    time: float = 0.0
    susceptibility: float = 0.0
    susceptibility_tau: float = 1.0
    susceptibility_gain: float = 0.0
    susceptibility_quadratic_gain: float = 0.0

    @classmethod
    def fit(cls, kernel, order=4, block=100):
        # Markov coefficients h[k]=C A^(k-1) B for k>=1, D=h[0]=0.
        h = kernel.coefficients
        if 2 * block + 1 >= len(h):
            raise ValueError("Kernel too short for requested Hankel blocks")
        h0 = np.concatenate(
            [h[i + 1 : i + 1 + block].reshape(1, -1) for i in range(block)]
        )
        h1 = np.concatenate(
            [h[i + 2 : i + 2 + block].reshape(1, -1) for i in range(block)]
        )
        u, s, vh = np.linalg.svd(h0, full_matrices=False)
        if order > len(s) or s[order - 1] < 1e-15:
            raise ValueError("Requested order exceeds resolved Hankel rank")
        root = np.sqrt(s[:order])
        uu, vv = u[:, :order], vh[:order]
        a = ((uu.T @ h1 @ vv.T) / root[:, None]) / root[None, :]
        b = root[:, None] * vv[:, : h.shape[1]]
        c = uu[0] * root
        return cls(kernel.step, a, b, c, kernel.degree)

    def initialize(self, separation):
        if not np.isfinite(separation):
            raise ValueError("Finite initial separation required")
        self.separation = float(separation)
        self.state = np.zeros(len(self.transition))
        self.pending = np.zeros(2)
        self.time = 0.0
        self.susceptibility = 0.0
        return self

    def jump(self, amplitude):
        if not np.isfinite(amplitude):
            raise ValueError("Finite declared pulse amplitude required")
        effective = amplitude * (
            1
            + self.susceptibility
            * (
                self.susceptibility_gain
                + self.susceptibility_quadratic_gain * amplitude
            )
        )
        self.pending += [effective, effective**2 if self.degree == 2 else 0]
        if self.susceptibility_gain or self.susceptibility_quadratic_gain:
            self.susceptibility += amplitude

    def advance(self, elapsed):
        ticks = round(elapsed / self.step)
        if ticks < 0 or not np.isclose(ticks * self.step, elapsed, atol=1e-10, rtol=0):
            raise ValueError("Advance must use nonnegative model-grid times")
        if not self.state.size:
            raise ValueError("Initialize before advancing")
        for _ in range(ticks):
            phi = input_features(self.separation, 1, self.degree)
            phi[:5] *= self.pending[0]
            if self.degree == 2:
                phi[5:] *= self.pending[1]
            self.state = self.transition @ self.state + self.injection @ phi
            self.pending[:] = 0
            self.susceptibility *= np.exp(-self.step / self.susceptibility_tau)
            self.time += self.step
        if not np.isfinite(self.state).all():
            raise FloatingPointError("Reduced recurrence diverged; not clipped")

    def output(self):
        return float(self.readout @ self.state)

    def rollout(self, separation, pulses, times):
        self.initialize(separation)
        events = {}
        last = -1.0
        for when, a in pulses:
            if when < 0 or when <= last:
                raise ValueError("Use distinct chronological nonnegative events")
            last = when
            if not np.isclose(round(when / self.step) * self.step, when, atol=1e-10):
                raise ValueError("Pulse off model grid")
            events.setdefault(round(when / self.step), []).append(a)
        result = []
        target = np.asarray(times)
        if target[0] != 0 or not np.allclose(np.diff(target), self.step):
            raise ValueError("Use contiguous model-grid output times")
        for i, _ in enumerate(target):
            if i:
                self.advance(self.step)
            for a in events.get(i, []):
                self.jump(a)
            result.append(self.output())
        return np.array(result)

    def save(self, path):
        data = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in vars(self).items()
        }
        path.write_text(json.dumps(data, allow_nan=False))

    @classmethod
    def load(cls, path):
        d = json.loads(path.read_text())
        for k in ("transition", "injection", "readout", "state", "pending"):
            d[k] = np.array(d[k])
        return cls(**d)


def effective_pulses(pulses, tau=1, gain=0, quadratic_gain=0):
    """One phenomenological susceptibility variable, driven only by prior inputs.

    This is a tested event-map approximation, not the true microscopic pulse
    increment. Returned effective amplitudes are for diagnostic kernel use.
    """
    memory, last, result = 0.0, 0.0, []
    for when, a in pulses:
        if when < last:
            raise ValueError("Events must be chronologically ordered")
        memory *= np.exp(-(when - last) / tau)
        result.append((when, a * (1 + memory * (gain + quadratic_gain * a))))
        memory += a
        last = when
    return result
