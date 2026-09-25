"""Fixed-size event-state correction. Deployment has no simulator imports."""

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .response_state import input_features


@dataclass
class EventResponse:
    step: float
    transition: np.ndarray
    injection: np.ndarray
    readout: np.ndarray
    event_maps: np.ndarray  # power x response-state x response-state
    separation: float = 0.0
    state: np.ndarray = field(default_factory=lambda: np.empty(0))
    pending: float = 0.0
    time: float = 0.0

    @classmethod
    def from_baseline(cls, model, powers=1):
        n = len(model.transition)
        return cls(
            model.step,
            model.transition.copy(),
            model.injection.copy(),
            model.readout.copy(),
            np.zeros((powers, n, n)),
        )

    def initialize(self, separation):
        if not np.isfinite(separation):
            raise ValueError("Finite initial separation required")
        self.separation = float(separation)
        self.state = np.zeros(len(self.transition))
        self.pending = 0.0
        self.time = 0.0
        return self

    def jump(self, amplitude):
        if not np.isfinite(amplitude) or self.pending:
            raise ValueError("One finite command per model tick")
        self.pending = float(amplitude)

    def advance(self, elapsed):
        ticks = round(elapsed / self.step)
        if ticks < 0 or not np.isclose(ticks * self.step, elapsed, atol=1e-10, rtol=0):
            raise ValueError("Advance must use nonnegative model-grid time")
        if not self.state.size:
            raise ValueError("Initialize first")
        with np.errstate(over="raise", invalid="raise"):
            for _ in range(ticks):
                a = self.pending
                before = self.state
                self.state = self.transition @ before + self.injection @ input_features(
                    self.separation, a
                )
                for power, matrix in enumerate(self.event_maps, 1):
                    self.state += a**power * (matrix @ before)
                self.pending = 0.0
                self.time += self.step
                if not np.isfinite(self.state).all():
                    raise FloatingPointError("Nonfinite reduced state; not clipped")

    def output(self):
        return float(self.readout @ self.state)

    def rollout(self, separation, pulses, times):
        self.initialize(separation)
        times = np.asarray(times)
        if times[0] != 0 or not np.allclose(np.diff(times), self.step):
            raise ValueError("Contiguous output grid starting at zero required")
        events = {}
        last = -1.0
        for when, a in pulses:
            tick = round(when / self.step)
            if (
                when < 0
                or when <= last
                or not np.isclose(tick * self.step, when, atol=1e-10, rtol=0)
            ):
                raise ValueError("Distinct chronological on-grid pulses required")
            events[tick] = a
            last = when
        values = []
        for i in range(len(times)):
            if i:
                self.advance(self.step)
            if i in events:
                self.jump(events[i])
            values.append(self.output())
        return np.array(values)

    def save(self, path):
        d = {
            k: v.tolist() if isinstance(v, np.ndarray) else v
            for k, v in vars(self).items()
        }
        Path(path).write_text(json.dumps(d, allow_nan=False))

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        for k in ("transition", "injection", "readout", "event_maps", "state"):
            d[k] = np.array(d[k])
        return cls(**d)


def fit_event_map(baseline, quartets, powers=1, rcond=1e-4):
    """Fit only resolved quartet cross terms; do not change A/B/C or first pulse.

    Two input powers optionally distinguish the next pulse's sign/magnitude.
    Column scaling and fixed SVD cutoff regularize the finite design. This is
    approximate model fitting, not an identification of physical state variables.
    """
    model = EventResponse.from_baseline(baseline, powers)
    n = len(model.transition)
    observability = []
    c = model.readout.copy()
    for _ in range(400):
        observability.append(c.copy())
        c = c @ model.transition
    observability = np.array(observability)
    design = []
    targets = []
    for r in quartets:
        (_, a), (gap, b) = r["pulses"]
        baseline.initialize(r["separation"])
        baseline.jump(a)
        baseline.advance(gap)
        x = np.concatenate([b**p * baseline.state for p in range(1, powers + 1)])
        start = round(gap / model.step) + 1
        y = np.array(r["terms"]["cross"])[start:]
        o = observability[: len(y)]
        # Tensor uses blocks M_power with row-major n-by-n matrices.
        block = np.concatenate(
            [
                np.einsum("ti,j->tij", o, x[p * n : (p + 1) * n]).reshape(len(y), -1)
                for p in range(powers)
            ],
            axis=1,
        )
        scale = np.sqrt(np.mean(np.array(r["terms"]["r1"]) ** 2)) + np.sqrt(
            np.mean(np.array(r["terms"]["r2"]) ** 2)
        )
        design.append(block / scale)
        targets.append(y / scale)
    x, y = np.concatenate(design), np.concatenate(targets)
    column_norm = np.linalg.norm(x, axis=0)
    column_norm[column_norm == 0] = 1
    coefficients, _, rank, sv = np.linalg.lstsq(x / column_norm, y, rcond=rcond)
    model.event_maps = (coefficients / column_norm).reshape(powers, n, n)
    return model, {
        "powers": powers,
        "coefficients": int(model.event_maps.size),
        "rank": int(rank),
        "rcond": rcond,
        "scaled_singular_values": sv.tolist(),
        "fit_normalized_rmse": float(
            np.sqrt(np.mean((x @ (coefficients / column_norm) - y) ** 2))
        ),
    }


@dataclass
class CrossResponse:
    """Two decaying input summaries drive a cross-only linear response injection.

    Unlike the earlier susceptibility repair this does not replace a by an
    effective amplitude or inadvertently square that correction. It adds mixed
    pulse products along a supplied response direction. No list of past events.
    """

    step: float
    transition: np.ndarray
    injection: np.ndarray
    readout: np.ndarray
    decay_times: np.ndarray
    cross_weights: np.ndarray
    separation: float = 0.0
    state: np.ndarray = field(default_factory=lambda: np.empty(0))
    memory: np.ndarray = field(default_factory=lambda: np.zeros(2))
    pending: float = 0.0
    pending_cross: float = 0.0
    time: float = 0.0
    separation_knots: np.ndarray = field(default_factory=lambda: np.empty(0))
    injection_knots: np.ndarray = field(default_factory=lambda: np.empty(0))

    def initialize(self, separation):
        if not np.isfinite(separation):
            raise ValueError("Finite separation required")
        self.separation = float(separation)
        if self.separation_knots.size:
            from scipy.interpolate import CubicSpline

            if not self.separation_knots[0] <= separation <= self.separation_knots[-1]:
                raise ValueError("Outside supplied geometry interpolation range")
            self.injection = CubicSpline(
                self.separation_knots, self.injection_knots, axis=0
            )(separation)
        self.state = np.zeros(len(self.transition))
        self.memory = np.zeros(2)
        self.pending = self.pending_cross = self.time = 0.0
        return self

    def jump(self, amplitude):
        a = float(amplitude)
        if not np.isfinite(a) or self.pending or self.pending_cross:
            raise ValueError("One finite event per model tick")
        self.pending = a
        weights = self.cross_weights
        if weights.ndim == 2:
            weights = geometry(self.separation) @ weights
        self.pending_cross = float(
            np.array(
                [
                    a * self.memory[0],
                    a * a * self.memory[0],
                    a * self.memory[1],
                    a * a * self.memory[1],
                ]
            )
            @ weights
        )
        self.memory += np.array([a, a * a])

    def advance(self, elapsed):
        ticks = round(elapsed / self.step)
        if ticks < 0 or not np.isclose(ticks * self.step, elapsed, atol=1e-10, rtol=0):
            raise ValueError("Nonnegative grid time required")
        if not self.state.size:
            raise ValueError("Initialize first")
        with np.errstate(over="raise", invalid="raise"):
            for _ in range(ticks):
                if self.separation_knots.size:
                    phi = np.array([self.pending**p for p in range(1, 5)])
                    direction = self.injection[:, 0]
                else:
                    phi = input_features(self.separation, self.pending)
                    direction = (
                        self.injection[:, :5] @ input_features(self.separation, 1)[:5]
                    )
                self.state = self.transition @ self.state + self.injection @ phi
                self.state += direction * self.pending_cross
                self.memory *= np.exp(-self.step / self.decay_times)
                self.pending = self.pending_cross = 0.0
                self.time += self.step
                if not np.isfinite(self.state).all():
                    raise FloatingPointError("Nonfinite state; not clipped")

    output = EventResponse.output
    rollout = EventResponse.rollout
    save = EventResponse.save

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        for k in (
            "transition",
            "injection",
            "readout",
            "decay_times",
            "cross_weights",
            "state",
            "memory",
        ):
            d[k] = np.array(d[k])
        for k in ("separation_knots", "injection_knots"):
            if k in d:
                d[k] = np.array(d[k])
        return cls(**d)


def geometry(separation):
    """Static supplied SH-tail/mediator ratio; never moving field measurements."""
    return np.r_[
        1.0, input_features(separation, 1)[3:5] / np.exp(-(separation - 28) / 8)
    ]


def fit_cross_response(base, quartets, conditioned=False):
    from scipy.optimize import least_squares

    shapes = []
    targets = []
    inputs = []
    for r in quartets:
        (t1, a), (t2, b) = r["pulses"]
        gap = t2 - t1
        n = round((r["time"][-1] - t2) / base.step)
        base.initialize(r["separation"])
        if isinstance(base, CrossResponse) and base.separation_knots.size:
            z = base.injection[:, 0].copy()
        else:
            z = base.injection[:, :5] @ input_features(r["separation"], 1)[:5]
        curve = []
        for _ in range(n):
            curve.append(base.readout @ z)
            z = base.transition @ z
        scale = np.sqrt(np.mean(np.array(r["terms"]["r1"]) ** 2)) + np.sqrt(
            np.mean(np.array(r["terms"]["r2"]) ** 2)
        )
        shapes.append(np.array(curve) / scale)
        targets.append(
            np.array(r["terms"]["cross"])[round(t2 / base.step) + 1 :] / scale
        )
        inputs.append((a, b, gap, r["separation"]))
    y = np.concatenate(targets)

    def design(tau):
        blocks = []
        for (a, b, gap, separation), shape in zip(inputs, shapes, strict=True):
            w1 = a * np.exp(-gap / tau[0])
            w2 = a * a * np.exp(-gap / tau[1])
            phi = np.array([b * w1, b * b * w1, b * w2, b * b * w2])
            if conditioned:
                phi = np.outer(geometry(separation), phi).ravel()
            blocks.append(shape[:, None] * phi)
        return np.concatenate(blocks)

    def residual(logtau):
        x = design(np.exp(logtau))
        beta = np.linalg.lstsq(x, y, rcond=1e-8)[0]
        return x @ beta - y

    fit = least_squares(
        residual,
        np.log([2.0, 3.0]),
        bounds=(np.log([0.25, 0.25]), np.log([10.0, 10.0])),
        ftol=1e-12,
        xtol=1e-12,
        gtol=1e-12,
    )
    tau = np.exp(fit.x)
    x = design(tau)
    beta = np.linalg.lstsq(x, y, rcond=1e-8)[0]
    m = CrossResponse(
        base.step,
        base.transition.copy(),
        base.injection.copy(),
        base.readout.copy(),
        tau,
        beta.reshape(3, 4) if conditioned else beta,
    )
    if isinstance(base, CrossResponse):
        m.separation_knots = base.separation_knots.copy()
        m.injection_knots = base.injection_knots.copy()
    return m, {
        "tau": tau.tolist(),
        "weights": beta.tolist(),
        "success": bool(fit.success),
        "normalized_cross_fit_rmse": float(np.sqrt(np.mean((x @ beta - y) ** 2))),
        "tau_bounds": [0.25, 10.0],
        "objective": "cross-only, normalized by sum of single-response RMS",
    }


def refit_single_inputs(base, records):
    """Quartic amplitude at nine measured geometry knots, projected into same A/C.

    Geometry interpolation occurs once at initialization. A and C, response
    dimension, pulse timing and physical task remain unchanged.
    """
    o = []
    c = base.readout.copy()
    for _ in range(400):
        o.append(c.copy())
        c = c @ base.transition
    o = np.array(o)
    knots = []
    injections = []
    scores = []
    for d in sorted({r["nominal_separation"] for r in records}):
        rows = [r for r in records if r["nominal_separation"] == d]
        x = np.array([[r["pulses"][0][1] ** p for p in range(1, 5)] for r in rows])
        y = np.array([r["response"] for r in rows])
        h = np.linalg.lstsq(x, y, rcond=None)[0].T
        b = np.linalg.lstsq(o, h[1:], rcond=None)[0]
        knots.append(float(np.mean([r["separation"] for r in rows])))
        injections.append(b)
        scores.append(
            {
                "nominal_separation": d,
                "basis_projection_relative": float(
                    np.linalg.norm(o @ b - h[1:]) / np.linalg.norm(h[1:])
                ),
            }
        )
    model = CrossResponse(
        base.step,
        base.transition.copy(),
        injections[0].copy(),
        base.readout.copy(),
        np.ones(2),
        np.zeros(4),
        separation_knots=np.array(knots),
        injection_knots=np.array(injections),
    )
    return model, scores
