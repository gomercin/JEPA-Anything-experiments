"""Three-coordinate autonomous updates. No field or reference solver access."""

import hashlib
import json

import numpy as np

from .present_state_model import design, save_json


def identity(model):
    return hashlib.sha256(
        json.dumps(model, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def fit(z, rates, kind="affine", ridge=1e-6, step=1.0):
    """Only caller-selected training rows; physical rates, no ages or labels."""
    z, rates = np.asarray(z, float), np.asarray(rates, float)
    if z.ndim != 2 or z.shape[1] != 3 or rates.shape != z.shape:
        raise ValueError("Expected matching (N,3) training centers/rates")
    if not np.isfinite(z).all() or not np.isfinite(rates).all():
        raise ValueError("Nonfinite training data")
    if (
        kind not in ("persistence", "drift", "affine", "quadratic")
        or ridge < 0
        or step <= 0
    ):
        raise ValueError("Unsupported update contract")
    mean = z.mean(axis=0) if kind in ("affine", "quadratic") else np.zeros(3)
    scale = (
        np.maximum(z.std(axis=0), 1e-14)
        if kind in ("affine", "quadratic")
        else np.ones(3)
    )
    a = np.ones((len(z), 1))
    if kind in ("affine", "quadratic"):
        a = design((z - mean) / scale, 2 if kind == "quadratic" else 1)
    penalty = np.eye(a.shape[1]) * ridge
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(a.T @ a + penalty, a.T @ rates)
    if kind == "persistence":
        coefficients[:] = 0
    return {
        "schema": 1,
        "kind": kind,
        "step": step,
        "ridge": ridge,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "coefficients": coefficients.tolist(),
        "condition": float(np.linalg.cond(a)),
        "training_rows": len(z),
        "retained_training_examples": 0,
    }


def velocity(model, z):
    """Shared autonomous law; no clock or metadata parameter."""
    x = (np.asarray(z, float) - model["mean"]) / model["scale"]
    a = (
        design(x[None], 2 if model["kind"] == "quadratic" else 1)[0]
        if model["kind"] in ("affine", "quadratic")
        else np.ones(1)
    )
    return a @ np.asarray(model["coefficients"])


class Continuation:
    """Fixed-grid RK4. State is three centers plus an integration step counter."""

    def __init__(self, model, z):
        self.model = json.loads(json.dumps(model, allow_nan=False))
        self.z = np.array(z, dtype=float, copy=True)
        if self.z.shape != (3,) or not np.isfinite(self.z).all():
            raise ValueError("Initialize with three finite current offsets only")
        if self.model["step"] <= 0:
            raise ValueError("Positive fixed step required")
        self.steps = 0

    def advance(self, steps):
        if (
            isinstance(steps, bool)
            or not isinstance(steps, (int, np.integer))
            or steps < 0
        ):
            raise ValueError("Advance by a nonnegative integer number of fixed steps")
        dt = self.model["step"]
        for _ in range(steps):
            k1 = velocity(self.model, self.z)
            k2 = velocity(self.model, self.z + dt * k1 / 2)
            k3 = velocity(self.model, self.z + dt * k2 / 2)
            k4 = velocity(self.model, self.z + dt * k3)
            self.z += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
            self.steps += 1
            if not np.isfinite(self.z).all():
                raise FloatingPointError("Unstable reduced rollout; no clipping")
        return self.z.copy()

    def checkpoint(self, path):
        save_json(
            path,
            {
                "schema": 1,
                "model_sha256": identity(self.model),
                "z": self.z.tolist(),
                "steps": self.steps,
            },
        )

    @classmethod
    def restore(cls, model, checkpoint):
        if set(checkpoint) != {"schema", "model_sha256", "z", "steps"}:
            raise ValueError("Unexpected checkpoint fields")
        if checkpoint["schema"] != 1 or checkpoint["model_sha256"] != identity(model):
            raise ValueError("Checkpoint model mismatch")
        result = cls(model, checkpoint["z"])
        steps = checkpoint["steps"]
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("Invalid checkpoint step counter")
        result.steps = steps
        return result


def rollout(model, z0, delays):
    """One uninterrupted rollout, observed without reinitialization at delays."""
    delays = np.asarray(delays, float)
    ticks = np.rint(delays / model["step"]).astype(int)
    if (
        delays.ndim != 1
        or (ticks < 0).any()
        or (np.diff(ticks) <= 0).any()
        or not np.allclose(ticks * model["step"], delays, rtol=0, atol=1e-12)
    ):
        raise ValueError("Increasing delays must lie on the fixed step grid")
    state = Continuation(model, z0)
    return np.array([state.advance(int(t - state.steps)) for t in ticks])
