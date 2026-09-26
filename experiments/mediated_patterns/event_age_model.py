"""Fixed-size causal response aging; no age, field or schedule in the readout."""

import json

import numpy as np

from . import geometry_model as gm
from . import present_state_model as pm
from .intervention_model import State as GeometryState


class State:
    def __init__(self, model, z):
        self.model = json.loads(json.dumps(model, allow_nan=False))
        decays = np.asarray(model["decays"], float)
        if (
            decays.ndim != 1
            or not len(decays)
            or not np.isfinite(decays).all()
            or np.any((decays <= 0) | (decays > 1))
        ):
            raise ValueError("Stable finite diagonal response transition required")
        if "readout" in model["geometry"]:
            raise ValueError("Old response memory must be removed, not hidden")
        self.geometry = GeometryState(model["geometry"], z)
        self.response_state = np.zeros((2, len(model["decays"])))

    def advance(self, steps):
        if (
            isinstance(steps, bool)
            or not isinstance(steps, (int, np.integer))
            or steps < 0
        ):
            raise ValueError("Nonnegative integer integration steps required")
        for _ in range(steps):
            self.geometry.advance(1)
            self.response_state *= self.model["decays"]
        return self.geometry.flow.z.copy()

    def event(self, amplitude):
        self.geometry.event(amplitude)
        b = amplitude / 0.02
        self.response_state += np.array([b, b * b])[:, None]

    def response(self):
        return readout(self.model, self.geometry.flow.z, self.response_state)

    def checkpoint(self, path):
        pm.save_json(
            path,
            {
                "schema": 1,
                "model_sha256": gm.identity(self.model),
                "z": self.geometry.flow.z.tolist(),
                "steps": self.geometry.flow.steps,
                "deformation": self.geometry.memory.tolist(),
                "response_state": self.response_state.tolist(),
            },
        )

    @classmethod
    def restore(cls, model, saved):
        if (
            set(saved)
            != {"schema", "model_sha256", "z", "steps", "deformation", "response_state"}
            or saved["schema"] != 1
            or saved["model_sha256"] != gm.identity(model)
        ):
            raise ValueError("Checkpoint/model mismatch")
        obj = cls(model, saved["z"])
        if (
            isinstance(saved["steps"], bool)
            or not isinstance(saved["steps"], int)
            or saved["steps"] < 0
        ):
            raise ValueError("Invalid counter")
        for key, shape in [
            ("deformation", (4,)),
            ("response_state", obj.response_state.shape),
        ]:
            value = np.asarray(saved[key], float)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Invalid retained coordinates")
        obj.geometry.flow.steps = saved["steps"]
        obj.geometry.memory = np.asarray(saved["deformation"], float)
        obj.response_state = np.asarray(saved["response_state"], float)
        return obj


def readout(model, centers, response_state):
    """Pure algebra also used by the explicitly separated evaluator diagnostic."""
    r = model["readout"]
    q = (np.asarray(centers) - r["mean_z"]) / r["scale_z"]
    q = q @ np.asarray(r.get("center_projection", np.eye(3)))
    features = np.outer(np.asarray(response_state).ravel(), np.r_[1.0, q]).ravel()
    weights = features @ np.asarray(r["coefficients"])
    result = pm.predict(model["geometry"]["F"], np.asarray(centers)[None])[0]
    result += np.einsum("ok,kt->to", weights.reshape(2, -1), r["basis"]) * r["scale_y"]
    if "prior" in model:
        prior = model["prior"]
        # Linear single-event realization of inherited p,d,p²,p*d features.
        # Squared excitation is injected per event, never inferred from net p.
        features = np.asarray(response_state)[:, :2].ravel()
        q = (np.asarray(centers) - prior["mean_z"]) / prior["scale_z"]
        w = np.outer(features, np.r_[1.0, q]).ravel() @ np.asarray(
            prior["coefficients"]
        )
        result += (
            np.einsum("ok,kt->to", w.reshape(2, -1), prior["basis"]) * prior["scale_y"]
        )
    return result


def vector(state):
    return np.r_[
        state.geometry.flow.z, state.geometry.memory, state.response_state.ravel()
    ]


def continue_state(state, events, final, checkpoint=None):
    """Schedule dispatch alone knows elapsed boundaries, never fitted dynamics."""
    times = [t for t, a in events]
    if (
        any(
            isinstance(t, bool) or not isinstance(t, (int, np.integer))
            for t in [*times, final]
        )
        or times != sorted(set(times))
        or any(t <= 0 or t >= final for t in times)
        or final < state.geometry.flow.steps
    ):
        raise ValueError("Ordered integer future event boundaries required")
    if any(not np.isfinite(a) or abs(a) > 0.020000000001 for t, a in events):
        raise ValueError("Per-event amplitude guard")
    trajectory = [vector(state)]
    while state.geometry.flow.steps < final:
        state.advance(1)
        for t, a in events:
            if state.geometry.flow.steps == t:
                state.event(a)
        trajectory.append(vector(state))
        if checkpoint and state.geometry.flow.steps in [t + 2 for t, a in events]:
            checkpoint(state)
    return np.asarray(trajectory), state.response()


def forecast(model, z0, events, final=40, checkpoint=None):
    return continue_state(State(model, z0), events, final, checkpoint)
