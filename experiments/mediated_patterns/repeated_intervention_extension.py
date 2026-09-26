"""Composition-informed separate squared-amplitude response summaries."""

import numpy as np

from . import geometry_model as gm
from . import present_state_model as pm
from .intervention_model import State


class SquaredState(State):
    """Eleven scientific scalars: original nine plus q=sum b² and decaying e.

    No new fit: use [p,d,q,e] with the original response coefficients. These
    diagonal event summaries are not physical energy or measured memory sites.
    """

    def __init__(self, model, z):
        super().__init__(model, z)
        self.squared_memory = np.zeros(2)

    def advance(self, steps):
        if (
            isinstance(steps, bool)
            or not isinstance(steps, (int, np.integer))
            or steps < 0
        ):
            raise ValueError("Nonnegative integer steps required")
        for _ in range(steps):
            super().advance(1)
            self.squared_memory[1] *= np.exp(-1.0 / 20.0)
        return self.flow.z.copy()

    def event(self, amplitude):
        super().event(amplitude)
        self.squared_memory += (amplitude / 0.02) ** 2

    def response(self):
        r = self.model["readout"]
        features = np.r_[self.response_memory, self.squared_memory]
        standardized = (self.flow.z - r["mean_z"]) / r["scale_z"]
        features = np.outer(features, np.r_[1.0, standardized]).ravel()
        weights = features @ np.asarray(r["coefficients"])
        return pm.predict(self.model["F"], self.flow.z[None])[0] + (
            np.einsum("ok,kt->to", weights.reshape(2, -1), np.asarray(r["basis"]))
            * r["scale_y"]
        )

    def checkpoint(self, path):
        pm.save_json(
            path,
            {
                "schema": 2,
                "model_sha256": gm.identity(self.model),
                "z": self.flow.z.tolist(),
                "steps": self.flow.steps,
                "memory": self.memory.tolist(),
                "response_memory": self.response_memory.tolist(),
                "squared_memory": self.squared_memory.tolist(),
            },
        )

    @classmethod
    def restore(cls, model, checkpoint):
        if checkpoint.get("schema") != 2 or "squared_memory" not in checkpoint:
            raise ValueError("Extended checkpoint required")
        base = dict(checkpoint)
        squared = np.asarray(base.pop("squared_memory"), float)
        base["schema"] = 1
        obj = super().restore(model, base)
        if squared.shape != (2,) or not np.isfinite(squared).all():
            raise ValueError("Two finite squared summaries required")
        obj.squared_memory = squared
        return obj
