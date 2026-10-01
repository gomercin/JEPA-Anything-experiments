"""Frozen atlas/G independent addition. Inference takes only t50 centers/schedule."""

import numpy as np

from . import event_atlas_model as am
from . import geometry_model as gm
from .repeated_intervention_model import validate_schedule


def forecast(model, g, centers, times, amplitudes):
    validate_schedule(times, amplitudes, 40)
    z = gm.rollout(g, centers, times)
    components = np.array(
        [
            am.predict(model, center, a, 40 - t, "quadratic")
            for center, t, amplitude in zip(z, times, amplitudes, strict=True)
            for a in (0.0, amplitude)
        ]
    )
    b1, r1, b2, r2 = components
    d2 = r2 - b2
    common = np.array([b1, r1, b1 + d2, r1 + d2])
    literal = np.array([b1, r1, r2, r1 + r2 - b1])
    return {
        "common": common,
        "literal": literal,
        "components": components,
        "centers": z,
    }
