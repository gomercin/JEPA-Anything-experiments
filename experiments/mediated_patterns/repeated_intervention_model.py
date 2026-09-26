"""Solver-free scheduling of the unchanged intervention State; no field inputs."""

import numpy as np

from .intervention_model import State

PREFIXES = ((0, 0), (1, 0), (0, 1), (1, 1))
CONTRASTS = {
    "R00": [1, 0, 0, 0],
    "R10": [0, 1, 0, 0],
    "R01": [0, 0, 1, 0],
    "R11": [0, 0, 0, 1],
    "D1": [-1, 1, 0, 0],
    "D2": [-1, 0, 1, 0],
    "D12": [-1, 0, 0, 1],
    "D2|1": [0, -1, 0, 1],
    "K12": [1, -1, -1, 1],
}


def validate_schedule(times, amplitudes, final):
    if len(times) != 2 or len(amplitudes) != 2:
        raise ValueError("Exactly two scheduled conditioning slots required")
    if any(
        isinstance(t, bool) or not isinstance(t, (int, np.integer))
        for t in (*times, final)
    ):
        raise ValueError("Integer runtime boundaries required")
    if not 0 < times[0] < times[1] < final:
        raise ValueError("Strictly increasing elapsed boundaries required")
    if (
        not np.isfinite(amplitudes).all()
        or max(abs(a) for a in amplitudes) > 0.020000000001
    ):
        raise ValueError("Per-event amplitude guard")


def state_class(model):
    if model.get("repeated_extension") == "separate-squared-amplitude":
        from .repeated_intervention_extension import SquaredState

        return SquaredState
    return State


def vector(state):
    return np.r_[
        state.flow.z,
        state.memory,
        state.response_memory,
        getattr(state, "squared_memory", np.zeros(0)),
    ]


def continue_state(state, times, amplitudes, final, checkpoint=None):
    """Consume future slots only when reached. Restored state is never reset.

    Slots at or before the current counter have already been consumed; saved
    checkpoints in this experiment are two ticks after each event.
    """
    validate_schedule(times, amplitudes, final)
    if state.flow.steps > final:
        raise ValueError("Cannot run backward")
    trajectory = [vector(state)]
    while state.flow.steps < final:
        state.advance(1)
        for t, a in zip(times, amplitudes, strict=True):
            if state.flow.steps == t:
                state.event(a)
        trajectory.append(vector(state))
        if checkpoint is not None and state.flow.steps in (times[0] + 2, times[1] + 2):
            checkpoint(state)
    # The final diagnostic probe is represented by response(), never event().
    return np.asarray(trajectory), state.response()


def forecast(model, z0, times, amplitudes, final=40, checkpoint=None):
    return continue_state(
        state_class(model)(model, z0), times, amplitudes, final, checkpoint
    )


def contrasts(responses):
    responses = np.asarray(responses)
    if responses.shape[0] != 4:
        raise ValueError("Prefix axis must be 00,10,01,11")
    return {k: np.einsum("p,p...->...", v, responses) for k, v in CONTRASTS.items()}


def comparators(responses):
    r = np.asarray(responses)
    addition, last = r.copy(), r.copy()
    addition[3] = r[1] + r[2] - r[0]
    last[3] = r[2]
    return {
        "unchanged-nine": r,
        "ignore-both": np.repeat(r[:1], 4, axis=0),
        "last-only": last,
        "independent-addition": addition,
    }
