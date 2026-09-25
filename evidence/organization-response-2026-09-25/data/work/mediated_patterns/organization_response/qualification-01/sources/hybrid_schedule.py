"""Finite declared C-event schedule around the unchanged spatial hybrid.

This module has no truth solver, extraction, replay, or readout dependencies.
Sampling is right-continuous: events at a sample time have already occurred.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from .hybrid_pair import PairHybrid


def event_ticks(events, dt, horizon=None):
    if len(events) > 3:
        raise ValueError("This experiment permits at most three declared events")
    result = []
    for t, a in events:
        if not np.isfinite([t, a]).all() or t < 0 or abs(a) > 0.1:
            raise ValueError(
                "Finite nonnegative time and qualified weak amplitude required"
            )
        k = round(t / dt)
        if abs(k * dt - t) > 1e-10:
            raise ValueError("Events must lie exactly on the integration grid")
        if horizon is not None and t > horizon:
            raise ValueError("Event beyond declared horizon")
        if result and k <= result[-1][0]:
            raise ValueError("Distinct increasing event times required")
        result.append((k, float(a)))
    return result


def controlled_responses(absolute):
    """Branches: sham, full stimulated, cut stimulated, cut sham."""
    a = np.asarray(absolute)
    full = a[:, 1] - a[:, 0]
    cut = a[:, 2] - a[:, 3]
    return full, cut, full - cut


class ScheduledHybrid:
    def __init__(self, model, center, events):
        if abs(model.time) > 1e-12:
            raise ValueError("Initialize schedule at prediction boundary only")
        self.model = model
        self.center = float(center)
        self.events = [list(v) for v in events]
        self.ticks = event_ticks(events, model.c.dt)
        self.tick = 0
        self.next_event = 0
        self.apply_current()

    def apply_current(self):
        if self.next_event < len(self.ticks):
            k, a = self.ticks[self.next_event]
            if k == self.tick:
                if a != 0:
                    self.model.pulse(self.center, a)
                self.next_event += 1

    def step(self):
        self.model.step()
        self.tick += 1
        self.apply_current()

    def save(self, directory):
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=False)
        self.model.save(path / "state.npz")
        metadata = {
            "center": self.center,
            "events": self.events,
            "tick": self.tick,
            "next_event": self.next_event,
            "state_sha256": hashlib.sha256(
                (path / "state.npz").read_bytes()
            ).hexdigest(),
            "contract": "declared external schedule only; no reference or future replay",
        }
        (path / "schedule.json").write_text(
            json.dumps(metadata, allow_nan=False, indent=2) + "\n"
        )

    @classmethod
    def load(cls, directory):
        path = Path(directory)
        meta = json.loads((path / "schedule.json").read_text())
        if (
            hashlib.sha256((path / "state.npz").read_bytes()).hexdigest()
            != meta["state_sha256"]
        ):
            raise ValueError("Checkpoint hash mismatch")
        obj = cls.__new__(cls)
        obj.model = PairHybrid.load(path / "state.npz")
        obj.center = meta["center"]
        obj.events = meta["events"]
        obj.ticks = event_ticks(obj.events, obj.model.c.dt)
        obj.tick = meta["tick"]
        obj.next_event = meta["next_event"]
        if abs(obj.model.time - obj.tick * obj.model.c.dt) > 1e-7:
            raise ValueError("Checkpoint clock mismatch")
        return obj
