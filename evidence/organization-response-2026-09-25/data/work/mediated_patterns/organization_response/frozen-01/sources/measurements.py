"""Readouts only. The field solver never imports this module."""

import numpy as np


def regions(x, centers, radius=8.0):
    """C2 windows: weight 1 through radius 5, smoothly taper to 0 at radius 8.

    Hard cutoffs produced resolution-dependent boundary quadrature in the pilot.
    These fixed disjoint readout weights are never passed to the dynamics.
    """
    if radius <= 5:
        raise ValueError("Window radius must exceed the flat core radius 5")
    d = np.array([abs(x - c) for c in centers])
    a = np.clip((d - 5) / (radius - 5), 0, 1)  # Measurement coordinate, not field clipping.
    masks = 1 - (6 * a**5 - 15 * a**4 + 10 * a**3)
    if ((masks > 0).sum(axis=0) > 1).any():
        raise ValueError("Measurement windows overlap")
    return masks


def describe(x, states, centers, radius=8.0):
    masks = regions(x, centers, radius)
    dx = x[1] - x[0]
    result = {k: [] for k in ("mass", "center", "width", "mediator", "peak")}
    for weights in masks:
        mask = weights > 0
        u, m = states[..., 0, mask], states[..., 1, mask]
        w = u * u * weights[mask]
        mass = w.sum(axis=-1) * dx
        if (mass <= 1e-12).any():
            raise ValueError("Pattern absent in declared measurement region")
        center = (w * x[mask]).sum(axis=-1) * dx / mass
        width = np.sqrt((w * (x[mask] - center[..., None]) ** 2).sum(axis=-1) * dx / mass)
        for k, v in zip(
            result,
            (mass, center, width, (m * w).sum(axis=-1) * dx / mass, abs(u).max(axis=-1)),
            strict=True,
        ):
            result[k].append(v)
    return {k: np.stack(v, axis=-1) for k, v in result.items()}


def shift(field, state, distance):
    """Preparation translation, not recentering during dynamics."""
    return np.fft.irfft(
        np.fft.rfft(state, axis=-1) * np.exp(-1j * field.k * distance), n=field.c.n, axis=-1
    )


def outward(x, states, receiver_center):
    """Mean mediator in a fixed two-unit window 12 units beyond the receiver.

    The C-infinity sensor lies outside the visible pattern, with no feedback
    into evolution. It is an outward-field readout, not a third participant.
    """
    d = (x - receiver_center - 12) / 2
    w = np.zeros_like(x)
    inside = abs(d) < 1
    w[inside] = np.exp(1 - 1 / (1 - d[inside] ** 2))
    return (states[..., 1, :] * w).sum(axis=-1) / w.sum()


def prepare(field, separation, seed, age=100.0, residual=0.0):
    """Two independently relaxed fields, then an unconstrained joint preparation.

    Optional smooth residuals are inserted at assembly, then relax for `age`.
    A and B labels belong to preparation/readout only.
    """
    centers = (-separation / 2, separation / 2)
    states = []
    for i, c in enumerate(centers):
        initial = field.seed(seed=seed + 100 * i, variation=0.01)
        isolated = field.evolve(initial, 150, sample=150)[1][-1]
        states.append(shift(field, isolated, c))
    joined = sum(states)
    if residual:
        rng = np.random.default_rng(seed + 3000)
        for c in centers:
            d = field.x - c
            joined[0] += (
                residual
                * np.exp(-((d / 5) ** 2))
                * sum(
                    rng.normal() * np.cos(j * d / 2 + rng.uniform(0, 2 * np.pi))
                    for j in range(1, 5)
                )
            )
        # This is a preparation disturbance of the medium, not ongoing noise.
        joined[1] += residual * np.exp(-((field.x / 20) ** 2)) * np.cos(field.x / 7)
    return field.evolve(joined, age, sample=age)[1][-1], states
