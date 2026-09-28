"""Finite context-indexed operators; no fields, history labels or reference access."""

import numpy as np
from scipy.interpolate import PchipInterpolator

from . import event_operator_model as om

COARSE = (10, 15, 18, 20, 30)
REFINED = (10, 14, 15, 17, 18, 20, 25, 30)


def nodes_checked(nodes):
    a = np.asarray(nodes, float)
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all() or (np.diff(a) <= 0).any():
        raise ValueError("Finite strictly increasing calibrated nodes required")
    return a


def neighbors(nodes, age, family):
    a = nodes_checked(nodes)
    if not np.isfinite(age) or age < a[0] or age > a[-1]:
        raise ValueError("Interpolation only inside calibrated context interval")
    if family not in ("linear", "quadratic", "pchip"):
        raise ValueError("Unknown local interpolation family")
    exact = np.flatnonzero(a == age)
    if len(exact):
        return exact
    right = int(np.searchsorted(a, age))
    if family == "linear":
        return np.array([right - 1, right])
    if family == "quadratic":
        if len(a) < 3:
            raise ValueError("Three nodes needed for local quadratic")
        # Bracketing pair plus closest third; smaller-age tie break for the third.
        pair = [right - 1, right]
        third = next(i for i in np.lexsort((a, abs(a - age))) if i not in pair)
        return np.sort([*pair, third])
    # Bracketing interval plus one adjacent node on either side if available.
    return np.arange(max(0, right - 2), min(len(a), right + 2))


def interpolate(nodes, values, age, family):
    a = nodes_checked(nodes)
    v = np.asarray(values, float)
    if len(v) != len(a) or not np.isfinite(v).all():
        raise ValueError("One finite coefficient or prediction array per node")
    ii = neighbors(a, age, family)
    if len(ii) == 1:
        return v[ii[0]].copy()
    x, y = a[ii], v[ii]
    if family == "pchip":
        return PchipInterpolator(x, y, axis=0, extrapolate=False)(age)
    weights = np.array(
        [
            np.prod([(age - x[k]) / (x[j] - x[k]) for k in range(len(x)) if k != j])
            for j in range(len(x))
        ]
    )
    return np.tensordot(weights, y, axes=(0, 0))


def fit(x, ages, y, nodes, ridge=0.001):
    nodes = nodes_checked(nodes)
    ages = np.asarray(ages)
    if set(ages) != set(nodes):
        raise ValueError("Training ages must equal calibrated nodes; no held target")
    rep = om.representation(x, y, rank=4)
    coefficients = []
    for age in nodes:
        tr = ages == age
        model = om.fit(x[tr], ages[tr], y[tr], "blind", ridge, 4, rep)
        coefficients.append(model["coefficients"])
    return dict(**rep, nodes=nodes.tolist(), ridge=ridge, coefficients=coefficients)


def predict(model, centers, amplitude, age, family="linear", direct=False):
    """direct is the multi-evaluation prediction-space diagnostic, not truth."""
    if not np.isfinite(amplitude) or abs(amplitude) > 0.020000000001:
        raise ValueError("Qualified amplitude required")
    z = np.asarray(centers, float)
    if z.shape != (3,) or not np.isfinite(z).all():
        raise ValueError("Three finite measured centers required")
    phi = om.matrix(z[None], [age], model, "blind")[0]
    coef = np.asarray(model["coefficients"])
    if direct:
        ii = neighbors(model["nodes"], age, family)
        # Only the selected neighboring operators are evaluated.
        curves = np.array([om.reconstruct(phi @ coef[i], model, amplitude) for i in ii])
        if len(ii) == 1:
            return curves[0]
        return interpolate(np.asarray(model["nodes"])[ii], curves, age, family)
    weights = phi @ interpolate(model["nodes"], coef, age, family)
    return om.reconstruct(weights, model, amplitude)


def predict_panel(model, x, ages, family="linear", direct=False):
    return np.array(
        [
            [predict(model, z, a, age, family, direct) for a in (0.0, -0.02, 0.02)]
            for z, age in zip(x, ages, strict=True)
        ]
    )
