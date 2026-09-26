"""Present-state extraction, response-blind matching and small snapshot models."""

import numpy as np

from . import present_state_model as pm
from .measurements import regions

SETS = {
    "centers": [0, 1, 2],
    "overlap": [0, 1, 2, 3],
    "mediator": [0, 1, 2, 3, 4],
    "mass": [0, 1, 2, 3, 4, 5],
}


def extract(state, profile, length=192.0):
    state = np.asarray(state, float)
    profile = np.asarray(profile, float)
    if (
        state.ndim != 2
        or state.shape[0] != 2
        or profile.shape != state.shape[1:]
        or not np.isfinite(state).all()
        or not np.isfinite(profile).all()
    ):
        raise ValueError("Finite current field and static profile required")
    n = state.shape[1]
    x = np.arange(n) * length / n - length / 2
    w = regions(x, pm.ANCHORS)[0]
    dx = length / n
    return np.r_[
        pm.extract(state, length, "geometry"),
        np.sum(w * state[0] * profile) * dx,
        np.sum(w * state[1]) / np.sum(w),
        np.sum(w * state[0] ** 2) * dx,
    ]


def choose_pairs(descriptors, groups, uncertainty):
    x = np.asarray(descriptors, float)
    u = np.asarray(uncertainty, float)
    if x.ndim != 2 or x.shape[1] < 4 or len(groups) != len(x) or u.shape != (3,):
        raise ValueError("Descriptor/group shape mismatch")
    scale = np.maximum.reduce(
        [x[:, :3].std(axis=0), 20 * u, 10 * np.array([5e-5, 2e-5, 2e-6])]
    )
    rows = []
    for i in range(len(x)):
        for j in range(i):
            if groups[i] == groups[j]:
                continue
            delta = abs(x[i, :3] - x[j, :3])
            q = delta / scale
            distance = float(np.linalg.norm(q) / np.sqrt(3))
            separation = float(
                np.linalg.norm(delta / np.maximum(u, 1e-15)) / np.sqrt(3)
            )
            near = bool(distance <= 0.15 and q.max() <= 0.25 and separation > 5)
            rows.append(
                {
                    "indices": [j, i],
                    "distance": distance,
                    "normalized_max": float(q.max()),
                    "uncertainty_distance": separation,
                    "near": near,
                    "overlap_difference": float(abs(x[i, 3] - x[j, 3])),
                }
            )
    rows.sort(key=lambda r: (r["distance"], r["indices"]))
    selected = rows[:5]
    near = [r for r in rows if r["near"]]
    if near:
        extra = max(near, key=lambda r: r["overlap_difference"])
        if extra not in selected:
            selected.append(extra)
    return {
        "scale": scale.tolist(),
        "uncertainty": u.tolist(),
        "candidate_pairs": selected,
        "near_pair_count": len(near),
        "all_pairs": rows,
    }


def design(x, kind, width=None, centers=None):
    if kind == "quadratic":
        return pm.design(x, 2)
    if kind == "rbf":
        d = ((x[:, None] - np.asarray(centers)[None]) ** 2).sum(axis=-1)
        return np.exp(-d / (2 * width**2))
    raise ValueError("Unknown transparent predictor family")


def fit(x, y, feature_set, kind="quadratic", ridge=1e-6, width_factor=1.0, rank=4):
    """Only caller-supplied training rows; targets [none,negative,positive]."""
    x = np.asarray(x, float)[:, SETS[feature_set]]
    y = np.asarray(y, float)
    if y.shape != (len(x), 3, 161, 2) or not np.isfinite(y).all():
        raise ValueError("Matched training triplets required")
    mean = x.mean(axis=0)
    scale = np.maximum(x.std(axis=0), 1e-10)
    q = (x - mean) / scale
    r = y[:, 0]
    dm = y[:, 1] - r
    dp = y[:, 2] - r
    odd = (dp - dm) / 2
    even = (dp + dm) / 2
    scales = [
        np.maximum(np.sqrt(np.mean(v * v, axis=(0, 1))), 1e-18)
        for v in [r, np.concatenate([dm, dp])]
    ]
    bases = []
    weights = []
    for v, sy in [(r, scales[0]), (np.concatenate([dm, dp]), scales[1])]:
        _, _, vt = np.linalg.svd(
            (v / sy).transpose(0, 2, 1).reshape(-1, 161), full_matrices=False
        )
        bases.append(vt[:rank])
    for v, sy, basis in [
        (r, scales[0], bases[0]),
        (odd, scales[1], bases[1]),
        (even, scales[1], bases[1]),
    ]:
        weights.append(np.einsum("nto,kt->nok", v / sy, basis).reshape(len(x), -1))
    target = np.concatenate(weights, axis=1)
    model = {
        "schema": 1,
        "feature_set": feature_set,
        "kind": kind,
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "rank": rank,
        "bases": np.array(bases).tolist(),
        "scales": np.array(scales).tolist(),
        "ridge": ridge,
    }
    if kind == "rbf":
        d = np.linalg.norm(q[:, None] - q[None], axis=-1)
        nonzero = d[np.triu_indices(len(q), 1)]
        width = width_factor * np.median(nonzero[nonzero > 0])
        model.update(centers=q.tolist(), width=float(width))
        matrix = design(q, kind, width, q)
        coef = np.linalg.solve(matrix + ridge * np.eye(len(q)), target)
    else:
        matrix = design(q, kind)
        penalty = ridge * np.eye(matrix.shape[1])
        penalty[0, 0] = 0
        coef = np.linalg.solve(matrix.T @ matrix + penalty, matrix.T @ target)
    model["coefficients"] = coef.tolist()
    return model


def predict(model, descriptor, amplitude):
    if not np.isfinite(amplitude) or abs(amplitude) > 0.020000000001:
        raise ValueError("Qualified amplitude required")
    x = np.asarray(descriptor, float)[SETS[model["feature_set"]]]
    if x.shape != np.asarray(model["mean"]).shape or not np.isfinite(x).all():
        raise ValueError("Finite measured descriptor required")
    q = (x - np.asarray(model["mean"])) / model["scale"]
    matrix = design(q[None], model["kind"], model.get("width"), model.get("centers"))
    weights = (matrix @ np.asarray(model["coefficients"]))[0].reshape(
        3, 2, model["rank"]
    )
    bases = np.asarray(model["bases"])
    scales = np.asarray(model["scales"])
    b = amplitude / 0.02
    base = np.einsum("ok,kt->to", weights[0], bases[0]) * scales[0]
    event = (
        np.einsum("ok,kt->to", b * weights[1] + b * b * weights[2], bases[1])
        * scales[1]
    )
    return base + event
