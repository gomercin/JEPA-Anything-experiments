"""Small contextual response family. Runtime inputs: centers, amplitude, known age."""

import numpy as np

from . import present_state_model as pm


def age_features(age, family):
    a = np.asarray(age, float).reshape(-1)
    if not np.isfinite(a).all() or ((a < 10) | (a > 30)).any():
        raise ValueError("Known age must be within [10,30]")
    t = (a - 20) / 10
    if family == "blind":
        return np.ones((len(a), 1))
    if family == "polynomial":
        return np.column_stack([np.ones(len(a)), t, t * t])
    if family == "spline":
        # Natural cubic regression basis; fixed knots, no timing IDs or table.
        k = np.array([-1.0, -0.5, 0.0, 1.0])

        def d(j):
            return (np.maximum(t - k[j], 0) ** 3 - np.maximum(t - k[-1], 0) ** 3) / (
                k[-1] - k[j]
            )

        return np.column_stack([np.ones(len(a)), t, d(0) - d(2), d(1) - d(2)])
    raise ValueError("Unknown contextual family")


def representation(x, y, rank=4):
    """Training rows only, common across their ages; no labels in features."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.shape != (len(y), 3) or y.shape[1:] != (3, 161, 2):
        raise ValueError(
            "Matched centers and [zero,negative,positive] responses required"
        )
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("Finite training arrays required")
    r = y[:, 0]
    dm, dp = y[:, 1] - r, y[:, 2] - r
    vectors = [r, np.concatenate([dm, dp])]
    scales = [np.maximum(np.sqrt(np.mean(v * v, axis=(0, 1))), 1e-18) for v in vectors]
    bases = [
        np.linalg.svd((v / s).transpose(0, 2, 1).reshape(-1, 161), full_matrices=False)[
            2
        ][:rank]
        for v, s in zip(vectors, scales, strict=True)
    ]
    return {
        "schema": 1,
        "mean": x.mean(0).tolist(),
        "scale": np.maximum(x.std(0), 1e-10).tolist(),
        "rank": rank,
        "bases": np.array(bases).tolist(),
        "scales": np.array(scales).tolist(),
    }


def targets(y, rep):
    r = y[:, 0]
    dm, dp = y[:, 1] - r, y[:, 2] - r
    parts = [r, (dp - dm) / 2, (dp + dm) / 2]
    return np.concatenate(
        [
            np.einsum(
                "nto,kt->nok",
                v / rep["scales"][int(j > 0)],
                np.asarray(rep["bases"])[int(j > 0)],
            ).reshape(len(y), -1)
            for j, v in enumerate(parts)
        ],
        axis=1,
    )


def matrix(x, age, rep, family):
    q = (np.asarray(x, float) - rep["mean"]) / rep["scale"]
    return np.einsum("ni,nj->nij", pm.design(q, 2), age_features(age, family)).reshape(
        len(q), -1
    )


def fit(x, age, y, family="polynomial", ridge=1e-6, rank=4, rep=None):
    rep = representation(x, y, rank) if rep is None else rep
    m = matrix(x, age, rep, family)
    penalty = np.eye(m.shape[1]) * ridge
    penalty[0, 0] = 0
    coef = np.linalg.solve(m.T @ m + penalty, m.T @ targets(y, rep))
    return dict(**rep, family=family, ridge=ridge, coefficients=coef.tolist())


def reconstruct(weights, model, amplitude):
    w = np.asarray(weights).reshape(3, 2, model["rank"])
    b = amplitude / 0.02
    basis, scale = np.asarray(model["bases"]), np.asarray(model["scales"])
    return (w[0] @ basis[0]).T * scale[0] + (
        (b * w[1] + b * b * w[2]) @ basis[1]
    ).T * scale[1]


def predict(model, centers, amplitude, age):
    if not np.isfinite(amplitude) or abs(amplitude) > 0.020000000001:
        raise ValueError("Qualified conditioning amplitude required")
    z = np.asarray(centers, float)
    if z.shape != (3,) or not np.isfinite(z).all():
        raise ValueError("Three finite centers required")
    m = matrix(z[None], [age], model, model["family"])
    return reconstruct((m @ np.asarray(model["coefficients"]))[0], model, amplitude)


def predict_panel(model, x, ages):
    return np.array(
        [
            [predict(model, z, a, t) for a in [0.0, -0.02, 0.02]]
            for z, t in zip(x, ages, strict=True)
        ]
    )


def training_mask(groups, ages, held_group, held_age=None):
    """Exclude every descendant of a group and optionally an entire age."""
    mask = np.asarray(groups) != held_group
    if held_age is not None:
        mask &= np.asarray(ages) != held_age
    return mask
