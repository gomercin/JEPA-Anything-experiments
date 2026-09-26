"""One-snapshot extraction and ordinary regression. No reference-solver import."""

import json
from pathlib import Path

import numpy as np

from .measurements import regions

ANCHORS = np.array([-28.0, -4.0, 28.0])
TIMES = np.arange(161) * 0.5
SETS = {
    "blind": [],
    "b": [1],
    "geometry": [0, 1, 2],
    "shape": list(range(9)),
    "mediator": list(range(15)),
}
NAMES = [
    f"{kind}_{site}"
    for kind in ("offset", "mass", "width", "mmean", "mmoment")
    for site in "ABC"
]


def extract(state, length=192.0, feature_set="mediator"):
    """Only one current state; fixed apparatus. No metadata or paired fields."""
    state = np.asarray(state, dtype=float)
    if state.ndim != 2 or state.shape[0] != 2 or not np.isfinite(state).all():
        raise ValueError("Expected one finite current (2,N) snapshot")
    n = state.shape[1]
    x = np.arange(n) * length / n - length / 2
    w = regions(x, ANCHORS)
    d = x[None] - ANCHORS[:, None]
    density = w * state[0] ** 2
    mass = density.sum(axis=1) * length / n
    if (mass < 1e-12).any():
        raise ValueError("Pattern missing at a fixed anchor")
    offset = (density * d).sum(axis=1) * length / n / mass
    if feature_set in ("blind", "b", "geometry"):
        return offset[SETS[feature_set]]
    width = np.sqrt(
        (density * (d - offset[:, None]) ** 2).sum(axis=1) * length / n / mass
    )
    if feature_set == "shape":
        return np.concatenate([offset, mass, width])
    means = (w * state[1]).sum(axis=1) / w.sum(axis=1)
    moments = (w * state[1] * d / 8).sum(axis=1) / w.sum(axis=1)
    return np.concatenate([offset, mass, width, means, moments])[SETS[feature_set]]


def load_checkpoint(path, wait, index):
    """The boundary selector is bookkeeping; only the selected current array exits."""
    with np.load(path, allow_pickle=False) as saved:
        times = saved["checkpoint_times"]
        match = np.flatnonzero(times == wait)
        if len(match) != 1:
            raise ValueError("Exact boundary absent or ambiguous")
        result = saved["checkpoints"][match[0], index].copy()
    if result.ndim != 2 or result.shape[0] != 2 or not np.isfinite(result).all():
        raise ValueError("Invalid current checkpoint")
    return result


def design(z, degree):
    terms = [np.ones((len(z), 1)), z]
    if degree == 2:
        terms.append(
            np.stack(
                [
                    z[:, i] * z[:, j]
                    for i in range(z.shape[1])
                    for j in range(i, z.shape[1])
                ],
                axis=1,
            )
        )
    return np.concatenate(terms, axis=1)


def grouped_folds(groups):
    groups = np.asarray(groups)
    if len(np.unique(groups)) < 3:
        raise ValueError("At least three complete development preparations required")
    return [
        (np.flatnonzero(groups != g), np.flatnonzero(groups == g))
        for g in np.unique(groups)
    ]


def fit(z, y, pairs, feature_set, degree=1, ridge=1e-6, rank=4):
    """All supplied rows must be training rows. Pairs are training indices only."""
    z, y = np.asarray(z), np.asarray(y)
    if y.shape != (len(z), len(TIMES), 2) or not np.isfinite(y).all():
        raise ValueError("Training response shape mismatch")
    columns = SETS[feature_set]
    scale_y = np.sqrt(np.mean(y * y, axis=(0, 1)))
    if (scale_y <= 0).any():
        raise ValueError("Unresolved training readout")
    normalized = y / scale_y
    _, singular, vt = np.linalg.svd(
        normalized.transpose(0, 2, 1).reshape(-1, len(TIMES)), full_matrices=False
    )
    basis = vt[:rank]
    target = np.einsum("nto,kt->nok", normalized, basis).reshape(len(z), -1)
    x = z[:, columns]
    mean, scale = x.mean(axis=0), x.std(axis=0)
    scale = np.maximum(scale, 1e-14)
    matrix = design((x - mean) / scale, degree if columns else 1)
    a, b = matrix, target
    if pairs and columns:
        left, right = np.asarray(pairs).T
        a = np.concatenate([a, 10 * (matrix[right] - matrix[left])])
        b = np.concatenate([b, 10 * (target[right] - target[left])])
    penalty = np.eye(a.shape[1]) * ridge
    penalty[0, 0] = 0
    coefficients = np.linalg.solve(a.T @ a + penalty, a.T @ b)
    projection = (
        np.einsum("nok,kt->nto", target.reshape(len(z), 2, -1), basis) * scale_y
    )
    return {
        "schema": 1,
        "feature_set": feature_set,
        "columns": columns,
        "names": [NAMES[i] for i in columns],
        "anchors": ANCHORS.tolist(),
        "probe": {"kind": "fixed_A_additive_even_unit_L2", "amplitude": 0.02},
        "times": TIMES.tolist(),
        "degree": degree if columns else 1,
        "ridge": ridge,
        "rank": len(basis),
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "scale_y": scale_y.tolist(),
        "basis": basis.tolist(),
        "coefficients": coefficients.tolist(),
        "training_rows": len(z),
        "retained_training_examples": 0,
        "condition": float(np.linalg.cond(a)),
        "singular_values": singular.tolist(),
        "projection_relative_rms": (
            np.sqrt(np.mean((projection - y) ** 2, axis=1))
            / np.sqrt(np.mean(y * y, axis=1))
        ).tolist(),
    }


def predict(model, z, probe=0.02, times=TIMES):
    """Descriptors, static data, declared probe/time only; no evaluator access."""
    if probe != model["probe"]["amplitude"]:
        raise ValueError("Probe outside the frozen assay")
    times = np.asarray(times)
    if times.ndim != 1 or not np.isin(times, model["times"]).all():
        raise ValueError("Forecast times outside the declared sample grid")
    z = np.asarray(z, dtype=float)
    if z.ndim != 2 or z.shape[1] != len(model["columns"]) or not np.isfinite(z).all():
        raise ValueError("Expected serialized descriptor rows only")
    x = (z - model["mean"]) / model["scale"]
    coeff = design(x, model["degree"]) @ np.asarray(model["coefficients"])
    basis = np.asarray(model["basis"])[:, np.searchsorted(model["times"], times)]
    return (
        np.einsum("nok,kt->nto", coeff.reshape(len(z), 2, -1), basis) * model["scale_y"]
    )


def save_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def scores(truth, prediction, pairs, metadata, floors=None):
    """R and Delta_R use their own denominators, never background/write offsets."""
    floors = np.zeros((2, 2)) if floors is None else np.asarray(floors)
    result = []
    comparisons = [("R", i, truth[i], prediction[i]) for i in range(len(truth))]
    comparisons += [
        ("Delta_R", j, truth[j] - truth[i], prediction[j] - prediction[i])
        for i, j in pairs
    ]
    for kind, i, actual, pred in comparisons:
        for k, mask in enumerate([TIMES >= 0, TIMES >= 40]):
            residual = pred[mask] - actual[mask]
            rms = np.sqrt(np.mean(actual[mask] ** 2, axis=0))
            error = np.sqrt(np.mean(residual**2, axis=0))
            for o, name in enumerate(["mass", "moment"]):
                resolved = bool(rms[o] > floors[k, o])
                relative = float(error[o] / rms[o]) if rms[o] else None
                result.append(
                    dict(
                        **metadata[i],
                        kind=kind,
                        window=["whole", "late"][k],
                        output=name,
                        rms=float(rms[o]),
                        residual_rms=float(error[o]),
                        residual_max=float(abs(residual[:, o]).max()),
                        relative_rms=relative,
                        floor=float(floors[k, o]),
                        resolved=resolved,
                        passed=bool(
                            resolved and relative <= (0.02 if kind == "R" else 0.10)
                        ),
                    )
                )
    return result
