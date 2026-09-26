"""Delayed probe evaluator, prediction seals and matched refinement."""

import time
from dataclasses import replace

import numpy as np
from scipy.signal import resample

from . import geometry_model as gm
from . import present_state_model as pm
from .geometry_evolution import (
    CFG,
    F_FILE,
    F_SHA,
    FRESH,
    digest,
    frozen_response,
    load,
    unforced,
)
from .history_conditioned_transmission import write_profile
from .hybrid_pair import save_npz_exclusive
from .organization_response import isolated_components
from .present_state_transmission import old_initial, reference
from .simulator import Field
from .source_receiver_relay import prepare

FAMILIES = {"none": [0.0, 0.0], "odd04": [0.4, 0.0], "heldmix": [0.2, -0.2]}
ALL_WRITES = dict(FAMILIES, negative=[-0.4, 0.0], mixed=[0.28, 0.28])


def initialize_written(initial, family, config):
    odd, even = ALL_WRITES[family]
    f = Field(config)
    q = odd * write_profile(f, "odd") + even * write_profile(f, "even")
    if np.sqrt(np.sum(q * q) * config.length / config.n) > 0.4 + 1e-12:
        raise ValueError("Write norm exceeds contract")
    state = initial.copy()
    state[0] += q
    return state


def forecasts(out, starts, models, origin, times):
    """Only serialized descriptors/static models enter this routine."""
    F = frozen_response()
    zhat = {k: [] for k in models}
    pred = {k: [] for k in models}
    start = time.process_time()
    for name, g in models.items():
        for z in starts:
            zhat[name].append(gm.rollout(g, z, np.array(times) - origin))
        zhat[name] = np.asarray(zhat[name])
        pred[name] = pm.predict(F, zhat[name].reshape(-1, 3)).reshape(
            len(starts), len(times), 161, 2
        )
    save_npz_exclusive(out / f"predictions-{origin}.npz", **pred)
    save_npz_exclusive(out / f"geometry-{origin}.npz", **zhat)
    # Each row resumes in a newly instantiated predictor, from serialized state.
    agreements = []
    for name, g in models.items():
        for i, z in enumerate(starts):
            state = gm.Continuation(g, z)
            midpoint = int((times[-1] - origin) / g["step"]) // 2
            state.advance(midpoint)
            path = out / f"checkpoint-{origin}-{name}-{i}.json"
            state.checkpoint(path)
            resumed = gm.Continuation.restore(g, load(path))
            endpoint = resumed.advance(
                round((times[-1] - origin) / g["step"]) - midpoint
            )
            agreements.append(float(abs(endpoint - zhat[name][i, -1]).max()))
    pm.save_json(
        out / f"seal-{origin}.json",
        {
            "initial_descriptors_sha256": digest(
                out / f"initial-descriptors-{origin}.npz"
            ),
            "predictions_sha256": digest(out / f"predictions-{origin}.npz"),
            "geometry_sha256": digest(out / f"geometry-{origin}.npz"),
            "response_model_sha256": digest(F_FILE),
            "updaters_sha256": {k: gm.identity(v) for k, v in models.items()},
            "cpu_seconds": time.process_time() - start,
            "max_resume_discrepancy": max(agreements),
            "order": "Saved all predictions at this initialization boundary before its future unforced continuation or probe references",
        },
    )


def fresh(out, frozen, budget):
    contract = load(frozen / "freeze.json")
    models = load(frozen / "models.json")
    if (
        contract["models_sha256"] != digest(frozen / "models.json")
        or contract["F_sha256"] != F_SHA
    ):
        raise ValueError("Frozen model identity changed")
    if contract["fresh_seeds"] != list(FRESH) or contract["families"] != FAMILIES:
        raise ValueError("Frozen preparation contract changed")
    starts, rows = [], []
    for seed in FRESH:
        f = Field(CFG)
        initial = prepare(f, isolated_components(f, seed, budget), seed, -4, budget)
        save_npz_exclusive(out / f"s{seed}-initial.npz", initial=initial)
        for family in FAMILIES:
            budget.begin(f"pre-boundary-{seed}-{family}")
            state = initialize_written(initial, family, CFG)
            history, screens = unforced(state, CFG, [0.0, 50.0], budget)
            starts.append(history[-1])
            rows.append({"seed": seed, "history": family})
            pm.save_json(
                out / f"pre-boundary-{seed}-{family}.json", {"regime": screens}
            )
            budget.finish()
    pm.save_json(out / "rows.json", rows)
    # A boundary input object contains no targets, rates, or hidden metadata.
    starts = np.array(starts)
    save_npz_exclusive(out / "boundary-50.npz", states=starts)
    z50 = np.array([pm.extract(s, feature_set="geometry") for s in starts])
    save_npz_exclusive(out / "initial-descriptors-50.npz", z=z50)
    forecasts(out, z50, models, 50, contract["times"]["50"])
    # Only now may future state arrays exist; collect shifted origin before its future.
    shifted = []
    for i, state in enumerate(starts):
        budget.begin(f"50-to-60-{i}")
        states, screens = unforced(state, CFG, [0.0, 10.0], budget)
        shifted.append(states[-1])
        pm.save_json(out / f"50-to-60-{i}.json", {"regime": screens})
        budget.finish()
    shifted = np.array(shifted)
    save_npz_exclusive(out / "boundary-60.npz", states=shifted)
    z60 = np.array([pm.extract(s, feature_set="geometry") for s in shifted])
    save_npz_exclusive(out / "initial-descriptors-60.npz", z=z60)
    forecasts(out, z60, models, 60, contract["times"]["60"])
    all_states = []
    target_times = [60.0, 75.0, 90.0, 100.0]
    for i, state in enumerate(shifted):
        budget.begin(f"60-to-100-{i}")
        states, screens = unforced(state, CFG, np.array(target_times) - 60.0, budget)
        all_states.append(states)
        pm.save_json(out / f"60-to-100-{i}.json", {"regime": screens})
        budget.finish()
    all_states = np.array(all_states)
    save_npz_exclusive(
        out / "evaluator-states.npz", time=target_times, states=all_states
    )
    ztrue = np.array(
        [
            [pm.extract(s, feature_set="geometry") for s in states]
            for states in all_states
        ]
    )
    save_npz_exclusive(out / "evaluator-geometry.npz", time=target_times, z=ztrue)
    y = []
    for i, states in enumerate(all_states):
        targets = []
        for ti, state in enumerate(states):
            budget.begin(f"reference-{i}-{target_times[ti]:g}")
            arrays, screens = reference(state, CFG, budget)
            save_npz_exclusive(
                out / f"reference-{i}-{target_times[ti]:g}.npz", **arrays
            )
            pm.save_json(
                out / f"reference-{i}-{target_times[ti]:g}.json",
                {"qualified": all(s["qualified"] for s in screens)},
            )
            targets.append(arrays["response"])
            budget.finish()
        y.append(targets)
    save_npz_exclusive(out / "targets.npz", time=target_times, y=y)


def refine(out, data, seed, family, target, budget, is_fresh=False):
    if is_fresh:
        initial_path = data / f"s{seed}-initial.npz"
        with np.load(initial_path, allow_pickle=False) as a:
            initial = a["initial"]
        rows = load(data / "rows.json")
        with np.load(data / "evaluator-geometry.npz", allow_pickle=False) as a:
            base_z = a["z"]
            times = a["time"]
        with np.load(data / "targets.npz", allow_pickle=False) as a:
            base_y = a["y"]
    else:
        initial = old_initial(seed)
        rows = load(data / "rows.json")
        with np.load(data / "trajectories.npz", allow_pickle=False) as a:
            base_z = a["z"]
            times = a["time"]
        rr = load(data / "response-rows.json")
        with np.load(data / "targets.npz", allow_pickle=False) as a:
            targets = a["y"]
        base_y = np.zeros((len(rows), len(times), 161, 2))
        for r, y in zip(rr, targets, strict=True):
            base_y[
                rows.index({"seed": r["seed"], "history": r["history"]}),
                list(times).index(r["wait"]),
            ] = y
    # Full-history refinement shares the prepared initial field, not a later reset.
    save_npz_exclusive(out / "prepared-initial.npz", initial=initial)
    panel_times = sorted({50.0, 60.0, target})
    y0 = []
    zi = []
    index = list(times).index(target)
    for history in ["none", family]:
        i = rows.index({"seed": seed, "history": history})
        y0.append(base_y[i, index])
        values = []
        for t in panel_times:
            if is_fresh and t == 50:
                with np.load(
                    data / "initial-descriptors-50.npz", allow_pickle=False
                ) as a:
                    values.append(a["z"][i])
            else:
                values.append(base_z[i, list(times).index(t)])
        zi.append(values)
    y0 = np.array(y0)
    zi = np.array(zi)
    records = []
    for label, config in [
        ("halfdt", replace(CFG, dt=CFG.dt / 2)),
        ("doubleN", replace(CFG, n=2 * CFG.n)),
    ]:
        ys, zs = [], []
        for history in ["none", family]:
            budget.begin(f"{label}-{history}-history")
            init = (
                initial if config.n == CFG.n else resample(initial, config.n, axis=-1)
            )
            state = initialize_written(init, history, config)
            states, screens = unforced(state, config, [0.0, *panel_times], budget)
            zs.append([pm.extract(s, config.length, "geometry") for s in states[1:]])
            save_npz_exclusive(out / f"{label}-{history}-current.npz", state=states[-1])
            pm.save_json(out / f"{label}-{history}-regime.json", {"regime": screens})
            budget.finish()
            budget.begin(f"{label}-{history}-probe")
            arrays, screens = reference(states[-1], config, budget)
            ys.append(arrays["response"])
            save_npz_exclusive(out / f"{label}-{history}-reference.npz", **arrays)
            pm.save_json(
                out / f"{label}-{history}-probe.json",
                {"qualified": all(s["qualified"] for s in screens)},
            )
            budget.finish()
        errors = np.array(ys) - y0
        floors = []
        for mask in [pm.TIMES >= 0, pm.TIMES >= 40]:
            floors.append(
                (
                    5 * np.sqrt(np.mean(errors[:, mask] ** 2, axis=1)).sum(axis=0)
                ).tolist()
            )
        records.append(
            {
                "refinement": label,
                "response_floor": floors,
                "coordinate_floor": (
                    5 * abs(np.array(zs) - zi).max(axis=(0, 1))
                ).tolist(),
                "coordinate_difference": (np.array(zs) - zi).tolist(),
                "contrast_residual_rms": np.sqrt(
                    np.mean((errors[1] - errors[0]) ** 2, axis=0)
                ).tolist(),
            }
        )
    pm.save_json(
        out / "refinement.json",
        {
            "seed": seed,
            "history": family,
            "target": target,
            "coordinate_times": panel_times,
            "prepared_initial_sha256": digest(out / "prepared-initial.npz"),
            "records": records,
            "response_floors": np.maximum(
                1e-12, np.max([r["response_floor"] for r in records], axis=0)
            ).tolist(),
            "coordinate_floors": np.maximum(
                1e-10, np.max([r["coordinate_floor"] for r in records], axis=0)
            ).tolist(),
        },
    )
