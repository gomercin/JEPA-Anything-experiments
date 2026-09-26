"""Freeze and score unforced continuation; nonlinear truth stays evaluator-only."""

import time
from pathlib import Path

import numpy as np

from . import geometry_model as gm
from . import present_state_model as pm
from .geometry_evolution import (
    F_FILE,
    F_SHA,
    FRESH,
    ROOT,
    SAMPLES,
    digest,
    frozen_response,
    load,
    response_pairs,
)
from .hybrid_pair import save_npz_exclusive

COORDINATE_TOLERANCES = np.array([2e-4, 2e-4, 5e-5])


def shared_floors(refinements):
    response = np.full((2, 2), 1e-12)
    coordinate = np.full(3, 1e-10)
    for path in refinements:
        r = load(path / "refinement.json")
        response = np.maximum(response, r["response_floors"])
        coordinate = np.maximum(coordinate, r["coordinate_floors"])
    return response, coordinate


def freeze(out, fit, refinements, budget):
    budget.begin("freeze-and-development-geometry-audit")
    report = load(fit / "fit.json")
    all_models = load(fit / "models.json")
    selected = report["selected"]
    models = {
        k: all_models[k]
        for k in dict.fromkeys(["persistence", "drift", "affine-1e-06", selected])
    }
    pm.save_json(out / "models.json", models)
    floors, coordinate_floor = shared_floors(refinements)
    # Development physical-coordinate gate, including relative-to-motion error.
    data = ROOT / "development-01"
    rows = load(data / "rows.json")
    with np.load(data / "trajectories.npz", allow_pickle=False) as f:
        z = f["z"]
    with np.load(fit / "rollouts.npz", allow_pickle=False) as f:
        predicted = f[selected]
    geometry = []
    step_differences = []
    for i, row in enumerate(rows):
        for b, start in enumerate([0, 2]):
            mask = ((SAMPLES <= 75) | (SAMPLES == 100)) & (SAMPLES > SAMPLES[start])
            error = predicted[i, b, mask] - z[i, mask]
            motion = z[i, mask] - z[i, start]
            geometry.append(
                dict(
                    **row,
                    origin=float(SAMPLES[start]),
                    max_error=abs(error).max(axis=0).tolist(),
                    relative_movement_rms=(
                        np.sqrt(np.mean(error**2, axis=0))
                        / np.sqrt(np.mean(motion**2, axis=0))
                    ).tolist(),
                )
            )
            g = models[selected]
            refined = dict(g, step=g["step"] / 2)
            step_differences.append(
                abs(
                    gm.rollout(g, z[i, start], [100 - SAMPLES[start]])
                    - gm.rollout(refined, z[i, start], [100 - SAMPLES[start]])
                ).max()
            )
    pm.save_json(
        out / "development-geometry.json",
        {"rows": geometry, "half_step_max_difference": float(max(step_differences))},
    )
    from .geometry_assay import FAMILIES

    pm.save_json(
        out / "freeze.json",
        {
            "selected": selected,
            "models_sha256": digest(out / "models.json"),
            "F_path": str(F_FILE),
            "F_sha256": F_SHA,
            "fit": str(fit),
            "fit_sha256": digest(fit / "fit.json"),
            "sources": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
            "fresh_seeds": list(FRESH),
            "families": FAMILIES,
            "times": {"50": [60, 75, 90, 100], "60": [75, 90, 100]},
            "coordinate_tolerances": COORDINATE_TOLERANCES.tolist(),
            "relative_movement_limit": 0.1,
            "coordinate_floors": coordinate_floor.tolist(),
            "response_floors": floors.tolist(),
            "response_limits": {"R": 0.02, "Delta_R": 0.1},
            "late_window": [40, 80],
            "floor_rule": "max(1e-12, 5*max_refinement(sum separate history response RMS errors)), shared per readout/window; coordinate max(1e-10,5*max absolute matched center discrepancy); movement floor twice coordinate floor",
            "withheld_interval": "Interior (75,100): no development rates/coordinates there enter fit/scaling/selection; fresh target90 is interpolation, not a new regime",
            "no_fresh_repair": True,
            "no_response_refit": True,
        },
    )
    budget.finish()


def error_terms(prediction, snapshot, truth):
    """Signed identity, with cross term retained; errors do not add in RMS."""
    update = np.asarray(prediction) - snapshot
    readout = np.asarray(snapshot) - truth
    return update, readout


def analyze(out, data, frozen, refinements, budget):
    budget.begin("scoring-and-figures")
    contract = load(frozen / "freeze.json")
    selected = contract["selected"]
    models = load(frozen / "models.json")
    floors, coordinate_floor = shared_floors(refinements)
    floors = np.maximum(floors, contract["response_floors"])
    coordinate_floor = np.maximum(coordinate_floor, contract["coordinate_floors"])
    base_rows = load(data / "rows.json")
    with np.load(data / "targets.npz", allow_pickle=False) as a:
        truth_all = a["y"]
        times = a["time"]
    with np.load(data / "evaluator-geometry.npz", allow_pickle=False) as a:
        ztrue_all = a["z"]
    rows, truth, ztrue, zinitial = [], [], [], []
    preds = {k: [] for k in models}
    zhats = {k: [] for k in models}
    geometry = []
    for origin in [50, 60]:
        with np.load(
            data / f"initial-descriptors-{origin}.npz", allow_pickle=False
        ) as a:
            z0 = a["z"]
        with np.load(data / f"predictions-{origin}.npz", allow_pickle=False) as a:
            pp = dict(a)
        with np.load(data / f"geometry-{origin}.npz", allow_pickle=False) as a:
            zz = dict(a)
        for i, row in enumerate(base_rows):
            indices = [list(times).index(t) for t in contract["times"][str(origin)]]
            movement = ztrue_all[i, indices] - z0[i]
            motion_rms = np.sqrt(np.mean(movement**2, axis=0))
            for name in models:
                residual = zz[name][i] - ztrue_all[i, indices]
                err_rms = np.sqrt(np.mean(residual**2, axis=0))
                resolved = motion_rms > 2 * coordinate_floor
                ratio = np.divide(
                    err_rms, motion_rms, out=np.full(3, np.nan), where=motion_rms > 0
                )
                geometry.append(
                    dict(
                        **row,
                        origin=origin,
                        model=name,
                        actual_displacements=movement.tolist(),
                        predicted_displacements=(zz[name][i] - z0[i]).tolist(),
                        max_error=abs(residual).max(axis=0).tolist(),
                        rms_error=err_rms.tolist(),
                        rms_movement=motion_rms.tolist(),
                        relative_movement_rms=[
                            float(x) if np.isfinite(x) else None for x in ratio
                        ],
                        movement_resolved=resolved.tolist(),
                        passed=(
                            resolved
                            & (ratio <= contract["relative_movement_limit"])
                            & (
                                abs(residual).max(axis=0)
                                <= contract["coordinate_tolerances"]
                            )
                        ).tolist(),
                    )
                )
            for ti, index in enumerate(indices):
                rows.append(dict(**row, origin=origin, wait=float(times[index])))
                truth.append(truth_all[i, index])
                ztrue.append(ztrue_all[i, index])
                zinitial.append(z0[i])
                for name in models:
                    preds[name].append(pp[name][i, ti])
                    zhats[name].append(zz[name][i, ti])
    truth, ztrue, zinitial = np.array(truth), np.array(ztrue), np.array(zinitial)
    preds = {k: np.asarray(v) for k, v in preds.items()}
    zhats = {k: np.asarray(v) for k, v in zhats.items()}
    snapshot = pm.predict(frozen_response(), ztrue)
    pairs = response_pairs(rows)
    scores = {
        k: pm.scores(truth, p, pairs, rows, floors)
        for k, p in dict(snapshot=snapshot, **preds).items()
    }
    terms = []
    signed = {
        "truth": truth,
        "snapshot": snapshot,
        "ztrue": ztrue,
        "zinitial": zinitial,
    }
    for name, pred in preds.items():
        u, r = error_terms(pred, snapshot, truth)
        signed[name] = pred
        signed[f"update_{name}"] = u
        signed[f"readout_{name}"] = r
        signed[f"z_{name}"] = zhats[name]
        for kind, indices in [
            ("R", [(None, i) for i in range(len(rows))]),
            ("Delta_R", pairs),
        ]:
            for left, right in indices:
                uu = u[right] if left is None else u[right] - u[left]
                rr = r[right] if left is None else r[right] - r[left]
                total = uu + rr
                for label, mask in [("whole", pm.TIMES >= 0), ("late", pm.TIMES >= 40)]:
                    for j, output in enumerate(["mass", "moment"]):
                        terms.append(
                            dict(
                                **rows[right],
                                model=name,
                                kind=kind,
                                window=label,
                                output=output,
                                update_mean=float(np.mean(uu[mask, j])),
                                readout_mean=float(np.mean(rr[mask, j])),
                                update_mse=float(np.mean(uu[mask, j] ** 2)),
                                readout_mse=float(np.mean(rr[mask, j] ** 2)),
                                cross_term=float(
                                    2 * np.mean(uu[mask, j] * rr[mask, j])
                                ),
                                total_mse=float(np.mean(total[mask, j] ** 2)),
                            )
                        )
    worst = {
        name: {
            f"{kind}_{output}": max(
                (r for r in score if r["kind"] == kind and r["output"] == output),
                key=lambda r: r["relative_rms"],
            )
            for kind in ["R", "Delta_R"]
            for output in ["mass", "moment"]
        }
        for name, score in scores.items()
    }
    per_preparation = {
        str(seed): {
            k: max(
                r["relative_rms"]
                for r in scores[selected]
                if r["seed"] == seed and r["kind"] == kind and r["output"] == output
            )
            for kind in ["R", "Delta_R"]
            for output in ["mass", "moment"]
            for k in [kind + "_" + output]
        }
        for seed in FRESH
    }
    pm.save_json(out / "rows.json", rows)
    save_npz_exclusive(out / "signed-predictions.npz", **signed)
    pm.save_json(
        out / "summary.json",
        {
            "selected": selected,
            "scores": scores,
            "geometry": geometry,
            "decomposition": terms,
            "worst": worst,
            "per_preparation": per_preparation,
            "response_floors": floors.tolist(),
            "coordinate_floors": coordinate_floor.tolist(),
            "max_identity_discrepancy": float(
                max(
                    abs(
                        (preds[k] - truth)
                        - (signed[f"update_{k}"] + signed[f"readout_{k}"])
                    ).max()
                    for k in preds
                )
            ),
            "all_response_gates": all(r["passed"] for r in scores[selected]),
            "all_geometry_gates": all(
                all(r["passed"]) for r in geometry if r["model"] == selected
            ),
        },
    )
    plot(
        out,
        data,
        models,
        selected,
        rows,
        truth,
        snapshot,
        preds,
        worst[selected]["Delta_R_mass"],
    )
    # Measured costs for this exact small workload, charged to this stage.
    with np.load(data / "boundary-50.npz", allow_pickle=False) as a:
        state = a["states"][0]
    started = time.process_time()
    z0 = pm.extract(state, feature_set="geometry")
    extraction = time.process_time() - started
    started = time.process_time()
    rollout = gm.rollout(models[selected], z0, [10, 25, 40, 50])
    updating = time.process_time() - started
    started = time.process_time()
    pm.predict(frozen_response(), rollout)
    forecast = time.process_time() - started
    pm.save_json(
        out / "accounting.json",
        {
            "initial_field_values": 1536,
            "initial_field_bytes": state.nbytes,
            "initial_full_field_scans": "finite check over both arrays; three weighted u-squared window reductions over u; no rate or FFT at inference",
            "dynamic_coordinates": 3,
            "dynamic_bytes": 24,
            "step_counter_scalars": 1,
            "integrator_step": models[selected]["step"],
            "updater_learned_scalars": 6
            + int(np.size(models[selected]["coefficients"])),
            "response_learned_scalars": 732,
            "retained_training_examples": 0,
            "spatial_templates": 0,
            "response_model_file_bytes": F_FILE.stat().st_size,
            "updater_file_bytes": (frozen / "models.json").stat().st_size,
            "extraction_cpu_seconds": extraction,
            "update_50_cpu_seconds": updating,
            "response_four_delays_cpu_seconds": forecast,
            "update_rhs_calls_for_50": 200,
            "training_rate_access": "Full current u,m plus periodic FFT RHS and three weighted reductions at selected development boundaries only",
            "timing_note": "Single-workload process timings; not a whole-system speed benchmark. Model-file sizes include controls and metadata.",
        },
    )
    budget.finish()


def plot(out, data, models, selected, rows, truth, snapshot, preds, worst):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    i = next(i for i, r in enumerate(rows) if all(r[k] == worst[k] for k in r))
    j = next(
        j
        for j, r in enumerate(rows)
        if r["seed"] == worst["seed"]
        and r["history"] == "none"
        and r["origin"] == worst["origin"]
        and r["wait"] == worst["wait"]
    )
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for o, output in enumerate(["C mass", "C signed moment"]):
        ax = axes[0, o]
        for index, label, color in [
            (j, "unwritten", "#35679a"),
            (i, worst["history"], "#d0792b"),
        ]:
            ax.plot(pm.TIMES, truth[index, :, o], color=color, label=label + " actual")
            ax.plot(
                pm.TIMES,
                preds[selected][index, :, o],
                "--",
                color=color,
                label=label + " updated",
            )
            ax.plot(
                pm.TIMES,
                preds["persistence"][index, :, o],
                ":",
                color=color,
                label=label + " persistence",
            )
        ax.set_title(output + " response R")
        ax.set_xlabel("h since hypothetical probe")
        ax = axes[1, o]
        for values, label, style, color in [
            (truth, "actual", "-", "#222222"),
            (preds[selected], "updated", "--", "#b04778"),
            (preds["persistence"], "persistence", ":", "#c18e2d"),
            (snapshot, "exact-center diagnostic", "-.", "#2a8b80"),
        ]:
            ax.plot(
                pm.TIMES,
                values[i, :, o] - values[j, :, o],
                style,
                color=color,
                label=label,
            )
        ax.set_title(output + " contrast Delta_R")
        ax.set_xlabel("h since hypothetical probe")
    axes[0, 0].legend(fontsize=8)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(
        f"Worst updated mass contrast: seed {worst['seed']}, {worst['history']}, {worst['origin']} to {worst['wait']:g}"
    )
    fig.savefig(out / "responses.png", dpi=150)
    plt.close(fig)
    with np.load(data / "initial-descriptors-50.npz", allow_pickle=False) as a:
        z0 = a["z"]
    with np.load(data / "evaluator-geometry.npz", allow_pickle=False) as a:
        z = a["z"]
        times = a["time"]
    base_rows = load(data / "rows.json")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), constrained_layout=True)
    for index, r in enumerate(base_rows):
        if r["seed"] != worst["seed"]:
            continue
        colors = {"none": "#35679a", "odd04": "#d0792b", "heldmix": "#6c589c"}
        prediction = gm.rollout(models[selected], z0[index], np.arange(51.0))
        for c, ax in enumerate(axes):
            ax.plot(
                np.r_[0, times - 50],
                np.r_[0, z[index, :, c] - z0[index, c]] * 1000,
                "o",
                color=colors[r["history"]],
                label=r["history"] + " actual",
            )
            ax.plot(
                np.arange(51),
                1000 * (prediction[:, c] - z0[index, c]),
                color=colors[r["history"]],
                label=r["history"] + " updated",
            )
            ax.axhline(0, color="#777777", linestyle=":", linewidth=0.8)
            ax.set_title("ABC"[c] + " center displacement")
            ax.set_xlabel("unobserved delay from 50")
            ax.set_ylabel("physical units × 1,000")
    axes[0].legend(fontsize=7)
    fig.suptitle(
        f"True center motion and uninterrupted three-coordinate update, seed {worst['seed']}; persistence = 0"
    )
    fig.savefig(out / "geometry.png", dpi=150)
    plt.close(fig)
