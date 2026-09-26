"""Grouped fixed-age snapshot fitting and sealed evaluation, never runtime truth."""

import numpy as np

from . import event_state_model as sm
from . import geometry_model as gm
from . import present_state_model as pm
from .event_age_response import WINDOWS, arrays, rms, score
from .event_state_aliasing import FRESH, ROOT, descriptor, reach_boundary, response_pair
from .geometry_assay import initialize_written
from .geometry_evolution import unforced
from .hybrid_pair import save_npz_exclusive as save_npz
from .present_state_transmission import digest, load
from .repeated_intervention_state import source_hashes
from .simulator import Field
from .source_receiver_relay import CFG


def development():
    base = ROOT / "boundaries-01"
    return (
        arrays(base / "descriptors.npz")["x"],
        arrays(ROOT / "responses-01/targets.npz")["y"],
        load(base / "rows.json"),
    )


def gates(y, prediction, rows, floors):
    result = []
    for i, row in enumerate(rows):
        for j, a in [(1, -0.02), (2, 0.02)]:
            for r in score(y[i, [0, j]], prediction[i, [0, j]], floors):
                result.append(
                    dict(**r, index=i, seed=row["seed"], history=row["history"], a=a)
                )
    return result


def summary(records):
    unique = []
    for r in records:
        if r["kind"] != "R_without" or r["a"] == -0.02:
            unique.append(r)
    return {
        "distinct_gates": len(unique),
        "failed": sum(r["passed"] is False for r in unique),
        "unresolved": sum(r["passed"] is None for r in unique),
        "worst_gate_ratio": max(
            (
                r["relative"] / (0.1 if r["kind"] == "D1" else 0.02)
                for r in unique
                if r["resolved"]
            ),
            default=0.0,
        ),
        "by_kind": {
            kind: {
                "failed": sum(
                    r["passed"] is False for r in unique if r["kind"] == kind
                ),
                "worst_relative": max(
                    (
                        r["relative"]
                        for r in unique
                        if r["kind"] == kind and r["resolved"]
                    ),
                    default=0.0,
                ),
            }
            for kind in ["R_without", "R_after", "D1"]
        },
    }


def prediction(model, x):
    return np.array([[sm.predict(model, v, a) for a in [0.0, -0.02, 0.02]] for v in x])


def variations(model, x, selected):
    """Absolute line-integral variation; empirical bound, not a sufficiency theorem."""
    values = []
    for pair in selected:
        i, j = pair["indices"]
        path = x[i][None] + np.linspace(0, 1, 17)[:, None] * (x[j] - x[i])[None]
        pp = prediction(model, path)
        d = pp[:, 1:] - pp[:, :1]
        integral = abs(np.diff(d, axis=0)).sum(axis=0)
        values.append(
            np.array([rms(integral[:, sl], axis=1) for sl in WINDOWS.values()])
        )
    return np.array(values)


def fit(out, args, budget):
    x, y, rows = development()
    groups = np.array([r["seed"] for r in rows])
    floors = load(ROOT / "refinement-01/refinement.json")["floors"]
    pairs = load(ROOT / "boundaries-01/selection.json")["candidate_pairs"]
    configs = (
        [
            {"kind": "rbf", "ridge": ridge, "width_factor": width}
            for width in [0.5, 1.0, 2.0]
            for ridge in [1e-6, 1e-3]
        ]
        if args.kernel
        else [{"kind": "quadratic", "ridge": ridge} for ridge in [1e-6, 1e-3, 1e-1]]
    )
    results = []
    for ci, config in enumerate(configs):
        budget.begin(f"grouped-{args.feature_set}-{ci}")
        pred = np.zeros_like(y)
        sensitivity = np.zeros((len(pairs), 2, 2, 2))
        folds = []
        for group in sorted(set(groups)):
            train = groups != group
            test = ~train
            m = sm.fit(x[train], y[train], args.feature_set, **config)
            pred[test] = prediction(m, x[test])
            if args.feature_set == "centers":
                sensitivity = np.maximum(sensitivity, variations(m, x, pairs))
            folds.append(
                {
                    "held_group": int(group),
                    "training_groups": sorted(set(groups[train].tolist())),
                    "mean": m["mean"],
                    "scale": m["scale"],
                    "model_sha256": gm.identity(m),
                }
            )
            budget.check()
        m = sm.fit(x, y, args.feature_set, **config)
        scores = gates(y, pred, rows, floors)
        pm.save_json(out / f"model-{ci}.json", m)
        pm.save_json(out / f"scores-{ci}.json", scores)
        pm.save_json(out / f"folds-{ci}.json", folds)
        save_npz(out / f"predictions-{ci}.npz", y=pred, sensitivity=sensitivity)
        result = dict(
            index=ci, config=config, **summary(scores), model_sha256=gm.identity(m)
        )
        results.append(result)
        budget.finish()
    selected = min(results, key=lambda r: (r["worst_gate_ratio"], r["index"]))
    # Assess temporal projection independently of descriptor regression.
    m = load(out / f"model-{selected['index']}.json")
    basis = np.array(m["bases"])[1]
    d = y[:, 1:] - y[:, :1]
    projected = np.einsum("nsht,kh,kj->nsjt", d, basis, basis)
    projection = {
        w: (
            rms((projected - d)[:, :, sl], axis=2)
            / np.maximum(rms(d[:, :, sl], axis=2), 1e-30)
        ).tolist()
        for w, sl in WINDOWS.items()
    }
    pm.save_json(
        out / "selection.json",
        {
            "feature_set": args.feature_set,
            "candidates": results,
            "selected_index": selected["index"],
            "selection_rule": "minimum worst grouped normalized gate error; deterministic index tie break",
            "projection_relative": projection,
            "development_groups": sorted(set(groups.tolist())),
            "fixed_D1_scale": rms(d, axis=(0, 1, 2)).tolist(),
            "rank": 4,
        },
    )


def pairs(out, args, budget):
    budget.begin("pair-consequence-analysis")
    _x, y, rows = development()
    selection = load(ROOT / "boundaries-01/selection.json")
    candidates = selection["candidate_pairs"]
    floor = load(ROOT / "refinement-01/refinement.json")["floors"]
    sens = np.zeros((len(candidates), 2, 2, 2))
    sources = []
    for p in sorted(ROOT.glob("fit-centers-*/predictions-*.npz")):
        sens = np.maximum(sens, arrays(p)["sensitivity"])
        sources.append({"path": str(p), "sha256": digest(p)})
    ds = y[:, 1:] - y[:, :1]
    fixed = rms(ds, axis=(0, 1, 2))
    results = []
    for k, pair in enumerate(candidates):
        i, j = pair["indices"]
        for ai, a in enumerate([-0.02, 0.02]):
            for wi, (window, sl) in enumerate(WINDOWS.items()):
                delta = ds[i, ai, sl] - ds[j, ai, sl]
                magnitude = rms(delta)
                both = np.minimum(rms(ds[i, ai, sl]), rms(ds[j, ai, sl]))
                uncertainty = np.array(floor["D1/" + window]) * 2
                local = 5 * sens[k, wi, ai]
                for o, name in enumerate(["mass", "moment"]):
                    screen = bool(
                        pair["near"]
                        and both[o] > uncertainty[o] / 2
                        and magnitude[o] > uncertainty[o] + local[o]
                        and magnitude[o] > 0.1 * fixed[o]
                    )
                    results.append(
                        {
                            "pair": k,
                            "indices": [i, j],
                            "preparations": [rows[i], rows[j]],
                            "a": a,
                            "window": window,
                            "output": name,
                            "near": pair["near"],
                            "difference_rms": float(magnitude[o]),
                            "difference_max": float(abs(delta[:, o]).max()),
                            "left_rms": float(rms(ds[i, ai, sl])[o]),
                            "right_rms": float(rms(ds[j, ai, sl])[o]),
                            "numerical_bound": float(uncertainty[o]),
                            "fitted_sensitivity_bound": float(local[o]),
                            "fixed_scale": float(fixed[o]),
                            "empirical_screen": screen,
                            "qualification": "empirical screen only; physical sensitivity control required for robust collision",
                        }
                    )
    pm.save_json(
        out / "pairs.json",
        {
            "selection_sha256": digest(ROOT / "boundaries-01/selection.json"),
            "sensitivity_sources": sources,
            "records": results,
        },
    )
    budget.finish()


def freeze(out, args, budget):
    budget.begin("freeze-snapshot-model")
    selection = load(args.data / "selection.json")
    model = load(args.data / f"model-{selection['selected_index']}.json")
    pm.save_json(out / "model.json", model)
    numerical = load(ROOT / "refinement-01/refinement.json")
    pm.save_json(
        out / "freeze.json",
        {
            "sources": source_hashes(),
            "model_sha256": gm.identity(model),
            "source_selection": str(args.data),
            "selection_sha256": digest(args.data / "selection.json"),
            "fresh_seeds": FRESH,
            "histories": ["none", "odd04"],
            "event_boundary": 72,
            "probe_boundary": 90,
            "amplitudes": [-0.02, 0.02],
            "floors": numerical["floors"],
            "fixed_D1_scale": selection["fixed_D1_scale"],
            "gates": {"R": 0.02, "D1": 0.1},
            "access": "actual event-boundary snapshot descriptor; no claim of t50 retention",
            "repeated_stage": "blocked unless snapshot and retention independently qualify",
        },
    )
    budget.finish()


def fresh(out, args, budget):
    from .organization_response import isolated_components
    from .source_receiver_relay import prepare

    frozen = load(args.freeze / "freeze.json")
    model = load(args.freeze / "model.json")
    if (
        frozen["sources"] != source_hashes()
        or gm.identity(model) != frozen["model_sha256"]
    ):
        raise ValueError("Frozen source or model changed")
    _, _, development_rows = development()
    if set(FRESH) & {r["seed"] for r in development_rows}:
        raise ValueError("Fresh groups overlap development")
    states, starts, rows, xx = [], [], [], []
    for seed in FRESH:
        f = Field(CFG)
        initial = prepare(f, isolated_components(f, seed, budget), seed, -4.0, budget)
        save_npz(out / f"initial-{seed}.npz", state=initial)
        for history in ["none", "odd04"]:
            budget.begin(f"fresh-boundary-{seed}-{history}")
            state50 = unforced(
                initialize_written(initial, history, CFG), CFG, [0.0, 50.0], budget
            )[0][-1]
            state72 = reach_boundary(state50, budget)
            starts.append(state50)
            states.append(state72)
            xx.append(descriptor(state72))
            rows.append({"seed": seed, "history": history})
            budget.finish()
    save_npz(out / "boundary-50.npz", states=starts)
    save_npz(out / "boundary-72.npz", states=states)
    save_npz(out / "descriptors.npz", x=xx)
    pm.save_json(out / "rows.json", rows)
    budget.begin("seal-all-snapshot-predictions")
    save_npz(out / "predictions.npz", y=prediction(model, xx))
    pm.save_json(
        out / "seal.json",
        {
            "prediction_sha256": digest(out / "predictions.npz"),
            "descriptor_sha256": digest(out / "descriptors.npz"),
            "model_sha256": gm.identity(model),
            "freeze_sha256": digest(args.freeze / "freeze.json"),
            "order": "all snapshot predictions persisted before generating any future references",
        },
    )
    budget.finish()
    y = []
    for i, state in enumerate(states):
        y.append(
            [
                response_pair(out, f"reference-{i}-{a:g}", state, a, budget)
                for a in [0.0, -0.02, 0.02]
            ]
        )
    save_npz(out / "targets.npz", y=y)


def analyze(out, args, budget):
    budget.begin("sealed-fresh-analysis")
    base = args.data
    frozen = load(args.freeze / "freeze.json")
    seal = load(base / "seal.json")
    if seal["prediction_sha256"] != digest(base / "predictions.npz") or seal[
        "descriptor_sha256"
    ] != digest(base / "descriptors.npz"):
        raise ValueError("Fresh seal changed")
    y = arrays(base / "targets.npz")["y"]
    p = arrays(base / "predictions.npz")["y"]
    rows = load(base / "rows.json")
    records = gates(y, p, rows, frozen["floors"])
    pm.save_json(out / "scores.json", records)
    pm.save_json(out / "summary.json", summary(records))
    figures(out, y, p, rows)
    budget.finish()


def figures(out, y, prediction, rows):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    targets = [y[:, 2], y[:, 1] - y[:, 0], y[:, 2] - y[:, 0]]
    forecasts = [
        prediction[:, 2],
        prediction[:, 1] - prediction[:, 0],
        prediction[:, 2] - prediction[:, 0],
    ]
    for col, (truth, pred, title) in enumerate(
        zip(
            targets,
            forecasts,
            ["R_after (+.02)", "D1 (−.02)", "D1 (+.02)"],
            strict=True,
        )
    ):
        for o, name in enumerate(["C mass", "C signed moment"]):
            ax = axes[o, col]
            for i, row in enumerate(rows):
                (line,) = ax.plot(
                    pm.TIMES,
                    truth[i, :, o],
                    label=f"{row['seed']} {row['history']}",
                    lw=1.3,
                )
                ax.plot(pm.TIMES, pred[i, :, o], "--", color=line.get_color(), lw=1)
            ax.set_title(title + " / " + name)
            ax.set_xlabel("h after final probe")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0, 0].legend(fontsize=7)
    fig.suptitle("Fresh fixed-age snapshot assay: solid reference; dashed prediction")
    fig.savefig(out / "fresh-responses.png", dpi=160)
    plt.close(fig)


def retention(out, args, budget):
    """Exposed diagnostic only; no added coordinates or refitting."""
    from .repeated_intervention_state import model as original_model

    if load(ROOT / "fresh-analysis-01/summary.json")["failed"]:
        raise ValueError("Snapshot single-event qualification required")
    model = load(args.freeze / "model.json")
    if model["feature_set"] != "centers":
        raise ValueError("Only unchanged center evolution is specified here")
    budget.begin("unchanged-G-exposed-retention-diagnostic")
    base = args.data
    starts = arrays(base / "boundary-50.npz")["states"]
    initial = np.array([pm.extract(s, feature_set="geometry") for s in starts])
    g = original_model()["G"]
    trajectories = np.array([gm.rollout(g, z, np.arange(23)) for z in initial])
    forecasts = prediction(model, trajectories[:, -1])
    save_npz(out / "predictions.npz", initial=initial, z=trajectories, y=forecasts)
    # Only after the predictions are saved does this function read later truth.
    true_x = arrays(base / "descriptors.npz")["x"][:, :3]
    y = arrays(base / "targets.npz")["y"]
    rows = load(base / "rows.json")
    frozen = load(args.freeze / "freeze.json")
    records = gates(y, forecasts, rows, frozen["floors"])
    pm.save_json(out / "scores.json", records)
    pm.save_json(
        out / "summary.json",
        dict(
            summary(records),
            max_center_error=abs(trajectories[:, -1] - true_x).max(0).tolist(),
            actual_motion=(true_x - initial).tolist(),
            errors=(trajectories[:, -1] - true_x).tolist(),
            G_sha256=gm.identity(g),
            model_sha256=gm.identity(model),
            exposure="exposed snapshot fresh panel; diagnostic, not fresh once-measured qualification",
        ),
    )
    budget.finish()


RETAIN_FRESH = [24111, 24112, 24113]


def retain_freeze(out, args, budget):
    from .repeated_intervention_state import model as original_model

    result = load(ROOT / "retention-01/summary.json")
    if result["failed"] or result["unresolved"]:
        raise ValueError("Successful exposed retention diagnostic required")
    budget.begin("freeze-unchanged-G-and-snapshot-readout")
    frozen = load(args.freeze / "freeze.json")
    model = load(args.freeze / "model.json")
    g = original_model()["G"]
    pm.save_json(out / "model.json", model)
    pm.save_json(out / "G.json", g)
    frozen.update(
        sources=source_hashes(),
        fresh_seeds=RETAIN_FRESH,
        G_sha256=gm.identity(g),
        snapshot_freeze_sha256=digest(args.freeze / "freeze.json"),
        acquisition="three centers once at t50; unchanged G to t72; fixed age18 conditional-response map",
        access="no field access after t50 for prediction; later fields evaluator-only",
        repeated_stage="not specified by this fixed-age snapshot response map",
    )
    pm.save_json(out / "freeze.json", frozen)
    budget.finish()


def retain_fresh(out, args, budget):
    from .organization_response import isolated_components
    from .source_receiver_relay import prepare

    frozen = load(args.freeze / "freeze.json")
    model = load(args.freeze / "model.json")
    g = load(args.freeze / "G.json")
    if (
        frozen["sources"] != source_hashes()
        or frozen["model_sha256"] != gm.identity(model)
        or frozen["G_sha256"] != gm.identity(g)
    ):
        raise ValueError("Retained freeze changed")
    _, _, exposed = development()
    if set(RETAIN_FRESH) & ({r["seed"] for r in exposed} | set(FRESH)):
        raise ValueError("Untouched retention groups required")
    starts, rows = [], []
    for seed in RETAIN_FRESH:
        f = Field(CFG)
        initial = prepare(f, isolated_components(f, seed, budget), seed, -4.0, budget)
        save_npz(out / f"initial-{seed}.npz", state=initial)
        for history in ["none", "odd04"]:
            budget.begin(f"retained-initial-{seed}-{history}")
            starts.append(
                unforced(
                    initialize_written(initial, history, CFG), CFG, [0.0, 50.0], budget
                )[0][-1]
            )
            rows.append({"seed": seed, "history": history})
            budget.finish()
    save_npz(out / "boundary-50.npz", states=starts)
    initial = np.array([pm.extract(s, feature_set="geometry") for s in starts])
    save_npz(out / "initial-descriptors.npz", x=initial)
    pm.save_json(out / "rows.json", rows)
    budget.begin("seal-retained-predictions-before-later-fields")
    trajectories = []
    for i, z in enumerate(initial):
        state = gm.Continuation(g, z)
        state.advance(10)
        state.checkpoint(out / f"checkpoint-{i}.json")
        restored = gm.Continuation.restore(g, load(out / f"checkpoint-{i}.json"))
        end = restored.advance(12)
        trajectory = gm.rollout(g, z, np.arange(23))
        np.testing.assert_array_equal(end, trajectory[-1])
        trajectories.append(trajectory)
    trajectories = np.array(trajectories)
    save_npz(
        out / "predictions.npz",
        y=prediction(model, trajectories[:, -1]),
        z=trajectories,
    )
    pm.save_json(
        out / "seal.json",
        {
            "prediction_sha256": digest(out / "predictions.npz"),
            "initial_descriptor_sha256": digest(out / "initial-descriptors.npz"),
            "model_sha256": gm.identity(model),
            "G_sha256": gm.identity(g),
            "freeze_sha256": digest(args.freeze / "freeze.json"),
            "checkpoints": {p.name: digest(p) for p in out.glob("checkpoint-*.json")},
            "order": "all forecasts from t50 saved before ANY later field arrays or responses",
        },
    )
    budget.finish()
    states = []
    for i, start in enumerate(starts):
        budget.begin(f"retained-reference-boundary-{i}")
        states.append(reach_boundary(start, budget))
        budget.finish()
    save_npz(out / "boundary-72.npz", states=states)
    save_npz(out / "descriptors.npz", x=[descriptor(s) for s in states])
    y = [
        [
            response_pair(out, f"reference-{i}-{a:g}", s, a, budget)
            for a in [0.0, -0.02, 0.02]
        ]
        for i, s in enumerate(states)
    ]
    save_npz(out / "targets.npz", y=y)


def retain_analyze(out, args, budget):
    budget.begin("retained-fresh-scoring")
    base = args.data
    frozen = load(args.freeze / "freeze.json")
    seal = load(base / "seal.json")
    if (
        digest(base / "predictions.npz") != seal["prediction_sha256"]
        or digest(base / "initial-descriptors.npz") != seal["initial_descriptor_sha256"]
    ):
        raise ValueError("Retained prediction seal changed")
    saved = arrays(base / "predictions.npz")
    y = arrays(base / "targets.npz")["y"]
    rows = load(base / "rows.json")
    records = gates(y, saved["y"], rows, frozen["floors"])
    exact = arrays(base / "descriptors.npz")["x"][:, :3]
    z0 = arrays(base / "initial-descriptors.npz")["x"]
    errors = saved["z"][:, -1] - exact
    pm.save_json(out / "scores.json", records)
    pm.save_json(
        out / "summary.json",
        dict(
            summary(records),
            max_center_error=abs(errors).max(0).tolist(),
            center_errors=errors.tolist(),
            actual_motion=(exact - z0).tolist(),
            fractional_motion_error=(
                abs(errors) / np.maximum(abs(exact - z0), 1e-15)
            ).tolist(),
            claim="once-measured fixed age18 conditional-response forecast; no post-event geometry or variable-age/event-stream closure",
        ),
    )
    figures(out, y, saved["y"], rows)
    budget.finish()


def report(out, args, budget):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .event_state_aliasing import AGE
    from .repeated_intervention_state import MODEL_FILE
    from .repeated_intervention_state import model as original_model

    budget.begin("final-evidence-tables-and-figures")
    model = load(ROOT / "frozen-01/model.json")
    paths = [
        AGE / "frozen-01/model.json",
        args.inherited / "frozen-01/model.json",
        MODEL_FILE,
    ]
    pm.save_json(
        out / "inherited-identities.json",
        {
            "files": {str(p): digest(p) for p in paths},
            "fifteen_state": gm.identity(load(paths[0])),
            "eleven_state": gm.identity(load(paths[1])),
            "nine_state": gm.identity(original_model()),
            "unchanged_G": gm.identity(original_model()["G"]),
            "new_snapshot": gm.identity(model),
        },
    )
    panels = {}
    for name in ["fresh-analysis-01", "retained-analysis-01"]:
        scores = load(ROOT / name / "scores.json")
        panels[name] = {}
        for kind in ["R_without", "R_after", "D1"]:
            panels[name][kind] = {}
            for output in ["mass", "moment"]:
                subset = [
                    r for r in scores if r["kind"] == kind and r["output"] == output
                ]
                panels[name][kind][output] = {
                    "signal_rms_range": [
                        min(r["magnitude"] for r in subset),
                        max(r["magnitude"] for r in subset),
                    ],
                    "max_error_rms": max(r["error_rms"] for r in subset),
                    "max_error_pointwise": max(r["error_max"] for r in subset),
                    "worst_relative": max(r["relative"] for r in subset),
                    "max_peak_timing_difference": max(
                        abs(r["predicted_peak_h"] - r["true_peak_h"]) for r in subset
                    ),
                    "worst_case": max(subset, key=lambda r: r["relative"]),
                    "residuals_below_floor": sum(
                        r["error_rms"] < r["floor"] for r in subset
                    ),
                }
    pm.save_json(out / "scales.json", panels)
    active_snapshot = sum(
        np.size(model[k]) for k in ["coefficients", "mean", "scale", "bases", "scales"]
    )
    g = load(ROOT / "retained-frozen-01/G.json")
    pm.save_json(
        out / "accounting.json",
        {
            "selected_measurements": 3,
            "diagnostic_descriptors": 6,
            "evolving_centers": 3,
            "step_counter": 1,
            "event_amplitude_input": 1,
            "snapshot_active_fitted_basis_scaling_values": int(active_snapshot),
            "G_active_values_including_step": sum(
                int(np.size(g[k])) for k in ["coefficients", "mean", "scale", "step"]
            ),
            "amplitude_normalization_constant": 0.02,
            "output_values_per_response": 322,
            "snapshot_model_json_bytes": (ROOT / "frozen-01/model.json").stat().st_size,
            "G_json_bytes": (ROOT / "retained-frozen-01/G.json").stat().st_size,
            "inference": "22 unit RK4 steps, 88 rate products of 10 by 3; per requested amplitude one 10 by 24 product and two 4 by 161 temporal reconstructions for each of two outputs; no PDE or response lookup",
            "state_after_event": "fixed-age forecast only; a pre-event center triple plus declared amplitude can be kept, or materialize 322 response values; no evolving event memory qualified",
        },
    )
    pairs = load(ROOT / "boundaries-01/selection.json")["candidate_pairs"]
    rows = load(ROOT / "boundaries-01/rows.json")
    y = arrays(ROOT / "responses-01/targets.npz")["y"]
    p = arrays(ROOT / "fit-centers-quadratic/predictions-0.npz")["y"]
    fig, axes = plt.subplots(2, len(pairs), figsize=(17, 6), constrained_layout=True)
    for k, pair in enumerate(pairs):
        i, j = pair["indices"]
        for o, output in enumerate(["mass", "signed moment"]):
            ax = axes[o, k]
            for s, a in [(1, -0.02), (2, 0.02)]:
                truth = (y[i, s] - y[i, 0]) - (y[j, s] - y[j, 0])
                pred = (p[i, s] - p[i, 0]) - (p[j, s] - p[j, 0])
                (line,) = ax.plot(pm.TIMES, truth[:, o], label=f"a={a:+g}")
                ax.plot(pm.TIMES, pred[:, o], "--", color=line.get_color())
            ax.set_title(
                f"{rows[i]['seed']}/{rows[i]['history']}\n− {rows[j]['seed']}/{rows[j]['history']}",
                fontsize=9,
            )
            ax.set_xlabel("h after probe")
            ax.set_ylabel("ΔD1 " + output)
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle(
        "All response-blind selected pairs: solid reference difference; dashed grouped prediction difference"
    )
    fig.savefig(out / "matched-pair-responses.png", dpi=160)
    plt.close(fig)
    retained = ROOT / "retained-fresh-01"
    z0 = arrays(retained / "initial-descriptors.npz")["x"]
    actual = arrays(retained / "descriptors.npz")["x"][:, :3] - z0
    pred = arrays(retained / "predictions.npz")["z"][:, -1] - z0
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    labels = [str(r["seed"]) + " " + r["history"] for r in load(retained / "rows.json")]
    for o, name in enumerate("ABC"):
        axes[o].plot(actual[:, o], "o", label="true boundary motion")
        axes[o].plot(pred[:, o], "x", label="unchanged G")
        axes[o].set_xticks(
            range(len(labels)), labels, rotation=45, ha="right", fontsize=8
        )
        axes[o].set_title(name + " center, t50→72")
        axes[o].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0].legend(fontsize=8)
    fig.savefig(out / "pre-event-geometry.png", dpi=160)
    plt.close(fig)
    budget.finish()
