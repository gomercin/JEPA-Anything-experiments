"""Grouped operator-family diagnostics. No reference access from inference code."""

from pathlib import Path

import numpy as np

from . import event_operator_model as om
from . import geometry_model as gm
from . import present_state_model as pm
from .event_age_operator_family import AGES, ROOT, scores
from .event_age_response import arrays
from .event_state_analysis import summary
from .hybrid_pair import save_npz_exclusive as save_npz
from .present_state_transmission import load


def dataset(path):
    x = arrays(path / "states.npz")["x"]
    y = arrays(path / "targets.npz")["y"]
    rows = load(path / "rows.json")
    return (
        x,
        y,
        rows,
        np.array([r["age"] for r in rows]),
        np.array([r["seed"] for r in rows]),
    )


def fixed_panel(x, y, ages, groups, ridge, rank, out):
    pred = np.empty_like(y)
    folds = []
    for group in sorted(set(groups)):
        train = om.training_mask(groups, ages, group)
        rep = om.representation(x[train], y[train], rank)
        models = {}
        for age in AGES:
            tr = train & (ages == age)
            test = (groups == group) & (ages == age)
            model = om.fit(x[tr], ages[tr], y[tr], "blind", ridge, rank, rep)
            pred[test] = om.predict_panel(model, x[test], ages[test])
            models[str(age)] = model
        pm.save_json(out / f"fixed-fold-{group}.json", models)
        folds.append(
            {
                "held_group": int(group),
                "training_indices": np.flatnonzero(train).tolist(),
                "test_indices": np.flatnonzero(~train).tolist(),
                "representation_sha256": gm.identity(rep),
            }
        )
    pm.save_json(out / "fixed-folds.json", folds)
    return pred


def shared_cv(x, y, ages, groups, family, ridge, rank, out):
    predictions = {}
    for held_age in [None, 18, 15, 20]:
        pred = np.full_like(y, np.nan)
        folds = []
        for group in sorted(set(groups)):
            train = om.training_mask(groups, ages, group, held_age)
            test = (groups == group) & (True if held_age is None else ages == held_age)
            model = om.fit(x[train], ages[train], y[train], family, ridge, rank)
            pred[test] = om.predict_panel(model, x[test], ages[test])
            filename = f"fold-{held_age}-{group}.json"
            pm.save_json(out / filename, model)
            folds.append(
                {
                    "held_group": int(group),
                    "held_age": held_age,
                    "training_indices": np.flatnonzero(train).tolist(),
                    "test_indices": np.flatnonzero(test).tolist(),
                    "model": filename,
                }
            )
        valid = np.isfinite(pred).all(axis=(1, 2, 3))
        predictions[str(held_age)] = (pred, valid)
        pm.save_json(out / f"folds-{held_age}.json", folds)
    return predictions


def fit(out, args, budget):
    budget.begin("fit-and-grouped-age-diagnostics")
    x, y, rows, ages, groups = dataset(args.data)
    floors = load(ROOT / "refine-01/refinement.json")["floors"]
    rank = args.rank
    rep = om.representation(x, y, rank)
    w = om.targets(y, rep)
    projected = np.array(
        [[om.reconstruct(v, rep, a) for a in [0.0, -0.02, 0.02]] for v in w]
    )
    proj = scores(y, projected, rows, floors)
    pm.save_json(
        out / "projection.json",
        {"rank": rank, "summary": summary(proj), "scores": proj},
    )
    if summary(proj)["failed"]:
        pm.save_json(out / "decision.json", {"status": "BASIS_LIMIT", "rank": rank})
        budget.finish()
        return
    fixed_results = []
    for ridge in [1e-6, 1e-3]:
        sub = out / f"fixed-{ridge:g}"
        sub.mkdir()
        pred = fixed_panel(x, y, ages, groups, ridge, rank, sub)
        gates = scores(y, pred, rows, floors)
        save_npz(sub / "predictions.npz", y=pred)
        by_age = {str(a): summary([r for r in gates if r["age"] == a]) for a in AGES}
        pm.save_json(sub / "scores.json", gates)
        pm.save_json(sub / "summary.json", by_age)
        fixed_results.append(
            (max(v["worst_gate_ratio"] for v in by_age.values()), ridge, by_age)
        )
    _, fixed_ridge, fixed_summaries = min(fixed_results, key=lambda v: v[0])
    for age in AGES:
        tr = ages == age
        model = om.fit(x[tr], ages[tr], y[tr], "blind", fixed_ridge, rank, rep)
        pm.save_json(out / f"fixed-age-{age}.json", model)
    if any(v["failed"] or v["unresolved"] for v in fixed_summaries.values()):
        pm.save_json(
            out / "decision.json",
            {
                "status": "FIXED_AGE_LIMIT"
                if any(v["failed"] for v in fixed_summaries.values())
                else "UNRESOLVED_FIXED_AGE",
                "ridge": fixed_ridge,
                "by_age": fixed_summaries,
            },
        )
        budget.finish()
        return
    results = []
    for family in ["blind", args.family]:
        for ridge in [1e-6, 1e-3]:
            sub = out / f"{family}-{ridge:g}"
            sub.mkdir()
            predictions = shared_cv(x, y, ages, groups, family, ridge, rank, sub)
            summaries = {}
            for label, (pred, valid) in predictions.items():
                ii = np.flatnonzero(valid)
                gates = scores(y[valid], pred[valid], [rows[i] for i in ii], floors)
                summaries[label] = summary(gates)
                save_npz(sub / f"predictions-{label}.npz", indices=ii, y=pred[valid])
                pm.save_json(sub / f"scores-{label}.json", gates)
            pm.save_json(sub / "summary.json", summaries)
            model = om.fit(x, ages, y, family, ridge, rank)
            pm.save_json(sub / "model.json", model)
            results.append(
                {
                    "family": family,
                    "ridge": ridge,
                    "path": str(sub),
                    "summaries": summaries,
                    "worst": max(s["worst_gate_ratio"] for s in summaries.values()),
                    "qualified": all(
                        s["failed"] == 0 and s["unresolved"] == 0
                        for s in summaries.values()
                    ),
                }
            )
    eligible = [r for r in results if r["qualified"]]
    # Prefer blind if it qualifies; context must earn its role.
    selected = (
        min(eligible, key=lambda r: (r["family"] != "blind", r["worst"]))
        if eligible
        else None
    )
    pm.save_json(
        out / "decision.json",
        {
            "status": "CREDIBLE" if selected else "INTERPOLATION_LIMIT",
            "fixed_ridge": fixed_ridge,
            "fixed_summaries": fixed_summaries,
            "candidates": results,
            "selected": selected,
        },
    )
    budget.finish()


def freeze(out, args, budget):
    from .event_age_operator_family import FRESH, OLD
    from .present_state_transmission import digest
    from .repeated_intervention_state import source_hashes

    budget.begin("freeze-contextual-contract")
    decision = load(args.data / "decision.json")
    if decision["status"] != "CREDIBLE":
        raise ValueError("No credible development candidate")
    selected = decision["selected"]
    blind = min(
        [c for c in decision["candidates"] if c["family"] == "blind"],
        key=lambda c: c["worst"],
    )
    for label, candidate in [("selected", selected), ("blind", blind)]:
        pm.save_json(
            out / (label + ".json"), load(Path(candidate["path"]) / "model.json")
        )
    g = load(OLD / "retained-frozen-01/G.json")
    if (
        gm.identity(g)
        != "a9af96ddb21bcd90e92cb8cd51fa1371c95166949fec6ffe54c0e1367dfdded9"
    ):
        raise ValueError("Inherited G changed")
    pm.save_json(out / "G.json", g)
    pm.save_json(
        out / "freeze.json",
        {
            "selected": selected,
            "blind": blind,
            "models": {p.name: digest(p) for p in out.glob("*.json")},
            "sources": source_hashes(),
            "floors": load(ROOT / "refine-01/refinement.json")["floors"],
            "fresh_seeds": FRESH,
            "ages": [18, 24],
            "heldout_age": 24,
            "amplitudes": [-0.02, 0.02],
            "probe": 90,
            "development": str(ROOT / "develop-01"),
            "development_rows_sha256": digest(ROOT / "develop-01/rows.json"),
            "contract": "Snapshot first; presealed unchanged-G forecasts scored only after snapshot qualification; no outcome tuning",
        },
    )
    budget.finish()


def fresh(out, args, budget):
    from .event_age_operator_family import FRESH, boundary, response_pair
    from .geometry_assay import initialize_written
    from .geometry_evolution import unforced
    from .organization_response import isolated_components
    from .present_state_transmission import digest
    from .repeated_intervention_state import source_hashes
    from .simulator import Field
    from .source_receiver_relay import CFG, prepare

    frozen = load(args.data / "freeze.json")
    if frozen["sources"] != source_hashes():
        raise ValueError("Frozen implementation changed")
    if any(digest(args.data / name) != sha for name, sha in frozen["models"].items()):
        raise ValueError("Frozen artifact changed")
    development = load(Path(frozen["development"]) / "rows.json")
    if set(FRESH) & {r["seed"] for r in development}:
        raise ValueError("Fresh preparation overlap")
    models = {
        label: load(args.data / (label + ".json")) for label in ["selected", "blind"]
    }
    g = load(args.data / "G.json")
    starts, rows, z0s, zgs = [], [], [], []
    for seed in FRESH:
        f = Field(CFG)
        initial = prepare(f, isolated_components(f, seed, budget), seed, -4.0, budget)
        save_npz(out / f"initial-{seed}.npz", state=initial)
        for history in ["none", "odd04"]:
            budget.begin(f"fresh-start-{seed}-{history}")
            state50 = unforced(
                initialize_written(initial, history, CFG), CFG, [0.0, 50.0], budget
            )[0][-1]
            start_index = len(starts)
            starts.append(state50)
            z0 = pm.extract(state50, feature_set="geometry")
            flow = gm.Continuation(g, z0)
            flow.advance(10)
            flow.checkpoint(out / f"checkpoint-{seed}-{history}.json")
            for age in [24, 18]:
                zg = flow.advance(40 - age - flow.steps)
                rows.append(
                    {
                        "seed": seed,
                        "history": history,
                        "age": age,
                        "start_index": start_index,
                    }
                )
                z0s.append(z0)
                zgs.append(zg)
            budget.finish()
    save_npz(out / "boundary-50.npz", states=starts)
    save_npz(out / "retained-centers.npz", z0=z0s, z=zgs)
    pm.save_json(out / "rows.json", rows)
    ages = np.array([r["age"] for r in rows])
    budget.begin("seal-G-forecasts-before-any-later-field")
    for label, model in models.items():
        save_npz(out / f"retained-{label}.npz", y=om.predict_panel(model, zgs, ages))
    pm.save_json(
        out / "retained-seal.json",
        {
            "predictions": {p.name: digest(p) for p in out.glob("retained-*.npz")},
            "checkpoints": {p.name: digest(p) for p in out.glob("checkpoint-*.json")},
            "freeze_sha256": digest(args.data / "freeze.json"),
            "order": "All t50-only G predictions persisted before generating any later field",
        },
    )
    budget.finish()
    states, x = [], []
    for row in rows:
        budget.begin(f"current-centers-{row['seed']}-{row['history']}-{row['age']}")
        state = boundary(starts[row["start_index"]], row["age"], budget)
        states.append(state)
        x.append(pm.extract(state, feature_set="geometry"))
        budget.finish()
    save_npz(out / "states.npz", states=states, x=x)
    budget.begin("seal-snapshot-forecasts-before-conditioning")
    for label, model in models.items():
        save_npz(out / f"snapshot-{label}.npz", y=om.predict_panel(model, x, ages))
    pm.save_json(
        out / "snapshot-seal.json",
        {
            "predictions": {p.name: digest(p) for p in out.glob("snapshot-*.npz")},
            "current_centers_sha256": digest(out / "states.npz"),
            "freeze_sha256": digest(args.data / "freeze.json"),
            "order": "All snapshot predictions persisted before generating conditioning/probe futures",
        },
    )
    budget.finish()
    yy, ledger = [], []
    for i, (row, state) in enumerate(zip(rows, states, strict=True)):
        yy.append(
            [
                response_pair(out, f"reference-{i}-{a:g}", state, row["age"], a, budget)
                for a in [0.0, -0.02, 0.02]
            ]
        )
        ledger.append(
            [str(out / f"reference-{i}-{a:g}.npz") for a in [0.0, -0.02, 0.02]]
        )
    save_npz(out / "targets.npz", y=yy)
    pm.save_json(out / "references.json", ledger)


def analyze(out, args, budget):
    from .present_state_transmission import digest

    budget.begin("frozen-fresh-scoring")
    x, y, rows, _ages, _ = dataset(args.data)
    frozen = load(ROOT / "frozen-01/freeze.json")
    results = {}
    for mode in ["snapshot", "retained"]:
        seal = load(args.data / (mode + "-seal.json"))
        if any(
            digest(args.data / name) != sha for name, sha in seal["predictions"].items()
        ):
            raise ValueError("Prediction seal mismatch")
        for label in ["blind", "selected"]:
            p = arrays(args.data / f"{mode}-{label}.npz")["y"]
            gates = scores(y, p, rows, frozen["floors"])
            pm.save_json(out / f"{mode}-{label}-scores.json", gates)
            results[mode + "-" + label] = {
                str(a): summary([r for r in gates if r["age"] == a]) for a in [18, 24]
            }
    qualified = lambda key: all(
        v["failed"] == 0 and v["unresolved"] == 0 for v in results[key].values()
    )
    pm.save_json(
        out / "decision.json",
        {
            "results": results,
            "snapshot_qualified": qualified("snapshot-selected"),
            "retained_qualified": qualified("snapshot-selected")
            and qualified("retained-selected"),
            "repeated_allowed": qualified("snapshot-selected")
            and qualified("retained-selected"),
        },
    )
    zg = arrays(args.data / "retained-centers.npz")
    err = zg["z"] - x
    pm.save_json(
        out / "geometry.json",
        {
            "max_absolute": abs(err).max(0).tolist(),
            "max_fractional": (abs(err) / np.maximum(abs(x - zg["z0"]), 1e-15))
            .max(0)
            .tolist(),
            "errors": err.tolist(),
            "true_motion": (x - zg["z0"]).tolist(),
            "predicted_motion": (zg["z"] - zg["z0"]).tolist(),
            "inherited_tolerances": [5e-5, 2e-5, 2e-6],
            "scope": "pre-event unforced centers only; old geometry failures preserved",
        },
    )
    gates = load(out / "snapshot-selected-scores.json")
    worst = max(
        (r for r in gates if r["age"] == 24 and r["kind"] == "D1"),
        key=lambda r: r["relative"],
    )
    pm.save_json(out / "adverse.json", worst)
    budget.finish()


def operators(out, args, budget):
    """Evaluate adjacent per-age operators on identical nearby supported centers."""
    budget.begin("common-support-operator-diagnostics")
    x, _y, _rows, ages, _ = dataset(ROOT / "develop-01")
    models = {a: load(args.data / f"fixed-age-{a}.json") for a in AGES}
    scale = x.std(0)
    candidate = x[ages == 18]
    # Empirical support: intersection of coordinate ranges and nearby training
    # examples at every age. No extrapolation claim based on a box alone.
    supported = []
    for z in candidate:
        if all(
            ((z >= x[ages == a].min(0)) & (z <= x[ages == a].max(0))).all()
            and np.min(np.linalg.norm((x[ages == a] - z) / scale, axis=1) / np.sqrt(3))
            <= 0.5
            for a in AGES
        ):
            supported.append(z)
    if not supported:
        pm.save_json(
            out / "support.json",
            {
                "count": 0,
                "interpretation": "No shared supported query; no operator extrapolation",
            },
        )
        budget.finish()
        return
    predictions = np.array(
        [om.predict_panel(models[a], supported, [a] * len(supported)) for a in AGES]
    )
    save_npz(
        out / "operators.npz",
        centers=supported,
        ages=AGES,
        y=predictions,
        coefficients=np.array([models[a]["coefficients"] for a in AGES]),
    )
    records = []
    for i in range(len(AGES) - 1):
        d = predictions[i, :, 1:] - predictions[i, :, 0, None]
        dn = predictions[i + 1, :, 1:] - predictions[i + 1, :, 0, None]
        records.append(
            {
                "ages": AGES[i : i + 2],
                "D_change_rms": np.sqrt(np.mean((dn - d) ** 2, axis=(0, 2))).tolist(),
                "D_rms": np.sqrt(np.mean(d * d, axis=(0, 2))).tolist(),
            }
        )
    pm.save_json(
        out / "support.json",
        {
            "count": len(supported),
            "rule": "All-age coordinate-range intersection and normalized nearest-neighbor RMS <= .5",
            "adjacent_changes": records,
            "meaning": "Prediction-space comparison only; coefficients have no physical mode interpretation",
        },
    )
    budget.finish()


def report(out, args, budget):
    """Small standalone scientific figures and explicit storage accounting."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .event_age_operator_family import OLD
    from .present_state_transmission import digest

    budget.begin("result-figures-and-accounting")
    decision = load(args.data / "decision.json")
    pm.save_json(out / "decision-copy.json", decision)
    identities = {
        str(p): digest(p)
        for p in [OLD / "frozen-01/model.json", OLD / "retained-frozen-01/G.json"]
    }
    pm.save_json(out / "inherited-identities.json", identities)
    if (ROOT / "operators-01/operators.npz").exists():
        d = arrays(ROOT / "operators-01/operators.npz")
        y = d["y"][:, 0]
        fig, axs = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
        for j, age in enumerate(AGES):
            for o in range(2):
                axs[0, o].plot(
                    pm.TIMES, (y[j, 2, :, o] - y[j, 1, :, o]) / 2, label=str(age)
                )
                axs[1, o].plot(
                    pm.TIMES, (y[j, 2, :, o] + y[j, 1, :, o] - 2 * y[j, 0, :, o]) / 2
                )
        for ax in axs.flat:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.set_xlabel("h since diagnostic probe")
        axs[0, 0].set_title("Odd component: mass")
        axs[0, 1].set_title("Odd component: signed moment")
        axs[1, 0].set_title("Even component: mass")
        axs[1, 1].set_title("Even component: signed moment")
        axs[0, 0].legend(title="Known age")
        fig.suptitle("Different age operators queried at the same supported centers")
        fig.savefig(out / "operator-age-curves.png", dpi=160)
        plt.close(fig)
    if "candidates" in decision:
        fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
        for c in decision["candidates"]:
            if c["ridge"] == 1e-6:
                vals = [
                    c["summaries"][str(a)]["by_kind"]["D1"]["worst_relative"] * 100
                    for a in [15, 18, 20]
                ]
                ax.plot([15, 18, 20], vals, "o-", label=c["family"])
        ax.axhline(10, color="black", linestyle="--", label="D1 target")
        ax.set(
            xlabel="Entire held-out development age",
            ylabel="Worst grouped D1 error (%)",
            yscale="log",
        )
        ax.legend()
        fig.savefig(out / "held-age-errors.png", dpi=160)
        plt.close(fig)
    fresh = ROOT / "fresh-01"
    if (fresh / "targets.npz").exists():
        x, y, rows, _ages, _ = dataset(fresh)
        p = arrays(fresh / "snapshot-selected.npz")["y"]
        pg = arrays(fresh / "retained-selected.npz")["y"]
        i = next(
            i
            for i, r in enumerate(rows)
            if r["seed"] == 26101 and r["history"] == "odd04" and r["age"] == 24
        )
        fig, axs = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
        for o in range(2):
            axs[0, o].plot(pm.TIMES, y[i, 2, :, o], label="Reference")
            axs[0, o].plot(pm.TIMES, p[i, 2, :, o], "--", label="Current centers")
            axs[0, o].plot(pm.TIMES, pg[i, 2, :, o], ":", label="t50 + unchanged G")
            for j, sign in [(1, "−.02"), (2, "+.02")]:
                axs[1, o].plot(
                    pm.TIMES, y[i, j, :, o] - y[i, 0, :, o], label="True " + sign
                )
                axs[1, o].plot(
                    pm.TIMES,
                    p[i, j, :, o] - p[i, 0, :, o],
                    "--",
                    label="Predicted " + sign,
                )
        for ax in axs.flat:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.set_xlabel("h since diagnostic probe")
        axs[0, 0].set_title("R_after: mass")
        axs[0, 1].set_title("R_after: signed moment")
        axs[1, 0].set_title("D1: mass")
        axs[1, 1].set_title("D1: signed moment")
        axs[0, 0].legend()
        axs[1, 0].legend()
        fig.suptitle(
            "Prospectively selected first written fresh preparation, unseen age24"
        )
        fig.savefig(out / "fresh-response.png", dpi=160)
        plt.close(fig)
        z = arrays(fresh / "retained-centers.npz")
        fig, axs = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)
        for o, name in enumerate("ABC"):
            axs[o].plot(x[:, o] - z["z0"][:, o], label="True displacement")
            axs[o].plot(z["z"][:, o] - z["z0"][:, o], "--", label="G displacement")
            axs[o].set(title=name, xlabel="Preparation/history/age row")
            axs[o].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        axs[0].legend()
        fig.savefig(out / "pre-event-geometry.png", dpi=160)
        plt.close(fig)
        m = load(ROOT / "frozen-01/selected.json")
        numeric = {
            k: int(np.asarray(m[k]).size)
            for k in ["mean", "scale", "bases", "scales", "coefficients"]
        }
        pm.save_json(
            out / "accounting.json",
            {
                "selected_family": m["family"],
                "active_response_values": numeric,
                "response_total": sum(numeric.values()),
                "G_active": 37,
                "amplitude_normalizer": 1,
                "age_normalization_values": 2,
                "age_spline_knots": 4 if m["family"] == "spline" else 0,
                "model_bytes": (ROOT / "frozen-01/selected.json").stat().st_size,
                "G_bytes": (ROOT / "frozen-01/G.json").stat().st_size,
                "retained_centers": 3,
                "counter": 1,
                "external_context": [
                    "conditioning amplitude",
                    "known age at requested probe",
                ],
                "output_values_per_query": 322,
                "amplitude_queries_per_triplet": 3,
                "feature_count": len(m["coefficients"]),
                "coefficient_columns": 6 * m["rank"],
                "basis_products_per_query": 2,
                "geometry_rate_evaluations_per_step": 4,
                "initial_acquisition": "one field scan at50 for retained path; current-event field scan for snapshot",
                "no_post_event_state": True,
            },
        )
    pm.save_json(
        out / "resource-previous.json",
        {
            "cpu_seconds_before_report": budget.previous,
            "scope": "All new sequential scientific stages including failed attempts plus30-second inspection allowance",
        },
    )
    budget.finish()


def repeated(out, args, budget):
    """Conditional exposed independent addition; no paired-event fit or field solve."""
    from . import repeated_intervention_model as rm
    from .present_state_transmission import digest
    from .repeated_intervention_analysis import score as repeat_score
    from .repeated_intervention_analysis import truth

    decision = load(ROOT / "fresh-analysis-01/decision.json")
    if not decision["repeated_allowed"]:
        raise ValueError(
            "Fresh single-event snapshot and retained gates must pass first"
        )
    budget.begin("exposed-independent-addition-diagnostic")
    frozen = ROOT / "frozen-01"
    model = load(frozen / "selected.json")
    g = load(frozen / "G.json")
    base = args.inherited / "fresh-01"
    z0 = arrays(base / "initial-descriptors.npz")["z"]
    cases = [
        c
        for c in load(base / "cases.json")
        if c["schedule"]["name"] in ["cancel", "timing"]
    ]
    predictions = []
    for k, c in enumerate(cases):
        time = c["schedule"]["times"]
        amps = c["schedule"]["amplitudes"]
        # Independent single-event histories share the unforced prefix; this
        # does not evolve the first physical conditioning through the second.
        z = gm.rollout(g, z0[c["index"]], time)
        r0a = om.predict(model, z[0], 0.0, 40 - time[0])
        r1 = om.predict(model, z[0], amps[0], 40 - time[0])
        r0b = om.predict(model, z[1], 0.0, 40 - time[1])
        r2 = om.predict(model, z[1], amps[1], 40 - time[1])
        raw = np.array([r0a, r1, r2, r1 + r2 - r0a])
        # Both no-event contexts describe the same final physical baseline.
        # Sum independently predicted contrasts on a single predicted baseline;
        # no true R00 is supplied. Keep raw addition and its baseline disagreement.
        centered = np.array([r0a, r1, r0a + (r2 - r0b), r1 + (r2 - r0b)])
        save_npz(
            out / f"prediction-{k}.npz",
            raw=raw,
            centered=centered,
            baseline_disagreement=r0b - r0a,
            components=[r0a, r1, r0b, r2],
        )
        predictions.append((raw, centered))
    pm.save_json(
        out / "seal.json",
        {
            "model_sha256": gm.identity(model),
            "G_sha256": gm.identity(g),
            "predictions": {p.name: digest(p) for p in out.glob("prediction-*.npz")},
            "order": "All predictions saved before opening exposed repeated future responses; no fitting",
        },
    )
    floors = load(args.inherited / "frozen-01/freeze.json")["floors"]["floors"]
    all_scores = []
    identities = {}
    for k, (c, (raw, centered)) in enumerate(zip(cases, predictions, strict=True)):
        y, _, _ = truth(base, c)
        for name in c["prefixes"]:
            p = base / (name + ".npz")
            identities[str(p)] = digest(p)
        for label, pred in [
            ("raw-addition", raw),
            ("common-baseline-addition", centered),
            (
                "true-single-event-addition",
                np.array([y[0], y[1], y[2], y[1] + y[2] - y[0]]),
            ),
        ]:
            rows = repeat_score(y, pred, floors)
            all_scores.extend(
                dict(
                    **r,
                    case=k,
                    seed=c["seed"],
                    history=c["history"],
                    schedule=c["schedule"]["name"],
                    model=label,
                )
                for r in rows
            )
            lhs = y[3] - pred[3]
            rhs = (
                rm.contrasts(y)["K12"]
                + ((y[1] + y[2] - y[0]) - (pred[1] + pred[2] - pred[0]))
                + ((pred[1] + pred[2] - pred[0]) - pred[3])
            )
            if not np.allclose(lhs, rhs, rtol=1e-12, atol=1e-21):
                raise ValueError("Signed error decomposition mismatch")
        comp = arrays(out / f"prediction-{k}.npz")["components"]
        from .event_age_response import score as isolated_score

        for j in [0, 1]:
            for r in isolated_score(
                y[[0, j + 1]],
                comp[2 * j : 2 * j + 2],
                load(frozen / "freeze.json")["floors"],
            ):
                all_scores.append(
                    dict(
                        **r,
                        case=k,
                        seed=c["seed"],
                        history=c["history"],
                        age=40 - c["schedule"]["times"][j],
                        a=c["schedule"]["amplitudes"][j],
                        schedule=c["schedule"]["name"],
                        model=f"isolated-component-{j + 1}",
                    )
                )
        save_npz(out / f"truth-{k}.npz", y=y)
    pm.save_json(out / "scores.json", all_scores)
    pm.save_json(out / "cases.json", cases)
    pm.save_json(out / "source-identities.json", identities)
    pm.save_json(
        out / "interpretation.json",
        {
            "exposed_only": True,
            "refitting": False,
            "comparisons": [
                "raw R10+R01−R00",
                "independently predicted D1+D2 on one predicted baseline",
                "evaluator true addition",
            ],
            "caveat": "Known-context single-event queries, not an event-updated autonomous state. K12 predicted zero by independent addition.",
        },
    )
    budget.finish()


def diagnose(out, args, budget):
    """Retain adverse held-age errors and common-support coefficient trajectories."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .event_age_response import rms

    budget.begin("adverse-interpolation-and-age-trajectory-diagnosis")
    _x, y, _rows, _ages, _ = dataset(ROOT / "develop-01")
    decision = load(args.data / "decision.json")
    selected = min(
        [c for c in decision["candidates"] if c["family"] != "blind"],
        key=lambda c: c["worst"],
    )
    folder = Path(selected["path"])
    records = []
    for age in [15, 18, 20]:
        gates = load(folder / f"scores-{age}.json")
        pred = arrays(folder / f"predictions-{age}.npz")
        for r in gates:
            r = dict(**r, development_index=int(pred["indices"][r["index"]]))
            records.append(r)
    worst = max((r for r in records if r["kind"] == "D1"), key=lambda r: r["relative"])
    pm.save_json(out / "adverse.json", worst)
    pm.save_json(out / "all-held-age-scores.json", records)
    idx = worst["development_index"]
    aindex = 1 if worst["a"] < 0 else 2
    pred = arrays(folder / f"predictions-{worst['age']}.npz")
    pi = int(np.flatnonzero(pred["indices"] == idx)[0])
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for o in range(2):
        axs[0, o].plot(pm.TIMES, y[idx, aindex, :, o], label="Reference R_after")
        axs[0, o].plot(
            pm.TIMES,
            pred["y"][pi, aindex, :, o],
            "--",
            label="Held-age/group prediction",
        )
        axs[1, o].plot(
            pm.TIMES, y[idx, aindex, :, o] - y[idx, 0, :, o], label="Reference D1"
        )
        axs[1, o].plot(
            pm.TIMES,
            pred["y"][pi, aindex, :, o] - pred["y"][pi, 0, :, o],
            "--",
            label="Prediction D1",
        )
    for ax in axs.flat:
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.set_xlabel("h since diagnostic probe")
        ax.legend()
    axs[0, 0].set_title("Mass: large conditional response")
    axs[0, 1].set_title("Signed moment: large conditional response")
    axs[1, 0].set_title("Mass: smaller event consequence")
    axs[1, 1].set_title("Signed moment: smaller event consequence")
    fig.suptitle(
        f"Worst spline holdout: seed{worst['seed']} {worst['history']}, age{worst['age']}, a={worst['a']}"
    )
    fig.savefig(out / "adverse-held-age.png", dpi=160)
    plt.close(fig)
    op = arrays(ROOT / "operators-01/operators.npz")
    models = [load(ROOT / "fit-01" / f"fixed-age-{a}.json") for a in AGES]
    z = op["centers"]
    # These weights are comparable because all fixed-age fits share the basis.
    w = np.array(
        [
            (
                om.matrix(z, [a] * len(z), m, "blind") @ np.asarray(m["coefficients"])
            ).reshape(len(z), 3, 2, m["rank"])
            for a, m in zip(AGES, models, strict=True)
        ]
    )
    save_npz(
        out / "common-basis-trajectories.npz",
        ages=AGES,
        centers=z,
        weights=w,
        coefficients=op["coefficients"],
    )
    fig, axs = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for part, label in [(1, "odd"), (2, "even")]:
        for o, name in enumerate(["mass", "moment"]):
            for k in range(w.shape[-1]):
                axs[part - 1, o].plot(
                    AGES, w[:, 0, part, o, k], "o-", label=f"basis{k + 1}"
                )
            axs[part - 1, o].set(
                title=f"{label}, {name}",
                xlabel="Known event age",
                ylabel="Normalized response-basis coefficient",
            )
    axs[0, 0].legend()
    fig.suptitle(
        "Fixed supported centers; common-basis weights, no physical-mode meaning"
    )
    fig.savefig(out / "coefficient-age-trajectories.png", dpi=160)
    plt.close(fig)

    def by_sign(entries):
        return {
            str(a): {
                str(age): {
                    kind: {
                        "worst_relative": max(
                            r["relative"]
                            for r in entries
                            if r["a"] == a and r["age"] == age and r["kind"] == kind
                        ),
                        "max_error_rms": max(
                            r["error_rms"]
                            for r in entries
                            if r["a"] == a and r["age"] == age and r["kind"] == kind
                        ),
                        "failures": sum(
                            r["passed"] is False
                            for r in entries
                            if r["a"] == a and r["age"] == age and r["kind"] == kind
                        ),
                    }
                    for kind in ["R_after", "D1"]
                }
                for age in [15, 18, 20]
            }
            for a in [-0.02, 0.02]
        }

    sy = rms((y[:, 1:] - y[:, 0, None]).reshape(-1, 161, 2), axis=(0, 1))
    pm.save_json(
        out / "diagnosis.json",
        {
            "best_tested_shared": selected,
            "by_sign": by_sign(records),
            "fixed_development_D1_scale": sy.tolist(),
            "error_over_floor": worst["error_rms"] / worst["floor"],
            "signal_over_floor": worst["magnitude"] / worst["floor"],
            "response_basis": "rank4 sufficient; no rank6 search",
            "stop": "Both bounded shared families fail full-age holdouts; no fresh freeze, age24 generation, retained or repeated evaluation",
        },
    )
    # Static costs count candidate models even though no deployment is qualified.
    accounting = {}
    for c in decision["candidates"]:
        m = load(Path(c["path"]) / "model.json")
        numeric = {
            k: int(np.asarray(m[k]).size)
            for k in ["mean", "scale", "bases", "scales", "coefficients"]
        }
        accounting[c["family"]] = {
            "values": numeric,
            "total": sum(numeric.values()),
            "bytes": (Path(c["path"]) / "model.json").stat().st_size,
            "coefficient_multiply_adds": len(m["coefficients"]) * 6 * m["rank"],
            "basis_multiply_adds": 4 * m["rank"] * 161,
            "runtime_inputs": "three current centers, conditioning amplitude, known age context (unused by blind features)",
            "measurement": "current field scan; retained G variant not qualified in this experiment",
            "output_values": 322,
            "temporal_grid_values": 161,
            "anchors": 3,
            "amplitude_normalizer": 1,
            "age_normalization": 2,
            "knots": 4 if m["family"] == "spline" else 0,
        }
    pm.save_json(out / "accounting.json", accounting)
    budget.finish()
