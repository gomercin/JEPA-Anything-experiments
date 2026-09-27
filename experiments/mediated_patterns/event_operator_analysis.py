"""Grouped operator-family diagnostics. No reference access from inference code."""

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
                "status": "FIXED_AGE_LIMIT",
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
