"""Grouped atlas evidence; current fields and future references stay evaluator-only."""

import numpy as np

from . import event_atlas_model as am
from . import event_operator_model as om
from . import present_state_model as pm
from .event_age_operator_atlas import ROOT
from .event_age_operator_family import ROOT as OLD
from .event_age_operator_family import scores
from .event_age_response import WINDOWS, rms
from .event_operator_analysis import dataset
from .event_state_analysis import summary
from .hybrid_pair import save_npz_exclusive as save_npz
from .present_state_transmission import digest, load


def combined():
    paths = [OLD / "develop-01", ROOT / "develop-01"]
    data = [dataset(p) for p in paths]
    x, y = [np.concatenate([d[j] for d in data]) for j in (0, 1)]
    rows = data[0][2] + data[1][2]
    ages, groups = [np.concatenate([d[j] for d in data]) for j in (3, 4)]
    return x, y, rows, ages, groups


def floors():
    records = [
        load(p / "refinement.json")["floors"]
        for p in (OLD / "refine-01", ROOT / "refine14-01", ROOT / "refine25-01")
    ]
    return {k: np.max([r[k] for r in records], axis=0).tolist() for k in records[0]}


def fit(out, args, budget):
    budget.begin("fixed-node-and-held-context-fits")
    x, y, rows, ages, groups = combined()
    floor = floors()
    pm.save_json(out / "floors.json", floor)
    pm.save_json(
        out / "input-identities.json",
        {
            str(p): digest(p)
            for base in (OLD / "develop-01", ROOT / "develop-01")
            for p in (base / "rows.json", base / "states.npz", base / "targets.npz")
        },
    )
    pm.save_json(out / "rows.json", rows)
    if args.family == "linear":
        rep = om.representation(x, y)
        projection = np.array(
            [
                [om.reconstruct(w, rep, a) for a in (0.0, -0.02, 0.02)]
                for w in om.targets(y, rep)
            ]
        )
        gates = scores(y, projection, rows, floor)
        pm.save_json(
            out / "projection.json", {"summary": summary(gates), "scores": gates}
        )
        pred = np.empty_like(y)
        folds = []
        for group in sorted(set(groups)):
            train = om.training_mask(groups, ages, group)
            test = ~train
            model = am.fit(x[train], ages[train], y[train], am.REFINED)
            pred[test] = am.predict_panel(model, x[test], ages[test])
            filename = f"fixed-{group}.json"
            pm.save_json(out / filename, model)
            folds.append(
                {
                    "held_group": int(group),
                    "training_indices": np.flatnonzero(train).tolist(),
                    "test_indices": np.flatnonzero(test).tolist(),
                    "model": filename,
                }
            )
            budget.check()
        gates = scores(y, pred, rows, floor)
        save_npz(out / "fixed-predictions.npz", y=pred)
        pm.save_json(out / "fixed-folds.json", folds)
        pm.save_json(out / "fixed-scores.json", gates)
        fixed = {
            str(a): summary([g for g in gates if g["age"] == a]) for a in am.REFINED
        }
        pm.save_json(out / "fixed-summary.json", fixed)
        all_model = am.fit(x, ages, y, am.REFINED)
        pm.save_json(out / "all-node-model.json", all_model)
        if summary(gates)["failed"] or summary(gates)["unresolved"]:
            pm.save_json(
                out / "decision.json", {"status": "FIXED_NODE_FAILURE", "fixed": fixed}
            )
            budget.finish()
            return
    else:
        previous = load(
            ROOT
            / ("fit-linear-01" if args.family == "quadratic" else "fit-quadratic-01")
            / "decision.json"
        )
        if previous["status"] != "LOCAL_INTERPOLATION_LIMIT":
            raise ValueError(
                "Further interpolation requires previous unresolved approximation failure"
            )
    results = {}
    for label, nodes in (("coarse", am.COARSE), ("refined", am.REFINED)):
        for held in (15, 18, 20):
            calibrated = [a for a in nodes if a != held]
            target = ages == held
            predictions = np.full_like(y, np.nan)
            direct = np.full_like(y, np.nan)
            folds = []
            for group in sorted(set(groups)):
                train = om.training_mask(groups, ages, group, held) & np.isin(
                    ages, calibrated
                )
                test = (groups == group) & target
                filename = f"atlas-{label}-{held}-{group}.json"
                if args.family == "linear":
                    model = am.fit(x[train], ages[train], y[train], calibrated)
                    pm.save_json(out / filename, model)
                    model_path = out / filename
                else:
                    model_path = ROOT / "fit-linear-01" / filename
                    model = load(model_path)
                predictions[test] = am.predict_panel(
                    model, x[test], ages[test], args.family
                )
                direct[test] = am.predict_panel(
                    model, x[test], ages[test], args.family, True
                )
                folds.append(
                    {
                        "held_group": int(group),
                        "held_age": held,
                        "nodes": calibrated,
                        "training_indices": np.flatnonzero(train).tolist(),
                        "test_indices": np.flatnonzero(test).tolist(),
                        "model": str(model_path),
                        "model_sha256": digest(model_path),
                    }
                )
                budget.check()
            ii = np.flatnonzero(target)
            tr = [rows[i] for i in ii]
            n = am.neighbors(calibrated, held, args.family)
            bracket = am.neighbors(calibrated, held, "linear")
            prefix = f"{label}-{held}"
            save_npz(
                out / f"{prefix}-predictions.npz",
                indices=ii,
                y=predictions[target],
                direct=direct[target],
            )
            pm.save_json(out / f"{prefix}-folds.json", folds)
            summaries = {}
            for mode, prediction in (("coefficient", predictions), ("direct", direct)):
                gates = scores(y[target], prediction[target], tr, floor)
                pm.save_json(out / f"{prefix}-{mode}-scores.json", gates)
                summaries[mode] = summary(gates)
            results[prefix] = {
                "neighbors": np.asarray(calibrated)[n].tolist(),
                "bracket": np.asarray(calibrated)[bracket].tolist(),
                "bracket_width": float(np.diff(np.asarray(calibrated)[bracket])[0]),
                "max_neighbor_distance": float(
                    max(abs(np.asarray(calibrated)[n] - held))
                ),
                "direct_max_abs_difference": float(
                    abs(predictions[target] - direct[target]).max()
                ),
                **summaries,
            }
    eligible = [
        mode
        for mode in ("coefficient", "direct")
        if all(
            results[f"refined-{a}"][mode]["failed"] == 0
            and results[f"refined-{a}"][mode]["unresolved"] == 0
            and results[f"refined-{a}"][mode]["by_kind"]["D1"]["worst_relative"]
            <= 0.8 * results[f"coarse-{a}"][mode]["by_kind"]["D1"]["worst_relative"]
            for a in (15, 18, 20)
        )
    ]
    pm.save_json(
        out / "decision.json",
        {
            "status": "DEVELOPMENT_QUALIFIED"
            if eligible
            else "LOCAL_INTERPOLATION_LIMIT",
            "family": args.family,
            "eligible_modes": eligible,
            "results": results,
        },
    )
    budget.finish()


def supported(x, ages, nodes, model):
    q = (x - model["mean"]) / model["scale"]
    keep = np.ones(len(x), bool)
    for age in nodes:
        z = q[ages == age]
        keep &= ((q >= z.min(0)) & (q <= z.max(0))).all(1)
        keep &= np.sqrt(np.mean((q[:, None] - z[None]) ** 2, axis=-1)).min(1) <= 0.5
    return np.flatnonzero(keep)


def curvature(out, args, budget):
    budget.begin("prediction-space-divided-difference-curvature")
    x, _y, _rows, ages, _groups = combined()
    model = load(ROOT / "fit-linear-01/all-node-model.json")
    records, curves, ids = [], [], []
    nodes = model["nodes"]
    for j in range(1, len(nodes) - 1):
        triple = nodes[j - 1 : j + 2]
        ii = supported(x, ages, triple, model)
        for i in ii:
            for amp in (-0.02, 0.02):
                p = np.array(
                    [
                        am.predict(model, x[i], amp, a)
                        - am.predict(model, x[i], 0.0, a)
                        for a in triple
                    ]
                )
                c = (p[2] - p[1]) / (triple[2] - triple[1]) - (p[1] - p[0]) / (
                    triple[1] - triple[0]
                )
                curves.append(p)
                ids.append([j, i, amp])
                record = {
                    "ages": triple,
                    "center_index": int(i),
                    "amplitude": amp,
                    "windows": {
                        win: rms(c[sl]).tolist() for win, sl in WINDOWS.items()
                    },
                    "second_divided_difference": (
                        2 * rms(c) / (triple[2] - triple[0])
                    ).tolist(),
                    "basis_coordinates": np.einsum(
                        "to,kt->ok", c, np.asarray(model["bases"])[1]
                    ).tolist(),
                }
                records.append(record)
    save_npz(out / "curvature.npz", curves=curves, ids=ids, x=x)
    pm.save_json(out / "curvature.json", records)
    pm.save_json(
        out / "summary.json",
        {
            "support_rule": "Coordinate intersection and nearest standardized RMS <= .5 at all three nodes",
            "triples": [
                {
                    "ages": nodes[j - 1 : j + 2],
                    "supported_inputs": len(
                        supported(x, ages, nodes[j - 1 : j + 2], model)
                    ),
                    "max_curvature": np.max(
                        [
                            r["windows"]["whole"]
                            for r in records
                            if r["ages"] == nodes[j - 1 : j + 2]
                        ],
                        axis=0,
                    ).tolist(),
                }
                for j in range(1, len(nodes) - 1)
            ],
        },
    )
    local_records = []
    for label in ("coarse", "refined"):
        for held in (15, 18, 20):
            folds = load(ROOT / "fit-linear-01" / f"{label}-{held}-folds.json")
            for fold in folds:
                m = load(fold["model"])
                ns = np.asarray(m["nodes"])
                triple = ns[am.neighbors(ns, held, "quadratic")]
                tr = np.asarray(fold["training_indices"])
                local_x, local_ages = x[tr], ages[tr]
                ii = supported(local_x, local_ages, triple, m)
                values = []
                for i in ii:
                    for amp in (-0.02, 0.02):
                        p = np.array(
                            [
                                am.predict(m, local_x[i], amp, a)
                                - am.predict(m, local_x[i], 0.0, a)
                                for a in triple
                            ]
                        )
                        c = (p[2] - p[1]) / (triple[2] - triple[1]) - (p[1] - p[0]) / (
                            triple[1] - triple[0]
                        )
                        values.append(rms(c))
                local_records.append(
                    {
                        "atlas": label,
                        "held_age": held,
                        "held_group": fold["held_group"],
                        "nodes": triple.tolist(),
                        "supported_inputs": len(ii),
                        "max_curvature": np.max(values, axis=0).tolist()
                        if values
                        else None,
                        "target_age_used": False,
                    }
                )
                budget.check()
    pm.save_json(out / "held-context-curvature.json", local_records)
    budget.finish()


def report(out, args, budget):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budget.begin("atlas-report-and-adverse-case")
    x, y, _rows, ages, _groups = combined()
    decisions = {}
    table = []
    for family in ("linear", "quadratic", "pchip"):
        folder = ROOT / f"fit-{family}-01"
        if not folder.exists():
            continue
        d = load(folder / "decision.json")
        decisions[family] = d
        for key, r in d.get("results", {}).items():
            for mode in ("coefficient", "direct"):
                table.append(
                    {
                        "family": family,
                        "atlas_target": key,
                        "mode": mode,
                        "bracket": r["bracket"],
                        "bracket_width": r["bracket_width"],
                        "neighbors": r["neighbors"],
                        "max_neighbor_distance": r["max_neighbor_distance"],
                        "summary": r[mode],
                    }
                )
    pm.save_json(out / "comparison.json", table)
    family = list(decisions)[-1]
    folder = ROOT / f"fit-{family}-01"
    # Diagnostic selection AFTER all predictions/gates are saved; never a fit input.
    records = []
    for age in (15, 18, 20):
        for g in load(folder / f"refined-{age}-coefficient-scores.json"):
            records.append(dict(**g, held_age=age))
    worst = max((g for g in records if g["kind"] == "D1"), key=lambda g: g["relative"])
    age = worst["held_age"]
    saved = np.load(folder / f"refined-{age}-predictions.npz", allow_pickle=False)
    local = worst["index"]
    i = int(saved["indices"][local])
    pred = saved["y"][local]

    save_npz(out / "adverse.npz", truth=y[i], prediction=pred, x=x[i])
    pm.save_json(
        out / "adverse.json",
        dict(
            **worst,
            family=family,
            global_index=i,
            error_over_floor=worst["error_rms"] / worst["floor"],
        ),
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    j = 1 if worst["a"] < 0 else 2
    for o, name in enumerate(("C mass", "C signed moment")):
        axes[0, o].plot(pm.TIMES, y[i, j, :, o], label="true R_after")
        axes[0, o].plot(pm.TIMES, pred[j, :, o], "--", label="atlas R_after")
        axes[1, o].plot(pm.TIMES, (y[i, j] - y[i, 0])[:, o], label="true D1")
        axes[1, o].plot(pm.TIMES, (pred[j] - pred[0])[:, o], "--", label="atlas D1")
        axes[0, o].set_title(name)
        for ax in axes[:, o]:
            ax.legend(fontsize=8)
            ax.set_xlabel("h since final probe")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.suptitle(
        f"Adverse held age {age}: {worst['seed']} {worst['history']}, a={worst['a']:+g}"
    )
    fig.savefig(out / "adverse-response.png", dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, held in zip(axes, (15, 18, 20), strict=True):
        for family, d in decisions.items():
            for mode, style in (("coefficient", "o-"), ("direct", "x--")):
                vals = [
                    d["results"][f"{atlas}-{held}"][mode]["by_kind"]["D1"][
                        "worst_relative"
                    ]
                    * 100
                    for atlas in ("coarse", "refined")
                ]
                ax.plot([0, 1], vals, style, label=family + "/" + mode)
        ax.axhline(10, color="black", linestyle=":")
        ax.set_xticks([0, 1], ["coarse", "refined"])
        ax.set_title(f"Held age {held}")
        ax.set_ylabel("Worst D1 relative RMS (%)")
    axes[-1].legend(fontsize=7)
    fig.savefig(out / "resolution-comparison.png", dpi=150)
    plt.close(fig)
    c = load(ROOT / "curvature-01/summary.json")["triples"]
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for o, name in enumerate(("mass", "signed moment")):
        ax.plot(
            [r["ages"][1] for r in c],
            [r["max_curvature"][o] for r in c],
            "o-",
            label=name,
        )
    ax.set_xlabel("Middle calibration age (unequal triple spacing)")
    ax.set_ylabel("Maximum RMS slope difference")
    ax.legend()
    fig.savefig(out / "operator-curvature.png", dpi=150)
    plt.close(fig)
    physical = []
    for p in (ROOT / "develop-01").glob("reference-*.json"):
        meta = load(p)
        physical.append(
            {
                "file": str(p),
                "age": meta["age"],
                "amplitude": meta["a"],
                "qualified": meta["qualified"],
                "event": meta["event"],
            }
        )
    pm.save_json(out / "physical-summary.json", physical)
    pm.save_json(
        out / "status.json",
        {
            "development": decisions[family]["status"],
            "fresh_age24": "NOT_RUN",
            "retained_G": "NOT_RUN",
            "repeated": "NOT_RUN",
            "reserved_seeds": [26101, 26102, 26103],
            "new_ages": [14, 17, 25],
            "preparations": 12,
        },
    )
    # Compact new evaluation tables refer to old inputs by their recorded identities.
    save_npz(out / "evaluation-table.npz", x=x, y=y, ages=ages)
    budget.finish()
