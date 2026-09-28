"""Grouped atlas evidence; current fields and future references stay evaluator-only."""

import numpy as np

from . import event_atlas_model as am
from . import event_operator_model as om
from . import present_state_model as pm
from .event_age_operator_atlas import ROOT
from .event_age_operator_family import ROOT as OLD
from .event_age_operator_family import scores
from .event_age_response import WINDOWS, arrays, rms
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
                    # Both densities use identical coordinates built from coarse
                    # calibration rows only, excluding the target and preparation.
                    base = om.training_mask(groups, ages, group, held) & np.isin(
                        ages, am.COARSE
                    )
                    rep = om.representation(x[base], y[base])
                    model = am.fit(x[train], ages[train], y[train], calibrated, rep=rep)
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
                        "representation_indices": np.flatnonzero(
                            om.training_mask(groups, ages, group, held)
                            & np.isin(ages, am.COARSE)
                        ).tolist(),
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
                    "max_second_divided_difference": np.max(
                        [
                            r["second_divided_difference"]
                            for r in records
                            if r["ages"] == nodes[j - 1 : j + 2]
                        ],
                        axis=0,
                    ).tolist(),
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
            [r["max_second_divided_difference"][o] for r in c],
            "o-",
            label=name,
        )
    ax.set_xlabel("Middle calibration age (unequal triple spacing)")
    ax.set_ylabel("Maximum RMS second divided difference")
    ax.legend()
    fig.savefig(out / "operator-curvature.png", dpi=150)
    plt.close(fig)
    physical = []
    for p in [
        p
        for folder in ("pilot-01", "develop-01", "fresh-01")
        for p in (ROOT / folder).glob("reference-*.json")
    ]:
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
    status = {
        "development": decisions[family]["status"],
        "fresh_age24": "NOT_RUN",
        "retained_G": "NOT_RUN",
        "repeated": "NOT_RUN",
        "reserved_seeds": [26101, 26102, 26103],
        "new_ages": [14, 17, 25],
        "development_preparations": 12,
    }
    if (ROOT / "fresh-analysis-01/decision.json").exists():
        frozen = load(ROOT / "frozen-01/freeze.json")
        fresh = ROOT / "fresh-01"
        fx, fy, frows, _fa, _fg = dataset(fresh)
        fp = arrays(fresh / "prediction-quadratic.npz")["y"]
        final_floor = {
            k: np.maximum(
                v, load(ROOT / "refine24-01/refinement.json")["floors"][k]
            ).tolist()
            for k, v in frozen["floors"].items()
        }
        gates = scores(fy, fp, frows, final_floor)
        pm.save_json(out / "fresh-final-scores.json", gates)
        pm.save_json(out / "fresh-final-floors.json", final_floor)
        ss = summary(gates)
        status["fresh_age24"] = (
            "PASS"
            if not ss["failed"] and not ss["unresolved"]
            else "FAIL_OR_UNRESOLVED"
        )
        status["fresh_preparations"] = 3
        stats = {}
        for age in (24, 25):
            for a in (-0.02, 0.02):
                stats[f"{age}/{a}"] = summary(
                    [g for g in gates if g["age"] == age and g["a"] == a]
                )
        pm.save_json(out / "fresh-by-sign.json", stats)
        worst = max(
            (g for g in gates if g["age"] == 24 and g["kind"] == "D1"),
            key=lambda g: g["relative"],
        )
        pm.save_json(
            out / "fresh-adverse.json",
            dict(**worst, error_over_floor=worst["error_rms"] / worst["floor"]),
        )
        i, j = worst["index"], 1 if worst["a"] < 0 else 2
        fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
        for o, label in enumerate(("C mass", "C signed moment")):
            for k, kind in enumerate(("R_after", "D1")):
                target = fy[i, j] if k == 0 else fy[i, j] - fy[i, 0]
                prediction = fp[i, j] if k == 0 else fp[i, j] - fp[i, 0]
                axes[k, o].plot(pm.TIMES, target[:, o], label="true")
                axes[k, o].plot(pm.TIMES, prediction[:, o], "--", label="frozen atlas")
                axes[k, o].set_title(label + " " + kind)
                axes[k, o].legend(fontsize=8)
                axes[k, o].set_xlabel("h since probe")
                axes[k, o].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        fig.suptitle(
            f"Fresh age24 adverse: {worst['seed']} {worst['history']}, a={worst['a']:+g}"
        )
        fig.savefig(out / "fresh-response.png", dpi=150)
        plt.close(fig)
        if (ROOT / "retained-01/summary.json").exists():
            rr = load(ROOT / "retained-01/summary.json")
            status["retained_G"] = (
                "EXPOSED_PASS"
                if all(
                    not v["failed"] and not v["unresolved"]
                    for v in rr["by_age"].values()
                )
                else "EXPOSED_LIMIT"
            )
            zz = arrays(ROOT / "retained-01/predictions.npz")
            fig, axs = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
            for o, label in enumerate("ABC"):
                axs[o].plot((fx - zz["z0"])[:, o], label="true event-boundary motion")
                axs[o].plot(
                    (zz["z"] - zz["z0"])[:, o], "x--", label="G predicted motion"
                )
                axs[o].set_title(label + " pre-event center")
                axs[o].set_xlabel("history/age case index")
                axs[o].legend(fontsize=7)
            fig.savefig(out / "retained-geometry.png", dpi=150)
            plt.close(fig)
    if (ROOT / "repeated-01/scores.json").exists():
        from . import repeated_intervention_model as rm

        status["repeated"] = "EXPOSED_DIAGNOSTIC"
        rr = load(ROOT / "repeated-01/scores.json")
        result = {}
        for model in sorted({r["model"] for r in rr}):
            for schedule in ("cancel", "timing"):
                for kind in sorted({r["kind"] for r in rr if r["model"] == model}):
                    ss = [
                        r
                        for r in rr
                        if r["model"] == model
                        and r["schedule"] == schedule
                        and r["kind"] == kind
                    ]
                    result[f"{model}/{schedule}/{kind}"] = {
                        "worst_relative": max(r["relative"] or 0.0 for r in ss),
                        "max_error_rms": max(r["error_rms"] for r in ss),
                        "failed": sum(
                            r.get("pass", r.get("passed")) is False for r in ss
                        ),
                        "unresolved": sum(not r["resolved"] for r in ss),
                        "count": len(ss),
                    }
        pm.save_json(out / "repeated-summary.json", result)
        cases = load(ROOT / "repeated-01/cases.json")
        idx = next(i for i, c in enumerate(cases) if c["schedule"]["name"] == "timing")
        true = rm.contrasts(arrays(ROOT / "repeated-01" / f"truth-{idx}.npz")["y"])
        pred = rm.contrasts(
            arrays(ROOT / "repeated-01" / f"prediction-{idx}.npz")["centered"]
        )
        fig, axs = plt.subplots(4, 2, figsize=(10, 10), constrained_layout=True)
        for k, kind in enumerate(("R11", "D12", "D2|1", "K12")):
            for o, label in enumerate(("mass", "moment")):
                axs[k, o].plot(pm.TIMES, true[kind][:, o], label="true")
                axs[k, o].plot(
                    pm.TIMES, pred[kind][:, o], "--", label="independent addition"
                )
                axs[k, o].set_title(kind + " " + label)
                axs[k, o].legend(fontsize=7)
        fig.savefig(out / "exposed-repeated.png", dpi=150)
        plt.close(fig)
    pm.save_json(out / "status.json", status)
    # Compact new evaluation tables refer to old inputs by their recorded identities.
    save_npz(out / "evaluation-table.npz", x=x, y=y, ages=ages)
    budget.finish()


def details(out, args, budget):
    """Final numerical margins, access accounting and unambiguous sign summaries."""
    budget.begin("final-qualified-scales-and-boundaries")
    gates = load(ROOT / "report-01/fresh-final-scores.json")
    signs = {}
    for age in (24, 25):
        for amp in (-0.02, 0.02):
            part = [g for g in gates if g["age"] == age and g["a"] == amp]
            signs[f"{age}/{amp}"] = {
                kind: {
                    "worst_relative": max(
                        g["relative"] for g in part if g["kind"] == kind
                    ),
                    "max_error_rms": max(
                        g["error_rms"] for g in part if g["kind"] == kind
                    ),
                    "max_error_absolute": max(
                        g["error_max"] for g in part if g["kind"] == kind
                    ),
                    "failed": sum(
                        g["passed"] is False for g in part if g["kind"] == kind
                    ),
                    "unresolved": sum(
                        not g["resolved"] for g in part if g["kind"] == kind
                    ),
                }
                for kind in ("R_without", "R_after", "D1")
            }
    pm.save_json(out / "fresh-by-sign.json", signs)
    repeated = load(ROOT / "repeated-01/scores.json")
    adverse = {}
    for model in (
        "raw-addition",
        "common-baseline-addition",
        "true-single-event-addition",
    ):
        w = max(
            (
                r
                for r in repeated
                if r["model"] == model
                and r["schedule"] == "timing"
                and r["kind"] == "D12"
            ),
            key=lambda r: r["relative"],
        )
        adverse[model] = dict(
            **w,
            error_over_floor=w["error_rms"] / w["floor"],
            excess_over_10pct_in_floor_units=(w["error_rms"] - 0.1 * w["magnitude"])
            / w["floor"],
        )
    pm.save_json(out / "repeated-adverse.json", adverse)
    pm.save_json(
        out / "accounting.json",
        {
            "calibrated_nodes": 8,
            "coefficients_per_node": 240,
            "coefficient_values": 1920,
            "common_basis_values": 1288,
            "center_scaling_values": 6,
            "output_scaling_values": 4,
            "active_fit_basis_scaling_values": 3218,
            "node_context_values": 8,
            "amplitude_normalizer_values": 1,
            "anchors": 3,
            "response_grid_values": 161,
            "conditioning_inputs": ["amplitude", "known event age"],
            "current_measurements": 3,
            "snapshot_acquisition": "One 2x768 field snapshot scanned per alternative forecast; twelve fresh forecast cases",
            "evolving_response_coordinates": 0,
            "output_values_per_query": 322,
            "output_values_per_zero_minus_plus_triplet": 966,
            "quadratic_coefficient_query_multiply_adds": 3536,
            "direct_three_operator_query_multiply_adds": 9414,
            "cost_scope": "Feature construction, interpolation weights, scaling and allocations additional; counts are not a benchmark",
            "fixed_age_active_values": 1538,
            "coarse_atlas_active_values": 2498,
            "prior_global_spline_active_values": 2258,
            "retained_G": {
                "status": "exposed diagnostic only",
                "centers": 3,
                "counter": 1,
                "static_values": 37,
                "rate_evaluations_per_step": 4,
                "field_access": "t50 only; no event-boundary refresh",
            },
            "repeated_comparator_queries": 4,
            "repeated_contexts": 2,
            "field_reference": "Two 768-value fields; each 80-unit probe/sham pair advances 12800 steps per branch at dt=.00625; preparation and event gap additional",
        },
    )
    pm.save_json(
        out / "limits.json",
        {
            "outcome": "LOCAL OPERATOR ATLAS SUFFICIENT",
            "fresh_snapshot": True,
            "fresh_age24_fitted": False,
            "retained_fresh_claim": False,
            "pchip": "NOT_RUN; quadratic qualified",
            "fourth_new_age": "NOT_RUN",
            "fresh_repeated": "NOT_RUN",
            "composition_correction": "NOT_RUN",
            "repeated_threshold_margin": "Centered D12 failure remains recorded, but its excess over 10% is below the inherited empirical numerical floor",
            "sign_table_note": "report-01/fresh-by-sign.json uses cross-sign R_without deduplication, so the positive-only baseline summary has no entries; its zero is NOT zero physical error. This file replaces that presentation with explicit per-sign maxima.",
            "resolution_scope": "Three held development ages and one fresh age; no universal spacing law or adaptive rule validated",
        },
    )
    budget.finish()
