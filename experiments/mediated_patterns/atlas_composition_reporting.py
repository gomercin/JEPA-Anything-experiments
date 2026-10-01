"""Read-only decision and numerical margin accounting, after sealed evaluation."""

import numpy as np

from . import present_state_model as pm
from .present_state_transmission import load
from .repeated_intervention_analysis import WINDOWS, rms, truth
from .repeated_intervention_model import contrasts


def numerical_margins(root, stage, base, case, prediction):
    records = load(root / stage / "refinement.json")["records"]
    y, _absolute, _z = truth(base, case)
    floors = load(root / "frozen-01/freeze.json")["floors"]
    result = []
    for window, sl in WINDOWS.items():
        target = contrasts(y)["D12"][sl]
        residual = contrasts(prediction)["D12"][sl] - target
        magnitude, error = rms(target), rms(residual)
        rr = [r for r in records if r["kind"] == "D12" and r["window"] == window]
        kk = [r for r in records if r["kind"] == "K12" and r["window"] == window]
        for o, output in enumerate(["mass", "moment"]):
            bound = max(floors[f"D12/{window}"][o], *(r["propagated"][o] for r in rr))
            margin = error[o] - 0.1 * magnitude[o]
            result.append(
                {
                    "window": window,
                    "output": output,
                    "magnitude": float(magnitude[o]),
                    "error_rms": float(error[o]),
                    "margin": float(margin),
                    "bound": float(bound),
                    "margin_over_1p1_bound": float(margin / (1.1 * bound)),
                    "classification": "robust-failure"
                    if margin > 1.1 * bound
                    else ("robust-pass" if margin < -1.1 * bound else "marginal"),
                    "observed_relative": [float(error[o] / magnitude[o])]
                    + [r["relative"][o] for r in rr],
                    "direct_D12_bounds": [r["direct"][o] for r in rr],
                    "direct_K12_bounds": [r["direct"][o] for r in kk],
                    "evaluator_addition_relative": [
                        r["true_addition_relative"][o] for r in rr
                    ],
                }
            )
    return result


def finish(out, root, budget):
    budget.begin("final-decision-and-cost-accounting")
    scores = load(root / "fresh-analysis-01/scores.json")
    primary = [s for s in scores if s["model"] == "common" and s["kind"] != "K12"]
    isolated = [s for s in scores if s["model"].startswith("isolated")]
    failures = [s for s in primary if s["pass"] is False]
    iso_failures = [s for s in isolated if s["pass"] is False]
    margins = []
    if (root / "fresh-refine-01/refinement.json").exists():
        case = load(root / "fresh-refine-01/selection.json")["case"]
        saved = np.load(
            root
            / "fresh-01"
            / f"prediction-{case['index']}-{case['schedule']['name']}.npz"
        )
        margins = numerical_margins(
            root, "fresh-refine-01", root / "fresh-01", case, saved["common"]
        )
    if iso_failures:
        decision = "ISOLATED_PREREQUISITE_FAILURE"
    elif not failures and all(s["resolved"] for s in primary + isolated):
        decision = "BOUNDED_ORDINARY_ADDITION_SUFFICIENT"
    elif failures and any(m["classification"] == "robust-failure" for m in margins):
        decision = "BOUNDED_COMPOSITION_LIMIT"
    else:
        decision = "INCONCLUSIVE_NUMERICAL_THRESHOLD_MARGIN"
    base = root / "exposed-01"
    case = next(
        c
        for c in load(base / "cases.json")
        if c["seed"] == 18103
        and c["history"] == "none"
        and c["schedule"]["name"] == "timing"
    )
    old_prediction = np.load(base / "prediction-4-timing.npz")["common"]
    exposed = numerical_margins(root, "refine-01", base, case, old_prediction)
    pm.save_json(
        out / "decision.json",
        {
            "decision": decision,
            "groups": 3,
            "histories": 6,
            "schedule_cases": 24,
            "primary_entries": len(primary),
            "primary_failures": failures,
            "isolated_entries": len(isolated),
            "isolated_failures": iso_failures,
            "unresolved_primary": sum(not s["resolved"] for s in primary),
            "fresh_margins": margins,
            "exposed_margins": exposed,
            "gates": "Original 2%/10% targets and inherited floors unchanged; K12 has no gate",
            "interpretation": "A marginal miss is retained, not a pass or a robust failure. No state-insufficiency inference, fit, correction or successor.",
            "scientific_cpu_before_reporting": budget.previous,
        },
    )
    budget.finish()
    budget.begin("scientific-response-figures")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for schedule in ["standard-cancel", "adverse-cancel", "adverse-reinforce"]:
        case = next(
            c
            for c in load(root / "fresh-01/cases.json")
            if c["index"] == 0 and c["schedule"]["name"] == schedule
        )
        y, _aa, _z = truth(root / "fresh-01", case)
        p = np.load(root / "fresh-01" / f"prediction-0-{schedule}.npz")
        oracle = y.copy()
        oracle[3] = y[1] + y[2] - y[0]
        fig, axes = plt.subplots(4, 2, figsize=(10, 9), sharex=True)
        for k, kind in enumerate(["R11", "D12", "D2|1", "K12"]):
            for o, output in enumerate(["mass", "moment"]):
                ax = axes[k, o]
                for label, data, style in [
                    ("true", y, "-"),
                    ("common addition", p["common"], "--"),
                    ("literal", p["literal"], ":"),
                    ("true addition", oracle, "-."),
                ]:
                    ax.plot(pm.TIMES, contrasts(data)[kind][:, o], style, label=label)
                ax.set_ylabel(kind + " " + output)
                ax.legend(fontsize=6)
        fig.suptitle("Fresh preparation 28101 / none / " + schedule)
        axes[-1, 0].set_xlabel("Time after final probe")
        axes[-1, 1].set_xlabel("Time after final probe")
        fig.tight_layout()
        fig.savefig(out / (schedule + ".png"), dpi=130)
        plt.close(fig)
    budget.finish()
