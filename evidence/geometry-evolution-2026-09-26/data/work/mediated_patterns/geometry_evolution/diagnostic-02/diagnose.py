"""Post-freeze explanatory diagnostics only: no refit, selection or repair."""

import os
import sys
import time
from pathlib import Path
import numpy as np

from experiments.mediated_patterns import geometry_model as gm
from experiments.mediated_patterns import present_state_model as pm
from experiments.mediated_patterns.geometry_evolution import (
    ROOT,
    center_rates,
    digest,
    load,
)
from experiments.mediated_patterns.history_conditioned_transmission import HistoryBudget

out = Path(__file__).parent
fd = os.open(ROOT / ".active", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
os.close(fd)
previous = 30 + sum(load(p)["cpu_seconds"] for p in ROOT.glob("*/budget.json"))
budget = HistoryBudget(previous, 1800)
status = "FAILED"
try:
    budget.begin("weak-motion-rate-and-error-diagnostics")
    frozen = ROOT / "frozen-01"
    fresh = ROOT / "fresh-01"
    analysis = ROOT / "analysis-final"
    models = load(frozen / "models.json")
    contract = load(frozen / "freeze.json")
    selected = contract["selected"]
    rows = load(fresh / "rows.json")
    i = rows.index({"seed": 14103, "history": "odd04"})
    with np.load(fresh / "boundary-50.npz", allow_pickle=False) as a:
        initial = a["states"][i]
    with np.load(fresh / "evaluator-states.npz", allow_pickle=False) as a:
        fields = np.concatenate([initial[None], a["states"][i]])
    times = np.array([50, 60, 75, 90, 100])
    z = np.array([pm.extract(s, feature_set="geometry") for s in fields])
    rate_start = time.process_time()
    rates = np.array([center_rates(s) for s in fields])
    rate_cpu = time.process_time() - rate_start
    fitted_rates = {
        k: np.array([gm.velocity(g, s) for s in z]) for k, g in models.items()
    }
    g = models[selected]
    prediction = gm.rollout(g, z[0], times - 50)
    refined = gm.rollout(dict(g, step=0.5), z[0], times - 50)
    summary = load(analysis / "summary.json")
    comparison = {}
    for model, scores in summary["scores"].items():
        comparison[model] = {}
        for label, subset in [
            ("whole", [r for r in scores if r["window"] == "whole"]),
            ("late", [r for r in scores if r["window"] == "late"]),
            ("withheld90", [r for r in scores if r["wait"] == 90]),
            ("origin50", [r for r in scores if r["origin"] == 50]),
            ("origin60", [r for r in scores if r["origin"] == 60]),
        ]:
            comparison[model][label] = {
                kind + "_" + output: max(
                    r["relative_rms"]
                    for r in subset
                    if r["kind"] == kind and r["output"] == output
                )
                for kind in ["R", "Delta_R"]
                for output in ["mass", "moment"]
            }
    with np.load(ROOT / "development-01/trajectories.npz", allow_pickle=False) as a:
        dz, features = a["z"], a["features"]
    fit = load(ROOT / "fit-02/fit.json")
    selected_times = [0, 1, 2, 3, 4, 5, 10]
    corr = np.array(fit["feature_age_correlation"])
    folds = {}
    for candidate in ["affine-1e-06", selected]:
        with np.load(ROOT / "fit-02/rollouts.npz", allow_pickle=False) as a:
            p = a[candidate]
        values = []
        for b, start in enumerate([0, 2]):
            ti = [k for k in selected_times if k > start]
            ratio = np.sqrt(np.mean((p[:, b, ti] - dz[:, ti]) ** 2, axis=1)) / np.sqrt(
                np.mean((dz[:, ti] - dz[:, start, None]) ** 2, axis=1)
            )
            values.append(ratio.tolist())
        folds[candidate] = values
    pm.save_json(
        out / "diagnostic.json",
        {
            "rate_calls": len(fields),
            "rate_cpu_seconds": rate_cpu,
            "seed": 14103,
            "history": "odd04",
            "time": times.tolist(),
            "z": z.tolist(),
            "instantaneous_reference_rates": rates.tolist(),
            "rates_at_true_geometry": {k: v.tolist() for k, v in fitted_rates.items()},
            "rate_residual_at_true_geometry": (fitted_rates[selected] - rates).tolist(),
            "rollout_coordinate_error": (prediction - z).tolist(),
            "half_step_max_difference": float(abs(refined - prediction).max()),
            "comparison": comparison,
            "development_relative_motion": folds,
            "geometry_to_other_feature_correlation": corr[:3, 3:15].tolist(),
            "geometry_to_age_correlation": corr[:3, -1].tolist(),
            "snapshot_update_error_identity": "Signed residuals and cross terms in analysis-final; no causal percentages",
            "exposure": "All fresh outcomes exposed. Diagnostic only; models, gates, coefficients and fresh predictions unchanged.",
        },
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax.plot(times - 50, 1e6 * (z[:, 0] - z[0, 0]), "ko", label="actual A displacement")
    for name, color in [
        (selected, "#b04778"),
        ("affine-1e-06", "#2a8b80"),
        ("persistence", "#777777"),
    ]:
        pp = gm.rollout(models[name], z[0], np.arange(51.0))
        ax.plot(np.arange(51.0), 1e6 * (pp[:, 0] - z[0, 0]), color=color, label=name)
    ax.set(
        xlabel="unobserved delay from 50",
        ylabel="A displacement × 1,000,000",
        title="Resolved small-motion miss: seed14103, odd +.4",
    )
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out / "weak-motion.png", dpi=150)
    plt.close(fig)
    pm.save_json(
        out / "protocol.json",
        {
            "command": sys.argv,
            "script_sha256": digest(__file__),
            "frozen_models_sha256": digest(frozen / "models.json"),
            "sources_unchanged_from_freeze": contract["sources"]
            == {
                p.name: digest(p)
                for p in Path("experiments/mediated_patterns").glob("*.py")
            },
        },
    )
    budget.finish()
    status = "COMPLETE"
finally:
    pm.save_json(out / "budget.json", dict(**budget.receipt(), status=status))
    (ROOT / ".active").unlink()
