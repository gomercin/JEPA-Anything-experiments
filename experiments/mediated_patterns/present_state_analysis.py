"""Metered saved-data selection, sealing, scoring and figures; no fresh refitting."""

import argparse
import time
from pathlib import Path

import numpy as np

from . import present_state_model as m
from . import present_state_transmission as run


def freeze(out, fit, refinement, selected):
    report = run.load(fit / "fit.json")
    models = run.load(fit / "models.json")
    best = report["best"]
    keep = {s: models[best[s]] for s in ["blind", "b", selected]}
    # Keep only deployment data: diagnostics live in the fit evidence.
    excluded = {"projection_relative_rms", "singular_values", "condition"}
    keep = {
        s: {k: v for k, v in value.items() if k not in excluded}
        for s, value in keep.items()
    }
    m.save_json(out / "models.json", keep)
    m.save_json(
        out / "freeze.json",
        {
            "selected": selected,
            "fit": str(fit),
            "fit_sha256": run.digest(fit / "fit.json"),
            "models_sha256": run.digest(out / "models.json"),
            "sources": {
                p.name: run.digest(p) for p in Path(__file__).parent.glob("*.py")
            },
            "fresh_seeds": list(run.FRESH),
            "families": {"none": [0, 0], "odd04": [0.4, 0], "heldmix": [0.2, -0.2]},
            "withheld_history_reason": "Intermediate positive odd component with opposite even component; neither mixture nor negative even history fitted",
            "limits": {"R": 0.02, "Delta_R": 0.10},
            "late_window": [40, 80],
            "floors": run.load(refinement / "refinement.json")["floors"],
            "refinement": str(refinement),
            "floor_rule": "max(1e-12,5*max_refinement(sum_two_separate_response_RMS_errors)); shared per readout/window; subfloor unresolved",
            "no_refit_after_fresh": True,
        },
    )


def evaluate(out, data, frozen, refinement=None):
    contract = run.load(frozen / "freeze.json")
    seal = run.load(data / "prediction-seal.json")
    if seal["predictions_sha256"] != run.digest(data / "predictions.npz"):
        raise ValueError("Sealed predictions changed")
    rows = run.load(data / "rows.json")
    y = run.npz(data / "targets.npz")["y"]
    predicted = run.npz(data / "predictions.npz")
    pairs = run.paired_indices(rows)
    floors = np.asarray(contract["floors"])
    if refinement:
        floors = np.maximum(floors, run.load(refinement / "refinement.json")["floors"])
    results = {
        name: m.scores(y, pred, pairs, rows, floors) for name, pred in predicted.items()
    }
    worst = {}
    for name, scores in results.items():
        worst[name] = {
            kind: {
                output: max(
                    (r for r in scores if r["kind"] == kind and r["output"] == output),
                    key=lambda r: r["relative_rms"],
                )
                for output in ["mass", "moment"]
            }
            for kind in ["R", "Delta_R"]
        }
    selected = contract["selected"]
    maximum = max(
        (r for r in results[selected] if r["kind"] == "Delta_R"),
        key=lambda r: r["relative_rms"],
    )
    m.save_json(
        out / "summary.json",
        {
            "selected": selected,
            "floors": floors.tolist(),
            "scores": results,
            "worst": worst,
            "worst_contrast_for_refinement": maximum,
            "all_selected_pass": all(r["passed"] for r in results[selected]),
            "fresh_seed_count": len({r["seed"] for r in rows}),
            "prediction_seal": seal,
        },
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for o, output in enumerate(["mass", "moment"]):
        row = worst[selected]["Delta_R"][output]
        j = next(
            i
            for i, r in enumerate(rows)
            if all(r[k] == row[k] for k in ["seed", "wait", "history"])
        )
        i = next(
            i
            for i, r in enumerate(rows)
            if r["seed"] == row["seed"]
            and r["wait"] == row["wait"]
            and r["history"] == "none"
        )
        p = predicted[selected]
        for index, color, label in [
            (i, "tab:blue", "unwritten"),
            (j, "tab:orange", row["history"]),
        ]:
            axes[o, 0].plot(
                m.TIMES, y[index, :, o], color=color, label=label + " actual"
            )
            axes[o, 0].plot(
                m.TIMES, p[index, :, o], "--", color=color, label=label + " predicted"
            )
        axes[o, 1].plot(m.TIMES, y[j, :, o] - y[i, :, o], label="actual contrast")
        axes[o, 1].plot(
            m.TIMES, p[j, :, o] - p[i, :, o], "--", label="predicted contrast"
        )
        axes[o, 1].axhline(0, color="grey", lw=0.6)
        for c, kind in enumerate(["R", "Delta R"]):
            axes[o, c].set_title(
                f"C {output}, {kind}: seed {row['seed']}, wait {row['wait']:g}, {row['history']}"
            )
            axes[o, c].set_xlabel("Time since probe")
            axes[o, c].legend(fontsize=8)
            axes[o, c].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.suptitle("Frozen snapshot predictor: worst history contrast for each readout")
    fig.savefig(out / "predictions.png", dpi=160)
    plt.close(fig)
    print(
        {
            "all_selected_pass": all(r["passed"] for r in results[selected]),
            "worst": worst,
            "refine": maximum,
        },
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["freeze", "evaluate"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--refinement", type=Path)
    p.add_argument("--freeze", type=Path)
    p.add_argument("--selected", choices=list(m.SETS), default="geometry")
    args = p.parse_args()
    previous = 30 + sum(
        run.load(x)["cpu_seconds"] for x in run.ROOT.glob("*/budget.json")
    )
    start = time.process_time()
    if args.output.parent.resolve() != run.ROOT.resolve() or any(
        x.is_symlink() for x in [args.output, *args.output.parents]
    ):
        raise ValueError("Unique direct output child required")
    args.output.mkdir(exist_ok=False)
    status = "FAILED"
    try:
        if previous >= (1200 if args.stage == "freeze" else 1790):
            raise RuntimeError("No analysis budget remains")
        m.save_json(
            args.output / "protocol.json",
            {
                "sources": {
                    p.name: run.digest(p) for p in Path(__file__).parent.glob("*.py")
                },
                "command": __import__("sys").argv,
            },
        )
        if args.stage == "freeze":
            freeze(args.output, args.data, args.refinement, args.selected)
        else:
            evaluate(args.output, args.data, args.freeze, args.refinement)
        status = "COMPLETE"
    finally:
        m.save_json(
            args.output / "budget.json",
            {
                "cpu_seconds": time.process_time() - start,
                "previous_cpu_seconds": previous,
                "status": status,
            },
        )


if __name__ == "__main__":
    main()
