"""Qualify the worst fresh intermediate-age history without changing frozen gates."""

import argparse
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import resample

from experiments.mediated_patterns import present_state_model as pm
from experiments.mediated_patterns.event_age_response import (
    ROOT,
    WINDOWS,
    arrays,
    paired,
    rms,
    slots,
)
from experiments.mediated_patterns.geometry_assay import initialize_written
from experiments.mediated_patterns.geometry_evolution import unforced
from experiments.mediated_patterns.history_conditioned_transmission import HistoryBudget
from experiments.mediated_patterns.present_state_transmission import load
from experiments.mediated_patterns.repeated_intervention_state import prefix_reference
from experiments.mediated_patterns.source_receiver_relay import CFG


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    out = args.output
    if out.parent.resolve() != ROOT.resolve() or any(
        p.is_symlink() for p in [out, *out.parents]
    ):
        raise ValueError("Unique nonsymlink output required")
    data = ROOT / "fresh-01"
    scores = load(ROOT / "fresh-analysis-01/scores.json")
    worst = max(
        (
            r
            for r in scores
            if r["model"] == "causal" and r["kind"] == "D1" and r["age"] == 18
        ),
        key=lambda r: r["relative"],
    )
    case = next(
        c
        for c in load(data / "cases.json")
        if all(c[k] == worst[k] for k in ["seed", "history", "age", "a"])
    )
    previous = (
        30
        + sum(load(p)["cpu_seconds"] for p in ROOT.glob("*/budget.json"))
        + load(ROOT / "runtime-audit.json")["cpu_seconds"]
    )
    budget = HistoryBudget(previous, 1800)
    out.mkdir(exist_ok=False)
    status = "FAILED"
    try:
        pm.save_json(
            out / "selection.json",
            {
                "case": case,
                "worst": worst,
                "rule": "worst causal D1 relative error at fresh intermediate age; frozen gates unchanged",
            },
        )
        initial = arrays(data / f"s{case['seed']}-initial.npz")["initial"]
        ds = [paired(data / (n + ".npz")) for n in case["prefixes"]]
        y = np.array([d["response"] for d in ds])
        absolute = np.array([d["absolute"] for d in ds])
        floors = {}
        details = []
        for label, cfg in [
            ("halfdt", replace(CFG, dt=CFG.dt / 2)),
            ("doubleN", replace(CFG, n=CFG.n * 2)),
        ]:
            sub = out / label
            sub.mkdir()
            budget.begin(label + "-prepare")
            init = initial if cfg.n == CFG.n else resample(initial, cfg.n, axis=-1)
            state = unforced(
                initialize_written(init, case["history"], cfg), cfg, [0.0, 50.0], budget
            )[0][-1]
            budget.finish()
            for a in [0.0, case["a"]]:
                t, aa = slots(case["age"], a)
                prefix_reference(sub, f"reference-{a:g}", state, t, aa, budget, cfg)
            rr = [paired(sub / f"reference-{a:g}.npz") for a in [0.0, case["a"]]]
            r = np.array([d["response"] for d in rr])
            ab = np.array([d["absolute"] for d in rr])
            for window, sl in WINDOWS.items():
                rounding = (
                    64
                    * np.finfo(float).eps
                    * abs(absolute[:, :, :, 2, :2]).max(axis=(0, 1, 2))
                )
                pair = np.maximum(5 * rms((r - y)[:, sl], axis=1), rounding)
                branches = 5 * rms((ab - absolute)[:, sl, :, 2, :2], axis=1)
                key = "D1/" + window
                floors[key] = np.maximum(floors.get(key, np.zeros(2)), pair.sum(axis=0))
                details.append(
                    {
                        "refinement": label,
                        "window": window,
                        "pair": pair.tolist(),
                        "floor": pair.sum(axis=0).tolist(),
                        "direct": (
                            5 * rms(((r[1] - r[0]) - (y[1] - y[0]))[sl])
                        ).tolist(),
                        "uncorrelated_absolute_bound": branches.sum(
                            axis=(0, 1)
                        ).tolist(),
                    }
                )
        pm.save_json(
            out / "refinement.json",
            {
                "floors": {k: v.tolist() for k, v in floors.items()},
                "records": details,
                "frozen_gates_changed": False,
            },
        )
        status = "COMPLETE"
    finally:
        pm.save_json(out / "budget.json", dict(budget.receipt(), status=status))


if __name__ == "__main__":
    main()
