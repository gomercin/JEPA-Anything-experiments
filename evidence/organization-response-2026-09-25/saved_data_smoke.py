"""Read saved configuration, coefficients, thresholds and traces; no experiment imports."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def inspect(root):
    work = root / "work/mediated_patterns/organization_response"
    freeze_path = work / "frozen-01/freeze.json"
    freeze = json.loads(freeze_path.read_text())
    summary = json.loads((work / "analysis-01/summary.json").read_text())
    checks = json.loads((work / "review-checks-01/checks.json").read_text())
    budget = json.loads((work / "review-checks-01/budget.json").read_text())
    if hashlib.sha256(freeze_path.read_bytes()).hexdigest() != summary["freeze_sha256"]:
        raise ValueError("Saved summary/freeze identity mismatch")
    if (work / "fresh-01/freeze-copy.json").read_bytes() != freeze_path.read_bytes():
        raise ValueError("Fresh panel's recorded freeze differs")
    preparations = []
    for path in sorted(work.glob("*/s*-initial.npz")):
        with np.load(path, allow_pickle=False) as state:
            if set(state.files) != {"state", "x"}:
                raise ValueError("Unexpected preparation serialization")
            shapes = {k: list(state[k].shape) for k in state.files}
            if shapes != {
                "state": [2, freeze["config"]["n"]],
                "x": [freeze["config"]["n"]],
            }:
                raise ValueError("Unexpected preparation dimensions")
            if not all(np.isfinite(state[k]).all() for k in state.files):
                raise ValueError("Nonfinite saved preparation")
            preparations.append({"path": str(path.relative_to(root)), "shapes": shapes})
    traces = []
    for path in sorted((work / "fresh-01").glob("s*.json")):
        row = json.loads(path.read_text())
        shape = np.asarray(row["full"]).shape
        if shape != (161, 2, 2) or row["time"][-1] != freeze["horizon"]:
            raise ValueError("Unexpected fresh saved trace")
        if not np.isfinite(np.asarray(row["return"])).all():
            raise ValueError("Nonfinite saved return contrast")
        traces.append({"preparation_id": row["preparation_id"], "shape": list(shape)})
    if len(preparations) != 10 or len(traces) != 6:
        raise ValueError("Incomplete saved preparation/trace inventory")
    return {
        "script": str(Path(__file__).resolve()),
        "root": str(root),
        "config": freeze["config"],
        "gain_fits": {k: v["gain_delay"] for k, v in freeze["models"].items()},
        "criteria": freeze["criteria"],
        "floors": freeze["floors"],
        "development_seeds": freeze["development_seeds"],
        "fresh_seeds": freeze["fresh_seeds"],
        "saved_comparator_table": [
            {k: row[k] for k in ("seed", "kind", "weighted_nrmse", "practical_failure")}
            for row in summary["fresh_comparator"]
        ],
        "historical_execution_checks": checks,
        "historical_budget_charge_seconds": budget["total_budget_charge_seconds"],
        "preparations": preparations,
        "fresh_traces": traces,
        "imports": "stdlib and numpy only; pickle disabled",
        "experiments_executed": False,
        "independent_scientific_reproduction": False,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    args = p.parse_args()
    print(json.dumps(inspect(args.root.resolve()), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
