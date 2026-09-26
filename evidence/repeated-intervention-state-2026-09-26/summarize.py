"""Read-only final panel summary; no selection or threshold changes."""

import argparse
import json
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.process_time()

    def load(path):
        return json.loads(path.read_text())

    scores = load(args.data / "fresh-analysis-01/scores.json")
    panels = {}
    for model in sorted({s["model"] for s in scores}):
        panels[model] = {}
        for panel in ["standard", "timing"]:
            rows = [
                s
                for s in scores
                if s["model"] == model
                and (s["schedule"] == "timing") == (panel == "timing")
            ]
            panels[model][panel] = {}
            for kind in ["R11", "D1", "D2", "D12", "D2|1", "K12"]:
                ss = [s for s in rows if s["kind"] == kind]
                panels[model][panel][kind] = {
                    "worst_relative": [
                        max(s["relative"] for s in ss if s["output"] == o)
                        for o in ["mass", "moment"]
                    ],
                    "error_rms_max": [
                        max(s["error_rms"] for s in ss if s["output"] == o)
                        for o in ["mass", "moment"]
                    ],
                    "error_max": [
                        max(s["error_max"] for s in ss if s["output"] == o)
                        for o in ["mass", "moment"]
                    ],
                    "magnitude_range": [
                        [
                            min(s["magnitude"] for s in ss if s["output"] == o),
                            max(s["magnitude"] for s in ss if s["output"] == o),
                        ]
                        for o in ["mass", "moment"]
                    ],
                    "resolved_worst_relative": [
                        max(
                            [
                                s["relative"]
                                for s in ss
                                if s["output"] == o and s["resolved"]
                            ],
                            default=None,
                        )
                        for o in ["mass", "moment"]
                    ],
                    "failed": sum(s["pass"] is False for s in ss),
                    "unresolved": sum(not s["resolved"] for s in ss),
                    "entries": len(ss),
                }
    physical = load(args.data / "fresh-analysis-01/physical.json")
    geom = {
        "absolute_max": np.max([r["absolute_max"] for r in physical], axis=0).tolist(),
        "induced_error_max": np.max(
            [r["induced_error_max"] for r in physical], axis=0
        ).tolist(),
        "fraction_of_peak_induced_motion_worst": np.max(
            [
                np.asarray(r["induced_error_max"])
                / np.max(abs(np.array(r["true_event_motion"])), axis=0)
                for r in physical
            ],
            axis=0,
        ).tolist(),
    }
    events = []
    screens = []
    for path in (args.data / "fresh-01").glob("reference-*.json"):
        r = load(path)
        events.extend(v for v in r["events"] if v["amplitude"])
        screens.extend(r["regime"])
    instrument = {
        "event_identity_max": max(r["identity_max"] for r in events),
        "source_jump_range": [
            min(r["source_jump"] for r in events),
            max(r["source_jump"] for r in events),
        ],
        "energy_jump_range": [
            min(r["energy_jump"] for r in events),
            max(r["energy_jump"] for r in events),
        ],
        "field_norm_range": [
            min(r["field_l2_before"] for r in events),
            max(r["field_l2_after"] for r in events),
        ],
        "center_jump_max": np.max(
            [abs(np.array(r["after"]) - r["before"]) for r in events], axis=0
        ).tolist(),
        "all_regimes_qualified": all(r["qualified"] for r in screens),
        "minimum_gap": min(r["minimum_gap"] for r in screens),
        "mass_range": [
            min(r["mass_range"][0] for r in screens),
            max(r["mass_range"][1] for r in screens),
        ],
        "width_range": [
            min(r["width_range"][0] for r in screens),
            max(r["width_range"][1] for r in screens),
        ],
    }
    ranges = load(args.data / "fresh-analysis-01/state-ranges.json")
    selected = [r for r in ranges if r.get("model") == "separate-squares"]
    states = {
        "min": np.min([r["min"] for r in selected], axis=0).tolist(),
        "max": np.max([r["max"] for r in selected], axis=0).tolist(),
    }
    true_addition = []
    for case in load(args.data / "fresh-01/cases.json"):
        ys = np.array(
            [
                np.load(args.data / f"fresh-01/{name}.npz")["response"]
                for name in case["prefixes"]
            ]
        )
        k = ys[3] - ys[1] - ys[2] + ys[0]
        for window, sl in [("whole", slice(None)), ("late", slice(80, None))]:
            error = np.sqrt(np.mean(k[sl] ** 2, axis=0))
            for kind, target in [
                ("R11", ys[3]),
                ("D12", ys[3] - ys[0]),
                ("D2|1", ys[3] - ys[1]),
            ]:
                magnitude = np.sqrt(np.mean(target[sl] ** 2, axis=0))
                true_addition.append(
                    {
                        "seed": case["seed"],
                        "history": case["history"],
                        "schedule": case["schedule"]["name"],
                        "window": window,
                        "kind": kind,
                        "relative": (error / magnitude).tolist(),
                    }
                )
    audit = load(args.data / "runtime-audit.json")
    report = {
        "panels": panels,
        "evaluator_only_true_addition": true_addition,
        "geometry": geom,
        "instrument": instrument,
        "selected_state_ranges": states,
        "runtime_checkpoints": len(audit["checks"]),
        "signed_error_identity_max": max(
            r["signed_identity_max"]
            for r in load(args.data / "fresh-analysis-01/identities.json")
        ),
        "budget_before_final_summary": 30
        + sum(load(p)["cpu_seconds"] for p in args.data.glob("*/budget.json"))
        + audit["own_cpu_seconds"]
        + audit["child_cpu_seconds"],
        "own_cpu_seconds": time.process_time() - started,
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
