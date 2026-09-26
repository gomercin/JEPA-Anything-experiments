"""Read-only panel/resource accounting; writes one exclusive scientific summary."""

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path("work/mediated_patterns/event_age_response")


def load(p):
    return json.loads(p.read_text())


def main():
    started = time.process_time()
    scores = load(ROOT / "fresh-analysis-01/scores.json")
    geometry = load(ROOT / "fresh-analysis-01/geometry.json")
    table = {}
    for name in ["causal", "original9", "original11", "explicit_age", "exact_centers"]:
        table[name] = {}
        for age in [10, 18, 30]:
            rows = [r for r in scores if r["model"] == name and r["age"] == age]
            table[name][str(age)] = {}
            for kind in ["R_without", "R_after", "D1"]:
                table[name][str(age)][kind] = {}
                for output in ["mass", "moment"]:
                    rr = [
                        r for r in rows if r["kind"] == kind and r["output"] == output
                    ]
                    table[name][str(age)][kind][output] = {
                        "worst_relative": max(r["relative"] for r in rr),
                        "max_error_rms": max(r["error_rms"] for r in rr),
                        "max_error": max(r["error_max"] for r in rr),
                        "rms_range": [
                            min(r["magnitude"] for r in rr),
                            max(r["magnitude"] for r in rr),
                        ],
                        "failed": sum(r["passed"] is False for r in rr),
                        "unresolved": sum(r["passed"] is None for r in rr),
                    }
    selected = [r for r in scores if r["model"] == "causal"]
    unique = {}
    for r in selected:
        key = (
            r["seed"],
            r["history"],
            r["age"],
            0 if r["kind"] == "R_without" else r["a"],
            r["kind"],
            r["output"],
            r["window"],
        )
        if key in unique:
            assert all(
                unique[key][k] == r[k] for k in ["relative", "magnitude", "passed"]
            )
        unique[key] = r
    events = []
    screens = []
    for p in (ROOT / "fresh-01").glob("reference-*.json"):
        r = load(p)
        screens.extend(r["regime"])
        events.extend(e for e in r["events"] if e["amplitude"])
    event = {
        key: [min(r[key] for r in events), max(r[key] for r in events)]
        for key in [
            "l2",
            "source_jump",
            "energy_jump",
            "field_l2_before",
            "field_l2_after",
        ]
    }
    stages = {p.parent.name: load(p) for p in ROOT.glob("*/budget.json")}
    sections = [s for r in stages.values() for s in r["sections"]]
    extra = time.process_time() - started
    summary = {
        "table": table,
        "distinct_gates": len(unique),
        "passed": sum(r["passed"] is True for r in unique.values()),
        "failed": sum(r["passed"] is False for r in unique.values()),
        "unresolved": sum(r["passed"] is None for r in unique.values()),
        "all_regimes_qualified": all(s["qualified"] for s in screens),
        "events": event,
        "geometry_absolute_max": np.max(
            [r["absolute_max"] for r in geometry], axis=0
        ).tolist(),
        "event_motion_error_max": np.max(
            [r["event_motion_error_max"] for r in geometry], axis=0
        ).tolist(),
        "stages_cpu": {k: v["cpu_seconds"] for k, v in stages.items()},
        "cpu_seconds": 30
        + sum(r["cpu_seconds"] for r in stages.values())
        + load(ROOT / "runtime-audit.json")["cpu_seconds"]
        + extra,
        "analysis_cpu_seconds": extra,
        "largest_section_cpu": max(r["cpu_seconds"] for r in sections),
        "largest_section_wall": max(r["wall_seconds"] for r in sections),
        "response_scalars": 8,
        "total_scientific_scalars": 15,
        "step_counter": 1,
        "active_static_numbers": 2423,
        "serialized_numeric_values": 2709,
        "repeated_stage": "NOT RUN: prospective fresh single-event contract failed",
    }
    with (ROOT / "final-summary.json").open("x") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps({k: v for k, v in summary.items() if k != "table"}))


if __name__ == "__main__":
    main()
