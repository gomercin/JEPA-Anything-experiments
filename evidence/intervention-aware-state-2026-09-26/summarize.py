"""Read-only arithmetic summary and complete compact-state accounting."""

import json
import time
from pathlib import Path

import numpy as np

ROOT = Path("work/mediated_patterns/intervention_aware_state")


def load(p):
    return json.loads(p.read_text())


def main():
    start = time.process_time()
    freeze = load(ROOT / "frozen-01/freeze.json")
    selected = freeze["selected"]
    fresh = ROOT / "fresh-01"
    scores = load(ROOT / "analysis-final/scores.json")
    s = [x for x in scores if x["model"] == selected]
    rows = load(fresh / "rows.json")
    geometry = []
    ratios = []
    norms = []
    for i, row in enumerate(rows):
        name = f"s{row['seed']}-{row['history']}"
        base = np.load(fresh / f"{name}-a0-geometry.npz")["z"]
        pred = np.load(fresh / f"prediction-{i}-{selected}.npz")["trajectory"]
        for ai, a in enumerate(freeze["amplitudes"], 1):
            actual = np.load(fresh / f"{name}-a{a:g}-geometry.npz")["z"]
            effect = actual - base
            estimate = pred[ai] - pred[0]
            rms = np.sqrt(np.mean(effect**2, axis=0))
            err = np.sqrt(np.mean((estimate - effect) ** 2, axis=0))
            geometry.append(
                dict(
                    **row,
                    a=a,
                    max_error=abs(pred[ai] - actual).max(axis=0).tolist(),
                    event_rms=rms.tolist(),
                    event_error_rms=err.tolist(),
                    event_relative=(err / rms).tolist(),
                )
            )
            norms.append(load(fresh / f"{name}-a{a:g}-event.json"))
            for gap in freeze["gaps"]:
                y = np.load(fresh / f"{name}-a{a:g}-gap{gap}.npz")["y"]
                ratios.append(
                    (
                        np.sqrt(np.mean((y[1] - y[0]) ** 2, axis=0))
                        / np.sqrt(np.mean(y[1] ** 2, axis=0))
                    ).tolist()
                )
    model = load(ROOT / "frozen-01/models.json")[selected]
    accounting = {
        "scientific_state_scalars": 3 + model["transient"]["order"] + 2,
        "integer_step_counters": 1,
        "initial_current_measurements": 3,
        "initial_full_snapshot_shape": [2, 768],
        "later_field_queries": 0,
        "stored_response_history": 0,
        "G_coefficients": 30,
        "G_scaling": 6,
        "J_coefficients": 8,
        "J_scaling": 6,
        "transient_matrices": sum(
            np.asarray(model["transient"][k]).size for k in ["A", "B", "C"]
        ),
        "duplicate_excitation_J_coefficients_and_scaling": 14,
        "F_coefficients": 80,
        "F_basis": 644,
        "F_scaling": 8,
        "correction_coefficients": int(
            np.asarray(model["readout"]["coefficients"]).size
        ),
        "correction_basis": 644,
        "correction_current_center_scaling": 6,
        "correction_output_scaling": 2,
        "fixed_response_times": 161,
        "fixed_anchors": 3,
        "fixed_decay_time": 20.0,
        "fixed_step": 1.0,
        "fixed_amplitude_scale": 0.02,
        "event_input_buffer": 1,
        "scheduled_delay_and_gaps": [10, 10, 30],
        "forecast_output_buffer": 322,
        "step_operations": "4 quadratic G evaluations (10x3 coefficients each), two3x4 C products, one4x4 A product, one scalar exponential decay; no microscopic reconstruction",
        "event_operations": "two8-term local feature maps, one8-term A jump and one4x8 excitation matrix product, plus2 scalar updates",
        "readout_operations": "10x8 original coefficient product and4x161 temporal reconstruction per output;16x8 correction product and another4x161 reconstruction per output",
    }
    accounting["active_coefficient_basis_scaling_scalars"] = sum(
        accounting[k]
        for k in [
            "G_coefficients",
            "G_scaling",
            "J_coefficients",
            "J_scaling",
            "transient_matrices",
            "duplicate_excitation_J_coefficients_and_scaling",
            "F_coefficients",
            "F_basis",
            "F_scaling",
            "correction_coefficients",
            "correction_basis",
            "correction_current_center_scaling",
            "correction_output_scaling",
        ]
    )

    def numeric_count(x):
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return 1
        if isinstance(x, list):
            return sum(map(numeric_count, x))
        if isinstance(x, dict):
            return sum(map(numeric_count, x.values()))
        return 0

    accounting["serialized_numeric_scalars_including_diagnostics_and_fixed_data"] = (
        numeric_count(model)
    )
    accounting["selected_model_json_bytes"] = len(
        json.dumps(model, sort_keys=True, allow_nan=False).encode()
    )
    runtime = load(ROOT / "runtime-audit.json")
    science = 30 + sum(load(p)["cpu_seconds"] for p in ROOT.glob("*/budget.json"))
    result = {
        "selected": selected,
        "scored_entries": len(s),
        "passed_entries": sum(x["passed"] for x in s),
        "independent_preparations": 3,
        "unique_Rs": 48,
        "unique_Ds": 36,
        "unique_scalar_gates": 336,
        "effect_over_R_range": [
            np.min(ratios, axis=0).tolist(),
            np.max(ratios, axis=0).tolist(),
        ],
        "geometry": geometry,
        "max_trajectory_error": np.max(
            [g["max_error"] for g in geometry], axis=0
        ).tolist(),
        "physical_trajectory_gates_pass": np.sum(
            np.array([g["max_error"] for g in geometry])
            <= freeze["coordinate_tolerances"],
            axis=0,
        ).tolist(),
        "event_fractional_error_max": np.max(
            [g["event_relative"] for g in geometry], axis=0
        ).tolist(),
        "moment_identity_max": max(n["identity_max"] for n in norms),
        "all_regimes_qualified": all(n["qualified"] for n in norms),
        "input_norm_range": [min(n["l2"] for n in norms), max(n["l2"] for n in norms)],
        "source_increment_range": [
            min(n["source_jump"] for n in norms),
            max(n["source_jump"] for n in norms),
        ],
        "energy_increment_range": [
            min(n["energy_jump"] for n in norms),
            max(n["energy_jump"] for n in norms),
        ],
        "worst_records": {
            k: {
                o: max(
                    [x for x in s if x["kind"] == k and x["output"] == o],
                    key=lambda x: x["relative"],
                )
                for o in ["mass", "moment"]
            }
            for k in ["R_after", "D_event"]
        },
        "panel": {
            w: {
                k: {
                    o: max(
                        x["relative"]
                        for x in s
                        if x["window"] == w and x["kind"] == k and x["output"] == o
                    )
                    for o in ["mass", "moment"]
                }
                for k in ["R_after", "D_event"]
            }
            for w in ["whole", "late"]
        },
        "withheld_D_worst": max(
            x["relative"] for x in s if x["kind"] == "D_event" and x["a"] == 0.01
        ),
        "peak_sign_agreement": all(
            x["true_peak_value"] * x["predicted_at_true_peak"] > 0
            for x in s
            if x["kind"] == "D_event"
        ),
        "D_peak_time_max_error": max(
            abs(x["true_peak_time"] - x["predicted_peak_time"])
            for x in s
            if x["kind"] == "D_event"
        ),
        "accounting": accounting,
        "science_cpu_seconds_including_30s_allowance": science,
        "solver_free_audit_cpu_seconds": runtime["own_cpu_seconds"]
        + runtime["child_cpu_seconds"],
        "summary_cpu_seconds": time.process_time() - start,
    }
    result["total_charged_cpu_seconds"] = (
        science
        + result["solver_free_audit_cpu_seconds"]
        + result["summary_cpu_seconds"]
    )
    with (ROOT / "audit-summary.json").open("x") as f:
        json.dump(result, f, indent=2, sort_keys=True)
        f.write("\n")
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k not in ["geometry", "worst_records", "accounting"]
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
