"""Reporting only. Read completed Phase B artifacts, never retrain or select models."""

import hashlib
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from experiments.coupled_oscillators.phase_b import protected_hashes
from experiments.coupled_oscillators.run_experiment import write_json

ROOT = Path(__file__).resolve().parent
b0 = json.loads((ROOT / "b0-refined/results.json").read_text())
b1 = json.loads((ROOT / "b1/results.json").read_text())
b2 = json.loads((ROOT / "b2-controls/results.json").read_text())
b3 = json.loads((ROOT / "b3-transform/results.json").read_text())
original_hashes = json.loads((ROOT / "b0/protocol.json").read_text())["protected_hashes_before"]
assert protected_hashes() == original_hashes


def stats(values):
    return {"mean": statistics.mean(values), "std": statistics.stdev(values), "values": values}


paired = []
for regime in ("LINEAR", "WEAK_NONLINEAR", "MODERATE_NONLINEAR"):
    for model in ("standard", "opf_fixed", "opf"):
        for split in ("interpolation", "amplitude_shift"):
            values = [r["trace"][-1]["physical_function_difference"][split]["mse_by_horizon"][-1]
                      for r in b1["paired_frames"] if r["model"] == model and r["regime"] == regime]
            paired.append({"regime": regime, "model": model, "split": split, "h16": stats(values)})

effects = []
for regime in ("WEAK_NONLINEAR", "MODERATE_NONLINEAR"):
    for frame in ("LATENT_NATIVE", "LATENT_ROTATED"):
        for split in ("interpolation", "amplitude_shift"):
            learned = [r for r in b1["records"] if r["regime"] == regime and r["frame"] == frame and r["model"] == "opf"]
            fixed = [r for r in b1["records"] if r["regime"] == regime and r["frame"] == frame and r["model"] == "opf_fixed"]
            gain = [100 * (1 - a["trace"][-1]["metrics"][split]["rollout_mse"]["16"] / b["trace"][-1]["metrics"][split]["rollout_mse"]["16"]) for a, b in zip(learned, fixed, strict=True)]
            effects.append({"regime": regime, "frame": frame, "split": split, "seeds": [11, 22, 33], "relative_improvement_percent": gain})

fine_deltas = [abs(r["refined_truth_metrics"][split][metric][h] / r["coarse_truth_metrics"][split][metric][h] - 1)
               for r in b2["conventional_checks"] for split in ("interpolation", "amplitude_shift")
               for metric in ("rollout_mse", "pulse_rollout_mse", "pulse_response_mse") for h in ("1", "4", "8", "16")]
analysis = {
    "exploratory": True,
    "primary_reading": "A: conventional nonlinear identification sufficient in this family and envelope",
    "neural_reading": "Small learned-basis gains are reproduced by ordinary output transformation; fixed basis already reduces coordinate sensitivity",
    "paired_coordinate_h16_function_mse": paired,
    "opf_vs_fixed_paired_effects": effects,
    "duration_summary": b2["summary"],
    "transform_control_summary": b3["summary"],
    "finer_truth_max_relative_metric_change": max(fine_deltas),
    "polynomial_rotation_max_physical_difference": max(v["polynomial_rotation_rollout_max_abs"] for r in b2["conventional_checks"] for v in r["coordinate_checks"].values()),
    "linear_normal_mode_max_physical_difference": max(v["linear_normal_mode_prediction_max_abs"] for r in b2["conventional_checks"] for v in r["coordinate_checks"].values()),
    "folded_opf_max_physical_difference": max(v for r in b3["folded_opf"] for v in r["max_physical_rollout_difference"].values()),
    "number_of_protected_files": len(original_hashes),
    "protected_files_unchanged": True,
    "neural_fits": 66,
    "optimizer_steps_including_replayed_prefixes": 90000,
    "summed_panel_wall_seconds": sum(b["duration_seconds"] for b in (b1, b2, b3)),
    "input_hashes": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob("*/results.json")},
}
write_json(ROOT / "analysis.json", analysis)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
models = [("linear", "PHYSICAL", "Linear ID", "#555555"),
          ("polynomial_cubic", "PHYSICAL", "Cubic ID", "#007f73"),
          ("standard", "LATENT_NATIVE", "Standard / fixed", "#526fb5"),
          ("opf", "LATENT_NATIVE", "Learned OPF", "#c37818")]
fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), layout="constrained", sharey=True)
for ax, split, title in zip(axes, ("interpolation", "amplitude_shift"), ("Held-out trajectories", "Initial amplitudes ×1.5")):
    for model, frame, label, color in models:
        row = next(r for r in b1["summary"] if r["regime"] == "MODERATE_NONLINEAR" and r["model"] == model and r["frame"] == frame and r["split"] == split)
        points = [row["metrics"]["rollout_mse"][str(h)] for h in (1, 4, 8, 16)]
        ax.errorbar([1, 4, 8, 16], [p["mean"] for p in points], yerr=[p["std"] for p in points], label=label, color=color, marker="o", capsize=3, linewidth=1.4)
    ax.set(title=title, xlabel="Rollout horizon (steps)", yscale="log", xticks=[1, 4, 8, 16])
    ax.grid(axis="y", alpha=0.2)
axes[0].set_ylabel("Physical state MSE · mean ± sample SD")
axes[1].legend(loc="lower right", frameon=False)
fig.suptitle("Moderate Duffing regime (α = 0.5) · 3 seeds · neural training: 1,000 steps")
for extension in ("png", "svg"):
    fig.savefig(ROOT / f"prediction_gap.{extension}", dpi=170)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.3), layout="constrained", sharey=True)
colors = ["#526fb5", "#007f73", "#c37818"]
for ax, regime, title in zip(axes, ("WEAK_NONLINEAR", "MODERATE_NONLINEAR"), ("Weak α = 0.1", "Moderate α = 0.5")):
    for i, (model, color) in enumerate(zip(("standard", "opf_fixed", "opf"), colors)):
        row = next(r for r in paired if r["regime"] == regime and r["model"] == model and r["split"] == "interpolation")
        for j, value in enumerate(row["h16"]["values"]):
            ax.scatter(i + (j - 1) * 0.09, value, marker=("o", "s", "^")[j], color=color, s=35)
        ax.plot([i - .2, i + .2], [row["h16"]["mean"]] * 2, color=color)
    ax.set(title=title, xticks=[0, 1, 2], xticklabels=["Standard", "Fixed basis", "Learned OPF"], ylim=(0, .020))
    ax.grid(axis="y", alpha=0.2)
axes[0].set_ylabel("Native vs rotated physical function MSE at h16")
fig.suptitle("Coordinate sensitivity · R seed 2718 · dots: seeds 11 / 22 / 33")
for extension in ("png", "svg"):
    fig.savefig(ROOT / f"coordinate_sensitivity.{extension}", dpi=170)
plt.close(fig)
print(json.dumps({k: v for k, v in analysis.items() if not isinstance(v, (list, dict))}, indent=2))
