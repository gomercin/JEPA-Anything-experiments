"""Read-only exploratory summaries and one scientific diagnostic figure; no training."""
import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
report = json.loads((HERE / "results.json").read_text())

def selected(variant, step):
    return [next(c for c in r["checkpoints"] if c["step"] == step)
            for r in report["records"] if r["variant"] == variant]

def limits(values):
    values = list(values)
    return [min(values), max(values)]

summary = {
    "label": "exploratory, post-run inspection; no new training or selection",
    "panel_sha256": hashlib.sha256((HERE / "results.json").read_bytes()).hexdigest(),
    "original_reproduction_max_h16_difference": max(
        r.get("original_h16_absolute_difference", 0) for r in report["records"]),
    "encoder_error_decomposition_max_absolute_residual": max(
        abs(sum(c["observable_mse_by_encoder_singular_direction"]) - c["metrics"]["one_step_mse"])
        for r in report["records"] for c in r["checkpoints"]),
    "soft_opf": {},
    "cross_coordinate_opf": {},
    "coordinate_probe_prediction_difference_ranges": {},
}
for step in (0, 250, 1000):
    values = selected("opf", step)
    summary["soft_opf"][str(step)] = {
        "basis_condition_range": limits(c["diagnostics"]["opf_geometry"]["basis"]["condition_number"] for c in values),
        "basis_max_orthogonality_error_range": limits(c["diagnostics"]["opf_geometry"]["basis"]["max_orthogonality_error"] for c in values),
        "mode_overlap_range": limits(c["orientation_null"]["observed"] for c in values),
        "orientation_null_exceedance_range": limits(c["orientation_null"]["fraction_null_at_least_observed"] for c in values),
        "inactive_coordinate_counts": [len(c["diagnostics"]["opf_geometry"]["factors"]["inactive_coordinates"]) for c in values],
        "cross_correlation_range": limits(c["diagnostics"]["opf_geometry"]["factors"]["max_cross_factor_correlation"] for c in values),
    }
    summary["cross_coordinate_opf"][str(step)] = [
        {"seed": c["seed"], "matched_overlap": c["alignment"]["best_permutation_mean_overlap"]}
        for c in report["cross_coordinate_subspaces"] if c["variant"] == "opf" and c["step"] == step]
for optimizer in ("Adam", "SGD"):
    summary["coordinate_probe_prediction_difference_ranges"][optimizer] = limits(
        c["trace"][optimizer][-1]["prediction_max_difference"] for c in report["coordinate_probes"])
summary["raw_seed33"] = []
for record in report["records"]:
    if record["seed"] == 33 and record["condition"] == "RAW" and record["variant"] in ("standard", "opf", "opf_qr"):
        for checkpoint in record["checkpoints"]:
            if checkpoint["step"] in (250, 1000):
                summary["raw_seed33"].append({"variant": record["variant"], "step": checkpoint["step"],
                    **{k: checkpoint[k] for k in ("target_encoder_condition", "target_latent_mse",
                        "observable_error_amplification", "weakest_direction_fraction", "metrics")}})
summary["original_artifacts_unchanged"] = all(
    hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest
    for path, digest in report["protocol"]["original_artifact_sha256"].items())
assert summary["original_artifacts_unchanged"]
assert hashlib.sha256((HERE / "interrogate.executed.py").read_bytes()).hexdigest() == report["protocol"]["source_sha256"]["experiments/coupled_oscillators/interrogate.py"]
(HERE / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), layout="constrained")
colors = {"standard": "#3574B1", "opf": "#179276", "opf_qr": "#BA4A46"}
labels = {"standard": "Standard JEPA", "opf": "OPF soft Gram", "opf_qr": "OPF QR retraction"}
for variant in colors:
    record = next(r for r in report["records"] if r["seed"] == 33 and r["condition"] == "RAW" and r["variant"] == variant)
    checkpoints = [c for c in record["checkpoints"] if c["step"] > 0]
    steps = [c["step"] for c in checkpoints]
    axes[0].plot(steps, [c["target_encoder_condition"] for c in checkpoints], "o-", color=colors[variant], label=labels[variant])
    axes[1].plot(steps, [c["metrics"]["rollout_mse"]["16"] for c in checkpoints], "o-", color=colors[variant], label=labels[variant])
for ax in axes:
    ax.set_yscale("log")
    ax.set_xlabel("Training step")
    ax.axvline(250, color="#999999", linestyle=":", linewidth=1)
    ax.grid(axis="y", which="major", alpha=.18)
axes[0].set_title("Target encoder conditioning")
axes[0].set_ylabel("Condition number (log scale)")
axes[0].legend(frameon=False)
axes[1].set_title("Recursive prediction error")
axes[1].set_ylabel("16-step observable MSE (log scale)")
fig.suptitle("Exploratory RAW seed 33: an orthogonal OPF basis can coexist with a poor encoder\nQR basis condition = 1 at every saved step; all original outcomes retained", fontsize=12)
fig.savefig(HERE / "conditioning.png", dpi=180)
print(json.dumps(summary, indent=2))
