"""Post-hoc, read-only analysis of the explicitly named frozen-encoder panels."""
import hashlib
import json
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from experiments.coupled_oscillators.dataset import make_dataset
from experiments.coupled_oscillators.evaluate import rollout
from experiments.coupled_oscillators.frozen_encoder import PhysicalPredictor, make_model
from experiments.coupled_oscillators.run_experiment import Config
from experiments.coupled_oscillators.simulator import Simulator

torch.set_num_threads(1)
PANELS = ("c1", "c2-c3", "alternate-rotation", "sgd-duration", "gram-controls", "fixed-alternate", "qr-native")
panels = {name: json.loads((HERE / name / "results.json").read_text()) for name in PANELS}
all_records = [r for panel in panels.values() for r in panel["records"]]

def last(record):
    return record["checkpoints"][-1]

def stats(values):
    values = list(values)
    return {"mean": statistics.mean(values), "std": statistics.stdev(values), "values": values}

def finite(value):
    if isinstance(value, dict):
        for child in value.values(): finite(child)
    elif isinstance(value, list):
        for child in value: finite(child)
    elif isinstance(value, float):
        assert __import__("math").isfinite(value)

for name, panel in panels.items():
    assert panel["status"] == "complete"
    finite(panel)
    for path, digest in panel["protocol"]["protected_artifact_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    assert hashlib.sha256((HERE / name / "results.json").read_bytes()).hexdigest() == (HERE / name / "results.sha256").read_text().split()[0]
    assert hashlib.sha256((HERE / name / "frozen_encoder.executed.py").read_bytes()).hexdigest() == panel["protocol"]["source_sha256"]["experiments/coupled_oscillators/frozen_encoder.py"]

result = {
    "label": "exploratory post-hoc analysis; no additional training",
    "protected_artifacts_unchanged": True,
    "source_panel_sha256": {name: hashlib.sha256((HERE / name / "results.json").read_bytes()).hexdigest() for name in PANELS},
    "fits": len(all_records),
    "optimizer_updates": sum(last(r)["step"] for r in all_records),
    "sum_panel_seconds": sum(p["duration_seconds"] for p in panels.values()),
    "c0_max_latent_difference": max(c["max_absolute_difference"] for c in panels["c1"]["c0"]),
    "c0_max_relative_difference": max(c["relative_frobenius_difference"] for c in panels["c1"]["c0"]),
    "c0_max_isometry_error": max(c["isometry_max_error"] for c in panels["c1"]["c0"]),
    "c1_max_physical_prediction_difference": max(max(c["physical_prediction_max_abs_by_horizon"]) for p in panels["c1"]["paired"] for c in p["checkpoints"]),
    "encoder_max_drift": max(c["encoder_max_drift"] for r in all_records for c in r["checkpoints"]),
    "target_encoder_max_difference": max(c["target_encoder_max_difference"] for r in all_records for c in r["checkpoints"]),
    "paired_rotation_errors": [],
    "performance": [],
    "sgd_growth": [],
}
for panel_name in ("c2-c3", "alternate-rotation", "gram-controls", "fixed-alternate"):
    panel = panels[panel_name]
    for optimizer, model in sorted({(p["optimizer"], p["model"]) for p in panel["paired"]}):
        group = [p for p in panel["paired"] if p["optimizer"] == optimizer and p["model"] == model]
        result["paired_rotation_errors"].append({
            "panel": panel_name, "rotation_seed": panel["protocol"]["rotation_seed"],
            "optimizer": optimizer, "model": model,
            "one_step_pair_mse": stats(last(p)["physical_prediction_mse_by_horizon"][0] for p in group),
            "h16_pair_mse": stats(last(p)["physical_prediction_mse_by_horizon"][3] for p in group),
            "max_abs_h16": max(last(p)["physical_prediction_max_abs_by_horizon"][3] for p in group),
        })
for panel_name, panel in panels.items():
    for optimizer, model, frame, obs in sorted({(r["optimizer"], r["model"], r["frame"], r["observation"]) for r in panel["records"]}):
        group = [r for r in panel["records"] if (r["optimizer"], r["model"], r["frame"], r["observation"]) == (optimizer, model, frame, obs)]
        for step in (250, 1000, 5000):
            if step not in panel["protocol"]["checkpoints"]: continue
            snapshots = [next(c for c in r["checkpoints"] if c["step"] == step) for r in group]
            row = {"panel": panel_name, "optimizer": optimizer, "model": model, "frame": frame, "observation": obs, "step": step,
                   "one_step_mse": stats(c["metrics"]["one_step_mse"] for c in snapshots),
                   "h16_mse": stats(c["metrics"]["rollout_mse"]["16"] for c in snapshots),
                   "pulse_response_h16": stats(c["metrics"]["pulse_response_mse"]["16"] for c in snapshots)}
            if model != "standard":
                row["basis_condition"] = stats(c["audit"]["basis"]["condition_number"] for c in snapshots)
                row["basis_orthogonality_error"] = stats(c["audit"]["basis"]["max_orthogonality_error"] for c in snapshots)
            result["performance"].append(row)

# Follow the dominant SGD seed, selected after inspecting the failures. No fitting:
# reconstruct saved models and inspect exact Jacobians and recursive predictions.
identity = torch.eye(6, dtype=torch.float64)
dataset = make_dataset(Simulator(), 22, 48, 16, 64)
test, train = dataset.physical[dataset.test_ids], dataset.physical[dataset.train_ids]
initial = test[:, :49].reshape(-1, 6)
result["true_transition_spectral_radius"] = float(torch.linalg.eigvals(Simulator().transition()).abs().max())
for family in ("standard", "opf"):
    record = next(r for r in panels["sgd-duration"]["records"] if r["seed"] == 22 and r["model"] == family)
    for step in (250, 1000, 5000):
        saved = next(c for c in record["checkpoints"] if c["step"] == step)
        model = make_model(22, family, identity, identity, Config())
        model.load_state_dict({k: torch.tensor(v, dtype=torch.float64) for k, v in saved["state_dict"].items()})
        predict = PhysicalPredictor(model, identity, identity)
        trace = rollout(predict, initial, 16)
        truth = test[:, 16:65].reshape(-1, 6)
        errors = (trace[:, -1] - truth).square().mean(-1)
        worst = int(errors.argmax())
        points = (torch.zeros(6, dtype=torch.float64), initial[worst], trace[worst, -1])
        radii = [float(torch.linalg.eigvals(torch.autograd.functional.jacobian(predict, x)).abs().max()) for x in points]
        result["sgd_growth"].append({"seed": 22, "model": family, "step": step,
            "worst_origin_trajectory_index": worst // 49, "worst_origin_time_index": worst % 49,
            "max_h16_mse_per_origin": float(errors.max()), "jacobian_rho_zero_start_end": radii,
            "maximum_train_state_norm": float(train.norm(dim=-1).max()),
            "worst_initial_norm": float(initial[worst].norm()), "predicted_norm_trace": trace[worst].norm(dim=-1).tolist()})

(HERE / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), layout="constrained")
colors = {"standard": "#3574B1", "opf": "#179276", "opf_fixed": "#B16B30"}
labels = {"standard": "Standard JEPA", "opf": "Learned soft OPF", "opf_fixed": "Fixed counter-rotated basis"}
for model in colors:
    values = []
    for rotation in (2718, 31415):
        row = next(r for r in result["paired_rotation_errors"] if r["optimizer"] == "Adam" and r["model"] == model and r["rotation_seed"] == rotation)
        values.extend(row["h16_pair_mse"]["values"])
    axes[0].plot(range(6), values, "o-", color=colors[model], label=labels[model])
axes[0].set_xticks(range(6), ["R1/11", "R1/22", "R1/33", "R2/11", "R2/22", "R2/33"])
axes[0].set_xlabel("Rotation / paired data and initialization seed")
axes[0].set_ylabel("Native–rotated physical prediction MSE, h=16")
axes[0].set_title("Adam coordinate sensitivity at 1,000 steps")
axes[0].legend(frameon=False, fontsize=9)
for family in ("standard", "opf"):
    r = next(r for r in panels["sgd-duration"]["records"] if r["seed"] == 22 and r["model"] == family)
    points = [c for c in r["checkpoints"] if c["step"] > 0]
    axes[1].plot([c["step"] for c in points], [c["metrics"]["rollout_mse"]["16"] for c in points], "o-", label=labels[family], color=colors[family])
axes[1].set_yscale("log")
axes[1].set_xlabel("Training step; unchanged SGD learning rate 0.03")
axes[1].set_ylabel("16-step physical MSE (log scale)")
axes[1].set_title("Coordinate equivalence does not ensure accuracy\nPost-hoc inspection: dominant SGD seed 22")
for ax in axes: ax.grid(axis="y", alpha=.2)
fig.suptitle("Exploratory frozen-isometric-encoder diagnostics — no physical factor claim", fontsize=13)
fig.savefig(HERE / "mechanisms.png", dpi=180)
print('fits / updates / seconds:', result['fits'],result['optimizer_updates'],result['sum_panel_seconds'])
print('C0/C1 differences:',result['c0_max_latent_difference'],result['c1_max_physical_prediction_difference'])
for r in result["paired_rotation_errors"]:
    print(r['rotation_seed'], r['optimizer'], r['model'], 'h16 pair MSE',r['h16_pair_mse'])
print('Original artifacts, result digests, and executed source snapshots verified.')
