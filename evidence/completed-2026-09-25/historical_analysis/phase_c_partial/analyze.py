"""Read-only aggregation of completed Phase C panels; no scientific fitting."""

import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
PANELS = ("c0", "c0-compression", "c0-ridge", "c1", "c2-duration", "c3-diagnostics")
data = {name: json.loads((ROOT / name / "results.json").read_text()) for name in PANELS}
for name, panel in data.items():
    assert panel["status"] == "complete" and panel["protected_artifacts_unchanged"]
    digest = hashlib.sha256((ROOT / name / "results.json").read_bytes()).hexdigest()
    assert digest == (ROOT / name / "results.sha256").read_text().strip()


def stats(values):
    valid = [value for value in values if value is not None]
    # An aggregate is unavailable if any seed failed; never silently drop it.
    return {
        "values": values,
        "mean": statistics.mean(valid) if len(valid) == len(values) else None,
        "sample_sd": statistics.stdev(valid) if len(valid) == len(values) and len(valid) > 1 else None,
        "failed_seeds": len(values) - len(valid),
    }


def fmt(record):
    return f"{record['mean']:.3g} ± {record['sample_sd']:.2g}" if record["mean"] is not None else f"FAILED ({record['failed_seeds']}/3)"


def select(panel, **filters):
    return [r for r in data[panel]["records"] if all(r.get(k) == v for k, v in filters.items())]


def metrics(records, neural=False):
    result = {}
    for split in ("interpolation", "amplitude_shift"):
        result[split] = {}
        for kind in ("rollout_mse", "pulse_rollout_mse", "pulse_response_mse"):
            result[split][kind] = {
                str(h): stats([(r["trace"][-1] if neural else r)["metrics"][split][kind][str(h)] for r in records])
                for h in (1, 4, 8, 16)
            }
    return result


summary = {"exploratory": True, "aggregate": "three paired seeds; arithmetic mean and sample SD; no failure dropping"}
summary["panels"] = {name: {"records": len(panel["records"]), "seconds": panel["duration_seconds"]} for name, panel in data.items()}
baseline = select("c0", alpha=0.5, observation="POSITIONS", history=2, model="svd_cubic_state")
summary["primary"] = {"Conventional 6D delay state (H2)": metrics(baseline)}
summary["primary"]["Linear AR (H2)"] = metrics(select("c0", alpha=0.5, observation="POSITIONS", history=2, model="linear_ar"))
labels = {"standard": "Standard JEPA", "opf_fixed": "Fixed factorized", "opf": "Learned soft OPF", "output_transform": "Ordinary output transform"}
for family, label in labels.items():
    summary["primary"][label] = metrics(select("c2-duration", model=family), neural=True)
summary["history"] = {}
for regime in ("POSITIONS", "LOCAL"):
    summary["history"][regime] = {}
    for family in ("linear_ar", "cubic_delay", "svd_cubic_state"):
        summary["history"][regime][family] = {
            str(h): metrics(select("c0", alpha=0.5, observation=regime, history=h, model=family)) for h in (1, 2, 4, 8)
        }
summary["coordinates_1000"] = {}
for family in labels:
    pairs = [p for p in data["c1"]["paired_frames"] if p["model"] == family]
    summary["coordinates_1000"][family] = {
        frame: metrics(select("c1", history=4, model=family, frame=frame), neural=True) for frame in ("native", "rotated")
    }
    summary["coordinates_1000"][family]["paired_function_mse_h16"] = stats([p["trace"][-1]["mse_by_horizon"][-1] for p in pairs])
summary["initial_function_max_abs"] = max(p["trace"][0]["max_abs"] for p in data["c1"]["paired_frames"])
summary["duration_prefix_checks"] = []
for record in data["c2-duration"]["records"]:
    previous = select("c1", seed=record["seed"], history=4, model=record["model"], frame="native")[0]["trace"][-1]
    replay = [point for point in record["trace"] if point["step"] == 1000][0]
    summary["duration_prefix_checks"].append({"seed": record["seed"], "model": record["model"], "identical_checkpoint": replay == previous})
assert all(r["identical_checkpoint"] for r in summary["duration_prefix_checks"])
summary["probes_3000"] = {}
for family in labels:
    records = select("c2-duration", model=family)
    summary["probes_3000"][family] = {
        name: stats([r["sufficiency"]["variants"][name]["metrics"]["interpolation"]["16"] for r in records])
        for name in ("z_only", "z_plus_old", "z_plus_shuffled_old")
    }
    summary["probes_3000"][family]["hidden_velocity_mse"] = stats([r["hidden_state_diagnostic"]["metrics"]["interpolation"]["velocity_mse"] for r in records])
summary["conventional_probes"] = {
    name: stats([r["sufficiency"]["variants"][name]["metrics"]["interpolation"]["16"] for r in data["c2-duration"]["conventional_probe_controls"]])
    for name in ("z_only", "z_plus_old", "z_plus_shuffled_old")
}
summary["probe_calibrations"] = {}
for alpha, history, degree in ((0.0, 1, 3), (0.0, 2, 3), (0.5, 2, 5)):
    records = [r for r in data["c3-diagnostics"]["probe_controls"] if (r["alpha"], r["history"], r["degree"]) == (alpha, history, degree)]
    summary["probe_calibrations"][f"alpha{alpha}-H{history}-degree{degree}"] = {
        name: stats([r["sufficiency"]["variants"][name]["metrics"]["interpolation"]["16"] for r in records])
        for name in ("z_only", "z_plus_old", "z_plus_shuffled_old")
    }
summary["recurrence_controls"] = {
    family: {kind: stats([r["metrics"][kind]["interpolation"]["rollout_mse"]["16"] for r in select("c3-diagnostics", model=family)]) for kind in ("history_feedback", "ema_aligned")}
    for family in labels
}
(ROOT / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

lines = ["# Phase C numerical review", "", "Exploratory; means ± sample SD over seeds 11, 22, 33. Native neural runs: 3,000 Adam steps. All metrics are position MSE.", "", "| Method | h1 | h4 | h8 | h16 | amplitude-shift h16 |", "|---|---:|---:|---:|---:|---:|"]
for label, record in summary["primary"].items():
    values = [fmt(record["interpolation"]["rollout_mse"][str(h)]) for h in (1, 4, 8, 16)]
    values += [fmt(record["amplitude_shift"]["rollout_mse"]["16"])]
    lines.append("| " + " | ".join([label] + values) + " |")
lines += ["", "Unknown-pulse forecasts start after eight post-pulse observation samples.", "", "| Method | pulsed trajectory h16 | response difference h16 | shifted pulsed trajectory h16 |", "|---|---:|---:|---:|"]
for label, record in summary["primary"].items():
    values = [fmt(record[s][kind]["16"]) for s, kind in (("interpolation", "pulse_rollout_mse"), ("interpolation", "pulse_response_mse"), ("amplitude_shift", "pulse_rollout_mse"))]
    lines.append("| " + " | ".join([label] + values) + " |")
lines += ["", "| 1,000-step neural model | native h16 | rotated h16 | paired physical-function difference h16 |", "|---|---:|---:|---:|"]
for family, record in summary["coordinates_1000"].items():
    lines.append("| " + " | ".join([labels[family]] + [fmt(record[frame]["interpolation"]["rollout_mse"]["16"]) for frame in ("native", "rotated")] + [fmt(record["paired_function_mse_h16"])]) + " |")
lines += ["", "Paired function differences use the same first forecast origin in each of the 16 held-out trajectories. Other reported MSEs use all 544 origins.", "", "| Frozen state, cubic future probe | z only h16 | z + older history h16 | z + shuffled older history h16 |", "|---|---:|---:|---:|"]
for family, record in {**summary["probes_3000"], "conventional": summary["conventional_probes"]}.items():
    lines.append("| " + " | ".join([labels.get(family, "Conventional 6D delay state")] + [fmt(record[k]) for k in ("z_only", "z_plus_old", "z_plus_shuffled_old")]) + " |")
lines += ["", "| Posthoc feedback control | original h16 | EMA-aligned h16 | observation re-encoding h16 |", "|---|---:|---:|---:|"]
for family, record in summary["recurrence_controls"].items():
    lines.append("| " + " | ".join([labels[family], fmt(summary["primary"][labels[family]]["interpolation"]["rollout_mse"]["16"]), fmt(record["ema_aligned"]), fmt(record["history_feedback"])]) + " |")
(ROOT / "tables.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
for index, split in enumerate(("interpolation", "amplitude_shift")):
    for family, label in (("linear_ar", "Linear delay"), ("svd_cubic_state", "SVD + cubic state (up to 6D)")):
        values = [summary["history"]["POSITIONS"][family][str(h)][split]["rollout_mse"]["16"]["mean"] for h in (1, 2, 4, 8)]
        axes[index].semilogy((1, 2, 4, 8), values, "o-", label=label)
    axes[index].set(title=("Ordinary test" if index == 0 else "Initial amplitude × 1.5"), xlabel="Position history samples", ylabel="16-step position MSE", xticks=[1, 2, 4, 8])
    axes[index].grid(alpha=0.2)
    axes[index].legend()
fig.suptitle("More history need not improve an estimated recursive model\nExploratory three-seed means; same trajectories and forecast origins")
fig.savefig(ROOT / "history-prediction.png", dpi=180)
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
colors = {"standard": "#376b9e", "opf": "#b94b49", "output_transform": "#49835c"}
for family, color in colors.items():
    records = select("c2-duration", model=family)
    for index, metric in enumerate(("training_loss", "rollout")):
        steps = [t["step"] for t in records[0]["trace"]][1:]
        for i, record in enumerate(records):
            values = [t["training_loss"] if metric == "training_loss" else t["metrics"]["interpolation"]["rollout_mse"]["16"] for t in record["trace"]][1:]
            axes[index].semilogy(steps, values, "o-", alpha=0.75, color=color, label=labels[family] if i == 0 else None)
axes[0].set(ylabel="Training objective (different OPF penalty)", xlabel="Adam steps", title="Loss improves")
axes[1].axhline(summary["primary"]["Conventional 6D delay state (H2)"]["interpolation"]["rollout_mse"]["16"]["mean"], color="black", linestyle="--", label="Conventional 6D delay state")
axes[1].set(ylabel="16-step position MSE", xlabel="Adam steps", title="Autonomous prediction remains poor")
for ax in axes:
    ax.set_xticks([250, 1000, 3000])
    ax.grid(alpha=0.2)
    ax.legend(fontsize=8)
fig.suptitle("Learned six-dimensional states: individual paired seeds\nFixed factorized native runs overlap Standard JEPA to roundoff")
fig.savefig(ROOT / "neural-training-rollout.png", dpi=180)
plt.close(fig)
