"""Read-only aggregation and scientific figures for the completed Phase D panels."""

import hashlib
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
NAMES = ("d0", "d1-filters", "d2-neural", "d3-output-error-contiguous", "d4-duration", "d5-normalization", "d6-output-error-duration")
panels = {name: json.loads((ROOT / name / "results.json").read_text()) for name in NAMES}
for name, panel in panels.items():
    assert panel["status"] == "complete" and panel["protected_artifacts_unchanged"]
    assert hashlib.sha256((ROOT / name / "results.json").read_bytes()).hexdigest() == (ROOT / name / "results.sha256").read_text().strip()


def select(panel, **criteria):
    return [r for r in panels[panel]["records"] if all(r.get(k) == v for k, v in criteria.items())]


def stats(values):
    valid = all(value is not None for value in values)
    return {"values": values, "mean": statistics.mean(values) if valid else None,
            "sample_sd": statistics.stdev(values) if valid and len(values) > 1 else None,
            "failed_seeds": sum(value is None for value in values)}


def fmt(value):
    return f"{value['mean']:.3g} ± {value['sample_sd']:.2g}" if value["mean"] is not None else f"unavailable ({value['failed_seeds']}/3 failed)"


def main_metrics(record):
    return record["trace"][-1]["metrics"] if "trace" in record else record["metrics"]


def aggregate(records):
    output = {}
    for split in ("interpolation", "amplitude_shift"):
        output[split] = {
            metric: {str(h): stats([main_metrics(r)[split][metric][str(h)] for r in records]) for h in (1, 4, 8, 16)}
            for metric in ("clean_truth_mse", "noisy_target_mse", "pulse_clean_mse", "pulse_noisy_mse", "pulse_response_clean_mse")
        }
    if all("unseen_noise" in r for r in records):
        output["unseen_noise_h16"] = stats([r["unseen_noise"]["metrics"]["interpolation"]["clean_truth_mse"]["16"] for r in records])
    output["probe"] = {kind: stats([r["state_probe"]["interpolation"][kind] for r in records]) for kind in ("position_mse", "velocity_mse")}
    if "uncertainty" in main_metrics(records[0])["interpolation"]:
        output["uncertainty"] = {
            str(h): {kind: stats([main_metrics(r)["interpolation"]["uncertainty"][str(h)][kind] for r in records])
                     for kind in ("clean_95pct_marginal_coverage", "noisy_95pct_marginal_coverage", "mean_position_variance")}
            for h in (1, 4, 8, 16)
        }
        output["unseen_noise_clean_coverage_h16"] = stats([r["unseen_noise"]["metrics"]["interpolation"]["uncertainty"]["16"]["clean_95pct_marginal_coverage"] for r in records])
        output["oracle_true_state_h16"] = stats([r["state_diagnostics"]["oracle_true_initial_state"]["interpolation"]["16"] for r in records])
    return output


groups = {
    "Noisy cubic delay H2": select("d0", model="cubic_delay_state", history=2, noise_ratio=0.15),
    "Noisy cubic delay H8": select("d0", model="cubic_delay_state", history=8, noise_ratio=0.15),
    "Causal polynomial state H8": select("d0", model="causal_polynomial_state", history=8, noise_ratio=0.15),
    "Identified EKF": select("d1-filters", model="identified_ekf", history=8, noise_ratio=0.15),
    "Output-error identified EKF": panels["d6-output-error-duration"]["records"],
    "ORACLE EKF": select("d1-filters", model="ORACLE_EKF", history=8, noise_ratio=0.15),
    "Standard JEPA": select("d4-duration", model="standard"),
    "Fixed factorized": select("d4-duration", model="opf_fixed"),
    "Learned soft OPF": select("d4-duration", model="opf"),
    "Ordinary output transform": select("d4-duration", model="output_transform"),
}
summary = {"exploratory": True, "seeds": [11,22,33], "aggregation": "arithmetic mean and sample SD; no failure dropping", "primary": {name: aggregate(records) for name, records in groups.items()}}
summary["panels"] = {name: {"records": len(panel["records"]), "duration_seconds": panel["duration_seconds"]} for name, panel in panels.items()}
summary["failed_attempt"] = json.loads((ROOT / "d3-output-error/failure.json").read_text())
summary["noise_ladder"] = {}
for ratio in (0.0, 0.05, 0.15):
    summary["noise_ladder"][str(ratio)] = {}
    for history in (2, 4, 8):
        records = select("d0", model="cubic_delay_state", history=history, noise_ratio=ratio)
        summary["noise_ladder"][str(ratio)][str(history)] = {
            "clean_h16": stats([r["metrics"]["interpolation"]["clean_truth_mse"]["16"] for r in records]),
            "oracle_clean_context_h16": stats([r["oracle_clean_context"]["interpolation"]["clean_truth_mse"]["16"] for r in records]),
        }
summary["neural_prefix_exact"] = []
for record in panels["d4-duration"]["records"]:
    earlier = select("d2-neural", seed=record["seed"], model=record["model"], history=8, noise_ratio=0.15)[0]
    exact = earlier["trace"][-1] == record["trace"][2]
    assert exact
    summary["neural_prefix_exact"].append({"seed": record["seed"], "model": record["model"], "exact": exact})
summary["output_error_loss_prefix_exact"] = []
for record in panels["d6-output-error-duration"]["records"]:
    previous = select("d3-output-error-contiguous", seed=record["seed"])[0]["fit"]["loss_evaluations"]
    exact = record["fit"]["loss_evaluations"][:len(previous)] == previous
    assert exact
    summary["output_error_loss_prefix_exact"].append({"seed": record["seed"], "exact": exact})
summary["normalization_control"] = {
    "noiseless_own_normalization_h16": stats([r["trace"][-1]["metrics"]["interpolation"]["clean_truth_mse"]["16"] for r in select("d2-neural", model="standard", noise_ratio=0.0)]),
    "noiseless_shared_normalization_h16": stats([r["trace"][-1]["metrics"]["interpolation"]["clean_truth_mse"]["16"] for r in panels["d5-normalization"]["records"]]),
    "noise_trained_clean_context_h16": stats([r["moderate_trained_clean_context"]["interpolation"]["clean_truth_mse"]["16"] for r in panels["d5-normalization"]["records"]]),
}
(ROOT / "analysis.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")

lines = ["# Phase D numerical review", "", "Exploratory. Mean ± sample SD over seeds 11,22,33. Moderate noise = 15% signal SD. Neural models: 3,000 steps, native frame. Filters/neural contexts H8 unless labelled H2.", "", "| Model | clean h1 | noisy h1 | clean h4 | clean h8 | clean h16 | noisy h16 |", "|---|---:|---:|---:|---:|---:|---:|"]
for label, record in summary["primary"].items():
    m = record["interpolation"]
    values = [fmt(m[k][str(h)]) for k,h in (("clean_truth_mse",1),("noisy_target_mse",1),("clean_truth_mse",4),("clean_truth_mse",8),("clean_truth_mse",16),("noisy_target_mse",16))]
    lines.append("| " + " | ".join([label] + values) + " |")
lines += ["", "| Model | amplitude-shift clean h16 | pulse clean h16 | pulse response h16 | unseen noise clean h16 |", "|---|---:|---:|---:|---:|"]
for label, record in summary["primary"].items():
    values = [fmt(record["amplitude_shift"]["clean_truth_mse"]["16"]), fmt(record["interpolation"]["pulse_clean_mse"]["16"]), fmt(record["interpolation"]["pulse_response_clean_mse"]["16"]), fmt(record["unseen_noise_h16"]) if "unseen_noise_h16" in record else "not run"]
    lines.append("| " + " | ".join([label]+values) + " |")
lines += ["", "Unknown pulse; only eight post-pulse noisy samples available before forecasting. Pulse and base branches have independent measurement noise. Unseen noise = 1.5 times training sigma; filter R remains fixed.", "", "| Representation, evaluator-only affine probe | current clean q MSE | hidden velocity MSE |", "|---|---:|---:|"]
for label, record in summary["primary"].items():
    lines.append("| " + " | ".join([label,fmt(record["probe"]["position_mse"]),fmt(record["probe"]["velocity_mse"])]) + " |")
lines += ["", "| Filter | clean-state oracle initialization h16 | empirical clean coverage h16 (nominal 95%) | noisy coverage h16 | unseen noise clean coverage h16 |", "|---|---:|---:|---:|---:|"]
for label in ("Identified EKF", "Output-error identified EKF", "ORACLE EKF"):
    r=summary["primary"][label]
    lines.append("| " + " | ".join([label,fmt(r["oracle_true_state_h16"]),fmt(r["uncertainty"]["16"]["clean_95pct_marginal_coverage"]),fmt(r["uncertainty"]["16"]["noisy_95pct_marginal_coverage"]),fmt(r["unseen_noise_clean_coverage_h16"])]) + " |")
lines += ["", "| Noise/signal SD | delay H2 clean h16 | delay H4 clean h16 | delay H8 clean h16 |", "|---|---:|---:|---:|"]
for ratio, row in summary["noise_ladder"].items():
    lines.append("| " + " | ".join([ratio] + [fmt(row[str(h)]["clean_h16"]) for h in (2,4,8)]) + " |")
lines += ["", "Normalization control: all scores below use clean test histories and clean futures. Matched initialization uses the same normalization fitted only from noisy training observations.", ""]
for name, value in summary["normalization_control"].items():
    lines.append(f"- {name}: {fmt(value)}")
(ROOT / "tables.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))

plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
fig, axes = plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
for h in (2,4,8):
    axes[0].semilogy([0,5,15],[summary["noise_ladder"][str(r)][str(h)]["clean_h16"]["mean"] for r in (0.0,0.05,0.15)],"o-",label=f"Cubic delay H{h}")
axes[0].set(xlabel="Measurement noise / signal SD (%)",ylabel="16-step clean position MSE",title="The noiseless delay solution fractures",xticks=[0,5,15])
for label in ("Identified EKF","Output-error identified EKF","ORACLE EKF","Standard JEPA","Learned soft OPF","Ordinary output transform"):
    axes[1].semilogy([1,4,8,16],[summary["primary"][label]["interpolation"]["clean_truth_mse"][str(h)]["mean"] for h in (1,4,8,16)],"o-",label=label)
axes[1].set(xlabel="Autonomous forecast horizon",ylabel="Clean position MSE",title="Identified filtering handles moderate noise",xticks=[1,4,8,16])
for ax in axes:
    ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle("Phase D: point prediction from noisy positions\nExploratory three-seed means; no clean training labels")
fig.savefig(ROOT/"prediction.png",dpi=180);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(11,4.3),constrained_layout=True)
order=["noiseless_own_normalization_h16","noiseless_shared_normalization_h16","noise_trained_clean_context_h16"]
for seed_index, seed in enumerate((11,22,33)):
    axes[0].plot(range(3),[summary["normalization_control"][key]["values"][seed_index] for key in order],"o-",label=f"seed {seed}")
axes[0].set(xticks=range(3),xticklabels=["No noise\nown normalization","No noise\nshared normalization","15% noise\nshared normalization"],ylabel="Clean-context 16-step MSE",title="Noisy training changes the learned function")
axes[0].legend();axes[0].grid(alpha=.2)
for label in ("Identified EKF","Output-error identified EKF","ORACLE EKF"):
    axes[1].plot([1,4,8,16],[100*summary["primary"][label]["uncertainty"][str(h)]["clean_95pct_marginal_coverage"]["mean"] for h in (1,4,8,16)],"o-",label=label)
axes[1].axhline(95,color="black",linestyle="--",label="Nominal 95%")
axes[1].set(xticks=[1,4,8,16],xlabel="Forecast horizon",ylabel="Clean-truth marginal coverage (%)",title="Point accuracy does not certify uncertainty",ylim=[40,100])
axes[1].legend(fontsize=8);axes[1].grid(alpha=.2)
fig.suptitle("Two mechanism checks: normalization and uncertainty\nNoisy-training controls use Standard JEPA at 1,000 steps")
fig.savefig(ROOT/"controls.png",dpi=180);plt.close(fig)
