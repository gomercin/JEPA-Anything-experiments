#!/usr/bin/env python3
"""Run the fixed Phase A oscillator panel; outputs never overwrite an earlier run."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import torch

# Support the documented direct-script command as well as module imports.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from jepa_anything_core.baselines import parameter_count  # noqa: E402
from jepa_anything_core.opf import validate_factorization  # noqa: E402

from experiments.coupled_oscillators.dataset import (  # noqa: E402
    make_dataset,
    observation_matrix,
    pairs,
)
from experiments.coupled_oscillators.evaluate import diagnostics, evaluate  # noqa: E402
from experiments.coupled_oscillators.models import (  # noqa: E402
    ObservablePredictor,
    build_models,
    fit_affine,
)
from experiments.coupled_oscillators.simulator import Simulator  # noqa: E402
from experiments.coupled_oscillators.train import train_neural  # noqa: E402


@dataclass(frozen=True)
class Config:
    preset: str = "quick"
    seeds: tuple[int, ...] = (11, 22, 33)
    conditions: tuple[str, ...] = ("RAW", "MIXED")
    mixing_seed: int = 1729
    train_count: int = 48
    test_count: int = 16
    trajectory_steps: int = 64
    training_steps: int = 250
    batch_size: int = 128
    learning_rate: float = 0.003
    ema_momentum: float = 0.99
    d: int = 6
    k: int = 3
    r: int = 2
    hidden_dim: int = 32
    gram_weight: float = 0.05
    activity_weight: float = 0.1
    min_std: float = 0.1
    max_correlation: float = 0.2
    horizons: tuple[int, ...] = (1, 4, 8, 16)
    pulse_time: int = 8
    pulse_oscillator: int = 0
    pulse_magnitude: float = 0.2

    @classmethod
    def full(cls):
        return cls(
            preset="full", train_count=256, test_count=64, trajectory_steps=128, training_steps=2000
        )


BOUNDARIES = [
    "Instrument/integration check; quick training is not a converged benchmark.",
    "A useful representation != a privileged physical ontology.",
    "Orthogonal learned factors != statistically independent causes.",
    "Good prediction after an intervention != causal discovery.",
    "MIXED coordinates != changed dynamics.",
    "PARTIAL observation != mere coordinate change; PARTIAL is not implemented.",
    "Lower training loss is not evidence of architectural advantage.",
    "Phase B nonlinear extension is intentionally deferred and not executed.",
]


def write_json(path, content):
    path.write_text(json.dumps(content, indent=2, sort_keys=True, allow_nan=False) + "\n")


def fingerprint(tensor):
    # Byte digest does not require numpy (core runtime depends only on torch).
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).flatten().tolist()
    return hashlib.sha256(bytes(raw)).hexdigest()


def aggregate(records):
    summaries = []
    for condition in sorted({r["condition"] for r in records}):
        for model in sorted({r["model"] for r in records}):
            group = [r for r in records if r["condition"] == condition and r["model"] == model]
            metrics = {}
            for family, first in group[0]["metrics"].items():
                keys = first.keys() if isinstance(first, dict) else (None,)
                stats = {}
                for key in keys:
                    values = [
                        r["metrics"][family][key] if key else r["metrics"][family] for r in group
                    ]
                    stats[key] = {
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else None,
                    }
                metrics[family] = stats if isinstance(first, dict) else stats[None]
            summaries.append(
                {"condition": condition, "model": model, "n_seeds": len(group), "metrics": metrics}
            )
    return summaries


def human_summary(report):
    lines = [
        "# Coupled oscillators: Phase A",
        "",
        "Mean ± sample SD over three paired seeds.",
        "MSE is in observable units, averaged over states and trajectory origins.",
        "",
        "| Condition | Model | h=1 | h=4 | h=8 | h=16 | Pulse h=16 | Response h=16 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    def cell(value):
        sd = value["std"]
        return f"{value['mean']:.3g} ± {sd:.2g}" if sd is not None else f"{value['mean']:.3g}"

    for row in report["summary"]:
        metrics = row["metrics"]
        values = [cell(metrics["rollout_mse"][str(h)]) for h in (1, 4, 8, 16)]
        values += [
            cell(metrics[name]["16"]) for name in ("pulse_rollout_mse", "pulse_response_mse")
        ]
        lines.append(f"| {row['condition']} | {row['model']} | " + " | ".join(values) + " |")
    lines += [
        "",
        "OPF audit outcomes (failures are retained diagnostics, not experiment failures):",
        "",
        "| Condition | Seed | Basis pass | Condition number | Round-trip NMSE | "
        "Transpose NMSE | Max cross correlation | Inactive coordinates | Mode overlap |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for record in report["records"]:
        if record["model"] != "opf":
            continue
        audit = record["diagnostics"]["opf_geometry"]
        alignment = record["diagnostics"]["mode_alignment"]
        lines.append(
            f"| {record['condition']} | {record['seed']} | {audit['basis']['passed']} | "
            f"{audit['basis']['condition_number']} | "
            f"{audit['round_trip']['normalized_mean_square_error']:.3g} | "
            f"{audit['transpose_synthesis']['normalized_mean_square_error']:.3g} | "
            f"{audit['factors']['max_cross_factor_correlation']:.3g} | "
            f"{len(audit['factors']['inactive_coordinates'])} | "
            f"{alignment['best_permutation_mean_overlap']:.3g} |"
        )
    lines += [
        "",
        *[f"- {boundary}" for boundary in BOUNDARIES],
        "",
        "See results.json for every seed, parameter counts, protocol, matrices and diagnostics.",
    ]
    return "\n".join(lines) + "\n"


def run(config, output):
    validate_factorization(config.d, config.k, config.r)
    if (config.d, config.k, config.r) != (6, 3, 2):
        raise ValueError("This bounded experiment fixes d=6, K=3, r=2")
    if config.training_steps < 1 or config.trajectory_steps <= max(config.horizons):
        raise ValueError("Need training steps and enough trajectory steps for evaluation")
    if not 0 <= config.pulse_time <= config.trajectory_steps:
        raise ValueError("Pulse time must be a held-out trajectory state")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    simulator = Simulator()
    matrices = {
        condition: observation_matrix(condition, config.mixing_seed)
        for condition in config.conditions
    }
    source_paths = sorted(Path(__file__).parent.rglob("*.py")) + [
        Path(__file__).with_name("README.md")
    ]
    source_paths += sorted((ROOT / "jepa-anything-core/src/jepa_anything_core").glob("*.py"))
    protocol = {
        "phase": "A_linear",
        "config": asdict(config),
        "simulator": asdict(simulator),
        "integrator": "torch.matrix_exp(dt * generator), exact linear flow up to roundoff",
        "generator_matrix": simulator.generator().tolist(),
        "transition_matrix": simulator.transition().tolist(),
        "mode_stiffness_eigenvalues": simulator.modes()[0].tolist(),
        "physical_mode_planes": simulator.modes()[1].tolist(),
        "observation_matrices": {name: matrix.tolist() for name, matrix in matrices.items()},
        "mixing_policy": "One seeded signed-QR orthogonal matrix, reused across all seeds",
        "dtype": "float64",
        "device": "cpu",
        "threads": 1,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "command": [sys.executable, *sys.argv],
        "git_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "git_status": subprocess.check_output(["git", "status", "--short"], cwd=ROOT, text=True),
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        },
        "optimizer": "Adam, default betas=(0.9,0.999), eps=1e-8, weight_decay=0",
        "preprocessing": "None; no normalization, augmentation, observation or process noise",
        "readout": "Frozen target encoder -> observation; affine SVD fit on train targets only",
        "rollout": "Decode to observations and re-encode at each step; no teacher forcing",
        "metric": "Endpoint MSE over 6 coordinates and all test origins t=0..T-max(h)",
        "perturbation": "At t=8 in every held-out trajectory, add +0.2 to p1; no pulse training",
        "seed_streams": {
            "initial_conditions": "seed",
            "split": "seed+10000",
            "initialization": "seed",
            "minibatches": "seed+20000",
        },
        "selection": "Fixed last step; no tuning, checkpoint selection or early stopping",
        "boundaries": BOUNDARIES,
    }
    # Persist the entire protocol before any training or outcome evaluation.
    write_json(output / "protocol.json", protocol)
    report = {
        "schema_version": 1,
        "status": "running",
        "protocol": protocol,
        "datasets": [],
        "records": [],
    }
    for seed in config.seeds:
        dataset = make_dataset(
            simulator, seed, config.train_count, config.test_count, config.trajectory_steps
        )
        report["datasets"].append(
            {
                "seed": seed,
                "physical_trajectories_sha256": fingerprint(dataset.physical),
                "train_ids": dataset.train_ids.tolist(),
                "test_ids": dataset.test_ids.tolist(),
                "initial_conditions": dataset.physical[:, 0].tolist(),
            }
        )
        for condition, matrix in matrices.items():
            train, test = dataset.observed(matrix)
            physical_test = dataset.physical[dataset.test_ids]
            models, match = build_models(seed, config)
            for name in (*models, "linear_transition"):
                model = models.get(name)
                if model is None:
                    started = perf_counter()
                    readout, fit_report = fit_affine(*pairs(train))
                    predict = ObservablePredictor(readout)
                    training = {
                        "duration_seconds": perf_counter() - started,
                        "least_squares_fit": fit_report,
                    }
                    counts = {"trainable_neural": 0, "fitted_linear": 42, "total_fitted": 42}
                    diagnostic = {}
                else:
                    predict, training = train_neural(model, train, seed, config)
                    count = parameter_count(model)
                    counts = {
                        "trainable_neural": count,
                        "frozen_ema": 42,
                        "fitted_linear": 42,
                        "total_fitted": count + 42,
                        "predictor": parameter_count(model.predictor),
                    }
                    diagnostic = diagnostics(model, test, matrix, simulator, config)
                metrics = evaluate(predict, test, physical_test, matrix, simulator, config)
                report["records"].append(
                    {
                        "seed": seed,
                        "condition": condition,
                        "model": name,
                        "parameters": counts,
                        "training": training,
                        "metrics": metrics,
                        "diagnostics": diagnostic,
                        "predictor_capacity_match": match if model is not None else None,
                        "capacity_scope": "Predictors matched; OPF adds 36 learned basis entries",
                        "predictor_flops": None,
                        "flops_note": "Not measured; no compute-equivalence claim",
                    }
                )
                # Preserve completed records even if a later model fails.
                write_json(output / "results.json", report)
    report["status"] = "complete"
    report["summary"] = aggregate(report["records"])
    write_json(output / "results.json", report)
    summary = human_summary(report)
    (output / "summary.md").write_text(summary)
    digest = hashlib.sha256((output / "results.json").read_bytes()).hexdigest()
    (output / "results.sha256").write_text(f"{digest}  results.json\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    preset = parser.add_mutually_exclusive_group(required=True)
    preset.add_argument("--quick", action="store_true")
    preset.add_argument("--full", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = Config.full() if args.full else Config()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    output = args.output_dir or ROOT / "work/coupled_oscillators" / f"{config.preset}-{stamp}"
    report = run(config, output)
    print(human_summary(report))
    print(f"Results: {output.resolve()}")


if __name__ == "__main__":
    main()
