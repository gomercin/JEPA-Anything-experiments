"""Bounded, exploratory Duffing oscillator panels; no changes to prior evidence."""

import argparse
import hashlib
import json
import statistics
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import torch
from jepa_anything_core.audit import audit_factor_geometry, audit_opf_geometry
from jepa_anything_core.baselines import parameter_count

from .dataset import make_dataset, observation_matrix, pairs
from .evaluate import evaluate, rollout
from .frozen_encoder import PhysicalPredictor, latent_equivalence, make_model
from .interrogate import orthogonal_draws
from .models import ObservablePredictor, fit_affine
from .nonlinear import DuffingSimulator, PolynomialPredictor
from .run_experiment import ROOT, Config, fingerprint, write_json
from .train import training_loss

IDENTITY = torch.eye(6, dtype=torch.float64)
ROTATION_SEED = 2718
AMPLITUDE_MULTIPLIER = 1.5
REGIMES = {"LINEAR": 0.0, "WEAK_NONLINEAR": 0.1, "MODERATE_NONLINEAR": 0.5}
PROTECTED = ("phase-a-quick", "interrogation", "frozen_encoder")


def protected_hashes():
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for name in PROTECTED
        for path in sorted((ROOT / "work/coupled_oscillators" / name).rglob("*"))
        if path.is_file()
    }


def sources():
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path(__file__).parent.glob("*.py"))
    }


def physical_splits(simulator, seed, config):
    data = make_dataset(
        simulator, seed, config.train_count, config.test_count, config.trajectory_steps
    )
    train, test = data.physical[data.train_ids], data.physical[data.test_ids]
    shifted = simulator.simulate(AMPLITUDE_MULTIPLIER * test[:, 0], config.trajectory_steps)
    return data, train, {"interpolation": test, "amplitude_shift": shifted}


def quantiles(values):
    return torch.quantile(
        values.flatten(), torch.tensor([0, 0.5, 0.95, 1], dtype=torch.float64)
    ).tolist()


def state_diagnostics(states, train, simulator):
    energy, train_energy = simulator.energy(states), simulator.energy(train)
    norms, train_norms = states.norm(dim=-1), train.norm(dim=-1)
    return {
        "quantile_probabilities": [0, 0.5, 0.95, 1],
        "state_norm_quantiles": quantiles(norms),
        "absolute_position_quantiles": quantiles(states[..., :3].abs()),
        "energy_quantiles": quantiles(energy),
        "maximum_energy_increase_per_interval": float((energy[:, 1:] - energy[:, :-1]).max()),
        "final_to_initial_energy_ratio_quantiles": quantiles(energy[:, -1] / energy[:, 0]),
        "fraction_above_train_max_norm": float((norms > train_norms.max()).double().mean()),
        "fraction_above_train_max_energy": float((energy > train_energy.max()).double().mean()),
        "fraction_outside_train_coordinate_box": float(
            ((states < train.amin((0, 1))) | (states > train.amax((0, 1)))).any(-1).double().mean()
        ),
    }


def score(predict, splits, simulator, config):
    return {
        name: evaluate(predict, states, states, IDENTITY, simulator, config)
        for name, states in splits.items()
    }


def stability_diagnostics(predict, states, train, simulator, config):
    """Common physical coordinates; local expansion is not a global stability test."""
    probes = states[:, [0, config.pulse_time]].reshape(-1, 6)
    with torch.no_grad():
        predicted = rollout(predict, probes, max(config.horizons))
        truth = simulator.simulate(probes, max(config.horizons))
        limit = train.norm(dim=-1).max()
        norms, truth_norms = predicted.norm(dim=-1), truth.norm(dim=-1)
        outside = norms > limit
        first_exit = [
            int(indices[0]) if len(indices := row.nonzero().flatten()) else None for row in outside
        ]
    # Fixed first four trajectories, t=0 and pulse origin, independent of outcomes.
    with torch.enable_grad():
        jacobians = torch.stack(
            [torch.autograd.functional.jacobian(predict, state) for state in probes[:8]]
        )
    radii = torch.linalg.eigvals(jacobians).abs().amax(-1)
    return {
        "train_max_state_norm": float(limit),
        "predicted_max_norm": float(norms.max()),
        "true_max_norm": float(truth_norms.max()),
        "max_norm_ratio_to_initial": float((norms / norms[:, :1]).max()),
        "predicted_fraction_outside_train_norm_by_step": outside.double().mean(0).tolist(),
        "true_fraction_outside_train_norm_by_step": (truth_norms > limit).double().mean(0).tolist(),
        "first_step_outside_train_norm_per_probe": first_exit,
        "local_jacobian_spectral_radii": radii.tolist(),
        "local_jacobian_max_singular_values": torch.linalg.svdvals(jacobians)[:, 0].tolist(),
    }


@torch.no_grad()
def neural_checkpoint(model, frame, train, splits, simulator, config, final=False):
    context, target = pairs(train)
    loss, components = training_loss(model, context, target, config)
    predict = PhysicalPredictor(model, frame, IDENTITY)
    encoded = model.target_encoder(splits["interpolation"].reshape(-1, 6))
    audit = (
        audit_opf_geometry(model.opf, encoded).to_dict()
        if hasattr(model, "opf")
        else audit_factor_geometry(encoded.reshape(-1, 3, 2)).to_dict()
    )
    report = {
        "training_loss": float(loss),
        "loss_components": components,
        "metrics": score(predict, splits, simulator, config),
        "audit": audit,
        "encoder_drift": float((model.context_encoder.weight - frame).abs().max()),
        "target_encoder_drift": float((model.target_encoder.weight - frame).abs().max()),
        "encoder_singular_values": torch.linalg.svdvals(model.context_encoder.weight).tolist(),
        "physical_signatures": {
            name: rollout(predict, states[:, [0, config.pulse_time]].reshape(-1, 6), 16)[
                :, list(config.horizons)
            ].tolist()
            for name, states in splits.items()
        },
        "state_dict": {key: value.tolist() for key, value in model.state_dict().items()},
    }
    if final:
        report["stability"] = {
            name: stability_diagnostics(predict, states, train, simulator, config)
            for name, states in splits.items()
        }
    return report


def train_frozen(model, frame, train, splits, simulator, seed, config):
    context, target = pairs(train)
    rng = torch.Generator().manual_seed(seed + 20000)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=config.learning_rate
    )
    checkpoints = sorted(
        {
            0,
            min(250, config.training_steps),
            min(1000, config.training_steps),
            config.training_steps,
        }
    )
    trace = [{"step": 0, **neural_checkpoint(model, frame, train, splits, simulator, config)}]
    for step in range(1, config.training_steps + 1):
        indices = torch.randint(len(context), (config.batch_size,), generator=rng)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = training_loss(model, context[indices], target[indices], config)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        if hasattr(model, "opf"):
            model.opf.after_optimizer_step(step)
        # As in Phase C, no EMA arithmetic on identical frozen encoder buffers.
        if step in checkpoints:
            trace.append(
                {
                    "step": step,
                    **neural_checkpoint(
                        model,
                        frame,
                        train,
                        splits,
                        simulator,
                        config,
                        final=step == config.training_steps,
                    ),
                }
            )
    return trace


def summarize(records):
    rows = []
    keys = sorted({(r["regime"], r["model"], r["frame"]) for r in records})
    for regime, model, frame in keys:
        group = [
            r for r in records if (r["regime"], r["model"], r["frame"]) == (regime, model, frame)
        ]
        for split in ("interpolation", "amplitude_shift"):
            metrics = {}
            for metric in ("rollout_mse", "pulse_rollout_mse", "pulse_response_mse"):
                metrics[metric] = {}
                for horizon in ("1", "4", "8", "16"):
                    values = [r["trace"][-1]["metrics"][split][metric][horizon] for r in group]
                    metrics[metric][horizon] = {
                        "mean": statistics.mean(values),
                        "std": statistics.stdev(values) if len(values) > 1 else None,
                        "values": values,
                    }
            rows.append(
                {
                    "regime": regime,
                    "model": model,
                    "frame": frame,
                    "split": split,
                    "seeds": [r["seed"] for r in group],
                    "metrics": metrics,
                }
            )
    return rows


def paired_frames(records):
    pairs_report = []
    for native in [r for r in records if r["frame"] == "LATENT_NATIVE"]:
        rotated = next(
            r
            for r in records
            if r["frame"] == "LATENT_ROTATED"
            and all(r[k] == native[k] for k in ("regime", "model", "seed"))
        )
        trace = []
        for left, right in zip(native["trace"], rotated["trace"], strict=True):
            signatures = {}
            for split in left["physical_signatures"]:
                delta = torch.tensor(
                    left["physical_signatures"][split], dtype=torch.float64
                ) - torch.tensor(right["physical_signatures"][split], dtype=torch.float64)
                signatures[split] = {
                    "mse_by_horizon": delta.square().mean((0, 2)).tolist(),
                    "max_abs_by_horizon": delta.abs().amax((0, 2)).tolist(),
                }
            trace.append({"step": left["step"], "physical_function_difference": signatures})
        pairs_report.append(
            {
                "regime": native["regime"],
                "model": native["model"],
                "seed": native["seed"],
                "trace": trace,
            }
        )
    return pairs_report


def run_panel(output, qualification, config):
    accepted = json.loads((qualification / "qualification.json").read_text())
    qualified_protocol = json.loads((qualification / "protocol.json").read_text())
    if not accepted["all_engineering_gates_pass"] or accepted["regimes"] != REGIMES:
        raise ValueError("Need the successful B0 qualification for this ladder")
    if qualified_protocol["config"] != json.loads(json.dumps(asdict(config))):
        raise ValueError("Panel and qualification must use the same dataset/training configuration")
    output.mkdir(parents=True, exist_ok=False)
    protected = protected_hashes()
    frames = {"LATENT_NATIVE": IDENTITY, "LATENT_ROTATED": orthogonal_draws(ROTATION_SEED, 1)[0]}
    protocol = {
        "exploratory": True,
        "stage": "B1 frozen-encoder model comparison",
        "regimes": REGIMES,
        "config": asdict(config),
        "simulator_substeps": qualified_protocol["integrator_substeps"],
        "qualification": str(qualification),
        "qualification_sha256": hashlib.sha256(
            (qualification / "results.json").read_bytes()
        ).hexdigest(),
        "alpha_selection": ".1/.5 accepted from B0 stability and linear error; no neural outcomes",
        "rotation_seed": ROTATION_SEED,
        "frames": {name: frame.tolist() for name, frame in frames.items()},
        "observation": "Physical-frame input; RAW/MIXED equivalence remains gated",
        "amplitude_multiplier": AMPLITUDE_MULTIPLIER,
        "optimizer": {
            "name": "Adam",
            "lr": config.learning_rate,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0,
        },
        "batch_seed_rule": "dataset/model seed + 20000; identical batches across frames/families",
        "models": ["linear", "polynomial_cubic", "standard", "opf_fixed", "opf"],
        "baseline_information": (
            "Same train-only (x_t,x_t+1) pairs; full monomials through degree 3, "
            "no simulator coefficients or derivatives"
        ),
        "matching": (
            "422 shared-trunk predictor parameters; learned OPF adds 36; initial physical "
            "functions matched; frozen encoder d=6,K=3,r=2"
        ),
        "normal_modes": (
            "Complete linear-mode coordinates plus unrestricted linear transition are an "
            "invertible re-expression of linear ID"
        ),
        "command": sys.argv,
        "protected_hashes_before": protected,
        "sources": sources(),
        "torch_version": torch.__version__,
    }
    write_json(output / "protocol.json", protocol)
    for name in ("phase_b.py", "nonlinear.py"):
        (output / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    records, datasets = [], []
    started = perf_counter()
    for regime, alpha in REGIMES.items():
        simulator = DuffingSimulator(alpha=alpha, substeps=protocol["simulator_substeps"])
        for seed in config.seeds:
            data, train, splits = physical_splits(simulator, seed, config)
            gates = {
                name: latent_equivalence(data.physical, frame, observation_matrix("MIXED"))
                for name, frame in frames.items()
            }
            if not all(g["passed"] for g in gates.values()):
                raise RuntimeError("Frozen encoder equivalence failed")
            datasets.append(
                {
                    "regime": regime,
                    "seed": seed,
                    "simulator": asdict(simulator),
                    "data_hash": fingerprint(data.physical),
                    "shifted_hash": fingerprint(splits["amplitude_shift"]),
                    "train_ids": data.train_ids.tolist(),
                    "test_ids": data.test_ids.tolist(),
                    "gates": gates,
                }
            )
            inputs, targets = pairs(train)
            weights, fit = fit_affine(inputs, targets)
            polynomial = PolynomialPredictor(inputs, targets)
            for family, predict, fit_report in (
                (
                    "linear",
                    ObservablePredictor(weights),
                    {**fit, "parameter_count": weights.numel(), "weights": weights.tolist()},
                ),
                ("polynomial_cubic", polynomial, polynomial.fit_report),
            ):
                baseline = {
                    "metrics": score(predict, splits, simulator, config),
                    "stability": {
                        name: stability_diagnostics(predict, states, train, simulator, config)
                        for name, states in splits.items()
                    },
                }
                records.append(
                    {
                        "regime": regime,
                        "seed": seed,
                        "model": family,
                        "frame": "PHYSICAL",
                        "fit": fit_report,
                        "parameter_count": fit_report["parameter_count"],
                        "trace": [baseline],
                    }
                )
                print(
                    regime,
                    seed,
                    family,
                    "h16",
                    baseline["metrics"]["interpolation"]["rollout_mse"]["16"],
                    flush=True,
                )
            for family in ("standard", "opf_fixed", "opf"):
                for frame_name, frame in frames.items():
                    model = make_model(seed, family, frame, IDENTITY, config)
                    begin = perf_counter()
                    trace = train_frozen(model, frame, train, splits, simulator, seed, config)
                    record = {
                        "regime": regime,
                        "seed": seed,
                        "model": family,
                        "frame": frame_name,
                        "parameter_count": parameter_count(model),
                        "duration_seconds": perf_counter() - begin,
                        "trace": trace,
                    }
                    records.append(record)
                    write_json(output / f"{regime}-{seed}-{family}-{frame_name}.json", record)
                    print(
                        regime,
                        seed,
                        family,
                        frame_name,
                        "h16",
                        trace[-1]["metrics"]["interpolation"]["rollout_mse"]["16"],
                        flush=True,
                    )
            write_json(
                output / "results.json",
                {"protocol": protocol, "datasets": datasets, "records": records},
            )
    assert protected_hashes() == protected, "Prior evidence changed"
    report = {
        "status": "complete",
        "exploratory": True,
        "duration_seconds": perf_counter() - started,
        "protocol": protocol,
        "datasets": datasets,
        "records": records,
        "summary": summarize(records),
        "paired_frames": paired_frames(records),
        "protected_artifacts_unchanged": True,
    }
    write_json(output / "results.json", report)
    digest = hashlib.sha256((output / "results.json").read_bytes()).hexdigest()
    (output / "results.sha256").write_text(digest + "\n")
    lines = [
        "# Phase B exploratory panel",
        "",
        "Mean ± sample SD, three paired seeds; physical state MSE.",
        "",
        "| Regime | Model | Frame | Split | h1 | h4 | h8 | h16 | Pulse response h16 |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in report["summary"]:
        entries = [row["metrics"]["rollout_mse"][str(h)] for h in config.horizons] + [
            row["metrics"]["pulse_response_mse"]["16"]
        ]
        cells = [
            f"{entry['mean']:.4g} ± {entry['std']:.3g}"
            if entry["std"] is not None
            else f"{entry['mean']:.4g}"
            for entry in entries
        ]
        lines.append(
            "| "
            + " | ".join([row["regime"], row["model"], row["frame"], row["split"], *cells])
            + " |"
        )
    (output / "summary.md").write_text("\n".join(lines) + "\n")


def qualify(output, config, substeps=8):
    """B0 runs only simulator and linear ID; no neural outcomes select alpha."""
    if substeps < 2 or substeps % 2:
        raise ValueError("Qualification needs an even substep count >= 2")
    output.mkdir(parents=True, exist_ok=False)
    protected = protected_hashes()
    protocol = {
        "exploratory": True,
        "stage": "B0 simulator qualification, before nonlinear/neural model comparison",
        "config": asdict(config),
        "candidate_regimes": REGIMES,
        "amplitude_multiplier": AMPLITUDE_MULTIPLIER,
        "amplitude_shift": "Same held-out initial directions scaled 1.5; overlapping support",
        "integrator_substeps": substeps,
        "integrator": "RK4 inside dt=.1; exact Phase A flow at alpha=0",
        "convergence_gate": (
            "max absolute chosen-vs-double substep state error < 1e-5 on train/test/shift/pulse"
        ),
        "stability_gate": "finite trajectories; energy cannot increase by more than 1e-9",
        "selection_rule": (
            "Accept .1/.5 if stable with distinct linear errors; no neural results consulted"
        ),
        "pulse": {"time_index": config.pulse_time, "velocity_index": 0, "magnitude": 0.2},
        "command": sys.argv,
        "protected_hashes_before": protected,
        "sources": sources(),
    }
    write_json(output / "protocol.json", protocol)
    for name in ("phase_b.py", "nonlinear.py"):
        (output / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    records = []
    for regime, alpha in REGIMES.items():
        simulator = DuffingSimulator(alpha=alpha, substeps=substeps)
        for seed in config.seeds:
            data, train, splits = physical_splits(simulator, seed, config)
            inputs, targets = pairs(train)
            weights, fit = fit_affine(inputs, targets)
            linear = ObservablePredictor(weights)
            initial = torch.cat((data.physical[:, 0], splits["amplitude_shift"][:, 0]))
            pulse_initial = torch.cat([states[:, config.pulse_time] for states in splits.values()])
            pulse_initial[:, 3] += config.pulse_magnitude
            initial = torch.cat((initial, pulse_initial))
            paths = {
                n: replace(simulator, substeps=n).simulate(initial, config.trajectory_steps)
                for n in (substeps // 2, substeps, 2 * substeps)
            }
            errors = {
                f"substeps_{a}_vs_{b}_max_abs": float((paths[a] - paths[b]).abs().max())
                for a, b in ((substeps // 2, substeps), (substeps, 2 * substeps))
            }
            energies = simulator.energy(paths[substeps])
            stable = (
                bool(torch.isfinite(paths[substeps]).all())
                and float((energies[:, 1:] - energies[:, :-1]).max()) < 1e-9
            )
            gates = {
                "integration": errors[f"substeps_{substeps}_vs_{2 * substeps}_max_abs"] < 1e-5,
                "stable_energy": stable,
                "latent_equivalence": latent_equivalence(
                    data.physical,
                    orthogonal_draws(ROTATION_SEED, 1)[0],
                    observation_matrix("MIXED"),
                ),
            }
            record = {
                "regime": regime,
                "seed": seed,
                "simulator": asdict(simulator),
                "data_hash": fingerprint(data.physical),
                "train_ids": data.train_ids.tolist(),
                "test_ids": data.test_ids.tolist(),
                "integration": errors,
                "gates": gates,
                "states": {
                    name: state_diagnostics(states, train, simulator)
                    for name, states in {"train": train, **splits}.items()
                },
                "linear_fit": fit,
                "linear_metrics": score(linear, splits, simulator, config),
            }
            records.append(record)
            write_json(output / "results.json", {"protocol": protocol, "records": records})
            print(
                regime,
                seed,
                errors,
                "linear h16",
                record["linear_metrics"]["interpolation"]["rollout_mse"]["16"],
                flush=True,
            )
    assert protected_hashes() == protected, "Prior evidence changed"
    passed = all(
        r["gates"]["integration"]
        and r["gates"]["stable_energy"]
        and r["gates"]["latent_equivalence"]["passed"]
        for r in records
    )
    write_json(
        output / "qualification.json", {"all_engineering_gates_pass": passed, "regimes": REGIMES}
    )
    if not passed:
        raise RuntimeError("B0 failed: inspect the saved qualification before any model comparison")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--qualify", action="store_true")
    mode.add_argument("--panel", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--substeps", type=int, default=8)
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = replace(Config(), training_steps=1000)
    if args.qualify:
        qualify(args.output, config, args.substeps)
    else:
        if args.qualification is None:
            parser.error("--panel requires --qualification")
        run_panel(args.output, args.qualification, config)


if __name__ == "__main__":
    main()
