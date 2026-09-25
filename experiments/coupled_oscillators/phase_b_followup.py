"""Two targeted checks prompted by B1: conventional invariance and training duration."""

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import torch
from jepa_anything_core.baselines import capacity_match_report, parameter_count
from torch import nn

from .dataset import pairs
from .evaluate import rollout
from .frozen_encoder import PhysicalPredictor, make_model
from .interrogate import orthogonal_draws
from .models import ObservablePredictor, fit_affine
from .nonlinear import DuffingSimulator, PolynomialPredictor
from .phase_b import (
    IDENTITY,
    REGIMES,
    ROTATION_SEED,
    physical_splits,
    protected_hashes,
    score,
    sources,
    summarize,
    train_frozen,
)
from .run_experiment import Config, write_json


def output_transform_control(seed, config):
    model = make_model(seed, "standard", IDENTITY, IDENTITY, config)
    with torch.random.fork_rng():
        torch.manual_seed(seed + 30000)
        transform = nn.Linear(6, 6, bias=False, dtype=torch.float64)
    with torch.no_grad():
        transform.weight.copy_(IDENTITY)
    model.predictor = nn.Sequential(model.predictor, transform)
    return model


def fold_opf_predictor(model, frame):
    """Fold the core's actual linear synthesis into an ordinary 422-parameter MLP.

    This is an equality of physical functions, not a factor interpretation.
    Obtain synthesis from core compose on coordinate unit vectors: no custom OPF.
    """
    trunk = copy.deepcopy(model.predictor.trunk)
    with torch.random.fork_rng():
        output = nn.Linear(trunk[0].out_features, 6, dtype=torch.float64)
    with torch.no_grad():
        trunk[0].weight.copy_(trunk[0].weight @ frame)
        decoder_rows = model.opf.compose(IDENTITY.reshape(6, 3, 2)) @ frame
        weights = torch.cat([head.weight for head in model.predictor.heads])
        biases = torch.cat([head.bias for head in model.predictor.heads])
        output.weight.copy_(decoder_rows.T @ weights)
        output.bias.copy_(biases @ decoder_rows)
    return nn.Sequential(trunk, output)


def transform_control(output, panel):
    parent = json.loads((panel / "results.json").read_text())
    if parent["status"] != "complete" or parent["protocol"]["config"]["training_steps"] != 3000:
        raise ValueError("Need the completed B2 duration control")
    config = replace(Config(), training_steps=3000)
    output.mkdir(parents=True, exist_ok=False)
    protected = protected_hashes()
    protocol = {
        "exploratory": True,
        "parent_sha256": hashlib.sha256((panel / "results.json").read_bytes()).hexdigest(),
        "config": asdict(config),
        "scope": "alpha=.5, native only, seeds 11/22/33, 3000 steps, unchanged Adam .003",
        "question": (
            "B2 gives OPF a small native-frame amplitude-shift gain. Does ordinary "
            "learnable output transformation with the same 36 extra parameters reproduce it?"
        ),
        "control": (
            "Standard JEPA followed by bias-free 6x6 linear map, initialized to identity; "
            "same initial physical function and parameter count as OPF, no Gram objective. "
            "Loss metric and optimization geometry are not matched."
        ),
        "folding": "Posthoc exact synthesis folding, no training or subspace interpretation",
        "command": sys.argv,
        "sources": sources(),
        "protected_hashes_before": protected,
    }
    write_json(output / "protocol.json", protocol)
    (output / "phase_b_followup.py").write_bytes(Path(__file__).read_bytes())
    records, folded_reports = [], []
    started = perf_counter()
    simulator = DuffingSimulator(alpha=0.5, substeps=8)
    for seed in config.seeds:
        _, train, splits = physical_splits(simulator, seed, config)
        control = output_transform_control(seed, config)
        opf = make_model(seed, "opf", IDENTITY, IDENTITY, config)
        match = capacity_match_report(opf, control).to_dict()
        assert match["matched"]
        with torch.no_grad():
            initial_difference = float(
                (
                    PhysicalPredictor(control, IDENTITY, IDENTITY)(train)
                    - PhysicalPredictor(opf, IDENTITY, IDENTITY)(train)
                )
                .abs()
                .max()
            )
        if initial_difference > 1e-12:
            raise RuntimeError("Control initialization mismatch")
        trace = train_frozen(control, IDENTITY, train, splits, simulator, seed, config)
        record = {
            "regime": "MODERATE_NONLINEAR",
            "seed": seed,
            "model": "output_transform",
            "frame": "LATENT_NATIVE",
            "parameter_count": parameter_count(control),
            "capacity_match": match,
            "initial_function_max_difference": initial_difference,
            "output_transform_singular_values": torch.linalg.svdvals(
                control.predictor[1].weight
            ).tolist(),
            "trace": trace,
        }
        records.append(record)
        write_json(output / f"{seed}-output_transform.json", record)
        earlier = next(r for r in parent["records"] if r["model"] == "opf" and r["seed"] == seed)
        opf.load_state_dict(
            {
                key: torch.tensor(value, dtype=torch.float64)
                for key, value in earlier["trace"][-1]["state_dict"].items()
            }
        )
        folded = fold_opf_predictor(opf, IDENTITY)
        predict = PhysicalPredictor(opf, IDENTITY, IDENTITY)
        checks = {}
        with torch.no_grad():
            for name, states in splits.items():
                probes = states[:, [0, config.pulse_time]].reshape(-1, 6)
                checks[name] = float(
                    (rollout(folded, probes, 16) - rollout(predict, probes, 16)).abs().max()
                )
        folded_reports.append(
            {
                "seed": seed,
                "folded_parameter_count": parameter_count(folded),
                "max_physical_rollout_difference": checks,
            }
        )
        print(
            seed,
            "output_transform h16",
            trace[-1]["metrics"]["interpolation"]["rollout_mse"]["16"],
            "folding error",
            checks,
            flush=True,
        )
    assert protected_hashes() == protected, "Prior evidence changed"
    report = {
        "status": "complete",
        "exploratory": True,
        "protocol": protocol,
        "records": records,
        "summary": summarize(records),
        "folded_opf": folded_reports,
        "duration_seconds": perf_counter() - started,
        "protected_artifacts_unchanged": True,
    }
    write_json(output / "results.json", report)
    (output / "results.sha256").write_text(
        hashlib.sha256((output / "results.json").read_bytes()).hexdigest() + "\n"
    )


def run(output, panel):
    parent = json.loads((panel / "results.json").read_text())
    if parent["status"] != "complete":
        raise ValueError("Need a completed B1 panel")
    config = replace(Config(), training_steps=3000)
    output.mkdir(parents=True, exist_ok=False)
    protected = protected_hashes()
    protocol = {
        "exploratory": True,
        "parent_sha256": hashlib.sha256((panel / "results.json").read_bytes()).hexdigest(),
        "config": asdict(config),
        "rationale": (
            "B1 cubic prediction is orders of magnitude better than neural prediction. "
            "Check coordinate choice/numerical truth as alternative explanations. "
            "Neural errors improve sharply 250->1000: replay to 3000 in moderate/native "
            "to separate unfinished learning from a representational ceiling."
        ),
        "conventional_checks": (
            "Both nonlinear strengths; 3 seeds; full cubic basis in R2718 and physical "
            "coordinates; 16-substep truth; linear mode re-expression"
        ),
        "duration_scope": (
            "Moderate alpha=.5 only, native frame, same 3 families and seeds, unchanged "
            "Adam .003; no selection of best checkpoint"
        ),
        "stop": "Stop strength search if ordinary cubic identification remains sufficient",
        "command": sys.argv,
        "protected_hashes_before": protected,
        "sources": sources(),
    }
    write_json(output / "protocol.json", protocol)
    for name in ("phase_b_followup.py", "phase_b.py", "nonlinear.py"):
        (output / name).write_bytes(Path(__file__).with_name(name).read_bytes())
    rotation = orthogonal_draws(ROTATION_SEED, 1)[0]
    checks, records = [], []
    started = perf_counter()
    for regime in ("WEAK_NONLINEAR", "MODERATE_NONLINEAR"):
        simulator = DuffingSimulator(alpha=REGIMES[regime], substeps=8)
        for seed in config.seeds:
            _, train, splits = physical_splits(simulator, seed, config)
            inputs, targets = pairs(train)
            native = PolynomialPredictor(inputs, targets)
            rotated = PolynomialPredictor(inputs @ rotation.T, targets @ rotation.T)

            def rotated_predict(state, fitted=rotated):
                return fitted(state @ rotation.T) @ rotation

            # Full linear mode coordinates carry all six dimensions. This is
            # an unrestricted reference, not three artificially independent fits.
            modes = simulator.modes()[1].permute(1, 0, 2).reshape(6, 6)
            linear = ObservablePredictor(fit_affine(inputs, targets)[0])
            modal = ObservablePredictor(fit_affine(inputs @ modes, targets @ modes)[0])
            refined_simulator = replace(simulator, substeps=16)
            refined_splits = {
                name: refined_simulator.simulate(states[:, 0], config.trajectory_steps)
                for name, states in splits.items()
            }
            coordinate_checks = {}
            for name, states in splits.items():
                probes = states[:, [0, config.pulse_time]].reshape(-1, 6)
                with torch.no_grad():
                    delta = rollout(native, probes, 16) - rollout(rotated_predict, probes, 16)
                    mode_delta = linear(states) - modal(states @ modes) @ modes.T
                coordinate_checks[name] = {
                    "polynomial_rotation_rollout_max_abs": float(delta.abs().max()),
                    "polynomial_rotation_h16_function_mse": float(delta[:, -1].square().mean()),
                    "linear_normal_mode_prediction_max_abs": float(mode_delta.abs().max()),
                    "coarse_vs_refined_state_max_abs": float(
                        (states - refined_splits[name]).abs().max()
                    ),
                }
            checks.append(
                {
                    "regime": regime,
                    "seed": seed,
                    "coordinate_checks": coordinate_checks,
                    "coarse_truth_metrics": score(native, splits, simulator, config),
                    "refined_truth_metrics": score(
                        native, refined_splits, refined_simulator, config
                    ),
                    "rotated_fit": rotated.fit_report,
                }
            )
            print(regime, seed, "conventional checks", coordinate_checks, flush=True)
            if regime == "MODERATE_NONLINEAR":
                for family in ("standard", "opf_fixed", "opf"):
                    model = make_model(seed, family, IDENTITY, IDENTITY, config)
                    begin = perf_counter()
                    trace = train_frozen(model, IDENTITY, train, splits, simulator, seed, config)
                    earlier = next(
                        r
                        for r in parent["records"]
                        if r["regime"] == regime
                        and r["seed"] == seed
                        and r["model"] == family
                        and r["frame"] == "LATENT_NATIVE"
                    )
                    replay = next(t for t in trace if t["step"] == 1000)
                    replay_delta = max(
                        abs(replay["metrics"][s][m][h] - earlier["trace"][-1]["metrics"][s][m][h])
                        for s in splits
                        for m in ("rollout_mse", "pulse_rollout_mse", "pulse_response_mse")
                        for h in ("1", "4", "8", "16")
                    )
                    if replay_delta > 1e-12:
                        raise RuntimeError("Duration control did not reproduce the B1 prefix")
                    record = {
                        "regime": regime,
                        "seed": seed,
                        "model": family,
                        "frame": "LATENT_NATIVE",
                        "parameter_count": parameter_count(model),
                        "replay_1000_max_metric_difference": replay_delta,
                        "duration_seconds": perf_counter() - begin,
                        "trace": trace,
                    }
                    records.append(record)
                    write_json(output / f"{seed}-{family}.json", record)
                    print(
                        seed,
                        family,
                        "3000 h16",
                        trace[-1]["metrics"]["interpolation"]["rollout_mse"]["16"],
                        flush=True,
                    )
    assert protected_hashes() == protected, "Prior evidence changed"
    report = {
        "status": "complete",
        "exploratory": True,
        "protocol": protocol,
        "conventional_checks": checks,
        "records": records,
        "summary": summarize(records),
        "duration_seconds": perf_counter() - started,
        "protected_artifacts_unchanged": True,
    }
    write_json(output / "results.json", report)
    (output / "results.sha256").write_text(
        hashlib.sha256((output / "results.json").read_bytes()).hexdigest() + "\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--transform-control", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.transform_control:
        transform_control(args.output, args.panel)
    else:
        run(args.output, args.panel)


if __name__ == "__main__":
    main()
