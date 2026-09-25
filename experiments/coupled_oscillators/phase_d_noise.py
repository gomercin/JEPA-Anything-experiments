"""Small staged Phase D panels; output directories are immutable after completion."""

import argparse
import hashlib
import json
import shutil
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import torch

from .noise_baselines import CausalPolynomialState
from .noise_data import NoiseConfig, noisy_data
from .noise_evaluate import evaluate_noise, filter_state_diagnostics, state_probe
from .noise_filters import WindowEKF, identify_mechanics, oracle_mechanics
from .partial_baselines import DelayRegression
from .phase_c_partial import preserved_hashes
from .run_experiment import ROOT, write_json


def protected_hashes():
    return {
        **preserved_hashes(),
        **{
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (ROOT / "work/coupled_oscillators/phase_c_partial").rglob("*")
            if p.is_file()
        },
    }


def begin(output, config, question):
    output.mkdir(parents=True, exist_ok=False)
    sources = list(Path(__file__).parent.glob("*.py"))
    protocol = {
        "exploratory": True,
        "question": question,
        "config": asdict(config),
        "command": sys.argv,
        "torch_version": torch.__version__,
        "contract": "Noisy positions only; targets noisy; clean labels evaluator-only",
        "pulse": "Unknown p1 +0.2 at t24; assimilate eight post-pulse samples; no action input",
        "protected_hashes": protected_hashes(),
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
    }
    for path in sources:
        if path.name.startswith(("noise_", "phase_d_")):
            shutil.copyfile(path, output / path.name)
    write_json(output / "protocol.json", protocol)
    return protocol


def finish(output, protocol, records, started, **extra):
    assert protected_hashes() == protocol["protected_hashes"]
    report = {
        "status": "complete",
        "exploratory": True,
        "protocol": protocol,
        "duration_seconds": perf_counter() - started,
        "records": records,
        "protected_artifacts_unchanged": True,
        **extra,
    }
    write_json(output / "results.json", report)
    (output / "results.sha256").write_text(
        hashlib.sha256((output / "results.json").read_bytes()).hexdigest() + "\n"
    )


def qualify(output, config):
    protocol = begin(output, config, "D0: what does 0/5/15 percent measurement noise break?")
    protocol["ladder_choice"] = "0%, 5%, 15% signal SD, chosen before outcome inspection"
    write_json(output / "protocol.json", protocol)
    records = []
    started = perf_counter()
    for ratio in config.noise_ratios:
        for seed in config.seeds:
            noisy, truth = noisy_data(config, seed, ratio)
            for history in config.histories:
                for family in ("cubic_delay_state", "causal_polynomial_state"):
                    if family == "causal_polynomial_state" and history == 2:
                        continue
                    model = (
                        DelayRegression(noisy.train, history, "svd_cubic_state", config.dt)
                        if family == "cubic_delay_state"
                        else CausalPolynomialState(noisy.train, history, config.dt)
                    )
                    metrics = evaluate_noise(model, noisy, truth, history)
                    record = {
                        "seed": seed,
                        "noise_ratio": ratio,
                        "history": history,
                        "model": family,
                        "oracle": False,
                        "noise": truth.metadata,
                        "fit": model.report(),
                        "metrics": metrics,
                        "oracle_clean_context": evaluate_noise(model, noisy, truth, history, True),
                        "state_probe": state_probe(model.encode, noisy, truth, history),
                    }
                    records.append(record)
                    print(
                        ratio,
                        seed,
                        history,
                        family,
                        "h16 clean",
                        metrics["interpolation"]["clean_truth_mse"]["16"],
                        flush=True,
                    )
            write_json(
                output / "results.json",
                {"status": "running", "protocol": protocol, "records": records},
            )
    finish(output, protocol, records, started)


def filters(output, config, qualification):
    parent = json.loads((qualification / "results.json").read_text())
    if parent["status"] != "complete":
        raise ValueError("Need completed D0 qualification")
    protocol = begin(
        output, config, "D1: observation-identified dynamics + causal EKF versus oracle"
    )
    protocol.update(
        {
            "qualification_sha256": hashlib.sha256(
                (qualification / "results.json").read_bytes()
            ).hexdigest(),
            "choices": "H2/H8; 5/15 percent noise; linear oracle calibration at 15 percent",
            "identification": (
                "Training-only centered local polynomial derivatives; no true coefficients"
            ),
            "evaluation": "Windowed online EKF reset at each context start; no future measurements",
        }
    )
    write_json(output / "protocol.json", protocol)
    records = []
    started = perf_counter()
    for alpha, ratios in ((config.alpha, (0.05, 0.15)), (0.0, (0.15,))):
        for ratio in ratios:
            for seed in config.seeds:
                world = replace(config, alpha=alpha)
                noisy, truth = noisy_data(world, seed, ratio)
                learned = identify_mechanics(noisy.train, config.dt) if alpha else None
                oracle = oracle_mechanics(alpha, truth.metadata["train_sigma"], config.dt)
                models = (
                    {"identified_ekf": learned, "ORACLE_EKF": oracle}
                    if alpha
                    else {"ORACLE_KALMAN": oracle}
                )
                for history in (2, 8):
                    for name, dynamics in models.items():
                        model = WindowEKF(dynamics)
                        metrics = evaluate_noise(model, noisy, truth, history)
                        record = {
                            "seed": seed,
                            "alpha": alpha,
                            "noise_ratio": ratio,
                            "history": history,
                            "model": name,
                            "oracle": model.oracle,
                            "noise": truth.metadata,
                            "fit": model.report(),
                            "metrics": metrics,
                            "state_diagnostics": filter_state_diagnostics(
                                model, noisy, truth, history
                            ),
                            "state_probe": state_probe(model.encode, noisy, truth, history),
                        }
                        if history == 8 and alpha:
                            unseen, unseen_truth = noisy_data(
                                world, seed, ratio, config.unseen_noise_multiplier
                            )
                            # Keep training variance fixed, even in the oracle mismatch test.
                            record["unseen_noise"] = {
                                "noise": unseen_truth.metadata,
                                "metrics": evaluate_noise(model, unseen, unseen_truth, history),
                            }
                        records.append(record)
                        print(
                            alpha,
                            ratio,
                            seed,
                            history,
                            name,
                            "h16",
                            metrics["interpolation"]["clean_truth_mse"]["16"],
                            flush=True,
                        )
                write_json(
                    output / "results.json",
                    {"status": "running", "protocol": protocol, "records": records},
                )
    finish(output, protocol, records, started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--qualify", action="store_true")
    mode.add_argument("--filters", action="store_true")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.qualify:
        qualify(args.output, NoiseConfig())
    else:
        if args.qualification is None:
            parser.error("--filters requires --qualification")
        filters(args.output, NoiseConfig(), args.qualification)


if __name__ == "__main__":
    main()
