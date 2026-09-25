"""Reuse Phase C JEPA adapters with exclusively noisy histories and targets."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch
from jepa_anything_core.baselines import parameter_count

from .noise_data import NoiseConfig, learner_config, noisy_data
from .noise_evaluate import evaluate_noise, state_probe
from .partial_data import windows
from .partial_models import HistoryJEPA, frozen_encoder, latent_diagnostics, train_history_model
from .phase_d_noise import begin, finish
from .run_experiment import write_json


@torch.no_grad()
def snapshot_noise(model, noisy, truth, step):
    train = windows(noisy.train, model.history)
    loss, components = model.loss(train["history"], train["next_history"], train["future"][:, 0])
    test = windows(noisy.test, model.history, 16)
    return {
        "step": step,
        "noisy_training_loss": float(loss),
        "loss_components": components,
        "metrics": evaluate_noise(model, noisy, truth, model.history),
        "diagnostics": latent_diagnostics(model, test["history"]),
        "state_dict": {key: value.tolist() for key, value in model.state_dict().items()},
    }


def neural_panel(output, config, qualification, duration=False):
    parent = json.loads((qualification / "results.json").read_text())
    if parent["status"] != "complete":
        raise ValueError("Need completed noise qualification")
    protocol = begin(
        output,
        config,
        "D neural: noisy self-supervision, learned state versus fixed/transform controls",
    )
    settings = learner_config(config)
    protocol.update(
        {
            "qualification_sha256": hashlib.sha256(
                (qualification / "results.json").read_bytes()
            ).hexdigest(),
            "learner_settings": vars(settings),
            "frame": "native",
            "input_and_target_contract": (
                "H past/current noisy positions; next noisy position "
                "and next noisy-history EMA embedding"
            ),
            "rollout": "autonomous six-dimensional latent recurrence",
            "choice": "H8 primary, H2 moderate-noise control, H8 noise-free Standard control",
            "hidden_state_probes": (
                "only after all main training; no encoder updates or model selection"
            ),
            "duration": duration,
        }
    )
    write_json(output / "protocol.json", protocol)
    families = ("standard", "opf_fixed", "opf", "output_transform")
    cases = [
        (ratio, 8, family)
        for ratio in ((0.15,) if duration else (0.05, 0.15))
        for family in families
    ]
    if not duration:
        cases += [(0.15, 2, "standard"), (0.0, 8, "standard")]
    records = []
    started = perf_counter()
    for seed in config.seeds:
        for ratio, history, family in cases:
            noisy, truth = noisy_data(config, seed, ratio)
            model = HistoryJEPA(
                noisy.train, history, family, seed, torch.eye(6, dtype=torch.float64), settings
            )

            def callback(current, step, sensor=noisy, evaluator=truth):
                return snapshot_noise(current, sensor, evaluator, step)

            trace = train_history_model(model, noisy.train, seed, settings, callback)
            record = {
                "seed": seed,
                "noise_ratio": ratio,
                "history": history,
                "model": family,
                "oracle": False,
                "frame": "native",
                "noise": truth.metadata,
                "trainable_parameters": parameter_count(model),
                "predictor_capacity_match": model.predictor_capacity_match,
                "trace": trace,
                "oracle_clean_context": evaluate_noise(model, noisy, truth, history, True),
            }
            if ratio:
                unseen, unseen_truth = noisy_data(
                    config, seed, ratio, config.unseen_noise_multiplier
                )
                record["unseen_noise"] = {
                    "noise": unseen_truth.metadata,
                    "metrics": evaluate_noise(model, unseen, unseen_truth, history),
                }
            records.append(record)
            write_json(output / f"{seed}-{ratio}-H{history}-{family}.json", record)
            print(
                seed,
                ratio,
                history,
                family,
                "h16",
                trace[-1]["metrics"]["interpolation"]["clean_truth_mse"]["16"],
                flush=True,
            )
    # Hidden labels become available only to frozen post-fit evaluator probes.
    for record in records:
        noisy, truth = noisy_data(config, record["seed"], record["noise_ratio"])
        model = HistoryJEPA(
            noisy.train,
            record["history"],
            record["model"],
            record["seed"],
            torch.eye(6, dtype=torch.float64),
            settings,
        )
        model.load_state_dict(
            {
                key: torch.tensor(value, dtype=torch.float64)
                for key, value in record["trace"][-1]["state_dict"].items()
            }
        )
        encoder = frozen_encoder(model)
        record["state_probe"] = state_probe(encoder, noisy, truth, record["history"])
    finish(output, protocol, records, started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qualification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--duration", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config = replace(NoiseConfig(), training_steps=3000 if args.duration else 1000)
    neural_panel(args.output, config, args.qualification, args.duration)


if __name__ == "__main__":
    main()
