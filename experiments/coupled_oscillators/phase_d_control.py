"""One normalization control for the observed noise-related neural rollout change."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from .noise_data import NoiseConfig, learner_config, noisy_data
from .noise_evaluate import evaluate_noise
from .partial_models import HistoryJEPA, train_history_model
from .phase_d_neural import snapshot_noise
from .phase_d_noise import begin, finish
from .run_experiment import write_json


def run(output, parent_path):
    config = NoiseConfig()
    parent = json.loads((parent_path / "results.json").read_text())
    protocol = begin(output, config, "D5: is noise-related rollout improvement only normalization?")
    protocol.update(
        {
            "parent_sha256": hashlib.sha256(
                (parent_path / "results.json").read_bytes()
            ).hexdigest(),
            "contract": (
                "Train no-noise calibration with moderate-noise observation-only normalization"
            ),
            "comparison": "Standard H8, 1000 steps, same initial weights and normalization buffers",
        }
    )
    write_json(output / "protocol.json", protocol)
    records = []
    started = perf_counter()
    for seed in config.seeds:
        noiseless, truth = noisy_data(config, seed, 0.0)
        noisy, _ = noisy_data(config, seed, 0.15)
        model = HistoryJEPA(
            noisy.train,
            8,
            "standard",
            seed,
            torch.eye(6, dtype=torch.float64),
            learner_config(config),
        )
        paired = next(
            r
            for r in parent["records"]
            if r["seed"] == seed
            and r["noise_ratio"] == 0.15
            and r["history"] == 8
            and r["model"] == "standard"
        )
        expected = paired["trace"][0]["state_dict"]
        assert all(
            torch.equal(value, torch.tensor(expected[key], dtype=value.dtype))
            for key, value in model.state_dict().items()
        )

        def callback(current, step, sensor=noiseless, evaluator=truth):
            return snapshot_noise(current, sensor, evaluator, step)

        trace = train_history_model(model, noiseless.train, seed, learner_config(config), callback)
        records.append(
            {
                "seed": seed,
                "model": "standard",
                "history": 8,
                "training_noise_ratio": 0.0,
                "normalization_noise_ratio": 0.15,
                "initial_state_dict_matches_moderate": True,
                "trace": trace,
                "moderate_trained_clean_context": paired["oracle_clean_context"],
                "cross_noise_evaluation": evaluate_noise(model, noisy, truth, 8),
            }
        )
        print(
            seed,
            "no-noise trained shared normalization h16",
            trace[-1]["metrics"]["interpolation"]["clean_truth_mse"]["16"],
            flush=True,
        )
    finish(output, protocol, records, started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--neural", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    run(args.output, args.neural)


if __name__ == "__main__":
    main()
