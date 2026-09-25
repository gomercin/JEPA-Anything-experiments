"""Small controls prompted by partial-observation results, not a parameter search."""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
from time import perf_counter

import torch

from .interrogate import orthogonal_draws
from .models import ObservablePredictor, fit_affine
from .nonlinear import polynomial_features
from .partial_baselines import DelayRegression, RecentTwoState, evaluate_observer
from .partial_data import PartialConfig, observer_data, windows
from .partial_models import HistoryJEPA
from .partial_probes import sufficiency_probes
from .phase_c_partial import begin, finish
from .run_experiment import write_json


def history_jacobian_radii(model, histories):
    history, width = histories.shape[-2:]

    def step(flat):
        past = flat.reshape(history, width)
        following = model.forecast(past[None], 1)[0, 0]
        return torch.cat((past[1:].flatten(), following))

    matrices = torch.stack(
        [torch.autograd.functional.jacobian(step, value.flatten()) for value in histories]
    )
    return torch.linalg.eigvals(matrices).abs().amax(-1).tolist()


def ridge_control(output):
    config = PartialConfig()
    protocol = begin(
        output,
        config,
        "C0 ridge control",
        "Ill-conditioned full cubic delays: does one fixed regularizer stabilize recurrence?",
    )
    protocol["ridge"] = {
        "lambda": 1e-6,
        "scale": "mean squared error",
        "intercept_penalty": False,
        "selection": "one fixed numerical regularizer, no grid or outcome-selected coefficient",
    }
    write_json(output / "protocol.json", protocol)
    records = []
    started = perf_counter()
    for seed in config.seeds:
        for regime in ("POSITIONS", "LOCAL"):
            data, _ = observer_data(config, seed, regime)
            model = DelayRegression(data.train, 4, "cubic_full", config.dt)
            test = windows(data.test, 4, 16)["history"][::34][:4]
            unregularized_radii = history_jacobian_radii(model, test)
            batch = windows(data.train, 4)
            features = polynomial_features(model.encode(batch["history"])) / model.predictor.scale
            target = batch["future"][:, 0]
            penalty = torch.eye(features.shape[-1], dtype=torch.float64)
            penalty[0, 0] = 0
            model.predictor.weights = torch.linalg.solve(
                features.T @ features + len(features) * 1e-6 * penalty,
                features.T @ target,
            )
            model.predictor.fit_report.update(
                {
                    "ridge_lambda": 1e-6,
                    "weights": model.predictor.weights.tolist(),
                    "train_mse": float(
                        (features @ model.predictor.weights - target).square().mean()
                    ),
                }
            )
            metrics = evaluate_observer(model, data, 4)
            records.append(
                {
                    "seed": seed,
                    "observation": regime,
                    "history": 4,
                    "model": "cubic_full_ridge",
                    "fit": model.report(),
                    "metrics": metrics,
                    "unregularized_jacobian_radii": unregularized_radii,
                    "ridge_jacobian_radii": history_jacobian_radii(model, test),
                }
            )
            print(
                seed,
                regime,
                "ridge",
                metrics["interpolation"]["rollout_mse"],
                "shift h16",
                metrics["amplitude_shift"]["rollout_mse"]["16"],
                flush=True,
            )
    finish(output, protocol, records, started)


class ReencodedForecast:
    def __init__(self, model):
        self.model = model

    def forecast(self, history, steps):
        return self.model.forecast(history, steps, history_feedback=True)


class AlignedEMAForecast:
    def __init__(self, model, observations):
        self.model = model
        batch = windows(observations, model.history)
        with torch.no_grad():
            source = model.jepa.target_encoder(batch["next_history"])
            target = model.encode(batch["next_history"])
            weights, self.fit = fit_affine(source, target)
        self.adapter = ObservablePredictor(weights)

    def forecast(self, history, steps):
        state = self.model.encode(history)
        result = []
        for _ in range(steps):
            prediction = self.model.predict_latent(state)
            result.append(self.model.decode(prediction))
            state = self.adapter(prediction)
        return torch.stack(result, 1)


def diagnose(output, panel):
    parent = json.loads((panel / "results.json").read_text())
    if parent["status"] != "complete":
        raise ValueError("Need a completed neural panel")
    config = PartialConfig(**parent["protocol"]["config"])
    protocol = begin(
        output,
        config,
        "C3 recurrence/probe checks",
        "Is failure EMA-frame mismatch, autonomous recurrence, or insufficient history?",
    )
    protocol.update(
        {
            "parent_sha256": hashlib.sha256((panel / "results.json").read_bytes()).hexdigest(),
            "controls": [
                "observation-history feedback",
                "train-only affine EMA-to-online feedback",
                "linear-world H1/H2 sufficiency calibration",
                "nonlinear H2 degree-five probe",
            ],
        }
    )
    write_json(output / "protocol.json", protocol)
    records, probe_controls = [], []
    started = perf_counter()
    for seed in config.seeds:
        data, _ = observer_data(config, seed, "POSITIONS")
        for record in [
            r
            for r in parent["records"]
            if r["seed"] == seed and r["frame"] == "native" and r["history"] == 4
        ]:
            frame = (
                torch.eye(6, dtype=torch.float64)
                if record["frame"] == "native"
                else orthogonal_draws(config.rotation_seed, 1)[0]
            )
            model = HistoryJEPA(data.train, 4, record["model"], seed, frame, config)
            model.load_state_dict(
                {
                    key: torch.tensor(value, dtype=torch.float64)
                    for key, value in record["trace"][-1]["state_dict"].items()
                }
            )
            model.eval().requires_grad_(False)
            aligned = AlignedEMAForecast(model, data.train)
            metrics = {
                "history_feedback": evaluate_observer(ReencodedForecast(model), data, 4),
                "ema_aligned": evaluate_observer(aligned, data, 4),
            }
            records.append(
                {
                    "seed": seed,
                    "model": record["model"],
                    "metrics": metrics,
                    "ema_alignment_fit": aligned.fit,
                }
            )
            print(
                seed,
                record["model"],
                [
                    (name, value["interpolation"]["rollout_mse"]["16"])
                    for name, value in metrics.items()
                ],
                flush=True,
            )
        for alpha, history, degree in ((0.0, 1, 3), (0.0, 2, 3), (0.5, 2, 5)):
            observed, _ = observer_data(replace(config, alpha=alpha), seed, "POSITIONS")
            if history == 1:

                def encode(values):
                    return values[:, -1]
            else:
                encode = RecentTwoState(observed.train, config.dt).encode
            probe_controls.append(
                {
                    "seed": seed,
                    "alpha": alpha,
                    "history": history,
                    "degree": degree,
                    "sufficiency": sufficiency_probes(
                        encode, observed, history, seed, config.dt, degree
                    ),
                }
            )
    finish(output, protocol, records, started, probe_controls=probe_controls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ridge", action="store_true")
    mode.add_argument("--diagnose", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--panel", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.ridge:
        ridge_control(args.output)
    else:
        if args.panel is None:
            parser.error("--diagnose requires --panel")
        diagnose(args.output, args.panel)


if __name__ == "__main__":
    main()
