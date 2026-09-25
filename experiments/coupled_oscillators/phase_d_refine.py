"""Targeted conventional output-error identification; no clean supervision."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import torch

from .noise_baselines import polynomial_window_matrix
from .noise_data import NoiseConfig, noisy_data
from .noise_evaluate import evaluate_noise, filter_state_diagnostics, state_probe
from .noise_filters import CubicMechanics, WindowEKF, identify_mechanics
from .phase_d_noise import begin, finish
from .run_experiment import write_json


def refine_mechanics(observations, initial, iterations=80):
    """Multiple shooting: unknown starts and one vector field fit noisy positions.

    Four 16-step blocks per training trajectory. The 6D start for each block is
    a nuisance fit parameter, never a clean simulator label. This is classical
    output-error identification with an explicit second-order mechanical prior.
    """
    starts = (0, 16, 32, 48)
    guesses, targets = [], []
    for start in starts:
        left = max(0, start - 5)
        matrix = polynomial_window_matrix(11, 5, initial.dt, location=start - left)
        coefficients = torch.einsum("kh,nhm->nkm", matrix, observations[:, left : left + 11])
        guesses.append(coefficients[:, :2].flatten(1))
        targets.append(observations[:, start : start + 17])
    state = torch.nn.Parameter(torch.cat(guesses))
    # gelsd can return column-major strides; PyTorch LBFGS flattens with view().
    weights = torch.nn.Parameter(initial.weights.clone().contiguous())
    target = torch.cat(targets)
    scale = observations.std((0, 1), correction=0).clamp_min(1e-8)
    dynamics = CubicMechanics(weights, initial.dt)
    optimizer = torch.optim.LBFGS(
        [weights, state],
        lr=0.5,
        max_iter=iterations,
        max_eval=iterations * 2,
        tolerance_grad=1e-8,
        tolerance_change=1e-11,
        history_size=20,
        line_search_fn="strong_wolfe",
    )
    losses = []

    def closure():
        optimizer.zero_grad(set_to_none=True)
        prediction = torch.cat((state[:, None, :3], dynamics.rollout(state, 16)), 1)
        loss = ((prediction - target) / scale).square().mean()
        loss = loss + 1e-6 * (weights - initial.weights).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite output-error identification trial")
        loss.backward()
        losses.append(float(loss.detach()))
        return loss

    optimizer.step(closure)
    refined = CubicMechanics(weights.detach().clone(), initial.dt)
    refined.noise_variance = initial.noise_variance.clone()
    refined.fit_report = {
        **initial.fit_report,
        "weights": refined.weights.tolist(),
        "refinement": "noisy-output multiple shooting with free training block starts",
        "block_origins": starts,
        "block_steps": 16,
        "nuisance_parameters": state.numel(),
        "optimized_training_starts": state.detach().tolist(),
        "iterations_requested": iterations,
        "iterations_used": optimizer.state[weights]["n_iter"],
        "loss_evaluations": losses,
        "coefficient_regularization": 1e-6,
        "optimizer": "LBFGS lr0.5 strong_wolfe history20",
        "fit_derivatives_only_for_initialization": True,
    }
    return refined


def run(output, parent_path, config, iterations=80):
    parent = json.loads((parent_path / "results.json").read_text())
    if parent["status"] != "complete":
        raise ValueError("Need completed filter panel")
    protocol = begin(
        output, config, "D3: does noisy-output identification close the filter/oracle gap?"
    )
    protocol.update(
        {
            "parent_sha256": hashlib.sha256(
                (parent_path / "results.json").read_bytes()
            ).hexdigest(),
            "fixed_choices": (
                f"moderate noise, H8, {iterations} LBFGS iterations, four 16-step blocks"
            ),
            "information": (
                "all identification inputs/targets noisy; no hidden state or true coefficients"
            ),
        }
    )
    write_json(output / "protocol.json", protocol)
    records = []
    started = perf_counter()
    for seed in config.seeds:
        noisy, truth = noisy_data(config, seed, 0.15)
        initial = identify_mechanics(noisy.train, config.dt)
        fitted = refine_mechanics(noisy.train, initial, iterations)
        model = WindowEKF(fitted)
        metrics = evaluate_noise(model, noisy, truth, 8)
        unseen, unseen_truth = noisy_data(config, seed, 0.15, config.unseen_noise_multiplier)
        record = {
            "seed": seed,
            "noise_ratio": 0.15,
            "history": 8,
            "model": "output_error_ekf",
            "oracle": False,
            "noise": truth.metadata,
            "fit": model.report(),
            "metrics": metrics,
            "state_diagnostics": filter_state_diagnostics(model, noisy, truth, 8),
            "state_probe": state_probe(model.encode, noisy, truth, 8),
            "unseen_noise": {
                "noise": unseen_truth.metadata,
                "metrics": evaluate_noise(model, unseen, unseen_truth, 8),
            },
        }
        records.append(record)
        write_json(output / f"seed-{seed}.json", record)
        print(
            seed,
            "train first/last",
            fitted.fit_report["loss_evaluations"][0],
            fitted.fit_report["loss_evaluations"][-1],
            "clean h16",
            metrics["interpolation"]["clean_truth_mse"]["16"],
            flush=True,
        )
    finish(output, protocol, records, started)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--filters", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=80)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    run(args.output, args.filters, NoiseConfig(), args.iterations)


if __name__ == "__main__":
    main()
