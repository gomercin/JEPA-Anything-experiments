"""Bounded exploratory interrogation of Phase A; never modifies original artifacts.

Run as: python -m experiments.coupled_oscillators.interrogate --output-dir NEW_PATH
"""

import argparse
import copy
import hashlib
import json
import platform
import statistics
import sys
from dataclasses import asdict, replace
from functools import partial
from itertools import permutations
from pathlib import Path
from time import perf_counter

import torch
from jepa_anything_core.baselines import capacity_match_report, parameter_count
from jepa_anything_core.opf import OrthogonalFactorProjection
from torch import nn

from .dataset import make_dataset, observation_matrix, pairs
from .evaluate import diagnostics, evaluate, subspace_alignment
from .models import ObservablePredictor, build_models, fit_affine
from .run_experiment import ROOT, Config, fingerprint, write_json
from .simulator import Simulator
from .train import training_loss

VARIANTS = ("standard", "output_transform", "opf", "opf_no_gram", "opf_qr")
CHECKPOINTS = (0, 25, 100, 250, 500, 1000)
NULL_SEED = 314159
NULL_COUNT = 1024


def file_hashes(directory):
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(directory.rglob("*"))
        if p.is_file() and "__pycache__" not in p.parts
    }


def orthogonal_draws(seed, count):
    rng = torch.Generator().manual_seed(seed)
    q, r = torch.linalg.qr(torch.randn(count, 6, 6, generator=rng, dtype=torch.float64))
    return q * torch.where(r.diagonal(dim1=-2, dim2=-1) < 0, -1.0, 1.0).unsqueeze(-2)


def overlap_scores(rows, modes):
    """Best-permutation overlaps for batches of three full-rank 2D row spans."""
    if (torch.linalg.matrix_rank(rows) != 2).any():
        raise ValueError("Null comparisons require full-rank two-dimensional blocks")
    orthogonal, _ = torch.linalg.qr(rows.transpose(-1, -2))
    overlaps = torch.einsum("...kda,mdb->...kmab", orthogonal, modes).square().sum((-1, -2)) / 2
    scores = torch.stack(
        [sum(overlaps[..., k, p[k]] for k in range(3)) / 3 for p in permutations(range(3))], dim=-1
    )
    return scores.max(dim=-1).values


def null_report(rows, modes, rotations):
    """Common right rotation preserves learned inter-factor geometry, removes orientation."""
    observed = float(overlap_scores(rows, modes))
    values = overlap_scores(rows.unsqueeze(0) @ rotations.unsqueeze(1), modes)
    return {
        "observed": observed,
        "median": float(values.median()),
        "q05_q95": torch.quantile(values, torch.tensor([0.05, 0.95], dtype=values.dtype)).tolist(),
        "fraction_null_at_least_observed": float((values >= observed).double().mean()),
        "scores": values.tolist(),
    }


def physical_rows(model, matrix):
    """Pull observation sensitivities back to x: factors = B W_target M x + bias."""
    basis = (
        model.opf.analysis_basis()
        if hasattr(model, "opf")
        else torch.eye(6, dtype=torch.float64).reshape(3, 2, 6)
    )
    return basis @ model.target_encoder.weight @ matrix


def coordinate_audit(dataset, matrix):
    raw = dataset.physical[dataset.train_ids].reshape(-1, 6)
    mixed = raw @ matrix.T

    def stats(values):
        centered = values - values.mean(0)
        covariance = centered.T @ centered / len(values)
        return {
            "coordinate_variances": values.var(0, correction=0).tolist(),
            "covariance_eigenvalues": torch.linalg.eigvalsh(covariance).tolist(),
            "covariance_condition": float(torch.linalg.cond(covariance)),
            "norm_quantiles": torch.quantile(
                values.norm(dim=1), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
            ).tolist(),
        }

    return {
        "raw": stats(raw),
        "mixed": stats(mixed),
        "max_norm_difference": float((raw.norm(dim=1) - mixed.norm(dim=1)).abs().max()),
    }


def make_variant(seed, name, config):
    models, _ = build_models(seed, config)
    model = models["opf"] if name.startswith("opf") else models["standard_jepa"]
    if name == "opf_no_gram":
        config = replace(config, gram_weight=0.0)
    if name == "opf_qr":
        model.opf = OrthogonalFactorProjection(
            6, 3, 2, learnable=True, orthogonality_mode="qr_retraction"
        ).double()
    if name == "output_transform":
        with torch.random.fork_rng():
            torch.manual_seed(seed + 30000)
            transform = nn.Linear(6, 6, bias=False, dtype=torch.float64)
        with torch.no_grad():
            transform.weight.copy_(torch.eye(6, dtype=torch.float64))
        model.predictor = nn.Sequential(model.predictor, transform)
        assert capacity_match_report(models["opf"], model).matched
    return model, config


@torch.no_grad()
def snapshot(model, train, test, physical_test, matrix, simulator, config):
    context, target = pairs(train)
    train_loss, components = training_loss(model, context, target, config)
    readout, fit = fit_affine(model.target_encoder(target), target)
    predict = ObservablePredictor(readout, model)
    metrics = evaluate(predict, test, physical_test, matrix, simulator, config)
    # Evaluate exactly the same origins as the one-step observable metric.
    n = test.shape[1] - max(config.horizons)
    current, following = test[:, :n].reshape(-1, 6), test[:, 1 : 1 + n].reshape(-1, 6)
    output = model(current, following)
    predicted = (
        model.opf.compose(output.head_predictions) if hasattr(model, "opf") else output.prediction
    )
    error = predicted - output.target
    u, singular, _ = torch.linalg.svd(model.target_encoder.weight)
    contributions = ((error @ u) / singular).square().mean(0) / 6
    return {
        "metrics": metrics,
        "train_loss": float(train_loss),
        "loss_components": components,
        "readout_fit": fit,
        "diagnostics": diagnostics(model, test, matrix, simulator, config),
        "physical_rows": physical_rows(model, matrix).tolist(),
        "target_latent_mse": float(error.square().mean()),
        "target_encoder_condition": float(singular[0] / singular[-1]),
        "observable_mse_by_encoder_singular_direction": contributions.tolist(),
        "observable_error_amplification": metrics["one_step_mse"] / float(error.square().mean()),
        "weakest_direction_fraction": float(contributions[-1] / contributions.sum()),
        "state_dict": {key: tensor.tolist() for key, tensor in model.state_dict().items()},
    }


def fit_with_trace(model, train, seed, config, observe, checkpoints=CHECKPOINTS):
    """Same optimizer/EMA/batch lifecycle as Phase A, with read-only checkpoints."""
    context, target = pairs(train)
    rng = torch.Generator().manual_seed(seed + 20000)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    records = [{"step": 0, **observe()}] if 0 in checkpoints else []
    for step in range(1, max(checkpoints) + 1):
        indices = torch.randint(len(context), (config.batch_size,), generator=rng)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = training_loss(model, context[indices], target[indices], config)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}; retain incomplete run")
        loss.backward()
        optimizer.step()
        model.update_target_encoder()
        if hasattr(model, "opf"):
            model.opf.after_optimizer_step(step)
        if step in checkpoints:
            records.append({"step": step, **observe()})
    return records


def equivariance_probe(seed, name, config, steps=25):
    """Equal physical initialization; compare Adam with rotation-equivariant SGD.

    This checks trajectories of parameters/functions, not tuned SGD performance.
    Both optimizers use the existing lr; no performance comparison between them.
    """
    matrix = observation_matrix("MIXED", config.mixing_seed)
    dataset = make_dataset(Simulator(), seed, 48, 16, 64)
    train, _ = dataset.observed(observation_matrix("RAW"))
    context, target = pairs(train)
    results = {}
    for optimizer_class in (torch.optim.Adam, torch.optim.SGD):
        raw, _ = make_variant(seed, name, config)
        mixed = copy.deepcopy(raw)
        with torch.no_grad():
            for field in ("context_encoder", "target_encoder"):
                encoder = getattr(mixed, field)
                encoder.weight.copy_(encoder.weight @ matrix.T)
        optimizers = [
            optimizer_class(m.parameters(), lr=config.learning_rate) for m in (raw, mixed)
        ]
        rng = torch.Generator().manual_seed(seed + 20000)
        trace = []
        for step in range(steps + 1):
            with torch.no_grad():
                first, second = (
                    raw(context[:128], target[:128]),
                    mixed(context[:128] @ matrix.T, target[:128] @ matrix.T),
                )
                trace.append(
                    {
                        "step": step,
                        "prediction_max_difference": float(
                            (first.prediction - second.prediction).abs().max()
                        ),
                        "physical_encoder_max_difference": float(
                            (raw.context_encoder.weight - mixed.context_encoder.weight @ matrix)
                            .abs()
                            .max()
                        ),
                    }
                )
            if step == steps:
                break
            indices = torch.randint(len(context), (config.batch_size,), generator=rng)
            for model, optimizer, transform in zip(
                (raw, mixed), optimizers, (torch.eye(6, dtype=torch.float64), matrix), strict=True
            ):
                optimizer.zero_grad(set_to_none=True)
                loss, _ = training_loss(
                    model, context[indices] @ transform.T, target[indices] @ transform.T, config
                )
                loss.backward()
                optimizer.step()
                model.update_target_encoder()
        results[optimizer_class.__name__] = trace
    return results


def summary_table(records):
    lines = [
        "# Exploratory Phase A interrogation",
        "",
        "Mean ± sample SD; three paired exposed seeds. No benchmark or selection claim.",
        "",
        "| Step | Condition | Variant | One-step MSE | h16 MSE | Pulse response h16 MSE |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for step in (250, 1000):
        for condition in ("RAW", "MIXED"):
            for name in VARIANTS:
                values = [
                    r for r in records if r["condition"] == condition and r["variant"] == name
                ]
                metrics = [
                    next(c for c in r["checkpoints"] if c["step"] == step)["metrics"]
                    for r in values
                ]
                cells = []
                for family in ("one_step_mse", "rollout_mse", "pulse_response_mse"):
                    numbers = [
                        m[family] if family == "one_step_mse" else m[family]["16"] for m in metrics
                    ]
                    cells.append(
                        f"{statistics.mean(numbers):.4g} ± {statistics.stdev(numbers):.3g}"
                    )
                lines.append(f"| {step} | {condition} | {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    started = perf_counter()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    config, simulator = Config(), Simulator()
    original = ROOT / "work/coupled_oscillators/phase-a-quick"
    originals = file_hashes(original)
    old = json.loads((original / "results.json").read_text())
    matrix = observation_matrix("MIXED", config.mixing_seed)
    protocol = {
        "status": "exploratory; exposed Phase A seeds, no preregistration claim",
        "config": asdict(config),
        "simulator": asdict(simulator),
        "variants": list(VARIANTS),
        "checkpoints": CHECKPOINTS,
        "variant_changes": {
            "output_transform": (
                "Identity 6x6 bias-free output matrix; +36 parameters, standard loss"
            ),
            "opf_no_gram": "Only Gram weight becomes zero; target activity retained",
            "opf_qr": "Core QR retraction after every step; Adam moments unchanged",
        },
        "coordinate_probe": (
            "25 steps, standard and OPF, physically matched initialization; Adam vs SGD, lr=.003"
        ),
        "null": {
            "seed": NULL_SEED,
            "count": NULL_COUNT,
            "kind": "Haar full basis and common right rotations of actual learned row geometry",
        },
        "original_artifact_sha256": originals,
        "source_sha256": {
            **file_hashes(Path(__file__).parent),
            **file_hashes(ROOT / "jepa-anything-core/src/jepa_anything_core"),
        },
        "python": platform.python_version(),
        "torch": torch.__version__,
        "command": [sys.executable, *sys.argv],
        "device": "cpu",
        "dtype": "float64",
    }
    write_json(output / "protocol.json", protocol)
    rotations = orthogonal_draws(NULL_SEED, NULL_COUNT)
    modes = simulator.modes()[1]
    report = {
        "label": "exploratory",
        "status": "running",
        "protocol": protocol,
        "mixing": {
            "matrix": matrix.tolist(),
            "determinant": float(torch.linalg.det(matrix)),
            "rank": int(torch.linalg.matrix_rank(matrix)),
            "singular_values": torch.linalg.svdvals(matrix).tolist(),
            "condition": float(torch.linalg.cond(matrix)),
            "orthogonality_max_error": float((matrix.T @ matrix - torch.eye(6)).abs().max()),
        },
        "mode_eigenvalues": simulator.modes()[0].tolist(),
        "haar_orthogonal_null_scores": overlap_scores(
            rotations.reshape(-1, 3, 2, 6), modes
        ).tolist(),
        "datasets": [],
        "records": [],
        "coordinate_probes": [],
    }
    for seed in config.seeds:
        dataset = make_dataset(
            simulator, seed, config.train_count, config.test_count, config.trajectory_steps
        )
        report["datasets"].append(
            {
                "seed": seed,
                "sha256": fingerprint(dataset.physical),
                "train_ids": dataset.train_ids.tolist(),
                "test_ids": dataset.test_ids.tolist(),
                "coordinate_audit": coordinate_audit(dataset, matrix),
            }
        )
        for condition in config.conditions:
            transform = observation_matrix(condition, config.mixing_seed)
            train, test = dataset.observed(transform)
            for name in VARIANTS:
                model, variant_config = make_variant(seed, name, config)
                checkpoints = fit_with_trace(
                    model,
                    train,
                    seed,
                    variant_config,
                    partial(
                        snapshot,
                        model,
                        train,
                        test,
                        dataset.physical[dataset.test_ids],
                        transform,
                        simulator,
                        variant_config,
                    ),
                )
                for checkpoint in checkpoints:
                    if checkpoint["step"] in (0, 250, 1000):
                        rows = torch.tensor(checkpoint["physical_rows"], dtype=torch.float64)
                        checkpoint["orientation_null"] = null_report(rows, modes, rotations)
                record = {
                    "seed": seed,
                    "condition": condition,
                    "variant": name,
                    "trainable_parameters": parameter_count(model),
                    "readout_coefficients": 42,
                    "checkpoints": checkpoints,
                }
                if name in ("standard", "opf"):
                    previous = next(
                        r
                        for r in old["records"]
                        if r["seed"] == seed
                        and r["condition"] == condition
                        and r["model"] == ("standard_jepa" if name == "standard" else name)
                    )
                    reproduced = next(c for c in checkpoints if c["step"] == 250)
                    record["original_h16_absolute_difference"] = abs(
                        reproduced["metrics"]["rollout_mse"]["16"]
                        - previous["metrics"]["rollout_mse"]["16"]
                    )
                report["records"].append(record)
                write_json(output / "results.json", report)
        for name in ("standard", "opf"):
            report["coordinate_probes"].append(
                {"seed": seed, "variant": name, "trace": equivariance_probe(seed, name, config)}
            )
    report["cross_coordinate_subspaces"] = []
    for seed in config.seeds:
        for name in VARIANTS:
            raw, mixed = [
                next(
                    r
                    for r in report["records"]
                    if r["seed"] == seed and r["variant"] == name and r["condition"] == condition
                )
                for condition in config.conditions
            ]
            for step in (0, 250, 1000):
                a, b = [
                    torch.tensor(
                        next(c for c in record["checkpoints"] if c["step"] == step)[
                            "physical_rows"
                        ],
                        dtype=torch.float64,
                    )
                    for record in (raw, mixed)
                ]
                b_planes = torch.linalg.qr(b.transpose(-1, -2))[0]
                report["cross_coordinate_subspaces"].append(
                    {
                        "seed": seed,
                        "variant": name,
                        "step": step,
                        "alignment": subspace_alignment(a, b_planes),
                    }
                )
    report["duration_seconds"] = perf_counter() - started
    report["original_artifacts_unchanged"] = originals == file_hashes(original)
    if not report["original_artifacts_unchanged"]:
        raise RuntimeError("Original artifact hashes changed")
    report["status"] = "complete"
    write_json(output / "results.json", report)
    (output / "summary.md").write_text(summary_table(report["records"]))
    (output / "results.sha256").write_text(
        hashlib.sha256((output / "results.json").read_bytes()).hexdigest() + "  results.json\n"
    )
    print(summary_table(report["records"]))
    print(f"Exploratory outputs: {output}; elapsed {report['duration_seconds']:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    run(parser.parse_args().output_dir)
