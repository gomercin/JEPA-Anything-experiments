"""Exploratory frozen-isometry controls for the existing linear oscillator task.

Each named panel is small and explicit, not an experiment configuration framework.
All evaluation is in physical [q,p] coordinates, using an exact isometric inverse.
"""

import argparse
import copy
import hashlib
import platform
import statistics
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import torch
from jepa_anything_core.audit import audit_factor_geometry, audit_opf_geometry
from jepa_anything_core.baselines import parameter_count
from jepa_anything_core.opf import OrthogonalFactorProjection
from torch import nn

from .dataset import make_dataset, observation_matrix, pairs
from .evaluate import evaluate, rollout
from .interrogate import orthogonal_draws
from .models import build_models
from .run_experiment import ROOT, Config, fingerprint, write_json
from .simulator import Simulator
from .train import training_loss

CHECKPOINTS = (0, 1, 25, 100, 250, 500, 1000)
ROTATION_SEED = 2718
LEARNING_RATES = {"Adam": 0.003, "SGD": 0.03}


class FrozenIsometry(nn.Module):
    """A fixed bias-free observation-to-latent map, stored as a buffer."""

    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight.detach().clone())

    def forward(self, observations):
        return observations @ self.weight.T


def protected_hashes():
    paths = [
        ROOT / "work/coupled_oscillators" / name for name in ("phase-a-quick", "interrogation")
    ]
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for directory in paths
        for p in sorted(directory.rglob("*"))
        if p.is_file()
    }


def latent_equivalence(physical, frame, observation):
    """C0 hard gate: independently execute both paths, including inverse mapping."""
    raw = FrozenIsometry(frame)(physical)
    mixed = FrozenIsometry(frame @ observation.T)(physical @ observation.T)
    flat_raw, flat_mixed = raw.reshape(-1, 6), mixed.reshape(-1, 6)
    singular = torch.linalg.svdvals(frame)

    def spectrum(values):
        centered = values - values.mean(0)
        return torch.linalg.eigvalsh(centered.T @ centered / len(values))

    difference = float((raw - mixed).abs().max())
    relative = float((raw - mixed).norm() / raw.norm())
    isometry = float((frame.T @ frame - torch.eye(6, dtype=torch.float64)).abs().max())
    inverse = float((raw @ frame - physical).abs().max())
    reference_spectrum = spectrum(physical.reshape(-1, 6))
    raw_spectrum, mixed_spectrum = spectrum(flat_raw), spectrum(flat_mixed)
    spectrum_error = float((reference_spectrum - raw_spectrum).abs().max())
    norm_error = float((raw.norm(dim=-1) - physical.norm(dim=-1)).abs().max())
    passed = max(difference, relative, isometry, inverse, spectrum_error, norm_error) < 1e-12
    passed = passed and float((raw_spectrum - mixed_spectrum).abs().max()) < 1e-12
    return {
        "passed": passed,
        "tolerance": 1e-12,
        "max_absolute_difference": difference,
        "relative_frobenius_difference": relative,
        "encoder_singular_values": singular.tolist(),
        "encoder_condition": float(singular[0] / singular[-1]),
        "isometry_max_error": isometry,
        "inverse_max_error": inverse,
        "physical_covariance_eigenvalues": reference_spectrum.tolist(),
        "raw_latent_covariance_eigenvalues": raw_spectrum.tolist(),
        "mixed_latent_covariance_eigenvalues": mixed_spectrum.tolist(),
        "max_norm_error_vs_physical": norm_error,
        "raw_norm_quantiles": torch.quantile(
            flat_raw.norm(dim=-1), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        ).tolist(),
        "mixed_norm_quantiles": torch.quantile(
            flat_mixed.norm(dim=-1), torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0], dtype=torch.float64)
        ).tolist(),
    }


def make_model(seed, family, frame, observation, config):
    """Match physical functions under z'=R z, keeping hidden coordinates fixed.

    Standard: A1'=A1 R.T, A2'=R A2, b2'=R b2.
    OPF: A1'=A1 R.T, B'=B R.T, factor output heads unchanged.
    """
    models, _ = build_models(seed, config)
    model = models["standard_jepa" if family == "standard" else "opf"]
    model.context_encoder = FrozenIsometry(frame @ observation.T)
    model.target_encoder = copy.deepcopy(model.context_encoder)
    model.target_encoder.requires_grad_(False)
    with torch.no_grad():
        first = model.predictor.trunk[0]
        first.weight.copy_(first.weight @ frame.T)
        if family == "standard":
            output = model.predictor.output
            output.weight.copy_(frame @ output.weight)
            output.bias.copy_(frame @ output.bias)
        else:
            basis = frame.T.reshape(3, 2, 6).clone()
            model.opf = OrthogonalFactorProjection(
                6,
                3,
                2,
                learnable=family != "opf_fixed",
                initial_basis=basis,
                orthogonality_mode="qr_retraction" if family == "opf_qr" else "soft_gram",
            )
    return model


class PhysicalPredictor:
    """Run physical input through the declared sensor path and exact latent inverse."""

    def __init__(self, model, frame, observation):
        self.model, self.frame, self.observation = model, frame, observation

    def __call__(self, physical):
        model = self.model
        encoded = model.context_encoder(physical @ self.observation.T)
        predicted = model.predictor(encoded)
        if hasattr(model, "opf"):
            predicted = model.opf.compose(predicted)
        return predicted @ self.frame


@torch.no_grad()
def checkpoint(model, dataset, frame, observation, config):
    train, test = dataset.physical[dataset.train_ids], dataset.physical[dataset.test_ids]
    context, target = pairs(train @ observation.T)
    loss, components = training_loss(model, context, target, config)
    predict = PhysicalPredictor(model, frame, observation)
    metrics = evaluate(predict, test, test, torch.eye(6, dtype=torch.float64), Simulator(), config)
    latent = model.target_encoder((test @ observation.T).reshape(-1, 6))
    audit = (
        audit_opf_geometry(model.opf, latent).to_dict()
        if hasattr(model, "opf")
        else audit_factor_geometry(latent.reshape(-1, 3, 2)).to_dict()
    )
    # Fixed probes include all test trajectories at t=0 and the pulse origin.
    probes = test[:, [0, config.pulse_time]].reshape(-1, 6)
    signature = rollout(predict, probes, 16)[:, [1, 4, 8, 16]]
    first = model.predictor.trunk[0]
    canonical = {"input_weight": (first.weight @ frame).tolist(), "input_bias": first.bias.tolist()}
    if hasattr(model, "opf"):
        canonical["analysis"] = (model.opf.analysis_basis().reshape(6, 6) @ frame).tolist()
        canonical["output_weight"] = torch.cat([h.weight for h in model.predictor.heads]).tolist()
        canonical["output_bias"] = torch.cat([h.bias for h in model.predictor.heads]).tolist()
    else:
        canonical["output_weight"] = (frame.T @ model.predictor.output.weight).tolist()
        canonical["output_bias"] = (frame.T @ model.predictor.output.bias).tolist()
    encoder = model.context_encoder.weight
    return {
        "metrics": metrics,
        "training_loss": float(loss),
        "loss_components": components,
        "audit": audit,
        "physical_prediction_signature": signature.tolist(),
        "canonical_parameters": canonical,
        "encoder_singular_values": torch.linalg.svdvals(encoder).tolist(),
        "encoder_max_drift": float((encoder - frame @ observation.T).abs().max()),
        "target_encoder_max_difference": float((encoder - model.target_encoder.weight).abs().max()),
        "state_dict": {k: v.tolist() for k, v in model.state_dict().items()},
    }


def train(
    model, dataset, seed, optimizer_name, frame, observation, config, checkpoints=CHECKPOINTS
):
    context, target = pairs(dataset.physical[dataset.train_ids] @ observation.T)
    rng = torch.Generator().manual_seed(seed + 20000)
    optimizer = getattr(torch.optim, optimizer_name)(
        [p for p in model.parameters() if p.requires_grad], lr=LEARNING_RATES[optimizer_name]
    )
    result = [{"step": 0, **checkpoint(model, dataset, frame, observation, config)}]
    for step in range(1, max(checkpoints) + 1):
        indices = torch.randint(len(context), (config.batch_size,), generator=rng)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = training_loss(model, context[indices], target[indices], config)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}")
        loss.backward()
        optimizer.step()
        # The target is an identical frozen map. EMA has no mathematical update
        # and is skipped to avoid unnecessary buffer roundoff or hidden state drift.
        if hasattr(model, "opf"):
            model.opf.after_optimizer_step(step)
        if step in checkpoints:
            result.append({"step": step, **checkpoint(model, dataset, frame, observation, config)})
    return result


def paired_differences(left, right):
    result = []
    for a, b in zip(left, right, strict=True):
        assert a["step"] == b["step"]
        delta = torch.tensor(
            a["physical_prediction_signature"], dtype=torch.float64
        ) - torch.tensor(b["physical_prediction_signature"], dtype=torch.float64)
        parameters = {
            k: float(
                (
                    torch.tensor(a["canonical_parameters"][k], dtype=torch.float64)
                    - torch.tensor(b["canonical_parameters"][k], dtype=torch.float64)
                )
                .abs()
                .max()
            )
            for k in a["canonical_parameters"]
        }
        result.append(
            {
                "step": a["step"],
                "physical_prediction_max_abs_by_horizon": delta.abs().amax((0, 2)).tolist(),
                "physical_prediction_mse_by_horizon": delta.square().mean((0, 2)).tolist(),
                "canonical_parameter_max_difference": parameters,
                "training_loss_difference": a["training_loss"] - b["training_loss"],
                "h16_mse_difference": a["metrics"]["rollout_mse"]["16"]
                - b["metrics"]["rollout_mse"]["16"],
            }
        )
    return result


def human_summary(report):
    lines = [
        f"# Frozen encoder: {report['protocol']['panel']} (exploratory)",
        "",
        "Mean ± sample SD across three paired, exposed seeds; no model selection.",
        "",
        "| Step | Optimizer | Model | Frame / sensor | One-step MSE | h16 MSE | Response h16 |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    groups = sorted(
        {(r["optimizer"], r["model"], r["frame"], r["observation"]) for r in report["records"]}
    )
    for step in (s for s in (250, 1000, 2000, 5000) if s in report["protocol"]["checkpoints"]):
        for optimizer, model, frame, obs in groups:
            group = [
                r
                for r in report["records"]
                if (r["optimizer"], r["model"], r["frame"], r["observation"])
                == (optimizer, model, frame, obs)
            ]
            metrics = [
                next(c for c in r["checkpoints"] if c["step"] == step)["metrics"] for r in group
            ]
            cells = []
            for family in ("one_step_mse", "rollout_mse", "pulse_response_mse"):
                values = [
                    m[family] if family == "one_step_mse" else m[family]["16"] for m in metrics
                ]
                cells.append(f"{statistics.mean(values):.4g} ± {statistics.stdev(values):.3g}")
            lines.append(
                f"| {step} | {optimizer} | {model} | {frame} / {obs} | " + " | ".join(cells) + " |"
            )
    return "\n".join(lines) + "\n"


def run(panel, output, rotation_seed=ROTATION_SEED):
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = perf_counter()
    checkpoints = CHECKPOINTS if panel != "sgd_duration" else (0, 250, 1000, 2000, 5000)
    config = replace(Config(), preset="exploratory_frozen", training_steps=max(checkpoints))
    identity = torch.eye(6, dtype=torch.float64)
    rotation = orthogonal_draws(rotation_seed, 1)[0]
    observation = observation_matrix("MIXED", config.mixing_seed)
    protected = protected_hashes()
    protocol = {
        "label": "exploratory",
        "panel": panel,
        "config": asdict(config),
        "learning_rates": LEARNING_RATES,
        "checkpoints": checkpoints,
        "panel_change": {
            "physical": "Adam standard/OPF; RAW versus MIXED; native latent frame",
            "rotation": "Standard/OPF, SGD/Adam, native versus rotated latent frame",
            "rotation_adam": "Alternate rotation, same seeds; Adam standard/OPF only",
            "sgd_duration": "Native standard/OPF SGD extended to 5000; unchanged lr",
            "gram": (
                "Adam both frames: Gram weight zero, or fixed counter-rotated basis; "
                "the latter keeps output factor coordinates canonical without learned basis entries"
            ),
            "fixed_alternate": "Fixed counter-rotated basis on the alternate rotation; Adam",
            "qr_native": "Frozen native encoder with QR-retracted OPF; Adam; all three seeds",
        }[panel],
        "simulator": asdict(Simulator()),
        "W": identity.tolist(),
        "Q": observation.tolist(),
        "R": rotation.tolist(),
        "rotation_seed": rotation_seed,
        "frozen_target": "Identical frozen copy; core stop-gradient path; no EMA arithmetic",
        "evaluation": (
            "Physical -> sensor -> encoder -> prediction/synthesis -> inverse; no fitted readout"
        ),
        "matched_initialization": (
            "Standard: first @ R.T, R @ output; OPF: first @ R.T, B @ R.T, heads unchanged"
        ),
        "optimizer_defaults": "Adam betas=(.9,.999),eps=1e-8; SGD momentum=0; no weight decay",
        "protected_artifact_sha256": protected,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "threads": 1,
        "dtype": "float64",
        "command": [sys.executable, *sys.argv],
    }
    sources = list(Path(__file__).parent.glob("*.py")) + list(
        (ROOT / "jepa-anything-core/src/jepa_anything_core").glob("*.py")
    )
    protocol["source_sha256"] = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    write_json(output / "protocol.json", protocol)
    # Keep executable bytes alongside each exploratory panel; later diagnostics
    # may legitimately extend this script without obscuring earlier provenance.
    (output / "frozen_encoder.executed.py").write_bytes(Path(__file__).read_bytes())
    report = {
        "label": "exploratory",
        "status": "preflight",
        "protocol": protocol,
        "c0": [],
        "datasets": [],
        "records": [],
        "paired": [],
    }
    datasets = {}
    for seed in config.seeds:
        dataset = make_dataset(Simulator(), seed, 48, 16, 64)
        datasets[seed] = dataset
        report["datasets"].append(
            {
                "seed": seed,
                "sha256": fingerprint(dataset.physical),
                "train_ids": dataset.train_ids.tolist(),
                "test_ids": dataset.test_ids.tolist(),
            }
        )
        for name, frame in (("LATENT_NATIVE", identity), ("LATENT_ROTATED", rotation)):
            report["c0"].append(
                {
                    "seed": seed,
                    "frame": name,
                    **latent_equivalence(dataset.physical, frame, observation),
                }
            )
    write_json(output / "results.json", report)
    if not all(r["passed"] for r in report["c0"]):
        raise RuntimeError("C0 equivalence failed; no training allowed")
    if panel == "physical":
        frames, observations, optimizers, families = (
            ("LATENT_NATIVE",),
            ("RAW", "MIXED"),
            ("Adam",),
            ("standard", "opf"),
        )
    elif panel in ("rotation", "rotation_adam"):
        frames, observations, optimizers, families = (
            ("LATENT_NATIVE", "LATENT_ROTATED"),
            ("RAW",),
            ("SGD", "Adam") if panel == "rotation" else ("Adam",),
            ("standard", "opf"),
        )
    elif panel == "sgd_duration":
        frames, observations, optimizers, families = (
            ("LATENT_NATIVE",),
            ("RAW",),
            ("SGD",),
            ("standard", "opf"),
        )
    elif panel == "gram":
        frames, observations, optimizers, families = (
            ("LATENT_NATIVE", "LATENT_ROTATED"),
            ("RAW",),
            ("Adam",),
            ("opf_no_gram", "opf_fixed"),
        )
    elif panel == "fixed_alternate":
        frames, observations, optimizers, families = (
            ("LATENT_NATIVE", "LATENT_ROTATED"),
            ("RAW",),
            ("Adam",),
            ("opf_fixed",),
        )
    elif panel == "qr_native":
        frames, observations, optimizers, families = (
            ("LATENT_NATIVE",),
            ("RAW",),
            ("Adam",),
            ("opf_qr",),
        )
    else:
        raise ValueError(panel)
    report["status"] = "running"
    for seed, dataset in datasets.items():
        for optimizer in optimizers:
            for family in families:
                variant_config = (
                    replace(config, gram_weight=0.0) if family == "opf_no_gram" else config
                )
                paired = []
                for frame_name in frames:
                    frame = identity if frame_name == "LATENT_NATIVE" else rotation
                    for obs_name in observations:
                        obs = identity if obs_name == "RAW" else observation
                        model = make_model(seed, family, frame, obs, variant_config)
                        snapshots = train(
                            model,
                            dataset,
                            seed,
                            optimizer,
                            frame,
                            obs,
                            variant_config,
                            checkpoints=checkpoints,
                        )
                        report["records"].append(
                            {
                                "seed": seed,
                                "optimizer": optimizer,
                                "model": family,
                                "frame": frame_name,
                                "observation": obs_name,
                                "trainable_parameters": parameter_count(model),
                                "gram_weight": variant_config.gram_weight
                                if family != "standard"
                                else None,
                                "checkpoints": snapshots,
                            }
                        )
                        paired.append(snapshots)
                        write_json(output / "results.json", report)
                if len(paired) == 1:
                    continue
                differences = paired_differences(*paired)
                report["paired"].append(
                    {
                        "seed": seed,
                        "optimizer": optimizer,
                        "model": family,
                        "comparison": "RAW minus MIXED"
                        if panel == "physical"
                        else "NATIVE minus ROTATED",
                        "checkpoints": differences,
                    }
                )
                if (
                    panel == "physical"
                    and max(max(c["physical_prediction_max_abs_by_horizon"]) for c in differences)
                    > 1e-9
                ):
                    write_json(output / "results.json", report)
                    raise RuntimeError("Frozen RAW/MIXED function equivalence failed; investigate")
    report["original_artifacts_unchanged"] = protected == protected_hashes()
    if not report["original_artifacts_unchanged"]:
        raise RuntimeError("Protected artifacts changed")
    report["duration_seconds"] = perf_counter() - started
    report["status"] = "complete"
    write_json(output / "results.json", report)
    summary = human_summary(report)
    (output / "summary.md").write_text(summary)
    (output / "results.sha256").write_text(
        hashlib.sha256((output / "results.json").read_bytes()).hexdigest() + "  results.json\n"
    )
    print(summary)
    print(f"Exploratory output: {output}; {report['duration_seconds']:.2f} seconds")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--panel",
        choices=(
            "physical",
            "rotation",
            "rotation_adam",
            "sgd_duration",
            "gram",
            "fixed_alternate",
            "qr_native",
        ),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rotation-seed", type=int, default=ROTATION_SEED)
    args = parser.parse_args()
    run(args.panel, args.output_dir, args.rotation_seed)
