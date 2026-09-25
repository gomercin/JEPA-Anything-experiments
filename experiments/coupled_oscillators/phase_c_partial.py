"""Bounded Phase C partial-observation study; explicit stages, immutable output paths."""

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

import torch
from jepa_anything_core.baselines import parameter_count

from .interrogate import orthogonal_draws
from .partial_baselines import DelayRegression, RecentTwoState, evaluate_observer
from .partial_data import (
    EXTRA_HISTORY,
    HORIZONS,
    OBSERVATIONS,
    ORIGIN_START,
    PULSE_ASSIMILATION,
    PULSE_TIME,
    PartialConfig,
    linear_observability,
    observer_data,
    windows,
)
from .partial_models import HistoryJEPA, frozen_encoder, latent_diagnostics, train_history_model
from .partial_probes import hidden_state_diagnostic, sufficiency_probes
from .run_experiment import ROOT, fingerprint, write_json


def preserved_hashes():
    return {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for name in ("phase-a-quick", "interrogation", "frozen_encoder", "phase_b")
        for p in sorted((ROOT / "work/coupled_oscillators" / name).rglob("*"))
        if p.is_file()
    }


def begin(output, config, stage, question):
    output.mkdir(parents=True, exist_ok=False)
    protected = preserved_hashes()
    sources = sorted(Path(__file__).parent.glob("*.py"))
    protocol = {
        "exploratory": True,
        "stage": stage,
        "question": question,
        "config": asdict(config),
        "observations": OBSERVATIONS,
        "common_origin_start": ORIGIN_START,
        "horizons": HORIZONS,
        "extra_history": EXTRA_HISTORY,
        "pulse": {
            "contract": "unknown, no action input",
            "time": PULSE_TIME,
            "velocity_index": 0,
            "magnitude": config.pulse_magnitude,
            "observed_steps_before_forecast": PULSE_ASSIMILATION,
        },
        "information_firewall": (
            "Main fit functions receive only observation tensors; hidden truth is evaluator-only"
        ),
        "command": sys.argv,
        "protected_hashes_before": protected,
        "source_hashes": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        },
        "torch_version": torch.__version__,
    }
    write_json(output / "protocol.json", protocol)
    for path in sources:
        if path.name.startswith(("partial_", "phase_c_")):
            (output / path.name).write_bytes(path.read_bytes())
    return protocol


def finish(output, protocol, records, started, **extra):
    assert preserved_hashes() == protocol["protected_hashes_before"], "Earlier evidence changed"
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
    protocol = begin(output, config, "C0", "Observability/history length before neural training")
    started = perf_counter()
    records = []
    observability = {
        regime: {str(h): linear_observability(regime, h, config.dt) for h in config.histories}
        for regime in ("FULL", "POSITIONS", "LOCAL")
    }
    for alpha in (0.0, config.alpha):
        for seed in config.seeds:
            for regime in ("POSITIONS", "LOCAL"):
                data, _ = observer_data(replace(config, alpha=alpha), seed, regime)
                for history in config.histories:
                    for family in ("linear_ar", "cubic_delay", "svd_cubic_state"):
                        # Linear calibration needs linear AR only, not redundant polynomial fits.
                        if alpha == 0 and family != "linear_ar":
                            continue
                        model = DelayRegression(data.train, history, family, config.dt)
                        record = {
                            "alpha": alpha,
                            "seed": seed,
                            "observation": regime,
                            "history": history,
                            "model": family,
                            "fields": data.fields,
                            "train_ids": data.train_ids,
                            "test_ids": data.test_ids,
                            "observations_hash": fingerprint(data.train),
                            "fit": model.report(),
                            "metrics": evaluate_observer(model, data, history),
                        }
                        records.append(record)
                        print(
                            alpha,
                            seed,
                            regime,
                            history,
                            family,
                            "h16",
                            record["metrics"]["interpolation"]["rollout_mse"]["16"],
                            flush=True,
                        )
            write_json(
                output / "results.json",
                {"status": "running", "protocol": protocol, "records": records},
            )
    finish(output, protocol, records, started, linear_observability=observability)


def compression_control(output, config):
    protocol = begin(
        output,
        config,
        "C0 compression control",
        "Does dropping delay directions explain longer-history instability?",
    )
    started = perf_counter()
    records = []
    for seed in config.seeds:
        for regime in ("POSITIONS", "LOCAL"):
            data, _ = observer_data(config, seed, regime)
            model = DelayRegression(data.train, 4, "cubic_full", config.dt)
            metrics = evaluate_observer(model, data, 4)
            records.append(
                {
                    "seed": seed,
                    "observation": regime,
                    "history": 4,
                    "model": "cubic_full",
                    "fit": model.report(),
                    "metrics": metrics,
                }
            )
            print(
                seed, regime, "full cubic H4", metrics["interpolation"]["rollout_mse"], flush=True
            )
            if regime == "POSITIONS":
                model = RecentTwoState(data.train, config.dt)
                for history in (4, 8):
                    metrics = evaluate_observer(model, data, history)
                    records.append(
                        {
                            "seed": seed,
                            "observation": regime,
                            "history": history,
                            "model": "recent_two_state",
                            "fit": model.model.report(),
                            "metrics": metrics,
                        }
                    )
    finish(output, protocol, records, started)


@torch.no_grad()
def snapshot(model, data, step):
    batch = windows(data.train, model.history)
    loss, components = model.loss(batch["history"], batch["next_history"], batch["future"][:, 0])
    test = windows(data.test, model.history, max(HORIZONS))
    report = {
        "step": step,
        "training_loss": float(loss),
        "loss_components": components,
        "metrics": evaluate_observer(model, data, model.history),
        "diagnostics": latent_diagnostics(model, test["history"]),
        "physical_prediction_signature": model.forecast(test["history"][::34], 16)[
            :, [h - 1 for h in HORIZONS]
        ].tolist(),
        "state_dict": {key: value.tolist() for key, value in model.state_dict().items()},
    }
    return report


def neural_panel(output, config, qualification, duration=False):
    qualified = json.loads((qualification / "results.json").read_text())
    if qualified["status"] != "complete":
        raise ValueError("Need completed observation qualification")
    protocol = begin(
        output,
        config,
        "C1 neural" if not duration else "C2 duration",
        "Can learned six-dimensional states replace position history beyond delay controls?",
    )
    protocol.update(
        {
            "qualified_results_sha256": hashlib.sha256(
                (qualification / "results.json").read_bytes()
            ).hexdigest(),
            "chosen_setting": "POSITIONS H4; H1 standard is an information-limited control",
            "choice_basis": "C0 observable prediction supports H2; H4 tests 12-to-6 compression",
            "objective": "observable next-step MSE plus core JEPA latent/Gram/activity objective",
            "primary_rollout": "autonomous latent recurrence, no history re-encoding after origin",
            "variance_frame": "canonical latent coordinates for invariant auxiliary activity",
            "probe_contract": (
                "frozen encoders; cubic(z) plus equal-width zero/real/shuffled old history"
            ),
            "hidden_probes": "post-training evaluator only, never read for model selection",
        }
    )
    identity = torch.eye(config.d, dtype=torch.float64)
    rotation = orthogonal_draws(config.rotation_seed, 1)[0]
    protocol["latent_rotation"] = rotation.tolist()
    write_json(output / "protocol.json", protocol)
    cases = [
        (4, family, name, frame)
        for family in ("standard", "opf_fixed", "opf", "output_transform")
        for name, frame in (("native", identity), ("rotated", rotation))
        if not duration or name == "native"
    ]
    if not duration:
        cases += [(1, "standard", "native", identity)]
    records, paired = [], []
    started = perf_counter()
    # Hidden-state labels are deliberately not retained while main models train.
    for seed in config.seeds:
        data, _ = observer_data(config, seed, "POSITIONS")
        for history, family, frame_name, frame in cases:
            model = HistoryJEPA(data.train, history, family, seed, frame, config)

            def callback(current, step, allowed=data):
                return snapshot(current, allowed, step)

            trace = train_history_model(model, data.train, seed, config, callback)
            encoder = frozen_encoder(model)

            def encode(values, frozen=encoder, fixed_frame=frame):
                return frozen(values) @ fixed_frame

            probes = sufficiency_probes(encode, data, history, seed, config.dt)
            record = {
                "seed": seed,
                "observation": "POSITIONS",
                "fields": data.fields,
                "history": history,
                "model": family,
                "frame": frame_name,
                "train_ids": data.train_ids,
                "test_ids": data.test_ids,
                "training_observation_hash": fingerprint(data.train),
                "trainable_parameters": parameter_count(model),
                "predictor_capacity_match": model.predictor_capacity_match,
                "trace": trace,
                "sufficiency": probes,
            }
            records.append(record)
            write_json(output / f"{seed}-H{history}-{family}-{frame_name}.json", record)
            print(
                seed,
                history,
                family,
                frame_name,
                "h16",
                trace[-1]["metrics"]["interpolation"]["rollout_mse"]["16"],
                flush=True,
            )
        write_json(
            output / "results.json", {"status": "running", "protocol": protocol, "records": records}
        )
    # All main fits finish before any hidden-label probe. These probes cannot feed
    # a training callback, normalizer, optimizer, target, or checkpoint selection.
    controls = []
    for seed in config.seeds:
        data, truth = observer_data(config, seed, "POSITIONS")
        control = RecentTwoState(data.train, config.dt)
        controls.append(
            {
                "seed": seed,
                "model": "recent_two_state",
                "history": 4,
                "sufficiency": sufficiency_probes(control.encode, data, 4, seed, config.dt),
                "hidden_state_diagnostic": hidden_state_diagnostic(control.encode, data, truth, 4),
            }
        )
        for record in [r for r in records if r["seed"] == seed]:
            frame = identity if record["frame"] == "native" else rotation
            model = HistoryJEPA(data.train, record["history"], record["model"], seed, frame, config)
            model.load_state_dict(
                {
                    key: torch.tensor(value, dtype=torch.float64)
                    for key, value in record["trace"][-1]["state_dict"].items()
                }
            )
            encoder = frozen_encoder(model)

            def encode(values, frozen=encoder, fixed_frame=frame):
                return frozen(values) @ fixed_frame

            record["hidden_state_diagnostic"] = hidden_state_diagnostic(
                encode, data, truth, record["history"]
            )
    if not duration:
        for native in [r for r in records if r["frame"] == "native" and r["history"] == 4]:
            rotated = next(
                r
                for r in records
                if r["seed"] == native["seed"]
                and r["model"] == native["model"]
                and r["frame"] == "rotated"
            )
            differences = []
            for a, b in zip(native["trace"], rotated["trace"], strict=True):
                delta = torch.tensor(
                    a["physical_prediction_signature"], dtype=torch.float64
                ) - torch.tensor(b["physical_prediction_signature"], dtype=torch.float64)
                differences.append(
                    {
                        "step": a["step"],
                        "mse_by_horizon": delta.square().mean((0, 2)).tolist(),
                        "max_abs": float(delta.abs().max()),
                        "loss_difference": a["training_loss"] - b["training_loss"],
                    }
                )
            if differences[0]["max_abs"] > 1e-12 or abs(differences[0]["loss_difference"]) > 1e-12:
                raise RuntimeError("Coordinate-equivalent initialization failed")
            paired.append({"seed": native["seed"], "model": native["model"], "trace": differences})
    finish(
        output,
        protocol,
        records,
        started,
        paired_frames=paired,
        conventional_probe_controls=controls,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--qualify", action="store_true")
    mode.add_argument("--compression-control", action="store_true")
    mode.add_argument("--neural", action="store_true")
    mode.add_argument("--duration", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--qualification", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if args.qualify:
        qualify(args.output, PartialConfig())
    elif args.compression_control:
        compression_control(args.output, PartialConfig())
    else:
        if args.qualification is None:
            parser.error("Neural panels require --qualification")
        config = PartialConfig(training_steps=3000 if args.duration else 1000)
        neural_panel(args.output, config, args.qualification, duration=args.duration)


if __name__ == "__main__":
    main()
