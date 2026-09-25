"""Post-training probes. Hidden labels enter only the explicitly named diagnostic."""

import torch

from .models import ObservablePredictor, fit_affine
from .nonlinear import polynomial_features
from .partial_data import EXTRA_HISTORY, HORIZONS, HistoryNormalizer, windows


def fit_whitener(train):
    mean = train.mean(0)
    _, singular, vh = torch.linalg.svd(train - mean, full_matrices=False)
    scale = (singular / len(train) ** 0.5).clamp_min(1e-8)
    return mean, vh.T, scale


def whiten(values, parameters):
    mean, basis, scale = parameters
    return ((values - mean) @ basis) / scale


def shuffled_old(old, trajectories, seed):
    """Whole other-trajectory histories at the same times; no same-trajectory future."""
    n = int(trajectories.max()) + 1
    generator = torch.Generator().manual_seed(seed)
    shift = int(torch.randint(1, n, (1,), generator=generator))
    return old.reshape(n, -1, old.shape[-1]).roll(shift, 0).reshape_as(old)


@torch.no_grad()
def sufficiency_probes(encode, data, history, seed, dt=0.1, degree=3):
    """Same-width linear heads on cubic(z) plus real/zero/shuffled older history.

    These are finite readout-class tests, not conditional-independence proofs.
    Positive controls use conventional delay states with known good prediction.
    """
    batches = {
        name: windows(obs, history, max(HORIZONS))
        for name, obs in (
            ("train", data.train),
            ("interpolation", data.test),
            ("amplitude_shift", data.shifted),
        )
    }
    old_windows = {
        name: windows(obs, history + EXTRA_HISTORY, max(HORIZONS))["history"][:, :EXTRA_HISTORY]
        for name, obs in (
            ("train", data.train),
            ("interpolation", data.test),
            ("amplitude_shift", data.shifted),
        )
    }
    z = {name: encode(batch["history"]) for name, batch in batches.items()}
    parameters = fit_whitener(z["train"])
    latent_features = {
        name: polynomial_features(whiten(values, parameters), degree)[..., 1:]
        for name, values in z.items()
    }
    old_normalizer = HistoryNormalizer(old_windows["train"], dt)
    old = {name: old_normalizer(values) for name, values in old_windows.items()}
    targets = {
        name: batch["future"][:, [h - 1 for h in HORIZONS]].flatten(1)
        for name, batch in batches.items()
    }
    records = {}
    for variant in ("z_only", "z_plus_old", "z_plus_shuffled_old"):
        features = {}
        for name in batches:
            supplement = old[name]
            if variant == "z_only":
                supplement = torch.zeros_like(supplement)
            elif variant == "z_plus_shuffled_old":
                supplement = shuffled_old(supplement, batches[name]["trajectory"], seed + 40000)
            features[name] = torch.cat((latent_features[name], supplement), -1)
        weights, fit = fit_affine(features["train"], targets["train"])
        predictor = ObservablePredictor(weights)
        metrics = {}
        for name in ("interpolation", "amplitude_shift"):
            prediction = predictor(features[name]).reshape(-1, len(HORIZONS), data.train.shape[-1])
            target = targets[name].reshape_as(prediction)
            metrics[name] = {
                str(h): float((prediction[:, i] - target[:, i]).square().mean())
                for i, h in enumerate(HORIZONS)
            }
        records[variant] = {"metrics": metrics, "fit": fit, "nominal_parameters": weights.numel()}
    return {
        "probe": "SVD affine head on polynomial whitened z + linear older history",
        "degree": degree,
        "latent_dimension": z["train"].shape[-1],
        "extra_history_samples": EXTRA_HISTORY,
        "target_horizons": HORIZONS,
        "separate_models_per_horizon": False,
        "variants": records,
    }


@torch.no_grad()
def hidden_state_diagnostic(encode, observed, truth, history):
    """Evaluator-only labels, invoked after all main training; never selects models."""
    train_batch = windows(observed.train, history, max(HORIZONS))
    inputs = encode(train_batch["history"])
    labels = truth.train[train_batch["trajectory"], train_batch["time"]]
    weights, fit = fit_affine(inputs, labels)
    probe = ObservablePredictor(weights)
    report = {}
    for name, observations, physical in (
        ("interpolation", observed.test, truth.test),
        ("amplitude_shift", observed.shifted, truth.shifted),
    ):
        batch = windows(observations, history, max(HORIZONS))
        prediction = probe(encode(batch["history"]))
        target = physical[batch["trajectory"], batch["time"]]
        error = (prediction - target).square().mean(0)
        report[name] = {
            "per_coordinate_mse": error.tolist(),
            "position_mse": float(error[:3].mean()),
            "velocity_mse": float(error[3:].mean()),
            "full_state_mse": float(error.mean()),
        }
    return {"evaluator_only": True, "linear_probe_fit": fit, "metrics": report}
