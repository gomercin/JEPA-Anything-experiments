"""Fixed-step latent prediction, followed by a frozen train-only linear readout."""

from time import perf_counter

import torch
from jepa_anything_core.losses import encoder_variance_loss, jepa_anything_objective

from .dataset import pairs
from .models import ObservablePredictor, affine_features, fit_affine


def training_loss(model, context, target, config):
    output = model(context, target)
    if hasattr(model, "opf"):
        # The core baseline already stops encoder-target gradients. Project AFTER
        # that boundary: target-side projector gradients must remain live.
        objective = jepa_anything_objective(
            output.head_predictions,
            model.opf(output.target),
            model.opf.analysis_basis(),
            output.context_state,
            orthogonality_weight=config.gram_weight,
            factor_activity_weight=config.activity_weight,
            encoder_variance_weight=config.activity_weight,
            factor_min_std=config.min_std,
            encoder_min_std=config.min_std,
        )
        return objective.total, {
            name: float(getattr(objective, name).detach())
            for name in ("prediction", "orthogonality", "factor_activity", "encoder_variance")
        }
    prediction = (output.prediction - output.target).square().mean()
    variance = encoder_variance_loss(output.context_state, min_std=config.min_std)
    return prediction + config.activity_weight * variance, {
        "prediction": float(prediction.detach()),
        "encoder_variance": float(variance.detach()),
    }


def train_neural(model, train, seed, config):
    started = perf_counter()
    context, target = pairs(train)
    rng = torch.Generator().manual_seed(seed + 20000)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.train()
    for step in range(config.training_steps):
        indices = torch.randint(len(context), (config.batch_size,), generator=rng)
        optimizer.zero_grad(set_to_none=True)
        loss, components = training_loss(model, context[indices], target[indices], config)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite training loss at step {step}")
        loss.backward()
        optimizer.step()
        model.update_target_encoder()
        if hasattr(model, "opf"):
            model.opf.after_optimizer_step(step + 1)
    model.eval()
    with torch.no_grad():
        latent = model.target_encoder(target)
        readout, fit_report = fit_affine(latent, target)
        reconstruction = affine_features(latent) @ readout
    return ObservablePredictor(readout, model), {
        "duration_seconds": perf_counter() - started,
        "last_batch_loss": float(loss.detach()),
        "last_batch_components": components,
        "readout_fit": fit_report,
        "readout_train_reconstruction_mse": float((reconstruction - target).square().mean()),
    }
