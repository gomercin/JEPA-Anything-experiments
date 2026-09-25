"""History JEPA adapters: observation-only supervision and autonomous latent rollout."""

import copy

import torch
from jepa_anything_core.audit import audit_factor_geometry, audit_opf_geometry
from jepa_anything_core.baselines import (
    StandardJEPABaseline,
    UnconstrainedMultiHeadJEPABaseline,
    build_capacity_matched_predictors,
    capacity_match_report,
)
from jepa_anything_core.losses import encoder_variance_loss, jepa_anything_objective
from jepa_anything_core.opf import OrthogonalFactorProjection
from torch import nn

from .partial_data import HistoryNormalizer, windows


class HistoryJEPA(nn.Module):
    def __init__(self, observations, history, family, seed, frame, config):
        super().__init__()
        if observations.ndim != 3 or frame.shape != (config.d, config.d):
            raise ValueError("Expected observation tensor and matching latent frame")
        self.history, self.family, self.config = history, family, config
        self.register_buffer("frame", frame.clone())
        batch = windows(observations, history)
        normalizer = HistoryNormalizer(batch["history"], config.dt)
        width = observations.shape[-1]
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            encoder = nn.Sequential(
                normalizer,
                nn.Linear(history * width, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, config.d),
            ).double()
            standard, multi = build_capacity_matched_predictors(
                config.d, config.d, config.k, config.r, config.hidden_dim, depth=2
            )
            standard, multi = standard.double(), multi.double()
            multi.trunk.load_state_dict(standard.trunk.state_dict())
            with torch.no_grad():
                for i, head in enumerate(multi.heads):
                    section = slice(i * config.r, (i + 1) * config.r)
                    head.weight.copy_(standard.output.weight[section])
                    head.bias.copy_(standard.output.bias[section])
            self.decoder = nn.Linear(config.d, width, dtype=torch.float64)
            self.predictor_capacity_match = capacity_match_report(standard, multi).to_dict()
            predictor = standard if family in ("standard", "output_transform") else multi
            with torch.no_grad():
                encoder[-1].weight.copy_(frame @ encoder[-1].weight)
                encoder[-1].bias.copy_(frame @ encoder[-1].bias)
                predictor.trunk[0].weight.copy_(predictor.trunk[0].weight @ frame.T)
                self.decoder.weight.copy_(self.decoder.weight @ frame.T)
                if predictor is standard:
                    predictor.output.weight.copy_(frame @ predictor.output.weight)
                    predictor.output.bias.copy_(frame @ predictor.output.bias)
            if family == "output_transform":
                transform = nn.Linear(config.d, config.d, bias=False, dtype=torch.float64)
                with torch.no_grad():
                    transform.weight.copy_(torch.eye(config.d, dtype=torch.float64))
                predictor = nn.Sequential(predictor, transform)
            wrapper = (
                StandardJEPABaseline
                if predictor is not multi
                else UnconstrainedMultiHeadJEPABaseline
            )
            self.jepa = wrapper(encoder, predictor, target_momentum=config.ema_momentum)
            if family in ("opf", "opf_fixed"):
                self.opf = OrthogonalFactorProjection(
                    config.d,
                    config.k,
                    config.r,
                    learnable=family == "opf",
                    initial_basis=frame.T.reshape(config.k, config.r, config.d).clone(),
                    orthogonality_mode="soft_gram",
                )
        self.register_buffer("observation_mean", observations.mean((0, 1)))
        self.register_buffer(
            "observation_scale", observations.std((0, 1), correction=0).clamp_min(1e-8)
        )

    def encode(self, history):
        return self.jepa.context_encoder(history)

    def predict_latent(self, state):
        prediction = self.jepa.predictor(state)
        return self.opf.compose(prediction) if hasattr(self, "opf") else prediction

    def decode(self, state):
        return self.decoder(state) * self.observation_scale + self.observation_mean

    def forecast(self, history, steps, history_feedback=False):
        state = self.encode(history)
        result = []
        for _ in range(steps):
            state = self.predict_latent(state)
            prediction = self.decode(state)
            result.append(prediction)
            if history_feedback:
                history = torch.cat((history[:, 1:], prediction[:, None]), 1)
                state = self.encode(history)
        return torch.stack(result, 1)

    def loss(self, context, next_context, target_observation):
        output = self.jepa(context, next_context)
        # Canonical activity makes this auxiliary term invariant under the
        # declared latent coordinate control, rather than adding another confound.
        canonical_context = output.context_state @ self.frame
        if hasattr(self, "opf"):
            objective = jepa_anything_objective(
                output.head_predictions,
                self.opf(output.target),
                self.opf.analysis_basis(),
                canonical_context,
                orthogonality_weight=self.config.gram_weight,
                factor_activity_weight=self.config.activity_weight,
                encoder_variance_weight=self.config.activity_weight,
                factor_min_std=self.config.min_std,
                encoder_min_std=self.config.min_std,
            )
            latent_loss = objective.total
            prediction = self.opf.compose(output.head_predictions)
            components = {
                name: float(getattr(objective, name).detach())
                for name in ("prediction", "orthogonality", "factor_activity", "encoder_variance")
            }
        else:
            prediction = output.prediction
            latent_prediction = (prediction - output.target).square().mean()
            activity = encoder_variance_loss(canonical_context, min_std=self.config.min_std)
            latent_loss = latent_prediction + self.config.activity_weight * activity
            components = {
                "prediction": float(latent_prediction.detach()),
                "encoder_variance": float(activity.detach()),
            }
        target = (target_observation - self.observation_mean) / self.observation_scale
        observation_loss = (self.decoder(prediction) - target).square().mean()
        components["observable_prediction"] = float(observation_loss.detach())
        return latent_loss + observation_loss, components


def train_history_model(model, observations, seed, config, callback):
    """Training boundary: only observed trajectories enter this function."""
    batch = windows(observations, model.history)
    generator = torch.Generator().manual_seed(seed + 20000)
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=config.learning_rate
    )
    checkpoints = sorted(
        {
            0,
            min(250, config.training_steps),
            min(1000, config.training_steps),
            config.training_steps,
        }
    )
    trace = [callback(model, 0)]
    model.train()
    for step in range(1, config.training_steps + 1):
        indices = torch.randint(len(batch["history"]), (config.batch_size,), generator=generator)
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.loss(
            batch["history"][indices], batch["next_history"][indices], batch["future"][indices, 0]
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"Nonfinite training loss at {step}")
        loss.backward()
        optimizer.step()
        model.jepa.update_target_encoder()
        if hasattr(model, "opf"):
            model.opf.after_optimizer_step(step)
        if step in checkpoints:
            trace.append(callback(model, step))
    model.eval()
    return trace


def spectrum(matrix):
    singular = torch.linalg.svdvals(matrix)
    rank = int(torch.linalg.matrix_rank(matrix))
    return {
        "singular_values": singular.tolist(),
        "rank": rank,
        "condition_number": float(singular[0] / singular[-1])
        if rank == min(matrix.shape)
        else None,
    }


@torch.no_grad()
def latent_diagnostics(model, context):
    latent = model.encode(context)
    canonical = latent @ model.frame
    centered = canonical - canonical.mean(0)
    report = {
        "latent_covariance": spectrum(centered.T @ centered / len(centered)),
        "latent_std": canonical.std(0, correction=0).tolist(),
        "encoder_final_linear": spectrum(model.jepa.context_encoder[-1].weight),
        "decoder": spectrum(model.decoder.weight),
        "target_online_latent_mse": float(
            (model.jepa.target_encoder(context) - latent).square().mean()
        ),
        "audit": audit_opf_geometry(model.opf, latent).to_dict()
        if hasattr(model, "opf")
        else audit_factor_geometry(canonical.reshape(-1, 3, 2)).to_dict(),
    }
    with torch.enable_grad():
        jacobians = torch.stack(
            [torch.autograd.functional.jacobian(model.predict_latent, z) for z in latent[:8]]
        )
    report["latent_transition_jacobian_radii"] = (
        torch.linalg.eigvals(jacobians).abs().amax(-1).tolist()
    )
    return report


def frozen_encoder(model):
    encoder = copy.deepcopy(model.jepa.context_encoder).eval().requires_grad_(False)
    return encoder
