"""Thin experiment adapters around public core APIs; no custom OPF geometry."""

import copy

import torch
from jepa_anything_core.baselines import (
    StandardJEPABaseline,
    UnconstrainedMultiHeadJEPABaseline,
    build_capacity_matched_predictors,
    capacity_match_report,
)
from jepa_anything_core.opf import OrthogonalFactorProjection
from torch import nn


def build_models(seed, config):
    # fork_rng keeps initialization reproducible without changing caller RNG state.
    with torch.random.fork_rng():
        torch.manual_seed(seed)
        encoder = nn.Linear(6, config.d, dtype=torch.float64)
        standard, multi = build_capacity_matched_predictors(
            config.d, config.d, config.k, config.r, config.hidden_dim, depth=2
        )
        standard, multi = standard.double(), multi.double()
        # Head grouping alone is algebraically the same architecture. Match actual
        # initial weights too, rather than attributing initialization noise to it.
        multi.trunk.load_state_dict(standard.trunk.state_dict())
        with torch.no_grad():
            for index, head in enumerate(multi.heads):
                section = slice(index * config.r, (index + 1) * config.r)
                head.weight.copy_(standard.output.weight[section])
                head.bias.copy_(standard.output.bias[section])
        match = capacity_match_report(standard, multi).to_dict()
        models = {
            "standard_jepa": StandardJEPABaseline(
                copy.deepcopy(encoder), standard, target_momentum=config.ema_momentum
            ),
            "unconstrained_multihead": UnconstrainedMultiHeadJEPABaseline(
                copy.deepcopy(encoder),
                copy.deepcopy(multi),
                target_momentum=config.ema_momentum,
            ),
            "opf": UnconstrainedMultiHeadJEPABaseline(
                copy.deepcopy(encoder), multi, target_momentum=config.ema_momentum
            ),
        }
        models["opf"].add_module(
            "opf",
            OrthogonalFactorProjection(
                config.d, config.k, config.r, learnable=True, orthogonality_mode="soft_gram"
            ).double(),
        )
    return models, match


def affine_features(values):
    return torch.cat((values, torch.ones_like(values[..., :1])), dim=-1)


def fit_affine(inputs, targets):
    """Unregularized double-precision SVD least squares, including an intercept."""
    features = affine_features(inputs)
    result = torch.linalg.lstsq(features, targets, driver="gelsd")
    singular = result.singular_values
    return result.solution, {
        "design_rank": int(result.rank),
        "design_singular_values": singular.tolist(),
        "design_condition_number": float(singular[0] / singular[-1]) if singular[-1] > 0 else None,
    }


class ObservablePredictor:
    """Decode predicted target latents, then re-encode observations on each step."""

    def __init__(self, readout, model=None):
        self.readout = readout
        self.model = model

    def __call__(self, observations):
        if self.model is None:
            latent = observations
        else:
            model = self.model
            latent = model.predictor(model.context_encoder(observations))
            if hasattr(model, "opf"):
                latent = model.opf.compose(latent)
            elif latent.ndim == observations.ndim + 1:
                latent = model.predictor.flatten_output(latent)
        return affine_features(latent) @ self.readout
