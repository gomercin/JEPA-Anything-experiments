"""Observable-space rollouts, declared pulses, and exploratory subspace geometry."""

from itertools import permutations

import torch
from jepa_anything_core.audit import audit_factor_geometry, audit_opf_geometry


@torch.no_grad()
def rollout(predict, initial, steps):
    states = [initial]
    for _ in range(steps):
        states.append(predict(states[-1]))
    return torch.stack(states, dim=-2)


def mse(left, right):
    return float((left - right).square().mean())


@torch.no_grad()
def evaluate(predict, test, physical_test, matrix, simulator, config):
    maximum = max(config.horizons)
    # Identical starting times at every horizon and for every method.
    origins = test.shape[1] - maximum
    initial = test[:, :origins].reshape(-1, 6)
    predicted = rollout(predict, initial, maximum)
    regular = {
        str(h): mse(predicted[:, h], test[:, h : h + origins].reshape(-1, 6))
        for h in config.horizons
    }
    base = physical_test[:, config.pulse_time].clone()
    pulsed = base.clone()
    pulsed[:, 3 + config.pulse_oscillator] += config.pulse_magnitude
    truth_base = simulator.simulate(base, maximum) @ matrix.T
    truth_pulse = simulator.simulate(pulsed, maximum) @ matrix.T
    predicted_base = rollout(predict, truth_base[:, 0], maximum)
    predicted_pulse = rollout(predict, truth_pulse[:, 0], maximum)
    return {
        "one_step_mse": regular["1"],
        "rollout_mse": regular,
        "pulse_rollout_mse": {
            str(h): mse(predicted_pulse[:, h], truth_pulse[:, h]) for h in config.horizons
        },
        "pulse_response_mse": {
            str(h): mse(
                predicted_pulse[:, h] - predicted_base[:, h], truth_pulse[:, h] - truth_base[:, h]
            )
            for h in config.horizons
        },
    }


def subspace_alignment(effective_rows, mode_planes):
    """Compare physical-coordinate analysis row spans, respecting rank loss.

    Both sides live in the declared Euclidean metric on [q, p]. We compare
    individual 2D spans, not the vacuously identical complete 6D spans.
    """
    overlaps, angles, ranks = [], [], []
    for rows in effective_rows:
        _, singular, vh = torch.linalg.svd(rows, full_matrices=False)
        rank = int((singular > singular[0] * max(rows.shape) * torch.finfo(rows.dtype).eps).sum())
        ranks.append(rank)
        learned = vh[:rank].T
        overlaps.append([float((learned.T @ mode).square().sum() / 2) for mode in mode_planes])
        angles.append(
            [
                torch.rad2deg(
                    torch.acos(torch.linalg.svdvals(learned.T @ mode).clamp(0, 1))
                ).tolist()
                if rank == 2
                else None
                for mode in mode_planes
            ]
        )
    assignments = list(permutations(range(3)))
    best = max(assignments, key=lambda p: sum(overlaps[k][p[k]] for k in range(3)))
    return {
        "physical_analysis_rows": effective_rows.tolist(),
        "factor_ranks": ranks,
        "projection_overlap_matrix": overlaps,
        "principal_angles_degrees_matrix": angles,
        "best_mode_permutation": list(best),
        "best_permutation_mean_overlap": sum(overlaps[k][best[k]] for k in range(3)) / 3,
        "interpretation": "Exploratory subspace alignment, no semantic identification or threshold",
    }


@torch.no_grad()
def diagnostics(model, test, matrix, simulator, config):
    observed = test.reshape(-1, 6)
    latent = model.target_encoder(observed)
    online = model.context_encoder(observed)
    report = {
        "split": "test; reporting only, never used for selection",
        "target_encoder_singular_values": torch.linalg.svdvals(
            model.target_encoder.weight
        ).tolist(),
        "online_encoder_coordinate_std": online.std(dim=0, correction=0).tolist(),
    }
    if hasattr(model, "opf"):
        report["opf_geometry"] = audit_opf_geometry(
            model.opf,
            latent,
            min_standard_deviation=config.min_std,
            max_cross_factor_correlation=config.max_correlation,
        ).to_dict()
        # z_target = W_target M x + b; factor-coordinate sensitivity = B W_target M.
        # Bias does not affect a subspace. No decoder or test-label fit is needed.
        rows = model.opf.analysis_basis() @ model.target_encoder.weight @ matrix
        report["mode_alignment"] = subspace_alignment(rows, simulator.modes()[1])
    else:
        report["anonymous_coordinate_groups"] = audit_factor_geometry(
            latent.reshape(-1, config.k, config.r),
            min_standard_deviation=config.min_std,
            max_cross_factor_correlation=config.max_correlation,
        ).to_dict()
    return report
