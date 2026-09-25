"""Noisy sensor data and evaluator-only truth; no clean training targets."""

from dataclasses import dataclass
from types import SimpleNamespace

import torch

from .partial_data import ObservedData, PartialConfig, observer_data
from .run_experiment import fingerprint


@dataclass(frozen=True)
class NoiseConfig(PartialConfig):
    noise_ratios: tuple[float, ...] = (0.0, 0.05, 0.15)
    histories: tuple[int, ...] = (2, 4, 8)
    unseen_noise_multiplier: float = 1.5


def learner_config(config):
    """Pass architecture/optimizer settings only, never simulator/noise parameters."""
    names = (
        "dt",
        "d",
        "k",
        "r",
        "hidden_dim",
        "ema_momentum",
        "gram_weight",
        "activity_weight",
        "min_std",
        "training_steps",
        "batch_size",
        "learning_rate",
    )
    return SimpleNamespace(**{name: getattr(config, name) for name in names})


def add_noise(observed, sigma, seed):
    if observed.shape[-1] != 3 or sigma < 0:
        raise ValueError("Noise is defined only on three observed positions")
    generator = torch.Generator().manual_seed(seed)
    return observed + sigma * torch.randn(observed.shape, generator=generator, dtype=observed.dtype)


def noisy_data(config, seed, ratio, test_multiplier=1.0):
    clean, physical = observer_data(config, seed, "POSITIONS")
    # Simulator-side calibration only: this scale never enters a learner.
    signal_std = float(clean.train.std((0, 1), correction=0).square().mean().sqrt())
    sigma = ratio * signal_std
    noise_seed = seed + 60000
    unseen_offset = 1000 if test_multiplier != 1 else 0
    test_sigma = sigma * test_multiplier
    pulses = {}
    for i, split in enumerate(("interpolation", "amplitude_shift")):
        pulses[split] = {
            branch: add_noise(values, test_sigma, noise_seed + unseen_offset + 10 + 2 * i + j)
            for j, (branch, values) in enumerate(clean.pulse_observations[split].items())
        }
    noisy = ObservedData(
        add_noise(clean.train, sigma, noise_seed),
        add_noise(clean.test, test_sigma, noise_seed + unseen_offset + 1),
        add_noise(clean.shifted, test_sigma, noise_seed + unseen_offset + 2),
        pulses,
        clean.train_ids,
        clean.test_ids,
        clean.fields,
    )
    metadata = {
        "ratio": ratio,
        "signal_std_simulator_calibration": signal_std,
        "train_sigma": sigma,
        "test_sigma": test_sigma,
        "test_multiplier": test_multiplier,
        "noise_seed_base": noise_seed,
        "unseen_seed_offset": unseen_offset,
        "branch_noise": "independent base/pulse noise; common standardized draws across ratios",
        "clean_train_hash": fingerprint(clean.train),
        "noisy_train_hash": fingerprint(noisy.train),
        "clean_test_hash": fingerprint(clean.test),
        "noisy_test_hash": fingerprint(noisy.test),
        "train_ids": clean.train_ids,
        "test_ids": clean.test_ids,
        "noise_is_learner_input": False,
    }
    return noisy, SimpleNamespace(observed=clean, physical=physical, metadata=metadata)
