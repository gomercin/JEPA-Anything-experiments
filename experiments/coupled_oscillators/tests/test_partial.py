"""Information boundaries and operational equivalences, never model superiority."""

import json
from dataclasses import asdict, replace

import pytest
import torch

from experiments.coupled_oscillators.interrogate import orthogonal_draws
from experiments.coupled_oscillators.partial_baselines import finite_float, prediction_metrics
from experiments.coupled_oscillators.partial_data import (
    HistoryNormalizer,
    PartialConfig,
    observe,
    observer_data,
    windows,
)
from experiments.coupled_oscillators.partial_models import (
    HistoryJEPA,
    frozen_encoder,
    train_history_model,
)
from experiments.coupled_oscillators.partial_probes import sufficiency_probes
from experiments.coupled_oscillators.phase_c_controls import AlignedEMAForecast
from experiments.coupled_oscillators.phase_c_partial import snapshot
from experiments.coupled_oscillators.run_experiment import write_json


def small_data():
    config = replace(PartialConfig(), train_count=8, test_count=3, trajectory_steps=40)
    return config, *observer_data(config, 11, "POSITIONS")


def test_observation_firewall_and_split():
    config, data, truth = small_data()
    assert data.fields == ("q1", "q2", "q3")
    assert data.train.shape[-1] == 3
    assert set(data.train_ids).isdisjoint(data.test_ids)
    changed = truth.train.clone()
    changed[..., 3:] = 100000
    assert torch.equal(observe(changed, "POSITIONS"), data.train)
    changed[..., 2] = 50000
    assert torch.equal(observe(changed, "LOCAL"), data.train[..., :2])
    first = HistoryNormalizer(windows(data.train, 4)["history"])
    second = HistoryNormalizer(windows(observe(truth.train, "POSITIONS"), 4)["history"])
    assert torch.equal(first.mean, second.mean)
    assert torch.equal(first.scale, second.scale)
    assert "physical" not in vars(data)
    # Saved configs reproduce allowed trajectories and split without hidden labels.
    reconstructed = PartialConfig(**json.loads(json.dumps(asdict(config))))
    repeated, _ = observer_data(reconstructed, 11, "POSITIONS")
    assert torch.equal(repeated.train, data.train)
    assert repeated.train_ids == data.train_ids


def test_windows_preserve_time_and_trajectory_boundaries():
    values = torch.arange(3 * 40 * 3, dtype=torch.float64).reshape(3, 40, 3)
    batch = windows(values, 4, 16)
    for index in (0, 8, len(batch["history"]) - 1):
        trajectory, time = batch["trajectory"][index], batch["time"][index]
        assert torch.equal(batch["history"][index], values[trajectory, time - 3 : time + 1])
        assert torch.equal(batch["future"][index], values[trajectory, time + 1 : time + 17])
        assert torch.equal(batch["next_history"][index], values[trajectory, time - 2 : time + 2])
    modified = values.clone()
    modified[:, 16:] = -999
    assert torch.equal(windows(modified, 4, 16)["history"][0], batch["history"][0])


def test_unknown_velocity_pulse_not_visible_at_time_zero():
    _, data, _ = small_data()
    for branches in data.pulse_observations.values():
        assert torch.equal(branches["pulse"][:, 0], branches["base"][:, 0])
        assert not torch.equal(branches["pulse"][:, 1:9], branches["base"][:, 1:9])
        assert branches["pulse"].shape == (3, 25, 3)
    assert not any("action" in name for name in vars(data))


@pytest.mark.parametrize("family", ["standard", "opf_fixed", "opf", "output_transform"])
def test_matched_latent_rotation_preserves_initial_function_and_loss(family):
    torch.set_num_threads(1)
    config, data, _ = small_data()
    identity = torch.eye(6, dtype=torch.float64)
    rotation = orthogonal_draws(2718, 1)[0]
    batch = windows(data.train, 4)
    native = HistoryJEPA(data.train, 4, family, 11, identity, config)
    rotated = HistoryJEPA(data.train, 4, family, 11, rotation, config)
    torch.testing.assert_close(rotation.T @ rotation, identity, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(
        native.forecast(batch["history"], 16),
        rotated.forecast(batch["history"], 16),
        atol=1e-13,
        rtol=1e-13,
    )
    a = native.loss(batch["history"], batch["next_history"], batch["future"][:, 0])[0]
    b = rotated.loss(batch["history"], batch["next_history"], batch["future"][:, 0])[0]
    torch.testing.assert_close(a, b, atol=1e-13, rtol=1e-13)
    a.backward()
    assert all(p.grad is None for p in native.jepa.target_encoder.parameters())
    assert native.jepa.context_encoder[-1].weight.grad is not None
    if family == "opf":
        assert native.opf.raw_basis.grad is not None


def test_frozen_probe_and_finite_saved_checkpoint(tmp_path):
    config, data, _ = small_data()
    model = HistoryJEPA(data.train, 4, "opf", 11, torch.eye(6, dtype=torch.float64), config)
    encoder = frozen_encoder(model)
    before = {key: value.clone() for key, value in encoder.state_dict().items()}
    probes = sufficiency_probes(encoder, data, 4, 11)
    assert all(not p.requires_grad and p.grad is None for p in encoder.parameters())
    assert all(torch.equal(value, encoder.state_dict()[key]) for key, value in before.items())
    assert len({r["nominal_parameters"] for r in probes["variants"].values()}) == 1
    report = snapshot(model, data, 0)
    path = tmp_path / "result.json"
    write_json(path, {"checkpoint": report, "probes": probes})
    assert json.loads(path.read_text())["checkpoint"]["metrics"] == report["metrics"]


def test_short_training_replay_and_ema_adapter_boundary():
    torch.set_num_threads(1)
    config, data, _ = small_data()
    config = replace(config, training_steps=3)
    runs = []
    for _ in range(2):
        model = HistoryJEPA(data.train, 4, "opf", 11, torch.eye(6, dtype=torch.float64), config)
        trace = train_history_model(
            model, data.train, 11, config, lambda current, step: snapshot(current, data, step)
        )
        assert trace[-1]["step"] == 3
        json.dumps(trace, allow_nan=False)
        runs.append(model)
    assert all(
        torch.equal(value, runs[1].state_dict()[key]) for key, value in runs[0].state_dict().items()
    )
    history = windows(data.test, 4, 16)["history"]
    adapter = AlignedEMAForecast(runs[0], data.train)
    # The adapter acts only on feedback; it cannot change the first prediction.
    torch.testing.assert_close(adapter.forecast(history, 1), runs[0].forecast(history, 1))


def test_unstable_metrics_are_explicit_nulls_not_clipped(tmp_path):
    prediction = torch.ones(2, 16, 3, dtype=torch.float64)
    prediction[:, 15] = float("inf")
    report = prediction_metrics(prediction, torch.zeros_like(prediction))
    assert report == {"1": 1.0, "4": 1.0, "8": 1.0, "16": None}
    assert finite_float(prediction.norm(dim=-1).max()) is None
    path = tmp_path / "unstable.json"
    write_json(path, report)
    assert json.loads(path.read_text()) == report
