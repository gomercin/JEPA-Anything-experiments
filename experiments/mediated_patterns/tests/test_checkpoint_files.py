"""Checkpoint file safety; no field evolution or scientific outcome assertions."""

import numpy as np
import pytest

from experiments.mediated_patterns.dual_pair import DualHybrid
from experiments.mediated_patterns.hybrid_pair import PairHybrid
from experiments.mediated_patterns.simulator import Config


def serializable_model(cls):
    # Only serialization fields are needed; no solver setup or evolution.
    model = cls.__new__(cls)
    model.c = Config()
    model.basis = np.eye(2)
    model.ab_basis = np.eye(2)
    model.c_basis = np.eye(2)
    model.c_position = 39.0
    model.template = np.zeros(4)
    model.q = np.zeros(4)
    model.m = np.zeros(3, dtype=complex)
    model.time = 2.0
    return model


@pytest.mark.parametrize("cls", [PairHybrid, DualHybrid])
@pytest.mark.parametrize("suffix", ["", ".npz"])
def test_checkpoint_refuses_actual_numpy_destination(tmp_path, cls, suffix):
    requested = tmp_path / ("checkpoint" + suffix)
    actual = tmp_path / "checkpoint.npz"
    model = serializable_model(cls)
    model.save(requested)
    before = actual.read_bytes()
    model.time = 3.0
    with pytest.raises(FileExistsError):
        model.save(requested)
    assert actual.read_bytes() == before


@pytest.mark.parametrize("cls", [PairHybrid, DualHybrid])
def test_checkpoint_refuses_dangling_symlink(tmp_path, cls):
    requested = tmp_path / "checkpoint.npz"
    elsewhere = tmp_path / "elsewhere.npz"
    requested.symlink_to(elsewhere)
    with pytest.raises(FileExistsError):
        serializable_model(cls).save(requested)
    assert requested.is_symlink()
    assert not elsewhere.exists()
