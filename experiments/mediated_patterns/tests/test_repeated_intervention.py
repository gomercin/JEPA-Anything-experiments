"""Instrument and information-boundary tests; no scientific superiority assertions."""

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.mediated_patterns import intervention_model as im
from experiments.mediated_patterns import repeated_intervention_model as rm
from experiments.mediated_patterns.hybrid_pair import save_npz_exclusive
from experiments.mediated_patterns.measurements import regions
from experiments.mediated_patterns.repeated_intervention_state import (
    DEV,
    FRESH,
    MODEL_FILE,
    eight_responses,
    model,
)
from experiments.mediated_patterns.simulator import Field
from experiments.mediated_patterns.source_receiver_relay import CFG, source_profile


def test_additive_source_and_ownership():
    f = Field(CFG)
    q = source_profile(f)
    assert np.isclose(np.sum(q * q) * CFG.length / CFG.n, 1)
    assert not np.any(q[regions(f.x, [-4, 28]).sum(axis=0) > 0])
    rng = np.random.default_rng(1)
    state = rng.normal(size=(2, CFG.n))
    after = state.copy()
    after[0] += 0.02 * q
    after[0] -= 0.02 * q
    np.testing.assert_allclose(after, state, rtol=0, atol=5e-16)
    np.testing.assert_array_equal(after[1], state[1])


@pytest.mark.parametrize("cross", [0.0, 0.7])
def test_eight_branches_linear_and_cross(cross):
    a = np.zeros((4, 161, 2, 3, 3))
    response = []
    for j, (i, k) in enumerate(rm.PREFIXES):
        baseline = 100 + 5 * i - 7 * k
        y = 3 + 2 * i - 4 * k + cross * i * k
        a[j, :, 0, 2, :2] = baseline
        a[j, :, 1, 2, :2] = baseline + y
        response.append(y)
    r = eight_responses(a)
    np.testing.assert_allclose(r[:, 0, 0], response)
    np.testing.assert_allclose(rm.contrasts(r)["K12"], cross, atol=1e-14)
    pred = r * 0.97
    k = rm.contrasts(r)["K12"]
    true_add, pred_add = r[1] + r[2] - r[0], pred[1] + pred[2] - pred[0]
    np.testing.assert_allclose(
        r[3] - pred[3], k + true_add - pred_add + pred_add - pred[3], atol=1e-14
    )


def test_continuous_state_causal_slots_and_zero_identity():
    m, z = model(), np.array([0.001, 0.01, -0.002])
    tr, y = rm.forecast(m, z, [10, 30], [0.01, 0.01])
    alternative, _ = rm.forecast(m, z, [10, 30], [0.01, -0.02])
    np.testing.assert_array_equal(tr[:30], alternative[:30])
    np.testing.assert_array_equal(tr[:10, 3:], 0.0)
    assert tr[10, 7] == 0.5 and tr[30, 7] == 1.0
    assert tr[30, 8] == pytest.approx(0.5 * np.exp(-1) + 0.5)
    state = im.State(m, z)
    state.advance(10)
    state.event(0.01)
    state.advance(7)
    state.advance(13)
    before = state.memory.copy()
    state.event(0.01)
    assert not np.array_equal(before, np.zeros(4))
    state.advance(10)
    np.testing.assert_array_equal(state.response(), y)
    zero, _ = rm.forecast(m, z, [10, 30], [0.0, 0.0])
    state = im.State(m, z)
    state.advance(40)
    np.testing.assert_array_equal(zero[-1], rm.vector(state))
    with pytest.raises(ValueError):
        rm.forecast(m, z, [10, 10], [0.01, 0.01])


def test_exposure_groups_and_frozen_identity():
    assert not set(DEV) & set(FRESH)
    assert len(FRESH) >= 3
    assert len(model()["transient"]["A"]) == 4
    assert MODEL_FILE.is_file()


def test_safe_output(tmp_path):
    path = tmp_path / "saved.npz"
    save_npz_exclusive(path, x=[1])
    with pytest.raises(FileExistsError):
        save_npz_exclusive(path, x=[2])
    link = tmp_path / "link.npz"
    link.symlink_to(path)
    with pytest.raises(FileExistsError):
        save_npz_exclusive(link, x=[2])
    np.testing.assert_array_equal(np.load(path)["x"], [1])


def test_fresh_solver_free_checkpoint_resume(tmp_path):
    # Copy only inference modules/model/JSON checkpoint. No field files, SciPy,
    # or evaluator module exists in the isolated runtime's module namespace.
    package = tmp_path / "runtime"
    package.mkdir()
    (package / "__init__.py").write_text("")
    source = Path(im.__file__).parent
    for name in [
        "intervention_model",
        "geometry_model",
        "present_state_model",
        "repeated_intervention_model",
    ]:
        shutil.copyfile(source / f"{name}.py", package / f"{name}.py")
    m, z = model(), np.array([0.001, 0.01, -0.002])
    (tmp_path / "model.json").write_text(json.dumps(m))
    checkpoints = []

    def checkpoint(state):
        path = tmp_path / f"checkpoint-{state.flow.steps}.json"
        state.checkpoint(path)
        checkpoints.append(path.name)

    _, expected = rm.forecast(m, z, [10, 30], [0.02, -0.02], checkpoint=checkpoint)
    script = """
import builtins,json,sys
sys.path.insert(0,'.')
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.startswith(('scipy','experiments')) or 'simulator' in name:
        raise RuntimeError('scientific imports forbidden')
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from runtime.intervention_model import State
from runtime.repeated_intervention_model import continue_state
model=json.load(open('model.json'))
checkpoint=json.load(open(sys.argv[1]))
state=State.restore(model,checkpoint)
_,y=continue_state(state,[10,30],[.02,-.02],40)
print(json.dumps(y.tolist()))
"""
    (tmp_path / "resume.py").write_text(script)
    for cp in checkpoints:
        output = subprocess.check_output(
            [sys.executable, "-I", "resume.py", cp], cwd=tmp_path, text=True
        )
        np.testing.assert_array_equal(json.loads(output), expected)
