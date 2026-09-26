"""Instrument invariants, not desired model superiority."""

import numpy as np
import pytest

from experiments.mediated_patterns import intervention_model as im
from experiments.mediated_patterns import present_state_model as pm
from experiments.mediated_patterns.intervention_aware_state import (
    four_responses,
    jump_instrument,
)
from experiments.mediated_patterns.source_receiver_relay import CFG


def specimen():
    x = np.arange(CFG.n) * CFG.length / CFG.n - CFG.length / 2
    u = sum(np.exp(-(((x - c) / 4) ** 2)) * np.cos(x - c + 0.1) for c in pm.ANCHORS)
    return np.array([u, 0.1 * u * u])


@pytest.mark.parametrize("a", [0.0, 0.02, -0.02])
def test_exact_additive_center_identity_and_locality(a):
    original = specimen()
    after, info = jump_instrument(original, a)
    assert info["identity_max"] < 1e-13
    assert np.array_equal(after[1], original[1])
    assert np.array_equal(np.array(info["after"])[1:], np.array(info["before"])[1:])
    assert abs(info["l2"] - abs(a)) < 1e-14


def test_four_branch_subtraction_removes_lingering_output():
    base = np.arange(12.0).reshape(3, 4)
    old = 7 * base
    probe = 2 * base
    interaction = 0.3 * base
    y = np.stack(
        [base, base + old, base + probe, base + old + probe + interaction], axis=1
    )
    r = four_responses(y)
    np.testing.assert_allclose(r[0], probe)
    np.testing.assert_allclose(r[1] - r[0], interaction)


def test_kicks_enforce_zero_and_locality_train_only_scaling():
    z = np.arange(18.0).reshape(6, 3) / 20
    a = np.array([0.02, -0.02] * 3)
    j = np.zeros_like(z)
    j[:, 0] = a * z[:, 0]
    model = im.fit_kick(z, a, j, "dependent")
    np.testing.assert_array_equal(model["mean"], z.mean(axis=0))
    for x in [z[0], np.array([50.0, 70.0, 90.0])]:
        np.testing.assert_array_equal(im.kick(model, x, 0), x)
        np.testing.assert_array_equal(im.kick(model, x, 0.02)[1:], x[1:])


def test_grouped_descendants_never_cross():
    groups = np.repeat([8101, 10101, 10102], 12)
    for train, test in pm.grouped_folds(groups):
        assert not set(groups[train]) & set(groups[test])


def tiny_model():
    from experiments.mediated_patterns.geometry_model import fit

    z = np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3], [0.2, 0.1, 0.4]])
    G = fit(z, np.zeros_like(z), kind="affine")
    F = pm.fit(z, np.ones((3, 161, 2)), [], "geometry", rank=1)
    J = im.fit_kick(
        z,
        [0.02, -0.02, 0.02],
        np.array([[0.001, 0, 0], [-0.001, 0, 0], [0.001, 0, 0]]),
        "constant",
    )
    return {"G": G, "F": F, "J": J}


def test_causal_events_segmented_and_zero_equivalence(tmp_path):
    model = tiny_model()
    z = np.array([0.1, 0.2, 0.3])
    a = im.State(model, z)
    b = im.State(model, z)
    # Runtime has no future schedule input; same prefix despite differing future events.
    np.testing.assert_array_equal(a.advance(4), b.advance(4))
    b.event(0.0)
    a.advance(6)
    b.advance(3)
    b.advance(3)
    np.testing.assert_array_equal(a.flow.z, b.flow.z)
    a.event(0.02)
    a.advance(2)
    a.checkpoint(tmp_path / "state.json")
    import json

    restored = im.State.restore(
        model, json.loads((tmp_path / "state.json").read_text())
    )
    np.testing.assert_array_equal(a.advance(8), restored.advance(8))
    with pytest.raises(FileExistsError):
        a.checkpoint(tmp_path / "state.json")
    (tmp_path / "dangling.json").symlink_to(tmp_path / "absent.json")
    with pytest.raises(FileExistsError):
        a.checkpoint(tmp_path / "dangling.json")
    assert not (tmp_path / "absent.json").exists()


def test_runtime_solver_free_fresh_process(tmp_path):
    import json
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    model = tiny_model()
    state = im.State(model, [0.1, 0.2, 0.3])
    state.event(0.02)
    state.advance(2)
    state.checkpoint(tmp_path / "checkpoint.json")
    pm.save_json(tmp_path / "model.json", model)
    package = tmp_path / "runtime"
    package.mkdir()
    (package / "__init__.py").write_text("")
    src = Path(im.__file__).parent
    for name in [
        "intervention_model.py",
        "geometry_model.py",
        "present_state_model.py",
        "measurements.py",
    ]:
        shutil.copyfile(src / name, package / name)
    code = """import sys,json,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if any(x in fullname for x in ['scipy','simulator','intervention_analysis','intervention_aware_state']): raise RuntimeError('scientific module unavailable')
sys.meta_path.insert(0,Block())
from runtime.intervention_model import State
m=json.load(open('model.json'));s=State.restore(m,json.load(open('checkpoint.json')))
print(json.dumps(s.advance(8).tolist()))
"""
    result = subprocess.check_output(
        [sys.executable, "-c", code], cwd=tmp_path, text=True
    )
    np.testing.assert_array_equal(json.loads(result), state.advance(8))


def test_extended_state_has_local_jump_and_zero_memory_preserves_baseline():
    model = tiny_model()
    model["transient"] = {
        "order": 1,
        "A": [[0.8]],
        "B": [[1.0, 0.0]],
        "C": [[0.001], [0.0], [0.0]],
        "J": model["J"],
    }
    model["readout"] = {
        "event_basis": True,
        "interaction": False,
        "coefficients": np.ones((4, 2)).tolist(),
        "basis": np.ones((1, 161)).tolist(),
        "scale_y": [1e-8, 1e-8],
    }
    baseline = im.State(tiny_model(), [0.1, 0.2, 0.3])
    extended = im.State(model, [0.1, 0.2, 0.3])
    np.testing.assert_array_equal(baseline.advance(30), extended.advance(30))
    np.testing.assert_array_equal(baseline.response(), extended.response())
    before = extended.flow.z.copy()
    extended.event(0.02)
    np.testing.assert_array_equal(extended.flow.z[1:], before[1:])
    assert len(extended.memory) == 1 and len(extended.response_memory) == 2
    before_memory = extended.response_memory.copy()
    extended.advance(2)
    np.testing.assert_allclose(
        extended.response_memory, [before_memory[0], before_memory[1] * np.exp(-2 / 20)]
    )
