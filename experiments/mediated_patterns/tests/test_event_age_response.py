"""Causal/instrument fixtures, without asserting exponential physical memory."""

import inspect
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from experiments.mediated_patterns import event_age_model as em
from experiments.mediated_patterns import present_state_model as pm
from experiments.mediated_patterns.event_age_analysis import fit_model, geometry_model
from experiments.mediated_patterns.event_age_response import (
    AGES,
    DEV,
    FRESH,
    FRESH_AGES,
    paired,
    score,
    slots,
)
from experiments.mediated_patterns.hybrid_pair import save_npz_exclusive


def fixture():
    c = np.zeros((24, 2))
    c[4] = [1.0, 2.0]
    return {
        "geometry": geometry_model(),
        "decays": [1.0, np.exp(-1 / 7), np.exp(-1 / 40)],
        "readout": {
            "mean_z": [0] * 3,
            "scale_z": [1] * 3,
            "basis": np.ones((1, 161)).tolist(),
            "scale_y": [1.0, 1.0],
            "coefficients": c.tolist(),
        },
    }


def test_known_exponential_unseen_age_and_no_clock_input():
    m = fixture()
    s = em.State(m, [0.001, 0.01, -0.002])
    s.advance(22)
    s.event(0.02)
    s.advance(18)
    expected = np.exp(-18 / 7) * np.array([1.0, 2.0])
    base = pm.predict(m["geometry"]["F"], s.geometry.flow.z[None])[0]
    np.testing.assert_allclose(
        s.response() - base, np.broadcast_to(expected, (161, 2)), rtol=1e-14
    )
    assert list(inspect.signature(em.State.response).parameters) == ["self"]
    saved = s.response().copy()
    s.geometry.flow.steps += 999
    np.testing.assert_array_equal(saved, s.response())
    assert "age" not in inspect.signature(em.readout).parameters


def test_causality_zero_segmented_and_no_reset():
    m = fixture()
    z = [0.001, 0.01, -0.002]
    a, y = em.forecast(m, z, [(10, 0.02), (25, -0.02)])
    b, _ = em.forecast(m, z, [(10, 0.02), (25, 0.01)])
    np.testing.assert_array_equal(a[:25], b[:25])
    s = em.State(m, z)
    s.advance(10)
    s.event(0.02)
    s.advance(7)
    s.advance(8)
    before = s.response_state.copy()
    s.event(0)
    np.testing.assert_array_equal(before, s.response_state)
    s.event(-0.02)
    s.advance(15)
    np.testing.assert_array_equal(s.response(), y)
    assert s.response_state[0, 0] == 0 and s.response_state[1, 0] == 2
    zero, _ = em.forecast(m, z, [(10, 0.0), (25, 0.0)])
    u = em.State(m, z)
    u.advance(40)
    np.testing.assert_array_equal(zero[-1], em.vector(u))
    for n in [-1, 0.5, True]:
        with pytest.raises(ValueError):
            s.advance(n)
    for bad in [0.0, 1.01, float("nan")]:
        broken = fixture()
        broken["decays"][1] = bad
        with pytest.raises(ValueError):
            em.State(broken, z)


def test_matched_four_branches_and_floor(tmp_path):
    absolute = np.zeros((161, 2, 3, 3))
    absolute[:, 0, 2, :2] = 100
    absolute[:, 1, 2, :2] = 103
    path = tmp_path / "pair.npz"
    save_npz_exclusive(path, absolute=absolute, response=np.full((161, 2), 3.0))
    np.testing.assert_array_equal(paired(path)["response"], 3.0)
    yy = np.array([np.ones((161, 2)), np.ones((161, 2)) * 1.01])
    result = score(yy, yy, {"D1/whole": [0.1, 0.1], "D1/late": [0.1, 0.1]})
    assert all(r["passed"] is None for r in result if r["kind"] == "D1")
    with pytest.raises(FileExistsError):
        save_npz_exclusive(path, x=[1])
    link = tmp_path / "link.npz"
    link.symlink_to(tmp_path / "missing")
    with pytest.raises(FileExistsError):
        save_npz_exclusive(link, x=[1])


def test_groups_age_and_exact_center_separation():
    assert not set(DEV) & set(FRESH) and len(FRESH) >= 3
    assert 18 not in AGES and 18 in FRESH_AGES
    assert slots(18, 0.02) == ([10, 22], [0.0, 0.02])
    with pytest.raises(ValueError):
        fit_model([{"seed": DEV[0], "age": 18}])
    s = em.State(fixture(), [0.001, 0.01, -0.002])
    s.event(0.02)
    before = em.vector(s).copy()
    em.readout(s.model, [0.0, 0.0, 0.0], s.response_state)
    np.testing.assert_array_equal(before, em.vector(s))


def test_solver_free_resume_and_serialization(tmp_path):
    package = tmp_path / "runtime"
    package.mkdir()
    (package / "__init__.py").write_text("")
    for name in [
        "event_age_model",
        "intervention_model",
        "geometry_model",
        "present_state_model",
        "measurements",
    ]:
        shutil.copyfile(
            Path(em.__file__).with_name(name + ".py"), package / (name + ".py")
        )
    m = fixture()
    (tmp_path / "model.json").write_text(json.dumps(m))
    cps = []

    def checkpoint(s):
        path = tmp_path / f"checkpoint-{s.geometry.flow.steps}.json"
        s.checkpoint(path)
        cps.append(path.name)
        with pytest.raises(FileExistsError):
            s.checkpoint(path)

    _, expected = em.forecast(
        m, [0.001, 0.01, -0.002], [(10, 0.02), (25, -0.02)], checkpoint=checkpoint
    )
    script = """
import builtins,json,sys
sys.path.insert(0,'.')
original=builtins.__import__
def guard(name,*args,**kwargs):
    if name.startswith(('scipy','experiments')) or 'simulator' in name:raise RuntimeError('solver unavailable')
    return original(name,*args,**kwargs)
builtins.__import__=guard
from runtime.event_age_model import State,continue_state
m=json.load(open('model.json'));s=State.restore(m,json.load(open(sys.argv[1])))
_,y=continue_state(s,[(10,.02),(25,-.02)],40)
print(json.dumps(y.tolist()))
"""
    (tmp_path / "resume.py").write_text(script)
    for cp in cps:
        out = subprocess.check_output(
            [sys.executable, "-I", "resume.py", cp], cwd=tmp_path, text=True
        )
        np.testing.assert_array_equal(json.loads(out), expected)
