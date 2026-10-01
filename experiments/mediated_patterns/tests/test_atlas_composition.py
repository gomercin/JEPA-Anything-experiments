"""Composition instrument tests; no field simulation or claimed scientific outcome."""

import numpy as np
import pytest

from experiments.mediated_patterns import event_atlas_model as am
from experiments.mediated_patterns import geometry_model as gm
from experiments.mediated_patterns.atlas_composition import decomposition
from experiments.mediated_patterns.atlas_composition_runtime import forecast
from experiments.mediated_patterns.repeated_intervention_model import contrasts


def test_signed_decomposition_retains_baseline_and_sse_cross_terms():
    rng = np.random.default_rng(541)
    y = rng.normal(size=(4, 161, 2))
    components = rng.normal(size=(4, 161, 2))
    d = decomposition(y, components)
    np.testing.assert_allclose(d["terms"].sum(0), d["common_residual"], atol=2e-15)
    np.testing.assert_allclose(
        d["terms"].sum(0) + d["baseline_inconsistency"],
        d["literal_residual"],
        atol=2e-15,
    )
    gram = np.einsum("ito,jto->oij", d["terms"], d["terms"]) / 161
    np.testing.assert_allclose(
        gram.sum((1, 2)), np.mean(d["common_residual"] ** 2, axis=0)
    )


def test_runtime_uses_one_common_baseline_and_known_age(monkeypatch):
    calls = []
    monkeypatch.setattr(gm, "rollout", lambda g, z, t: np.array([z + 1, z + 2]))

    def query(model, z, a, age, family):
        calls.append((z.copy(), a, age, family))
        return np.full((161, 2), z[0] * 3 + a * age)

    monkeypatch.setattr(am, "predict", query)
    p = forecast({}, {}, np.zeros(3), [10, 25], [0.02, -0.02])
    assert [c[2] for c in calls] == [30, 30, 15, 15]
    assert all(c[3] == "quadratic" for c in calls)
    common, literal = contrasts(p["common"]), contrasts(p["literal"])
    np.testing.assert_allclose(common["D12"], 0.02 * 30 - 0.02 * 15)
    np.testing.assert_allclose(common["D2|1"], -0.02 * 15)
    np.testing.assert_allclose(literal["D12"] - common["D12"], 3)
    np.testing.assert_allclose(common["K12"], 0, atol=2e-15)
    with pytest.raises(ValueError):
        forecast({}, {}, np.zeros(3), [25, 10], [0.02, -0.02])


def test_fresh_process_forecast_with_solvers_and_array_reads_denied(tmp_path):
    import json
    import shutil
    import subprocess
    import sys
    from pathlib import Path

    from experiments.mediated_patterns.tests.test_event_atlas import fixture

    x, ages, _groups, y, _truth = fixture()
    model = am.fit(x, ages, y, sorted(set(ages)))
    g = gm.fit(x, np.zeros_like(x), kind="quadratic")
    z = [0.2, -0.3, 0.7]
    expected = forecast(model, g, z, [10, 25], [0.02, -0.02])["common"]
    package = tmp_path / "runtime"
    package.mkdir()
    (package / "__init__.py").write_text("")
    src = Path(__file__).parents[1]
    for name in [
        "atlas_composition_runtime",
        "event_atlas_model",
        "event_operator_model",
        "geometry_model",
        "present_state_model",
        "measurements",
        "repeated_intervention_model",
        "intervention_model",
    ]:
        shutil.copyfile(src / (name + ".py"), package / (name + ".py"))
    (tmp_path / "input.json").write_text(json.dumps({"model": model, "g": g, "z": z}))
    (tmp_path / "run.py").write_text("""
import sys, json, importlib.abc
sys.path.insert(0, '.')
import numpy as np
class NoScience(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path, target=None):
        if any(s in fullname for s in ['simulator','source_receiver_relay','geometry_evolution','organization_response','atlas_composition.']):
            raise RuntimeError('physical/evaluator imports forbidden')
sys.meta_path.insert(0, NoScience())
def denied(*args, **kwargs):
    raise RuntimeError('field/response arrays forbidden')
np.load = denied
from runtime.atlas_composition_runtime import forecast
v = json.load(open('input.json'))
print(json.dumps(forecast(v['model'],v['g'],v['z'],[10,25],[.02,-.02])['common'].tolist()))
""")
    actual = json.loads(
        subprocess.check_output(
            [sys.executable, "-I", "run.py"], cwd=tmp_path, text=True
        )
    )
    np.testing.assert_array_equal(actual, expected)
