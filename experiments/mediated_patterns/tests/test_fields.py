"""Engineering contracts; no assertions of binding, emergence, or model superiority."""

import hashlib
import inspect
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from experiments.mediated_patterns import simulator
from experiments.mediated_patterns.explore import response, start_panel, write_json
from experiments.mediated_patterns.measurements import describe, outward, regions
from experiments.mediated_patterns.reduced import HORIZONS, ResponseModel, features
from experiments.mediated_patterns.simulator import Config, Field


def test_seeded_evolution():
    f = Field(Config(n=128, length=64, dt=0.025))
    a = f.seed(seed=11, variation=0.01)
    assert np.array_equal(a, f.seed(seed=11, variation=0.01))
    assert not np.array_equal(a, f.seed(seed=22, variation=0.01))
    t, y, _ = f.evolve(a, 1, sample=0.25)
    assert y.shape == (5, 2, 128)
    assert np.isfinite(y).all()
    assert np.array_equal(y, f.evolve(a, 1, sample=0.25)[1])
    assert t[-1] == 1


def test_mediator_analytic_decay():
    f = Field(Config(n=128, length=64))
    s = np.array([np.zeros(128), 1 + 0.2 * np.cos(2 * np.pi * f.x / 64)])
    actual = f.evolve(s, 1)[1][-1]
    expected = np.exp(-1 / f.c.tau) + 0.2 * np.cos(2 * np.pi * f.x / 64) * np.exp(
        -(1 + f.c.diffusion * (2 * np.pi / 64) ** 2) / f.c.tau
    )
    np.testing.assert_allclose(actual[1], expected, atol=1e-13)
    assert np.array_equal(actual[0], np.zeros(128))


def test_pathway_switches_and_compact_pulse():
    c = Config(n=128, length=64)
    f = Field(c)
    state = f.seed()
    p = f.pulse(state, 0, 0.1, width=8, relative=True, compact=True)
    assert np.array_equal(p[:, abs(f.x) >= 8], state[:, abs(f.x) >= 8])
    assert np.array_equal(p[1], state[1])
    assert np.max(abs(p[0] - state[0])) > 0
    source_off = Field(replace(c, source=0))
    assert np.array_equal(source_off.evolve(state, 1)[1][-1, 1], state[1])
    feedback_off = Field(replace(c, feedback=0))
    m = state.copy()
    m[1] = 1
    np.testing.assert_array_equal(
        feedback_off.evolve(state, 1)[1][:, 0], feedback_off.evolve(m, 1)[1][:, 0]
    )


def test_replay_preserves_sham():
    f = Field(Config(n=128, length=64))
    s = f.seed()
    _, a, replay = f.evolve(s, 1, record=True)
    _, b, _ = f.evolve(s, 1, replay=replay)
    np.testing.assert_array_equal(a[:, 0], b[:, 0])
    with pytest.raises(ValueError):
        f.evolve(s, 1, replay=replay[:-1])


def test_spatial_replay_information_boundary():
    f = Field(Config(n=128, length=64))
    s = f.seed()
    _, base, stages = f.evolve(s, 1, record=True)
    p = f.pulse(s, 0, 0.05)
    full = f.evolve(p, 1)[1]
    zero = f.evolve(p, 1, replay=stages, replay_weight=np.zeros(128))[1]
    np.testing.assert_array_equal(full, zero)
    replay = f.evolve(p, 1, replay=stages)[1]
    one = f.evolve(p, 1, replay=stages, replay_weight=np.ones(128))[1]
    np.testing.assert_array_equal(replay, one)
    sham = f.evolve(s, 1, replay=stages, replay_weight=(f.x > 0).astype(float))[1]
    np.testing.assert_array_equal(sham, base)


def test_exterior_sensor_is_local_and_normalized():
    f = Field(Config(n=256, length=64))
    states = np.zeros((2, 2, 256))
    states[0, 1] = 2
    states[1, 1] = 3
    np.testing.assert_allclose(outward(f.x, states, 0), [2, 3], atol=1e-15)
    # Changes around the pattern and in u cannot alter the exterior m readout.
    changed = states.copy()
    changed[:, 0] = 999
    changed[:, 1, abs(f.x) < 8] = 999
    np.testing.assert_array_equal(outward(f.x, states, 0), outward(f.x, changed, 0))


def test_measurement_is_separate_and_refines():
    assert "measurements" not in inspect.getsource(simulator)
    answers = []
    for n in [256, 512]:
        f = Field(Config(n=n, length=64))
        s = np.array([np.cos(f.x) * np.exp(-((f.x / 4) ** 2)), np.ones(n)])
        answers.append(describe(f.x, s, [0]))
    np.testing.assert_allclose(answers[0]["mass"], answers[1]["mass"], rtol=1e-6)
    with pytest.raises(ValueError):
        regions(f.x, [0, 2])


def synthetic_records(seed):
    rows = []
    coeff = np.arange(30).reshape(10, 3) * 1e-6
    for d in [24, 26, 28, 30, 32]:
        for a in [-0.1, 0.05, 0.1]:
            truth = features([d, a]) @ coeff
            series = np.zeros((51, 2))
            series[list(HORIZONS), 1] = truth
            rows.append(
                {
                    "initial_descriptors": {"center": [-d / 2, d / 2]},
                    "amplitude": a,
                    "response": series.tolist(),
                    "independent_response": (series * 0).tolist(),
                    "preparation_id": f"{seed}-{d}",
                }
            )
    return rows


def test_reduced_boundary_and_run_split(tmp_path):
    rows = synthetic_records(11)
    model = ResponseModel.fit(rows)
    r = rows[0]
    y = model.predict(r)
    # Only boundary separation and declared input are available to prediction.
    np.testing.assert_array_equal(
        y, model.predict({k: r[k] for k in ("initial_descriptors", "amplitude")})
    )
    assert model.score(synthetic_records(22))["fresh_preparations_disjoint"]
    assert not model.score(rows)["fresh_preparations_disjoint"]
    p = tmp_path / "model.json"
    model.save(p)
    np.testing.assert_array_equal(model.predict(r), ResponseModel.load(p).predict(r))
    write_json(tmp_path / "score.json", model.score(rows))
    assert json.loads((tmp_path / "score.json").read_text())
    with pytest.raises(ValueError):
        write_json(tmp_path / "bad.json", {"bad": float("nan")})


def test_existing_output_refused_without_mutation(tmp_path):
    p = tmp_path / "prior"
    p.mkdir()
    (p / "evidence.txt").write_text("keep")
    before = {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in p.iterdir()}
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.mediated_patterns.explore",
            "--quick",
            "--output",
            str(p),
        ],
        capture_output=True,
    )
    assert result.returncode != 0
    after = {x.name: hashlib.sha256(x.read_bytes()).hexdigest() for x in p.iterdir()}
    assert before == after
    with pytest.raises(FileExistsError):
        start_panel(p, {})


def test_short_response_serializes(tmp_path):
    f = Field(Config(n=256, length=96))
    s = f.seed((-14, 14))
    result, fields = response(f, s, [-14, 14], 0.05, horizon=1)
    assert fields["base"].shape == (2, 2, 256)
    assert result["replay_sham_error"] == 0
    write_json(tmp_path / "results.json", result)


def test_prior_artifact_preservation():
    manifest = Path("work/mediated_patterns/protected.json")
    if not manifest.exists():
        pytest.skip("Session-local immutable evidence manifest absent in fresh checkout")
    for filename, digest in json.loads(manifest.read_text()).items():
        assert hashlib.sha256(Path(filename).read_bytes()).hexdigest() == digest, filename
