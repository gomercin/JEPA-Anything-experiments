"""Fast immutable-data replay; never execute the scientific reference."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.mediated_patterns import intervention_model as im
from experiments.mediated_patterns import present_state_model as pm

HERE = Path(__file__).resolve().parents[1]
DATA = HERE / "data/work/mediated_patterns/intervention_aware_state"


def load(p):
    return json.loads(p.read_text())


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def test_portable_hashes_and_exclusive_restore(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "intervention_restore", HERE / "restore.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = load(HERE / "artifacts.json")
    assert sum(r["bytes"] for r in manifest["files"]) < 25 * 1024**2
    assert module.helpers.verify(manifest, HERE / "data") == len(manifest["files"])
    subset = {"files": manifest["files"][:1]}
    module.helpers.restore(subset, HERE / "data", tmp_path / "restored")
    with pytest.raises(FileExistsError):
        module.helpers.restore(subset, HERE / "data", tmp_path / "restored")


def test_prospective_seal_and_saved_checkpoint_replay():
    frozen = DATA / "frozen-01"
    fresh = DATA / "fresh-01"
    freeze = load(frozen / "freeze.json")
    models = load(frozen / "models.json")
    seal = load(fresh / "seal.json")
    assert freeze["sources"] == load(fresh / "protocol.json")["sources"]
    assert (
        sha(frozen / "models.json") == freeze["models_sha256"] == seal["models_sha256"]
    )
    assert not set(freeze["fresh_seeds"]) & set(freeze["development_seeds"])
    for name, digest in {**seal["predictions"], **seal["checkpoint_hashes"]}.items():
        assert sha(fresh / name) == digest
    assert sha(fresh / "initial-descriptors.npz") == seal["initialization_sha256"]
    z0 = np.load(fresh / "initial-descriptors.npz")["z"]
    for i, z in enumerate(z0):
        model = models[freeze["selected"]]
        predictions = np.load(fresh / f"prediction-{i}-{freeze['selected']}.npz")
        for ai, a in enumerate([0.0, *freeze["amplitudes"]]):
            state = im.State.restore(
                model, load(fresh / f"checkpoint-{i}-{freeze['selected']}-a{a:g}.json")
            )
            np.testing.assert_array_equal(state.advance(8), predictions["z"][ai, 0])
            np.testing.assert_array_equal(state.response(), predictions["y"][ai, 0])
            np.testing.assert_array_equal(state.advance(20), predictions["z"][ai, 1])
            np.testing.assert_array_equal(state.response(), predictions["y"][ai, 1])
        # Before any new event, added state preserves prior G/F capability exactly.
        base = im.State(models["ignore"], z)
        augmented = im.State(model, z)
        np.testing.assert_array_equal(base.advance(40), augmented.advance(40))
        np.testing.assert_array_equal(base.response(), augmented.response())


def test_all_four_branch_targets_and_score_arithmetic():
    fresh = DATA / "fresh-01"
    contract = load(DATA / "frozen-01/freeze.json")
    rows = load(fresh / "rows.json")
    scores = load(DATA / "analysis-final/scores.json")
    for p in fresh.glob("*gap*.npz"):
        d = np.load(p)
        a = d["absolute"]
        truth = np.stack(
            [a[:, 2, 2, :2] - a[:, 0, 2, :2], a[:, 3, 2, :2] - a[:, 1, 2, :2]]
        )
        np.testing.assert_array_equal(d["y"], truth)
    for s in scores:
        i = rows.index({"seed": s["seed"], "history": s["history"]})
        if s["model"] == "exact_centers_F":
            continue
        ai = [0.0, *contract["amplitudes"]].index(s["a"])
        gi = contract["gaps"].index(s["gap"])
        actual = np.load(
            fresh / f"s{s['seed']}-{s['history']}-a{s['a']:g}-gap{s['gap']}.npz"
        )["y"]
        pred = np.load(fresh / f"prediction-{i}-{s['model']}.npz")["y"][[0, ai], gi]
        t, p = (
            (actual[1] - actual[0], pred[1] - pred[0])
            if s["kind"] == "D_event"
            else (
                actual[int(s["kind"] == "R_after")],
                pred[int(s["kind"] == "R_after")],
            )
        )
        mask = pm.TIMES >= (40 if s["window"] == "late" else 0)
        o = ["mass", "moment"].index(s["output"])
        rms = float(np.sqrt(np.mean(t[mask, o] ** 2)))
        err = float(np.sqrt(np.mean((p[mask, o] - t[mask, o]) ** 2)))
        assert s["rms"] == rms and s["error_rms"] == err
        assert s["resolved"] == (rms > s["floor"])
        assert s["passed"] == (
            rms > s["floor"] and err / rms <= (0.1 if s["kind"] == "D_event" else 0.02)
        )
