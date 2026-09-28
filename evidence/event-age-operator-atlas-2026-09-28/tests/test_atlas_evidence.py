"""Replay published atlas evidence only; no scientific simulator execution."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.mediated_patterns import event_atlas_model as am
from experiments.mediated_patterns.event_age_operator_family import scores
from experiments.mediated_patterns.event_age_response import (
    WINDOWS,
    arrays,
    paired,
    rms,
)
from experiments.mediated_patterns.event_state_analysis import summary

HERE = Path(__file__).resolve().parents[1]


def load(p):
    return json.loads(p.read_text())


def restorer():
    spec = importlib.util.spec_from_file_location("atlas_restore", HERE / "restore.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("atlas-evidence")
    restorer().unpack(root, fixtures=True)
    return root / "work/mediated_patterns/event_age_operator_atlas"


def test_archive_exclusive_safe_and_budgeted(data, tmp_path):
    m = restorer().manifest(fixtures=True)
    assert sum(a["bytes"] for a in m["archives"]) < 25 * 1024**2
    assert restorer().helpers.verify(m, data.parents[2]) == len(m["files"])
    with pytest.raises(FileExistsError):
        restorer().unpack(data.parents[2], fixtures=True)
    link = tmp_path / "linked"
    link.symlink_to(tmp_path / "absent")
    with pytest.raises(ValueError):
        restorer().unpack(link, fixtures=True)
    assert 30 + sum(load(p)["cpu_seconds"] for p in data.glob("*/budget.json")) < 1800


def test_target_and_preparation_exclusion_with_same_coordinates(data):
    rows = load(data / "fit-linear-01/rows.json")
    groups = np.array([r["seed"] for r in rows])
    ages = np.array([r["age"] for r in rows])
    x = arrays(data / "report-01/evaluation-table.npz")["x"]
    assert len(set(groups)) == 12
    assert set(ages) == set(am.REFINED)
    for held in (15, 18, 20):
        for left, right in zip(
            load(data / f"fit-linear-01/coarse-{held}-folds.json"),
            load(data / f"fit-linear-01/refined-{held}-folds.json"),
            strict=True,
        ):
            models = []
            for fold in (left, right):
                tr = np.asarray(fold["training_indices"])
                rp = np.asarray(fold["representation_indices"])
                te = np.asarray(fold["test_indices"])
                assert not set(tr) & set(te)
                assert not (groups[tr] == fold["held_group"]).any()
                assert not (ages[tr] == held).any()
                assert set(rp) <= set(tr)
                assert not (ages[rp] == held).any()
                model = load(data / "fit-linear-01" / Path(fold["model"]).name)
                np.testing.assert_allclose(
                    model["mean"], x[rp].mean(0), rtol=1e-13, atol=1e-15
                )
                np.testing.assert_allclose(
                    model["scale"], x[rp].std(0), rtol=1e-13, atol=1e-15
                )
                assert held not in model["nodes"]
                models.append(model)
            for key in ("mean", "scale", "bases", "scales"):
                assert models[0][key] == models[1][key]
            for j, age in enumerate(models[0]["nodes"]):
                k = models[1]["nodes"].index(age)
                np.testing.assert_array_equal(
                    models[0]["coefficients"][j], models[1]["coefficients"][k]
                )


def test_scores_and_inference_reconstruct(data):
    rows = load(data / "fit-linear-01/rows.json")
    table = arrays(data / "report-01/evaluation-table.npz")
    y, x, ages = table["y"], table["x"], table["ages"]
    floors = load(data / "fit-linear-01/floors.json")
    for family in ("linear", "quadratic", "pchip"):
        folder = data / f"fit-{family}-01"
        if not folder.exists():
            continue
        decision = load(folder / "decision.json")
        for density in ("coarse", "refined"):
            for age in (15, 18, 20):
                key = f"{density}-{age}"
                saved = arrays(folder / f"{key}-predictions.npz")
                ii = saved["indices"]
                for mode, field in (("coefficient", "y"), ("direct", "direct")):
                    rebuilt = scores(y[ii], saved[field], [rows[i] for i in ii], floors)
                    original = load(folder / f"{key}-{mode}-scores.json")
                    assert (
                        summary(rebuilt)["failed"]
                        == decision["results"][key][mode]["failed"]
                    )
                    assert [r["passed"] for r in rebuilt] == [
                        r["passed"] for r in original
                    ]
                    np.testing.assert_allclose(
                        [r["error_rms"] for r in rebuilt],
                        [r["error_rms"] for r in original],
                        rtol=1e-12,
                        atol=1e-24,
                    )
                fold = next(
                    f
                    for f in load(folder / f"{key}-folds.json")
                    if f["held_group"] == 20101
                )
                model = load(data / "fit-linear-01" / Path(fold["model"]).name)
                for i in fold["test_indices"]:
                    local = int(np.flatnonzero(ii == i)[0])
                    pred = am.predict_panel(
                        model, x[i : i + 1], ages[i : i + 1], family
                    )[0]
                    np.testing.assert_allclose(
                        pred, saved["y"][local], rtol=1e-11, atol=1e-21
                    )
                if family != "pchip":
                    np.testing.assert_allclose(
                        saved["y"], saved["direct"], rtol=1e-12, atol=1e-21
                    )


def test_new_refinement_uses_matched_R_then_conservative_D(data):
    for age, base, index in ((14, "pilot-01", 0), (25, "develop-01", 23)):
        y = arrays(data / base / "targets.npz")["y"][index]
        absolute = np.array(
            [
                paired(data / base / f"reference-{index}-{a:g}.npz")["absolute"]
                for a in (0.0, -0.02, 0.02)
            ]
        )
        floors = {}
        for label in ("halfdt", "doubleN"):
            r = np.array(
                [
                    paired(data / f"refine{age}-01" / label / f"reference-{a:g}.npz")[
                        "response"
                    ]
                    for a in (0.0, -0.02, 0.02)
                ]
            )
            for win, sl in WINDOWS.items():
                err = np.maximum(
                    5 * rms((r - y)[:, sl], axis=1),
                    64
                    * np.finfo(float).eps
                    * abs(absolute[:, :, :, 2, :2]).max(axis=(0, 1, 2)),
                )
                key = "D1/" + win
                for j in (1, 2):
                    floors[key] = np.maximum(
                        floors.get(key, np.zeros(2)), err[0] + err[j]
                    )
        published = load(data / f"refine{age}-01/refinement.json")["floors"]
        for key, value in floors.items():
            np.testing.assert_allclose(value, published[key], rtol=1e-12, atol=1e-24)


def test_fresh_protection_and_regime_records(data):
    status = load(data / "report-01/status.json")
    if status["development"] != "DEVELOPMENT_QUALIFIED":
        assert status["fresh_age24"] == "NOT_RUN"
        assert not (data / "fresh-01").exists()
        assert not (data / "repeated-01").exists()
    assert status["reserved_seeds"] == [26101, 26102, 26103]
    assert all(r["qualified"] for r in load(data / "report-01/physical-summary.json"))


def test_fresh_seal_interpolation_and_no_age24_calibration(data):
    from experiments.mediated_patterns.present_state_transmission import digest

    frozen = load(data / "frozen-01/freeze.json")
    model = load(data / "frozen-01/model.json")
    assert 24 not in model["nodes"]
    assert model["nodes"] == list(am.REFINED)
    assert am.neighbors(model["nodes"], 24, "quadratic").tolist() == [4, 5, 6]
    base = data / "fresh-01"
    seal = load(base / "seal.json")
    assert seal["model_sha256"] == digest(data / "frozen-01/model.json")
    assert all(digest(base / name) == sha for name, sha in seal["predictions"].items())
    rows = load(base / "rows.json")
    assert {r["seed"] for r in rows} == {26101, 26102, 26103}
    assert not {r["seed"] for r in rows} & set(frozen["development_groups"])
    assert {r["age"] for r in rows} == {24, 25}
    x = arrays(base / "states.npz")["x"]
    y = arrays(base / "targets.npz")["y"]
    for family in ("linear", "quadratic"):
        pred = arrays(base / f"prediction-{family}.npz")["y"]
        replay = am.predict_panel(model, x, [r["age"] for r in rows], family)
        np.testing.assert_allclose(replay, pred, rtol=1e-12, atol=1e-22)
        gates = scores(y, pred, rows, frozen["floors"])
        original = load(data / f"fresh-analysis-01/{family}-scores.json")
        assert [g["passed"] for g in gates] == [g["passed"] for g in original]
        np.testing.assert_allclose(
            [g["error_rms"] for g in gates],
            [g["error_rms"] for g in original],
            rtol=1e-12,
            atol=1e-24,
        )
