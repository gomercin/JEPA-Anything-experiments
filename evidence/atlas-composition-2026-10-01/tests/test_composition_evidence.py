"""Portable sealed forecast, branch algebra, scores and margin replay; no PDE run."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.mediated_patterns.atlas_composition import decomposition
from experiments.mediated_patterns.atlas_composition_reporting import numerical_margins
from experiments.mediated_patterns.atlas_composition_runtime import forecast
from experiments.mediated_patterns.event_age_response import score as isolated_score
from experiments.mediated_patterns.repeated_intervention_analysis import (
    WINDOWS,
    rms,
    score,
    truth,
)
from experiments.mediated_patterns.repeated_intervention_model import (
    CONTRASTS,
    contrasts,
)

HERE = Path(__file__).resolve().parents[1]


def load(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def restorer():
    spec = importlib.util.spec_from_file_location(
        "composition_restore", HERE / "restore.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    root = tmp_path_factory.mktemp("composition-evidence")
    restorer().unpack(root)
    return root / "work/mediated_patterns/atlas_composition"


def test_archive_identity_and_safe_exclusive_restore(data, tmp_path):
    m = load(HERE / "artifacts.json")
    assert m["bytes"] == sum(r["bytes"] for r in m["files"])
    assert sum(a["bytes"] for a in m["archives"]) < 25 * 1024**2
    module = restorer()
    root = data.parents[2]
    assert module.helpers.verify(m, root) == len(m["files"])
    subset = {"files": m["files"][:1]}
    module.helpers.restore(subset, root, tmp_path / "copy")
    with pytest.raises(FileExistsError):
        module.helpers.restore(subset, root, tmp_path / "copy")
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "copy", target_is_directory=True)
    with pytest.raises(ValueError):
        module.unpack(link)


def test_frozen_models_sources_provenance_and_seals(data):
    f = load(data / "frozen-01/freeze.json")
    fresh = data / "fresh-01"
    s = load(fresh / "seal.json")
    assert f["seeds"] == [28101, 28102, 28103]
    p = load(data / "frozen-01/provenance.json")
    assert p["hits"] == [] and len(p["checked"]) == 1963
    assert not set(f["seeds"]) & set(p["recorded_groups"])
    assert len({r["seed"] for r in load(fresh / "rows.json")}) == 3
    assert f["models"] == {
        "G.json": "39b498baaa3814dfb589f45fe6b75588ebd1ac0c80180777154166df26982456",
        "model.json": "8635f9a4cfecd21ea0a7da4af82c155231d1cfb141d4126442fcdd33192e816d",
    }
    for name, h in f["models"].items():
        assert sha(data / "frozen-01" / name) == h
    assert s["model_sha256"] == f["models"]["model.json"]
    assert s["G_sha256"] == f["models"]["G.json"]
    for name, h in s["predictions"].items():
        assert sha(fresh / name) == h
    assert sha(fresh / "initial-descriptors.npz") == s["descriptors_sha256"]
    assert s["sealed_time_ns"] < load(fresh / "completion.json")["completed_time_ns"]
    assert len(s["predictions"]) == 24
    assert f["sources"] == load(fresh / "budget.json")["sources"]
    for stage in data.iterdir():
        if not (stage / "budget.json").exists():
            continue
        for name, h in load(stage / "budget.json")["sources"].items():
            assert sha(stage / "sources" / name) == h


def test_all_forecasts_replay_from_only_t50_and_schedule(data):
    model = load(data / "frozen-01/model.json")
    g = load(data / "frozen-01/G.json")
    fresh = data / "fresh-01"
    z = np.load(fresh / "initial-descriptors.npz")["z"]
    for case in load(fresh / "cases.json"):
        s = case["schedule"]
        p = forecast(model, g, z[case["index"]], s["times"], s["amplitudes"])
        saved = np.load(fresh / f"prediction-{case['index']}-{s['name']}.npz")
        for k in p:
            np.testing.assert_allclose(p[k], saved[k], rtol=1e-12, atol=1e-20)
        common = contrasts(saved["common"])
        np.testing.assert_allclose(common["K12"], 0, atol=2e-22)
        components = saved["components"]
        np.testing.assert_allclose(
            common["D12"],
            (components[1] - components[0]) + (components[3] - components[2]),
            atol=2e-22,
        )


def test_eight_branch_subtraction_all_comparators_and_scores(data):
    fresh = data / "fresh-01"
    f = load(data / "frozen-01/freeze.json")
    all_scores = load(data / "fresh-analysis-01/scores.json")
    for c in load(fresh / "cases.json"):
        y, aa, _z = truth(fresh, c)
        np.testing.assert_array_equal(y, aa[:, :, 1, 2, :2] - aa[:, :, 0, 2, :2])
        p = np.load(fresh / f"prediction-{c['index']}-{c['schedule']['name']}.npz")
        oracle = y.copy()
        oracle[3] = y[1] + y[2] - y[0]
        for name, pred in [
            ("common", p["common"]),
            ("literal", p["literal"]),
            ("true-addition", oracle),
        ]:
            computed = score(y, pred, f["floors"])
            stored = [
                s
                for s in all_scores
                if s["index"] == c["index"]
                and s["schedule"] == c["schedule"]["name"]
                and s["model"] == name
            ]
            assert len(computed) == len(stored)
            for s, e in zip(stored, computed, strict=True):
                for k, v in e.items():
                    assert s[k] == v
        for j in range(2):
            computed = isolated_score(
                y[[0, j + 1]], p["components"][2 * j : 2 * j + 2], f["isolated_floors"]
            )
            stored = [
                s
                for s in all_scores
                if s["index"] == c["index"]
                and s["schedule"] == c["schedule"]["name"]
                and s["model"] == f"isolated-{j + 1}"
            ]
            for s, e in zip(stored, computed, strict=True):
                assert s["pass"] == e.pop("passed")
                for k, v in e.items():
                    assert s[k] == v
        d = decomposition(y, p["components"])
        for sl in WINDOWS.values():
            terms = d["terms"][:, sl]
            gram = np.einsum("ito,jto->oij", terms, terms) / len(y[0, sl])
            np.testing.assert_allclose(
                gram.sum((1, 2)),
                rms(d["common_residual"][sl]) ** 2,
                rtol=1e-10,
                atol=1e-40,
            )
    for p in fresh.glob("reference-*.json"):
        row = load(p)
        assert row["qualified"]
        for e in row["events"]:
            assert e["identity_max"] < 1e-13
            np.testing.assert_array_equal(e["before"][1:], e["after"][1:])


def test_refinement_direct_and_propagated_bounds_and_decision(data):
    for stage, base in [("refine-01", "exposed-01"), ("fresh-refine-01", "fresh-01")]:
        if not (data / stage).exists():
            continue
        if stage == "refine-01":
            case = next(
                c
                for c in load(data / base / "cases.json")
                if c["seed"] == 18103
                and c["history"] == "none"
                and c["schedule"]["name"] == "timing"
            )
        else:
            case = load(data / stage / "selection.json")["case"]
            assert (
                sha(data / stage / "fixed-forecast.npz")
                == load(data / stage / "selection.json")["fixed_forecast_sha256"]
            )
        y, absolute, _z = truth(data / base, case)
        for record in load(data / stage / "refinement.json")["records"]:
            sub = data / stage / record["refinement"]
            r, _aa, _zz = truth(sub, load(sub / "cases.json")[0])
            sl = WINDOWS[record["window"]]
            ro = 64 * np.finfo(float).eps * abs(absolute[:, :, :, 2, :2]).max((0, 1, 2))
            pair = np.maximum(5 * rms((r - y)[:, sl], axis=1), ro)
            np.testing.assert_array_equal(pair, record["matched_pair_bounds"])
            np.testing.assert_array_equal(
                abs(np.array(CONTRASTS[record["kind"]])) @ pair, record["propagated"]
            )
            np.testing.assert_array_equal(
                5 * rms(contrasts(r - y)[record["kind"]][sl]), record["direct"]
            )
        pred = np.load(
            data / base / f"prediction-{case['index']}-{case['schedule']['name']}.npz"
        )["common"]
        margins = numerical_margins(data, stage, data / base, case, pred)
        key = "exposed_margins" if stage == "refine-01" else "fresh_margins"
        assert margins == load(data / "report-01/decision.json")[key]
    d = load(data / "report-01/decision.json")
    assert d["groups"] == 3 and d["schedule_cases"] == 24
    if d["decision"] == "INCONCLUSIVE_NUMERICAL_THRESHOLD_MARGIN":
        assert d["primary_failures"] and not d["isolated_failures"]
        assert any(m["classification"] == "marginal" for m in d["fresh_margins"])
        assert all(m["classification"] != "robust-failure" for m in d["fresh_margins"])
    receipts = [load(p) for p in data.glob("*/budget.json")]
    assert 30 + sum(r["cpu_seconds"] for r in receipts) < 1800
    assert all(r["status"] == "complete" for r in receipts)
    assert all(
        s["cpu_seconds"] < 120 and s["wall_seconds"] < 180
        for r in receipts
        for s in r["sections"]
    )


def test_critical_audit_preserves_nominal_pass_and_marginal_qualification(data):
    audit = load(data / "review-01/audit.json")
    nominal = load(data / "report-01/decision.json")
    assert nominal["decision"] == "BOUNDED_ORDINARY_ADDITION_SUFFICIENT"
    assert audit["robust_threshold_decision"] == "INCONCLUSIVE_AT_MARGINAL_CASES"
    scores = load(data / "fresh-analysis-01/scores.json")
    primary = [s for s in scores if s["model"] == "common" and s["kind"] != "K12"]
    assert len(primary) == 768 and all(s["pass"] for s in primary)
    near = []
    for s in primary:
        limit = 0.02 if s["kind"].startswith("R") else 0.1
        headroom = limit * s["magnitude"] - s["error_rms"]
        if headroom <= 1.1 * s["floor"]:
            near.append(
                dict(
                    **s,
                    pass_headroom=headroom,
                    headroom_over_floor=headroom / s["floor"],
                )
            )
    assert near == audit["marginal_fresh_passes"] and len(near) == 3
    order = audit["order"]
    assert (
        order["last_forecast_mtime_ns"]
        < order["seal_time_ns"]
        < order["first_reference_mtime_ns"]
    )
