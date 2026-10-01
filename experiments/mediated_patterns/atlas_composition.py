"""Opt-in bounded composition cycle; scientific panels are serial and budgeted."""

import argparse
import os
import shutil
import subprocess
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import geometry_model as gm
from . import present_state_model as pm
from . import repeated_intervention_model as rm
from .atlas_composition_runtime import forecast
from .event_age_response import score as isolated_score
from .geometry_assay import initialize_written
from .geometry_evolution import unforced
from .history_conditioned_transmission import HistoryBudget
from .hybrid_pair import save_npz_exclusive as save_npz
from .present_state_transmission import digest, load
from .repeated_intervention_analysis import WINDOWS, rms, score, truth
from .repeated_intervention_state import references, source_hashes
from .simulator import Field
from .source_receiver_relay import CFG

ROOT = Path("work/mediated_patterns/atlas_composition")
ATLAS = Path(
    "/private/tmp/jepa-compose-inherited-atlas/work/mediated_patterns/event_age_operator_atlas"
)
REPEAT = Path(
    "/private/tmp/jepa-compose-inherited-repeat/work/mediated_patterns/repeated_intervention_state"
)
SEEDS = [28101, 28102, 28103]
SCHEDULES = [
    {"name": f"{timing}-{sign}", "times": [10, second], "amplitudes": [0.02, a]}
    for timing, second in [("standard", 30), ("adverse", 25)]
    for sign, a in [("cancel", -0.02), ("reinforce", 0.02)]
]


def models():
    f = load(ATLAS / "frozen-01/freeze.json")
    for name in ["model.json", "G.json"]:
        if digest(ATLAS / "frozen-01" / name) != f["models"][name]:
            raise ValueError("Inherited model bytes changed")
    return load(ATLAS / "frozen-01/model.json"), load(ATLAS / "frozen-01/G.json")


def sealed(out, starts, rows, schedules, budget, frozen=ATLAS / "frozen-01"):
    budget.begin("t50-only-forecasts-before-future-access")
    model, g = load(frozen / "model.json"), load(frozen / "G.json")
    z0 = np.array([pm.extract(s, feature_set="geometry") for s in starts])
    save_npz(out / "initial-descriptors.npz", z=z0)
    for i, z in enumerate(z0):
        for schedule in schedules:
            save_npz(
                out / f"prediction-{i}-{schedule['name']}.npz",
                **forecast(model, g, z, schedule["times"], schedule["amplitudes"]),
            )
    pm.save_json(out / "rows.json", rows)
    pm.save_json(
        out / "seal.json",
        {
            "predictions": {
                p.name: digest(p) for p in sorted(out.glob("prediction-*.npz"))
            },
            "descriptors_sha256": digest(out / "initial-descriptors.npz"),
            "model_sha256": digest(frozen / "model.json"),
            "G_sha256": digest(frozen / "G.json"),
            "order": "Every forecast sealed before ANY corresponding future reference",
            "access": "Exactly one t50 field acquisition per history; only three centers retained",
            "sealed_time_ns": time.time_ns(),
        },
    )
    budget.finish()


def decomposition(y, components):
    """Signed true-minus-predicted D12 terms, with baseline inconsistency explicit."""
    b1, r1, b2, r2 = components
    d1, d2 = y[1] - y[0], y[2] - y[0]
    terms = np.array([d1 - (r1 - b1), d2 - (r2 - b2), rm.contrasts(y)["K12"]])
    inconsistency = b1 - b2
    common_residual = rm.contrasts(y)["D12"] - ((r1 - b1) + (r2 - b2))
    literal_residual = rm.contrasts(y)["D12"] - (r1 + r2 - 2 * b1)
    if not np.allclose(terms.sum(0), common_residual, rtol=1e-10, atol=1e-21):
        raise ValueError("Common residual identity")
    if not np.allclose(
        terms.sum(0) + inconsistency, literal_residual, rtol=1e-10, atol=1e-21
    ):
        raise ValueError("Literal residual identity")
    return {
        "terms": terms,
        "baseline_inconsistency": inconsistency,
        "common_residual": common_residual,
        "literal_residual": literal_residual,
    }


def evaluate(out, data, budget, exposed=False):
    budget.begin("score-and-signed-decomposition")
    floors = load(REPEAT / "frozen-01/freeze.json")["floors"]["floors"]
    single_floors = load(ATLAS / "frozen-01/freeze.json")["floors"]
    cases = load(data / "cases.json")
    scores, decompositions, physical, identities = [], [], [], {}
    for case in cases:
        y, aa, z = truth(data, case)
        p = data / f"prediction-{case['index']}-{case['schedule']['name']}.npz"
        if digest(p) != load(data / "seal.json")["predictions"][p.name]:
            raise ValueError("Forecast seal changed")
        saved = np.load(p, allow_pickle=False)
        meta = {k: case[k] for k in ["index", "seed", "history"]}
        meta["schedule"] = case["schedule"]["name"]
        oracle = np.array([y[0], y[1], y[2], y[1] + y[2] - y[0]])
        for name, pred in [
            ("common", saved["common"]),
            ("literal", saved["literal"]),
            ("true-addition", oracle),
        ]:
            scores.extend(dict(**s, **meta, model=name) for s in score(y, pred, floors))
        for j in range(2):
            for s in isolated_score(
                y[[0, j + 1]], saved["components"][2 * j : 2 * j + 2], single_floors
            ):
                s["pass"] = s.pop("passed")
                scores.append(dict(**s, **meta, model=f"isolated-{j + 1}"))
        d = decomposition(y, saved["components"])
        save_npz(
            out / f"decomposition-{meta['index']}-{meta['schedule']}.npz", y=y, **d
        )
        for window, sl in WINDOWS.items():
            terms = np.concatenate(
                [d["terms"], d["baseline_inconsistency"][None]], axis=0
            )[:, sl]
            gram = np.einsum("ito,jto->oij", terms, terms) / terms.shape[1]
            decompositions.append(
                dict(
                    **meta,
                    window=window,
                    term_order=[
                        "single-event-1-error",
                        "single-event-2-error",
                        "true-K12",
                        "zero-context-inconsistency",
                    ],
                    terms_rms=rms(terms, axis=1).tolist(),
                    signed_gram=gram.tolist(),
                    common_residual_rms=rms(d["common_residual"][sl]).tolist(),
                    literal_residual_rms=rms(d["literal_residual"][sl]).tolist(),
                    identity_max=float(
                        abs(d["terms"].sum(0) - d["common_residual"]).max()
                    ),
                )
            )
        model, g = models()
        z0 = np.load(data / "initial-descriptors.npz")["z"][case["index"]]
        unforced_centers = gm.rollout(g, z0, np.arange(1, 41))
        # G predicts the unforced counterfactual only; actual post-event geometry is evaluator-only.
        physical.append(
            dict(
                **meta,
                unforced_max_absolute=abs(unforced_centers - z[0, 1:]).max(0).tolist(),
                combined_vs_unforced_max=abs(unforced_centers - z[3, 1:])
                .max(0)
                .tolist(),
                actual_event_motion_max=abs(z[3] - z[0]).max(0).tolist(),
                absolute_tolerances=[5e-5, 2e-5, 2e-6],
            )
        )
        for name in case["prefixes"]:
            identities[name + ".npz"] = digest(data / (name + ".npz"))
        if not np.array_equal(y, aa[:, :, 1, 2, :2] - aa[:, :, 0, 2, :2]):
            raise ValueError("Eight branch algebra")
    pm.save_json(out / "scores.json", scores)
    pm.save_json(out / "decomposition.json", decompositions)
    pm.save_json(out / "physical.json", physical)
    pm.save_json(out / "reference-identities.json", identities)
    result = {}
    for schedule in sorted({s["schedule"] for s in scores}):
        result[schedule] = {}
        for model in sorted({s["model"] for s in scores}):
            result[schedule][model] = {}
            for kind in sorted({s["kind"] for s in scores if s["model"] == model}):
                ss = [
                    s
                    for s in scores
                    if s["schedule"] == schedule
                    and s["model"] == model
                    and s["kind"] == kind
                ]
                result[schedule][model][kind] = {
                    "count": len(ss),
                    "failed": sum(s["pass"] is False for s in ss),
                    "unresolved": sum(not s["resolved"] for s in ss),
                    "worst": max(ss, key=lambda s: s["relative"] or 0),
                }
    pm.save_json(out / "summary.json", {"exposed": exposed, "results": result})
    budget.finish()


def exposed(out, args, budget):
    models()
    base = REPEAT / "fresh-01"
    starts = np.load(base / "boundary-50.npz")["states"]
    rows = load(base / "rows.json")
    schedules = [
        dict(
            name=c["schedule"]["name"],
            **{k: c["schedule"][k] for k in ["times", "amplitudes"]},
        )
        for c in load(base / "cases.json")
        if c["index"] == 0 and c["schedule"]["name"] in ["cancel", "timing"]
    ]
    sealed(out, starts, rows, schedules, budget)
    cases = [
        c
        for c in load(base / "cases.json")
        if c["schedule"]["name"] in ["cancel", "timing"]
    ]
    pm.save_json(out / "cases.json", cases)
    # Exposed source files copied byte-for-byte, with hashes; no historical rewrite.
    for name in sorted({name for c in cases for name in c["prefixes"]}):
        for suffix in [".npz", ".json"]:
            shutil.copyfile(base / (name + suffix), out / (name + suffix))
    pm.save_json(
        out / "inputs.json",
        {str(base / p.name): digest(p) for p in out.glob("reference-*.npz")},
    )
    evaluate(out, out, budget, True)


def pilot(out, args, budget):
    base = REPEAT / "fresh-01"
    starts = np.load(base / "boundary-50.npz")["states"][4:5]
    rows = [{"seed": 18103, "history": "none"}]
    sealed(out, starts, rows, SCHEDULES, budget)
    references(out, starts, rows, SCHEDULES, budget)
    evaluate(out, out, budget, True)


def refine(out, args, budget):
    from scipy.signal import resample

    if args.stage == "fresh-refine":
        base = ROOT / "fresh-01"
        scored = load(ROOT / "fresh-analysis-01/scores.json")
        if any(
            s["pass"] is not True for s in scored if s["model"].startswith("isolated")
        ):
            raise ValueError(
                "Isolated prerequisite does not qualify a composition refinement"
            )
        worst = max(
            (
                s
                for s in scored
                if s["model"] == "common"
                and s["kind"] == "D12"
                and s["schedule"] == "adverse-cancel"
            ),
            key=lambda s: s["relative"],
        )
        if worst["pass"] is not False:
            raise ValueError("No adverse common D12 miss to diagnose")
        case = next(
            c
            for c in load(base / "cases.json")
            if c["seed"] == worst["seed"]
            and c["history"] == worst["history"]
            and c["schedule"]["name"] == worst["schedule"]
        )
        initial_path = base / f"s{case['seed']}-initial.npz"
        variants = [
            ("halfdt", replace(CFG, dt=CFG.dt / 2)),
            ("doubleN", replace(CFG, n=CFG.n * 2)),
        ]
        scope = "Fresh adverse case diagnosed only after fixed predictions/reference; retained prepared initial, no refit"
    else:
        base = ROOT / "exposed-01"
        case = next(
            c
            for c in load(base / "cases.json")
            if c["seed"] == 18103
            and c["history"] == "none"
            and c["schedule"]["name"] == "timing"
        )
        initial_path = REPEAT / "fresh-01/s18103-initial.npz"
        variants = [
            ("halfdt", replace(CFG, dt=CFG.dt / 2)),
            ("doubleN", replace(CFG, n=CFG.n * 2)),
            ("quarterdt", replace(CFG, dt=CFG.dt / 4)),
        ]
        scope = "Exposed fixed prepared initial; full none-to50 and eight future branches; no fresh qualification"
    y, absolute, _ = truth(base, case)
    forecast_path = base / f"prediction-{case['index']}-{case['schedule']['name']}.npz"
    prediction = np.load(forecast_path)["common"]
    shutil.copyfile(forecast_path, out / "fixed-forecast.npz")
    pm.save_json(
        out / "selection.json",
        {
            "case": case,
            "fixed_forecast_sha256": digest(forecast_path),
            "initial_sha256": digest(initial_path),
            "scope": scope,
        },
    )
    initial = np.load(initial_path)["initial"]
    records = []
    for label, cfg in variants:
        sub = out / label
        sub.mkdir()
        budget.begin(label + "-matched-initial-to50")
        init = initial if cfg.n == CFG.n else resample(initial, cfg.n, axis=-1)
        state = unforced(
            initialize_written(init, case["history"], cfg), cfg, [0.0, 50.0], budget
        )[0][-1]
        save_npz(sub / "boundary-50.npz", state=state)
        budget.finish()
        references(
            sub,
            [state],
            [{"seed": case["seed"], "history": case["history"]}],
            [case["schedule"]],
            budget,
            cfg,
        )
        r, aa, _ = truth(sub, load(sub / "cases.json")[0])
        budget.begin(label + "-direct-and-propagated-numerics")
        for window, sl in WINDOWS.items():
            roundoff = (
                64
                * np.finfo(float).eps
                * abs(absolute[:, :, :, 2, :2]).max(axis=(0, 1, 2))
            )
            pair = np.maximum(5 * rms((r - y)[:, sl], axis=1), roundoff)
            for kind, weights in rm.CONTRASTS.items():
                target = rm.contrasts(r)[kind][sl]
                error = rms(
                    (rm.contrasts(prediction)[kind] - rm.contrasts(r)[kind])[sl]
                )
                records.append(
                    {
                        "refinement": label,
                        "window": window,
                        "kind": kind,
                        "magnitude": rms(target).tolist(),
                        "error_rms": error.tolist(),
                        "relative": (error / rms(target)).tolist(),
                        "matched_pair_bounds": pair.tolist(),
                        "propagated": (abs(np.array(weights)) @ pair).tolist(),
                        "direct": (5 * rms(rm.contrasts(r - y)[kind][sl])).tolist(),
                        "absolute_branch_difference": (
                            5 * rms((aa - absolute)[:, sl, :, 2, :2], axis=1)
                        ).tolist(),
                        "true_addition_relative": (
                            rms(rm.contrasts(r)["K12"][sl]) / rms(target)
                        ).tolist()
                        if kind == "D12"
                        else None,
                    }
                )
        save_npz(sub / "contrasts.npz", **rm.contrasts(r))
        budget.finish()
    pm.save_json(
        out / "refinement.json",
        {
            "records": records,
            "scope": scope,
            "rule": "5x RMS per matched R; triangle propagation; direct D12/K12 retained separately, never substituted for gate floors",
        },
    )


def provenance():
    """Search seed metadata and archive path ledgers, ignoring incidental float/hash digits."""
    wanted = set(
        SEEDS
        + [s + 100 * j for s in SEEDS for j in range(3)]
        + [s + 3000 for s in SEEDS]
    )
    hits, groups, checked = [], set(), []
    paths = list(Path("evidence").rglob("*.json"))
    paths += list(ATLAS.rglob("*.json")) + list(REPEAT.rglob("*.json"))

    def scan(value, path, key=""):
        if isinstance(value, dict):
            for k, v in value.items():
                scan(v, path, k)
        elif isinstance(value, list):
            for v in value:
                scan(v, path, key)
        elif isinstance(value, int) and ("seed" in key or "group" in key):
            groups.add(value)
            if value in wanted:
                hits.append({"path": str(path), "key": key, "value": value})
        elif isinstance(value, str) and key == "path":
            import re

            found = {
                int(s) for s in re.findall(r"(?:s|initial-|seed-)(\d+)(?=[-./])", value)
            }
            groups.update(found)
            if found & wanted:
                hits.append({"path": str(path), "value": value})

    for p in paths:
        scan(load(p), p)
        checked.append({"path": str(p), "sha256": digest(p)})
    if hits:
        raise ValueError(f"Reserved seed provenance overlap: {hits}")
    return {
        "seeds": SEEDS,
        "component_seeds": [s + 100 * j for s in SEEDS for j in range(3)],
        "perturbation_rng_seeds": [s + 3000 for s in SEEDS],
        "recorded_groups": sorted(groups),
        "checked": checked,
        "hits": hits,
        "boundary": "Current-main evidence indexes plus restored seed metadata; runtime source constants also inspected. No historical field descendants with these groups found.",
    }


def freeze(out, args, budget):
    budget.begin("freeze-unchanged-comparators-and-bounded-panel")
    models()
    for name in ["model.json", "G.json"]:
        shutil.copyfile(ATLAS / "frozen-01" / name, out / name)
    pm.save_json(out / "provenance.json", provenance())
    pilot_cost = load(ROOT / "pilot-01/budget.json")["cpu_seconds"]
    # Four schedule pilot; preparation upper reserve 25 CPU sec/group, plus 40% panel margin.
    projected = 6 * pilot_cost * 1.4 + 3 * 25 + 60
    used = budget.previous + time.process_time() - budget.start_cpu
    if used + projected + 180 > 1800:
        raise RuntimeError(
            "Full three-group panel cannot fit conservative budget; stop before outcomes"
        )
    pm.save_json(
        out / "freeze.json",
        {
            "source_revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "sources": source_hashes(),
            "models": {n: digest(out / n) for n in ["model.json", "G.json"]},
            "seeds": SEEDS,
            "histories": ["none", "odd04"],
            "schedules": SCHEDULES,
            "comparison": "common baseline r0a; isolated deltas r1-r0a/r2-r0b; literal and true addition separate",
            "addition": "R11_hat=r0a+(r1-r0a)+(r2-r0b)",
            "floors": load(REPEAT / "frozen-01/freeze.json")["floors"]["floors"],
            "isolated_floors": load(ATLAS / "frozen-01/freeze.json")["floors"],
            "R_limit": 0.02,
            "D_limit": 0.1,
            "windows": ["whole0-80", "late40-80"],
            "K_gate": None,
            "uncertainty": "No floors/relative targets changed. Fresh adverse 60/75/90 cancellation at worst common D12 gets matched halfdt/doubleN if failed and isolated pass, budget permitting. Margin classification uses 1.1x max inherited/propagated bound. No fit/correction.",
            "access": "Measure three centers once t50; known timing and amplitudes only; no later fields or true baseline",
            "exposure": "Fresh preparations at declared schedules; no unseen timing claim",
            "projected_cpu_seconds": projected,
            "remaining_cpu_seconds": 1800 - used,
            "stop": "After this cycle; no correction, hidden state or automatic successor",
        },
    )
    budget.finish()


def fresh(out, args, budget):
    from .organization_response import isolated_components
    from .source_receiver_relay import prepare

    frozen = ROOT / "frozen-01"
    f = load(frozen / "freeze.json")
    if f["sources"] != source_hashes() or any(
        digest(frozen / n) != h for n, h in f["models"].items()
    ):
        raise ValueError("Frozen runtime/model drift")
    starts, rows = [], []
    for seed in SEEDS:
        field = Field(CFG)
        initial = prepare(
            field, isolated_components(field, seed, budget), seed, -4.0, budget
        )
        save_npz(out / f"s{seed}-initial.npz", initial=initial)
        for history in f["histories"]:
            budget.begin(f"t50-{seed}-{history}")
            starts.append(
                unforced(
                    initialize_written(initial, history, CFG), CFG, [0.0, 50.0], budget
                )[0][-1]
            )
            rows.append({"seed": seed, "history": history})
            budget.finish()
    save_npz(out / "boundary-50.npz", states=starts)
    sealed(out, starts, rows, f["schedules"], budget, frozen)
    references(out, starts, rows, f["schedules"], budget)
    pm.save_json(
        out / "completion.json",
        {
            "completed_time_ns": time.time_ns(),
            "cases": 24,
            "computed_pairs": len(list(out.glob("reference-*.npz"))),
        },
    )


def fresh_refine(out, args, budget):
    refine(out, args, budget)


def report(out, args, budget):
    from .atlas_composition_reporting import finish

    finish(out, ROOT, budget)


def main():
    global ATLAS, REPEAT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=[
            "exposed",
            "pilot",
            "refine",
            "freeze",
            "fresh",
            "analyze",
            "fresh-refine",
            "report",
        ],
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--atlas", type=Path, default=ATLAS)
    parser.add_argument("--inherited", type=Path, default=REPEAT)
    args = parser.parse_args()
    ATLAS, REPEAT = args.atlas, args.inherited
    args.out.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    (args.out / "sources").mkdir()
    for p in Path(__file__).parent.glob("*.py"):
        shutil.copyfile(p, args.out / "sources" / p.name)
    previous = 30 + sum(load(p)["cpu_seconds"] for p in ROOT.glob("*/budget.json"))
    budget = HistoryBudget(previous=previous)
    status = "complete"
    try:
        if args.stage == "analyze":
            evaluate(args.out, ROOT / "fresh-01", budget)
        else:
            globals()[args.stage.replace("-", "_")](args.out, args, budget)
    except BaseException:
        status = "failed"
        raise
    finally:
        pm.save_json(
            args.out / "budget.json",
            dict(
                **budget.receipt(),
                status=status,
                stage=args.stage,
                source_revision=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                sources=sources,
                inspection_allowance=30,
            ),
        )


if __name__ == "__main__":
    if any(
        int(os.environ.get(k, "1")) != 1
        for k in ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS"]
    ):
        raise RuntimeError("Serial panels require one BLAS/OMP thread")
    main()
