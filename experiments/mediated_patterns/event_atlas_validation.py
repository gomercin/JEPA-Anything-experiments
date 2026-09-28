"""Conditional fresh snapshot validation and separately gated retained diagnostics."""

from pathlib import Path

import numpy as np

from . import event_atlas_model as am
from . import event_operator_model as om
from . import geometry_model as gm
from . import present_state_model as pm
from .event_age_operator_atlas import ROOT
from .event_age_operator_family import boundary, response_pair, scores
from .event_age_response import arrays
from .event_atlas_analysis import combined, floors
from .event_operator_analysis import dataset
from .event_state_analysis import summary
from .hybrid_pair import save_npz_exclusive as save_npz
from .present_state_transmission import digest, load
from .repeated_intervention_state import source_hashes

FRESH = (26101, 26102, 26103)


def freeze(out, args, budget):
    budget.begin("freeze-local-atlas-and-fresh-contract")
    d = load(args.data / "decision.json")
    if (
        d["status"] != "DEVELOPMENT_QUALIFIED"
        or "coefficient" not in d["eligible_modes"]
    ):
        raise ValueError(
            "Complete coefficient-atlas development qualification required"
        )
    x, y, _rows, ages, groups = combined()
    if set(FRESH) & set(groups):
        raise ValueError("Reserved preparations already exposed")
    coarse = np.isin(ages, am.COARSE)
    rep = om.representation(x[coarse], y[coarse])
    model = am.fit(x, ages, y, am.REFINED, rep=rep)
    pm.save_json(out / "model.json", model)
    g = load(
        Path("work/mediated_patterns/event_state_aliasing/retained-frozen-01/G.json")
    )
    if (
        gm.identity(g)
        != "a9af96ddb21bcd90e92cb8cd51fa1371c95166949fec6ffe54c0e1367dfdded9"
    ):
        raise ValueError("Inherited G changed")
    pm.save_json(out / "G.json", g)
    pm.save_json(
        out / "freeze.json",
        {
            "family": d["family"],
            "mode": "coefficient",
            "nodes": model["nodes"],
            "neighbor_rule": "bracketing pair plus closest third; smaller third-age tie",
            "fresh_seeds": list(FRESH),
            "ages": [24, 25],
            "amplitudes": [-0.02, 0.02],
            "probe_boundary": 90,
            "floors": floors(),
            "models": {p.name: digest(p) for p in out.glob("*.json")},
            "sources": source_hashes(),
            "selection": str(args.data / "decision.json"),
            "selection_sha256": digest(args.data / "decision.json"),
            "development_groups": sorted({int(g) for g in groups}),
            "information": "Actual pre-event centers, amplitude and known age; no true baseline",
            "retained_plan": "Only after snapshot pass: unchanged G from saved t50, exposed diagnostic, no refit",
            "repeated_plan": "Only after snapshot pass: exposed independent addition, no repeated fitting",
        },
    )
    budget.finish()


def fresh(out, args, budget):
    from .geometry_assay import initialize_written
    from .geometry_evolution import unforced
    from .organization_response import isolated_components
    from .simulator import Field
    from .source_receiver_relay import CFG, prepare

    frozen = load(args.data / "freeze.json")
    if frozen["sources"] != source_hashes():
        raise ValueError("Frozen sources changed before fresh access")
    for name, sha in frozen["models"].items():
        if digest(args.data / name) != sha:
            raise ValueError("Frozen artifact changed")
    if set(FRESH) & set(frozen["development_groups"]):
        raise ValueError("Fresh preparation overlap")
    model = load(args.data / "model.json")
    starts, rows, states, x = [], [], [], []
    for seed in FRESH:
        f = Field(CFG)
        initial = prepare(f, isolated_components(f, seed, budget), seed, -4.0, budget)
        save_npz(out / f"initial-{seed}.npz", state=initial)
        for history in ("none", "odd04"):
            budget.begin(f"fresh-t50-{seed}-{history}")
            state50 = unforced(
                initialize_written(initial, history, CFG), CFG, [0.0, 50.0], budget
            )[0][-1]
            start_index = len(starts)
            starts.append(state50)
            budget.finish()
            for age in frozen["ages"]:
                budget.begin(f"fresh-current-{seed}-{history}-{age}")
                current = boundary(state50, age, budget)
                rows.append(
                    {
                        "seed": seed,
                        "history": history,
                        "age": age,
                        "start_index": start_index,
                    }
                )
                states.append(current)
                x.append(pm.extract(current, feature_set="geometry"))
                budget.finish()
    save_npz(out / "boundary-50.npz", states=starts)
    save_npz(out / "states.npz", states=states, x=x)
    pm.save_json(out / "rows.json", rows)
    ages = np.array([r["age"] for r in rows])
    budget.begin("seal-all-snapshot-predictions-before-conditioning-futures")
    for family in ("linear", frozen["family"]):
        save_npz(
            out / f"prediction-{family}.npz", y=am.predict_panel(model, x, ages, family)
        )
    pm.save_json(
        out / "seal.json",
        {
            "predictions": {p.name: digest(p) for p in out.glob("prediction-*.npz")},
            "model_sha256": digest(args.data / "model.json"),
            "freeze_sha256": digest(args.data / "freeze.json"),
            "current_states_sha256": digest(out / "states.npz"),
            "order": "All predictions persisted before any conditioning/probe future reference",
        },
    )
    budget.finish()
    yy, ledger = [], []
    for i, (row, state) in enumerate(zip(rows, states, strict=True)):
        yy.append(
            [
                response_pair(out, f"reference-{i}-{a:g}", state, row["age"], a, budget)
                for a in (0.0, -0.02, 0.02)
            ]
        )
        ledger.append(
            [str(out / f"reference-{i}-{a:g}.npz") for a in (0.0, -0.02, 0.02)]
        )
        print(
            f"fresh completed {row['seed']} {row['history']} age{row['age']}",
            flush=True,
        )
    save_npz(out / "targets.npz", y=yy)
    pm.save_json(out / "references.json", ledger)


def analyze(out, args, budget):
    budget.begin("frozen-snapshot-fresh-gates")
    _x, y, rows, _ages, _groups = dataset(args.data)
    frozen = load(ROOT / "frozen-01/freeze.json")
    seal = load(args.data / "seal.json")
    for name, sha in seal["predictions"].items():
        if digest(args.data / name) != sha:
            raise ValueError("Prediction seal changed")
    results = {}
    for family in ("linear", frozen["family"]):
        pred = arrays(args.data / f"prediction-{family}.npz")["y"]
        gates = scores(y, pred, rows, frozen["floors"])
        pm.save_json(out / f"{family}-scores.json", gates)
        results[family] = {
            str(a): summary([g for g in gates if g["age"] == a]) for a in frozen["ages"]
        }
    qualified = all(
        s["failed"] == 0 and s["unresolved"] == 0
        for s in results[frozen["family"]].values()
    )
    pm.save_json(
        out / "decision.json", {"snapshot_qualified": qualified, "results": results}
    )
    worst = max(
        (g for g in gates if g["age"] == 24 and g["kind"] == "D1"),
        key=lambda g: g["relative"],
    )
    pm.save_json(out / "adverse.json", worst)
    budget.finish()


def retained(out, args, budget):
    if not load(ROOT / "fresh-analysis-01/decision.json")["snapshot_qualified"]:
        raise ValueError("Snapshot prerequisite failed")
    budget.begin("unchanged-G-exposed-retained-diagnostic")
    frozen = load(ROOT / "frozen-01/freeze.json")
    model = load(ROOT / "frozen-01/model.json")
    g = load(ROOT / "frozen-01/G.json")
    starts = arrays(args.data / "boundary-50.npz")["states"]
    rows = load(args.data / "rows.json")
    zg, z0 = [], []
    for i, row in enumerate(rows):
        z = pm.extract(starts[row["start_index"]], feature_set="geometry")
        flow = gm.Continuation(g, z)
        flow.advance(10)
        flow.checkpoint(out / f"checkpoint-{i}.json")
        z0.append(z)
        zg.append(flow.advance(30 - row["age"]))
    ages = np.array([r["age"] for r in rows])
    pred = am.predict_panel(model, zg, ages, frozen["family"])
    save_npz(out / "predictions.npz", y=pred, z=zg, z0=z0)
    pm.save_json(
        out / "seal.json",
        {
            "prediction_sha256": digest(out / "predictions.npz"),
            "access": "Only t50 snapshots used before seal; later arrays below evaluator-only",
            "exposure": "Same snapshot-fresh preparations now exposed; no refit",
        },
    )
    x, y, _rows, _ages, _groups = dataset(args.data)
    gates = scores(y, pred, rows, frozen["floors"])
    pm.save_json(out / "scores.json", gates)
    err = np.asarray(zg) - x
    pm.save_json(
        out / "summary.json",
        {
            "exposed_diagnostic": True,
            "by_age": {
                str(a): summary([r for r in gates if r["age"] == a]) for a in (24, 25)
            },
            "geometry_max_absolute": abs(err).max(0).tolist(),
            "geometry_errors": err.tolist(),
            "true_motion": (x - z0).tolist(),
            "predicted_motion": (np.asarray(zg) - z0).tolist(),
            "old_geometry_failures": "Unchanged; this diagnostic is pre-event only",
        },
    )
    budget.finish()


def repeated(out, args, budget):
    """Conditional exposed independent addition; no paired-event fit or field solve."""
    from . import repeated_intervention_model as rm
    from .present_state_transmission import digest
    from .repeated_intervention_analysis import score as repeat_score
    from .repeated_intervention_analysis import truth

    decision = load(ROOT / "fresh-analysis-01/decision.json")
    if not decision["snapshot_qualified"]:
        raise ValueError("Fresh single-event snapshot gates must pass first")
    budget.begin("exposed-independent-addition-diagnostic")
    frozen = ROOT / "frozen-01"
    model = load(frozen / "model.json")
    family = load(frozen / "freeze.json")["family"]

    def query(z, amplitude, age):
        return am.predict(model, z, amplitude, age, family)

    g = load(frozen / "G.json")
    base = args.inherited / "fresh-01"
    z0 = arrays(base / "initial-descriptors.npz")["z"]
    cases = [
        c
        for c in load(base / "cases.json")
        if c["schedule"]["name"] in ["cancel", "timing"]
    ]
    predictions = []
    for k, c in enumerate(cases):
        time = c["schedule"]["times"]
        amps = c["schedule"]["amplitudes"]
        # Independent single-event histories share the unforced prefix; this
        # does not evolve the first physical conditioning through the second.
        z = gm.rollout(g, z0[c["index"]], time)
        r0a = query(z[0], 0.0, 40 - time[0])
        r1 = query(z[0], amps[0], 40 - time[0])
        r0b = query(z[1], 0.0, 40 - time[1])
        r2 = query(z[1], amps[1], 40 - time[1])
        raw = np.array([r0a, r1, r2, r1 + r2 - r0a])
        # Both no-event contexts describe the same final physical baseline.
        # Sum independently predicted contrasts on a single predicted baseline;
        # no true R00 is supplied. Keep raw addition and its baseline disagreement.
        centered = np.array([r0a, r1, r0a + (r2 - r0b), r1 + (r2 - r0b)])
        save_npz(
            out / f"prediction-{k}.npz",
            raw=raw,
            centered=centered,
            baseline_disagreement=r0b - r0a,
            components=[r0a, r1, r0b, r2],
        )
        predictions.append((raw, centered))
    pm.save_json(
        out / "seal.json",
        {
            "model_sha256": gm.identity(model),
            "G_sha256": gm.identity(g),
            "predictions": {p.name: digest(p) for p in out.glob("prediction-*.npz")},
            "order": "All predictions saved before opening exposed repeated future responses; no fitting",
        },
    )
    floors = load(args.inherited / "frozen-01/freeze.json")["floors"]["floors"]
    all_scores = []
    identities = {}
    for k, (c, (raw, centered)) in enumerate(zip(cases, predictions, strict=True)):
        y, _, _ = truth(base, c)
        for name in c["prefixes"]:
            p = base / (name + ".npz")
            identities[str(p)] = digest(p)
        for label, pred in [
            ("raw-addition", raw),
            ("common-baseline-addition", centered),
            (
                "true-single-event-addition",
                np.array([y[0], y[1], y[2], y[1] + y[2] - y[0]]),
            ),
        ]:
            rows = repeat_score(y, pred, floors)
            all_scores.extend(
                dict(
                    **r,
                    case=k,
                    seed=c["seed"],
                    history=c["history"],
                    schedule=c["schedule"]["name"],
                    model=label,
                )
                for r in rows
            )
            lhs = y[3] - pred[3]
            rhs = (
                rm.contrasts(y)["K12"]
                + ((y[1] + y[2] - y[0]) - (pred[1] + pred[2] - pred[0]))
                + ((pred[1] + pred[2] - pred[0]) - pred[3])
            )
            if not np.allclose(lhs, rhs, rtol=1e-12, atol=1e-21):
                raise ValueError("Signed error decomposition mismatch")
        comp = arrays(out / f"prediction-{k}.npz")["components"]
        from .event_age_response import score as isolated_score

        for j in [0, 1]:
            for r in isolated_score(
                y[[0, j + 1]],
                comp[2 * j : 2 * j + 2],
                load(frozen / "freeze.json")["floors"],
            ):
                all_scores.append(
                    dict(
                        **r,
                        case=k,
                        seed=c["seed"],
                        history=c["history"],
                        age=40 - c["schedule"]["times"][j],
                        a=c["schedule"]["amplitudes"][j],
                        schedule=c["schedule"]["name"],
                        model=f"isolated-component-{j + 1}",
                    )
                )
        save_npz(out / f"truth-{k}.npz", y=y)
    pm.save_json(out / "scores.json", all_scores)
    pm.save_json(out / "cases.json", cases)
    pm.save_json(out / "source-identities.json", identities)
    pm.save_json(
        out / "interpretation.json",
        {
            "exposed_only": True,
            "refitting": False,
            "comparisons": [
                "raw R10+R01−R00",
                "independently predicted D1+D2 on one predicted baseline",
                "evaluator true addition",
            ],
            "caveat": "Known-context single-event queries, not an event-updated autonomous state. K12 predicted zero by independent addition.",
        },
    )
    budget.finish()
