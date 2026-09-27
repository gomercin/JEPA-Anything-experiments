"""Bounded contextual response assay; nonlinear references are evaluator-only."""

import argparse
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import resample

from . import present_state_model as pm
from .event_age_response import WINDOWS, arrays, paired, rms, score
from .geometry_evolution import unforced
from .history_conditioned_transmission import HistoryBudget
from .hybrid_pair import save_npz_exclusive as save_npz
from .intervention_aware_state import jump_instrument
from .present_state_transmission import digest, load, reference
from .repeated_intervention_state import source_hashes
from .source_receiver_relay import CFG

ROOT = Path("work/mediated_patterns/event_age_operator_family")
OLD = Path("work/mediated_patterns/event_state_aliasing")
AGE = Path("work/mediated_patterns/event_age_response")
AGES = [10, 15, 18, 20, 30]
FRESH = [26101, 26102, 26103]


def sources(inherited):
    result = []
    for folder in [
        inherited / "fresh-01",
        AGE / "fresh-01",
        OLD / "fresh-01",
        OLD / "retained-fresh-01",
    ]:
        states = arrays(folder / "boundary-50.npz")["states"]
        for i, (row, state) in enumerate(
            zip(load(folder / "rows.json"), states, strict=True)
        ):
            result.append(
                (
                    {
                        "seed": row["seed"],
                        "history": row["history"],
                        "source": str(folder / "boundary-50.npz"),
                        "source_index": i,
                        "source_sha256": digest(folder / "boundary-50.npz"),
                    },
                    state,
                )
            )
    return result


def boundary(state, age, budget, config=CFG):
    for steps in [10, 30 - age]:
        state = unforced(state, config, np.arange(steps + 1, dtype=float), budget)[0][
            -1
        ]
    return state


def response_pair(out, name, state, age, a, budget, config=CFG):
    budget.begin(name + "-gap")
    hit, event = jump_instrument(state, a, config)
    tr, checks = unforced(hit, config, np.arange(age + 1, dtype=float), budget)
    budget.finish()
    budget.begin(name + "-probe")
    data, screens = reference(tr[-1], config, budget)
    checks.extend(screens)
    save_npz(
        out / (name + ".npz"),
        **data,
        z=[pm.extract(s, config.length, "geometry") for s in tr],
    )
    pm.save_json(
        out / (name + ".json"),
        {
            "age": age,
            "a": a,
            "event": event,
            "regime": checks,
            "qualified": all(c["qualified"] for c in checks),
        },
    )
    if not all(c["qualified"] for c in checks):
        raise RuntimeError("Regime failure retained")
    budget.finish()
    return data["response"]


def old_paths(row, age, inherited):
    seed, hist = row["seed"], row["history"]
    if age == 18:
        if seed < 24000:
            folder = OLD / "boundaries-01"
            i = next(
                i
                for i, r in enumerate(load(folder / "rows.json"))
                if (r["seed"], r["history"]) == (seed, hist)
            )
            ledger = load(OLD / "responses-01/references.json")
            return [
                Path(next(r["path"] for r in ledger if r["index"] == i and r["a"] == a))
                for a in [0.0, -0.02, 0.02]
            ]
        folder = OLD / ("fresh-01" if seed < 24110 else "retained-fresh-01")
        i = next(
            i
            for i, r in enumerate(load(folder / "rows.json"))
            if (r["seed"], r["history"]) == (seed, hist)
        )
        return [folder / f"reference-{i}-{a:g}.npz" for a in [0.0, -0.02, 0.02]]
    if 20101 <= seed <= 20103 and age in [10, 30]:
        i = row["source_index"]
        return [
            AGE / "fresh-01" / f"reference-{i}-{age}-{a:g}.npz"
            for a in [0.0, -0.02, 0.02]
        ]
    if 18101 <= seed <= 18103:
        folder = inherited / "fresh-01"
        i = row["source_index"]
        result = []
        for a in [-0.02, 0.02]:
            if age == 20 or (age == 15 and a > 0):
                paths = [
                    AGE / "develop-01" / f"reference-{i}-{age}-{v:g}.npz"
                    for v in [0.0, a]
                ]
                if age == 15:
                    c = next(
                        c
                        for c in load(folder / "cases.json")
                        if c["index"] == i and c["schedule"]["name"] == "timing"
                    )
                    paths[0] = folder / (c["prefixes"][0] + ".npz")
            else:
                sched = (
                    "timing"
                    if age == 15
                    else ("cancel" if (age == 30) == (a > 0) else "reverse")
                )
                c = next(
                    c
                    for c in load(folder / "cases.json")
                    if c["index"] == i and c["schedule"]["name"] == sched
                )
                paths = [
                    folder / (c["prefixes"][j] + ".npz")
                    for j in [0, 1 if age == 30 else 2]
                ]
            if not result:
                result.extend(paths)
            else:
                if not np.array_equal(
                    paired(result[0])["response"], paired(paths[0])["response"]
                ):
                    raise ValueError("Historical baseline partitions differ")
                result.append(paths[1])
        return result
    return None


def pilot(out, args, budget):
    row, start = next(
        (r, s)
        for r, s in sources(args.inherited)
        if r["seed"] == 20101 and r["history"] == "odd04"
    )
    budget.begin("pilot-boundary")
    state = boundary(start, 20, budget)
    save_npz(
        out / "states.npz",
        states=[state],
        x=[pm.extract(state, feature_set="geometry")],
    )
    pm.save_json(out / "rows.json", [dict(**row, age=20)])
    budget.finish()
    y = [
        [
            response_pair(out, f"reference-0-{a:g}", state, 20, a, budget)
            for a in [0.0, -0.02, 0.02]
        ]
    ]
    save_npz(out / "targets.npz", y=y)
    pm.save_json(
        out / "references.json",
        [[str(out / f"reference-0-{a:g}.npz") for a in [0.0, -0.02, 0.02]]],
    )


def develop(out, args, budget):
    rows, states, x = [], [], []
    for row, start in sources(args.inherited):
        for age in AGES:
            budget.begin(f"boundary-{row['seed']}-{row['history']}-{age}")
            state = boundary(start, age, budget)
            rows.append(dict(**row, age=age))
            states.append(state)
            x.append(pm.extract(state, feature_set="geometry"))
            budget.finish()
    save_npz(out / "states.npz", states=states, x=x)
    pm.save_json(out / "rows.json", rows)
    yy, ledger, identities = [], [], {}
    for i, (row, state) in enumerate(zip(rows, states, strict=True)):
        age = row["age"]
        paths = old_paths(row, age, args.inherited)
        if row["seed"] == 20101 and row["history"] == "odd04" and age == 20:
            paths = [
                ROOT / "pilot-01" / f"reference-0-{a:g}.npz" for a in [0.0, -0.02, 0.02]
            ]
        if paths:
            yy.append([paired(p)["response"] for p in paths])
            for p in paths:
                meta = load(p.with_suffix(".json"))
                event = meta.get("event")
                if event is None:
                    event = meta["events"][0 if age == 30 else 1]
                if not np.allclose(event["before"], x[i], rtol=0, atol=1e-12):
                    raise ValueError("Reused reference pre-event centers differ")
                identities[str(p)] = digest(p)
                identities[str(p.with_suffix(".json"))] = digest(p.with_suffix(".json"))
        else:
            yy.append(
                [
                    response_pair(out, f"reference-{i}-{a:g}", state, age, a, budget)
                    for a in [0.0, -0.02, 0.02]
                ]
            )
            paths = [out / f"reference-{i}-{a:g}.npz" for a in [0.0, -0.02, 0.02]]
        ledger.append([str(p) for p in paths])
    save_npz(out / "targets.npz", y=yy)
    pm.save_json(out / "references.json", ledger)
    pm.save_json(out / "reused-identities.json", identities)


def refine(out, args, budget):
    base = args.data
    rows = load(base / "rows.json")
    i = args.index
    row = rows[i]
    state = arrays(base / "states.npz")["states"][i]
    y = arrays(base / "targets.npz")["y"][i]
    paths = load(base / "references.json")[i]
    absolute = np.array([paired(Path(p))["absolute"] for p in paths])
    floors, records = {}, []
    for label, cfg in [
        ("halfdt", replace(CFG, dt=CFG.dt / 2)),
        ("doubleN", replace(CFG, n=CFG.n * 2)),
    ]:
        sub = out / label
        sub.mkdir()
        st = state if cfg.n == CFG.n else resample(state, cfg.n, axis=-1)
        r = np.array(
            [
                response_pair(sub, f"reference-{a:g}", st, row["age"], a, budget, cfg)
                for a in [0.0, -0.02, 0.02]
            ]
        )
        refined = np.array(
            [
                paired(sub / f"reference-{a:g}.npz")["absolute"]
                for a in [0.0, -0.02, 0.02]
            ]
        )
        for win, sl in WINDOWS.items():
            err = np.maximum(
                5 * rms((r - y)[:, sl], axis=1),
                64
                * np.finfo(float).eps
                * abs(absolute[:, :, :, 2, :2]).max(axis=(0, 1, 2)),
            )
            for j in [1, 2]:
                for kind, bound in [
                    ("R_without", err[0]),
                    ("R_after", err[j]),
                    ("D1", err[0] + err[j]),
                ]:
                    key = kind + "/" + win
                    floors[key] = np.maximum(floors.get(key, np.zeros(2)), bound)
                records.append(
                    {
                        "refinement": label,
                        "window": win,
                        "a": [0.0, -0.02, 0.02][j],
                        "R_bounds": err[[0, j]].tolist(),
                        "D1_bound": (err[0] + err[j]).tolist(),
                        "direct_D1": (
                            5 * rms(((r[j] - r[0]) - (y[j] - y[0]))[sl])
                        ).tolist(),
                        "absolute_Y_bound": (
                            5
                            * rms((refined - absolute)[:, sl, :, 2, :2], axis=1)[
                                [0, j]
                            ].sum(axis=(0, 1))
                        ).tolist(),
                    }
                )
    pm.save_json(
        out / "refinement.json",
        {
            "row": row,
            "index": i,
            "floors": {k: v.tolist() for k, v in floors.items()},
            "records": records,
            "scope": "Matched branches from fixed current snapshot; upstream preparation not requalified",
        },
    )


def scores(y, p, rows, floors):
    return [
        dict(
            **g, index=i, seed=row["seed"], history=row["history"], age=row["age"], a=a
        )
        for i, row in enumerate(rows)
        for j, a in [(1, -0.02), (2, 0.02)]
        for g in score(y[i, [0, j]], p[i, [0, j]], floors)
    ]


def safe_output(path):
    if path.parent.resolve() != ROOT.resolve() or any(
        p.is_symlink() for p in [path, *path.parents]
    ):
        raise ValueError("Unique nonsymlink direct output child required")
    path.mkdir(exist_ok=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        required=True,
        choices=[
            "pilot",
            "develop",
            "refine",
            "fit",
            "freeze",
            "fresh",
            "analyze",
            "operators",
            "diagnose",
            "repeated",
            "report",
        ],
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument(
        "--inherited",
        type=Path,
        default=Path(
            "/private/tmp/jepa-event-age-inherited/work/mediated_patterns/repeated_intervention_state"
        ),
    )
    p.add_argument("--data", type=Path)
    p.add_argument("--index", type=int, default=0)
    p.add_argument(
        "--family", choices=["blind", "polynomial", "spline"], default="polynomial"
    )
    p.add_argument("--rank", type=int, default=4)
    args = p.parse_args()
    if subprocess.check_output(
        ["git", "diff", "HEAD", "--", "experiments/mediated_patterns"], text=True
    ):
        raise RuntimeError("Commit scientific sources and living note before science")
    if any(
        p.is_symlink() for p in [ROOT, *ROOT.parents, args.output, *args.output.parents]
    ):
        raise ValueError("Symlink output roots forbidden before lock creation")
    ROOT.mkdir(parents=True, exist_ok=True)
    fd = os.open(ROOT / ".active", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    previous = 30 + sum(load(p)["cpu_seconds"] for p in ROOT.glob("*/budget.json"))
    budget = HistoryBudget(
        previous,
        1800
        if args.stage in ["fresh", "analyze", "repeated", "report"]
        or (args.stage == "refine" and args.data and "fresh" in args.data.name)
        else 1200,
    )
    started = False
    status = "FAILED"
    try:
        safe_output(args.output)
        started = True
        pm.save_json(
            args.output / "protocol.json",
            {
                "command": sys.argv,
                "revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "sources": source_hashes(),
                "note_sha256": digest(
                    Path(__file__).with_name("EVENT_AGE_OPERATOR_FAMILY.md")
                ),
            },
        )
        budget.check()
        if args.stage in ["pilot", "develop", "refine"]:
            globals()[args.stage](args.output, args, budget)
        else:
            from . import event_operator_analysis as analysis

            getattr(analysis, args.stage)(args.output, args, budget)
        status = "COMPLETE"
    finally:
        if started:
            pm.save_json(
                args.output / "budget.json", dict(budget.receipt(), status=status)
            )
        (ROOT / ".active").unlink()


if __name__ == "__main__":
    main()
