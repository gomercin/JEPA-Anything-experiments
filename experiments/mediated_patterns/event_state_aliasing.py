"""Bounded fixed-age present-state assay; references stay outside the predictor."""

import argparse
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import resample

from . import event_state_model as sm
from . import present_state_model as pm
from .event_age_response import WINDOWS, arrays, paired, rms
from .geometry_evolution import development_sources, unforced
from .history_conditioned_transmission import HistoryBudget
from .hybrid_pair import save_npz_exclusive as save_npz
from .intervention_aware_state import jump_instrument
from .present_state_transmission import digest, load, reference
from .repeated_intervention_state import source_hashes
from .simulator import Field
from .source_receiver_relay import CFG, source_profile

ROOT = Path("work/mediated_patterns/event_state_aliasing")
AGE = Path("work/mediated_patterns/event_age_response")
FRESH = [24101, 24102, 24103]


def descriptor(state, config=CFG):
    return sm.extract(state, source_profile(Field(config)), config.length)


def boundary_sources(inherited):
    starts = []
    rows = []
    inputs = {}
    for row, p, index in development_sources():
        starts.append(pm.load_checkpoint(p, 50.0, index))
        rows.append(dict(**row, source=str(p), source_index=index))
        inputs[str(p)] = digest(p)
    sources = [
        Path(
            "evidence/geometry-evolution-2026-09-26/data/work/mediated_patterns/geometry_evolution/fresh-01"
        ),
        Path(
            "evidence/intervention-aware-state-2026-09-26/data/work/mediated_patterns/intervention_aware_state/fresh-01"
        ),
        inherited / "fresh-01",
        AGE / "fresh-01",
    ]
    for base in sources:
        states = arrays(base / "boundary-50.npz")["states"]
        metadata = load(base / "rows.json")
        inputs[str(base / "boundary-50.npz")] = digest(base / "boundary-50.npz")
        inputs[str(base / "rows.json")] = digest(base / "rows.json")
        for index, (row, state) in enumerate(zip(metadata, states, strict=True)):
            starts.append(state)
            rows.append(
                dict(**row, source=str(base / "boundary-50.npz"), source_index=index)
            )
    if len({(r["seed"], r["history"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate initial histories")
    return np.array(starts), rows, inputs


def reach_boundary(state, budget):
    # Retain historical t60 sham boundary exactly, not a new event or field law.
    for steps in [10, 12]:
        state = unforced(state, CFG, np.arange(steps + 1, dtype=float), budget)[0][-1]
    return state


def boundaries(out, args, budget):
    starts, rows, inputs = boundary_sources(args.inherited)
    if args.stage == "pilot":
        starts, rows = starts[:1], rows[:1]
    states = []
    descriptors = []
    uncertainty = []
    for state, row in zip(starts, rows, strict=True):
        budget.begin(f"boundary-{row['seed']}-{row['history']}")
        at72 = reach_boundary(state, budget)
        states.append(at72)
        x = descriptor(at72)
        descriptors.append(x)
        high = descriptor(resample(at72, 2 * CFG.n, axis=-1), replace(CFG, n=2 * CFG.n))
        uncertainty.append(5 * abs(high - x))
        budget.finish()
    save_npz(out / "boundary-72.npz", states=states)
    save_npz(out / "initial-50.npz", states=starts)
    save_npz(out / "descriptors.npz", x=descriptors, extraction_uncertainty=uncertainty)
    pm.save_json(out / "rows.json", rows)
    pm.save_json(out / "inputs.json", inputs)
    if args.stage != "pilot":
        selection = sm.choose_pairs(
            descriptors, [r["seed"] for r in rows], np.max(uncertainty, axis=0)[:3]
        )
        for r in selection["candidate_pairs"]:
            i, j = r["indices"]
            r["field_l2_difference"] = float(
                np.linalg.norm(states[i] - states[j]) * np.sqrt(CFG.length / CFG.n)
            )
            r["descriptor_difference"] = (
                np.array(descriptors[i]) - descriptors[j]
            ).tolist()
        selection.update(
            descriptors_sha256=digest(out / "descriptors.npz"),
            rows_sha256=digest(out / "rows.json"),
            order="current descriptors only; before opening future reference responses",
        )
        pm.save_json(out / "selection.json", selection)


def response_pair(out, name, initial, amplitude, budget, config=CFG):
    budget.begin(name + "-event-and-gap")
    state, event = jump_instrument(initial, amplitude, config)
    f = Field(config)
    q = source_profile(f)
    x = descriptor(initial, config)
    from .measurements import regions

    w = regions(f.x, pm.ANCHORS)[0]
    dx = config.length / config.n
    weighted = float(np.sum(w * (state[0] ** 2 - initial[0] ** 2)) * dx * config.source)
    formula = float(
        config.source * (2 * amplitude * x[3] + amplitude**2 * np.sum(w * q * q) * dx)
    )
    states, checks = unforced(state, config, np.arange(19, dtype=float), budget)
    budget.finish()
    budget.begin(name + "-probe-pair")
    data, screens = reference(states[-1], config, budget)
    checks.extend(screens)
    save_npz(
        out / (name + ".npz"),
        **data,
        z=np.array([pm.extract(s, config.length, "geometry") for s in states]),
    )
    pm.save_json(
        out / (name + ".json"),
        {
            "event": event,
            "weighted_source_jump": weighted,
            "overlap_formula": formula,
            "overlap_error": abs(weighted - formula),
            "qualified": all(s["qualified"] for s in checks),
            "regime": checks,
        },
    )
    if not all(s["qualified"] for s in checks):
        raise RuntimeError("Physical regime failure retained")
    budget.finish()
    return data["response"]


def references(out, args, budget):
    base = args.data
    selection = load(base / "selection.json")
    if digest(base / "descriptors.npz") != selection["descriptors_sha256"]:
        raise ValueError("Response-blind descriptor seal changed")
    rows = load(base / "rows.json")
    states = arrays(base / "boundary-72.npz")["states"]
    x = arrays(base / "descriptors.npz")["x"]
    ys = []
    sources = []
    oldrows = load(AGE / "fresh-01/rows.json")
    for i, (row, state) in enumerate(zip(rows, states, strict=True)):
        y = []
        for a in [0.0, -0.02, 0.02]:
            if row["seed"] in [20101, 20102, 20103]:
                index = next(
                    j
                    for j, r in enumerate(oldrows)
                    if r["seed"] == row["seed"] and r["history"] == row["history"]
                )
                p = AGE / "fresh-01" / f"reference-{index}-18-{a:g}.npz"
                info = load(p.with_suffix(".json"))["events"][1]
                if (
                    max(abs(np.array(info["before"]) - x[i, :3])) > 1e-12
                    or abs(info["I0"][0] - x[i, 3]) > 1e-12
                ):
                    raise ValueError("Old event-boundary identity mismatch")
                y.append(paired(p)["response"])
                sources.append(
                    {
                        "index": i,
                        "a": a,
                        "path": str(p),
                        "sha256": digest(p),
                        "reused": True,
                    }
                )
            else:
                name = f"reference-{i}-{a:g}"
                y.append(response_pair(out, name, state, a, budget))
                sources.append(
                    {
                        "index": i,
                        "a": a,
                        "path": str(out / (name + ".npz")),
                        "sha256": digest(out / (name + ".npz")),
                        "reused": False,
                    }
                )
        ys.append(y)
    save_npz(out / "targets.npz", y=ys)
    pm.save_json(out / "references.json", sources)
    pm.save_json(out / "selection-copy.json", selection)


def refine(out, args, budget):
    base = args.data
    selection = load(base / "selection.json")
    pair = selection["candidate_pairs"][0]
    indices = pair["indices"] if pair["near"] else pair["indices"][:1]
    rows = load(base / "rows.json")
    states = arrays(base / "boundary-72.npz")["states"]
    targets = arrays(ROOT / "responses-01/targets.npz")["y"]
    refledger = load(ROOT / "responses-01/references.json")
    floors = {}
    records = []
    for i in indices:
        absolute = np.array(
            [
                paired(
                    Path(
                        next(
                            r["path"]
                            for r in refledger
                            if r["index"] == i and r["a"] == a
                        )
                    )
                )["absolute"]
                for a in [0.0, -0.02, 0.02]
            ]
        )
        y = targets[i]
        for label, cfg in [
            ("halfdt", replace(CFG, dt=CFG.dt / 2)),
            ("doubleN", replace(CFG, n=CFG.n * 2)),
        ]:
            sub = out / f"{i}-{label}"
            sub.mkdir()
            state = states[i] if cfg.n == CFG.n else resample(states[i], cfg.n, axis=-1)
            r = np.array(
                [
                    response_pair(sub, f"reference-{a:g}", state, a, budget, cfg)
                    for a in [0.0, -0.02, 0.02]
                ]
            )
            refined_abs = np.array(
                [
                    paired(sub / f"reference-{a:g}.npz")["absolute"]
                    for a in [0.0, -0.02, 0.02]
                ]
            )
            for window, sl in WINDOWS.items():
                roundoff = (
                    64
                    * np.finfo(float).eps
                    * abs(absolute[:, :, :, 2, :2]).max(axis=(0, 1, 2))
                )
                err = np.maximum(5 * rms((r - y)[:, sl], axis=1), roundoff)
                abserr = 5 * rms((refined_abs - absolute)[:, sl, :, 2, :2], axis=1)
                for j in [1, 2]:
                    for kind, bound in [
                        ("R_without", err[0]),
                        ("R_after", err[j]),
                        ("D1", err[0] + err[j]),
                    ]:
                        key = kind + "/" + window
                        floors[key] = np.maximum(floors.get(key, np.zeros(2)), bound)
                    records.append(
                        {
                            "index": i,
                            "history": rows[i]["history"],
                            "seed": rows[i]["seed"],
                            "refinement": label,
                            "window": window,
                            "a": [0.0, -0.02, 0.02][j],
                            "pair_uncertainty": err[[0, j]].tolist(),
                            "D1_bound": (err[0] + err[j]).tolist(),
                            "direct_D1": (
                                5 * rms(((r[j] - r[0]) - (y[j] - y[0]))[sl])
                            ).tolist(),
                            "uncorrelated_absolute_bound": abserr[[0, j]]
                            .sum(axis=(0, 1))
                            .tolist(),
                        }
                    )
    pm.save_json(
        out / "refinement.json",
        {
            "indices": indices,
            "floors": {k: v.tolist() for k, v in floors.items()},
            "records": records,
            "scope": "matched future response from declared event-boundary initial field; upstream preparation not requalified",
        },
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        required=True,
        choices=[
            "pilot",
            "boundaries",
            "response-pilot",
            "references",
            "refine",
            "fit",
            "freeze",
            "fresh",
            "analyze",
            "pairs",
            "retention",
            "retain_freeze",
            "retain_fresh",
            "retain_analyze",
        ],
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inherited", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--freeze", type=Path)
    p.add_argument("--feature-set", choices=list(sm.SETS), default="centers")
    p.add_argument("--kernel", action="store_true")
    args = p.parse_args()
    if args.output.parent.resolve() != ROOT.resolve() or any(
        p.is_symlink() for p in [args.output, *args.output.parents]
    ):
        raise ValueError("Unique nonsymlink direct output child required")
    if subprocess.check_output(
        ["git", "diff", "HEAD", "--", "experiments/mediated_patterns"], text=True
    ):
        raise RuntimeError("Commit scientific sources and note before science")
    ROOT.mkdir(parents=True, exist_ok=True)
    fd = os.open(ROOT / ".active", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    previous = 30 + sum(load(p)["cpu_seconds"] for p in ROOT.glob("*/budget.json"))
    budget = HistoryBudget(
        previous,
        1800
        if args.stage in ["fresh", "analyze", "retain_fresh", "retain_analyze"]
        else 1200,
    )
    started = False
    status = "FAILED"
    try:
        args.output.mkdir(exist_ok=False)
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
                    Path(__file__).with_name("EVENT_STATE_ALIASING.md")
                ),
            },
        )
        budget.check()
        if args.stage in ["pilot", "boundaries"]:
            boundaries(args.output, args, budget)
        elif args.stage == "response-pilot":
            state = arrays(args.data / "boundary-72.npz")["states"][0]
            response_pair(args.output, "pilot-pair", state, 0.0, budget)
        elif args.stage in ["references", "refine"]:
            globals()[args.stage](args.output, args, budget)
        else:
            from . import event_state_analysis as sa

            getattr(sa, args.stage)(args.output, args, budget)
        status = "COMPLETE"
    finally:
        if started:
            pm.save_json(
                args.output / "budget.json", dict(budget.receipt(), status=status)
            )
        (ROOT / ".active").unlink()


if __name__ == "__main__":
    main()
