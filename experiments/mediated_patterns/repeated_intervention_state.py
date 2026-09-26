"""Opt-in bounded eight-branch evaluator; never imported by retained inference."""

import argparse
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import geometry_model as gm
from . import present_state_model as pm
from . import repeated_intervention_model as rm
from .geometry_assay import initialize_written
from .geometry_evolution import unforced
from .history_conditioned_transmission import HistoryBudget
from .hybrid_pair import save_npz_exclusive as save_npz
from .intervention_aware_state import jump_instrument
from .present_state_transmission import digest, load, reference
from .simulator import Field
from .source_receiver_relay import CFG, source_profile

ROOT = Path("work/mediated_patterns/repeated_intervention_state")
OLD = Path(
    "evidence/intervention-aware-state-2026-09-26/data/work/mediated_patterns/intervention_aware_state"
)
MODEL_FILE = OLD / "frozen-01/models.json"
MODEL_SHA = "9524c674fb9247b97284fc2bd8fa8082c5dc953d989f1db3e44f68a101e9d56a"
SELECTED_SHA = "167c1229f39a2f29fbda0c502c7fc434e2f6d83e42b42ef62e66c7eae54cd76c"
DEV, FRESH = [16101, 16102, 16103], [18101, 18102, 18103]
SCHEDULES = [
    {"name": "same", "times": [10, 30], "amplitudes": [0.01, 0.01]},
    {"name": "cancel", "times": [10, 30], "amplitudes": [0.02, -0.02]},
    {"name": "reverse", "times": [10, 30], "amplitudes": [-0.02, 0.02]},
]
WITHHELD = {"name": "timing", "times": [10, 25], "amplitudes": [0.02, -0.02]}


def model():
    if digest(MODEL_FILE) != MODEL_SHA:
        raise ValueError("Original artifact changed")
    m = load(MODEL_FILE)["transient4-readout"]
    if gm.identity(m) != SELECTED_SHA:
        raise ValueError("Selected model changed")
    return m


def eight_responses(absolute):
    """Axes prefix[00,10,01,11], h, probe[0,1], site[A,B,C], readout."""
    a = np.asarray(absolute)
    if a.shape != (4, 161, 2, 3, 3):
        raise ValueError("Eight logical branch shape required")
    return a[:, :, 1, 2, :2] - a[:, :, 0, 2, :2]


def source_hashes():
    return {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}


def predictions(out, starts, rows, schedules, budget):
    budget.begin("seal-all-prefix-predictions")
    z = np.array([pm.extract(s, feature_set="geometry") for s in starts])
    save_npz(out / "initial-descriptors.npz", z=z)
    pm.save_json(out / "rows.json", rows)
    for i, z0 in enumerate(z):
        for s in schedules:
            trajectories, responses = [], []
            for j, flags in enumerate(rm.PREFIXES):
                amplitudes = [
                    a * b for a, b in zip(s["amplitudes"], flags, strict=True)
                ]

                def checkpoint(state, i=i, name=s["name"], j=j):
                    state.checkpoint(
                        out / f"checkpoint-{i}-{name}-{j}-t{state.flow.steps}.json"
                    )

                tr, y = rm.forecast(
                    model(), z0, s["times"], amplitudes, checkpoint=checkpoint
                )
                trajectories.append(tr)
                responses.append(y)
            save_npz(
                out / f"prediction-{i}-{s['name']}.npz",
                trajectory=trajectories,
                y=responses,
            )
    pm.save_json(
        out / "seal.json",
        {
            "predictions": {
                p.name: digest(p) for p in sorted(out.glob("prediction-*.npz"))
            },
            "checkpoints": {
                p.name: digest(p) for p in sorted(out.glob("checkpoint-*.json"))
            },
            "initialization_sha256": digest(out / "initial-descriptors.npz"),
            "model_sha256": SELECTED_SHA,
            "order": "All independently initialized prefix forecasts saved before any future reference arrays",
        },
    )
    budget.finish()


def prefix_reference(out, name, initial, times, amplitudes, budget, config=CFG):
    state = initial.copy()
    trajectory, events, screens = [pm.extract(state, config.length, "geometry")], [], []
    previous = 0
    for t, a in [*zip(times, amplitudes, strict=True), (40, 0.0)]:
        budget.begin(name + f"-advance-{t}")
        states, checks = unforced(
            state, config, np.arange(t - previous + 1, dtype=float), budget
        )
        trajectory.extend(pm.extract(x, config.length, "geometry") for x in states[1:])
        state = states[-1]
        screens.extend(checks)
        if t != 40:
            state, info = jump_instrument(state, a, config)
            events.append(dict(elapsed=t, **info))
            trajectory[-1] = pm.extract(state, config.length, "geometry")
        previous = t
        budget.finish()
    budget.begin(name + "-final-probe-pair")
    arrays, checks = reference(state, config, budget)
    screens.extend(checks)
    save_npz(out / f"{name}.npz", **arrays, z=trajectory)
    pm.save_json(
        out / f"{name}.json",
        {
            "events": events,
            "regime": screens,
            "qualified": all(x["qualified"] for x in screens),
        },
    )
    if not all(x["qualified"] for x in screens):
        raise RuntimeError("Physical regime failed; preserve failed run")
    budget.finish()


def references(out, starts, rows, schedules, budget, config=CFG):
    cases = []
    for i, (initial, row) in enumerate(zip(starts, rows, strict=True)):
        cache = {}
        for s in schedules:
            names = []
            for flags in rm.PREFIXES:
                aa = tuple(a * b for a, b in zip(s["amplitudes"], flags, strict=True))
                # Retain boundaries in cache identity, including sham slots.
                key = (tuple(s["times"]), aa)
                if key not in cache:
                    name = f"reference-{i}-{len(cache)}"
                    prefix_reference(out, name, initial, s["times"], aa, budget, config)
                    cache[key] = name
                names.append(cache[key])
            cases.append(dict(**row, index=i, schedule=s, prefixes=names))
    pm.save_json(out / "cases.json", cases)


def panel(out, args, budget):
    if args.stage == "fresh":
        from .organization_response import isolated_components
        from .source_receiver_relay import prepare

        frozen = load(args.freeze / "freeze.json")
        if (
            frozen["sources"] != source_hashes()
            or frozen["selected_sha256"] != SELECTED_SHA
        ):
            raise ValueError("Frozen source/model mismatch")
        if set(FRESH) & set(DEV) or frozen["fresh_seeds"] != FRESH:
            raise ValueError("Grouped exposure split changed")
        starts, rows = [], []
        for seed in FRESH:
            f = Field(CFG)
            initial = prepare(
                f, isolated_components(f, seed, budget), seed, -4.0, budget
            )
            save_npz(out / f"s{seed}-initial.npz", initial=initial)
            for history in ["none", "odd04"]:
                budget.begin(f"prepare-{seed}-{history}")
                state = unforced(
                    initialize_written(initial, history, CFG), CFG, [0.0, 50.0], budget
                )[0][-1]
                starts.append(state)
                rows.append({"seed": seed, "history": history})
                budget.finish()
        save_npz(out / "boundary-50.npz", states=starts)
        schedules = frozen["schedules"]
    else:
        source = OLD / "fresh-01/boundary-50.npz"
        starts = np.load(source, allow_pickle=False)["states"]
        rows = load(OLD / "fresh-01/rows.json")
        pm.save_json(
            out / "inputs.json",
            {str(source): digest(source), str(MODEL_FILE): digest(MODEL_FILE)},
        )
        if args.stage == "pilot":
            starts, rows = starts[1:2], rows[1:2]
        schedules = SCHEDULES
    predictions(out, starts, rows, schedules, budget)
    references(out, starts, rows, schedules, budget)


def algebra(out, args, budget):
    from .intervention_model import State

    budget.begin("zero-gap-instrument-and-accumulator")
    state = np.load(OLD / "fresh-01/boundary-50.npz")["states"][1]
    f, z = Field(CFG), pm.extract(state, feature_set="geometry")
    q = source_profile(f)
    records = []
    for a, b in [(0.01, 0.01), (0.02, -0.02), (-0.02, 0.02)]:
        twice = state.copy()
        twice[0] += a * q
        twice[0] += b * q
        once = state.copy()
        once[0] += (a + b) * q
        ss, one = State(model(), z), State(model(), z)
        ss.event(a)
        ss.event(b)
        one.event(a + b)
        records.append(
            {
                "amplitudes": [a, b],
                "field_max_difference": float(abs(twice - once).max()),
                "model_vector_difference": (rm.vector(ss) - rm.vector(one)).tolist(),
                "model_response_difference_rms": np.sqrt(
                    np.mean((ss.response() - one.response()) ** 2, axis=0)
                ).tolist(),
                "p_d": ss.response_memory.tolist(),
                "p_squared_cross_term": 2 * (a / 0.02) * (b / 0.02),
            }
        )
    pm.save_json(out / "algebra.json", records)
    budget.finish()


def refine(out, args, budget):
    from scipy.signal import resample

    budget.begin("refinement-initial")
    initial = np.load(OLD / "fresh-01/s16101-initial.npz")["initial"]
    budget.finish()
    for label, config in [
        ("halfdt", replace(CFG, dt=CFG.dt / 2)),
        ("doubleN", replace(CFG, n=CFG.n * 2)),
    ]:
        sub = out / label
        sub.mkdir()
        budget.begin(label + "-write-to-50")
        init = initial if config.n == CFG.n else resample(initial, config.n, axis=-1)
        state = unforced(
            initialize_written(init, "odd04", config), config, [0.0, 50.0], budget
        )[0][-1]
        budget.finish()
        references(
            sub,
            [state],
            [{"seed": 16101, "history": "odd04"}],
            [SCHEDULES[1]],
            budget,
            config,
        )
    from .repeated_intervention_analysis import refinement

    refinement(out, args.data, budget)


def freeze(out, args, budget):
    from .repeated_intervention_analysis import WINDOWS, rms, truth

    scales = {window: np.zeros(2) for window in WINDOWS}
    development = ROOT / "develop-01"
    for case in load(development / "cases.json"):
        y, _, _ = truth(development, case)
        for window, sl in WINDOWS.items():
            scales[window] = np.maximum(
                scales[window],
                np.max([rms((y[1] - y[0])[sl]), rms((y[2] - y[0])[sl])], axis=0),
            )
    pm.save_json(out / "model.json", model())
    pm.save_json(
        out / "freeze.json",
        {
            "selected": "unchanged-nine",
            "selected_sha256": SELECTED_SHA,
            "original_file_sha256": MODEL_SHA,
            "sources": source_hashes(),
            "development_seeds": DEV,
            "fresh_seeds": FRESH,
            "schedules": [*SCHEDULES, WITHHELD],
            "t0": 50,
            "probe_time": 90,
            "initial_measurement": "three centers; six auxiliary zeros",
            "runtime_step": 1,
            "floors": load(args.data / "refinement.json"),
            "R_limit": 0.02,
            "D_limit": 0.1,
            "physical_tolerances": [5e-5, 2e-5, 2e-6],
            "no_fit": True,
            "development_single_event_scale": {
                k: v.tolist() for k, v in scales.items()
            },
            "development_cases_sha256": digest(development / "cases.json"),
            "withheld": "second event at75 for +.02,-.02 only",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "algebra",
            "pilot",
            "develop",
            "refine",
            "freeze",
            "fresh",
            "analyze",
            "repair",
            "separate-squares",
        ],
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--freeze", type=Path)
    args = parser.parse_args()
    if args.output.parent.resolve() != ROOT.resolve() or any(
        p.is_symlink() for p in [args.output, *args.output.parents]
    ):
        raise ValueError("Unique nonsymlink direct output child required")
    if subprocess.check_output(
        ["git", "diff", "HEAD", "--", "experiments/mediated_patterns"], text=True
    ):
        raise RuntimeError("Commit code and prospective note before science")
    ROOT.mkdir(parents=True, exist_ok=True)
    fd = os.open(ROOT / ".active", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    previous = 30 + sum(load(p)["cpu_seconds"] for p in ROOT.glob("*/budget.json"))
    budget = HistoryBudget(
        previous, 1800 if args.stage in ("fresh", "analyze") else 1200
    )
    started, status = False, "FAILED"
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
                "model_sha256": SELECTED_SHA,
                "note_sha256": digest(
                    Path(__file__).with_name("REPEATED_INTERVENTION_STATE.md")
                ),
            },
        )
        model()
        budget.check()
        if args.stage in ("pilot", "develop", "fresh"):
            panel(args.output, args, budget)
        elif args.stage == "separate-squares":
            from .repeated_intervention_repair import separate_squares

            separate_squares(args.output, args, budget)
        elif args.stage == "repair":
            from .repeated_intervention_repair import run

            run(args.output, args, budget)
        elif args.stage == "analyze":
            from .repeated_intervention_analysis import analyze

            analyze(args.output, args, budget)
        else:
            globals()[args.stage](args.output, args, budget)
        status = "COMPLETE"
    finally:
        if started:
            pm.save_json(
                args.output / "budget.json", dict(budget.receipt(), status=status)
            )
        (ROOT / ".active").unlink()


if __name__ == "__main__":
    main()
