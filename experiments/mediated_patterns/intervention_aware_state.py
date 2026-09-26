"""Budgeted evaluator for actual conditioning events; never imported by inference."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import present_state_model as pm
from .geometry_evolution import F_FILE, development_sources, frozen_response, unforced
from .history_conditioned_transmission import HistoryBudget, event
from .hybrid_pair import save_npz_exclusive
from .measurements import regions
from .present_state_transmission import digest, load, reference
from .simulator import Field
from .source_receiver_relay import CFG, source_profile

ROOT = Path("work/mediated_patterns/intervention_aware_state")
G_FILE = Path(
    "evidence/geometry-evolution-2026-09-26/data/work/mediated_patterns/geometry_evolution/frozen-01/models.json"
)
G_SHA = "4b42006337f4e27c1149211e724ffeef022e9b6391c46aee27d96ecb9c038477"
DEV = [8101, 10101, 10102, 10103]
FRESH = [16101, 16102, 16103]


def originals():
    if digest(G_FILE) != G_SHA:
        raise ValueError("Original G changed")
    return load(G_FILE), frozen_response()


def jump_instrument(state, amplitude, config=CFG):
    f = Field(config)
    q = source_profile(f)
    w = regions(f.x, pm.ANCHORS)
    ell = f.x[None] - pm.ANCHORS[:, None]
    dx = config.length / config.n
    m = (w * state[0] ** 2).sum(axis=1) * dx
    z = pm.extract(state, config.length, "geometry")
    i0 = (w * state[0] * q).sum(axis=1) * dx
    i1 = (w * ell * state[0] * q).sum(axis=1) * dx
    q0 = (w * q * q).sum(axis=1) * dx
    q1 = (w * ell * q * q).sum(axis=1) * dx
    after, info = event(f, state, q, amplitude)
    exact = (m * z + 2 * amplitude * i1 + amplitude**2 * q1) / (
        m + 2 * amplitude * i0 + amplitude**2 * q0
    )
    measured = pm.extract(after, config.length, "geometry")
    info.update(
        M=m.tolist(),
        I0=i0.tolist(),
        I1=i1.tolist(),
        Q0=q0.tolist(),
        Q1=q1.tolist(),
        before=z.tolist(),
        after=measured.tolist(),
        formula=exact.tolist(),
        identity_max=float(abs(exact - measured).max()),
        field_l2_before=float(np.linalg.norm(state) * np.sqrt(dx)),
        field_l2_after=float(np.linalg.norm(after) * np.sqrt(dx)),
    )
    return after, info


def four_responses(absolute):
    """Branch axis order Y00,Y10,Y01,Y11; never subtract Y11-Y00 as R."""
    return np.stack([absolute[:, 2] - absolute[:, 0], absolute[:, 3] - absolute[:, 1]])


def case(
    out,
    state,
    row,
    budget,
    config=CFG,
    delay=10,
    gaps=(10, 30),
    amplitudes=(0.02, -0.02),
):
    """All t2 branches originate from the same conditioning-only continuation."""
    budget.begin("prefix")
    before = unforced(state, config, [0.0, float(delay)], budget)[0][-1]
    budget.finish()
    Field(config)
    y_without, absolute_without = {}, {}
    for a in (0.0, *amplitudes):
        budget.begin(f"conditioning-{a}")
        after, info = jump_instrument(before, a, config)
        times = np.arange(max(gaps) + 1, dtype=float)
        states, screens = unforced(after, config, times, budget)
        z = np.array([pm.extract(s, config.length, "geometry") for s in states])
        features = np.array([pm.extract(s, config.length) for s in states])
        name = f"s{row['seed']}-{row['history']}-a{a:g}"
        save_npz_exclusive(
            out / f"{name}-geometry.npz", time=times, z=z, features=features
        )
        pm.save_json(
            out / f"{name}-event.json",
            dict(
                **info, qualified=all(s["qualified"] for s in screens), regime=screens
            ),
        )
        budget.finish()
        for gap in gaps:
            budget.begin(f"probe-{a}-{gap}")
            arrays, screens = reference(states[gap], config, budget)
            if a == 0:
                y_without[gap] = arrays["response"]
                absolute_without[gap] = arrays["absolute"]
            else:
                # Each pair was stepped identically by reference; combine the four
                # absolute readouts before subtraction as an instrument cross-check.
                absolute = np.stack(
                    [
                        absolute_without[gap][:, 0],
                        arrays["absolute"][:, 0],
                        absolute_without[gap][:, 1],
                        arrays["absolute"][:, 1],
                    ],
                    axis=1,
                )
                ys = four_responses(absolute)[:, :, 2, :2]
                save_npz_exclusive(
                    out / f"{name}-gap{gap}.npz",
                    absolute=absolute,
                    y=ys,
                    z0=pm.extract(state, config.length, "geometry"),
                    z_before=info["before"],
                    z_after=info["after"],
                    z_later=z[gap],
                    time=pm.TIMES,
                )
            if not all(s["qualified"] for s in screens):
                raise RuntimeError("Regime failed")
            budget.finish()


def pilot(out, budget):
    row, path, index = next(
        v for v in development_sources() if v[0] == {"seed": 8101, "history": "odd04"}
    )
    state = pm.load_checkpoint(path, 50.0, index)
    pm.save_json(
        out / "inputs.json",
        {
            str(path): digest(path),
            str(F_FILE): digest(F_FILE),
            str(G_FILE): digest(G_FILE),
        },
    )
    case(out, state, row, budget)
    _, F = originals()
    summary = []
    for p in out.glob("*gap*.npz"):
        with np.load(p, allow_pickle=False) as d:
            r = d["y"]
            pred = pm.predict(F, d["z_later"][None])[0]
            baseline_z = np.load(out / "s8101-odd04-a0-geometry.npz")["z"][
                int(p.stem.split("gap")[1])
            ]
            r0hat = pm.predict(F, baseline_z[None])[0]
            summary.append(
                {
                    "case": p.name,
                    "r_rms": np.sqrt(np.mean(r * r, axis=1)).tolist(),
                    "D_rms": np.sqrt(np.mean((r[1] - r[0]) ** 2, axis=0)).tolist(),
                    "D_over_R": (
                        np.sqrt(np.mean((r[1] - r[0]) ** 2, axis=0))
                        / np.sqrt(np.mean(r[1] ** 2, axis=0))
                    ).tolist(),
                    "exact_F_R_relative": (
                        np.sqrt(np.mean((pred - r[1]) ** 2, axis=0))
                        / np.sqrt(np.mean(r[1] ** 2, axis=0))
                    ).tolist(),
                    "exact_F_D_relative": (
                        np.sqrt(np.mean((pred - r0hat - r[1] + r[0]) ** 2, axis=0))
                        / np.sqrt(np.mean((r[1] - r[0]) ** 2, axis=0))
                    ).tolist(),
                }
            )
    pm.save_json(out / "pilot.json", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True)
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
        raise RuntimeError("Commit scientific code and note before running")
    ROOT.mkdir(parents=True, exist_ok=True)
    fd = os.open(ROOT / ".active", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    previous = 30 + sum(load(p)["cpu_seconds"] for p in ROOT.glob("*/budget.json"))
    budget = HistoryBudget(
        previous, 1800 if args.stage in ("fresh", "analyze", "fresh-refine") else 1200
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
                "sources": {
                    p.name: digest(p) for p in Path(__file__).parent.glob("*.py")
                },
                "note_sha256": digest(
                    Path(__file__).with_name("INTERVENTION_AWARE_STATE.md")
                ),
                "original_G_sha256": digest(G_FILE),
                "original_F_sha256": digest(F_FILE),
            },
        )
        budget.check()
        if args.stage == "pilot":
            pilot(args.output, budget)
        else:
            from . import intervention_analysis as analysis

            getattr(analysis, args.stage.replace("-", "_"))(args.output, args, budget)
        status = "COMPLETE"
    finally:
        if started:
            pm.save_json(
                args.output / "budget.json", dict(budget.receipt(), status=status)
            )
        (ROOT / ".active").unlink()


if __name__ == "__main__":
    main()
