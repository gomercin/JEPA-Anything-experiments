"""Bounded local-context calibration, reference runner and evidence receipts."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from . import event_age_operator_family as inherited
from . import present_state_model as pm
from .event_age_response import arrays
from .history_conditioned_transmission import HistoryBudget
from .hybrid_pair import save_npz_exclusive as save_npz
from .present_state_transmission import digest, load
from .repeated_intervention_state import source_hashes

ROOT = Path("work/mediated_patterns/event_age_operator_atlas")
NEW_AGES = (14, 17, 25)


def safe_output(path):
    if path.parent.resolve() != ROOT.resolve() or any(
        p.is_symlink() for p in [path, *path.parents]
    ):
        raise ValueError("Unique nonsymlink direct output child required")
    path.mkdir(exist_ok=False)


def panel(out, args, budget, pilot=False):
    rows, states, xx, yy, ledger = [], [], [], [], []
    reused = {}
    for row, start in inherited.sources(args.inherited):
        if pilot and (row["seed"], row["history"]) != (20101, "odd04"):
            continue
        for age in (14,) if pilot else NEW_AGES:
            i = len(rows)
            from_pilot = not pilot and (row["seed"], row["history"], age) == (
                20101,
                "odd04",
                14,
            )
            if from_pilot:
                base = ROOT / "pilot-01"
                saved = arrays(base / "states.npz")
                state, x = saved["states"][0], saved["x"][0]
                y = arrays(base / "targets.npz")["y"][0]
                paths = load(base / "references.json")[0]
                for f in base.iterdir():
                    if f.is_file():
                        reused[str(f)] = digest(f)
            else:
                budget.begin(f"boundary-{row['seed']}-{row['history']}-{age}")
                state = inherited.boundary(start, age, budget)
                x = pm.extract(state, feature_set="geometry")
                budget.finish()
                y = [
                    inherited.response_pair(
                        out, f"reference-{i}-{a:g}", state, age, a, budget
                    )
                    for a in (0.0, -0.02, 0.02)
                ]
                paths = [
                    str(out / f"reference-{i}-{a:g}.npz") for a in (0.0, -0.02, 0.02)
                ]
            rows.append(dict(**row, age=age))
            states.append(state)
            xx.append(x)
            yy.append(y)
            ledger.append(paths)
            print(f"completed {row['seed']} {row['history']} age{age}", flush=True)
    save_npz(out / "states.npz", states=states, x=xx)
    save_npz(out / "targets.npz", y=yy)
    pm.save_json(out / "rows.json", rows)
    pm.save_json(out / "references.json", ledger)
    pm.save_json(out / "reused-identities.json", reused)


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
            "curvature",
            "report",
            "details",
            "freeze",
            "fresh",
            "analyze",
            "retained",
            "repeated",
        ],
    )
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--data", type=Path)
    p.add_argument("--index", type=int, default=0)
    p.add_argument(
        "--family", choices=["linear", "quadratic", "pchip"], default="linear"
    )
    p.add_argument(
        "--inherited",
        type=Path,
        default=Path(
            "/private/tmp/jepa-event-age-inherited/work/mediated_patterns/repeated_intervention_state"
        ),
    )
    args = p.parse_args()
    if subprocess.check_output(
        ["git", "diff", "HEAD", "--", "experiments/mediated_patterns"], text=True
    ):
        raise RuntimeError("Commit sources and living note before scientific execution")
    if any(
        p.is_symlink() for p in [ROOT, *ROOT.parents, args.output, *args.output.parents]
    ):
        raise ValueError("Symlink output roots forbidden")
    ROOT.mkdir(parents=True, exist_ok=True)
    fd = os.open(ROOT / ".active", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    previous = 30 + sum(load(p)["cpu_seconds"] for p in ROOT.glob("*/budget.json"))
    fresh_stage = args.stage in ("fresh", "analyze", "retained", "repeated") or (
        args.stage == "refine"
        and args.data is not None
        and args.data.name == "fresh-01"
    )
    fresh_stage |= args.stage in ("report", "details") and (ROOT / "fresh-01").is_dir()
    if fresh_stage and not (ROOT / "frozen-01/freeze.json").is_file():
        (ROOT / ".active").unlink()
        raise ValueError("Qualified frozen contract required before fresh-stage budget")
    budget = HistoryBudget(previous, 1800 if fresh_stage else 1260)
    started, status = False, "FAILED"
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
                    Path(__file__).with_name("EVENT_AGE_OPERATOR_ATLAS.md")
                ),
            },
        )
        budget.check()
        if args.stage in ("pilot", "develop"):
            panel(args.output, args, budget, args.stage == "pilot")
        elif args.stage == "refine":
            inherited.refine(args.output, args, budget)
        elif args.stage in ("freeze", "fresh", "analyze", "retained", "repeated"):
            from . import event_atlas_validation as validation

            getattr(validation, args.stage)(args.output, args, budget)
        else:
            from . import event_atlas_analysis as analysis

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
