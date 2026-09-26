"""Adverse fresh timing refinement only: no fitting, new gates or fresh relabeling."""

import argparse
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.signal import resample

from experiments.mediated_patterns import present_state_model as pm
from experiments.mediated_patterns import repeated_intervention_model as rm
from experiments.mediated_patterns.geometry_assay import initialize_written
from experiments.mediated_patterns.geometry_evolution import unforced
from experiments.mediated_patterns.history_conditioned_transmission import HistoryBudget
from experiments.mediated_patterns.present_state_transmission import digest, load
from experiments.mediated_patterns.repeated_intervention_analysis import (
    WINDOWS,
    rms,
    truth,
)
from experiments.mediated_patterns.repeated_intervention_state import ROOT, references
from experiments.mediated_patterns.source_receiver_relay import CFG


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    if out.parent.resolve() != ROOT.resolve() or any(
        p.is_symlink() for p in [out, *out.parents]
    ):
        raise ValueError("Unique nonsymlink output child required")
    fd = os.open(ROOT / ".active", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    audit = load(ROOT / "runtime-audit.json")
    previous = (
        30
        + sum(load(p)["cpu_seconds"] for p in ROOT.glob("*/budget.json"))
        + audit["own_cpu_seconds"]
        + audit["child_cpu_seconds"]
    )
    budget = HistoryBudget(previous, 1800)
    started, status = False, "FAILED"
    try:
        out.mkdir(exist_ok=False)
        started = True
        case = next(
            c
            for c in load(ROOT / "fresh-01/cases.json")
            if c["seed"] == 18102
            and c["history"] == "odd04"
            and c["schedule"]["name"] == "timing"
        )
        pm.save_json(
            out / "protocol.json",
            {
                "command": sys.argv,
                "revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "script_sha256": digest(Path(__file__)),
                "case": case,
                "selection": "largest fresh D2|1 mass relative error; no model or threshold change",
            },
        )
        initial = np.load(ROOT / "fresh-01/s18102-initial.npz")["initial"]
        y, absolute, z = truth(ROOT / "fresh-01", case)
        records = []
        for label, config in [
            ("halfdt", replace(CFG, dt=CFG.dt / 2)),
            ("doubleN", replace(CFG, n=CFG.n * 2)),
        ]:
            sub = out / label
            sub.mkdir()
            budget.begin(label + "-full-history")
            init = (
                initial if config.n == CFG.n else resample(initial, config.n, axis=-1)
            )
            state = unforced(
                initialize_written(init, "odd04", config), config, [0.0, 50.0], budget
            )[0][-1]
            budget.finish()
            references(
                sub,
                [state],
                [{"seed": 18102, "history": "odd04"}],
                [case["schedule"]],
                budget,
                config,
            )
            yy, aa, zz = truth(sub, load(sub / "cases.json")[0])
            for window, sl in WINDOWS.items():
                pair = 5 * rms((yy - y)[:, sl], axis=1)
                roundoff = (
                    64
                    * np.finfo(float).eps
                    * abs(absolute[:, :, :, 2, :2]).max(axis=(0, 1, 2))
                )
                pair = np.maximum(pair, roundoff)
                raw = 5 * rms((aa - absolute)[:, sl, :, 2, :2], axis=1)
                for kind, weights in rm.CONTRASTS.items():
                    records.append(
                        {
                            "refinement": label,
                            "kind": kind,
                            "window": window,
                            "propagated": (abs(np.array(weights)) @ pair).tolist(),
                            "direct": (
                                5 * rms(rm.contrasts(yy - y)[kind][sl])
                            ).tolist(),
                            "absolute_branch_bound": (
                                abs(np.array(weights)) @ raw.sum(axis=1)
                            ).tolist(),
                            "coordinate_max": abs(zz - z).max(axis=(0, 1)).tolist(),
                        }
                    )
        pm.save_json(
            out / "qualification.json",
            {
                "records": records,
                "frozen_floors_changed": False,
                "fresh_panel_remains_failed": True,
            },
        )
        status = "COMPLETE"
    finally:
        if started:
            pm.save_json(out / "budget.json", dict(budget.receipt(), status=status))
        (ROOT / ".active").unlink()


if __name__ == "__main__":
    main()
