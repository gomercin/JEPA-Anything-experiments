"""Supplemental fresh worst-case refinement; does not change frozen scores/floors."""

from dataclasses import replace

import numpy as np
from scipy.signal import resample

from experiments.mediated_patterns import present_state_model as pm
from experiments.mediated_patterns.geometry_assay import initialize_written
from experiments.mediated_patterns.geometry_evolution import unforced
from experiments.mediated_patterns.history_conditioned_transmission import HistoryBudget
from experiments.mediated_patterns.intervention_aware_state import CFG, ROOT, case
from experiments.mediated_patterns.present_state_transmission import digest, load


def main():
    out = ROOT / "fresh-refine-01"
    out.mkdir(exist_ok=False)
    previous = 30 + sum(load(p)["cpu_seconds"] for p in ROOT.glob("*/budget.json"))
    budget = HistoryBudget(previous, 1800)
    status = "FAILED"
    try:
        pm.save_json(
            out / "protocol.json",
            {
                "source_sha256": digest(__file__),
                "target": "fresh worst selected mass D: seed16101 none, gap10; include withheld+.01",
                "prepared_initial_sha256": digest(ROOT / "fresh-01/s16101-initial.npz"),
            },
        )
        initial = np.load(ROOT / "fresh-01/s16101-initial.npz")["initial"]
        for label, cfg in [
            ("halfdt", replace(CFG, dt=CFG.dt / 2)),
            ("doubleN", replace(CFG, n=CFG.n * 2)),
        ]:
            target = out / label
            target.mkdir()
            budget.begin(label + "-prefix")
            init = initial if cfg.n == CFG.n else resample(initial, cfg.n, axis=-1)
            boundary = unforced(
                initialize_written(init, "none", cfg), cfg, [0.0, 50.0], budget
            )[0][-1]
            budget.finish()
            case(
                target,
                boundary,
                {"seed": 16101, "history": "none"},
                budget,
                config=cfg,
                gaps=(10,),
                amplitudes=(-0.02, 0.01),
            )
        estimates = []
        for a in [-0.02, 0.01]:
            name = f"s16101-none-a{a:g}-gap10.npz"
            base = np.load(ROOT / "fresh-01" / name)["y"]
            for label in ["halfdt", "doubleN"]:
                delta = np.load(out / label / name)["y"] - base
                estimates.append(
                    {
                        "amplitude": a,
                        "refinement": label,
                        "five_sum_R_rms": [
                            (
                                5
                                * np.sqrt(np.mean(delta[:, mask] ** 2, axis=1)).sum(
                                    axis=0
                                )
                            ).tolist()
                            for mask in [pm.TIMES >= 0, pm.TIMES >= 40]
                        ],
                        "D_difference_rms": np.sqrt(
                            np.mean((delta[1] - delta[0]) ** 2, axis=0)
                        ).tolist(),
                    }
                )
        pm.save_json(
            out / "refinement.json",
            {
                "records": estimates,
                "supplemental_max": np.max(
                    [x["five_sum_R_rms"] for x in estimates], axis=0
                ).tolist(),
                "frozen_gates_unchanged": True,
            },
        )
        status = "COMPLETE"
    finally:
        pm.save_json(out / "budget.json", dict(budget.receipt(), status=status))


if __name__ == "__main__":
    main()
