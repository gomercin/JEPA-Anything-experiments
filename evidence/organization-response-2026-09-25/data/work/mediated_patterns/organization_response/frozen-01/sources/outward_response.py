"""Add an actual exterior mediator readout to the same development preparations.

This follow-up reuses saved initial fields, never future fields, for simulation.
The fitted terminal map itself takes only separation and declared pulse amplitude.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .explore import start_panel, write_json
from .measurements import outward
from .reduced import HORIZONS, extract_inputs, features
from .simulator import Config, Field


def develop(development, output):
    out = start_panel(
        output,
        {
            "purpose": "outward mediator response, exterior sensor, same development runs",
            "development": str(development),
            "readout": "C-infinity mean m at B+12, support radius 2",
            "primary_horizon": 20,
            "horizons": HORIZONS,
            "frozen_tolerances": {"h20_nrmse": 0.15, "h20_max_absolute_error": 2e-6},
            "prediction": "terminal, equation-informed 10-feature separation/pulse response map",
            "fresh_plan": "seeds 101,202,303; separations 25,27,29,31; pulses +/-.075",
        },
    )
    start = time.perf_counter()
    data = json.loads((development / "results.json").read_text())["records"]
    f = Field(Config(**json.loads((development / "protocol.json").read_text())["config"]))
    rows = []
    for prep in sorted({r["preparation_id"] for r in data}):
        same = [r for r in data if r["preparation_id"] == prep]
        s = np.load(development / f"{prep}-initial.npz")["state"]
        _, base, _ = f.evolve(s, 50)
        for row in same:
            d, a = row["separation"], row["amplitude"]
            _, full, _ = f.evolve(f.pulse(s, -d / 2, a, width=8, relative=True, compact=True), 50)
            y = outward(f.x, full, d / 2) - outward(f.x, base, d / 2)
            rows.append(
                {
                    "preparation_id": prep,
                    "inputs": extract_inputs(row).tolist(),
                    "target": y[list(HORIZONS)].tolist(),
                }
            )
        print(prep, flush=True)
    design = features(np.array([r["inputs"] for r in rows]))
    y = np.array([r["target"] for r in rows])
    coeff, _, rank, sv = np.linalg.lstsq(design, y, rcond=None)
    write_json(
        out / "model.json",
        {
            "coefficients": coeff.tolist(),
            "rank": int(rank),
            "condition": float(sv[0] / sv[-1]),
            "horizons": HORIZONS,
            "training_preparations": sorted({r["preparation_id"] for r in rows}),
            "contract": "one-shot exterior response; no recursive state claim",
        },
    )
    write_json(
        out / "results.json",
        {
            "records": rows,
            "seconds": time.perf_counter() - start,
            "fit_rmse": np.sqrt(np.mean((design @ coeff - y) ** 2, axis=0)).tolist(),
            "signal_rms": np.sqrt(np.mean(y * y, axis=0)).tolist(),
        },
    )


def evaluate(panel, model, output):
    out = start_panel(
        output,
        {
            "purpose": "evaluate frozen exterior readout model",
            "panel": str(panel),
            "model": str(model),
        },
    )
    data = json.loads((panel / "results.json").read_text())["records"]
    fit = json.loads(model.read_text())
    assert not set(fit["training_preparations"]) & {r["preparation_id"] for r in data}
    design = features(np.array([extract_inputs(r) for r in data]))
    prediction = design @ np.array(fit["coefficients"])
    truth = np.array([[r["outward_response"][h] for h in HORIZONS] for r in data])
    independent = np.array([[r["independent_outward_response"][h] for h in HORIZONS] for r in data])
    result = {
        "truth": truth.tolist(),
        "prediction": prediction.tolist(),
        "independent": independent.tolist(),
        "horizons": HORIZONS,
        "preparation_ids": [r["preparation_id"] for r in data],
        "signal_rms": np.sqrt(np.mean(truth**2, axis=0)).tolist(),
    }
    for name, p in (("compact", prediction), ("independent", independent)):
        result[name + "_score"] = {
            "rmse": np.sqrt(np.mean((truth - p) ** 2, axis=0)).tolist(),
            "max_abs": np.max(abs(truth - p), axis=0).tolist(),
            "nrmse": np.sqrt(
                np.mean((truth - p) ** 2, axis=0) / np.mean(truth**2, axis=0)
            ).tolist(),
        }
    result["h20_pass"] = bool(
        result["compact_score"]["nrmse"][1] <= 0.15
        and result["compact_score"]["max_abs"][1] <= 2e-6
    )
    write_json(out / "results.json", result)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--development", type=Path)
    p.add_argument("--evaluate", type=Path)
    p.add_argument("--model", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.development:
        develop(args.development, args.output)
    else:
        evaluate(args.evaluate, args.model, args.output)
