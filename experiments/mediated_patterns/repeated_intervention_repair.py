"""One composition-informed scalar readout repair, grouped by preparation."""

import json

import numpy as np

from . import intervention_model as im
from . import present_state_model as pm
from . import repeated_intervention_model as rm
from .hybrid_pair import save_npz_exclusive
from .present_state_transmission import load
from .repeated_intervention_analysis import rms, score, truth


def rescale(original, alpha):
    model = json.loads(json.dumps(original))
    coefficients = np.array(model["readout"]["coefficients"])
    # Feature order is [p,d,p²,p*d] outer [1,z_A,z_B,z_C].
    coefficients[8:] *= alpha
    model["readout"]["coefficients"] = coefficients.tolist()
    return model


def fit_alpha(examples, train_seeds, floor):
    """Train-only scalar LS on resolved mass K; no state/basis refitting."""
    pairs = [
        (x["delta"], x["target"])
        for x in examples
        if x["seed"] in train_seeds and rms(x["truth_k"])[0] > floor
    ]
    if not pairs:
        raise ValueError("No resolved training K")
    x = np.concatenate([p[0][:, 0] for p in pairs])
    y = np.concatenate([p[1][:, 0] for p in pairs])
    return float(x @ y / (x @ x))


def run(out, args, budget):
    from .intervention_analysis import records
    from .intervention_analysis import score as old_score
    from .repeated_intervention_state import DEV, OLD, ROOT, model

    budget.begin("grouped-single-scalar-readout-repair")
    original, data = model(), ROOT / "develop-01"
    cases = load(data / "cases.json")
    z0 = np.load(data / "initial-descriptors.npz")["z"]
    floors = load(ROOT / "refine-01/refinement.json")["floors"]
    examples = []
    for case in cases:
        s = case["schedule"]
        y, _, _ = truth(data, case)
        y1 = np.load(data / f"prediction-{case['index']}-{s['name']}.npz")["y"]
        y0 = []
        for flags in rm.PREFIXES:
            aa = [a * b for a, b in zip(s["amplitudes"], flags, strict=True)]
            y0.append(
                rm.forecast(rescale(original, 0), z0[case["index"]], s["times"], aa)[1]
            )
        y0 = np.array(y0)
        k0, k1, kt = (rm.contrasts(v)["K12"] for v in [y0, y1, y])
        examples.append(
            {
                "seed": case["seed"],
                "case": case,
                "y": y,
                "y0": y0,
                "y1": y1,
                "delta": k1 - k0,
                "target": kt - k0,
                "truth_k": kt,
            }
        )
    selected = fit_alpha(examples, DEV, floors["K12/whole"][0])
    models = {
        "original": original,
        "linear-only": rescale(original, 0),
        "selected": rescale(original, selected),
    }
    alphas = {"selected": selected, "linear-only": 0.0, "original": 1.0}
    for seed in DEV:
        alpha = fit_alpha(examples, set(DEV) - {seed}, floors["K12/whole"][0])
        alphas[f"held-{seed}"] = alpha
        models[f"held-{seed}"] = rescale(original, alpha)
    scores = []
    for i, e in enumerate(examples):
        arrays = {"truth": e["y"], "original": e["y1"], "linear_only": e["y0"]}
        case = e["case"]
        for name, alpha in alphas.items():
            if name.startswith("held-") and name != f"held-{e['seed']}":
                continue
            pred = e["y0"] + alpha * (e["y1"] - e["y0"])
            arrays[name] = pred
            scores.extend(
                dict(
                    seed=e["seed"],
                    history=case["history"],
                    schedule=case["schedule"]["name"],
                    model=name,
                    **s,
                )
                for s in score(e["y"], pred, floors)
            )
        save_npz_exclusive(out / f"case-{i}.npz", **arrays)
    old = []
    for row in records(OLD / "fresh-01"):
        baseline = im.forecast(models["selected"], row["z0"], 0.0, gaps=(row["gap"],))[
            1
        ][0]
        after = im.forecast(
            models["selected"], row["z0"], row["a"], gaps=(row["gap"],)
        )[1][0]
        old.extend(
            dict(
                seed=row["seed"],
                history=row["history"],
                amplitude=row["a"],
                gap=row["gap"],
                **s,
            )
            for s in old_score(
                row["y"],
                np.array([baseline, after]),
                load(OLD / "frozen-01/freeze.json")["response_floors"],
            )
        )
    pm.save_json(out / "models.json", models)
    pm.save_json(out / "alphas.json", alphas)
    pm.save_json(out / "scores.json", scores)
    pm.save_json(out / "one-event-regression.json", old)
    pm.save_json(
        out / "selection.json",
        {
            "family": "scale both quadratic response feature blocks by one fitted scalar; same nine state scalars",
            "fit": "least squares on numerically resolved whole-window mass K, train seeds only; no preprocessing learned",
            "alphas": alphas,
            "new_variables": 0,
            "changed_runtime_coefficients": 64,
            "grouped_main_failures": sum(
                s["pass"] is False for s in scores if s["model"].startswith("held-")
            ),
            "selected_main_failures": sum(
                s["pass"] is False for s in scores if s["model"] == "selected"
            ),
            "one_event_failures": sum(not s["passed"] for s in old),
            "one_event_worst": max(s["relative"] for s in old),
        },
    )
    budget.finish()


def separate_squares(out, args, budget):
    from .intervention_analysis import records
    from .intervention_analysis import score as old_score
    from .repeated_intervention_state import OLD, ROOT, model

    budget.begin("separate-square-summary-repair")
    candidate = model()
    candidate["repeated_extension"] = "separate-squared-amplitude"
    data = ROOT / "develop-01"
    z0 = np.load(data / "initial-descriptors.npz")["z"]
    floors = load(ROOT / "refine-01/refinement.json")["floors"]
    scores = []
    for i, case in enumerate(load(data / "cases.json")):
        s = case["schedule"]
        y, _, _ = truth(data, case)
        ys = []
        trajectories = []
        for flags in rm.PREFIXES:
            aa = [a * b for a, b in zip(s["amplitudes"], flags, strict=True)]
            tr, p = rm.forecast(candidate, z0[case["index"]], s["times"], aa)
            ys.append(p)
            trajectories.append(tr)
        save_npz_exclusive(out / f"case-{i}.npz", y=ys, trajectory=trajectories)
        scores.extend(
            dict(
                seed=case["seed"],
                history=case["history"],
                schedule=s["name"],
                model="separate-squares",
                **v,
            )
            for v in score(y, np.array(ys), floors)
        )
    old = []
    equivalence = []
    for row in records(OLD / "fresh-01"):
        yy = []
        for amp in [0.0, row["a"]]:
            state = rm.state_class(candidate)(candidate, row["z0"])
            state.advance(10)
            state.event(amp)
            state.advance(row["gap"])
            yy.append(state.response())
            original = im.forecast(model(), row["z0"], amp, gaps=(row["gap"],))[1][0]
            equivalence.append(float(abs(original - yy[-1]).max()))
        old.extend(
            dict(
                seed=row["seed"],
                history=row["history"],
                amplitude=row["a"],
                gap=row["gap"],
                **v,
            )
            for v in old_score(
                row["y"],
                np.array(yy),
                load(OLD / "frozen-01/freeze.json")["response_floors"],
            )
        )
    pm.save_json(out / "model.json", candidate)
    pm.save_json(out / "scores.json", scores)
    pm.save_json(out / "one-event-regression.json", old)
    pm.save_json(
        out / "selection.json",
        {
            "family": "separate squared-amplitude summaries; no coefficient fitting",
            "main_failures": sum(v["pass"] is False for v in scores),
            "one_event_failures": sum(not v["passed"] for v in old),
            "one_event_original_max_difference": max(equivalence),
            "scientific_scalars": 11,
            "active_coefficient_values": 1636,
            "initialization": "three measured centers and eight zero auxiliary coordinates",
            "exposure": "composition-informed after the original and scalar-rescale development failures; all descendants grouped",
        },
    )
    budget.finish()
