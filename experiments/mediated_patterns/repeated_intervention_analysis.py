"""Read-only scoring and numerical qualification of sealed prefix forecasts."""

import numpy as np

from . import present_state_model as pm
from . import repeated_intervention_model as rm
from .present_state_transmission import digest, load

WINDOWS = {"whole": slice(None), "late": slice(80, None)}


def rms(x, axis=0):
    return np.sqrt(np.mean(np.asarray(x) ** 2, axis=axis))


def truth(data, case):
    from .repeated_intervention_state import eight_responses

    arrays = [
        np.load(data / f"{name}.npz", allow_pickle=False) for name in case["prefixes"]
    ]
    absolute = np.array([a["absolute"] for a in arrays])
    y = eight_responses(absolute)
    if not np.array_equal(y, np.array([a["response"] for a in arrays])):
        raise ValueError("Matched branch subtraction mismatch")
    return y, absolute, np.array([a["z"] for a in arrays])


def refinement(out, base, budget):
    budget.begin("matched-refinement-analysis")
    case = next(
        c for c in load(base / "cases.json") if c["schedule"]["name"] == "cancel"
    )
    y, absolute, z = truth(base, case)
    records, floors, coordinate = [], {}, []
    for label in ["halfdt", "doubleN"]:
        r, aa, zz = truth(out / label, load(out / label / "cases.json")[0])
        coordinate.append(5 * abs(zz - z).max(axis=(0, 1)))
        for window, sl in WINDOWS.items():
            # Each R is a separately matched probe/sham pair. Never use a
            # fortuitously smaller direct multi-prefix contrast as its floor.
            roundoff = (
                64
                * np.finfo(float).eps
                * abs(absolute[:, :, :, 2, :2]).max(axis=(0, 1, 2))
            )
            pair = np.maximum(5 * rms((r - y)[:, sl], axis=1), roundoff)
            branches = 5 * rms((aa - absolute)[:, sl, :, 2, :2], axis=1)
            for kind, weights in rm.CONTRASTS.items():
                weights = np.asarray(weights)
                bound = abs(weights) @ pair
                key = f"{kind}/{window}"
                floors[key] = np.maximum(floors.get(key, np.zeros(2)), bound)
                direct = 5 * rms(rm.contrasts(r - y)[kind][sl])
                records.append(
                    {
                        "refinement": label,
                        "window": window,
                        "kind": kind,
                        "matched_pair_uncertainty": pair.tolist(),
                        "floor": bound.tolist(),
                        "direct_contrast": direct.tolist(),
                        "absolute_branch_differences": branches.tolist(),
                        "uncorrelated_absolute_branch_bound": (
                            abs(weights) @ branches.sum(axis=1)
                        ).tolist(),
                    }
                )
    pm.save_json(
        out / "refinement.json",
        {
            "floors": {k: v.tolist() for k, v in floors.items()},
            "coordinate_floors": np.max(coordinate, axis=0).tolist(),
            "records": records,
            "rule": "5x matched per-prefix R refinement RMS, propagated by triangle inequality; max halfdt/doubleN and 64eps absolute readout; direct contrast not substituted",
        },
    )
    budget.finish()


def score(y, pred, floors):
    result = []
    for kind, target in rm.contrasts(y).items():
        prediction = rm.contrasts(pred)[kind]
        for window, sl in WINDOWS.items():
            magnitude, error = rms(target[sl]), rms((prediction - target)[sl])
            for o, name in enumerate(["mass", "moment"]):
                floor = floors.get(f"{kind}/{window}", [0.0, 0.0])[o]
                resolved = bool(magnitude[o] > floor)
                limit = (
                    0.02 if kind.startswith("R") else (0.1 if kind != "K12" else None)
                )
                relative = float(error[o] / magnitude[o]) if magnitude[o] else None
                peak = int(np.argmax(abs(target[sl, o]))) + (
                    80 if window == "late" else 0
                )
                pred_peak = int(np.argmax(abs(prediction[sl, o]))) + (
                    80 if window == "late" else 0
                )
                result.append(
                    {
                        "kind": kind,
                        "window": window,
                        "output": name,
                        "magnitude": float(magnitude[o]),
                        "error_rms": float(error[o]),
                        "error_max": float(abs(prediction - target)[sl, o].max()),
                        "relative": relative,
                        "floor": float(floor),
                        "resolved": resolved,
                        "pass": bool(relative <= limit)
                        if resolved and limit is not None
                        else None,
                        "true_peak": float(target[peak, o]),
                        "true_peak_h": float(pm.TIMES[peak]),
                        "predicted_peak": float(prediction[pred_peak, o]),
                        "predicted_peak_h": float(pm.TIMES[pred_peak]),
                    }
                )
    return result


def analyze(out, args, budget):
    budget.begin("score-eight-branch-panel")
    data = args.data
    floors = (
        load(args.freeze / "freeze.json")["floors"]["floors"] if args.freeze else {}
    )
    seal = load(data / "seal.json")
    if any(digest(data / name) != sha for name, sha in seal["predictions"].items()):
        raise ValueError("Sealed forecast changed")
    all_scores, physical, identities, ranges = [], [], [], []
    cases = load(data / "cases.json")
    for case in cases:
        y, _absolute, z = truth(data, case)
        meta = {k: case[k] for k in ("seed", "history")}
        meta["schedule"] = case["schedule"]["name"]
        saved = np.load(
            data / f"prediction-{case['index']}-{meta['schedule']}.npz",
            allow_pickle=False,
        )
        pred, tr = saved["y"], saved["trajectory"]
        for name, p in rm.comparators(pred).items():
            for s in score(y, p, floors):
                if args.freeze:
                    scale = load(args.freeze / "freeze.json")[
                        "development_single_event_scale"
                    ][s["window"]]
                    s["error_over_fixed_development_single_event_scale"] = (
                        s["error_rms"] / scale[["mass", "moment"].index(s["output"])]
                    )
                all_scores.append(dict(**meta, model=name, **s))
        true_add, pred_add = y[1] + y[2] - y[0], pred[1] + pred[2] - pred[0]
        terms = np.array(
            [rm.contrasts(y)["K12"], true_add - pred_add, pred_add - pred[3]]
        )
        residual = y[3] - pred[3]
        identities.append(
            dict(
                **meta,
                signed_identity_max=float(abs(terms.sum(axis=0) - residual).max()),
                terms_rms=rms(terms, axis=1).tolist(),
                squared_error_gram=np.einsum("ito,jto->oij", terms, terms).tolist(),
            )
        )
        physical.append(
            dict(
                **meta,
                absolute_max=abs(tr[:, :, :3] - z).max(axis=(0, 1)).tolist(),
                final_errors=(tr[:, -1, :3] - z[:, -1]).tolist(),
                true_event_motion=(z[3] - z[0]).tolist(),
                predicted_event_motion=(tr[3, :, :3] - tr[0, :, :3]).tolist(),
                induced_error_max=abs((tr[3, :, :3] - tr[0, :, :3]) - (z[3] - z[0]))
                .max(axis=0)
                .tolist(),
            )
        )
        ranges.append(
            dict(
                **meta,
                min=tr.min(axis=(0, 1)).tolist(),
                max=tr.max(axis=(0, 1)).tolist(),
                total_absolute_input=sum(
                    abs(a) for a in case["schedule"]["amplitudes"]
                ),
            )
        )
    pm.save_json(out / "scores.json", all_scores)
    pm.save_json(out / "physical.json", physical)
    pm.save_json(out / "identities.json", identities)
    pm.save_json(out / "state-ranges.json", ranges)
    summary = {}
    for model_name in rm.comparators(np.zeros((4, 161, 2))):
        summary[model_name] = {}
        for kind in rm.CONTRASTS:
            ss = [
                s for s in all_scores if s["model"] == model_name and s["kind"] == kind
            ]
            summary[model_name][kind] = {
                "worst_relative": max(s["relative"] or 0 for s in ss),
                "failed": sum(s["pass"] is False for s in ss),
                "unresolved": sum(not s["resolved"] for s in ss),
                "count": len(ss),
                "worst": max(ss, key=lambda s: s["relative"] or 0),
            }
    pm.save_json(out / "summary.json", summary)
    # Static scientific plots keep geometry, common response and small contrasts separate.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for case in [c for c in cases if c["index"] == 0]:
        y, _, z = truth(data, case)
        s = case["schedule"]["name"]
        saved = np.load(data / f"prediction-0-{s}.npz")
        pred, tr = saved["y"], saved["trajectory"]
        fig, axes = plt.subplots(3, 1, figsize=(8, 7), sharex=True)
        for j, ax in enumerate(axes):
            ax.plot(np.arange(41) + 50, z[3, :, j], label="true combined")
            ax.plot(np.arange(41) + 50, tr[3, :, j], "--", label="nine-state")
            ax.plot(np.arange(41) + 50, z[0, :, j], ":", label="true no event")
            ax.set_ylabel("ABC"[j] + " center offset")
            ax.legend(fontsize=8)
        axes[-1].set_xlabel("Absolute time")
        fig.tight_layout()
        fig.savefig(out / f"geometry-{s}.png", dpi=140)
        plt.close(fig)
        fig, axes = plt.subplots(4, 2, figsize=(10, 10), sharex=True)
        for k, kind in enumerate(["R11", "D12", "D2|1", "K12"]):
            for o in range(2):
                ax = axes[k, o]
                ax.plot(pm.TIMES, rm.contrasts(y)[kind][:, o], label="true")
                ax.plot(
                    pm.TIMES, rm.contrasts(pred)[kind][:, o], "--", label="nine-state"
                )
                addition = rm.comparators(pred)["independent-addition"]
                ax.plot(
                    pm.TIMES,
                    rm.contrasts(addition)[kind][:, o],
                    ":",
                    label="single-event addition",
                )
                ax.set_ylabel(kind + " " + ["mass", "moment"][o])
                ax.legend(fontsize=7)
        for ax in axes[-1]:
            ax.set_xlabel("Time since final probe")
        fig.tight_layout()
        fig.savefig(out / f"response-{s}.png", dpi=140)
        plt.close(fig)
    budget.finish()
