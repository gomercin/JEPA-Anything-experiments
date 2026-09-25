"""Saved-data qualification, prospective freeze, and fresh relay analysis."""

import argparse
from pathlib import Path

import numpy as np

from .explore import write_json
from .organization_response import Budget, gain_delay, rms, shifted
from .source_receiver_relay import (
    CFG,
    DEV,
    EPS,
    FRESH,
    POSITIONS,
    ROOT,
    A,
    C,
    digest,
    load,
    start_output,
)

KINDS = ("full", "cut", "qb")


def read_case(root, seed, b, suffix=""):
    name = f"s{seed}-b{b:g}{suffix}"
    with np.load(root / f"{name}.npz", allow_pickle=False) as z:
        arrays = dict(z)
    return arrays, load(root / f"{name}.json")


def score(truth, prediction, floor):
    err = rms(prediction - truth, axis=0)
    sig = rms(truth, axis=0)
    return {
        "rmse": err.tolist(),
        "signal": sig.tolist(),
        "relative": (err / np.maximum(sig, 1e-30)).tolist(),
        "resolved": (sig > floor).tolist(),
        "maximum": abs(prediction - truth).max(axis=0).tolist(),
    }


def geometry(meta):
    g = meta["sham_geometry"]
    centers = np.array([x["center"] for x in g])
    end = np.array([x["center"] for x in meta["final_geometry"]])
    all_centers = np.concatenate([centers, end])
    anchors = np.array([A, meta["b"], C])
    widths = np.array(
        [x["width"] for x in g] + [x["width"] for x in meta["final_geometry"]]
    )
    masses = np.array(
        [x["mass"] for x in g] + [x["mass"] for x in meta["final_geometry"]]
    )
    minimum_gap = float(np.diff(all_centers, axis=-1).min())
    anchor_offset = float(abs(all_centers - anchors).max())
    drift = float(abs(all_centers - centers[0]).max())
    return {
        "minimum_gap": minimum_gap,
        "maximum_anchor_offset": anchor_offset,
        "maximum_drift": drift,
        "maximum_width_change": float(abs(widths - widths[0]).max()),
        "minimum_mass": float(masses.min()),
        "qualified": bool(
            minimum_gap > 22
            and anchor_offset < 0.5
            and drift < 0.5
            and masses.min() > 1
        ),
        "initial": g[0],
        "final": g[-1],
    }


def freeze(out, data, refinement, mask):
    dev = [read_case(data, DEV[0], b) for b in POSITIONS]
    floors, delta_floors = np.full((3, 3), 1e-11), np.full((3, 3), 1e-11)
    refinements = []
    for label in ("half-dt", "double-n"):
        errors, direct = [], []
        for b, (base, _) in zip(POSITIONS, dev, strict=True):
            fine, _ = read_case(refinement, DEV[0], b, "-" + label)
            df = fine["full"][:, 0] - base["full"][:, 0]
            dk = fine["cut"][:, 0] - base["cut"][:, 0]
            conservative = rms(df, axis=0) + rms(dk, axis=0)
            errors.append(conservative)
            direct.append((df, dk))
            floors = np.maximum(floors, 5 * conservative)
            refinements.append(
                {
                    "b": b,
                    "refinement": label,
                    "conservative": conservative.tolist(),
                    "direct_q_rmse": rms(df - dk, axis=0).tolist(),
                }
            )
        delta_floors = np.maximum(delta_floors, 5 * sum(errors))
        refinements.append(
            {
                "refinement": label,
                "delta_full_rmse": rms(direct[1][0] - direct[0][0], axis=0).tolist(),
                "delta_q_rmse": rms(
                    (direct[1][0] - direct[1][1]) - (direct[0][0] - direct[0][1]),
                    axis=0,
                ).tolist(),
            }
        )
    models, comparisons, tangent_checks, masks = {}, [], [], []
    for kind in KINDS:
        x, y = [d[kind][:, 0, 2, :2] for d, _ in dev]
        scales = np.maximum(rms(np.stack([x, y]), axis=(0, 1)), floors[2, :2])
        fit = gain_delay(
            [(x, y)], dev[0][0]["time"], scales, np.arange(-8, 8.0001, 0.025)
        )
        fit["scales"] = scales.tolist()
        models[kind] = fit
        pred = fit["gain"] * shifted(x, dev[0][0]["time"], fit["delay"])
        comparisons.append(dict(kind=kind, fit=fit, **score(y, pred, floors[2, :2])))
    for b, (arr, meta) in zip(POSITIONS, dev, strict=True):
        if not geometry(meta)["qualified"] or meta["sham_field_max"] > 1e-12:
            raise RuntimeError("Preparation or matched sham did not qualify")
        for j, amplitude in enumerate(meta["amplitudes"]):
            for kind in KINDS:
                truth = arr[kind][:, j, 2, :2]
                pred = arr["tangent_" + kind][:, 2, :2] * amplitude / EPS
                s = score(truth, pred, floors[2, :2])
                tangent_checks.append(dict(b=b, amplitude=amplitude, kind=kind, **s))
                if kind == "full" and np.any(
                    (np.array(s["relative"]) > 0.02)
                    & (np.array(s["rmse"]) > floors[2, :2])
                ):
                    raise RuntimeError("Nonlinear tangent check exceeds declared gate")
        alternative, _ = read_case(mask, DEV[0], b)
        masks.append(
            {
                "b": b,
                "score": score(
                    arr["qb"][:, 0, 2, :2],
                    alternative["qb"][:, 0, 2, :2],
                    floors[2, :2],
                ),
            }
        )
    delta_q = dev[1][0]["qb"][:, 0, 2, :2] - dev[0][0]["qb"][:, 0, 2, :2]
    mask_delta = [read_case(mask, DEV[0], b)[0]["qb"][:, 0, 2, :2] for b in POSITIONS]
    mask_delta_score = score(
        delta_q, mask_delta[1] - mask_delta[0], delta_floors[2, :2]
    )
    result = {
        "development_seeds": list(DEV),
        "fresh_seeds": list(FRESH),
        "positions": list(POSITIONS),
        "config": vars(CFG),
        "eps": EPS,
        "withheld_amplitude": 0.03,
        "models": models,
        "floors": floors.tolist(),
        "delta_floors": delta_floors.tolist(),
        "candidate_resolved": (rms(delta_q, axis=0) > delta_floors[2, :2]).tolist(),
        "prediction_gates": {"full": 0.02, "qb": 0.10},
        "sources": {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
        "inputs": {
            str(p): digest(p)
            for root in (data, refinement, mask)
            for p in sorted(root.glob("*"))
            if p.is_file()
        },
        "refinements": refinements,
        "development_comparisons": comparisons,
        "tangent_checks": tangent_checks,
        "mask_checks": masks,
        "mask_delta_score": mask_delta_score,
    }
    write_json(out / "freeze.json", result)
    print(json_summary(result), flush=True)


def json_summary(f):
    return str(
        {
            "floors": f["floors"][2],
            "delta_floors": f["delta_floors"][2],
            "candidate": f["candidate_resolved"],
            "models": f["models"],
            "mask_delta": f["mask_delta_score"],
        }
    )


def analyze(out, data, frozen):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    freeze = load(frozen / "freeze.json")
    floors, dfloors = np.asarray(freeze["floors"]), np.asarray(freeze["delta_floors"])
    rows, organizations = [], []
    for seed in FRESH:
        pair = [read_case(data, seed, b) for b in POSITIONS]
        for b, (arr, meta) in zip(POSITIONS, pair, strict=True):
            organizations.append(
                {
                    "seed": seed,
                    "b": b,
                    "geometry": geometry(meta),
                    "jumps": meta["jumps"],
                    "sham_max": meta["sham_field_max"],
                }
            )
            pred, _ = read_case(data, seed, b, "-prediction")
            for j, amplitude in enumerate(meta["amplitudes"]):
                for kind in KINDS:
                    truth = arr[kind][:, j, 2, :2]
                    prediction = pred["tangent_" + kind][:, 2, :2] * amplitude / EPS
                    s = score(truth, prediction, floors[2, :2])
                    gate = freeze["prediction_gates"].get(kind, 0.02)
                    passed = bool(
                        np.all(
                            (np.asarray(s["relative"]) <= gate)
                            | (np.asarray(s["rmse"]) <= floors[2, :2])
                        )
                    )
                    rows.append(
                        dict(
                            seed=seed,
                            b=b,
                            amplitude=amplitude,
                            kind=kind,
                            test="tangent",
                            passed=passed,
                            **s,
                        )
                    )
        for kind in KINDS:
            x, y = [d[kind][:, 0, 2, :2] for d, _ in pair]
            fit = freeze["models"][kind]
            predicted = fit["gain"] * shifted(x, pair[0][0]["time"], fit["delay"])
            rows.append(
                dict(
                    seed=seed,
                    kind=kind,
                    test="frozen_gain_delay",
                    **score(y, predicted, floors[2, :2]),
                )
            )
            delta = y - x
            tdelta = (
                pair[1][0]["tangent_" + kind][:, 2, :2]
                - pair[0][0]["tangent_" + kind][:, 2, :2]
            )
            rows.append(
                dict(
                    seed=seed,
                    kind=kind,
                    test="organization_difference",
                    **score(delta, tdelta, dfloors[2, :2]),
                )
            )
        for b, (arr, _) in zip(POSITIONS, pair, strict=True):
            for node, index in [("A", 0), ("B", 1), ("C", 2)]:
                f, q = arr["full"][:, 0, index], arr["qb"][:, 0, index]
                rows.append(
                    {
                        "seed": seed,
                        "b": b,
                        "test": "pathway_fraction",
                        "node": node,
                        "full_rms": rms(f, axis=0).tolist(),
                        "qb_rms": rms(q, axis=0).tolist(),
                        "fraction": (
                            rms(q, axis=0) / np.maximum(rms(f, axis=0), 1e-30)
                        ).tolist(),
                        "resolved": (rms(q, axis=0) > floors[index]).tolist(),
                    }
                )
    write_json(
        out / "summary.json",
        {
            "rows": rows,
            "organizations": organizations,
            "freeze_sha256": digest(frozen / "freeze.json"),
        },
    )
    pair = [read_case(data, FRESH[0], b)[0] for b in POSITIONS]
    fig, axes = plt.subplots(3, 3, figsize=(13, 9))
    colors = ("#205b92", "#c26026")
    for col, (node, index) in enumerate([("A", 0), ("B", 1), ("C", 2)]):
        for b, arr, color in zip(POSITIONS, pair, colors, strict=True):
            t = arr["time"]
            axes[0, col].plot(
                t, arr["full"][:, 0, index, 0], color=color, label=f"B={b:g} full"
            )
            axes[0, col].plot(
                t,
                arr["cut"][:, 0, index, 0],
                "--",
                color=color,
                label=f"B={b:g} feedback cut",
            )
            axes[1, col].plot(
                t, arr["qb"][:, 0, index, 0], color=color, label=f"B={b:g}"
            )
        for kind, color in [("full", "#555555"), ("qb", "#8c398c")]:
            delta = pair[1][kind][:, 0, index, 0] - pair[0][kind][:, 0, index, 0]
            axes[2, col].plot(t, delta, color=color, label="delta " + kind)
        axes[1, col].axhspan(
            -floors[index, 0], floors[index, 0], color="gray", alpha=0.2
        )
        axes[2, col].axhspan(
            -dfloors[index, 0], dfloors[index, 0], color="gray", alpha=0.2
        )
        axes[0, col].set_title(f"{node}: weighted mass response")
        axes[2, col].set_xlabel("Time after fixed A input")
    for row, label in zip(
        axes, ["F and K", "Q_B = F - K", "B=-4 minus B=0"], strict=True
    ):
        row[0].set_ylabel(label)
        for ax in row:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.grid(alpha=0.18)
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    axes[2, 0].legend(fontsize=7)
    fig.suptitle(
        f"Actual nonlinear fields: fresh seed {FRESH[0]}, epsilon=.02; shared axes across organizations"
    )
    fig.tight_layout()
    fig.savefig(out / "relay.png", dpi=160)
    plt.close(fig)
    print("Saved fresh summary and relay figure", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["freeze", "analyze"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--refinement", type=Path)
    p.add_argument("--mask", type=Path)
    p.add_argument("--freeze", type=Path)
    args = p.parse_args()
    if args.stage == "freeze" and (args.refinement is None or args.mask is None):
        p.error("--refinement and --mask required")
    if args.stage == "analyze" and args.freeze is None:
        p.error("--freeze required")
    previous = sum(load(f)["cpu_seconds"] for f in ROOT.glob("*/budget.json"))
    budget = Budget(previous, 1200 if args.stage == "freeze" else 1800)
    start_output(args.output, args.stage)
    status = "FAILED"
    try:
        budget.check()
        if args.stage == "freeze":
            freeze(args.output, args.data, args.refinement, args.mask)
        else:
            analyze(args.output, args.data, args.freeze)
        status = "COMPLETE"
    finally:
        write_json(args.output / "budget.json", dict(**budget.receipt(), status=status))


if __name__ == "__main__":
    main()
