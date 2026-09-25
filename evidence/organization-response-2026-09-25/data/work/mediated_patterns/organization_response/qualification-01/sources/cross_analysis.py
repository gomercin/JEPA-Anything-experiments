"""Read-only scientific synthesis of the saved cross-pulse panels; no fitting."""

import argparse
import hashlib
import json
import shlex
import time
from pathlib import Path

import numpy as np

from .cross_pulse import FROZEN, assess
from .event_response import CrossResponse
from .explore import start_panel, write_json
from .measurements import describe
from .recursive_response import error_metrics

ROOT = Path("work/mediated_patterns/cross_pulse_repair")


def read(relative):
    return json.loads((ROOT / relative).read_text())


def rms(x):
    return float(np.sqrt(np.mean(np.asarray(x) ** 2)))


def grouped(rows, predictions):
    result = []
    for role in ("short", "regression", "three"):
        group = [r for r in rows if r["role"] == role]
        for name in predictions:
            scores = [r["scores"][name] for r in group]
            result.append(
                {
                    "role": role,
                    "model": name,
                    "runs": len(group),
                    "passed": sum(s["pass"] for s in scores),
                    "pooled": error_metrics(
                        np.concatenate([r["response"] for r in group]),
                        np.concatenate([r["predictions"][name] for r in group]),
                    ),
                    "worst_nrmse": max(s["metrics"]["all"]["nrmse"] for s in scores),
                    "worst_after_last_nrmse": max(
                        s["metrics"]["after_last"]["nrmse"] for s in scores
                    ),
                    "worst_absolute": max(
                        s["metrics"]["all"]["max_absolute"] for s in scores
                    ),
                    "worst_early_absolute": max(
                        s["metrics"]["early"]["max_absolute"] for s in scores
                    ),
                }
            )
    return result


def checkpoint(model_path, row, out):
    """No field or future readout passed to the predictor; truth used afterward."""
    model = CrossResponse.load(model_path).initialize(row["separation"])
    events = dict(row["pulses"])
    stop = row["pulses"][1][0]  # includes pending second event
    before = []
    for t in row["time"]:
        if t > stop:
            break
        if t:
            model.advance(model.step)
        if t in events:
            model.jump(events[t])
        before.append(model.output())
    model.save(out / "checkpoint.json")
    resumed = CrossResponse.load(out / "checkpoint.json")
    tail, resumed_tail = [], []
    for t in row["time"][len(before) :]:
        model.advance(model.step)
        resumed.advance(resumed.step)
        if t in events:
            model.jump(events[t])
            resumed.jump(events[t])
        tail.append(model.output())
        resumed_tail.append(resumed.output())
    prediction = before + resumed_tail
    return {
        "run_id": row["run_id"],
        "time": stop,
        "resumed_max_difference": float(np.max(abs(np.array(tail) - resumed_tail))),
        "saved_prediction_max_difference": float(
            np.max(abs(np.array(prediction) - row["predictions"]["candidate"]))
        ),
        "accuracy_against_independent_fields": assess(
            row, prediction, read("frozen-02/freeze.json")["gates"]
        ),
        "serialized_keys": list(json.loads((out / "checkpoint.json").read_text())),
        "checkpoint_bytes": (out / "checkpoint.json").stat().st_size,
    }


def plots(rows, diagnostic, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10})
    names = ["s1101-d25.5-short_np275", "s1202-d25.5-three_close_withheld"]
    colors = {"kernel": "#d28b00", "original": "#ad4475", "candidate": "#187b72"}
    labels = {
        "kernel": "Original kernel",
        "original": "Original recurrence",
        "candidate": "Repaired recurrence",
    }
    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    for col, name in enumerate(names):
        row = next(r for r in rows if r["run_id"] == name)
        t, y = np.array(row["time"]), np.array(row["response"])
        axes[0, col].plot(
            t, y * 1e6, color="black", lw=2, label="Full fields minus sham"
        )
        for m, c in colors.items():
            p = np.array(row["predictions"][m])
            axes[0, col].plot(
                t,
                p * 1e6,
                color=c,
                ls="--" if m != "candidate" else "-",
                label=labels[m],
            )
            axes[1, col].plot(t, (y - p) * 1e6, color=c, label=labels[m])
            axes[2, col].plot(t, p * 1e9, color=c)
        axes[2, col].plot(t, y * 1e9, color="black", lw=2)
        axes[0, col].set_title(
            name.replace("s1101-d25.5-", "Two pulses, separation 25.5\n").replace(
                "s1202-d25.5-", "Three pulses, separation 25.5\n"
            )
        )
        for ax in axes[:, col]:
            for when, a in row["pulses"]:
                ax.axvline(when, color="#666666", lw=0.8, alpha=0.65)
            ax.axhline(0, color="#999999", lw=0.5)
            ax.grid(alpha=0.2)
            ax.set_xlabel("Time after prediction boundary")
        axes[2, col].set_xlim(0, 8)
        axes[2, col].set_ylim(-100, 100)
        axes[2, col].set_title(
            "Early response (magnified)\nRelative accuracy unresolved below the 100-nanounit floor",
            fontsize=9,
        )
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_ylabel("Signed response × 10⁶")
    axes[1, 0].set_ylabel("Truth − prediction × 10⁶")
    axes[2, 0].set_ylabel("Signed response × 10⁹")
    fig.suptitle(
        "Fresh predictions without resets or field refresh; fixed pulse and sensor anchors",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out / "fresh_sequences.png", dpi=160)
    plt.close(fig)
    row = diagnostic["records"][0]
    t = np.array(row["time"])
    terms = {k: np.asarray(v) for k, v in row["terms"].items() if isinstance(v, list)}
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    for k, label in [
        ("r1", "First pulse alone"),
        ("r2", "Second at its original absolute time"),
        ("r12", "Both pulses"),
    ]:
        axes[0].plot(t, terms[k] * 1e6, label=label)
    axes[0].set_ylabel("Signed response × 10⁶")
    axes[0].set_title(
        f"Cancellation: pair RMS {rms(terms['r12']):.2e}; fixed development single RMS {diagnostic['diagnostic_scale']:.2e}"
    )
    for k, label in [
        ("cross", "Full-field cross response"),
        ("kernel_fit", "Single-response fit / age"),
        ("realization", "Kernel − recurrence"),
        ("total_error", "Total error"),
    ]:
        axes[1].plot(t, terms[k] * 1e7, label=label)
    axes[1].set_ylabel("Signed error terms × 10⁷")
    axes[1].set_xlabel("Time")
    for ax in axes:
        ax.axvline(3, color="black", lw=0.8, ls=":")
        ax.axhline(0, color="#999999", lw=0.5)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Matched quartet, seed 411 / separation 29: −0.075 then +0.075 at time 3"
    )
    fig.tight_layout()
    fig.savefig(out / "error_decomposition.png", dpi=160)
    plt.close(fig)


def analyze(output):
    out = start_panel(
        output,
        {
            "purpose": "read-only synthesis; no retuning or new fields",
            "fresh": "fresh-02",
        },
    )
    rows = read("fresh-02/results.json")["records"]
    gates = read("frozen-02/freeze.json")["gates"]
    single_only = CrossResponse.load(ROOT / "conditioned-02/single-only.json")
    for r in rows:
        p = single_only.rollout(r["separation"], r["pulses"], r["time"])
        r["predictions"]["single_fit_only"] = p.tolist()
        r["scores"]["single_fit_only"] = assess(r, p, gates)
    names = ("kernel", "original", "single_fit_only", "candidate")
    summary = grouped(rows, names)
    per_preparation = []
    for prep in sorted({r["preparation_id"] for r in rows}):
        for v in grouped([r for r in rows if r["preparation_id"] == prep], names):
            per_preparation.append({"preparation": prep, **v})
    paired = [
        {
            "run_id": r["run_id"],
            "role": r["role"],
            "scores": r["scores"],
            "nrmse_change_vs_original": r["scores"]["candidate"]["metrics"]["all"][
                "nrmse"
            ]
            - r["scores"]["original"]["metrics"]["all"]["nrmse"],
            "nrmse_change_vs_kernel": r["scores"]["candidate"]["metrics"]["all"][
                "nrmse"
            ]
            - r["scores"]["kernel"]["metrics"]["all"]["nrmse"],
        }
        for r in rows
    ]
    diagnostic = read("diagnosis-01/results.json")
    decomposition = []
    for r in diagnostic["records"]:
        parts = np.array(
            [r["terms"][k] for k in ("cross", "kernel_fit", "realization")]
        )
        windows = {}
        t = np.array(r["time"])
        for name, mask in [
            ("before_second", t < r["pulses"][1][0]),
            ("after_second", t >= r["pulses"][1][0]),
        ]:
            y = np.array(r["terms"]["r12"])[mask]
            e = np.array(r["terms"]["total_error"])[mask]
            windows[name] = {
                "signal_rms": rms(y),
                "rmse": rms(e),
                "max_error": float(max(abs(e))),
            }
        decomposition.append(
            {
                "run_id": r["run_id"],
                "pulses": r["pulses"],
                "rms": r["rms"],
                "max_absolute": {
                    k: float(max(abs(np.array(r["terms"][k]))))
                    for k in ("cross", "kernel_fit", "realization", "total_error")
                },
                "windows": windows,
                "fixed_scale_nrmse": r["rms"]["total_error"]
                / diagnostic["diagnostic_scale"],
                "cancellation_ratio_pair_over_single_rms_sum": r["rms"]["r12"]
                / (r["rms"]["r1"] + r["rms"]["r2"]),
                "mean_product_matrix": (parts @ parts.T / parts.shape[1]).tolist(),
                "identity_max_error": r["terms"]["identity_max_error"],
                "note": "sum of all entries of mean-product matrix gives total MSE; diagonal alone does not",
            }
        )
    old_refinement = read("refinement-01/results.json")["records"]
    bounds = []
    for mesh in range(1, 4):
        for q in range(2):
            a, b = old_refinement[q], old_refinement[2 * mesh + q]
            bounds.append(
                {
                    "mesh": mesh,
                    "quartet": q,
                    "paired_response_conservative_bound": sum(
                        float(max(abs(np.array(a["terms"][k]) - b["terms"][k])))
                        for k in ("r1", "r2", "r12")
                    ),
                }
            )
    # Cost ledger: literal stored timers, no double-counting nested derived records.
    ledger = []
    for path in sorted(ROOT.glob("*/results.json")):
        data = json.loads(path.read_text())
        if isinstance(data, dict) and "costs" in data:
            ledger.append(
                {
                    "panel": path.parent.name,
                    "simulation_cpu_seconds": sum(
                        c.get("cpu_seconds", 0) + c.get("preparation_cpu_seconds", 0)
                        for c in data["costs"]
                    ),
                    "simulation_wall_seconds": sum(
                        c.get("wall_seconds", 0) + c.get("preparation_wall_seconds", 0)
                        for c in data["costs"]
                    ),
                    "prediction_cpu_seconds": sum(
                        c.get("prediction_cpu_seconds", 0) for c in data["costs"]
                    ),
                }
            )
        elif isinstance(data, dict) and "cpu_seconds" in data:
            ledger.append(
                {
                    "panel": path.parent.name,
                    "fit_cpu_seconds": data["cpu_seconds"],
                    "fit_wall_seconds": data.get("wall_seconds", 0),
                }
            )
    for path in sorted(ROOT.glob("*/fit.json")):
        d = json.loads(path.read_text())
        ledger.append(
            {
                "panel": path.parent.name,
                "fit_cpu_seconds": d.get("cpu_seconds", 0),
                "fit_wall_seconds": d.get("wall_seconds", 0),
            }
        )
    model_path = ROOT / "frozen-02/model.json"
    model = CrossResponse.load(model_path)
    pathway = []
    old_rows = json.loads((FROZEN.parent / "fresh-01/results.json").read_text())[
        "records"
    ]
    for old in old_rows:
        if "replay_response" not in old:
            continue
        p = model.rollout(old["separation"], old["pulses"], old["time"])
        contrast = np.array(old["response"]) - old["replay_response"]
        pathway.append(
            {
                "run_id": old["run_id"],
                "role": old["role"],
                "contrast_rms": rms(contrast),
                "total_error_rms": rms(p - old["response"]),
                "error_over_contrast": rms(p - old["response"]) / rms(contrast),
                "status": "SCALE_COMPARISON_ONLY; controlled contrast not predicted",
            }
        )
    r = next(r for r in rows if r["run_id"] == "s1202-d25.5-three_close_withheld")
    check = checkpoint(model_path, r, out)
    initial = np.load(ROOT / "fresh-02/seed1202-d25.5-initial.npz")
    benchmarks = {}
    for name, count, fn in [
        (
            "initial_field_scan",
            100,
            lambda: describe(initial["x"], initial["state"], [-12.75, 12.75]),
        ),
        (
            "initialize_static_interpolation",
            100,
            lambda: model.initialize(r["separation"]),
        ),
        (
            "forecast_100_units",
            100,
            lambda: model.rollout(r["separation"], r["pulses"], r["time"]),
        ),
    ]:
        st, sw = time.process_time(), time.perf_counter()
        for _ in range(count):
            fn()
        benchmarks[name] = {
            "repeats": count,
            "mean_cpu_seconds": (time.process_time() - st) / count,
            "mean_wall_seconds": (time.perf_counter() - sw) / count,
        }
    model.initialize(r["separation"])
    st, sw = time.process_time(), time.perf_counter()
    for _ in range(10000):
        model.advance(0.25)
    benchmarks["one_state_tick"] = {
        "repeats": 10000,
        "mean_cpu_seconds": (time.process_time() - st) / 10000,
        "mean_wall_seconds": (time.perf_counter() - sw) / 10000,
    }
    drift = []
    for r in rows:
        center = np.array(r["descriptors"]["center"])
        mass = np.array(r["descriptors"]["mass"])
        drift.append(
            {
                "run_id": r["run_id"],
                "maximum_separation_change": float(
                    max(abs(np.diff(center, axis=1)[:, 0] - r["separation"]))
                ),
                "minimum_mass_ratio": float((mass / mass[0]).min()),
                "maximum_mass_ratio": float((mass / mass[0]).max()),
            }
        )
    protected = read("protected.json")
    changed = [
        p
        for p, h in protected.items()
        if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != h
    ]
    commands = []
    for path in sorted(ROOT.glob("*/protocol.json"), key=lambda p: p.stat().st_mtime):
        command = json.loads(path.read_text())["command"]
        module = "experiments.mediated_patterns." + Path(command[0]).stem
        commands.append(
            "OPENBLAS_NUM_THREADS=1 .venv/bin/python -m "
            + module
            + " "
            + shlex.join(command[1:])
        )
    (out / "commands.txt").write_text("\n".join(commands) + "\n")
    write_json(
        out / "summary.json",
        {
            "fresh_summary": summary,
            "per_preparation": per_preparation,
            "paired": paired,
            "diagnosis": decomposition,
            "diagnostic_fixed_scale": diagnostic["diagnostic_scale"],
            "paired_refinement_bounds": bounds,
            "checkpoint": check,
            "historical_pathway_scale_only": pathway,
            "cost_ledger": ledger,
            "benchmarks": benchmarks,
            "drift": drift,
            "resources": {
                "response_variables": 10,
                "memory_variables": 2,
                "pending_input_scalars": 2,
                "clock": 1,
                "fixed_separation": 1,
                "independent_static_numeric_values": 493,
                "cached_injection_values": 40,
                "model_step": 0.25,
                "stored_model_json_bytes": model_path.stat().st_size,
                "full_field_scalars": 1024,
                "input_history_buffer": 0,
                "kernel_history": "original kernel retains/replays each previous pulse; not fixed size",
            },
            "protected_artifacts": {"checked": len(protected), "changed": changed},
            "limitations": [
                "early relative accuracy uncertified",
                "no prediction of receiver-replay contrast",
                "finite tested input family, no arbitrary-history closure",
            ],
        },
    )
    plots(rows, diagnostic, out)
    print(json.dumps(summary, indent=2))
    print(
        "protected",
        len(protected),
        "changed",
        changed,
        "checkpoint",
        check["resumed_max_difference"],
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    analyze(p.parse_args().output)


if __name__ == "__main__":
    main()
