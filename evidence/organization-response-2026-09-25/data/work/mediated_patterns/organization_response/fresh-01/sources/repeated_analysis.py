"""Read saved repeated-hybrid panels; no fitting or scientific gate changes."""

import argparse
import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .explore import start_panel, write_json
from .hybrid_analysis import sum_times
from .repeated_hybrid import OLD, ROOT, load


def plots(row, out, floor):
    t = np.array(row["truth"]["time"])
    events = row["plan"]["events"]
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for ax, col, label in [
        (axes[0], 2, "C total mass response"),
        (axes[1], 1, "B source-region mass response"),
    ]:
        tr, hy = (
            np.array(row["truth"]["response"])[:, col],
            np.array(row["hybrid"]["response"])[:, col],
        )
        ax.plot(t, tr, color="black", label="Full field")
        ax.plot(t, hy, "--", color="teal", label="Frozen 20-mode hybrid")
        ax.set_ylabel(label)
        ax.legend()
    for col, label in [(1, "B"), (2, "C")]:
        residual = (
            np.array(row["truth"]["response"]) - np.array(row["hybrid"]["response"])
        )[:, col]
        axes[2].plot(t, residual, label=label + " signed residual")
    axes[2].set_ylabel("Truth minus hybrid")
    axes[2].legend()
    for ax in axes:
        for when, a in events:
            ax.axvline(when, color="darkorange", alpha=0.6, lw=1)
        ax.grid(alpha=0.2)
    axes[0].set_title(
        row["run_id"] + "; inputs " + str(events) + "; no resets or field refresh"
    )
    axes[-1].set_xlabel("Time since first C stimulus")
    fig.tight_layout()
    fig.savefig(out / "repeated_responses.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    tr = np.array(row["truth"]["return_contrast"])[:, 2]
    hy = np.array(row["hybrid"]["return_contrast"])[:, 2]
    axes[0].plot(t, tr, color="black", label="Full field, own matched control")
    axes[0].plot(t, hy, "--", color="teal", label="Hybrid, own matched control")
    axes[0].axhline(0, color="grey", ls=":", label="Zero-return comparator")
    axes[0].set_ylabel("Selected return contrast at C")
    axes[0].legend()
    axes[1].plot(t, tr - hy, color="teal")
    axes[1].set_ylabel("Signed return residual")
    axes[2].plot(t, tr, color="black")
    axes[2].plot(t, hy, "--", color="teal")
    axes[2].axhspan(
        -floor,
        floor,
        color="grey",
        alpha=0.18,
        label="± frozen RMS resolution floor (scale only)",
    )
    axes[2].set_ylim(-2 * floor, 2 * floor)
    axes[2].set_ylabel("Small-signal view")
    axes[2].legend(loc="upper right", fontsize=8)
    # Shade only the originally unresolved first-five interval. Later-window
    # resolution is assessed on window RMS, not pointwise crossings of this band.
    for ax in axes:
        ax.axvspan(0, 5, color="grey", alpha=0.12)
        for when, a in events:
            ax.axvline(when, color="darkorange", alpha=0.6, lw=1)
        ax.grid(alpha=0.2)
    axes[0].set_title(row["run_id"] + ": pathway-specific response, not all feedback")
    axes[-1].set_xlabel(
        "Time; orange lines are declared C events, grey first interval unresolved"
    )
    fig.tight_layout()
    fig.savefig(out / "repeated_return.png", dpi=160)
    plt.close(fig)


def analyze(output):
    out = start_panel(
        output, {"purpose": "report-only saved repeated-hybrid results; no fitting"}
    )
    rows = load(ROOT / "fresh-01/results.json")
    frozen = load(ROOT / "frozen-01/freeze.json")
    summaries = []
    for r in rows:
        a = r["assessment"]
        eligible = {k: v for k, v in a["windows"].items() if not k.endswith("early5")}
        worst = {
            name: max(
                v[name]["nrmse"] for v in eligible.values() if v[name]["resolved"]
            )
            for name in ["B_response", "C_response", "C_return"]
        }
        tr = np.array(r["truth"]["response"])[:, 2]
        cut = np.array(r["truth"]["cut_response"])[:, 2]
        no_return = float(np.linalg.norm(tr - cut) / np.linalg.norm(tr))
        early = {
            k: v["C_return"] for k, v in a["windows"].items() if k.endswith("early5")
        }
        summaries.append(
            {
                "run_id": r["run_id"],
                "preparation_id": r["preparation_id"],
                "plan": r["plan"],
                "assessment": a,
                "worst_resolved_window_nrmse": worst,
                "geometry": r["geometry"],
                "coverage": {
                    k: v for k, v in r["coverage"].items() if k != "joint_distance"
                },
                "no_return_total_C_nrmse": no_return,
                "zero_return_contrast_nrmse": 1.0,
                "early_return_windows": early,
                "cumulative_amplitude_abs": sum(abs(v[1]) for v in r["plan"]["events"]),
                "command_sum": sum(v[1] for v in r["plan"]["events"]),
                "peak_growth_ratio": max(
                    r["truth"]["geometry"]["peak"][i][2]
                    / r["truth"]["geometry"]["peak"][0][2]
                    for i in range(len(r["truth"]["time"]))
                ),
            }
        )
    grouped = {}
    for name in frozen["plans"]:
        subset = [r for r in summaries if r["run_id"].endswith(name)]
        grouped[name] = {
            "runs": len(subset),
            "resolved_task_passes": sum(
                r["assessment"]["resolved_task_pass"] for r in subset
            ),
            "worst_whole_nrmse": {
                key: max(
                    r["assessment"]["windows"]["whole"][key]["nrmse"] for r in subset
                )
                for key in ["B_response", "C_response", "C_return"]
            },
            "worst_window_nrmse": {
                key: max(r["worst_resolved_window_nrmse"][key] for r in subset)
                for key in ["B_response", "C_response", "C_return"]
            },
            "worst_whole_absolute": {
                key: max(
                    r["assessment"]["windows"]["whole"][key]["max_absolute"]
                    for r in subset
                )
                for key in ["B_response", "C_response", "C_return"]
            },
        }
    # Select an illustrative trace by the declared return metric; no sensor or
    # gate selection. All cases remain in JSON and the per-preparation table.
    representative = max(
        rows, key=lambda r: r["assessment"]["windows"]["whole"]["C_return"]["nrmse"]
    )
    plots(representative, out, frozen["gates"]["return_signal_floor"])
    costs = []
    paths = (
        list(ROOT.glob("*/results.json"))
        + list(ROOT.glob("*/*-preparation.json"))
        + [ROOT / "development-01/first-only.json"]
    )
    for p in sorted(paths):
        if p.is_relative_to(out):
            continue
        cpu, wall = sum_times(load(p))
        costs.append({"file": str(p), "cpu_seconds": cpu, "wall_seconds": wall})
    protected = load(ROOT / "protected.json")
    changed = [
        p
        for p, h in protected.items()
        if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != h
    ]
    old = load(OLD / "fresh-01/results.json")[0]
    repeat = load(ROOT / "baseline-01/results.json")[0]
    first = load(ROOT / "development-01/first-only.json")
    reproduction = {
        k: float(abs(np.array(old[k]["absolute"]) - repeat[k]["absolute"]).max())
        for k in ["truth", "hybrid"]
    }
    reproduction["new_runner_single_max"] = max(
        float(abs(np.array(old[k]["absolute"]) - first[k]["absolute"]).max())
        for k in ["truth", "hybrid"]
    )
    if (ROOT / "quick-01/results.json").exists():
        quick = load(ROOT / "quick-01/results.json")[0]
        full = next(r for r in rows if r["run_id"] == quick["run_id"])
        reproduction["quick_sequence_max"] = max(
            float(abs(np.array(quick[k]["absolute"]) - full[k]["absolute"]).max())
            for k in ["truth", "hybrid"]
        )
    qualification = []
    for r in load(ROOT / "qualification-01/results.json"):
        times = np.array(r["truth"]["time"])
        plan = load(ROOT / "development-01/protocol.json")["cases"][r["run_id"]]
        late = np.ones(len(times), bool)
        for event, _ in plan["events"]:
            late &= ~((times >= event) & (times < event + 10))
        v = {
            "case": r["run_id"],
            "kind": r["kind"],
            "truth_difference": {
                k: v for k, v in r["truth_difference"].items() if k != "bound_trace"
            },
            "conservative_C_bound_excluding_first10_after_events": float(
                np.array(r["truth_difference"]["bound_trace"])[late, 2].max()
            ),
        }
        if "hybrid_difference" in r:
            v["hybrid_difference"] = {
                k: v for k, v in r["hybrid_difference"].items() if k != "bound_trace"
            }
        qualification.append(v)
    extra_qualification = None
    if (ROOT / "fresh-refinement-01/results.json").exists():
        extra = load(ROOT / "fresh-refinement-01/results.json")
        extra_qualification = {
            "run_id": extra["run_id"],
            "truth_difference": {
                k: v for k, v in extra["truth_difference"].items() if k != "bound_trace"
            },
            "hybrid_difference": {
                k: v
                for k, v in extra["hybrid_difference"].items()
                if k != "bound_trace"
            },
        }
    write_json(
        out / "summary.json",
        {
            "status": "EXPLORATORY_UNCHANGED_MODEL_REUSE",
            "preparations": len({r["preparation_id"] for r in summaries}),
            "seeds": frozen["seeds"],
            "runs": len(rows),
            "passed_resolved_tasks": sum(
                r["assessment"]["resolved_task_pass"] for r in rows
            ),
            "grouped": grouped,
            "per_run": summaries,
            "reproduction": reproduction,
            "qualification": qualification,
            "fresh_triple_qualification": extra_qualification,
            "representative": representative["run_id"],
            "costs": costs,
            "cpu_seconds": sum(c["cpu_seconds"] for c in costs),
            "wall_seconds_sum": sum(c["wall_seconds"] for c in costs),
            "protected": {"count": len(protected), "changed": changed},
            "model_sha256": frozen["model_sha256"],
            "checkpoint": load(ROOT / "checkpoint-01/results.json"),
            "development_duration_bookkeeping": "development snapshot counted .5 per sampled point including endpoints; fresh uses trapezoidal time integration; no dynamics or gates changed",
        },
    )
    lines = [
        "| Preparation/schedule | B whole % | C whole % | Return whole % | Worst resolved return-window % | Status |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for r in summaries:
        w = r["assessment"]["windows"]["whole"]
        lines.append(
            f"| {r['run_id']} | {w['B_response']['nrmse'] * 100:.4g} | {w['C_response']['nrmse'] * 100:.4g} | {w['C_return']['nrmse'] * 100:.4g} | {r['worst_resolved_window_nrmse']['C_return'] * 100:.4g} | {r['assessment']['status']} |"
        )
    (out / "per_preparation.md").write_text("\n".join(lines) + "\n")
    commands = []
    for p in sorted(ROOT.glob("*/protocol.json"), key=lambda p: p.stat().st_mtime):
        data = load(p)
        module = "experiments.mediated_patterns." + Path(data["command"][0]).stem
        commands.append(
            "OPENBLAS_NUM_THREADS=1 .venv/bin/python -m "
            + module
            + " "
            + " ".join(data["command"][1:])
        )
    (out / "commands.txt").write_text("\n".join(commands) + "\n")
    print(
        "runs",
        len(rows),
        "passes",
        sum(r["assessment"]["resolved_task_pass"] for r in rows),
        "CPU",
        sum(c["cpu_seconds"] for c in costs),
        "protected changes",
        changed,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", required=True)
    analyze(p.parse_args().output)
