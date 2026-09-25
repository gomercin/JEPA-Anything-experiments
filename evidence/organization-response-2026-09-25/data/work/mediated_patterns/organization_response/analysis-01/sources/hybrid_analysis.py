"""Saved hybrid evidence, complete restart, resource/access audit, and plots."""

import argparse
import hashlib
import json
import shlex
import time
from pathlib import Path

import numpy as np

from .explore import start_panel, write_json
from .hybrid_pair import PairHybrid
from .simulator import Config
from .two_way_hybrid import ROOT, readouts, score


def load(path):
    return json.loads((ROOT / path).read_text())


def sum_times(value):
    cpu = wall = 0.0
    if isinstance(value, dict):
        cpu = value.get("cpu_seconds", 0) + value.get("preparation_cpu_seconds", 0)
        wall = value.get("wall_seconds", 0) + value.get("preparation_wall_seconds", 0)
        for v in value.values():
            if isinstance(v, (dict, list)):
                c, w = sum_times(v)
                cpu += c
                wall += w
    elif isinstance(value, list):
        for v in value:
            c, w = sum_times(v)
            cpu += c
            wall += w
    return cpu, wall


def checkpoint(frozen, row, out):
    cfg = Config(**frozen["config"])
    basis = np.load(ROOT / "frozen-01/model.npz")["basis"]
    initial = np.load(ROOT / "fresh-01" / f"{row['run_id']}-initial.npz")["state"]
    st, sw = time.process_time(), time.perf_counter()
    model = PairHybrid.initialize(cfg, basis, initial)
    setup_cpu = time.process_time() - st
    model.pulse(row["c_position"], row["amplitude"])
    values = []
    resume = None
    difference = 0.0
    q_difference = 0.0
    stride = round(0.5 / cfg.dt)
    steps = round(80 / cfg.dt)
    for k in range(steps + 1):
        if k == round(20 / cfg.dt):
            model.save(out / "hybrid-checkpoint.npz")
            resume = PairHybrid.load(out / "hybrid-checkpoint.npz")
        if k % stride == 0:
            values.append(readouts(model.x, model.fields(), row["c_position"]))
            if resume is not None:
                difference = max(
                    difference, float(abs(model.fields() - resume.fields()).max())
                )
                q_difference = max(q_difference, float(abs(model.q - resume.q).max()))
        if k == steps:
            break
        model.step()
        if resume is not None:
            resume.step()
    values = np.array(values)
    stored = np.array(row["hybrid"]["absolute"])[:, 1]
    # Sham arrays enter evaluator scoring only, after autonomous prediction.
    response = values - np.array(row["hybrid"]["absolute"])[:, 0]
    cpu, wall = time.process_time() - st, time.perf_counter() - sw
    arrays = {}
    for key, value in vars(model).items():
        if isinstance(value, np.ndarray):
            arrays[key] = {"shape": list(value.shape), "bytes": value.nbytes}
        elif isinstance(value, tuple):
            arrays[key] = {
                "array_shapes": [list(v.shape) for v in value],
                "bytes": sum(v.nbytes for v in value),
            }
    st, sw = time.process_time(), time.perf_counter()
    for _ in range(1000):
        model.step()
    tick = {
        "cpu_seconds": (time.process_time() - st) / 1000,
        "wall_seconds": (time.perf_counter() - sw) / 1000,
    }
    # These extra ticks measure cost only, not validated forecast accuracy.
    return {
        "run_id": row["run_id"],
        "time": 20,
        "field_max_resume_difference": difference,
        "state_max_resume_difference": q_difference,
        "uninterrupted_vs_saved_readout_max": float(abs(values - stored).max()),
        "response_accuracy": score(row["truth"]["response"], response),
        "cpu_seconds": cpu,
        "wall_seconds": wall,
        "initialization_cpu_seconds": setup_cpu,
        "mean_tick_cost": tick,
        "arrays": arrays,
        "checkpoint_bytes": (out / "hybrid-checkpoint.npz").stat().st_size,
        "serialization_keys": np.load(
            out / "hybrid-checkpoint.npz", allow_pickle=False
        ).files,
        "field_access": "only initial snapshot; truth/sham accessed afterward by evaluator",
    }


def envelope_diagnostic(rows, basis):
    # Post-evaluation explanation of excursions, not a new acceptance gate.
    cfg = Config(**load("frozen-01/freeze.json")["config"])
    x = np.arange(cfg.n) * cfg.length / cfg.n - cfg.length / 2
    idx = np.flatnonzero((x >= -34) & (x < 28))
    extidx = np.setdiff1d(np.arange(cfg.n), idx)
    k = 2 * np.pi * np.fft.rfftfreq(cfg.n, cfg.length / cfg.n)
    linear = cfg.r - (1 - k * k) ** 2
    dev = []
    for r in load("calibration-01/results.json"):
        snap = np.load(ROOT / "calibration-01" / f"{r['run_id']}-snapshots.npz")[
            "fields"
        ]
        for branch in range(3):
            channels = []
            for u, m in snap[:, branch]:
                e = np.zeros(cfg.n)
                e[extidx] = u[extidx]
                tail = np.fft.irfft(linear * np.fft.rfft(e), n=cfg.n)[idx]
                channels.append(
                    np.r_[cfg.feedback * basis.T @ (m[idx] * u[idx]), basis.T @ tail]
                )
            a = np.array(channels)
            dev.append(a - a[0])
    dev = np.concatenate(dev)
    lo, hi = dev.min(axis=0), dev.max(axis=0)
    result = []
    for r in rows:
        a = np.array(
            [
                np.r_[p["incident_mediator_force"], p["incident_pattern_force"]]
                for p in r["hybrid"]["ports"]
            ]
        )
        a -= a[0]
        excursion = np.maximum(np.maximum(lo - a, a - hi), 0)
        result.append(
            {
                "run_id": r["run_id"],
                "absolute_port_excursion": r["port_excursion"],
                "centered_span_fraction": float(
                    (excursion / np.maximum(hi - lo, 1e-12)).max()
                ),
                "centered_max_absolute": float(excursion.max()),
                "centered_outside_fraction": float(
                    np.mean((excursion > 1e-12).any(axis=1))
                ),
            }
        )
    return result


def plots(rows, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    row = next(r for r in rows if r["run_id"] == "s2022-c39")
    cfg = Config(**load("frozen-01/freeze.json")["config"])
    x = np.arange(cfg.n) * cfg.length / cfg.n - cfg.length / 2
    initial = np.load(ROOT / "fresh-01" / f"{row['run_id']}-initial.npz")["state"]
    t = np.array(row["truth"]["time"])
    truth = np.array(row["truth"]["response"])
    pred = np.array(row["hybrid"]["response"])
    rt = np.array(row["truth"]["return_contrast"])
    rp = np.array(row["hybrid"]["return_contrast"])
    fig, ax = plt.subplots(3, 2, figsize=(12, 10))
    ax[0, 0].plot(x, initial[0], label="Pattern u", color="#355f98")
    ax[0, 0].plot(x, initial[1], label="Mediator m", color="#ab557c")
    ax[0, 0].axvspan(-34, 28, alpha=0.09, color="#187b72", label="AB projection region")
    ax[0, 0].axvspan(
        31, 47, alpha=0.12, color="#d78724", label="C pulse support at t=0"
    )
    ax[0, 0].set_xlim(-40, 60)
    ax[0, 0].set_xlabel("Position")
    ax[0, 0].set_title("Prepared triple; +0.075 relative pulse at C=39")
    ax[0, 0].legend(fontsize=8)
    ax[0, 1].plot(t, truth[:, 2], color="black", label="Full fields")
    ax[0, 1].plot(t, pred[:, 2], "--", color="#187b72", label="20-mode pair hybrid")
    ax[0, 1].set_xlim(0, 20)
    ax[0, 1].set_title("C mass response: mostly its local relaxation")
    ax[0, 1].set_ylabel("Pulse minus own sham")
    for (i, j), y, p, title, scale in [
        ((1, 0), truth[:, 1], pred[:, 1], "Pair B source-region mass response", 1e4),
        (
            (1, 1),
            rt[:, 2],
            rp[:, 2],
            "C return contrast: full minus matched AB replay",
            1e6,
        ),
    ]:
        ax[i, j].plot(t, y * scale, color="black", label="Full fields")
        ax[i, j].plot(
            t,
            p * scale,
            "--",
            color="#187b72",
            label="Closed-loop hybrid contrast" if j else "Closed-loop hybrid",
        )
        ax[i, j].set_title(title)
        ax[i, j].set_ylabel(f"Signed response × {scale:.0e}")
    ax[1, 1].axhline(
        0, color="#888888", ls=":", label="Zero return contrast comparator"
    )
    ax[2, 0].plot(t, (truth[:, 1] - pred[:, 1]) * 1e6, color="#187b72")
    ax[2, 0].set_title("Pair response residual")
    ax[2, 0].set_ylabel("Truth − hybrid × 10⁶")
    ax[2, 1].plot(t, (rt[:, 2] - rp[:, 2]) * 1e9, color="#187b72")
    ax[2, 1].set_title("Return-contrast residual")
    ax[2, 1].set_ylabel("Truth − hybrid × 10⁹")
    for i in range(3):
        for j in range(2):
            ax[i, j].grid(alpha=0.2)
            if (i, j) != (0, 0):
                ax[i, j].axvline(0, color="#d78724", lw=1)
                ax[i, j].set_xlabel("Time since stimulus at C")
    ax[0, 1].legend(fontsize=8)
    ax[1, 1].legend(fontsize=8)
    fig.suptitle(
        "Fresh seed 2022: AB fine evolution removed, C and shared mediator remain resolved",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out / "two_way_response.png", dpi=160)
    plt.close(fig)


def analyze(output):
    out = start_panel(
        output,
        {
            "purpose": "reporting and complete hybrid restart; no refitting",
            "fresh_source": "fresh-01",
            "frozen_model": "frozen-01",
            "checkpoint_case": "s2022-c39",
        },
    )
    frozen = load("frozen-01/freeze.json")
    rows = load("fresh-01/results.json")
    basis = np.load(ROOT / "frozen-01/model.npz")["basis"]
    metrics = []
    for r in rows:
        y = np.array(r["truth"]["response"])
        p = np.array(r["hybrid"]["response"])
        contrast = np.array(r["truth"]["return_contrast"])
        # Matched replay is an evaluator-only comparator, not a deployable model.
        controlled = (
            np.array(r["truth"]["absolute"])[:, 2]
            - np.array(r["truth"]["absolute"])[:, 0]
        )
        t = np.array(r["truth"]["time"])
        early = t <= 5
        centers = np.array(r["truth"]["center"])
        metrics.append(
            {
                "run_id": r["run_id"],
                "scores": r["scores"],
                "no_return_C_response": score(y[:, 2:3], controlled[:, 2:3]),
                "zero_return_contrast": score(contrast[:, 2:3], np.zeros((len(t), 1))),
                "early_C_return_signal_rms": float(
                    np.sqrt(np.mean(contrast[early, 2] ** 2))
                ),
                "early_C_return_max_error": float(
                    abs(
                        contrast[early, 2]
                        - np.array(r["hybrid"]["return_contrast"])[early, 2]
                    ).max()
                ),
                "center_drift_max": float(abs(centers - centers[0]).max()),
                "AB_separation_drift_max": float(
                    abs(
                        (centers[:, 1] - centers[:, 0])
                        - (centers[0, 1] - centers[0, 0])
                    ).max()
                ),
                "C_response_error_rms": float(
                    np.sqrt(np.mean((y[:, 2] - p[:, 2]) ** 2))
                ),
            }
        )
    check = checkpoint(frozen, next(r for r in rows if r["run_id"] == "s2022-c39"), out)
    envelope = envelope_diagnostic(rows, basis)
    costs = []
    paths = (
        list(ROOT.glob("*/results.json"))
        + list(ROOT.glob("*/fit.json"))
        + list(ROOT.glob("*/exchange-refinement.json"))
    )
    for path in sorted(paths):
        c, w = sum_times(json.loads(path.read_text()))
        if c or w:
            costs.append({"file": str(path), "cpu_seconds": c, "wall_seconds": w})
    costs.append(
        {
            "file": "this analysis restart",
            "cpu_seconds": check["cpu_seconds"],
            "wall_seconds": check["wall_seconds"],
        }
    )
    protected = load("protected.json")
    changed = [
        p
        for p, h in protected.items()
        if not Path(p).exists() or hashlib.sha256(Path(p).read_bytes()).hexdigest() != h
    ]
    summary = {
        "status": "EXPLORATORY",
        "passed": sum(r["scores"]["pass"] for r in rows),
        "preparations": len(rows),
        "seeds": frozen["seeds"],
        "metrics": metrics,
        "checkpoint": check,
        "envelope_diagnostic": envelope,
        "costs": costs,
        "cpu_seconds_total": sum(r["cpu_seconds"] for r in costs),
        "wall_seconds_sum": sum(r["wall_seconds"] for r in costs),
        "protected": {"count": len(protected), "changed": changed},
        "resources": {
            "pair_dynamic": 20,
            "removed_pair_grid": 248,
            "exterior_u_grid": 520,
            "mediator_grid": 768,
            "full_dynamic_scalars": 1536,
            "hybrid_physical_dynamic_scalars": 1308,
            "stored_dynamic_float64_equivalents": 1310,
            "clock": 1,
            "pair_static_template": 248,
            "pair_basis_coefficients": 4960,
            "incoming_force_channels": 40,
            "outgoing_coordinates": 20,
            "force_projection": "state-dependent exact equation projection; dense global linear exchange integrated jointly",
            "shared_field_source": "template plus basis*z squared once; no old response-kernel injection",
            "no_speedup_claim": True,
        },
        "limits": [
            "new spatial pair model, not old pulse-model reuse",
            "fixed AB geometry and tested C positions only",
            "early return below resolution",
            "input channel excursions retained",
            "mediator-specific AB-replay contrast, not every return pathway",
        ],
    }
    write_json(out / "summary.json", summary)
    commands = []
    for p in sorted(ROOT.glob("*/protocol.json"), key=lambda p: p.stat().st_mtime):
        argv = json.loads(p.read_text())["command"]
        module = "experiments.mediated_patterns." + Path(argv[0]).stem
        commands.append(
            "OPENBLAS_NUM_THREADS=1 .venv/bin/python -m "
            + module
            + " "
            + shlex.join(argv[1:])
        )
    (out / "commands.txt").write_text("\n".join(commands) + "\n")
    plots(rows, out)
    print(
        "Passed",
        summary["passed"],
        "/",
        len(rows),
        "CPU",
        summary["cpu_seconds_total"],
        "protected changed",
        changed,
    )
    print(
        "Checkpoint",
        check["field_max_resume_difference"],
        "saved",
        check["uninterrupted_vs_saved_readout_max"],
    )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument("--output", required=True)
    analyze(p.parse_args().output)


if __name__ == "__main__":
    main()
