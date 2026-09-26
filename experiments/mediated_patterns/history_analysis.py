"""Saved-data resolution, prospective prediction freeze, and history diagnostics."""

import argparse
import os
from pathlib import Path

import numpy as np
from scipy.signal import resample

from . import history_conditioned_transmission as h
from .explore import write_json
from .organization_response import rms
from .relay_analysis import score
from .simulator import Field


def case(root, seed, wait, written, suffix="", prediction=False):
    name = f"s{seed}" + ("-" + suffix if suffix else "")
    name += f"-t{wait:g}-" + ("written" if written else "unwritten")
    if prediction:
        name += "-prediction"
    return h.read_saved(root, name)


def pair(root, seed, wait, suffix="", prediction=False):
    return [
        case(root, seed, wait, written, suffix, prediction) for written in (False, True)
    ]


def qualify(meta):
    if not meta["qualified"] or meta["sham_field_max"] > 1e-12:
        raise RuntimeError("Saved regime or history-matched sham failed")


def state_summary(arrays, meta):
    """Present-state diagnostics; no replacement state is evolved."""
    f = Field(h.CFG)
    dx = f.c.length / f.c.n
    rows = []
    for wait in (0.0, 10.0, 50.0, 100.0, 180.0):
        k = int(np.flatnonzero(arrays["time"] == wait)[0])
        g0, gw = meta["geometry"][k]
        rows.append(
            {
                "time": wait,
                "changes": arrays["changes"][k, 0].tolist(),
                "center_difference": (np.array(gw["center"]) - g0["center"]).tolist(),
                "mass_difference": (np.array(gw["mass"]) - g0["mass"]).tolist(),
                "width_difference": (np.array(gw["width"]) - g0["width"]).tolist(),
                "source_total_difference": gw["source_total"] - g0["source_total"],
                "mediator_mean_difference": (
                    np.array(gw["mediator_means"]) - g0["mediator_means"]
                ).tolist(),
                "write_only_C": (
                    arrays["absolute"][k, 1, 2] - arrays["absolute"][k, 0, 2]
                ).tolist(),
            }
        )
    # Exact homogeneous propagator of the actual spectral mediator operator.
    # The remainder equals the source-history integral (up to the trajectory's
    # ETDRK4 discretization); it is not an independent causal response fraction.
    m50 = np.diff(h.checkpoint(arrays, 50.0)[:, 1], axis=0)[0]
    m100 = np.diff(h.checkpoint(arrays, 100.0)[:, 1], axis=0)[0]
    inherited = np.fft.irfft(np.exp(f.linear[1] * 50) * np.fft.rfft(m50), n=f.c.n)
    sourced = m100 - inherited
    norm = lambda v: float(np.sqrt(np.sum(v * v) * dx))
    return {
        "states": rows,
        "mediator_50_to_100": {
            "delta_m_50_l2": norm(m50),
            "delta_m_100_l2": norm(m100),
            "inherited_l2": norm(inherited),
            "continuing_source_remainder_l2": norm(sourced),
            "operator": "(D partial_xx - 1)/tau",
            "interpretation": "conditional decomposition of recorded coupled histories, not causal fractions",
        },
        "write": meta["writes"][0],
        "qualified": meta["qualified"],
    }


def freeze(out, data, refinement):
    floors = np.full((3, 3), 1e-12)
    refinement_rows, state_rows = [], []
    coarse_history, _ = h.read_saved(data, f"s{h.DEV[0]}-history")
    for suffix in ("half-dt", "double-n"):
        fine_history, _ = h.read_saved(refinement, f"s{h.DEV[0]}-{suffix}-history")
        difference = fine_history["checkpoints"] - resample(
            coarse_history["checkpoints"],
            fine_history["checkpoints"].shape[-1],
            axis=-1,
        )
        state_rows.append(
            {
                "refinement": suffix,
                "checkpoint_max_abs_error_by_field": abs(difference)
                .max(axis=(0, 1, 3))
                .tolist(),
                "change_descriptor_max_error": abs(
                    fine_history["changes"] - coarse_history["changes"]
                )
                .max(axis=(0, 1))
                .tolist(),
            }
        )
        for wait in h.WAITS:
            coarse, fine = (
                pair(data, h.DEV[0], wait),
                pair(refinement, h.DEV[0], wait, suffix),
            )
            separate_errors, direct = [], []
            for (a, m), (b, fm) in zip(coarse, fine, strict=True):
                qualify(m)
                qualify(fm)
                df = b["full"][:, 0] - a["full"][:, 0]
                dk = b["cut"][:, 0] - a["cut"][:, 0]
                separate_errors.append(rms(df, axis=0) + rms(dk, axis=0))
                direct.append((df, dk))
            conservative = sum(separate_errors)
            floors = np.maximum(floors, 5 * conservative)
            refinement_rows.append(
                {
                    "wait": wait,
                    "refinement": suffix,
                    "sum_branch_response_errors": conservative.tolist(),
                    "delta_R_error": rms(direct[1][0] - direct[0][0], axis=0).tolist(),
                    "delta_Q_error": rms(
                        (direct[1][0] - direct[1][1]) - (direct[0][0] - direct[0][1]),
                        axis=0,
                    ).tolist(),
                }
            )
    checks = []
    for wait in h.WAITS:
        paired = pair(data, h.DEV[0], wait)
        for written, (a, m) in enumerate(paired):
            qualify(m)
            for j, eps in enumerate(m["amplitudes"]):
                for kind in ("full", "cut", "qb"):
                    checks.append(
                        {
                            "wait": wait,
                            "written": bool(written),
                            "amplitude": eps,
                            "kind": kind,
                            **score(
                                a[kind][:, j, 2, :2],
                                a["tangent_" + kind][:, 2, :2] * eps / h.EPS,
                                floors[2, :2],
                            ),
                        }
                    )
        for j, eps in enumerate(paired[0][1]["amplitudes"]):
            for kind in ("full", "qb"):
                truth = h.history_contrasts(
                    *[x[0] for x in paired], key=kind, input_index=j
                )[2][:, 2, :2]
                prediction = (
                    (paired[1][0]["tangent_" + kind] - paired[0][0]["tangent_" + kind])[
                        :, 2, :2
                    ]
                    * eps
                    / h.EPS
                )
                checks.append(
                    {
                        "wait": wait,
                        "amplitude": eps,
                        "kind": "delta_" + kind,
                        **score(truth, prediction, floors[2, :2]),
                    }
                )
    result = {
        "source_hashes": {
            p.name: h.digest(p) for p in Path(h.__file__).parent.glob("*.py")
        },
        "development_seeds": list(h.DEV),
        "fresh_seeds": list(h.FRESH),
        "configuration": vars(h.CFG),
        "B": h.B,
        "write_kind": "odd",
        "write_amplitude": h.WRITE,
        "probe_amplitude": h.EPS,
        "waits": list(h.WAITS),
        "horizon": 80.0,
        "floors": floors.tolist(),
        "floor_rule": "max(1e-12,5*max_refinement_wait(sum_2histories(RMS(full error)+RMS(cut error))))",
        "prediction_gates": {"response": 0.02, "history_difference": 0.10},
        "retention_rule": {
            "late_waits": [50.0, 100.0],
            "delta_R_rms_ratio_min": 0.5,
            "displacement_ratio_min": 0.75,
            "requirements": "Delta_R above shared floor at both late waits; same sign correlation; translation-like present-state difference remains; no indefinite retention claim",
        },
        "refinements": refinement_rows,
        "state_refinements": state_rows,
        "development_tangent_checks": checks,
        "inputs": {
            str(p): h.digest(p)
            for root in (data, refinement)
            for p in sorted(root.iterdir())
            if p.is_file()
        },
    }
    write_json(out / "freeze.json", result)
    print(
        {"C_shared_floor": floors[2].tolist(), "development_checks": checks}, flush=True
    )


def analyze(out, data, frozen):
    fz = h.load(frozen / "freeze.json")
    floors = np.asarray(fz["floors"])
    rows, histories, comparisons = [], [], []
    for seed in h.FRESH:
        a, m = h.read_saved(data, f"s{seed}-history")
        histories.append({"seed": seed, **state_summary(a, m)})
        late = {}
        for wait in h.WAITS:
            paired = pair(data, seed, wait)
            predictions = pair(data, seed, wait, prediction=True)
            for written, ((a, m), (p, pm)) in enumerate(
                zip(paired, predictions, strict=True)
            ):
                qualify(m)
                qualify(pm)
                if m["initial_sha256"] != pm["initial_sha256"]:
                    raise RuntimeError(
                        "Prediction/reference have different pre-probe state"
                    )
                for kind in ("full", "cut", "qb"):
                    metric = score(
                        a[kind][:, 0, 2, :2],
                        p["tangent_" + kind][:, 2, :2],
                        floors[2, :2],
                    )
                    gate = 0.10 if kind == "qb" else 0.02
                    rows.append(
                        {
                            "seed": seed,
                            "wait": wait,
                            "written": bool(written),
                            "kind": kind,
                            "gate": gate,
                            "passed": bool(
                                np.all(
                                    (np.asarray(metric["relative"]) <= gate)
                                    | (np.asarray(metric["rmse"]) <= floors[2, :2])
                                )
                            ),
                            **metric,
                        }
                    )
            aa = [x[0] for x in paired]
            r0, rw, delta = h.history_contrasts(*aa)
            q0, qw, dq = h.history_contrasts(*aa, key="qb")
            pp = [x[0] for x in predictions]
            scores = {}
            for kind, truth in (("full", delta), ("qb", dq)):
                predicted = pp[1]["tangent_" + kind] - pp[0]["tangent_" + kind]
                metric = score(truth[:, 2, :2], predicted[:, 2, :2], floors[2, :2])
                scores[kind] = metric
                rows.append(
                    {
                        "seed": seed,
                        "wait": wait,
                        "kind": "delta_" + kind,
                        "gate": 0.10,
                        "passed": bool(
                            np.all(
                                (np.asarray(metric["relative"]) <= 0.10)
                                | (np.asarray(metric["rmse"]) <= floors[2, :2])
                            )
                        ),
                        **metric,
                    }
                )
            # A per-readout scalar is descriptive only; no new held-out fit claim.
            gain = np.sum(r0[:, 2, :2] * rw[:, 2, :2], axis=0) / np.sum(
                r0[:, 2, :2] ** 2, axis=0
            )
            outgoing_q = []
            for a in aa:
                o = a["outgoing"]
                outgoing_q.append((o[:, 2] - o[:, 0]) - (o[:, 3] - o[:, 1]))
            comparison = {
                "seed": seed,
                "wait": wait,
                "R0_rms": rms(r0, axis=0).tolist(),
                "RW_rms": rms(rw, axis=0).tolist(),
                "delta_R_rms": rms(delta, axis=0).tolist(),
                "relative_change": (
                    rms(delta, axis=0) / np.maximum(rms(r0, axis=0), 1e-30)
                ).tolist(),
                "Q0_rms": rms(q0, axis=0).tolist(),
                "QW_rms": rms(qw, axis=0).tolist(),
                "delta_Q_rms": rms(dq, axis=0).tolist(),
                "delta_R_resolved": (rms(delta, axis=0) > floors).tolist(),
                "delta_Q_resolved": (rms(dq, axis=0) > floors).tolist(),
                "C_descriptive_gains": gain.tolist(),
                "C_after_gain_relative_residual": (
                    rms(rw[:, 2, :2] - r0[:, 2, :2] * gain, axis=0)
                    / rms(rw[:, 2, :2], axis=0)
                ).tolist(),
                "A_outgoing_Q_rms": [float(rms(x)) for x in outgoing_q],
                "probe_jumps": [m["jumps"][0] for _, m in paired],
                "sham_error": max(m["sham_field_max"] for _, m in paired),
            }
            comparisons.append(comparison)
            if wait in (50.0, 100.0):
                late[wait] = delta[:, 2, :2]
        retention_ratio = rms(late[100.0], axis=0) / rms(late[50.0], axis=0)
        correlation = np.sum(late[100.0] * late[50.0], axis=0) / np.sqrt(
            np.sum(late[100.0] ** 2, axis=0) * np.sum(late[50.0] ** 2, axis=0)
        )
        histories[-1]["late_Delta_R_ratio"] = retention_ratio.tolist()
        histories[-1]["late_Delta_R_correlation"] = correlation.tolist()
        positions = histories[-1]["states"]
        displacement_ratio = abs(
            positions[3]["changes"][2] / positions[2]["changes"][2]
        )
        histories[-1]["late_displacement_ratio"] = displacement_ratio
        histories[-1]["finite_retention_qualified"] = bool(
            (retention_ratio >= 0.5).all()
            and (correlation > 0).all()
            and displacement_ratio >= 0.75
            and all((rms(v, axis=0) > floors[2, :2]).all() for v in late.values())
        )
    summary = {
        "histories": histories,
        "comparisons": comparisons,
        "prediction_checks": rows,
        "floors": floors.tolist(),
        "freeze_sha256": h.digest(frozen / "freeze.json"),
    }
    write_json(out / "summary.json", summary)
    figure(out, data, frozen)
    print(
        {
            "retention": [x["finite_retention_qualified"] for x in histories],
            "prediction_failures": [x for x in rows if not x["passed"]],
            "late_comparisons": [x for x in comparisons if x["wait"] == 100.0],
        },
        flush=True,
    )


def figure(out, data, frozen):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    floors = np.asarray(h.load(frozen / "freeze.json")["floors"])
    seed = h.FRESH[0]
    hist, _meta = h.read_saved(data, f"s{seed}-history")
    f = Field(h.CFG)
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    ax = axes[0, 0]
    ax.plot(f.x, h.WRITE * h.write_profile(f, "odd"))
    ax.set(
        xlim=(-15, 7),
        title="Supplied B write at time 0",
        xlabel="Physical x",
        ylabel="Additive u",
    )
    ax = axes[0, 1]
    ax.plot(hist["time"], hist["changes"][:, 0, 2], label="Translation projection")
    ax.plot(hist["time"], hist["changes"][:, 0, 3], label="Local residual L2")
    for wait in h.WAITS:
        ax.axvline(wait, color=".7", lw=0.6)
    ax.legend(fontsize=8)
    ax.set(title="Written minus unwritten state", xlabel="Absolute time")
    ax = axes[0, 2]
    for j, label in enumerate(("u", "m")):
        ax.semilogy(
            hist["time"], hist["changes"][:, 0, j], label=f"Difference {label} L2"
        )
    ax.legend(fontsize=8)
    ax.set(title="Present-state differences persist", xlabel="Absolute time")
    colors = ("#3975a6", "#d07a25", "#528747")
    for col, (node, index) in enumerate((("A", 0), ("B", 1), ("C", 2))):
        paired = pair(data, seed, 100.0)
        for written, (a, _) in enumerate(paired):
            ax = axes[1, col]
            label = "Written" if written else "Unwritten"
            ax.plot(
                a["time"],
                a["full"][:, 0, index, 0],
                color=colors[written],
                label=label + " full",
            )
            ax.plot(
                a["time"],
                a["cut"][:, 0, index, 0],
                color=colors[written],
                ls="--",
                label=label + " cut",
            )
        ax.set(title=f"{node} mass response; wait 100", xlabel="Time after probe")
        ax.legend(fontsize=7)
    for wait, color in zip(h.WAITS, colors, strict=True):
        paired = pair(data, seed, wait)
        aa = [x[0] for x in paired]
        delta = h.history_contrasts(*aa)[2]
        dq = h.history_contrasts(*aa, key="qb")[2]
        for col, (signal, readout, title) in enumerate(
            (
                (delta, 0, "C mass Delta_R"),
                (delta, 1, "C moment Delta_R"),
                (dq, 0, "C mass Delta_Q"),
            )
        ):
            ax = axes[2, col]
            ax.plot(
                aa[0]["time"],
                signal[:, 2, readout],
                color=color,
                label=f"wait {wait:g}",
            )
            ax.set(title=title, xlabel="Time after probe")
    for col, readout in enumerate((0, 1, 0)):
        ax = axes[2, col]
        floor = floors[2, readout]
        ax.axhspan(
            -floor, floor, color=".6", alpha=0.4, label="Shared RMS resolution scale"
        )
        ax.axhline(0, color=".5", lw=0.5)
        ax.legend(fontsize=7)
    fig.suptitle(
        f"Actual matched histories, fresh preparation {seed}; other seeds retained in tables"
    )
    fig.tight_layout()
    fig.savefig(out / "history.png", dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["freeze", "analyze"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refinement", type=Path)
    parser.add_argument("--freeze", type=Path)
    a = parser.parse_args()
    if a.stage == "freeze" and a.refinement is None:
        parser.error("--refinement required")
    if a.stage == "analyze" and a.freeze is None:
        parser.error("--freeze required")
    h.ROOT.mkdir(parents=True, exist_ok=True)
    lock = h.ROOT / ".active"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    budget = h.HistoryBudget(
        sum(h.load(x)["cpu_seconds"] for x in h.ROOT.glob("*/budget.json"))
    )
    status = "FAILED"
    started = False
    try:
        h.start(a.output, a.stage, budget)
        started = True
        if a.stage == "freeze":
            freeze(a.output, a.data, a.refinement)
        else:
            analyze(a.output, a.data, a.freeze)
        budget.check()
        status = "COMPLETE"
    finally:
        if started:
            write_json(
                a.output / "budget.json", dict(**budget.receipt(), status=status)
            )
        lock.unlink()


if __name__ == "__main__":
    main()
