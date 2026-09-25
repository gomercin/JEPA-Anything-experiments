"""Offline diagnostics and figures; no channel back into deployed models."""

import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from . import dual_reduction as task
from .dual_pair import DualHybrid, c_indices
from .dual_reduction import OLD, ROOT, charge, frozen_c
from .explore import start_panel, write_json
from .hybrid_pair import PairHybrid
from .measurements import regions
from .repeated_hybrid import baseline, geometry_scores, load
from .two_way_hybrid import readouts


def projection(output, qualification, order):
    out = start_panel(
        output, {"stage": "FR failure best-in-basis diagnosis only", "order": order}
    )
    _, cfg, _ = baseline()
    x = np.arange(cfg.n) * cfg.length / cfg.n - cfg.length / 2
    b = np.load(ROOT / "calibration-01/basis.npz")["basis"][:, :order]
    rows = []
    for r in load(Path(qualification) / "results.json"):
        tag, name = r["preparation_id"], r["name"]
        pos = float(tag.split("-c")[1])
        idx = c_indices(x, pos)
        initial = np.load(OLD / "fresh-01" / f"{tag}-initial.npz")["state"]
        snap = np.load(ROOT / "c-reference" / f"{tag}-{name}.npz")["fields"]
        local = snap[:, :, 0, idx]
        delta = local - initial[0, idx]
        projected = initial[0, idx] + (delta @ b) @ b.T
        v = []
        for k, states in enumerate(snap):
            values = []
            for j, state in enumerate(states):
                p = state.copy()
                p[0, idx] = projected[k, j]
                values.append(readouts(x, p, pos))
            v.append(values)
        v = np.array(v)
        truth = np.array(load(r["FF_path"])["absolute"])
        error = v[:, 1, 2] - v[:, 0, 2] - truth[:, 1, 2] + truth[:, 0, 2]
        rows.append(
            {
                "preparation": tag,
                "schedule": name,
                "discarded_C_field_relative_rms": float(
                    np.linalg.norm(projected - local) / np.linalg.norm(local)
                ),
                "projected_C_response_max_error": float(abs(error).max()),
                "evolved": r["assessment"],
            }
        )
    write_json(out / "results.json", rows)


def c_coverage(cb, cfg):
    x = np.arange(cfg.n) * cfg.length / cfg.n - cfg.length / 2
    wave = 2 * np.pi * np.fft.rfftfreq(cfg.n, cfg.length / cfg.n)
    lu = cfg.r - (1 - wave**2) ** 2
    blocks = []
    for row in load(OLD / "calibration-01/results.json"):
        idx = c_indices(x, row["c_position"])
        fields = np.load(OLD / "calibration-01" / f"{row['run_id']}-snapshots.npz")[
            "fields"
        ]
        for j in range(3):
            values = []
            for u, m in fields[:, j]:
                other = u.copy()
                other[idx] = 0
                tail = np.fft.irfft(lu * np.fft.rfft(other), n=cfg.n)[idx]
                values.append(
                    np.r_[cfg.feedback * cb.T @ (m[idx] * u[idx]), cb.T @ tail]
                )
            blocks.append(np.array(values))
    raw = np.concatenate(blocks)
    centered = np.concatenate([b - b[0] for b in blocks])
    scale = np.maximum(centered.std(axis=0), 1e-10)
    rates = abs(np.concatenate([np.diff(b, axis=0) / 0.5 for b in blocks])).max(axis=0)
    return {
        "lo": raw.min(axis=0),
        "hi": raw.max(axis=0),
        "scale": scale,
        "rates": np.maximum(rates, 1e-12),
        "cloud": centered / scale,
    }


def coverage(record, training):
    p = np.array(record["C_ports"])
    exc = np.maximum(np.maximum(training["lo"] - p, p - training["hi"]), 0)
    joint = cKDTree(training["cloud"]).query((p - p[0]) / training["scale"])[
        0
    ] / np.sqrt(p.shape[1])
    return {
        "max_training_span_excursion": float(
            (exc / np.maximum(training["hi"] - training["lo"], 1e-12)).max()
        ),
        "outside_sample_fraction": float((exc > 1e-12).any(axis=1).mean()),
        "max_standardized_joint_distance": float(joint.max()),
        "max_rate_training_ratio": float(
            (abs(np.diff(p, axis=0)) / 0.5 / training["rates"]).max()
        ),
        "clipped": False,
    }


def summarize(output):
    out = start_panel(
        output, {"stage": "post-freeze reporting only; no fitting or selection"}
    )
    f, cb = frozen_c()
    _, cfg, _ = baseline()
    rows = load(task.PANEL / "results.json")
    training = c_coverage(cb, cfg)
    kinds = ["RF", "FR", "RR"] + (["RR_original"] if "RR_original" in rows[0] else [])
    summaries = []
    for row in rows:
        item = {
            "run_id": row["run_id"],
            "preparation": row["preparation_id"],
            "models": {},
        }
        for kind in kinds:
            a = row["assessments"][kind]
            item["models"][kind] = {
                "pass": a["resolved_task_pass"],
                "status": a["status"],
                "failures": a["failures"],
                "whole": a["windows"]["whole"],
                "additional": a["additional"],
                "worst_resolved_return_nrmse": max(
                    [
                        s["C_return"]["nrmse"]
                        for label, s in a["windows"].items()
                        if s["C_return"]["resolved"] and not label.endswith("early5")
                    ],
                    default=None,
                ),
                "unresolved": a["unresolved"],
                "geometry": geometry_scores(row[kind]),
                "cpu_seconds": row[kind]["cpu_seconds"],
                "wall_seconds": row[kind]["wall_seconds"],
                "sham_replay_max": row[kind]["sham_replay_max"],
            }
            if kind in ["FR", "RR"]:
                item["models"][kind]["coverage"] = coverage(row[kind], training)
        item["FF_cpu_seconds"] = row["FF"]["cpu_seconds"]
        local = np.array(row["FF"]["response"])[:, 2]
        ret = np.array(row["FF"]["return_contrast"])[:, 2]
        t = np.array(row["FF"]["time"])
        item["no_return_C_response_nrmse"] = {
            name: float(np.linalg.norm(ret[mask]) / np.linalg.norm(local[mask]))
            for name, mask in [("whole", t >= 0), ("final20", t >= t[-1] - 20)]
        }
        # Evaluator-only clean C geometry is recoverable from saved local fields:
        # the entire fixed readout support is inside this region.
        x = np.arange(cfg.n) * cfg.length / cfg.n - cfg.length / 2
        idx = c_indices(x, row["position"])
        w = regions(x, [row["position"]])[0, idx]
        u = np.array(row["FF"]["C_fields"])[:, 1]
        weights = w * u * u
        center = (weights * x[idx]).sum(axis=1) / weights.sum(axis=1)
        width = np.sqrt(
            (weights * (x[idx] - center[:, None]) ** 2).sum(axis=1)
            / weights.sum(axis=1)
        )
        item["truth_C_geometry"] = {
            "max_center_drift": float(abs(center - center[0]).max()),
            "max_width_change": float(abs(width - width[0]).max()),
        }
        item["nonadditivity"] = {}
        for key, col in [("response", 2), ("return_contrast", 2)]:
            signed = sum(
                sign * np.array(row[k][key])[:, col]
                for k, sign in [("RR", 1), ("RF", -1), ("FR", -1), ("FF", 1)]
            )
            item["nonadditivity"][key] = {
                "rmse": float(np.sqrt(np.mean(signed**2))),
                "max": float(abs(signed).max()),
                "signed_trace": signed.tolist(),
            }
        summaries.append(item)
    aggregates = {}
    for kind in kinds:
        models = [r["models"][kind] for r in summaries]
        aggregates[kind] = {
            "passed": sum(m["pass"] for m in models),
            "cases": len(models),
            "gate_failed": sum(bool(m["failures"]) for m in models),
            "partial_unresolved": sum(
                not m["pass"] and not m["failures"] for m in models
            ),
            "worst_whole": {
                key: max(
                    m["whole"][key]["nrmse"]
                    for m in models
                    if m["whole"][key]["resolved"]
                )
                for key in ["B_response", "C_response", "C_return"]
            },
            "worst_return_window": max(
                m["worst_resolved_return_nrmse"]
                for m in models
                if m["worst_resolved_return_nrmse"] is not None
            ),
            "max_absolute": {
                key: max(m["whole"][key]["max_absolute"] for m in models)
                for key in ["B_response", "C_response", "C_return"]
            },
            "cpu_seconds": sum(m["cpu_seconds"] for m in models),
        }
    write_json(
        out / "results.json",
        {
            "freeze": f,
            "aggregate": aggregates,
            "cases": summaries,
            "coverage_contract": "Training-only centered standardized nearest-neighbor distance and marginal/rate excursions; diagnostics, not error bounds.",
        },
    )
    plot(rows[1], out)


def plot(row, out):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = np.array(row["FF"]["time"])
    colors = {"FF": "black", "RF": "#e08a20", "FR": "#2374b0", "RR": "#b63e78"}
    styles = {"FF": "-", "RF": ":", "FR": "--", "RR": "-."}
    if "RR_original" in row:
        colors["RR_original"], styles["RR_original"] = "#999999", "--"
    fig, ax = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for kind, color in colors.items():
        y = np.array(row[kind]["response"])
        ax[0].plot(t, y[:, 2], styles[kind], color=color, label=kind, lw=1.7)
        ax[1].plot(t, y[:, 1], styles[kind], color=color, label=kind, lw=1.7)
        if kind != "FF":
            ax[2].plot(
                t,
                y[:, 2] - np.array(row["FF"]["response"])[:, 2],
                styles[kind],
                color=color,
                label=kind,
            )
    for a in ax:
        for when, amplitude in row["plan"]["events"]:
            a.axvline(when, color="gray", lw=0.8)
        a.grid(alpha=0.2)
    ax[0].legend(ncol=4)
    ax[0].set_ylabel("C mass response")
    ax[1].set_ylabel("B mass response")
    ax[2].set_ylabel("C signed error")
    ax[2].set_xlabel("Model time")
    ax[0].set_title(row["run_id"] + " | " + str(row["plan"]["events"]))
    fig.tight_layout()
    fig.savefig(out / "responses.png", dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ref = np.array(row["FF"]["return_contrast"])[:, 2]
    for kind, color in colors.items():
        value = np.array(row[kind]["return_contrast"])[:, 2]
        ax[0].plot(t, value, styles[kind], color=color, label=kind, lw=1.7)
        if kind != "FF":
            ax[1].plot(t, value - ref, styles[kind], color=color, label=kind)
    for a in ax:
        a.fill_between(
            t,
            0,
            1,
            where=abs(ref) < 2e-7,
            transform=a.get_xaxis_transform(),
            color="gray",
            alpha=0.15,
        )
        for when, _ in row["plan"]["events"]:
            a.axvline(when, color="gray", lw=0.8)
        a.grid(alpha=0.2)
    ax[0].legend(ncol=4)
    ax[0].set_ylabel("Selected return contrast")
    ax[0].set_title(
        "Own-sham matched controls; gray: instantaneous signal below 2e-7 (window scores use RMS)"
    )
    ax[1].set_ylabel("Signed contrast error")
    ax[1].set_xlabel("Model time")
    fig.tight_layout()
    fig.savefig(out / "return_contrast.png", dpi=160)
    plt.close(fig)


def preservation(output):
    out = Path(output)
    manifest = load(ROOT / "protected.json")
    hashes = manifest.get("hashes", manifest)
    changed = []
    missing = []
    for name, digest in hashes.items():
        p = Path(name)
        if not p.exists():
            missing.append(name)
        elif hashlib.sha256(p.read_bytes()).hexdigest() != digest:
            changed.append(name)
    write_json(
        out / "preservation.json",
        {
            "count": len(hashes),
            "changed": changed,
            "missing": missing,
            "authorized": "Only additive experiment README handoff is authorized; old results/models/source must match.",
        },
    )


def resources(output):
    out = start_panel(
        output, {"stage": "static and evolving array accounting, no fitting"}
    )
    _, cb = frozen_c()
    _, cfg, ab = baseline()
    initial = np.load(OLD / "fresh-01/s2011-c39-initial.npz")["state"]
    rows = {}
    st, sw = time.process_time(), time.perf_counter()
    for kind in ["RF", "FR", "RR"]:
        obj = (
            PairHybrid.initialize(cfg, ab, initial)
            if kind == "RF"
            else DualHybrid.initialize(
                cfg, 39, None if kind == "FR" else ab, cb, initial
            )
        )
        arrays = {}
        for name, value in vars(obj).items():
            if isinstance(value, np.ndarray):
                arrays[name] = {
                    "shape": list(value.shape),
                    "bytes": value.nbytes,
                    "scalars": value.size,
                }
            elif isinstance(value, tuple) and all(
                isinstance(v, np.ndarray) for v in value
            ):
                arrays[name] = {
                    "bytes": sum(v.nbytes for v in value),
                    "scalars": sum(v.size for v in value),
                }
        rows[kind] = {
            "arrays": arrays,
            "total_owned_array_bytes": sum(v["bytes"] for v in arrays.values()),
            "dynamic_pattern_scalars": obj.q.size,
            "dynamic_mediator_real_dofs": cfg.n,
            "mediator_fft_storage_real_scalars": 2 * obj.m.size,
            "clock_scalars": 1,
            "AB_coordinates": 20 if kind != "FR" else 0,
            "C_coordinates": cb.shape[1] if kind != "RF" else 0,
            "resolved_u": len(obj.outside),
            "static_template_scalars": len(obj.template),
            "owned_dynamic_bytes": obj.q.nbytes + obj.m.nbytes,
            "checkpoint_no_history_buffer": True,
        }
    tail_fractions = {}
    x = np.arange(cfg.n) * cfg.length / cfg.n - cfg.length / 2
    for path in task.PANEL.glob("*-initial.npz"):
        position = float(path.name.split("-c")[1].split("-")[0])
        u = np.load(path)["state"][0]
        distances = abs(
            (x[:, None] - np.array([-14, 14, position]) + cfg.length / 2) % cfg.length
            - cfg.length / 2
        )
        cell = distances.argmin(axis=1) == 2
        interior = (x >= position - 9) & (x < position + 9)
        tail_fractions[path.stem] = float(
            (u[cell & ~interior] ** 2).sum() / (u[cell] ** 2).sum()
        )
    write_json(
        out / "results.json",
        {
            "initial_C_nearest_center_cell_energy_outside_interior": tail_fractions,
            "configurations": rows,
            "FF_physical_dynamic_scalars": 2 * cfg.n,
            "semantics": "C impulse projected from current reconstruction; no input history. Per-stage whole-field reconstruction/nonlinear quadrature, dense lift/projection, spectral mediator. Four branches for controlled evaluation; primary deployment needs one.",
        },
    )
    charge(
        time.process_time() - st,
        time.perf_counter() - sw,
        "resource inventory initialization",
    )


def exposed(output):
    """Diagnose first frozen panel only; its preparations become exposed."""
    out = start_panel(
        output,
        {
            "stage": "first-panel failure diagnosis; exposed cases",
            "candidate_orders": [24, 32],
            "no_model_selected_from_fresh_second_panel": True,
        },
    )
    _, cfg, _ = baseline()
    x = np.arange(cfg.n) * cfg.length / cfg.n - cfg.length / 2
    basis = np.load(ROOT / "calibration-01/basis.npz")["basis"]
    records = []
    for row in load(ROOT / "fresh-01/results.json"):
        if not (
            row["assessments"]["FR"]["failures"] or row["assessments"]["RR"]["failures"]
        ):
            continue
        initial = np.load(ROOT / "fresh-01" / f"{row['preparation_id']}-initial.npz")[
            "state"
        ]
        idx = c_indices(x, row["position"])
        w = regions(x, [row["position"]])[0, idx] * (cfg.length / cfg.n)
        local = np.array(row["FF"]["C_fields"])
        delta = local - initial[0, idx]
        sample = np.array(row["FF"]["time"]) >= row["plan"]["horizon"] - 20
        item = {
            "run_id": row["run_id"],
            "FR": row["assessments"]["FR"],
            "RR": row["assessments"]["RR"],
            "projection": {},
        }
        for order in [24, 32]:
            b = basis[:, :order]
            projected = initial[0, idx] + (delta @ b) @ b.T
            mass = (projected**2 * w).sum(axis=2)
            response = mass[:, 1] - mass[:, 0]
            error = response - np.array(row["FF"]["response"])[:, 2]
            item["projection"][str(order)] = {
                "C_field_relative_rms": float(
                    np.linalg.norm(projected - local) / np.linalg.norm(local)
                ),
                "C_response_max_error": float(abs(error).max()),
                "C_late_response_rmse": float(np.sqrt(np.mean(error[sample] ** 2))),
            }
        records.append(item)
    write_json(out / "results.json", records)


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "mode",
        choices=["projection", "summarize", "preservation", "resources", "exposed"],
    )
    p.add_argument("--output", required=True)
    p.add_argument("--qualification")
    p.add_argument("--order", type=int, default=8)
    p.add_argument("--freeze", default="frozen-01")
    p.add_argument("--panel", default="fresh-01")
    a = p.parse_args()
    task.FROZEN, task.PANEL = ROOT / a.freeze, ROOT / a.panel
    if a.mode == "projection":
        projection(a.output, a.qualification, a.order)
    else:
        globals()[a.mode](a.output)


if __name__ == "__main__":
    main()
