"""Bounded input-driven response experiment; all output directories are new.

The field evaluator may inspect truth and run a sham. Reduced prediction lives
in response_state, which receives only initial separation and declared inputs.
"""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .explore import start_panel, write_json
from .measurements import describe, outward, prepare
from .response_state import CausalKernel, ResponseState, effective_pulses
from .simulator import Config, Field

ROOT = Path("work/mediated_patterns")
SAMPLE = 0.25
DEV_SCHEDULES = [
    [(0, 0.1), (2, 0.1)],
    [(0, 0.1), (2, -0.1)],
    [(0, -0.1), (10, 0.1)],
    [(0, 0.1), (10, 0.1)],
    [(0, 0.1), (40, -0.1)],
    [(0, -0.1), (40, -0.1)],
]
CONTRACT = {
    "input": "u <- u*(1+a*b(x)); compact C-infinity bump, radius 8, fixed left anchor",
    "output": "fixed exterior mediator sensor at right anchor+12, minus own no-pulse sham",
    "initial_access": "one field scan for u^2-weighted initial separation; nominal anchors known",
    "deployment": "only initial separation, retained state, static coefficients, clock and pulses",
    "sensor": "old smooth radius-2 sensor; never recentered",
    "sample": SAMPLE,
    "event_order": "jump at exact event time before sampling; mediator is continuous at event",
    "preparation": "two isolated 150-unit relaxations, assembly residual .02, age20 odd/60 even",
    "new_horizon": 100,
    "historical_gates": "unchanged; terminal exterior max2e-6 and NRMSE15% at20/50",
}


def grid_index(t, step):
    i = round(t / step)
    if t < 0 or not np.isclose(i * step, t, atol=1e-10, rtol=0):
        raise ValueError("Time must be nonnegative and on the integration grid")
    return i


def trace(
    field, initial, separation, schedules, horizon=100, sample=SAMPLE, replay=False
):
    """Evaluate complete sequences, sharing only their matched *evaluator* sham.

    Replay is a separate diagnostic, streaming sham RK stages. It never enters
    the reduced predictor. Dynamics still has no pattern labels or trackers.
    """
    start_cpu, start_wall = time.process_time(), time.perf_counter()
    nsteps, stride = grid_index(horizon, field.c.dt), grid_index(sample, field.c.dt)
    if stride < 1 or nsteps % stride:
        raise ValueError("Use aligned positive samples and horizon")
    anchors = [-separation / 2, separation / 2]
    events = []
    for schedule in schedules:
        ev = {}
        for when, amplitude in schedule:
            k = grid_index(when, field.c.dt)
            if k > nsteps or not np.isfinite(amplitude):
                raise ValueError("Invalid event")
            if k in ev:
                raise ValueError(
                    "Simultaneous multiplicative pulses need explicit ordering"
                )
            ev[k] = amplitude
        events.append(ev)
    initial_desc = describe(field.x, initial, anchors)
    measured_sep = float(np.diff(initial_desc["center"])[0])
    v = np.repeat(np.fft.rfft(initial, axis=-1)[None], len(schedules) + 1, axis=0)
    control = v[1:].copy() if replay else None
    d = abs(field.x - anchors[1])
    ramp = np.clip(
        (d - 8) / 4, 0, 1
    )  # Fixed diagnostic replay weight, not field clipping.
    weight = 1 - (6 * ramp**5 - 15 * ramp**4 + 10 * ramp**3)
    times, sensors, descriptors, controls = [], [], [], []
    for k in range(nsteps + 1):
        for j, ev in enumerate(events):
            if k in ev:
                for values in ([v[j + 1]], [control[j]] if replay else []):
                    for value in values:
                        state = np.fft.irfft(value, n=field.c.n, axis=-1)
                        value[:] = np.fft.rfft(
                            field.pulse(
                                state,
                                anchors[0],
                                ev[k],
                                width=8,
                                relative=True,
                                compact=True,
                            ),
                            axis=-1,
                        )
        if k % stride == 0:
            states = np.fft.irfft(v, n=field.c.n, axis=-1)
            times.append(k * field.c.dt)
            sensors.append(outward(field.x, states, anchors[1]))
            descriptors.append(describe(field.x, states, anchors))
            if replay:
                controls.append(
                    outward(
                        field.x, np.fft.irfft(control, n=field.c.n, axis=-1), anchors[1]
                    )
                )
        if k == nsteps:
            break
        v[0], stages = field.step(v[0], record=replay)
        for j in range(len(schedules)):
            v[j + 1], _ = field.step(v[j + 1])
            if replay:
                control[j], _ = field.step(
                    control[j], replay=stages, replay_weight=weight
                )
    sensor = np.array(sensors)
    desc = {key: np.stack([r[key] for r in descriptors]) for key in descriptors[0]}
    records = []
    for j, schedule in enumerate(schedules):
        row = {
            "time": times,
            "pulses": schedule,
            "nominal_separation": separation,
            "separation": measured_sep,
            "response": (sensor[:, j + 1] - sensor[:, 0]).tolist(),
            "sensor_full": sensor[:, j + 1].tolist(),
            "sensor_sham": sensor[:, 0].tolist(),
            "receiver_mass_response": (
                desc["mass"][:, j + 1, 1] - desc["mass"][:, 0, 1]
            ).tolist(),
            "descriptors": {
                key: value[:, j + 1].tolist() for key, value in desc.items()
            },
            "sham_descriptors": {
                key: value[:, 0].tolist() for key, value in desc.items()
            },
        }
        if replay:
            row["replay_response"] = (np.array(controls)[:, j] - sensor[:, 0]).tolist()
        records.append(row)
    return records, {
        "cpu_seconds": time.process_time() - start_cpu,
        "wall_seconds": time.perf_counter() - start_wall,
    }


def existing_initial(seed, separation):
    path = ROOT / "development-01" / f"seed{seed}-d{separation}-initial.npz"
    return (
        np.load(path)["state"],
        str(path),
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )


def error_metrics(y, prediction, mask=None):
    y, prediction = np.asarray(y), np.asarray(prediction)
    if mask is not None:
        y, prediction = y[mask], prediction[mask]
    if not np.isfinite(prediction).all():
        return {"status": "FAILED_NONFINITE"}
    rms = float(np.sqrt(np.mean(y * y)))
    err = prediction - y
    return {
        "signal_rms": rms,
        "rmse": float(np.sqrt(np.mean(err * err))),
        "max_absolute": float(np.max(abs(err))),
        "nrmse": float(np.sqrt(np.mean(err * err)) / rms) if rms else None,
    }


def compare(records, kernel, model):
    rows = []
    for r in records:
        t = np.array(r["time"])
        pk = kernel.predict(r["separation"], r["pulses"], t)
        ps = model.rollout(r["separation"], r["pulses"], t)
        masks = {
            "all": np.ones(len(t), dtype=bool),
            "early_first": t <= 5,
            "after_last": t >= r["pulses"][-1][0],
            "late": t >= 20,
        }
        rows.append(
            {
                "run_id": r["run_id"],
                "preparation_id": r["preparation_id"],
                "kernel": pk.tolist(),
                "recursive": ps.tolist(),
                "metrics": {
                    name: {
                        label: error_metrics(r["response"], pred, mask)
                        for label, pred in [
                            ("kernel", pk),
                            ("recursive", ps),
                            ("zero", np.zeros(len(t))),
                        ]
                    }
                    for name, mask in masks.items()
                },
            }
        )
    return rows


def run_development(output, pilot=False):
    out = start_panel(
        output,
        {
            "contract": CONTRACT,
            "config": asdict(Config()),
            "seeds": [11, 22],
            "separations": [24, 26, 28, 30, 32],
            "single_amplitudes": [-0.1, 0.05, 0.1],
            "sequence_schedules": DEV_SCHEDULES,
            "split": "development only; old preparations reused",
            "pilot": pilot,
        },
    )
    records, costs = [], []
    cases = (
        [(11, 28)]
        if pilot
        else [(s, d) for s in [11, 22] for d in [24, 26, 28, 30, 32]]
    )
    for seed, d in cases:
        initial, path, digest = existing_initial(seed, d)
        schedules = [[(0, a)] for a in [-0.1, 0.05, 0.1]]
        if pilot or d in (24, 28, 32):
            schedules += DEV_SCHEDULES
        try:
            rows, cost = trace(Field(), initial, d, schedules, replay=pilot)
            for i, row in enumerate(rows):
                row.update(
                    run_id=f"dev-s{seed}-d{d}-input{i}",
                    preparation_id=f"dev-s{seed}-d{d}",
                    seed=seed,
                    split="development",
                    initial_path=path,
                    initial_sha256=digest,
                )
            records.extend(rows)
            costs.append(cost)
            write_json(out / "results.json", {"records": records, "costs": costs})
            print(seed, d, cost, flush=True)
        except (FloatingPointError, ValueError) as exc:
            write_json(
                out / f"failure-s{seed}-d{d}.json",
                {"status": "FAILED", "error": str(exc)},
            )
            raise
    return out


def fit_models(data, output):
    data = Path(data)
    records = json.loads((data / "results.json").read_text())["records"]
    out = start_panel(
        output,
        {
            "training_data": str(data),
            "contract": CONTRACT,
            "orders": [4, 6, 8],
            "kernel_degrees": [1, 2],
        },
    )
    singles = [r for r in records if len(r["pulses"]) == 1]
    start = time.process_time()
    summary = []
    for degree in (1, 2):
        kernel = CausalKernel.fit(singles, degree)
        kernel.save(out / f"kernel-degree{degree}.json")
        for order in (4, 6, 8):
            model = ResponseState.fit(kernel, order=order, block=160)
            model.save(out / f"state-degree{degree}-order{order}.json")
            rows = compare(records, kernel, model)
            write_json(out / f"predictions-degree{degree}-order{order}.json", rows)
            for kind in ("single", "sequence"):
                selected = [
                    i
                    for i, r in enumerate(records)
                    if (len(r["pulses"]) == 1) == (kind == "single")
                ]
                y = np.concatenate([records[i]["response"] for i in selected])
                row = {
                    "degree": degree,
                    "order": order,
                    "kind": kind,
                    "spectral_radius": float(
                        max(abs(np.linalg.eigvals(model.transition)))
                    ),
                    **{
                        label: error_metrics(
                            y, np.concatenate([rows[i][label] for i in selected])
                        )
                        for label in ("kernel", "recursive")
                    },
                }
                summary.append(row)
                print(row, flush=True)
    write_json(
        out / "fit.json",
        {"rows": summary, "fit_cpu_seconds": time.process_time() - start},
    )


def qualify(output, triple=False):
    configs = [
        Config(),
        replace(Config(), dt=0.00625),
        replace(Config(), n=1024, dt=0.00625),
        replace(Config(), length=256, n=1024, dt=0.00625),
    ]
    schedules = (
        [[(0, 0.06), (7, -0.075), (23, 0.05)]]
        if triple
        else [DEV_SCHEDULES[i] for i in [0, 1, 4]]
    )
    separations = [29] if triple else [24, 28]
    out = start_panel(
        output,
        {
            "contract": CONTRACT,
            "configs": [asdict(c) for c in configs],
            "separations": separations,
            "seed": 11,
            "age": 20,
            "residual": 0.02,
            "schedules": schedules,
            "qualification": "new preparations at each mesh; fixed nominal pulse/sensor anchors",
        },
    )
    records, costs = [], []
    for d in separations:
        for ci, cfg in enumerate(configs):
            begin = time.process_time()
            f = Field(cfg)
            state, _ = prepare(f, d, 11, age=20, residual=0.02)
            prep_cpu = time.process_time() - begin
            rows, cost = trace(f, state, d, schedules)
            for j, row in enumerate(rows):
                row.update(
                    run_id=f"qualification-d{d}-mesh{ci}-input{j}",
                    preparation_id=f"qualification-d{d}-mesh{ci}",
                    config=asdict(cfg),
                )
            records.extend(rows)
            costs.append({**cost, "preparation_cpu_seconds": prep_cpu})
            write_json(out / "results.json", {"records": records, "costs": costs})
            print(d, ci, costs[-1], flush=True)
    comparisons = []
    for d in separations:
        for j in range(len(schedules)):
            group = [
                r
                for r in records
                if r["nominal_separation"] == d and r["run_id"].endswith(f"input{j}")
            ]
            for k in range(1, 4):
                delta = np.array(group[k]["response"]) - group[0]["response"]
                comparisons.append(
                    {
                        "separation": d,
                        "schedule_index": j,
                        "mesh": k,
                        "max_difference": float(max(abs(delta))),
                        "early_max_difference": float(max(abs(delta[:21]))),
                        "rms_difference": float(np.sqrt(np.mean(delta**2))),
                    }
                )
    write_json(out / "comparison.json", comparisons)


def diagnose(output, extra=False):
    """Cheap discriminator: true delayed single pulses versus time-invariant kernels."""
    schedules = (
        (
            [[(0, -0.1), (2, a)] for a in (-0.1, 0.1)]
            + [[(0, a), (5, -a)] for a in (-0.1, 0.1)]
        )
        if extra
        else [[(t, a)] for t in (0, 2, 10, 40) for a in (-0.1, 0.1)]
    )
    out = start_panel(
        output,
        {
            "contract": CONTRACT,
            "schedules": schedules,
            "purpose": "separate aging, pulse interaction and response fitting",
        },
    )
    records, costs = [], []
    for d in [24, 28, 32] if extra else [28]:
        initial, path, digest = existing_initial(11, d)
        rows, cost = trace(Field(), initial, d, schedules)
        for j, r in enumerate(rows):
            r.update(
                run_id=f"diagnostic-d{d}-input{j}",
                preparation_id=f"dev-s11-d{d}",
                initial_path=path,
                initial_sha256=digest,
            )
        records.extend(rows)
        costs.append(cost)
        write_json(out / "results.json", {"records": records, "costs": costs})
        print(d, cost, flush=True)


def repair(data, extra, output):
    from scipy.optimize import least_squares

    out = start_panel(
        output,
        {
            "training_data": [data, extra],
            "contract": CONTRACT,
            "repair": "one decaying susceptibility with bilinear/quadratic event correction",
            "order": 10,
            "fit": "ordinary least squares on development sequences; no fresh data",
        },
    )
    records = json.loads((Path(data) / "results.json").read_text())["records"]
    added = json.loads((Path(extra) / "results.json").read_text())["records"]
    kernel = CausalKernel.fit([r for r in records if len(r["pulses"]) == 1], 2)
    sequences = [r for r in records + added if len(r["pulses"]) > 1]

    def residual(params):
        tau, gain, quad = params
        return np.concatenate(
            [
                (
                    kernel.predict(
                        r["separation"],
                        effective_pulses(r["pulses"], tau, gain, quad),
                        r["time"],
                    )
                    - np.array(r["response"])
                )
                / 1e-6
                for r in sequences
            ]
        )

    start = time.process_time()
    fit = least_squares(
        residual, [1.0, 1.0, 0.0], bounds=([0.1, -20, -200], [10, 20, 200])
    )
    tau, gain, quad = fit.x
    model = ResponseState.fit(kernel, order=10, block=160)
    model.save(out / "state-uncorrected.json")
    model.susceptibility_tau = float(tau)
    model.susceptibility_gain = float(gain)
    model.susceptibility_quadratic_gain = float(quad)
    model.save(out / "state-corrected.json")
    kernel.save(out / "kernel.json")
    rows = []
    for r in sequences:
        plain = kernel.predict(r["separation"], r["pulses"], r["time"])
        pred = kernel.predict(
            r["separation"], effective_pulses(r["pulses"], tau, gain, quad), r["time"]
        )
        dynamic = model.rollout(r["separation"], r["pulses"], r["time"])
        rows.append(
            {
                "run_id": r["run_id"],
                "preparation_id": r["preparation_id"],
                "plain": error_metrics(r["response"], plain),
                "corrected": error_metrics(r["response"], pred),
                "recursive": error_metrics(r["response"], dynamic),
            }
        )
    write_json(
        out / "repair.json",
        {
            "parameters": fit.x.tolist(),
            "fit_success": bool(fit.success),
            "cpu_seconds": time.process_time() - start,
            "records": rows,
        },
    )
    print(
        "susceptibility",
        fit.x,
        "worst plain/corrected/recursive",
        [
            max(r[key]["nrmse"] for r in rows)
            for key in ["plain", "corrected", "recursive"]
        ],
        flush=True,
    )


def freeze(data, output):
    """Create immutable model copies and the fresh-case contract before evaluation."""
    data = Path(data)
    out = start_panel(
        output,
        {
            "purpose": "freeze after development, before fresh preparations",
            "contract": CONTRACT,
        },
    )
    for name in ("kernel.json", "state-uncorrected.json"):
        (out / name).write_bytes((data / name).read_bytes())
    protocol = {
        "status": "FROZEN_EXPLORATORY_WITHIN_FAMILY",
        "contract": CONTRACT,
        "models": {
            name: hashlib.sha256((out / name).read_bytes()).hexdigest()
            for name in ("kernel.json", "state-uncorrected.json")
        },
        "selected_model": "quadratic input, order10 ERA, no susceptibility correction",
        "rejected_repair": "one decaying event-susceptibility state worsened worst development error",
        "seeds": [411, 522, 633],
        "separations": [25, 29, 31],
        "sample": SAMPLE,
        "horizon": 100,
        "schedules": [
            {
                "name": "same_sign_wait6",
                "role": "primary",
                "pulses": [[0, 0.075], [6, 0.075]],
            },
            {
                "name": "opposite_wait15",
                "role": "primary",
                "pulses": [[0, -0.075], [15, 0.075]],
            },
            {
                "name": "unequal_wait30",
                "role": "primary",
                "pulses": [[0, 0.05], [30, -0.1]],
            },
            {
                "name": "opposite_wait3",
                "role": "short_wait_challenge",
                "pulses": [[0, -0.075], [3, 0.075]],
            },
            {
                "name": "irregular_three",
                "role": "continued_reuse",
                "pulses": [[0, 0.06], [7, -0.075], [23, 0.05]],
            },
        ],
        "validity": "initial nominal separation24..32, |a|<=.1, pulse waits>=5, horizon<=100; tested family only",
        "gates": {
            "response_resolution_floor": 1e-7,
            "floor_rationale": "absolute readout contract rounded above the 7.87e-8 domain-refinement bound",
            "per_run_trace_nrmse": 0.05,
            "per_run_max_absolute": 2e-6,
            "early_first_0_to_5_max_absolute": 2e-8,
            "relative_rule": "no relative accuracy verdict if window signal RMS<=1e-7",
            "after_last": "same NRMSE/absolute gates, separate from whole trace",
        },
        "historical_comparison": "single-pulse 5/20/50 maps remain terminal only; report post-last errors without resetting",
        "receiver_diagnostic": "stream sham RK stages into receiver-local replay for seed411,d29 only; never into model",
        "preparation_split": "development seeds11/22 vs fresh411/522/633; complete prepared states kept together",
        "no_retuning": "exposed fresh cases will not change these fitted models or gates",
    }
    write_json(out / "freeze.json", protocol)
    print("Frozen", out, flush=True)


def assert_run_split(training_ids, evaluation_ids):
    if set(training_ids) & set(evaluation_ids):
        raise ValueError("Preparation leakage across split")


def fresh(data, output, quick=False):
    data = Path(data)
    frozen = json.loads((data / "freeze.json").read_text())
    for name, digest in frozen["models"].items():
        if hashlib.sha256((data / name).read_bytes()).hexdigest() != digest:
            raise ValueError("Frozen model changed")
    out = start_panel(
        output,
        {
            "freeze_path": str(data / "freeze.json"),
            "freeze_sha256": hashlib.sha256(
                (data / "freeze.json").read_bytes()
            ).hexdigest(),
            "frozen": frozen,
            "quick": quick,
        },
    )
    kernel = CausalKernel.load(data / "kernel.json")
    model = ResponseState.load(data / "state-uncorrected.json")
    schedules = frozen["schedules"]
    records, costs = [], []
    cases = (
        [(411, 29)]
        if quick
        else [(s, d) for s in frozen["seeds"] for d in frozen["separations"]]
    )
    assert_run_split(
        [f"s{s}-d{d}" for s in [11, 22] for d in [24, 26, 28, 30, 32]],
        [f"s{s}-d{d}" for s, d in cases],
    )
    for seed, d in cases:
        begin = time.process_time()
        f = Field()
        initial, _ = prepare(f, d, seed, age=20 if seed % 2 else 60, residual=0.02)
        prep_cpu = time.process_time() - begin
        np.savez_compressed(out / f"seed{seed}-d{d}-initial.npz", state=initial, x=f.x)
        try:
            rows, cost = trace(
                f,
                initial,
                d,
                [s["pulses"] for s in schedules],
                replay=seed == 411 and d == 29,
            )
            for row, s in zip(rows, schedules, strict=True):
                row.update(
                    seed=seed,
                    run_id=f"fresh-s{seed}-d{d}-{s['name']}",
                    preparation_id=f"fresh-s{seed}-d{d}",
                    role=s["role"],
                    split="fresh",
                    initial_sha256=hashlib.sha256(initial.tobytes()).hexdigest(),
                )
            records.extend(rows)
            costs.append({**cost, "preparation_cpu_seconds": prep_cpu})
            write_json(out / "results.json", {"records": records, "costs": costs})
            print(seed, d, costs[-1], flush=True)
        except (ValueError, FloatingPointError) as exc:
            write_json(
                out / f"failure-s{seed}-d{d}.json",
                {"status": "FAILED", "error": str(exc)},
            )
            raise
    start = time.process_time()
    predictions = compare(records, kernel, model)
    predict_cpu = time.process_time() - start
    write_json(out / "predictions.json", predictions)
    write_json(
        out / "forecast_cost.json",
        {
            "cpu_seconds": predict_cpu,
            "runs": len(records),
            "includes": "kernel+recurrence and metric calculation, excludes field simulation",
        },
    )
    summarize(out, frozen)


def summarize(output, frozen):
    out = Path(output)
    records = json.loads((out / "results.json").read_text())["records"]
    preds = json.loads((out / "predictions.json").read_text())
    summary = []
    floor = frozen["gates"]["response_resolution_floor"]
    for role in ("primary", "short_wait_challenge", "continued_reuse"):
        ids = [i for i, r in enumerate(records) if r["role"] == role]
        if not ids:
            continue
        for name in ("kernel", "recursive"):
            selected = [preds[i]["metrics"]["all"][name] for i in ids]
            y = np.concatenate([records[i]["response"] for i in ids])
            p = np.concatenate([preds[i][name] for i in ids])
            worst = max(ids, key=lambda i: preds[i]["metrics"]["all"][name]["nrmse"])
            failures = []
            for i in ids:
                checks = {}
                for window in ("all", "after_last"):
                    m = preds[i]["metrics"][window][name]
                    checks[window + "_absolute"] = (
                        m["max_absolute"] <= frozen["gates"]["per_run_max_absolute"]
                    )
                    checks[window + "_relative"] = (
                        m["signal_rms"] <= floor
                        or m["nrmse"] <= frozen["gates"]["per_run_trace_nrmse"]
                    )
                checks["early_absolute"] = (
                    preds[i]["metrics"]["early_first"][name]["max_absolute"]
                    <= frozen["gates"]["early_first_0_to_5_max_absolute"]
                )
                if not all(checks.values()):
                    failures.append(
                        {
                            "run_id": records[i]["run_id"],
                            "failed": [k for k, v in checks.items() if not v],
                        }
                    )
            summary.append(
                {
                    "role": role,
                    "model": name,
                    "runs": len(ids),
                    "pooled": error_metrics(y, p),
                    "worst_run": records[worst]["run_id"],
                    "worst_nrmse": max(m["nrmse"] for m in selected),
                    "failed_runs": failures,
                }
            )
    write_json(out / "summary.json", summary)
    for r in summary:
        print(
            r["role"],
            r["model"],
            "pooled",
            r["pooled"]["nrmse"],
            "worst",
            r["worst_nrmse"],
            "failed",
            len(r["failed_runs"]),
            flush=True,
        )


def report(data, output):
    """Read-only analysis of frozen predictions; no model fitting or field runs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .reduced import features

    data = Path(data)
    out = start_panel(
        output, {"purpose": "report only; no retuning", "data": str(data)}
    )
    rec = json.loads((data / "results.json").read_text())["records"]
    pred = json.loads((data / "predictions.json").read_text())
    protocol = json.loads((data / "protocol.json").read_text())
    model_dir = Path(protocol["freeze_path"]).parent
    model = ResponseState.load(model_dir / "state-uncorrected.json")
    kernel = CausalKernel.load(model_dir / "kernel.json")
    seed_scores = []
    horizons = []
    pathway = []
    stability = []
    peaks = []
    for role in ("primary", "continued_reuse", "short_wait_challenge"):
        for seed in sorted({r["seed"] for r in rec}):
            ids = [
                i for i, r in enumerate(rec) if r["seed"] == seed and r["role"] == role
            ]
            for name in ("kernel", "recursive"):
                seed_scores.append(
                    {
                        "role": role,
                        "seed": seed,
                        "model": name,
                        **error_metrics(
                            np.concatenate([rec[i]["response"] for i in ids]),
                            np.concatenate([pred[i][name] for i in ids]),
                        ),
                    }
                )
    for r, p in zip(rec, pred, strict=True):
        t = np.array(r["time"])
        y = np.array(r["response"])
        for h in (5, 20, 50):
            idx = grid_index(r["pulses"][-1][0] + h, SAMPLE)
            horizons.append(
                {
                    "run_id": r["run_id"],
                    "role": r["role"],
                    "after_last": h,
                    "truth": float(y[idx]),
                    "kernel_error": float(p["kernel"][idx] - y[idx]),
                    "recursive_error": float(p["recursive"][idx] - y[idx]),
                }
            )
        peaks.append(
            {
                "run_id": r["run_id"],
                "role": r["role"],
                **{
                    label: {
                        "signed_value": float(v[np.argmax(abs(v))]),
                        "time": float(t[np.argmax(abs(v))]),
                        "end_value": float(v[-1]),
                    }
                    for label, v in [
                        ("field", y),
                        ("kernel", np.array(p["kernel"])),
                        ("recursive", np.array(p["recursive"])),
                    ]
                },
            }
        )
        if "replay_response" in r:
            contrast = y - np.array(r["replay_response"])
            error = np.array(p["recursive"]) - y
            pathway.append(
                {
                    "run_id": r["run_id"],
                    "role": r["role"],
                    "contrast_rms": float(np.sqrt(np.mean(contrast**2))),
                    "contrast_fraction_of_total": float(
                        np.linalg.norm(contrast) / np.linalg.norm(y)
                    ),
                    "model_error_over_contrast_rms": float(
                        np.linalg.norm(error) / np.linalg.norm(contrast)
                    ),
                    "claim": "total-output accuracy only; controlled contrast not predicted by this model",
                }
            )
        center = np.array(r["descriptors"]["center"])
        mass = np.array(r["descriptors"]["mass"])
        sham_mass = np.array(r["sham_descriptors"]["mass"])
        stability.append(
            {
                "run_id": r["run_id"],
                "max_center_drift": float(abs(center - center[0]).max()),
                "max_separation_drift": float(
                    abs(np.diff(center, axis=1)[:, 0] - r["separation"]).max()
                ),
                "terminal_mass_relative_to_sham": (mass[-1] / sham_mass[-1]).tolist(),
                "min_mass": float(mass.min()),
                "terminal_width": r["descriptors"]["width"][-1],
            }
        )
    # Repeat inexpensive operations to measure access/update costs apart from field work.
    sample_rec = next(
        r for r in rec if r["seed"] == 411 and r["nominal_separation"] == 29
    )
    initial = np.load(data / "seed411-d29-initial.npz")["state"]
    f = Field()
    begin = time.process_time()
    for _ in range(1000):
        describe(f.x, initial, [-14.5, 14.5])
    extraction = (time.process_time() - begin) / 1000
    begin = time.process_time()
    for _ in range(100):
        model.rollout(
            sample_rec["separation"], sample_rec["pulses"], sample_rec["time"]
        )
    forecast = (time.process_time() - begin) / 100
    begin = time.process_time()
    for _ in range(1000):
        kernel.predict(
            sample_rec["separation"], sample_rec["pulses"], sample_rec["time"]
        )
    kernel_forecast = (time.process_time() - begin) / 1000
    cost = {
        "initial_field_extraction_cpu_seconds": extraction,
        "recursive_100_unit_forecast_cpu_seconds": forecast,
        "kernel_100_unit_forecast_cpu_seconds": kernel_forecast,
        "recursive_scalars": 10,
        "pending_event_buffer_scalars": 2,
        "clock_scalars": 1,
        "initial_geometry_scalars": 1,
        "stored_matrix_coefficients": 210,
        "matrix_binary_bytes": model.transition.nbytes
        + model.injection.nbytes
        + model.readout.nbytes,
        "frozen_model_json_bytes": (model_dir / "state-uncorrected.json")
        .stat()
        .st_size,
        "kernel_coefficients": int(kernel.coefficients.size),
        "kernel_binary_bytes": kernel.coefficients.nbytes,
        "kernel_json_bytes": (model_dir / "kernel.json").stat().st_size,
        "kernel_history": "all past event time/amplitude pairs; grows with event count",
        "field_state_scalars": 1024,
        "field_state_binary_bytes": 8192,
        "spectral_radius": float(max(abs(np.linalg.eigvals(model.transition)))),
        "stability_prior": "none; fitted A happens to be stable; validated only to time100",
        "unused_serialized_repair_fields": "susceptibility=0 plus 3 static constants; no active correction",
    }
    # Old horizon maps are evaluated only on single-pulse development traces.
    old = json.loads((ROOT / "outward-development-01/model.json").read_text())
    dev = json.loads(
        (ROOT / "recursive_response/development-01/results.json").read_text()
    )["records"]
    singles = [r for r in dev if len(r["pulses"]) == 1]
    historic = []
    for j, h in enumerate((5, 20, 50)):
        truth = [r["response"][grid_index(h, SAMPLE)] for r in singles]
        yy = [
            (
                features([r["separation"], r["pulses"][0][1]])
                @ np.array(old["coefficients"])
            )[j]
            for r in singles
        ]
        historic.append(
            {
                "horizon": h,
                "scope": "old terminal model, single-pulse development only",
                **error_metrics(truth, yy),
            }
        )
    # A checkpoint at t=15 immediately after the second event includes pending inputs.
    model.initialize(sample_rec["separation"])
    model.jump(-0.075)
    model.advance(15)
    model.jump(0.075)
    model.save(out / "checkpoint-t15.json")
    resumed = ResponseState.load(out / "checkpoint-t15.json")
    max_resume_error = 0.0
    for _ in range(340):
        model.advance(SAMPLE)
        resumed.advance(SAMPLE)
        max_resume_error = max(max_resume_error, abs(model.output() - resumed.output()))
    write_json(
        out / "diagnostics.json",
        {
            "seed_scores": seed_scores,
            "post_last_horizons": horizons,
            "receiver_pathway": pathway,
            "stability": stability,
            "peaks": peaks,
            "cost": cost,
            "old_terminal_calibration": historic,
            "checkpoint_max_difference": max_resume_error,
            "initial_access": "one field scan for static separation; zero later field reads by predictor",
        },
    )
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), layout="constrained")
    chosen = ["fresh-s411-d29-opposite_wait15", "fresh-s411-d29-opposite_wait3"]
    colors = {"kernel": "#d97900", "recursive": "#007a9c"}
    for col, run in enumerate(chosen):
        i = next(i for i, r in enumerate(rec) if r["run_id"] == run)
        r, p = rec[i], pred[i]
        t = np.array(r["time"])
        y = np.array(r["response"])
        axes[0, col].plot(
            t, y * 1e6, color="black", lw=2, label="Full field minus sham"
        )
        for name, color in colors.items():
            value = np.array(p[name])
            axes[0, col].plot(
                t,
                value * 1e6,
                color=color,
                ls="--" if name == "kernel" else ":",
                lw=2,
                label=name.capitalize(),
            )
            axes[1, col].plot(
                t, (value - y) * 1e6, color=color, label=name.capitalize()
            )
            axes[2, col].plot(
                t[t <= 5],
                value[t <= 5] * 1e8,
                color=color,
                ls="--" if name == "kernel" else ":",
                lw=2,
            )
        axes[1, col].axhspan(
            -0.1, 0.1, color="gray", alpha=0.18, label="± resolution floor"
        )
        if "replay_response" in r:
            contrast = (y - np.array(r["replay_response"])) * 1e6
            axes[1, col].plot(
                t,
                contrast,
                color="#8b459c",
                ls="-.",
                lw=1,
                label="Full minus receiver replay",
            )
        axes[2, col].plot(t[t <= 5], y[t <= 5] * 1e8, color="black", lw=2)
        for a in axes[:, col]:
            for when, amp in r["pulses"]:
                if when <= (5 if a is axes[2, col] else 100):
                    a.axvline(when, color="#bd3333", lw=1, alpha=0.6)
            a.axhline(0, color="gray", lw=0.6)
            a.grid(alpha=0.2)
            a.set_xlabel("Time (mediator τ = 10)")
        axes[0, col].set_title(
            "Fresh two pulses: wait15 (within envelope)"
            if col == 0
            else "Fresh two pulses: wait3 (failed challenge)"
        )
        axes[0, col].text(
            0.52,
            0.06,
            "a₁ = −0.075; a₂ = +0.075",
            transform=axes[0, col].transAxes,
            fontsize=10,
        )
        axes[0, col].set_ylim(-25, 8)
        axes[1, col].set_ylim(-1.4, 0.6)
        axes[2, col].set_xlim(0, 5)
        axes[2, col].set_ylim(-2, 0.2)
        axes[2, col].set_title(
            "Early transient: below the 10⁻⁷ readout floor", fontsize=10
        )
    axes[0, 0].set_ylabel("Exterior response × 10⁶")
    axes[1, 0].set_ylabel("Prediction error / contrast × 10⁶")
    axes[2, 0].set_ylabel("Exterior response × 10⁸")
    axes[0, 0].legend(fontsize=9)
    axes[1, 0].legend(fontsize=8)
    fig.suptitle(
        "A small state carries the response forward; closely spaced cancellation remains a limit",
        fontsize=13,
    )
    fig.savefig(out / "response_sequences.png", dpi=160)
    plt.close(fig)
    print(
        "Report artifacts", out, "checkpoint difference", max_resume_error, flush=True
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "mode",
        choices=[
            "pilot",
            "development",
            "fit",
            "qualify",
            "qualify-three",
            "diagnose",
            "extra",
            "repair",
            "freeze",
            "fresh",
            "quick",
            "report",
        ],
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--data")
    parser.add_argument("--extra")
    args = parser.parse_args()
    if args.mode in ("pilot", "development"):
        run_development(args.output, args.mode == "pilot")
    elif args.mode == "fit":
        fit_models(args.data, args.output)
    elif args.mode in ("qualify", "qualify-three"):
        qualify(args.output, args.mode == "qualify-three")
    elif args.mode == "repair":
        repair(args.data, args.extra, args.output)
    elif args.mode == "freeze":
        freeze(args.data, args.output)
    elif args.mode in ("fresh", "quick"):
        fresh(args.data, args.output, args.mode == "quick")
    elif args.mode == "report":
        report(args.data, args.output)
    else:
        diagnose(args.output, args.mode == "extra")


if __name__ == "__main__":
    main()
