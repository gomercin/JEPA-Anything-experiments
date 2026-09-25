"""Matched cross-pulse diagnostics and bounded response-interface repairs."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .event_response import (
    CrossResponse,
    EventResponse,
    fit_cross_response,
    fit_event_map,
    refit_single_inputs,
)
from .explore import start_panel, write_json
from .measurements import outward, prepare
from .recursive_response import CONTRACT, error_metrics, trace
from .response_state import CausalKernel, ResponseState
from .simulator import Config, Field

ROOT = Path("work/mediated_patterns")
OLD = ROOT / "recursive_response"
FROZEN = OLD / "frozen-01"
CONTRASTS = [
    (-0.075, 0.075, 3),
    (0.075, -0.075, 3),
    (0.075, 0.075, 3),
    (-0.075, 0.075, 15),
]


def initial_path(seed, d):
    base = OLD / "fresh-01" if seed in (411, 522, 633) else ROOT / "development-01"
    return base / f"seed{seed}-d{d}-initial.npz"


def baseline():
    return (
        CausalKernel.load(FROZEN / "kernel.json"),
        ResponseState.load(FROZEN / "state-uncorrected.json"),
    )


def cross_identity(y0, y1, y2, y12, kernel, recursive):
    y0, y1, y2, y12, kernel, recursive = map(
        np.asarray, (y0, y1, y2, y12, kernel, recursive)
    )
    r1, r2, r12 = y1 - y0, y2 - y0, y12 - y0
    cross = r12 - r1 - r2
    fitting = r1 + r2 - kernel
    realization = kernel - recursive
    error = r12 - recursive
    return {
        "r1": r1,
        "r2": r2,
        "r12": r12,
        "cross": cross,
        "kernel_fit": fitting,
        "realization": realization,
        "total_error": error,
        "identity_max_error": float(max(abs(error - cross - fitting - realization))),
    }


def quartets(field, state, d, contrasts, tag, source):
    schedules = []
    for a, b, gap in contrasts:
        schedules.extend(
            [
                [(0, 0), (gap, 0)],
                [(0, a), (gap, 0)],
                [(0, 0), (gap, b)],
                [(0, a), (gap, b)],
            ]
        )
    rows, cost = trace(field, state, d, schedules)
    kernel, model = baseline()
    result = []
    for j, (a, b, gap) in enumerate(contrasts):
        group = rows[4 * j : 4 * j + 4]
        pair = group[3]
        pk = kernel.predict(pair["separation"], pair["pulses"], pair["time"])
        ps = model.rollout(pair["separation"], pair["pulses"], pair["time"])
        terms = cross_identity(*(r["sensor_full"] for r in group), pk, ps)
        # The fixed scale is supplied separately by the development diagnostic.
        rms = lambda v: float(np.sqrt(np.mean(np.asarray(v) ** 2)))
        result.append(
            {
                "run_id": f"{tag}-q{j}",
                "preparation_id": tag,
                "source": source,
                "config": asdict(field.c),
                "pulses": pair["pulses"],
                "separation": pair["separation"],
                "nominal_separation": d,
                "time": pair["time"],
                "terms": {
                    k: v.tolist() if isinstance(v, np.ndarray) else v
                    for k, v in terms.items()
                },
                "absolute_sensor_traces": [r["sensor_full"] for r in group],
                "kernel": pk.tolist(),
                "recursive": ps.tolist(),
                "rms": {
                    k: rms(terms[k])
                    for k in (
                        "r1",
                        "r2",
                        "r12",
                        "cross",
                        "kernel_fit",
                        "realization",
                        "total_error",
                    )
                },
                "baseline_metrics": error_metrics(terms["r12"], ps),
                "descriptors": pair["descriptors"],
                "sham_descriptors": pair["sham_descriptors"],
            }
        )
    return result, cost


def diagnose(output):
    path = initial_path(411, 29)
    # Fixed diagnostic scale from already exposed, single-pulse development data.
    past = json.loads((OLD / "development-01/results.json").read_text())["records"]
    single = next(r for r in past if r["run_id"] == "dev-s11-d28-input0")
    scale = float(np.sqrt(np.mean(np.array(single["response"]) ** 2)))
    out = start_panel(
        output,
        {
            "contract": CONTRACT,
            "contrasts": CONTRASTS,
            "source": str(path),
            "source_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "diagnostic_scale": scale,
            "scale_reference": "dev-s11-d28-input0 (-.1 single)",
            "note": "all prior fresh/challenge cases are now exposed development evidence",
        },
    )
    try:
        rows, cost = quartets(
            Field(), np.load(path)["state"], 29, CONTRASTS, "s411-d29", str(path)
        )
        write_json(
            out / "results.json",
            {"records": rows, "costs": [cost], "diagnostic_scale": scale},
        )
        for r in rows:
            print(
                r["pulses"],
                r["rms"],
                "NRMSE",
                r["baseline_metrics"]["nrmse"],
                flush=True,
            )
        print(cost, flush=True)
    except Exception as exc:
        write_json(out / "failure.json", {"status": "FAILED", "error": repr(exc)})
        raise


def refine(output):
    configs = [
        Config(),
        replace(Config(), dt=0.00625),
        replace(Config(), n=1024, dt=0.00625),
        replace(Config(), length=256, n=1024, dt=0.00625),
    ]
    # Representative failing quartet and reversed order; all four trajectories refined together.
    out = start_panel(
        output,
        {
            "contract": CONTRACT,
            "contrasts": CONTRASTS[:2],
            "configs": [asdict(c) for c in configs],
            "seed": 411,
            "separation": 29,
            "preparation": "same seed recipe at each mesh; age20,residual.02",
        },
    )
    rows, costs = [], []
    try:
        for j, cfg in enumerate(configs):
            startcpu, startwall = time.process_time(), time.perf_counter()
            f = Field(cfg)
            state, _ = prepare(f, 29, 411, age=20, residual=0.02)
            pc, pw = time.process_time() - startcpu, time.perf_counter() - startwall
            r, c = quartets(
                f, state, 29, CONTRASTS[:2], f"mesh{j}", "prepared at this resolution"
            )
            rows.extend(r)
            costs.append(
                {**c, "preparation_cpu_seconds": pc, "preparation_wall_seconds": pw}
            )
            write_json(out / "results.json", {"records": rows, "costs": costs})
            print(j, costs[-1], flush=True)
        scores = []
        for mesh in range(1, 4):
            for q in range(2):
                a, b = rows[q], rows[2 * mesh + q]
                diff = (
                    np.array(a["absolute_sensor_traces"]) - b["absolute_sensor_traces"]
                )
                scores.append(
                    {
                        "mesh": mesh,
                        "quartet": q,
                        "cross_difference_max": float(
                            max(
                                abs(np.array(a["terms"]["cross"]) - b["terms"]["cross"])
                            )
                        ),
                        "conservative_four_trace_bound": float(
                            abs(diff).max(axis=1).sum()
                        ),
                        "cross_timewise_bound_max": float(abs(diff).sum(axis=0).max()),
                    }
                )
        write_json(out / "refinement.json", scores)
    except Exception as exc:
        write_json(out / "failure.json", {"status": "FAILED", "error": repr(exc)})
        raise


def development(output):
    contrasts = [
        (-0.1, 0.1, 2),
        (0.1, -0.1, 2),
        (0.1, 0.1, 2),
        (-0.1, -0.1, 2),
        (-0.1, 0.1, 5),
        (0.1, -0.1, 5),
        (-0.1, 0.1, 10),
    ]
    out = start_panel(
        output,
        {
            "contract": CONTRACT,
            "contrasts": contrasts,
            "separations": [24, 28, 32],
            "seed": 11,
            "purpose": "fit cross-only state-dependent event maps; ordinary A/B/C remain frozen",
        },
    )
    rows, costs = [], []
    try:
        for d in (24, 28, 32):
            path = initial_path(11, d)
            r, cost = quartets(
                Field(), np.load(path)["state"], d, contrasts, f"s11-d{d}", str(path)
            )
            rows.extend(r)
            costs.append(cost)
            write_json(out / "results.json", {"records": rows, "costs": costs})
            print(d, cost, flush=True)
    except Exception as exc:
        write_json(out / "failure.json", {"status": "FAILED", "error": repr(exc)})
        raise


def fit(data, extra, output, powers=1, cross=False):
    paths = [Path(data), Path(extra)]
    out = start_panel(
        output,
        {
            "purpose": "fit cross-only event map, baseline A/B/C unchanged",
            "data": [str(p) for p in paths],
            "powers": powers,
            "rcond": 1e-4,
            "contract": CONTRACT,
            "family": "two-input-summary cross response"
            if cross
            else "existing-state matrix event map",
        },
    )
    rows = [
        r
        for p in paths
        for r in json.loads((p / "results.json").read_text())["records"]
    ]
    startcpu, startwall = time.process_time(), time.perf_counter()
    _, base = baseline()
    model, fit_info = (
        fit_cross_response(base, rows) if cross else fit_event_map(base, rows, powers)
    )
    model.save(out / "model.json")
    records = []
    for r in rows:
        yp = model.rollout(r["separation"], r["pulses"], r["time"])
        records.append(
            {
                "run_id": r["run_id"],
                "preparation_id": r["preparation_id"],
                "prediction": yp.tolist(),
                "baseline": r["baseline_metrics"],
                "candidate": error_metrics(r["terms"]["r12"], yp),
                "cross_approximation": error_metrics(
                    r["terms"]["cross"], yp - np.array(r["recursive"])
                ),
            }
        )
    write_json(
        out / "results.json",
        {
            "records": records,
            "fit": fit_info,
            "cpu_seconds": time.process_time() - startcpu,
            "wall_seconds": time.perf_counter() - startwall,
        },
    )
    print(fit_info, flush=True)
    for r in records:
        print(
            r["run_id"],
            "base",
            r["baseline"]["nrmse"],
            "candidate",
            r["candidate"]["nrmse"],
            flush=True,
        )


def regression(data, output, cross=False):
    out = start_panel(
        output,
        {
            "model": str(data),
            "purpose": "exposed historical regression, NOT fresh validation",
            "cross": cross,
        },
    )
    model = (CrossResponse if cross else EventResponse).load(Path(data) / "model.json")
    _, base = baseline()
    results = []
    for panel in ("fresh-01", "development-01", "diagnostic-02"):
        rows = json.loads((OLD / panel / "results.json").read_text())["records"]
        for r in rows:
            pp = model.rollout(r["separation"], r["pulses"], r["time"])
            old = base.rollout(r["separation"], r["pulses"], r["time"])
            results.append(
                {
                    "panel": panel,
                    "run_id": r["run_id"],
                    "role": r.get("role", "development"),
                    "baseline": error_metrics(r["response"], old),
                    "candidate": error_metrics(r["response"], pp),
                    "prediction": pp.tolist(),
                }
            )
    write_json(out / "results.json", results)
    for role in ("primary", "continued_reuse", "short_wait_challenge"):
        group = [r for r in results if r["role"] == role]
        print(
            role,
            "worst",
            max(r["candidate"]["nrmse"] for r in group),
            "maxabs",
            max(r["candidate"]["max_absolute"] for r in group),
            flush=True,
        )


def pulse_audit(output):
    out = start_panel(
        output,
        {
            "purpose": "changed-input diagnostic only, NEVER primary task",
            "input": "second additive increment is J_b(sham_at_gap)-sham_at_gap; evaluator privileged",
            "seed": 411,
            "separation": 29,
            "contrasts": CONTRASTS[:2],
        },
    )
    f = Field()
    initial = np.load(initial_path(411, 29))["state"]
    startcpu, startwall = time.process_time(), time.perf_counter()
    rows = []
    for a, b, gap in CONTRASTS[:2]:
        for kind in ("relative", "reference_additive"):
            first = []
            for amp in (0, a, 0, a):
                p = f.pulse(initial, -14.5, amp, width=8, relative=True, compact=True)
                _, states, _ = f.evolve(p, gap, sample=0.25)
                first.append(states)
            reference = first[0][-1]
            increment = (
                f.pulse(reference, -14.5, b, width=8, relative=True, compact=True)
                - reference
            )
            values = []
            for j, amp in enumerate((0, 0, b, b)):
                state = first[j][-1]
                if kind == "relative":
                    state = f.pulse(
                        state, -14.5, amp, width=8, relative=True, compact=True
                    )
                elif j >= 2:
                    state = state + increment
                _, states, _ = f.evolve(state, 100 - gap, sample=0.25)
                values.append(
                    np.concatenate(
                        [outward(f.x, first[j], 14.5), outward(f.x, states[1:], 14.5)]
                    )
                )
            v = np.array(values)
            cross = v[3] - v[1] - v[2] + v[0]
            rows.append(
                {
                    "pulses": [[0, a], [gap, b]],
                    "input_kind": kind,
                    "absolute_sensor_traces": v.tolist(),
                    "cross": cross.tolist(),
                    "cross_rms": float(np.sqrt(np.mean(cross**2))),
                }
            )
    # Exact no-evolution composition on this same prepared state.
    a = 0.1
    b = -0.1
    wa = (
        f.pulse(np.ones_like(initial), -14.5, 1, width=8, relative=True, compact=True)[
            0
        ]
        - 1
    )
    composed = f.pulse(
        f.pulse(initial, -14.5, a, width=8, relative=True, compact=True),
        -14.5,
        b,
        width=8,
        relative=True,
        compact=True,
    )
    expected = initial.copy()
    expected[0] *= 1 + (a + b) * wa + a * b * wa * wa
    write_json(
        out / "results.json",
        {
            "records": rows,
            "costs": [
                {
                    "cpu_seconds": time.process_time() - startcpu,
                    "wall_seconds": time.perf_counter() - startwall,
                }
            ],
            "composition_identity_max": float(abs(composed - expected).max()),
            "opposite_pulse_nonidentity_max": float(abs(composed - initial).max()),
            "mediator_unchanged": bool(np.array_equal(composed[1], initial[1])),
        },
    )
    for r in rows:
        print(r["pulses"], r["input_kind"], r["cross_rms"], flush=True)


def assess(row, prediction, gates):
    t = np.array(row["time"])
    y = np.array(row["response"])
    windows = {
        "all": np.ones(len(t), bool),
        "after_last": t >= row["pulses"][-1][0],
        "early": t <= 5,
    }
    metrics = {k: error_metrics(y, prediction, mask) for k, mask in windows.items()}
    failed = []
    for k in ("all", "after_last"):
        m = metrics[k]
        if m.get("status") == "FAILED_NONFINITE":
            failed.append(k + "_nonfinite")
            continue
        if m["max_absolute"] > gates["per_run_max_absolute"]:
            failed.append(k + "_absolute")
        if (
            m["signal_rms"] > gates["response_resolution_floor"]
            and m["nrmse"] > gates["per_run_trace_nrmse"]
        ):
            failed.append(k + "_relative")
    if (
        metrics["early"].get("status") == "FAILED_NONFINITE"
        or metrics["early"]["max_absolute"] > gates["early_first_0_to_5_max_absolute"]
    ):
        failed.append("early_absolute")
    return {"metrics": metrics, "failed": failed, "pass": not failed}


def freeze(data, output, refined=False):
    out = start_panel(
        output,
        {"purpose": "freeze selected cross response before new preparation outcomes"},
    )
    model_path = Path(data) / "model.json"
    (out / "model.json").write_bytes(model_path.read_bytes())
    old = json.loads((FROZEN / "freeze.json").read_text())
    schedules = [
        {"name": "short_np25", "role": "short", "pulses": [[0, -0.08], [2.5, 0.08]]},
        {"name": "short_pn35", "role": "short", "pulses": [[0, 0.09], [3.5, -0.09]]},
        {"name": "strong_np2", "role": "short", "pulses": [[0, -0.1], [2, 0.1]]},
        {
            "name": "same_sign_wait6",
            "role": "regression",
            "pulses": [[0, 0.075], [6, 0.075]],
        },
        {
            "name": "opposite_wait15",
            "role": "regression",
            "pulses": [[0, -0.075], [15, 0.075]],
        },
        {
            "name": "unequal_wait30",
            "role": "regression",
            "pulses": [[0, 0.05], [30, -0.1]],
        },
        {
            "name": "three_close",
            "role": "three",
            "pulses": [[0, 0.08], [2.5, -0.09], [7.5, 0.06]],
        },
    ]
    if refined:
        schedules[0] = {
            "name": "short_np275",
            "role": "short",
            "pulses": [[0, -0.085], [2.75, 0.085]],
        }
        schedules[1] = {
            "name": "short_pn325",
            "role": "short",
            "pulses": [[0, 0.095], [3.25, -0.095]],
        }
        schedules[-1] = {
            "name": "three_close_withheld",
            "role": "three",
            "pulses": [[0, 0.08], [2.25, -0.09], [7.0, 0.06]],
        }
    write_json(
        out / "freeze.json",
        {
            "status": "FROZEN_EXPLORATORY",
            "contract": CONTRACT,
            "model_sha256": hashlib.sha256(
                (out / "model.json").read_bytes()
            ).hexdigest(),
            "baseline_model_hashes": old["models"],
            "baseline_directory": str(FROZEN),
            "gates": old["gates"],
            "seeds": [1101, 1202, 1303] if refined else [741, 852, 963],
            "separations": [25.5, 27.5, 30.5] if refined else [25, 27, 31],
            "schedules": schedules,
            "horizon": 100,
            "sample": 0.25,
            "fit_sources": (
                [f"s11-d{d}" for d in [24, 26, 28, 30, 32]]
                + ["s741-d25", "s741-d27", "s741-d31", "s411-d29", "s852-d25"]
            )
            if refined
            else ["s11-d24", "s11-d28", "s11-d32", "s411-d29"],
            "selection": (
                "same A/C and response order; quartic single response at nine geometry knots; 12 conditioned cross weights and two decay times"
            )
            if refined
            else "two command-memory scalars, 6 coefficients; first pulse and A/B/C unchanged",
            "validation_round": 2 if refined else 1,
            "split": "whole preparations; all earlier outcomes exposed and only these seeds fresh",
            "joint_success": "short-gap improvement under unchanged gates AND retained regression and three-pulse gates",
            "pathway": "no pathway-fidelity claim; total-response task only",
        },
    )
    print("Frozen", out, flush=True)


def validate(data, output, quick=False):
    data = Path(data)
    frozen = json.loads((data / "freeze.json").read_text())
    if (
        hashlib.sha256((data / "model.json").read_bytes()).hexdigest()
        != frozen["model_sha256"]
    ):
        raise ValueError("Frozen candidate changed")
    for name, h in frozen["baseline_model_hashes"].items():
        if hashlib.sha256((FROZEN / name).read_bytes()).hexdigest() != h:
            raise ValueError("Baseline changed")
    out = start_panel(
        output,
        {
            "freeze": frozen,
            "freeze_path": str(data / "freeze.json"),
            "freeze_sha256": hashlib.sha256(
                (data / "freeze.json").read_bytes()
            ).hexdigest(),
            "quick": quick,
        },
    )
    model = CrossResponse.load(data / "model.json")
    kernel, base = baseline()
    cases = (
        [(frozen["seeds"][0], frozen["separations"][1])]
        if quick
        else [(s, d) for s in frozen["seeds"] for d in frozen["separations"]]
    )
    if any(f"s{s}-d{d}" in frozen["fit_sources"] for s, d in cases):
        raise ValueError("Preparation leakage")
    records, costs = [], []
    try:
        for seed, d in cases:
            st, sw = time.process_time(), time.perf_counter()
            f = Field()
            state, _ = prepare(f, d, seed, age=20 if seed % 2 else 60, residual=0.02)
            pc, pw = time.process_time() - st, time.perf_counter() - sw
            np.savez_compressed(
                out / f"seed{seed}-d{d}-initial.npz", state=state, x=f.x
            )
            rows, cost = trace(f, state, d, [s["pulses"] for s in frozen["schedules"]])
            cost.update(preparation_cpu_seconds=pc, preparation_wall_seconds=pw)
            st, sw = time.process_time(), time.perf_counter()
            for r, s in zip(rows, frozen["schedules"], strict=True):
                r.update(
                    seed=seed,
                    preparation_id=f"s{seed}-d{d}",
                    run_id=f"s{seed}-d{d}-{s['name']}",
                    role=s["role"],
                )
                r["predictions"] = {
                    "kernel": kernel.predict(
                        r["separation"], r["pulses"], r["time"]
                    ).tolist(),
                    "original": base.rollout(
                        r["separation"], r["pulses"], r["time"]
                    ).tolist(),
                    "candidate": model.rollout(
                        r["separation"], r["pulses"], r["time"]
                    ).tolist(),
                }
                r["scores"] = {
                    k: assess(r, v, frozen["gates"])
                    for k, v in r["predictions"].items()
                }
            cost.update(
                prediction_cpu_seconds=time.process_time() - st,
                prediction_wall_seconds=time.perf_counter() - sw,
            )
            records.extend(rows)
            costs.append(cost)
            write_json(out / "results.json", {"records": records, "costs": costs})
            print(seed, d, cost, flush=True)
        summary = []
        for role in ("short", "regression", "three"):
            group = [r for r in records if r["role"] == role]
            for name in ("kernel", "original", "candidate"):
                summary.append(
                    {
                        "role": role,
                        "model": name,
                        "runs": len(group),
                        "passed": sum(r["scores"][name]["pass"] for r in group),
                        "pooled": error_metrics(
                            np.concatenate([r["response"] for r in group]),
                            np.concatenate([r["predictions"][name] for r in group]),
                        ),
                        "worst_nrmse": max(
                            r["scores"][name]["metrics"]["all"]["nrmse"] for r in group
                        ),
                        "failures": [
                            {"run": r["run_id"], "failed": r["scores"][name]["failed"]}
                            for r in group
                            if not r["scores"][name]["pass"]
                        ],
                    }
                )
        write_json(out / "summary.json", summary)
        for row in summary:
            print(
                row["role"],
                row["model"],
                row["pooled"]["nrmse"],
                row["worst_nrmse"],
                "pass",
                row["passed"],
                "/",
                row["runs"],
                flush=True,
            )
    except Exception as exc:
        write_json(out / "failure.json", {"status": "FAILED", "error": repr(exc)})
        raise


def triple_diagnostic(output, seed=741, d=27):
    """Full inclusion-exclusion; exposed failed fresh cases become development."""
    pulses = [(0, 0.08), (2.5, -0.09), (7.5, 0.06)]
    out = start_panel(
        output,
        {
            "purpose": "diagnose exposed triple failure, not fresh validation",
            "pulses": pulses,
            "seed": seed,
            "separation": d,
            "contract": CONTRACT,
        },
    )
    path = ROOT / f"cross_pulse_repair/fresh-01/seed{seed}-d{d}-initial.npz"
    schedules = [
        [(t, a if mask & (1 << j) else 0) for j, (t, a) in enumerate(pulses)]
        for mask in range(8)
    ]
    rows, cost = trace(Field(), np.load(path)["state"], d, schedules)
    ys = np.array([r["sensor_full"] for r in rows])
    rr = ys - ys[0]
    singles = rr[1] + rr[2] + rr[4]
    pair_terms = {
        "12": rr[3] - rr[1] - rr[2],
        "13": rr[5] - rr[1] - rr[4],
        "23": rr[6] - rr[2] - rr[4],
    }
    pair = sum(pair_terms.values())
    third = rr[7] - singles - pair
    kernel, base = baseline()
    model = CrossResponse.load(ROOT / "cross_pulse_repair/frozen-01/model.json")
    t = rows[0]["time"]
    d = rows[0]["separation"]
    k = kernel.predict(d, pulses, t)
    p = base.rollout(d, pulses, t)
    c = model.rollout(d, pulses, t)
    terms = {
        "single_sum": singles,
        "pair_sum": pair,
        "third_order": third,
        "single_fit": singles - k,
        "realization": k - p,
        "predicted_cross": c - p,
        "candidate_error": rr[7] - c,
        "truth": rr[7],
        **pair_terms,
    }
    metrics = {
        name: {"rms": float(np.sqrt(np.mean(v * v))), "max": float(max(abs(v)))}
        for name, v in terms.items()
    }
    write_json(
        out / "results.json",
        {
            "records": rows,
            "costs": [cost],
            "terms": {k: v.tolist() for k, v in terms.items()},
            "metrics": metrics,
        },
    )
    print(metrics, flush=True)


def single_refinement(output):
    out = start_panel(
        output,
        {
            "purpose": "second repair: amplitude/geometry approximation; exposed cases are development",
            "amplitudes": [-0.1, -0.05, 0.05, 0.1],
            "nominal_geometry_knots": list(range(24, 33)),
            "state_order": 10,
            "source_A_C": "unchanged original",
            "contract": CONTRACT,
        },
    )
    oldrows = json.loads((OLD / "development-01/results.json").read_text())["records"]
    rows, costs = [], []
    for d in range(24, 33):
        if d % 2 == 0:
            path = initial_path(11, d)
            amps = [-0.05]
            reused = [
                r
                for r in oldrows
                if r["seed"] == 11
                and r["nominal_separation"] == d
                and len(r["pulses"]) == 1
            ]
            rows.extend(
                [
                    {
                        **r,
                        "source_artifact": str(OLD / "development-01/results.json"),
                        "reused": True,
                    }
                    for r in reused
                ]
            )
            seed = 11
        else:
            seed = 411 if d == 29 else 741
            path = (
                initial_path(411, 29)
                if d == 29
                else ROOT / f"cross_pulse_repair/fresh-01/seed741-d{d}-initial.npz"
            )
            amps = [-0.1, -0.05, 0.05, 0.1]
        try:
            rr, cost = trace(
                Field(), np.load(path)["state"], d, [[(0, a)] for a in amps]
            )
            for i, r in enumerate(rr):
                r.update(
                    run_id=f"single-s{seed}-d{d}-a{amps[i]}",
                    preparation_id=f"s{seed}-d{d}",
                    source_artifact=str(path),
                    reused=False,
                )
            rows.extend(rr)
            costs.append(cost)
            write_json(out / "results.json", {"records": rows, "costs": costs})
            print(d, cost, flush=True)
        except Exception as exc:
            write_json(out / "failure.json", {"status": "FAILED", "error": repr(exc)})
            raise


def refit(data, output):
    out = start_panel(
        output,
        {
            "purpose": "one follow-up after exposed triple failure: conditioned cross and refined input fit",
            "singles": str(data),
            "cross_sources": [
                "development-01",
                "diagnosis-01",
                "triple-diagnostic-01",
                "triple-diagnostic-02",
            ],
            "changes": "quartic amplitudes, static cubic geometry interpolation, 3 supplied cross-geometry features; same A/C and dimension",
        },
    )
    st, sw = time.process_time(), time.perf_counter()
    _, base = baseline()
    singles = json.loads((Path(data) / "results.json").read_text())["records"]
    improved, fit_info = refit_single_inputs(base, singles)
    improved.save(out / "single-only.json")
    root = ROOT / "cross_pulse_repair"
    qs = [
        r
        for name in ["development-01", "diagnosis-01"]
        for r in json.loads((root / name / "results.json").read_text())["records"]
    ]
    for name in ["triple-diagnostic-01", "triple-diagnostic-02"]:
        records = json.loads((root / name / "results.json").read_text())["records"]
        y = np.array([r["sensor_full"] for r in records])
        pulses = [(0, 0.08), (2.5, -0.09), (7.5, 0.06)]
        for i, j in [(0, 1), (0, 2), (1, 2)]:
            r1 = y[1 << i] - y[0]
            r2 = y[1 << j] - y[0]
            r12 = y[(1 << i) | (1 << j)] - y[0]
            qs.append(
                {
                    "run_id": f"{name}-pair{i}{j}",
                    "pulses": [pulses[i], pulses[j]],
                    "time": records[0]["time"],
                    "separation": records[0]["separation"],
                    "terms": {
                        "r1": r1.tolist(),
                        "r2": r2.tolist(),
                        "r12": r12.tolist(),
                        "cross": (r12 - r1 - r2).tolist(),
                    },
                }
            )
    model, cross_info = fit_cross_response(improved, qs, conditioned=True)
    model.save(out / "model.json")
    write_json(
        out / "fit.json",
        {
            "single_fit": fit_info,
            "cross_fit": cross_info,
            "cpu_seconds": time.process_time() - st,
            "wall_seconds": time.perf_counter() - sw,
        },
    )
    results = []
    for r in json.loads((root / "fresh-01/results.json").read_text())["records"]:
        yp = model.rollout(r["separation"], r["pulses"], r["time"])
        results.append(
            {
                "run_id": r["run_id"],
                "role": r["role"],
                "prediction": yp.tolist(),
                "candidate": assess(
                    r, yp, json.loads((FROZEN / "freeze.json").read_text())["gates"]
                ),
            }
        )
    write_json(out / "exposed-regression.json", results)
    for role in ["short", "regression", "three"]:
        group = [r for r in results if r["role"] == role]
        print(
            role,
            "worst",
            max(r["candidate"]["metrics"]["all"]["nrmse"] for r in group),
            "passed",
            sum(r["candidate"]["pass"] for r in group),
            "/",
            len(group),
            flush=True,
        )


def qualify_final(output):
    """Refine a new strong reversal quartet and close triple together.

    All variants use their own matched sham. These are numerical diagnostics,
    not new training data or a relaxation of the frozen scientific gates.
    """
    configs = [
        Config(),
        replace(Config(), dt=0.00625),
        replace(Config(), n=1024, dt=0.00625),
        replace(Config(), length=256, n=1024, dt=0.00625),
    ]
    schedules = [
        [(0, 0), (2, 0)],
        [(0, -0.1), (2, 0)],
        [(0, 0), (2, 0.1)],
        [(0, -0.1), (2, 0.1)],
        [(0, 0.08), (2.25, -0.09), (7, 0.06)],
    ]
    out = start_panel(
        output,
        {
            "purpose": "post-freeze numerical qualification only",
            "configs": [asdict(c) for c in configs],
            "seed": 1101,
            "separation": 27.5,
            "schedules": schedules,
            "contract": CONTRACT,
        },
    )
    panels, costs = [], []
    try:
        for c in configs:
            st, sw = time.process_time(), time.perf_counter()
            f = Field(c)
            initial, _ = prepare(f, 27.5, 1101, age=20, residual=0.02)
            pc, pw = time.process_time() - st, time.perf_counter() - sw
            rows, cost = trace(f, initial, 27.5, schedules)
            panels.append(rows)
            costs.append(
                {**cost, "preparation_cpu_seconds": pc, "preparation_wall_seconds": pw}
            )
            write_json(out / "results.json", {"panels": panels, "costs": costs})
            print(asdict(c), costs[-1], flush=True)
        checks = []
        for i, panel in enumerate(panels[1:], 1):
            raw_diff = np.array([r["sensor_full"] for r in panel[:4]]) - np.array(
                [r["sensor_full"] for r in panels[0][:4]]
            )
            response_diff = raw_diff[1:] - raw_diff[0]
            checks.append(
                {
                    "mesh": i,
                    "raw_four_trace_conservative_max_sum": float(
                        abs(raw_diff).max(axis=1).sum()
                    ),
                    "paired_three_response_conservative_max_sum": float(
                        abs(response_diff).max(axis=1).sum()
                    ),
                    "cross_difference_max": float(
                        abs(
                            response_diff[2] - response_diff[0] - response_diff[1]
                        ).max()
                    ),
                    "pair_response_difference_max": float(abs(response_diff[2]).max()),
                    "triple_response_difference_max": float(
                        abs(
                            np.array(panel[4]["response"]) - panels[0][4]["response"]
                        ).max()
                    ),
                }
            )
        write_json(out / "refinement.json", checks)
    except Exception as exc:
        write_json(out / "failure.json", {"status": "FAILED", "error": repr(exc)})
        raise


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "mode",
        choices=[
            "diagnose",
            "refine",
            "development",
            "fit",
            "fit-cross",
            "regression",
            "regression-cross",
            "pulse-audit",
            "freeze",
            "freeze-refined",
            "validate",
            "quick",
            "triple-diagnostic",
            "single-refinement",
            "refit",
            "qualify-final",
        ],
    )
    p.add_argument("--output", required=True)
    p.add_argument("--data")
    p.add_argument("--extra")
    p.add_argument("--powers", type=int, default=1)
    p.add_argument("--seed", type=int, default=741)
    p.add_argument("--separation", type=float, default=27)
    args = p.parse_args()
    if args.mode in ("fit", "fit-cross"):
        fit(args.data, args.extra, args.output, args.powers, args.mode == "fit-cross")
    elif args.mode.startswith("regression"):
        regression(args.data, args.output, args.mode == "regression-cross")
    elif args.mode in ("freeze", "freeze-refined"):
        freeze(args.data, args.output, args.mode == "freeze-refined")
    elif args.mode in ("validate", "quick"):
        validate(args.data, args.output, args.mode == "quick")
    elif args.mode == "triple-diagnostic":
        triple_diagnostic(args.output, args.seed, int(args.separation))
    elif args.mode == "refit":
        refit(args.data, args.output)
    else:
        {
            "diagnose": diagnose,
            "refine": refine,
            "development": development,
            "pulse-audit": pulse_audit,
            "single-refinement": single_refinement,
            "qualify-final": qualify_final,
        }[args.mode](args.output)


if __name__ == "__main__":
    main()
