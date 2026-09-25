"""Bounded three-pattern reference, projection calibration, and hybrid panels."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .explore import start_panel, write_json
from .hybrid_pair import PairHybrid, etd_coefficients, pair_indices
from .measurements import describe, regions, shift
from .simulator import Config, Field

ROOT = Path("work/mediated_patterns/two_way_hybrid")
CFG = Config(length=192, n=768)
HYBRID_CFG = replace(CFG, dt=0.00625)


def prepare_triple(config, seed, c_position, age=40):
    f = Field(config)
    centers = [-14.0, 14.0, float(c_position)]
    states = []
    for j, c in enumerate(centers):
        initial = f.seed(seed=seed + 100 * j, variation=0.01)
        isolated = f.evolve(initial, 150, sample=150)[1][-1]
        states.append(shift(f, isolated, c))
    joined = sum(states)
    rng = np.random.default_rng(seed + 3000)
    for c in centers:
        d = f.x - c
        joined[0] += (
            0.02
            * np.exp(-((d / 5) ** 2))
            * sum(
                rng.normal() * np.cos(j * d / 2 + rng.uniform(0, 2 * np.pi))
                for j in range(1, 5)
            )
        )
    joined[1] += 0.02 * np.exp(-((f.x / 40) ** 2)) * np.cos(f.x / 7)
    return f.evolve(joined, age, sample=age)[1][-1]


def replay_weight(x):
    # Targeted mediator feedback replay around AB only, not a pattern clamp.
    d = np.minimum(abs(x + 14), abs(x - 14))
    z = np.clip((d - 8) / 4, 0, 1)
    return 1 - (6 * z**5 - 15 * z**4 + 10 * z**3)


def readouts(x, fields, c_position):
    centers = [-14.0, 14.0, float(c_position)]
    w = regions(x, centers)
    dx = x[1] - x[0]
    u, m = fields
    return np.r_[(w * u * u).sum(axis=1) * dx, (w * m).sum(axis=1) / w.sum(axis=1)]


def run_reference(
    config,
    initial,
    c_position,
    amplitude=0.1,
    horizon=80,
    pulse_at="C",
    snapshots=False,
):
    st, sw = time.process_time(), time.perf_counter()
    f = Field(config)
    anchor = c_position if pulse_at == "C" else -14
    pulsed = f.pulse(initial, anchor, amplitude, width=8, relative=True, compact=True)
    states = [np.fft.rfft(s, axis=-1) for s in [initial, pulsed, pulsed, initial]]
    weight = replay_weight(f.x)
    times = []
    outputs = []
    saved = []
    descs = []
    stride = round(0.5 / config.dt)
    steps = round(horizon / config.dt)
    for k in range(steps + 1):
        if k % stride == 0:
            fields = [np.fft.irfft(v, n=config.n, axis=-1) for v in states]
            times.append(k * config.dt)
            outputs.append([readouts(f.x, s, c_position) for s in fields])
            descs.append(describe(f.x, fields[1], [-14, 14, c_position]))
            if snapshots:
                saved.append(np.array(fields)[:3])
        if k == steps:
            break
        states[0], stages = f.step(states[0], record=True)
        states[1], _ = f.step(states[1])
        states[2], _ = f.step(states[2], replay=stages, replay_weight=weight)
        states[3], _ = f.step(states[3], replay=stages, replay_weight=weight)
    outputs = np.array(outputs)
    result = {
        "time": times,
        "observables": [
            "A_mass",
            "B_mass",
            "C_mass",
            "A_mediator",
            "B_mediator",
            "C_mediator",
        ],
        "absolute": outputs.tolist(),
        "response": (outputs[:, 1] - outputs[:, 0]).tolist(),
        "return_contrast": (outputs[:, 1] - outputs[:, 2]).tolist(),
        "sham_replay_max": float(abs(outputs[:, 0] - outputs[:, 3]).max()),
        "center": np.array([v["center"] for v in descs]).tolist(),
        "width": np.array([v["width"] for v in descs]).tolist(),
        "cpu_seconds": time.process_time() - st,
        "wall_seconds": time.perf_counter() - sw,
    }
    return result, np.array(saved)


def run_hybrid(
    config, initial, c_position, basis, amplitude=0.1, horizon=80, pulse_at="C"
):
    st, sw = time.process_time(), time.perf_counter()
    # Each branch uses the same permitted initial snapshot once. Sham/replay are
    # evaluator controls, never supplied to the primary closed-loop branch.
    models = [PairHybrid.initialize(config, basis, initial) for _ in range(4)]
    setup_cpu = time.process_time() - st
    anchor = c_position if pulse_at == "C" else -14
    for j in (1, 2):
        models[j].pulse(anchor, amplitude)
    times = []
    outputs = []
    ports = []
    weight = replay_weight(models[0].x)
    stride = round(0.5 / config.dt)
    steps = round(horizon / config.dt)
    for k in range(steps + 1):
        if k % stride == 0:
            times.append(k * config.dt)
            outputs.append([readouts(m.x, m.fields(), c_position) for m in models])
            ports.append({k: v.tolist() for k, v in models[1].ports().items()})
        if k == steps:
            break
        stages = models[0].step(record=True)
        models[1].step()
        models[2].step(replay=stages, replay_weight=weight)
        models[3].step(replay=stages, replay_weight=weight)
    outputs = np.array(outputs)
    return {
        "time": times,
        "absolute": outputs.tolist(),
        "response": (outputs[:, 1] - outputs[:, 0]).tolist(),
        "return_contrast": (outputs[:, 1] - outputs[:, 2]).tolist(),
        "sham_replay_max": float(abs(outputs[:, 0] - outputs[:, 3]).max()),
        "cpu_seconds": time.process_time() - st,
        "wall_seconds": time.perf_counter() - sw,
        "setup_cpu_seconds": setup_cpu,
        "ports": ports,
    }


def score(truth, prediction):
    t = np.array(truth)
    p = np.array(prediction)
    signal = np.sqrt(np.mean(t * t, axis=0))
    error = np.sqrt(np.mean((t - p) ** 2, axis=0))
    return {
        "signal_rms": signal.tolist(),
        "rmse": error.tolist(),
        "max_absolute": abs(t - p).max(axis=0).tolist(),
        "nrmse": np.divide(
            error, signal, out=np.zeros_like(error), where=signal > 0
        ).tolist(),
    }


def pilot(output):
    out = start_panel(
        output,
        {
            "purpose": "three-pattern persistence and return-influence pilot BEFORE reduction",
            "config": asdict(CFG),
            "seed": 71,
            "C_positions": [38, 42],
            "amplitude": 0.1,
            "horizon": 80,
            "controls": "mediator feedback replay around A/B, own sham, matched four stages",
            "finite_cap": "pilot two preparations; expand only after measuring cost and signal",
        },
    )
    rows = []
    for c in [38, 42]:
        st, sw = time.process_time(), time.perf_counter()
        initial = prepare_triple(CFG, 71, c)
        pc, pw = time.process_time() - st, time.perf_counter() - sw
        np.savez_compressed(out / f"initial-c{c}.npz", state=initial)
        row, snap = run_reference(CFG, initial, c, snapshots=True)
        row.update(
            seed=71,
            c_position=c,
            preparation_cpu_seconds=pc,
            preparation_wall_seconds=pw,
        )
        rows.append(row)
        np.savez_compressed(out / f"traces-c{c}.npz", fields=snap, time=row["time"])
        write_json(out / "results.json", rows)
        print(
            c,
            "CPU",
            pc + row["cpu_seconds"],
            "return RMS",
            np.sqrt(np.mean(np.array(row["return_contrast"]) ** 2, axis=0)),
            flush=True,
        )


def consistency(output):
    """Resolved trial space vs monolithic solver, before model fitting."""
    out = start_panel(
        output,
        {
            "purpose": "resolved-interface equivalence and runtime pilot",
            "config": asdict(CFG),
            "source": "pilot-01/initial-c38.npz",
            "horizon": 10,
            "construction": "full-rank AB block, same Fourier collocation operator, affine ETDRK4",
        },
    )
    initial = np.load(ROOT / "pilot-01/initial-c38.npz")["state"]
    x = Field(CFG).x
    basis = np.eye(len(pair_indices(x)))
    truth, _ = run_reference(CFG, initial, 38, horizon=10)
    candidate = run_hybrid(CFG, initial, 38, basis, horizon=10)
    write_json(
        out / "results.json",
        {
            "truth": truth,
            "hybrid": candidate,
            "response": score(truth["response"], candidate["response"]),
            "return": score(truth["return_contrast"], candidate["return_contrast"]),
            "raw_max_error": float(
                abs(np.array(truth["absolute"]) - candidate["absolute"]).max()
            ),
        },
    )
    print(
        "CPU",
        candidate["cpu_seconds"],
        "rawerror",
        float(abs(np.array(truth["absolute"]) - candidate["absolute"]).max()),
        flush=True,
    )


def calibrate(output):
    out = start_panel(
        output,
        {
            "purpose": "shared spatial basis calibration; all fields are training access",
            "config": asdict(CFG),
            "cases": [
                [71, 38, 0.1, "C"],
                [71, 42, 0.1, "C"],
                [82, 38, -0.1, "C"],
                [82, 42, -0.1, "C"],
                [71, 38, 0.05, "A"],
            ],
            "horizon": 80,
            "candidate_orders": [4, 8, 12],
            "budget": "maximum 6 development reference runs and 6 hybrid candidates before review",
            "normalization": "each trajectory drift and pulse-response snapshot block receives unit Frobenius norm",
            "prospective_task_gates": {
                "C_response_nrmse": 0.01,
                "C_response_max": 0.001,
                "B_response_nrmse": 0.1,
                "B_response_max": 2e-5,
                "C_return_nrmse": 0.1,
                "C_return_max": 1e-6,
                "return_signal_floor": 1e-8,
            },
            "note": "no receiver sensor search; primary return readout is fixed C weighted u^2 mass",
        },
    )
    x = Field(CFG).x
    idx = pair_indices(x)
    blocks = []
    records = []
    for seed, c, a, where in [
        (71, 38, 0.1, "C"),
        (71, 42, 0.1, "C"),
        (82, 38, -0.1, "C"),
        (82, 42, -0.1, "C"),
        (71, 38, 0.05, "A"),
    ]:
        if seed == 71:
            initial = np.load(ROOT / f"pilot-01/initial-c{c}.npz")["state"]
            pc = pw = 0.0
        else:
            st, sw = time.process_time(), time.perf_counter()
            initial = prepare_triple(CFG, seed, c, age=60)
            pc, pw = time.process_time() - st, time.perf_counter() - sw
        tag = f"s{seed}-c{c}-{where}"
        np.savez_compressed(out / f"{tag}-initial.npz", state=initial)
        if seed == 71 and where == "C":
            records0 = json.loads((ROOT / "pilot-01/results.json").read_text())
            row = next(r for r in records0 if r["c_position"] == c).copy()
            row.update(cpu_seconds=0.0, wall_seconds=0.0, reused="pilot-01")
            snap = np.load(ROOT / f"pilot-01/traces-c{c}.npz")["fields"]
        else:
            row, snap = run_reference(
                CFG, initial, c, a, pulse_at=where, snapshots=True
            )
        row.update(
            seed=seed,
            c_position=c,
            amplitude=a,
            pulse_at=where,
            preparation_cpu_seconds=pc,
            preparation_wall_seconds=pw,
            run_id=tag,
        )
        records.append(row)
        np.savez_compressed(out / f"{tag}-snapshots.npz", fields=snap, time=row["time"])
        for block in [
            snap[:, 0, 0, idx] - initial[0, idx],
            snap[:, 1, 0, idx] - snap[:, 0, 0, idx],
            snap[:, 2, 0, idx] - snap[:, 0, 0, idx],
        ]:
            blocks.append(block.T / np.linalg.norm(block))
        write_json(out / "results.json", records)
        print(tag, "CPU", pc + row["cpu_seconds"], flush=True)
    st, sw = time.process_time(), time.perf_counter()
    basis, singular, _ = np.linalg.svd(
        np.concatenate(blocks, axis=1), full_matrices=False
    )
    np.savez_compressed(out / "basis.npz", basis=basis, singular=singular)
    write_json(
        out / "fit.json",
        {
            "singular": singular.tolist(),
            "cpu_seconds": time.process_time() - st,
            "wall_seconds": time.perf_counter() - sw,
        },
    )


def development(data, output, order=4):
    data = Path(data)
    basis = np.load(data / "basis.npz")["basis"][:, :order]
    out = start_panel(
        output,
        {
            "purpose": "closed-loop Galerkin development",
            "order": order,
            "calibration": str(data),
            "cases": ["s71-c38-C", "s71-c42-C"],
            "initialization": "own initial pair template + zero deviation; exterior/mediator retained",
            "no_field_refresh": True,
            "dt": CFG.dt,
        },
    )
    records = []
    truth_rows = json.loads((data / "results.json").read_text())
    for tag in ["s71-c38-C", "s71-c42-C"]:
        row = next(r for r in truth_rows if r["run_id"] == tag)
        initial = np.load(data / f"{tag}-initial.npz")["state"]
        h = run_hybrid(CFG, initial, row["c_position"], basis)
        records.append(
            {
                "run_id": tag,
                "hybrid": h,
                "response": score(row["response"], h["response"]),
                "return": score(row["return_contrast"], h["return_contrast"]),
            }
        )
        write_json(out / "results.json", records)
        print(
            tag,
            "CPU",
            h["cpu_seconds"],
            "response",
            records[-1]["response"]["nrmse"],
            "return",
            records[-1]["return"]["nrmse"],
            flush=True,
        )


def qualify(output):
    configs = [
        replace(CFG, dt=0.00625),
        replace(CFG, n=1536, dt=0.00625),
        replace(CFG, length=256, n=1024, dt=0.00625),
    ]
    out = start_panel(
        output,
        {
            "purpose": "new triple and matched return-control numerical qualification",
            "configs": [asdict(c) for c in configs],
            "seed": 71,
            "C_position": 38,
            "amplitude": 0.1,
            "horizon": 80,
            "all_branches_refined": True,
            "reference": "pilot-01",
        },
    )
    base = json.loads((ROOT / "pilot-01/results.json").read_text())[0]
    rows = []
    for c in configs:
        st, sw = time.process_time(), time.perf_counter()
        initial = prepare_triple(c, 71, 38)
        pc, pw = time.process_time() - st, time.perf_counter() - sw
        row, _ = run_reference(c, initial, 38)
        row.update(
            config=asdict(c),
            preparation_cpu_seconds=pc,
            preparation_wall_seconds=pw,
            response_difference=score(base["response"], row["response"]),
            return_difference=score(base["return_contrast"], row["return_contrast"]),
        )
        # Conservative comparison on pulse-minus-own-sham differences, not on
        # huge common static backgrounds. Preserve both accounting choices.
        raw = np.array(base["absolute"]) - row["absolute"]
        paired = raw[:, 1:3] - raw[:, 0, None, :]
        row["conservative_raw_two_branch_bound"] = (
            abs(raw[:, 1:3]).max(axis=0).sum(axis=0).tolist()
        )
        row["conservative_paired_two_branch_bound"] = (
            abs(paired).max(axis=0).sum(axis=0).tolist()
        )
        rows.append(row)
        write_json(out / "results.json", rows)
        print(
            asdict(c),
            "C return delta",
            row["return_difference"]["max_absolute"][2],
            "paired bound",
            row["conservative_paired_two_branch_bound"][2],
            flush=True,
        )


def repair_check(data, output, order=16):
    data = Path(data)
    basis = np.load(data / "basis.npz")["basis"][:, :order]
    out = start_panel(
        output,
        {
            "purpose": "targeted +4 modes for missed fast AB response; halve exchange timestep",
            "reason": "lower orders reproduce return but miss fast B response; retain 2e-5 maximum gate",
            "order": order,
            "calibration": str(data),
            "config": asdict(HYBRID_CFG),
            "cases": ["s71-c38-C", "s71-c42-C", "s82-c38-C", "s82-c42-C"],
            "horizon": 80,
            "finite_cap": "4 development cases; no new coefficients fitted",
        },
    )
    records = []
    truth_rows = json.loads((data / "results.json").read_text())
    for tag in ["s71-c38-C", "s71-c42-C", "s82-c38-C", "s82-c42-C"]:
        old = next(r for r in truth_rows if r["run_id"] == tag)
        initial = np.load(data / f"{tag}-initial.npz")["state"]
        truth, _ = run_reference(
            HYBRID_CFG, initial, old["c_position"], old["amplitude"]
        )
        hybrid = run_hybrid(
            HYBRID_CFG, initial, old["c_position"], basis, old["amplitude"]
        )
        records.append(
            {
                "run_id": tag,
                "truth": truth,
                "hybrid": hybrid,
                "response": score(truth["response"], hybrid["response"]),
                "return": score(truth["return_contrast"], hybrid["return_contrast"]),
            }
        )
        write_json(out / "results.json", records)
        print(
            tag,
            "B max",
            records[-1]["response"]["max_absolute"][1],
            "Cret NRMSE",
            records[-1]["return"]["nrmse"][2],
            flush=True,
        )
    # Same permitted initial state, dt/2 on both reference and hybrid.
    initial = np.load(data / "s71-c38-C-initial.npz")["state"]
    fine = replace(HYBRID_CFG, dt=0.003125)
    truth, _ = run_reference(fine, initial, 38)
    hybrid = run_hybrid(fine, initial, 38, basis)
    raw = np.array(records[0]["truth"]["absolute"]) - truth["absolute"]
    paired = raw[:, 1:3] - raw[:, 0, None, :]
    write_json(
        out / "exchange-refinement.json",
        {
            "truth": truth,
            "hybrid": hybrid,
            "reference_return_difference": score(
                records[0]["truth"]["return_contrast"], truth["return_contrast"]
            ),
            "hybrid_return_difference": score(
                records[0]["hybrid"]["return_contrast"], hybrid["return_contrast"]
            ),
            "reference_paired_conservative_bound": abs(paired)
            .max(axis=0)
            .sum(axis=0)
            .tolist(),
            "hybrid_response_difference": score(
                records[0]["hybrid"]["response"], hybrid["response"]
            ),
        },
    )


def freeze(data, output, order=20):
    data = Path(data)
    out = start_panel(
        output,
        {
            "purpose": "freeze shared pair basis and new response/return task BEFORE fresh outcomes",
            "qualification": "consistency-01, qualification-01, order20-01/exchange-refinement.json",
            "calibration": str(data),
            "order": order,
        },
    )
    basis = np.load(data / "basis.npz")["basis"][:, :order]
    np.savez_compressed(out / "model.npz", basis=basis)
    # Envelope of physically defined force channels, using development truth.
    envelope = []
    for row in json.loads((data / "results.json").read_text()):
        tag = row["run_id"]
        initial = np.load(data / f"{tag}-initial.npz")["state"]
        model = PairHybrid.initialize(HYBRID_CFG, basis, initial)
        snapshots = np.load(data / f"{tag}-snapshots.npz")["fields"]
        for state in snapshots[:, :3].reshape(-1, 2, HYBRID_CFG.n):
            u, m = state
            ext = np.zeros(HYBRID_CFG.n)
            ext[model.outside] = u[model.outside]
            force = np.fft.irfft(
                (HYBRID_CFG.r - (1 - model.k**2) ** 2) * np.fft.rfft(ext),
                n=HYBRID_CFG.n,
            )
            envelope.append(
                np.r_[
                    HYBRID_CFG.feedback * basis.T @ (m[model.inside] * u[model.inside]),
                    basis.T @ force[model.inside],
                ]
            )
    envelope = np.array(envelope)
    write_json(
        out / "freeze.json",
        {
            "status": "FROZEN_EXPLORATORY",
            "config": asdict(HYBRID_CFG),
            "order": order,
            "model_sha256": hashlib.sha256(
                (out / "model.npz").read_bytes()
            ).hexdigest(),
            "training_preparations": ["s71-c38", "s71-c42", "s82-c38", "s82-c42"],
            "seeds": [2011, 2022, 2033],
            "cases": [[39.0, 0.075], [41.0, -0.08]],
            "ages": "20 odd seed /60 even seed",
            "horizon": 80,
            "sample": 0.5,
            "anchors": [-14, 14],
            "initial_access": "one initial snapshot; retain AB static template only, C/exterior u and full mediator",
            "gates": {
                "C_response_nrmse": 0.01,
                "C_response_max": 0.001,
                "B_response_nrmse": 0.1,
                "B_response_max": 2e-5,
                "C_return_nrmse": 0.1,
                "C_return_max": 1e-6,
                "return_signal_floor": 2e-7,
            },
            "floor_change_from_prospective": "raised before fresh evaluation to 2e-7, above 1.105e-7 conservative dt-refinement paired bound; accuracy gates unchanged",
            "return_definition": "full minus own-sham-mediator replay in A/B feedback only; identical control in hybrid",
            "primary_contract": "no replay, sham, future samples or AB dynamic grid enter closed-loop primary branch",
            "port_calibration_min": envelope.min(axis=0).tolist(),
            "port_calibration_max": envelope.max(axis=0).tolist(),
            "no_claim": "old pulse-model reuse, arbitrary input family, minimality, unique physical state",
        },
    )


def assess_case(truth, hybrid, gates):
    response = score(truth["response"], hybrid["response"])
    contrast = score(truth["return_contrast"], hybrid["return_contrast"])
    failed = []
    for label, index in [("C", 2), ("B", 1)]:
        if response["nrmse"][index] > gates[label + "_response_nrmse"]:
            failed.append(label + "_relative")
        if response["max_absolute"][index] > gates[label + "_response_max"]:
            failed.append(label + "_absolute")
    resolved = contrast["signal_rms"][2] > gates["return_signal_floor"]
    if resolved and contrast["nrmse"][2] > gates["C_return_nrmse"]:
        failed.append("return_relative")
    if contrast["max_absolute"][2] > gates["C_return_max"]:
        failed.append("return_absolute")
    return {
        "response": response,
        "return": contrast,
        "failed": failed,
        "return_resolved": resolved,
        "pass": not failed and resolved,
    }


def fresh(data, output, quick=False):
    data = Path(data)
    frozen = json.loads((data / "freeze.json").read_text())
    if (
        hashlib.sha256((data / "model.npz").read_bytes()).hexdigest()
        != frozen["model_sha256"]
    ):
        raise ValueError("Frozen model changed")
    out = start_panel(
        output,
        {
            "purpose": "new whole-preparation closed-loop evaluation"
            if not quick
            else "reproduction only",
            "freeze": frozen,
            "freeze_path": str(data),
            "quick": quick,
        },
    )
    cfg = Config(**frozen["config"])
    basis = np.load(data / "model.npz")["basis"]
    rows = []
    cases = [(s, c, a) for s in frozen["seeds"] for c, a in frozen["cases"]]
    if quick:
        cases = cases[:1]
    for seed, c, a in cases:
        tag = f"s{seed}-c{c:g}"
        if tag in frozen["training_preparations"]:
            raise ValueError("Preparation split overlap")
        st, sw = time.process_time(), time.perf_counter()
        initial = prepare_triple(cfg, seed, c, age=20 if seed % 2 else 60)
        pc, pw = time.process_time() - st, time.perf_counter() - sw
        np.savez_compressed(out / f"{tag}-initial.npz", state=initial)
        truth, _ = run_reference(cfg, initial, c, a)
        hybrid = run_hybrid(cfg, initial, c, basis, a)
        values = np.array(
            [
                np.r_[p["incident_mediator_force"], p["incident_pattern_force"]]
                for p in hybrid["ports"]
            ]
        )
        low, high = (
            np.array(frozen["port_calibration_min"]),
            np.array(frozen["port_calibration_max"]),
        )
        excursion = np.maximum(low - values, values - high)
        excursion = np.maximum(excursion, 0)
        rows.append(
            {
                "run_id": tag,
                "seed": seed,
                "c_position": c,
                "amplitude": a,
                "truth": truth,
                "hybrid": hybrid,
                "scores": assess_case(truth, hybrid, frozen["gates"]),
                "port_excursion": {
                    "max_absolute": float(excursion.max()),
                    "max_training_span_fraction": float(
                        (excursion / np.maximum(high - low, 1e-12)).max()
                    ),
                    "fraction_samples_outside_any_channel": float(
                        np.mean((excursion > 1e-12).any(axis=1))
                    ),
                    "clipped": False,
                },
                "preparation_cpu_seconds": pc,
                "preparation_wall_seconds": pw,
            }
        )
        write_json(out / "results.json", rows)
        print(
            tag,
            rows[-1]["scores"],
            "port excursion",
            rows[-1]["port_excursion"],
            flush=True,
        )


def openloop(output):
    """Diagnostic only: true RK-stage exterior/mediator drive a projected AB.

    No input trace from this function enters the primary hybrid. This compares
    missing spatial dynamics with feedback amplification, on exposed seed71.
    """
    from scipy.linalg import eigh

    out = start_panel(
        output,
        {
            "purpose": "teacher-forced AB diagnostic, NOT deployment",
            "config": asdict(HYBRID_CFG),
            "orders": [4, 20],
            "seed": 71,
            "C_position": 38,
            "amplitude": 0.1,
            "input_access": "true exterior u and mediator at each reference RK stage",
            "no_fitting": True,
        },
    )
    st, sw = time.process_time(), time.perf_counter()
    f = Field(HYBRID_CFG)
    idx = pair_indices(f.x)
    outside = np.setdiff1d(np.arange(f.c.n), idx)
    initial = np.load(ROOT / "calibration-01/s71-c38-C-initial.npz")["state"]
    pulsed = f.pulse(initial, 38, 0.1, width=8, relative=True, compact=True)
    v = [np.fft.rfft(s, axis=-1) for s in [initial, pulsed]]
    allbasis = np.load(ROOT / "calibration-01/basis.npz")["basis"]
    template = initial[0, idx]
    linear = f.linear[0]

    def apply_l(u):
        return np.fft.irfft(linear * np.fft.rfft(u), n=f.c.n)

    T = np.zeros(f.c.n)
    T[idx] = template
    solvers = []
    for rank in [4, 20]:
        b = allbasis[:, :rank]
        embed = np.zeros((f.c.n, rank))
        embed[idx] = b
        lb = np.fft.irfft(linear[:, None] * np.fft.rfft(embed, axis=0), n=f.c.n, axis=0)
        eig, rotation = eigh(b.T @ lb[idx])
        basis = b @ rotation
        solvers.append(
            {
                "rank": rank,
                "basis": basis,
                "constant": basis.T @ apply_l(T)[idx],
                "coeff": etd_coefficients(eig, f.c.dt),
                "z": np.zeros((2, rank)),
            }
        )
    times = []
    truth = []
    outputs = {str(r["rank"]): [] for r in solvers}
    projected = {str(r["rank"]): [] for r in solvers}
    weights = regions(f.x, [-14, 14, 38])[1, idx] * (f.x[1] - f.x[0])
    for tick in range(round(80 / f.c.dt) + 1):
        if tick % round(0.5 / f.c.dt) == 0:
            fields = [np.fft.irfft(q, n=f.c.n, axis=-1) for q in v]
            truth.append([float(weights @ (s[0, idx] ** 2)) for s in fields])
            times.append(tick * f.c.dt)
            for s in solvers:
                b = s["basis"]
                outputs[str(s["rank"])].append(
                    [float(weights @ ((template + b @ z) ** 2)) for z in s["z"]]
                )
                projected[str(s["rank"])].append(
                    [
                        float(
                            weights
                            @ ((template + b @ (b.T @ (u[0, idx] - template))) ** 2)
                        )
                        for u in fields
                    ]
                )
        if tick == round(80 / f.c.dt):
            break
        for branch in range(2):
            old = v[branch]
            nv = f.nonlinear(old)
            a = f.e2 * old + f.q * nv
            na = f.nonlinear(a)
            b = f.e2 * old + f.q * na
            nb = f.nonlinear(b)
            c = f.e2 * a + f.q * (2 * nb - nv)
            stages = [np.fft.irfft(q, n=f.c.n, axis=-1) for q in [old, a, b, c]]
            v[branch], _ = f.step(old)
            forcing = []
            for u, m in stages:
                ext = np.zeros(f.c.n)
                ext[outside] = u[outside]
                forcing.append((apply_l(ext)[idx], m[idx]))
            for s in solvers:
                e, e2, q, f1, f2, f3 = s["coeff"]
                basis = s["basis"]
                z = s["z"][branch]

                def nl(z, j, basis=basis, forcing=forcing, constant=s["constant"]):
                    u = template + basis @ z
                    tail, m = forcing[j]
                    return constant + basis.T @ (
                        tail + f.c.cubic * u**3 - u**5 + f.c.feedback * m * u
                    )

                nz = nl(z, 0)
                aa = e2 * z + q * nz
                an = nl(aa, 1)
                bb = e2 * z + q * an
                bn = nl(bb, 2)
                cc = e2 * aa + q * (2 * bn - nz)
                cn = nl(cc, 3)
                s["z"][branch] = e * z + f1 * nz + 2 * f2 * (an + bn) + f3 * cn
    truth = np.array(truth)
    response = truth[:, 1] - truth[:, 0]
    rows = []
    for s in solvers:
        y = np.array(outputs[str(s["rank"])])
        p = np.array(projected[str(s["rank"])])
        rows.append(
            {
                "rank": s["rank"],
                "predicted_response": (y[:, 1] - y[:, 0]).tolist(),
                "response_error": score(
                    response[:, None], (y[:, 1] - y[:, 0])[:, None]
                ),
                "instantaneous_truth_projection_error": score(
                    response[:, None], (p[:, 1] - p[:, 0])[:, None]
                ),
            }
        )
    write_json(
        out / "results.json",
        {
            "time": times,
            "truth_response": response.tolist(),
            "records": rows,
            "cpu_seconds": time.process_time() - st,
            "wall_seconds": time.perf_counter() - sw,
        },
    )
    print(rows, flush=True)


def weak_refinement(output):
    from scipy.signal import resample

    out = start_panel(
        output,
        {
            "purpose": "post-freeze weak-return numerical spot check; no fitting",
            "source": "fresh-01/s2022-c41-initial.npz",
            "refinements": ["dt/2", "dt/2 and N*2"],
            "C_position": 41,
            "amplitude": -0.08,
            "gates": "unchanged frozen-01",
        },
    )
    initial = np.load(ROOT / "fresh-01/s2022-c41-initial.npz")["state"]
    base = next(
        r
        for r in json.loads((ROOT / "fresh-01/results.json").read_text())
        if r["run_id"] == "s2022-c41"
    )["truth"]
    rows = []
    for cfg in [
        replace(HYBRID_CFG, dt=0.003125),
        replace(HYBRID_CFG, dt=0.003125, n=1536),
    ]:
        state = initial if cfg.n == HYBRID_CFG.n else resample(initial, cfg.n, axis=-1)
        truth, _ = run_reference(cfg, state, 41, -0.08)
        raw = np.array(base["absolute"]) - truth["absolute"]
        paired = raw[:, 1:3] - raw[:, 0, None, :]
        rows.append(
            {
                "config": asdict(cfg),
                "truth": truth,
                "response_difference": score(base["response"], truth["response"]),
                "return_difference": score(
                    base["return_contrast"], truth["return_contrast"]
                ),
                "paired_conservative_bound": abs(paired)
                .max(axis=0)
                .sum(axis=0)
                .tolist(),
                "paired_bound_after10": abs(paired[np.array(base["time"]) >= 10])
                .max(axis=0)
                .sum(axis=0)
                .tolist(),
            }
        )
        write_json(out / "results.json", rows)
        print(
            cfg.n,
            "return delta",
            rows[-1]["return_difference"]["max_absolute"][2],
            flush=True,
        )


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument(
        "mode",
        choices=[
            "pilot",
            "consistency",
            "calibrate",
            "development",
            "qualify",
            "repair-check",
            "freeze",
            "fresh",
            "quick",
            "openloop",
            "weak-refinement",
        ],
    )
    p.add_argument("--output", required=True)
    p.add_argument("--data")
    p.add_argument("--order", type=int, default=4)
    args = p.parse_args()
    try:
        if args.mode == "development":
            development(args.data, args.output, args.order)
        elif args.mode == "repair-check":
            repair_check(args.data, args.output, args.order)
        elif args.mode == "freeze":
            freeze(args.data, args.output, args.order)
        elif args.mode in ("fresh", "quick"):
            fresh(args.data, args.output, args.mode == "quick")
        elif args.mode == "openloop":
            openloop(args.output)
        elif args.mode == "weak-refinement":
            weak_refinement(args.output)
        else:
            {
                "pilot": pilot,
                "consistency": consistency,
                "calibrate": calibrate,
                "qualify": qualify,
            }[args.mode](args.output)
    except Exception as exc:
        if Path(args.output).exists():
            write_json(
                Path(args.output) / "failure.json",
                {"status": "FAILED", "error": repr(exc)},
            )
        raise


if __name__ == "__main__":
    main()
