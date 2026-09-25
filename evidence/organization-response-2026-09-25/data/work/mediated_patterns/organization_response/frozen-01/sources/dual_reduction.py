"""Staged separate-C qualification and frozen-module composition, CPU only."""

import argparse
import hashlib
import resource
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .dual_pair import DualHybrid, c_indices
from .explore import start_panel, write_json
from .hybrid_pair import PairHybrid, pulse_window
from .hybrid_schedule import controlled_responses, event_ticks
from .measurements import describe
from .repeated_hybrid import (
    assess,
    baseline,
    get_scales,
    load,
    metric,
    refinement_difference,
)
from .two_way_hybrid import prepare_triple, readouts, replay_weight

ROOT = Path("work/mediated_patterns/dual_reduction")
OLD = Path("work/mediated_patterns/two_way_hybrid")
FROZEN = ROOT / "frozen-01"
PANEL = ROOT / "fresh-01"
PLANS = {
    "single": {"events": [[0, 0.075]], "horizon": 80},
    "reverse": {"events": [[0, 0.075], [8, -0.075]], "horizon": 80},
}


def rss_bytes():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
        1 if sys.platform == "darwin" else 1024
    )


def charge(cpu, wall, label):
    path = ROOT / "budget.json"
    log = load(path) if path.exists() else []
    log.append({"label": label, "cpu_seconds": cpu, "wall_seconds": wall})
    write_json(path, log)
    if sum(v["cpu_seconds"] for v in log) > 2700:
        raise RuntimeError("Aggregate CPU cap exhausted; stop for review")


def model(config, initial, position, kind, cbasis):
    if kind not in {"RF", "FR", "RR"}:
        raise ValueError("Hybrid kind must be RF, FR or RR")
    _, _, ab = baseline()
    if kind == "RF":
        return PairHybrid.initialize(config, ab, initial)
    return DualHybrid.initialize(
        config, position, ab if kind == "RR" else None, cbasis, initial
    )


def run(config, initial, position, plan, kind, cbasis=None, save_fields=False):
    st, sw = time.process_time(), time.perf_counter()
    limits = (
        load(ROOT / "limits.json")
        if (ROOT / "limits.json").exists()
        else {"cpu_per_run": 120, "wall_per_run": 180, "rss_bytes": 768 * 1024**2}
    )
    if (ROOT / "budget.json").exists() and sum(
        v["cpu_seconds"] for v in load(ROOT / "budget.json")
    ) > 2580:
        raise RuntimeError("Insufficient reserved CPU for another capped run")
    models = [model(config, initial, position, kind, cbasis) for _ in range(4)]
    setup = time.process_time() - st
    x = models[0].x
    ticks = dict(event_ticks(plan["events"], config.dt, plan["horizon"]))
    nsteps = round(plan["horizon"] / config.dt)
    stride = round(0.5 / config.dt)
    mask = replay_weight(x)
    idx = c_indices(x, position)
    ts = []
    values = []
    shape = []
    fields = []
    ports = []
    ab_ports = []
    coordinates = []
    jumps = []
    initial_error = float(abs(models[1].fields() - initial).max())
    for k in range(nsteps + 1):
        if ticks.get(k):
            for j in [1, 2]:
                before = models[j].fields()
                exact = before.copy()
                exact[0] += (
                    ticks[k] * pulse_window(x, position, config.length) * before[0]
                )
                diag = models[j].pulse(position, ticks[k])
                if j == 1:
                    jumps.append(
                        {
                            "time": k * config.dt,
                            "amplitude": ticks[k],
                            "C_mass_projection_error": float(
                                abs(
                                    readouts(x, models[j].fields(), position)[2]
                                    - readouts(x, exact, position)[2]
                                )
                            ),
                            "field_increment": diag,
                        }
                    )
        if k % stride == 0:
            current = [m.fields() for m in models]
            ts.append(k * config.dt)
            values.append([readouts(x, s, position) for s in current])
            shape.append(describe(x, current[1], [-14, 14, position]))
            fields.append(np.array(current)[:, 0, idx])
            if isinstance(models[1], DualHybrid):
                ports.append(models[1].block_ports("C").tolist())
                ab_ports.append(np.asarray(models[1].block_ports("AB")).tolist())
                native = models[1].rotation @ models[1].q
                coordinates.append(native[: models[1].rab + models[1].rc].tolist())
            else:
                port = models[1].ports()
                ab_ports.append(
                    np.r_[
                        port["incident_mediator_force"], port["incident_pattern_force"]
                    ].tolist()
                )
                coordinates.append(port["outgoing_coordinates"].tolist())
            if (
                time.process_time() - st > limits["cpu_per_run"]
                or time.perf_counter() - sw > limits["wall_per_run"]
                or rss_bytes() > limits["rss_bytes"]
            ):
                raise RuntimeError("Per-run resource cap exceeded; no continuation")
        if k == nsteps:
            break
        stages = models[0].step(record=True)
        models[1].step()
        models[2].step(replay=stages, replay_weight=mask)
        models[3].step(replay=stages, replay_weight=mask)
    v = np.array(values)
    full, cut, contrast = controlled_responses(v)
    result = {
        "time": ts,
        "absolute": v.tolist(),
        "response": full.tolist(),
        "cut_response": cut.tolist(),
        "return_contrast": contrast.tolist(),
        "sham_replay_max": float(abs(v[:, 0] - v[:, 3]).max()),
        "geometry": {
            key: np.array([s[key] for s in shape]).tolist() for key in shape[0]
        },
        "C_fields": np.array(fields).tolist(),
        "C_ports": ports,
        "AB_ports": ab_ports,
        "coordinates": coordinates,
        "initial_field_max_error": initial_error,
        "jump_projection": jumps,
        "setup_cpu_seconds": setup,
        "peak_rss_bytes": rss_bytes(),
        "cpu_seconds": time.process_time() - st,
        "wall_seconds": time.perf_counter() - sw,
    }
    charge(result["cpu_seconds"], result["wall_seconds"], kind)
    return result, models[1] if save_fields else None


def full_training(config, initial, position, plan):
    """Resolved AB+C snapshots at exact shared sample/event times, evaluator only."""
    from .simulator import Field

    st, sw = time.process_time(), time.perf_counter()
    f = Field(config)
    ticks = dict(event_ticks(plan["events"], config.dt, plan["horizon"]))
    states = [np.fft.rfft(initial, axis=-1) for _ in range(4)]
    outputs = []
    saved = []
    times = []
    for k in range(round(plan["horizon"] / config.dt) + 1):
        if ticks.get(k):
            for j in [1, 2]:
                states[j] = np.fft.rfft(
                    f.pulse(
                        np.fft.irfft(states[j], n=config.n, axis=-1),
                        position,
                        ticks[k],
                        width=8,
                        relative=True,
                        compact=True,
                    ),
                    axis=-1,
                )
        if k % round(0.5 / config.dt) == 0:
            cur = np.array([np.fft.irfft(v, n=config.n, axis=-1) for v in states])
            saved.append(cur)
            times.append(k * config.dt)
            outputs.append([readouts(f.x, s, position) for s in cur])
            if (
                time.process_time() - st > 120
                or time.perf_counter() - sw > 180
                or rss_bytes() > 768 * 1024**2
            ):
                raise RuntimeError("Full-reference per-run resource cap exceeded")
        if k == round(plan["horizon"] / config.dt):
            break
        states[0], stage = f.step(states[0], record=True)
        states[1], _ = f.step(states[1])
        states[2], _ = f.step(states[2], replay=stage, replay_weight=replay_weight(f.x))
        states[3], _ = f.step(states[3], replay=stage, replay_weight=replay_weight(f.x))
    v = np.array(outputs)
    full, cut, contrast = controlled_responses(v)
    saved = np.array(saved)
    geometry = [describe(f.x, s[1], [-14, 14, position]) for s in saved]
    result = {
        "time": times,
        "absolute": v.tolist(),
        "response": full.tolist(),
        "cut_response": cut.tolist(),
        "return_contrast": contrast.tolist(),
        "sham_replay_max": float(abs(v[:, 0] - v[:, 3]).max()),
        "C_fields": saved[:, :, 0, c_indices(f.x, position)].tolist(),
        "geometry": {
            key: np.array([s[key] for s in geometry]).tolist() for key in geometry[0]
        },
        "cpu_seconds": time.process_time() - st,
        "wall_seconds": time.perf_counter() - sw,
    }
    charge(result["cpu_seconds"], result["wall_seconds"], "FF snapshots")
    return result, saved


def evaluate(ff, pred, plan):
    f, _, _ = baseline()
    a = assess(ff, pred, **plan, gates=f["gates"], scales=get_scales())
    ctrue = np.array(ff["C_fields"])
    cpred = np.array(pred["C_fields"])
    local = float(np.linalg.norm(cpred - ctrue) / np.linalg.norm(ctrue))
    sham = float(
        abs(
            np.array(pred["absolute"])[:, 0, 2] - np.array(ff["absolute"])[:, 0, 2]
        ).max()
    )
    jump = max(
        (j["C_mass_projection_error"] for j in pred["jump_projection"]), default=0
    )
    additional = {
        "C_field_relative_rms": local,
        "C_sham_mass_max_error": sham,
        "C_event_projection_max": jump,
        "initial_field_max": pred["initial_field_max_error"],
    }
    for name, limit in [
        ("C_field_relative_rms", 0.01),
        ("C_sham_mass_max_error", 0.005),
        ("C_event_projection_max", 0.001),
        ("initial_field_max", 1e-10),
    ]:
        if additional[name] > limit:
            a["failures"].append(name)
    a["resolved_task_pass"] = a["resolved_task_pass"] and not a["failures"]
    if a["failures"]:
        a["status"] = "FAIL"
    a["additional"] = additional
    return a


def calibrate(output):
    out = start_panel(
        output,
        {
            "stage": "C construction only; no RR outcomes",
            "source": "saved full AB+C development snapshots",
            "C_region": "center +/-9, 72 points",
            "initial_order": 8,
            "selection": "FR only",
            "AB": "frozen",
        },
    )
    st, sw = time.process_time(), time.perf_counter()
    _, cfg, _ = baseline()
    blocks = []
    rows = load(OLD / "calibration-01/results.json")
    x = np.arange(cfg.n) * cfg.length / cfg.n - cfg.length / 2
    for row in rows:
        idx = c_indices(x, row["c_position"])
        tag = row["run_id"]
        initial = np.load(OLD / "calibration-01" / f"{tag}-initial.npz")["state"]
        snap = np.load(OLD / "calibration-01" / f"{tag}-snapshots.npz")["fields"]
        for a in [
            snap[:, 0, 0, idx] - initial[0, idx],
            snap[:, 1, 0, idx] - snap[:, 0, 0, idx],
            snap[:, 1, 0, idx] - snap[:, 2, 0, idx],
        ]:
            if np.linalg.norm(a) > 1e-14:
                blocks.append(a.T / np.linalg.norm(a))
        # Physical event increments, not a prescribed post-event trajectory.
        for amplitude in [-0.1, 0.1]:
            jump = (
                amplitude
                * pulse_window(x, row["c_position"], cfg.length)[idx]
                * initial[0, idx]
            )
            blocks.append(jump[:, None] / np.linalg.norm(jump))
    matrix = np.concatenate(blocks, axis=1)
    b, s, _ = np.linalg.svd(matrix, full_matrices=False)
    np.savez_compressed(out / "basis.npz", basis=b, singular=s)
    write_json(
        out / "fit.json",
        {
            "singular": s.tolist(),
            "training_preparations": ["s71-c38", "s71-c42", "s82-c38", "s82-c42"],
            "source_runs": [r["run_id"] for r in rows],
            "weights": "unit Frobenius per drift/direct response/return block and per jump",
            "cpu_seconds": time.process_time() - st,
            "wall_seconds": time.perf_counter() - sw,
        },
    )
    charge(time.process_time() - st, time.perf_counter() - sw, "C POD construction")


def pilot(output):
    out = start_panel(
        output,
        {
            "stage": "partition consistency, pilot cost, FR only; no RR predictive outcomes",
            "horizon": 10,
        },
    )
    _, cfg, ab = baseline()
    initial = np.load(OLD / "fresh-01/s2011-c39-initial.npz")["state"]
    basis = np.load(ROOT / "calibration-01/basis.npz")["basis"][:, :8]
    plan = {"events": [[0, 0.075]], "horizon": 10}
    ff, _ = full_training(cfg, initial, 39, plan)
    fr, _ = run(cfg, initial, 39, plan, "FR", basis)
    # Full-rank C is a resolved partition; no reduced AB+C accuracy inspected.
    resolved, _ = run(
        cfg,
        initial,
        39,
        {"events": [[0, 0.075]], "horizon": 2},
        "FR",
        np.eye(len(basis)),
    )
    partition = float(
        abs(np.array(resolved["absolute"]) - np.array(ff["absolute"])[:5]).max()
    )
    # AB-only assembly regression: C remains identity, no new C approximation.
    generalized = DualHybrid.initialize(cfg, 39, ab, None, initial)
    original = PairHybrid.initialize(cfg, ab, initial)
    for _ in range(10):
        generalized.step()
        original.step()
    ab_error = float(abs(generalized.fields() - original.fields()).max())
    write_json(
        out / "results.json",
        {
            "FF": ff,
            "FR": fr,
            "resolved_interface": resolved,
            "partition_readout_max_error": partition,
            "AB_only_assembly_max_error": ab_error,
            "estimated_FR_107_four_branches_cpu": fr["cpu_seconds"] * 10.7,
            "peak_rss_bytes": rss_bytes(),
        },
    )
    if partition > 1e-9 or ab_error > 1e-11:
        raise RuntimeError("Resolved partition/AB regression failed")
    limits = {
        "cpu_per_run": 120,
        "wall_per_run": 180,
        "rss_bytes": 768 * 1024**2,
        "aggregate_cpu": 2700,
        "reserve_fresh_cpu": 900,
        "pilot_FR10_cpu": fr["cpu_seconds"],
        "projection": "12 fresh cases x FF/RF/FR/RR plus qualification expected <1100 CPU seconds; total allowance <2700",
        "stop": "no run beyond limits without substantive review",
    }
    write_json(ROOT / "limits.json", limits)


def qualify_c(output, order):
    out = start_panel(
        output,
        {
            "stage": "closed-loop FR qualification before C freeze; RR unavailable",
            "C_order": order,
            "cases": [
                ["s2011-c39", "single"],
                ["s2011-c39", "reverse"],
                ["s2022-c41", "reverse"],
                ["s2022-c41", "single"],
            ],
            "reason": "initial candidate"
            if order == 8
            else "modest C enrichment after FR failure; no RR exposure",
        },
    )
    _, cfg, _ = baseline()
    cb = np.load(ROOT / "calibration-01/basis.npz")["basis"][:, :order]
    rows = []
    for tag, name in [
        ("s2011-c39", "single"),
        ("s2011-c39", "reverse"),
        ("s2022-c41", "reverse"),
        ("s2022-c41", "single"),
    ]:
        pos = float(tag.split("-c")[1])
        initial = np.load(OLD / "fresh-01" / f"{tag}-initial.npz")["state"]
        plan = PLANS[name]
        cache = ROOT / "c-reference"
        cache.mkdir(exist_ok=True)
        file = cache / f"{tag}-{name}.json"
        if file.exists():
            ff = load(file)
        else:
            ff, snap = full_training(cfg, initial, pos, plan)
            write_json(file, ff)
            np.savez_compressed(cache / f"{tag}-{name}.npz", fields=snap)
        fr, _ = run(cfg, initial, pos, plan, "FR", cb)
        row = {
            "preparation_id": tag,
            "plan": plan,
            "name": name,
            "FR": fr,
            "assessment": evaluate(ff, fr, plan),
            "FF_path": str(file),
        }
        rows.append(row)
        write_json(out / "results.json", rows)
        print(tag, name, order, row["assessment"]["failures"], flush=True)


def freeze_c(output, qualification, order):
    rows = load(Path(qualification) / "results.json")
    qualification_protocol = load(Path(qualification) / "protocol.json")
    assert qualification_protocol["C_order"] == order
    adapted = qualification_protocol.get("composition_informed", False)
    if any(r["assessment"]["failures"] for r in rows):
        raise ValueError("C failed a gate; no RR freeze")
    for name in ["single", "reverse"]:
        if not any(
            r["name"] == name and r["assessment"]["resolved_task_pass"] for r in rows
        ):
            raise ValueError("No resolved FR qualification for " + name)
    out = start_panel(
        output,
        {
            "stage": "freeze FR requalified composition-informed C"
            if adapted
            else "freeze C BEFORE any RR predictive outcomes",
            "qualification": qualification,
            "C_order": order,
        },
    )
    cb = np.load(ROOT / "calibration-01/basis.npz")["basis"][:, :order]
    np.savez_compressed(out / "model.npz", basis=cb)
    f, cfg, _ = baseline()
    write_json(
        out / "freeze.json",
        {
            "status": "COMPOSITION_INFORMED_FR_REQUALIFIED"
            if adapted
            else "SEPARATELY_FR_QUALIFIED",
            "composition_informed": adapted,
            "selection_exposure": [r["preparation_id"] for r in rows],
            "C_order": order,
            "C_model_sha256": hashlib.sha256(
                (out / "model.npz").read_bytes()
            ).hexdigest(),
            "AB_model_sha256": f["model_sha256"],
            "config": asdict(cfg),
            "gates": f["gates"],
            "additional_gates": {
                "C_field_relative_rms": 0.01,
                "C_sham_mass_max": 0.005,
                "C_jump_mass_max": 0.001,
                "initial_max": 1e-10,
            },
            "training": ["s71-c38", "s71-c42", "s82-c38", "s82-c42"],
            "FR_development": ["s2011-c39", "s2022-c41"],
            "RR_exposure_before_freeze": adapted,
            "qualification_resolution": {
                r["preparation_id"] + "-" + r["name"]: r["assessment"]["status"]
                for r in rows
            },
            "fresh_seeds": [5011, 5022, 5033] if adapted else [4011, 4022, 4033],
            "positions": [39, 41],
            "age": "20 odd /60 even",
            "plans": {
                "two": {"events": [[0, 0.08], [12, -0.07]], "horizon": 80},
                "three": {
                    "events": [[0, -0.075], [11, 0.07], [29, -0.08]],
                    "horizon": 109,
                },
            }
            if adapted
            else {
                "two": {"events": [[0, 0.075], [10, -0.065]], "horizon": 80},
                "three": {
                    "events": [[0, -0.08], [9, 0.065], [27, -0.075]],
                    "horizon": 107,
                },
            },
            "AB_unchanged": True,
            "joint_fit": False,
            "windows": "same repeated-hybrid whole/event20/last+10/final20; early5 absolute-only",
            "stop": "first independently frozen composition preserved; adaptation if any separately named",
        },
    )


def frozen_c():
    f = load(FROZEN / "freeze.json")
    p = FROZEN / "model.npz"
    if hashlib.sha256(p.read_bytes()).hexdigest() != f["C_model_sha256"]:
        raise ValueError("Frozen C changed")
    return f, np.load(p)["basis"]


def adapt(output, order):
    """C-only order repair after exposed first-panel failure; never joint fit."""
    out = start_panel(
        output,
        {
            "stage": "composition-informed C order enrichment, FR requalification",
            "composition_informed": True,
            "C_order": order,
            "basis": "unchanged original C POD, additional retained columns only",
            "AB": "unchanged frozen20",
            "cases": [
                "s2011-c39-single",
                "s2011-c39-reverse",
                "s4011-c41-three",
                "s4022-c39-three",
            ],
        },
    )
    _, cfg, _ = baseline()
    cb = np.load(ROOT / "calibration-01/basis.npz")["basis"][:, :order]
    rows = []
    for name in ["single", "reverse"]:
        initial = np.load(OLD / "fresh-01/s2011-c39-initial.npz")["state"]
        ff = load(ROOT / "c-reference" / f"s2011-c39-{name}.json")
        fr, _ = run(cfg, initial, 39, PLANS[name], "FR", cb)
        rows.append(
            {
                "preparation_id": "s2011-c39",
                "name": name,
                "plan": PLANS[name],
                "FR": fr,
                "assessment": evaluate(ff, fr, PLANS[name]),
            }
        )
        write_json(out / "results.json", rows)
        print("FR repair", name, rows[-1]["assessment"]["failures"], flush=True)
    exposed = load(ROOT / "fresh-01/results.json")
    for tag in ["s4011-c41-three", "s4022-c39-three"]:
        row = next(r for r in exposed if r["run_id"] == tag)
        initial = np.load(ROOT / "fresh-01" / f"{row['preparation_id']}-initial.npz")[
            "state"
        ]
        fr, _ = run(cfg, initial, row["position"], row["plan"], "FR", cb)
        rows.append(
            {
                "preparation_id": row["preparation_id"],
                "name": "three",
                "plan": row["plan"],
                "FR": fr,
                "assessment": evaluate(row["FF"], fr, row["plan"]),
            }
        )
        write_json(out / "results.json", rows)
        print("FR repair", tag, rows[-1]["assessment"]["failures"], flush=True)


def compose(output, fresh=False, quick=False):
    f, cb = frozen_c()
    _, cfg, _ = baseline()
    out = start_panel(
        output,
        {
            "stage": "quick reproduction of already exposed frozen cases"
            if quick
            else "fresh frozen composition"
            if fresh
            else "first frozen RR development",
            "freeze": f,
            "quick": quick,
        },
    )
    cases = (
        [(s, c) for s in f["fresh_seeds"] for c in f["positions"]]
        if fresh
        else [(2011, 39), (2022, 41)]
    )
    if quick:
        cases = cases[:1]
    records = []
    for seed, pos in cases:
        tag = f"s{seed}-c{pos}"
        if fresh and tag in f["training"] + f["FR_development"] + f.get(
            "selection_exposure", []
        ):
            raise ValueError("Exposure overlap")
        if fresh:
            st, sw = time.process_time(), time.perf_counter()
            initial = prepare_triple(cfg, seed, pos, age=20 if seed % 2 else 60)
            charge(
                time.process_time() - st,
                time.perf_counter() - sw,
                "fresh preparation " + tag,
            )
        else:
            initial = np.load(OLD / "fresh-01" / f"{tag}-initial.npz")["state"]
        np.savez_compressed(out / f"{tag}-initial.npz", state=initial)
        plans = f["plans"] if fresh else {"reverse": PLANS["reverse"]}
        if quick:
            plans = {"three": f["plans"]["three"]}
        for name, plan in plans.items():
            ff, _ = full_training(cfg, initial, pos, plan)
            row = {
                "run_id": tag + "-" + name,
                "preparation_id": tag,
                "position": pos,
                "plan": plan,
                "FF": ff,
                "assessments": {},
            }
            for kind in ["RF", "FR", "RR"]:
                result, _ = run(cfg, initial, pos, plan, kind, cb)
                row[kind] = result
                row["assessments"][kind] = evaluate(ff, result, plan)
            if fresh and f.get("composition_informed", False):
                original = ROOT / "frozen-01/model.npz"
                if (
                    hashlib.sha256(original.read_bytes()).hexdigest()
                    != load(ROOT / "frozen-01/freeze.json")["C_model_sha256"]
                ):
                    raise ValueError("Original C model changed")
                result, _ = run(
                    cfg, initial, pos, plan, "RR", np.load(original)["basis"]
                )
                row["RR_original"] = result
                row["assessments"]["RR_original"] = evaluate(ff, result, plan)
            records.append(row)
            write_json(out / "results.json", records)
            print(
                row["run_id"],
                {k: v["failures"] for k, v in row["assessments"].items()},
                flush=True,
            )


def refine(output):
    _, cb = frozen_c()
    _, cfg, _ = baseline()
    out = start_panel(
        output, {"stage": "matched composition timestep qualification, no retuning"}
    )
    row = load(ROOT / "composition-01/results.json")[0]
    initial = np.load(ROOT / "composition-01/s2011-c39-initial.npz")["state"]
    fine = replace(cfg, dt=cfg.dt / 2)
    ff, _ = full_training(fine, initial, 39, row["plan"])
    results = {"FF": ff}
    for kind in ["FR", "RR"]:
        results[kind], _ = run(fine, initial, 39, row["plan"], kind, cb)
    write_json(
        out / "results.json",
        {
            "config": asdict(fine),
            "runs": results,
            "differences": {
                k: refinement_difference(row[k], r) for k, r in results.items()
            },
        },
    )


def failure_refine(output):
    """Refine the exposed local-C failure before choosing an adaptation."""
    _, cb = frozen_c()
    _, cfg, _ = baseline()
    out = start_panel(
        output, {"stage": "failed C24 triple timestep diagnosis; no retuning"}
    )
    row = next(
        r
        for r in load(ROOT / "fresh-01/results.json")
        if r["run_id"] == "s4011-c41-three"
    )
    initial = np.load(ROOT / "fresh-01/s4011-c41-initial.npz")["state"]
    fine = replace(cfg, dt=cfg.dt / 2)
    ff, _ = full_training(fine, initial, 41, row["plan"])
    fr, _ = run(fine, initial, 41, row["plan"], "FR", cb)
    write_json(
        out / "results.json",
        {
            "FF": ff,
            "FR": fr,
            "assessment": evaluate(ff, fr, row["plan"]),
            "differences": {
                k: refinement_difference(row[k], r) for k, r in [("FF", ff), ("FR", fr)]
            },
        },
    )


def validate_refine(output):
    """Selected repaired model, fresh distant triple; no changes after outcomes."""
    freeze, cb = frozen_c()
    _, cfg, _ = baseline()
    out = start_panel(
        output,
        {"stage": "selected model numerical check only; no retuning", "freeze": freeze},
    )
    row = next(
        r
        for r in load(PANEL / "results.json")
        if r["run_id"] == f"s{freeze['fresh_seeds'][0]}-c41-three"
    )
    initial = np.load(PANEL / (row["preparation_id"] + "-initial.npz"))["state"]
    fine = replace(cfg, dt=cfg.dt / 2)
    ff, _ = full_training(fine, initial, 41, row["plan"])
    runs = {"FF": ff}
    for kind in ["FR", "RR"]:
        runs[kind], _ = run(fine, initial, 41, row["plan"], kind, cb)
    write_json(
        out / "results.json",
        {
            "runs": runs,
            "differences": {
                k: refinement_difference(row[k], r) for k, r in runs.items()
            },
            "assessments": {
                k: evaluate(ff, r, row["plan"]) for k, r in runs.items() if k != "FF"
            },
        },
    )


def repair_regression(output):
    """Previously exposed RR failures are regression evidence, never fresh."""
    freeze, cb = frozen_c()
    _, cfg, _ = baseline()
    out = start_panel(
        output,
        {
            "stage": "exposed original composition failures, selected-model regression",
            "freeze": freeze,
        },
    )
    records = []
    for row in load(ROOT / "fresh-01/results.json"):
        if not row["assessments"]["RR"]["failures"]:
            continue
        initial = np.load(ROOT / "fresh-01" / (row["preparation_id"] + "-initial.npz"))[
            "state"
        ]
        record = {
            "run_id": row["run_id"],
            "original": row["assessments"],
            "selected": {},
            "runs": {},
        }
        for kind in ["FR", "RR"]:
            result, _ = run(cfg, initial, row["position"], row["plan"], kind, cb)
            record["runs"][kind] = result
            record["selected"][kind] = evaluate(row["FF"], result, row["plan"])
        records.append(record)
        write_json(out / "results.json", records)


def incremental(output):
    """First-only continuation; no reset at input two and no new fitting."""
    freeze, cb = frozen_c()
    _, cfg, _ = baseline()
    out = start_panel(
        output,
        {
            "stage": "second-event increment versus first-only continuation",
            "windows": [[8, 80], [18, 80]],
            "gates": "inherited, sub-floor unresolved",
        },
    )
    repeated = load(PANEL / "results.json")[0]
    initial = np.load(PANEL / "s2011-c39-initial.npz")["state"]
    single = {"FF": load(ROOT / "c-reference/s2011-c39-single.json")}
    qualification = load(FROZEN / "protocol.json")["qualification"]
    single["FR"] = next(
        r["FR"]
        for r in load(Path(qualification) / "results.json")
        if r["preparation_id"] == "s2011-c39" and r["name"] == "single"
    )
    for kind in ["RF", "RR"]:
        single[kind], _ = run(cfg, initial, 39, PLANS["single"], kind, cb)
    t = np.array(repeated["FF"]["time"])
    scores = {}
    for kind in ["RF", "FR", "RR"]:
        scores[kind] = {}
        for start in [8, 18]:
            scores[kind][str(start)] = {}
            for name, key, column in [
                ("B_response", "response", 1),
                ("C_response", "response", 2),
                ("C_return", "return_contrast", 2),
            ]:
                truth = (
                    np.array(repeated["FF"][key])[:, column]
                    - np.array(single["FF"][key])[:, column]
                )
                prediction = (
                    np.array(repeated[kind][key])[:, column]
                    - np.array(single[kind][key])[:, column]
                )
                value = metric(
                    truth[t >= start],
                    prediction[t >= start],
                    get_scales()[name],
                    2e-7 if name == "C_return" else 0,
                )
                value["pass_if_resolved"] = (
                    None
                    if not value["resolved"]
                    else value["nrmse"] <= freeze["gates"][name + "_nrmse"]
                    and value["max_absolute"] <= freeze["gates"][name + "_max"]
                )
                scores[kind][str(start)][name] = value
    write_json(out / "results.json", {"single": single, "incremental_scores": scores})


def checkpoint(output):
    frozen, cb = frozen_c()
    _, cfg, ab = baseline()
    out = start_panel(
        output, {"stage": "RR restart in fresh process before second input"}
    )
    row = next(
        r
        for r in load(PANEL / "results.json")
        if r["run_id"] == f"s{frozen['fresh_seeds'][0]}-c39-three"
    )
    initial = np.load(PANEL / (row["preparation_id"] + "-initial.npz"))["state"]
    events = row["plan"]["events"]
    horizon = row["plan"]["horizon"]
    ticks = dict(event_ticks(events, cfg.dt))
    obj = DualHybrid.initialize(cfg, 39, ab, cb, initial)
    values = []
    st, sw = time.process_time(), time.perf_counter()
    for k in range(round(horizon / cfg.dt) + 1):
        if k in ticks:
            obj.pulse(39, ticks[k])
        if k == round(6 / cfg.dt):
            obj.save(out / "state.npz")
            write_json(
                out / "schedule.json",
                {"events": events, "tick": k, "horizon": horizon, "position": 39},
            )
        if k % round(0.5 / cfg.dt) == 0:
            values.append(readouts(obj.x, obj.fields(), 39))
        if k < round(horizon / cfg.dt):
            obj.step()
    charge(
        time.process_time() - st,
        time.perf_counter() - sw,
        "RR uninterrupted checkpoint",
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.mediated_patterns.dual_reduction",
            "resume",
            "--output",
            str(out / "resumed"),
            "--checkpoint",
            str(out),
        ],
        check=True,
    )
    d = np.load(out / "resumed/trace.npz")
    values = np.array(values)
    start = 12
    write_json(
        out / "results.json",
        {
            "resume_readout_max": float(abs(values[start:] - d["values"]).max()),
            "saved_primary_max": float(
                abs(values - np.array(row["RR"]["absolute"])[:, 1]).max()
            ),
            "FF_accuracy": row["assessments"]["RR"],
            "state_keys": np.load(out / "state.npz").files,
            "checkpoint_bytes": (out / "state.npz").stat().st_size,
        },
    )


def resume(output, checkpoint_path):
    out = Path(output)
    out.mkdir(exist_ok=False, parents=True)
    path = Path(checkpoint_path)
    obj = DualHybrid.load(path / "state.npz")
    s = load(path / "schedule.json")
    ticks = dict(event_ticks(s["events"], obj.c.dt))
    values = []
    # Disable construction/stepping of full truth after loading the checkpoint.
    from .simulator import Field

    def forbidden(*a, **kw):
        raise AssertionError("No truth or microstate access in restart")

    Field.__init__ = forbidden
    Field.step = forbidden
    st, sw = time.process_time(), time.perf_counter()
    for k in range(s["tick"], round(s["horizon"] / obj.c.dt) + 1):
        if k > s["tick"] and k in ticks:
            obj.pulse(s["position"], ticks[k])
        if k % round(0.5 / obj.c.dt) == 0:
            values.append(readouts(obj.x, obj.fields(), s["position"]))
        if k < round(s["horizon"] / obj.c.dt):
            obj.step()
    np.savez_compressed(out / "trace.npz", values=values)
    charge(
        time.process_time() - st, time.perf_counter() - sw, "fresh-process RR resume"
    )


def main():
    global FROZEN, PANEL
    p = argparse.ArgumentParser()
    p.add_argument(
        "mode",
        choices=[
            "calibrate",
            "adapt",
            "pilot",
            "qualify_c",
            "freeze_c",
            "compose",
            "fresh",
            "quick",
            "refine",
            "failure_refine",
            "validate_refine",
            "repair_regression",
            "checkpoint",
            "incremental",
            "resume",
        ],
    )
    p.add_argument("--output", required=True)
    p.add_argument("--order", type=int, default=8)
    p.add_argument("--qualification")
    p.add_argument("--checkpoint")
    p.add_argument("--freeze", default="frozen-01")
    p.add_argument("--panel", default="fresh-01")
    a = p.parse_args()
    FROZEN, PANEL = ROOT / a.freeze, ROOT / a.panel
    try:
        if a.mode == "adapt":
            adapt(a.output, a.order)
        elif a.mode == "qualify_c":
            qualify_c(a.output, a.order)
        elif a.mode == "freeze_c":
            freeze_c(a.output, a.qualification, a.order)
        elif a.mode in ["compose", "fresh", "quick"]:
            compose(a.output, a.mode != "compose", a.mode == "quick")
        elif a.mode == "resume":
            resume(a.output, a.checkpoint)
        else:
            globals()[a.mode](a.output)
    except (FloatingPointError, ValueError, RuntimeError) as e:
        if Path(a.output).exists():
            write_json(
                Path(a.output) / "failure.json",
                {"type": type(e).__name__, "error": str(e)},
            )
        raise


if __name__ == "__main__":
    main()
