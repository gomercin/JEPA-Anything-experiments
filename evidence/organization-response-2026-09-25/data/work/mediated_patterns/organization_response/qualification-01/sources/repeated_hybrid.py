"""Repeated C events through the frozen pair; reference/control runs are evaluators."""

import argparse
import hashlib
import json
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.signal import resample
from scipy.spatial import cKDTree

from .explore import start_panel, write_json
from .hybrid_pair import PairHybrid
from .hybrid_schedule import ScheduledHybrid, controlled_responses, event_ticks
from .measurements import describe
from .simulator import Config, Field
from .two_way_hybrid import prepare_triple, readouts, replay_weight

ROOT = Path("work/mediated_patterns/repeated_hybrid")
OLD = Path("work/mediated_patterns/two_way_hybrid")
DEVELOPMENT = {
    "overlap_same": {"events": [[0, 0.075], [8, 0.075]], "horizon": 80},
    "overlap_reverse": {"events": [[0, 0.075], [8, -0.075]], "horizon": 80},
    "relaxed_same": {"events": [[0, 0.075], [60, 0.075]], "horizon": 140},
}


def load(path):
    return json.loads(Path(path).read_text())


def baseline():
    frozen = load(OLD / "frozen-01/freeze.json")
    path = OLD / "frozen-01/model.npz"
    if hashlib.sha256(path.read_bytes()).hexdigest() != frozen["model_sha256"]:
        raise ValueError("Protected frozen model changed")
    return frozen, Config(**frozen["config"]), np.load(path)["basis"]


def run_sequence(config, initial, c_position, events, horizon, basis=None):
    """Four matched branches; primary branch never receives replay or truth.

    All branches share fixed timesteps, including omitted-event shams. Events
    are jumps before recording the corresponding time. No partition changes.
    """
    st, sw = time.process_time(), time.perf_counter()
    ticks = dict(event_ticks(events, config.dt, horizon))
    nstep, stride = round(horizon / config.dt), round(0.5 / config.dt)
    if (
        abs(nstep * config.dt - horizon) > 1e-10
        or abs(stride * config.dt - 0.5) > 1e-10
    ):
        raise ValueError("Horizon and sampling must match integration grid")
    if basis is None:
        solver = Field(config)
        x = solver.x
        # At t=0 preserve the original initial-field pulse semantics exactly.
        first = solver.pulse(
            initial, c_position, ticks.get(0, 0), width=8, relative=True, compact=True
        )
        states = [np.fft.rfft(v, axis=-1) for v in (initial, first, first, initial)]
    else:
        models = [PairHybrid.initialize(config, basis, initial) for _ in range(4)]
        x = models[0].x
        for j in (1, 2):
            if ticks.get(0, 0):
                models[j].pulse(c_position, ticks[0])
    setup_cpu = time.process_time() - st
    weight = replay_weight(x)
    times, values, shapes, ports, coordinates = [], [], [], [], []
    for k in range(nstep + 1):
        if k > 0 and k in ticks and ticks[k] != 0:
            for j in (1, 2):
                if basis is None:
                    state = np.fft.irfft(states[j], n=config.n, axis=-1)
                    state = solver.pulse(
                        state,
                        c_position,
                        ticks[k],
                        width=8,
                        relative=True,
                        compact=True,
                    )
                    states[j] = np.fft.rfft(state, axis=-1)
                else:
                    models[j].pulse(c_position, ticks[k])
        if k % stride == 0:
            fields = (
                [np.fft.irfft(v, n=config.n, axis=-1) for v in states]
                if basis is None
                else [m.fields() for m in models]
            )
            times.append(k * config.dt)
            values.append([readouts(x, v, c_position) for v in fields])
            # Measurements are output-only; neither solver accepts them.
            shapes.append(describe(x, fields[1], [-14, 14, c_position]))
            if basis is not None:
                p = models[1].ports()
                ports.append(
                    np.r_[p["incident_mediator_force"], p["incident_pattern_force"]]
                )
                coordinates.append(p["outgoing_coordinates"])
        if k == nstep:
            break
        if basis is None:
            states[0], stages = solver.step(states[0], record=True)
            states[1], _ = solver.step(states[1])
            states[2], _ = solver.step(states[2], replay=stages, replay_weight=weight)
            states[3], _ = solver.step(states[3], replay=stages, replay_weight=weight)
        else:
            stages = models[0].step(record=True)
            models[1].step()
            models[2].step(replay=stages, replay_weight=weight)
            models[3].step(replay=stages, replay_weight=weight)
    values = np.array(values)
    full, cut, contrast = controlled_responses(values)
    return {
        "time": times,
        "absolute": values.tolist(),
        "response": full.tolist(),
        "cut_response": cut.tolist(),
        "return_contrast": contrast.tolist(),
        "sham_replay_max": float(abs(values[:, 0] - values[:, 3]).max()),
        "geometry": {
            key: np.array([v[key] for v in shapes]).tolist() for key in shapes[0]
        },
        "ports": np.array(ports).tolist(),
        "coordinates": np.array(coordinates).tolist(),
        "setup_cpu_seconds": setup_cpu,
        "cpu_seconds": time.process_time() - st,
        "wall_seconds": time.perf_counter() - sw,
    }


def windows(events, horizon):
    result = {"whole": (0, horizon)}
    for j, (t, _) in enumerate(events):
        result[f"event{j + 1}_first20"] = (t, min(t + 20, horizon))
        result[f"event{j + 1}_early5"] = (t, min(t + 5, horizon))
    result["last_return"] = (events[-1][0] + 10, horizon)
    result["final_relaxation"] = (horizon - 20, horizon)
    return result


def metric(truth, prediction, scale, floor=0):
    truth, prediction = np.array(truth), np.array(prediction)
    signal = float(np.sqrt(np.mean(truth**2)))
    rmse = float(np.sqrt(np.mean((truth - prediction) ** 2)))
    return {
        "signal_rms": signal,
        "rmse": rmse,
        "max_absolute": float(abs(truth - prediction).max()),
        "nrmse": rmse / signal if signal > 0 else None,
        "fixed_single_scale_error": rmse / scale,
        "resolved": signal > floor,
    }


def assess(truth, hybrid, events, horizon, gates, scales):
    t = np.array(truth["time"])
    scores, failures, unresolved = {}, [], []
    for label, (lo, hi) in windows(events, horizon).items():
        mask = (t >= lo) & (t <= hi)
        scores[label] = {}
        for name, key, col in [
            ("B_response", "response", 1),
            ("C_response", "response", 2),
            ("C_return", "return_contrast", 2),
        ]:
            v = metric(
                np.array(truth[key])[mask, col],
                np.array(hybrid[key])[mask, col],
                scales[name],
                gates["return_signal_floor"] if name == "C_return" else 0,
            )
            scores[label][name] = v
            if v["max_absolute"] > gates[name + "_max"]:
                failures.append(label + ":" + name + ":max")
            if not v["resolved"]:
                unresolved.append(label + ":" + name)
            elif not label.endswith("early5") and v["nrmse"] > gates[name + "_nrmse"]:
                failures.append(label + ":" + name + ":nrmse")
    main_resolved = all(
        scores[w]["C_return"]["resolved"] for w in ["whole", "last_return"]
    )
    return {
        "windows": scores,
        "failures": failures,
        "unresolved": unresolved,
        "resolved_task_pass": not failures and main_resolved,
        "status": "failed"
        if failures
        else "pass_with_unresolved_windows"
        if main_resolved
        else "partial_unresolved_return",
    }


def calibration_coverage(basis, config):
    """Training-only ranges, rate scales and standardized joint NN reference."""
    idx = np.flatnonzero(
        (np.arange(config.n) * config.length / config.n - config.length / 2 >= -34)
        & (np.arange(config.n) * config.length / config.n - config.length / 2 < 28)
    )
    outside = np.setdiff1d(np.arange(config.n), idx)
    k = 2 * np.pi * np.fft.rfftfreq(config.n, config.length / config.n)
    linear = config.r - (1 - k * k) ** 2
    blocks = []
    for row in load(OLD / "calibration-01/results.json"):
        snapshots = np.load(OLD / "calibration-01" / f"{row['run_id']}-snapshots.npz")[
            "fields"
        ]
        for j in range(3):
            v = []
            for u, m in snapshots[:, j]:
                exterior = np.zeros(config.n)
                exterior[outside] = u[outside]
                tail = np.fft.irfft(linear * np.fft.rfft(exterior), n=config.n)[idx]
                v.append(
                    np.r_[config.feedback * basis.T @ (m[idx] * u[idx]), basis.T @ tail]
                )
            blocks.append(np.array(v))
    raw = np.concatenate(blocks)
    centered = np.concatenate([v - v[0] for v in blocks])
    scale = np.maximum(centered.std(axis=0), 1e-10)
    cloud = centered / scale
    tree = cKDTree(cloud)
    near = tree.query(cloud, k=2)[0][:, 1] / np.sqrt(cloud.shape[1])
    rates = np.concatenate([np.diff(v, axis=0) / 0.5 for v in blocks])
    return {
        "raw_min": raw.min(axis=0).tolist(),
        "raw_max": raw.max(axis=0).tolist(),
        "centered_scale": scale.tolist(),
        "cloud": cloud.tolist(),
        "rate_max": abs(rates).max(axis=0).tolist(),
        "training_nearest_nonself_max": float(near.max()),
        "contract": "per-trajectory centered 40-channel Euclidean NN after training-only std scaling; no error-bound interpretation",
    }


def coverage_scores(hybrid, coverage):
    p = np.array(hybrid["ports"])
    lo, hi = np.array(coverage["raw_min"]), np.array(coverage["raw_max"])
    exc = np.maximum(np.maximum(lo - p, p - hi), 0)
    mask = (exc > 1e-12).any(axis=1)
    joint = cKDTree(coverage["cloud"]).query((p - p[0]) / coverage["centered_scale"])[
        0
    ] / np.sqrt(p.shape[1])
    rates = abs(np.diff(p, axis=0)) / 0.5
    return {
        "max_marginal_span_excursion": float((exc / np.maximum(hi - lo, 1e-12)).max()),
        "outside_sample_fraction": float(mask.mean()),
        "outside_sample_duration": float(
            np.sum((mask[:-1].astype(float) + mask[1:]) * np.diff(hybrid["time"]) / 2)
        ),
        "max_standardized_joint_distance": float(joint.max()),
        "joint_distance": joint.tolist(),
        "max_rate_relative_to_training_max": float(
            (rates / np.maximum(coverage["rate_max"], 1e-12)).max()
        ),
        "coordinate_max_abs": float(abs(np.array(hybrid["coordinates"])).max()),
        "clipped": False,
    }


def geometry_scores(record):
    g = {k: np.array(v) for k, v in record["geometry"].items()}
    return {
        "center_drift_max": float(abs(g["center"] - g["center"][0]).max()),
        "AB_separation_drift_max": float(
            abs(
                np.diff(g["center"][:, :2], axis=1)[:, 0]
                - np.diff(g["center"][0, :2])[0]
            ).max()
        ),
        "width_change_max": float(abs(g["width"] - g["width"][0]).max()),
        "min_region_mass": g["mass"].min(axis=0).tolist(),
        "peak_max": g["peak"].max(axis=0).tolist(),
    }


def get_scales():
    old = load(OLD / "fresh-01/results.json")[0]["truth"]
    return {
        "B_response": float(np.sqrt(np.mean(np.array(old["response"])[:, 1] ** 2))),
        "C_response": float(np.sqrt(np.mean(np.array(old["response"])[:, 2] ** 2))),
        "C_return": float(
            np.sqrt(np.mean(np.array(old["return_contrast"])[:, 2] ** 2))
        ),
    }


def case(config, initial, position, plan, basis, gates, scales, coverage):
    truth = run_sequence(config, initial, position, **plan)
    hybrid = run_sequence(config, initial, position, **plan, basis=basis)
    return {
        "truth": truth,
        "hybrid": hybrid,
        "assessment": assess(truth, hybrid, **plan, gates=gates, scales=scales),
        "coverage": coverage_scores(hybrid, coverage),
        "geometry": geometry_scores(truth),
    }


def development(output):
    frozen, cfg, basis = baseline()
    out = start_panel(
        output,
        {
            "purpose": "unchanged twenty-mode repeated-input development",
            "config": asdict(cfg),
            "model_sha256": frozen["model_sha256"],
            "cases": DEVELOPMENT,
            "preparation": "saved s2011-c39",
            "timing_rationale": "B peaks at7, C return at23.5; gap8 overlaps B peak, gap60 has B0.85% and return20% of peaks",
            "gates": frozen["gates"],
            "window_rules": "whole, each event first20 and early5, last+10 onward, final20; same gates except early5 relative not certified",
            "budget_cpu_seconds": 2700,
            "first_order": "unchanged frozen model only",
        },
    )
    coverage = calibration_coverage(basis, cfg)
    write_json(out / "coverage.json", coverage)
    scales = get_scales()
    write_json(out / "scales.json", scales)
    initial = np.load(OLD / "fresh-01/s2011-c39-initial.npz")["state"]
    rows = []
    for name, plan in DEVELOPMENT.items():
        row = case(cfg, initial, 39, plan, basis, frozen["gates"], scales, coverage)
        row.update(run_id=name, preparation_id="s2011-c39", c_position=39, plan=plan)
        rows.append(row)
        write_json(out / "results.json", rows)
        print(
            name, row["assessment"]["status"], row["assessment"]["failures"], flush=True
        )
    # One first-only continuation at the same absolute times; no reset at event2.
    first_plan = {"events": [[0, 0.075], [8, 0]], "horizon": 80}
    first = case(cfg, initial, 39, first_plan, basis, frozen["gates"], scales, coverage)
    write_json(out / "first-only.json", first)
    incremental = {}
    for row in rows[:2]:
        t = np.array(row["truth"]["time"])
        incremental[row["run_id"]] = {}
        for key, col in [("response", 1), ("response", 2), ("return_contrast", 2)]:
            tr = (
                np.array(row["truth"][key])[:, col]
                - np.array(first["truth"][key])[:, col]
            )
            pr = (
                np.array(row["hybrid"][key])[:, col]
                - np.array(first["hybrid"][key])[:, col]
            )
            label = f"{key}_{col}"
            incremental[row["run_id"]][label] = metric(
                tr[t >= 8],
                pr[t >= 8],
                scales[
                    "C_return"
                    if key == "return_contrast"
                    else "B_response"
                    if col == 1
                    else "C_response"
                ],
                frozen["gates"]["return_signal_floor"]
                if key == "return_contrast"
                else 0,
            )
    write_json(out / "incremental.json", incremental)


def refinement_difference(base, fine):
    a, b = np.array(base["absolute"]), np.array(fine["absolute"])
    raw = a - b
    # Sum of absolute errors in the two separately baseline-subtracted effects.
    bound = abs(raw[:, 1] - raw[:, 0]) + abs(raw[:, 2] - raw[:, 3])
    return {
        "response_max_difference": abs(np.array(base["response"]) - fine["response"])
        .max(axis=0)
        .tolist(),
        "contrast_max_difference": abs(
            np.array(base["return_contrast"]) - fine["return_contrast"]
        )
        .max(axis=0)
        .tolist(),
        "conservative_paired_bound": bound.max(axis=0).tolist(),
        "bound_trace": bound.tolist(),
    }


def qualify(output):
    frozen, cfg, basis = baseline()
    out = start_panel(
        output,
        {
            "purpose": "matched repeated-event and extended-horizon qualification",
            "cases": ["overlap_reverse", "relaxed_same"],
            "refinements": [
                "dt/2 reference and hybrid both cases",
                "N*2 dt/2 reference reversal",
                "L256 same dx long reference",
            ],
            "gates_unchanged": frozen["gates"],
        },
    )
    rows = load(ROOT / "development-01/results.json")
    initial = np.load(OLD / "fresh-01/s2011-c39-initial.npz")["state"]
    records = []
    for name in ["overlap_reverse", "relaxed_same"]:
        row = next(r for r in rows if r["run_id"] == name)
        fine = replace(cfg, dt=cfg.dt / 2)
        tr = run_sequence(fine, initial, 39, **row["plan"])
        hy = run_sequence(fine, initial, 39, **row["plan"], basis=basis)
        records.append(
            {
                "run_id": name,
                "kind": "dt/2",
                "truth": tr,
                "hybrid": hy,
                "truth_difference": refinement_difference(row["truth"], tr),
                "hybrid_difference": refinement_difference(row["hybrid"], hy),
            }
        )
        write_json(out / "results.json", records)
        print(
            name,
            "dt/2",
            records[-1]["truth_difference"]["contrast_max_difference"][2],
            flush=True,
        )
    row = rows[1]
    fine = replace(cfg, n=cfg.n * 2, dt=cfg.dt / 2)
    tr = run_sequence(fine, resample(initial, fine.n, axis=-1), 39, **row["plan"])
    records.append(
        {
            "run_id": "overlap_reverse",
            "kind": "N*2 dt/2",
            "truth": tr,
            "truth_difference": refinement_difference(row["truth"], tr),
        }
    )
    write_json(out / "results.json", records)
    # Zero-background extension of the same interior initial profile. Store
    # the nonzero initial mediator boundary magnitude; this is a boundary/image
    # diagnostic, not an exactly identical periodic initial-value problem.
    wide = replace(cfg, length=256, n=1024)
    enlarged = np.zeros((2, wide.n))
    enlarged[:, 128:896] = initial
    tr = run_sequence(wide, enlarged, 39, **rows[2]["plan"])
    records.append(
        {
            "run_id": "relaxed_same",
            "kind": "L256",
            "truth": tr,
            "initial_boundary_max": float(abs(initial[:, [0, -1]]).max()),
            "truth_difference": refinement_difference(rows[2]["truth"], tr),
        }
    )
    write_json(out / "results.json", records)


def freeze(output):
    frozen, cfg, _ = baseline()
    plans = {
        "overlap_same": {"events": [[0, 0.08], [10, 0.065]], "horizon": 80},
        "overlap_reverse": {"events": [[0, -0.08], [10, 0.065]], "horizon": 80},
        "relaxed_reverse": {"events": [[0, 0.075], [55, -0.075]], "horizon": 135},
        "irregular_three": {
            "events": [[0, 0.075], [9, -0.065], [27, 0.08]],
            "horizon": 107,
        },
    }
    out = start_panel(
        output, {"purpose": "freeze unchanged repeated-use panel before new outcomes"}
    )
    coverage = load(ROOT / "development-01/coverage.json")
    write_json(out / "coverage.json", coverage)
    write_json(
        out / "freeze.json",
        {
            "status": "FROZEN_EXPLORATORY",
            "model_version": "original twenty-mode spatial pair unchanged",
            "model_path": str(OLD / "frozen-01/model.npz"),
            "model_sha256": frozen["model_sha256"],
            "config": asdict(cfg),
            "seeds": [3011, 3022, 3033],
            "positions": [39, 41],
            "age": "20 odd /60 even",
            "plans": plans,
            "gates": frozen["gates"],
            "scales": get_scales(),
            "windows": "whole, each event first20, early5 absolute-only, last event+10 onward, final20",
            "floor": "retain 2e-7 subject to matched refinement; unresolved windows are not relative passes",
            "preparation_training_ids": frozen["training_preparations"],
            "development_preparations": ["s2011-c39"],
            "not_independent": "schedules share one preparation; six new preparations across three seeds",
            "events": "known external finite table; original radius8 multiplicative u jump; no m jump, no continuous forcing",
            "no_updates": "no fitting, basis changes, true field refresh, recentering, or gate changes",
        },
    )


def fresh(output, quick=False):
    f = load(ROOT / "frozen-01/freeze.json")
    _, cfg, basis = baseline()
    out = start_panel(
        output,
        {
            "purpose": "quick reproduction"
            if quick
            else "fresh whole-preparation repeated-use evaluation",
            "freeze": f,
        },
    )
    coverage = load(ROOT / "frozen-01/coverage.json")
    rows = []
    for seed in f["seeds"][:1] if quick else f["seeds"]:
        for position in f["positions"][:1] if quick else f["positions"]:
            preparation_id = f"s{seed}-c{position}"
            if (
                preparation_id
                in f["preparation_training_ids"] + f["development_preparations"]
            ):
                raise ValueError("Preparation overlap")
            st, sw = time.process_time(), time.perf_counter()
            initial = prepare_triple(cfg, seed, position, age=20 if seed % 2 else 60)
            prep = {
                "cpu_seconds": time.process_time() - st,
                "wall_seconds": time.perf_counter() - sw,
            }
            np.savez_compressed(out / f"{preparation_id}-initial.npz", state=initial)
            write_json(out / f"{preparation_id}-preparation.json", prep)
            plans = (
                {"irregular_three": f["plans"]["irregular_three"]}
                if quick
                else f["plans"]
            )
            for name, plan in plans.items():
                row = case(
                    cfg,
                    initial,
                    position,
                    plan,
                    basis,
                    f["gates"],
                    f["scales"],
                    coverage,
                )
                row.update(
                    run_id=preparation_id + "-" + name,
                    preparation_id=preparation_id,
                    seed=seed,
                    c_position=position,
                    plan=plan,
                )
                rows.append(row)
                write_json(out / "results.json", rows)
                print(
                    row["run_id"],
                    row["assessment"]["status"],
                    row["assessment"]["failures"],
                    flush=True,
                )


def checkpoint(output):
    _, cfg, basis = baseline()
    out = start_panel(
        output,
        {
            "purpose": "restart while response active before later external event; no truth handle in prediction"
        },
    )
    row = next(
        r
        for r in load(ROOT / "fresh-01/results.json")
        if r["run_id"] == "s3011-c39-irregular_three"
    )
    initial = np.load(ROOT / "fresh-01/s3011-c39-initial.npz")["state"]
    st, sw = time.process_time(), time.perf_counter()
    run = ScheduledHybrid(
        PairHybrid.initialize(cfg, basis, initial), 39, row["plan"]["events"]
    )
    resume = None
    delta = 0.0
    values = []
    for k in range(round(row["plan"]["horizon"] / cfg.dt) + 1):
        if k == round(6 / cfg.dt):
            run.save(out / "checkpoint")
            resume = ScheduledHybrid.load(out / "checkpoint")
        if k % round(0.5 / cfg.dt) == 0:
            values.append(readouts(run.model.x, run.model.fields(), 39))
            if resume is not None:
                delta = max(
                    delta,
                    float(abs(run.model.q - resume.model.q).max()),
                    float(abs(run.model.m - resume.model.m).max()),
                )
        if k == round(row["plan"]["horizon"] / cfg.dt):
            break
        run.step()
        if resume is not None:
            resume.step()
    values = np.array(values)
    write_json(
        out / "results.json",
        {
            "checkpoint_time": 6,
            "next_event": 9,
            "max_resume_difference": delta,
            "stored_primary_readout_max_difference": float(
                abs(values - np.array(row["hybrid"]["absolute"])[:, 1]).max()
            ),
            "reference_accuracy": row["assessment"],
            "state_keys": np.load(out / "checkpoint/state.npz").files,
            "scheduler_keys": list(load(out / "checkpoint/schedule.json")),
            "cpu_seconds": time.process_time() - st,
            "wall_seconds": time.perf_counter() - sw,
        },
    )


def fresh_refine(output):
    """Post-freeze numerical check, not a model/gate repair."""
    _, cfg, basis = baseline()
    rows = load(ROOT / "fresh-01/results.json")
    row = max(
        (r for r in rows if r["run_id"].endswith("irregular_three")),
        key=lambda r: r["assessment"]["windows"]["whole"]["C_return"]["nrmse"],
    )
    out = start_panel(
        output,
        {
            "purpose": "qualify the worst whole-return fresh triple at half exchange timestep",
            "selected_case": row["run_id"],
            "model_gates_and_floor": "unchanged frozen-01",
            "selection": "maximum declared whole-return NRMSE among all fresh triples; diagnostic only",
        },
    )
    initial = np.load(ROOT / "fresh-01" / (row["preparation_id"] + "-initial.npz"))[
        "state"
    ]
    fine = replace(cfg, dt=cfg.dt / 2)
    truth = run_sequence(fine, initial, row["c_position"], **row["plan"])
    hybrid = run_sequence(fine, initial, row["c_position"], **row["plan"], basis=basis)
    write_json(
        out / "results.json",
        {
            "run_id": row["run_id"],
            "truth": truth,
            "hybrid": hybrid,
            "truth_difference": refinement_difference(row["truth"], truth),
            "hybrid_difference": refinement_difference(row["hybrid"], hybrid),
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode",
        choices=[
            "development",
            "qualify",
            "freeze",
            "fresh",
            "quick",
            "checkpoint",
            "fresh_refine",
        ],
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        if args.mode in ["fresh", "quick"]:
            fresh(args.output, args.mode == "quick")
        else:
            globals()[args.mode](args.output)
    except (FloatingPointError, ValueError, RuntimeError) as e:
        if Path(args.output).exists():
            write_json(
                Path(args.output) / "failure.json",
                {"type": type(e).__name__, "error": str(e)},
            )
        raise


if __name__ == "__main__":
    main()
