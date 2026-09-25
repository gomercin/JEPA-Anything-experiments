"""Small sequential panels; new output directories only. CPU, no neural fitting."""

import argparse
import hashlib
import json
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from .measurements import describe, outward, prepare
from .simulator import Config, Field


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def start_panel(output, protocol):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    sources = output / "sources"
    sources.mkdir()
    for p in Path(__file__).parent.glob("*.py"):
        (sources / p.name).write_bytes(p.read_bytes())
    protocol = {
        "status": "EXPLORATORY",
        "command": sys.argv,
        "base_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "source_hashes": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources.glob("*.py")
        },
        **protocol,
    }
    write_json(output / "protocol.json", protocol)
    return output


def lists(d):
    return {k: v.tolist() for k, v in d.items()}


def response(f, state, centers, amplitude, horizon=50, replay=True, independent=None):
    """Pulse minus each condition's own sham. No recurrent reduced predictions."""
    times, base, stages = f.evolve(state, horizon, record=replay)
    pulse = f.pulse(state, centers[0], amplitude, width=8, relative=True, compact=True)
    _, full, _ = f.evolve(pulse, horizon)
    b = describe(f.x, base, centers)
    a = describe(f.x, full, centers)
    records = {
        "time": times.tolist(),
        "response": (a["mass"] - b["mass"]).tolist(),
        "initial_descriptors": lists(describe(f.x, state, centers)),
        "sham_descriptors": lists(b),
    }
    records["outward_response"] = (
        outward(f.x, full, centers[1]) - outward(f.x, base, centers[1])
    ).tolist()
    fields = {"x": f.x, "time": times, "base": base, "full": full}
    if replay:
        _, openloop, _ = f.evolve(pulse, horizon, replay=stages)
        _, sham_replay, _ = f.evolve(state, horizon, replay=stages)
        records["replay_sham_error"] = float(np.max(abs(sham_replay[:, 0] - base[:, 0])))
        rr = describe(f.x, openloop, centers)["mass"] - b["mass"]
        records["openloop_response"] = rr.tolist()
        fields["openloop"] = openloop
    if independent is not None:
        sa, sb = independent
        _, ya, _ = f.evolve(sa, horizon)
        _, yb, _ = f.evolve(sb, horizon)
        _, pa, _ = f.evolve(
            f.pulse(sa, centers[0], amplitude, width=8, relative=True, compact=True), horizon
        )
        # The homogeneous background is exactly u=m=0, counted once (not twice).
        delta = describe(f.x, pa + yb, centers)["mass"] - describe(f.x, ya + yb, centers)["mass"]
        records["independent_response"] = delta.tolist()
        records["independent_outward_response"] = (
            outward(f.x, pa + yb, centers[1]) - outward(f.x, ya + yb, centers[1])
        ).tolist()
    return records, fields


def qualification(output, quick=False):
    cfg = Config()
    configs = [
        cfg,
        replace(cfg, dt=cfg.dt / 2),
        replace(cfg, n=1024, dt=cfg.dt / 2),
        replace(cfg, n=1024, length=256, dt=cfg.dt / 2),
    ]
    out = start_panel(
        output,
        {
            "purpose": "persistence, feedback, and numerical qualification",
            "configs": [asdict(c) for c in configs],
            "separation": 28,
            "seed": 11,
            "preparation": 150,
            "joint_age": 100,
            "persistence": 150 if quick else 500,
            "pulse_relative_amplitude": 0.1,
            "pulse_support_radius": 8,
            "horizon": 50,
            "gates": {
                "max_response_refinement": 1e-6,
                "max_response_boundary": 1e-6,
                "max_relative_mass_drift": 0.05,
                "max_center_drift": 1.0,
            },
            "quick": quick,
        },
    )
    start = time.perf_counter()
    rows = []
    for i, c in enumerate(configs[:2] if quick else configs):
        f = Field(c)
        state, _ = prepare(f, 28, 11)
        row, fields = response(f, state, [-14, 14], 0.1)
        t, y, _ = f.evolve(state, 150 if quick else 500, sample=10)
        row.update(config=asdict(c), persistence=lists(describe(f.x, y, [-14, 14])))
        fields.update(long_time=t, long_fields=y)
        np.savez_compressed(out / f"case{i}.npz", **fields)
        rows.append(row)
        print("qualified case", i, flush=True)
    single = []
    for g in (0, 0.05):
        f = Field(replace(cfg, feedback=g))
        t, y, _ = f.evolve(f.seed(), 300, sample=5)
        desc = describe(f.x, y, [0])
        single.append(
            {
                "feedback": g,
                "time": t.tolist(),
                "descriptors": lists(desc),
                "late_field_change": float(np.max(abs(y[-1] - y[-11]))),
            }
        )
        np.savez_compressed(out / f"single-g{g}.npz", x=f.x, time=t, fields=y)
    diffs = [float(np.max(abs(np.array(r["response"]) - rows[0]["response"]))) for r in rows[1:]]
    persistence_ok = all(
        np.max(
            abs(np.array(r["persistence"]["mass"])[-1] / np.array(r["persistence"]["mass"])[0] - 1)
        )
        < 0.05
        and np.max(
            abs(np.array(r["persistence"]["center"])[-1] - np.array(r["persistence"]["center"])[0])
        )
        < 1
        for r in rows
    )
    result = {
        "records": rows,
        "single": single,
        "max_response_differences": diffs,
        "qualified": bool(max(diffs) < 1e-6 and persistence_ok),
        "seconds": time.perf_counter() - start,
    }
    write_json(out / "results.json", result)
    if not result["qualified"]:
        raise RuntimeError("Qualification failed; saved, not suppressed")
    if quick:
        from .analyze import quick_plot

        quick_plot(out)
    print("Qualification passed; maximum response differences:", diffs, flush=True)
    print("B response at t=20:", rows[0]["response"][20][1], flush=True)
    return result


def panel(output, qualification_path, fresh=False, model_path=None):
    qualified = json.loads((Path(qualification_path) / "results.json").read_text())
    if not qualified["qualified"]:
        raise ValueError("A passing qualification is required")
    from .reduced import ResponseModel

    cfg = Config()
    if fresh:
        model = ResponseModel.load(model_path)
        seeds, separations, amplitudes = [101, 202, 303], [25, 27, 29, 31], [0.075, -0.075]
    else:
        model = None
        seeds, separations, amplitudes = [11, 22], [24, 26, 28, 30, 32], [0.05, 0.1, -0.1]
    protocol = {
        "purpose": "fresh within-family evaluation" if fresh else "development response panel",
        "config": asdict(cfg),
        "seeds": seeds,
        "separations": separations,
        "amplitudes": amplitudes,
        "pulse": "relative u amplitude, C-infinity compact radius 8",
        "age": "20 for odd seed, 60 for even seed",
        "residual": 0.02,
        "readout": "C2 weighted u-squared in fixed B window; pulse minus own sham",
        "horizons": [5, 20, 50],
        "primary_horizon": 20,
        "split": "whole independently prepared pair; no adjacent-snapshot split",
        "frozen_tolerances": {"h20_max_absolute_error": 2e-5, "h20_nrmse": 0.15},
        "model_path": None if model_path is None else str(model_path),
        "model_sha256": None
        if model_path is None
        else hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
    }
    out = start_panel(output, protocol)
    start = time.perf_counter()
    rows = []
    f = Field(cfg)
    for seed in seeds:
        for d in separations:
            age = 20 if seed % 2 else 60
            state, isolated = prepare(f, d, seed, age=age, residual=0.02)
            independent = [f.evolve(s, age, sample=age)[1][-1] for s in isolated]
            centers = [-d / 2, d / 2]
            prep_id = f"seed{seed}-d{d}"
            np.savez_compressed(out / f"{prep_id}-initial.npz", state=state, x=f.x)
            for amp in amplitudes:
                row, fields = response(f, state, centers, amp, replay=True, independent=independent)
                row.update(
                    preparation_id=prep_id,
                    seed=seed,
                    separation=d,
                    amplitude=amp,
                    age=age,
                    split="fresh" if fresh else "development",
                )
                if model is not None:
                    row["compact_prediction"] = model.predict(row).tolist()
                rows.append(row)
                # One full trajectory per preparation; all response/readout traces in JSON.
                if amp == amplitudes[0]:
                    np.savez_compressed(out / f"{prep_id}-response.npz", **fields)
            print(prep_id, flush=True)
    result = {"records": rows, "seconds": time.perf_counter() - start}
    write_json(out / "results.json", result)
    if not fresh:
        fitted = ResponseModel.fit(rows)
        fitted.save(out / "response_model.json")
        write_json(out / "fit.json", fitted.score(rows))
    else:
        write_json(out / "score.json", model.score(rows))
    return result


def receiver_control(output):
    out = start_panel(
        output,
        {
            "purpose": "localize induced mediator feedback at receiver, preserving sender feedback",
            "config": asdict(Config()),
            "separation": 28,
            "seed": 11,
            "age": 100,
            "pulse_relative_amplitude": 0.1,
            "pulse_support_radius": 8,
            "horizon": 50,
            "control": (
                "replay sham mediator ONLY in fixed receiver region; "
                "C2 weight flat radius 8, zero beyond 12"
            ),
            "boundary": (
                "prescribed spatial pathway intervention only; primary law remains homogeneous"
            ),
        },
    )
    start = time.perf_counter()
    f = Field()
    state, _ = prepare(f, 28, 11)
    t, base, stages = f.evolve(state, 50, record=True)
    perturbed = f.pulse(state, -14, 0.1, width=8, relative=True, compact=True)
    _, full, _ = f.evolve(perturbed, 50)
    d = abs(f.x - 14)
    a = np.clip((d - 8) / 4, 0, 1)
    weight = 1 - (6 * a**5 - 15 * a**4 + 10 * a**3)
    _, control, _ = f.evolve(perturbed, 50, replay=stages, replay_weight=weight)
    _, sham, _ = f.evolve(state, 50, replay=stages, replay_weight=weight)
    records = {
        "full_response": (
            describe(f.x, full, [-14, 14])["mass"] - describe(f.x, base, [-14, 14])["mass"]
        ).tolist(),
        "receiver_control_response": (
            describe(f.x, control, [-14, 14])["mass"] - describe(f.x, sham, [-14, 14])["mass"]
        ).tolist(),
        "sham_max_error": float(np.max(abs(sham - base))),
        "seconds": time.perf_counter() - start,
    }
    np.savez_compressed(
        out / "fields.npz",
        x=f.x,
        time=t,
        base=base,
        full=full,
        control=control,
        sham=sham,
        replay_weight=weight,
    )
    write_json(out / "results.json", records)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--qualify", action="store_true")
    parser.add_argument("--panel", action="store_true")
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--receiver-control", action="store_true")
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preexisting = args.output.exists()
    try:
        if args.quick or args.qualify:
            qualification(args.output, quick=args.quick)
        elif args.panel or args.fresh:
            panel(args.output, args.qualification, fresh=args.fresh, model_path=args.model)
        elif args.receiver_control:
            receiver_control(args.output)
        else:
            parser.error("Select --quick, --qualify, --panel, or --fresh")
    except Exception as e:
        if not preexisting and args.output.is_dir() and not (args.output / "failure.json").exists():
            write_json(args.output / "failure.json", {"type": type(e).__name__, "message": str(e)})
        raise


if __name__ == "__main__":
    main()
