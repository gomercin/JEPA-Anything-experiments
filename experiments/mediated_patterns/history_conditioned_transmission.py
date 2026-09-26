"""Bounded matched-history write/wait/probe experiment; no historical mutation."""

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.signal import resample

from .explore import write_json
from .organization_response import Budget, isolated_components
from .simulator import Field
from .source_receiver_relay import (
    CFG,
    A,
    C,
    descriptors,
    energy,
    prepare,
    read,
    simulate,
    source_profile,
)

ROOT = Path("work/mediated_patterns/history_conditioned_transmission")
REPORT = Path(__file__).with_name("HISTORY_CONDITIONED_TRANSMISSION.md")
B = -4.0
DEV = (8101,)
FRESH = (10101, 10102, 10103)
WAITS = (10.0, 50.0, 100.0)
WRITE = 0.4
EPS = 0.02


def load(p):
    return json.loads(Path(p).read_text())


def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


class HistoryBudget(Budget):
    def check(self):
        super().check()
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_bytes = rss if sys.platform == "darwin" else 1024 * rss
        if rss_bytes > 1024**3:
            raise RuntimeError("One GiB resident-memory stop")


def write_profile(field, kind):
    d = field.x - B
    q = np.zeros_like(d)
    inside = abs(d) < 8
    bump = np.exp(1 - 1 / (1 - (d[inside] / 8) ** 2))
    if kind not in ("even", "odd"):
        raise ValueError("Only two declared write families")
    q[inside] = bump * (np.cos(d[inside]) if kind == "even" else np.sin(d[inside]))
    return q / np.sqrt(np.sum(q * q) * field.c.length / field.c.n)


def event(field, state, profile, amplitude):
    result = state.copy()
    result[0] += amplitude * profile
    dx = field.c.length / field.c.n
    return result, {
        "amplitude": amplitude,
        "l2": float(np.sqrt(np.sum((result[0] - state[0]) ** 2) * dx)),
        "source_jump": float(
            field.c.source * np.sum(result[0] ** 2 - state[0] ** 2) * dx
        ),
        "energy_jump": energy(field, result) - energy(field, state),
    }


def grid_steps(duration, dt):
    n = round(duration / dt)
    if duration < 0 or not np.isclose(n * dt, duration, rtol=0, atol=1e-10):
        raise ValueError("Event/duration must lie exactly on the step grid")
    return n


def regime(field, states):
    d = [descriptors(field, s, B) for s in states]
    centers = np.array([x["center"] for x in d])
    masses = np.array([x["mass"] for x in d])
    widths = np.array([x["width"] for x in d])
    outside = np.min(abs(field.x[None] - np.array([A, B, C])[:, None]), axis=0) >= 10
    outside_peak = float(abs(states[:, 0, outside]).max())
    offset = abs(centers - np.array([A, B, C]))
    qualified = bool(
        (np.diff(centers, axis=-1) > 22).all()
        and (offset < np.array([0.5, 0.75, 0.5])).all()
        and ((masses > 4) & (masses < 12)).all()
        and ((widths > 2) & (widths < 4)).all()
        and outside_peak < 0.25
    )
    return d, {
        "qualified": qualified,
        "minimum_gap": float(np.diff(centers, axis=-1).min()),
        "max_offsets": offset.max(axis=0).tolist(),
        "outside_peak": outside_peak,
        "mass_range": [float(masses.min()), float(masses.max())],
        "width_range": [float(widths.min()), float(widths.max())],
    }


def change_descriptors(field, states):
    """Read-only local translation projection; never feeds back or recenters."""
    from .measurements import regions

    dx = field.c.length / field.c.n
    w = regions(field.x, [B])[0]
    u = states[0, 0]
    direction = -np.fft.irfft(1j * field.k * np.fft.rfft(u), n=field.c.n)
    delta = states[1:] - states[0]
    denom = np.sum(w * direction**2) * dx
    displacement = np.sum(w * direction * delta[:, 0], axis=-1) * dx / denom
    residual = delta[:, 0] - displacement[:, None] * direction
    return np.stack(
        [
            np.sqrt(np.sum(delta[:, 0] ** 2, axis=-1) * dx),
            np.sqrt(np.sum(delta[:, 1] ** 2, axis=-1) * dx),
            displacement,
            np.sqrt(np.sum(w * residual**2, axis=-1) * dx),
        ],
        axis=-1,
    )


def histories(config, initial, writes, waits, budget, end=180.0):
    """One complete ancestry, writes at t=0 only, no future event/recentering."""
    f = Field(config)
    starts, jumps = [initial.copy()], []
    for kind, amplitude in writes:
        state, jump = event(f, initial, write_profile(f, kind), amplitude)
        starts.append(state)
        jumps.append(dict(kind=kind, **jump))
    starts = np.asarray(starts)
    v = np.fft.rfft(starts)
    steps, stride = grid_steps(end, config.dt), grid_steps(1.0, config.dt)
    wanted = {grid_steps(t, config.dt): t for t in (*waits, end)}
    if max(wanted) > steps:
        raise ValueError("Wait beyond continuation")
    checkpoint, times, absolute, changes, geometry, regimes = {}, [], [], [], [], []
    for k in range(steps + 1):
        if k % stride == 0 or k in wanted:
            states = np.fft.irfft(v, n=config.n)
            if k in wanted:
                checkpoint[str(wanted[k])] = states.copy()
            times.append(k * config.dt)
            absolute.append(read(f, states, B))
            changes.append(change_descriptors(f, states))
            d, r = regime(f, states)
            geometry.append(d)
            regimes.append(r)
        if k == steps:
            break
        for j in range(len(v)):
            v[j], _ = f.step(v[j])
        if k % 100 == 0:
            budget.check()
    arrays = {
        "time": np.asarray(times),
        "absolute": np.asarray(absolute),
        "changes": np.asarray(changes),
        "initial": initial,
        "post_write": starts,
        "checkpoint_times": np.asarray(list(map(float, checkpoint))),
        "checkpoints": np.asarray(list(checkpoint.values())),
    }
    meta = {
        "writes": jumps,
        "config": asdict(config),
        "waits": list(waits),
        "end": end,
        "geometry": geometry,
        "regime": regimes,
        "qualified": all(x["qualified"] for x in regimes),
    }
    return arrays, meta


def save(out, name, arrays, meta):
    for suffix in (".npz", ".json"):
        p = out / (name + suffix)
        if p.exists() or p.is_symlink():
            raise FileExistsError(p)
    np.savez_compressed(out / (name + ".npz"), **arrays)
    write_json(out / (name + ".json"), meta)


def read_saved(root, name):
    with np.load(root / (name + ".npz"), allow_pickle=False) as z:
        a = dict(z)
    return a, load(root / (name + ".json"))


def checkpoint(arrays, wait):
    ix = np.flatnonzero(arrays["checkpoint_times"] == wait)
    if len(ix) != 1:
        raise ValueError("Exact wait checkpoint absent")
    return arrays["checkpoints"][ix[0]]


def history_contrasts(unwritten, written, key="full", input_index=0):
    """Each argument already uses its own history-specific unprobed baseline."""
    if not np.array_equal(unwritten["time"], written["time"]):
        raise ValueError("Matched response timelines differ")
    r0 = unwritten[key][:, input_index]
    rw = written[key][:, input_index]
    return r0, rw, rw - r0


def probe(out, name, config, state, amplitudes, budget, absolute_time):
    budget.begin(name)
    arrays, meta = simulate(config, state, B, amplitudes, budget, translated=True)
    arrays["absolute_time"] = arrays["time"] + absolute_time
    meta["probe_time"] = absolute_time
    # Check all saved centroids/masses without moving any window.
    absolute = arrays["absolute"]
    centers = np.array([A, B, C]) + 8 * absolute[..., 1] / absolute[..., 0]
    meta["all_sample_max_offsets"] = (
        abs(centers - np.array([A, B, C])).max(axis=(0, 1)).tolist()
    )
    meta["all_sample_min_gap"] = float(np.diff(centers, axis=-1).min())
    meta["all_sample_mass_range"] = [
        float(absolute[..., 0].min()),
        float(absolute[..., 0].max()),
    ]
    _, end_regime = regime(Field(config), arrays["final_states"])
    meta["endpoint_regime"] = end_regime
    meta["qualified"] = bool(
        end_regime["qualified"]
        and meta["all_sample_min_gap"] > 22
        and (
            np.array(meta["all_sample_max_offsets"]) < np.array([0.5, 0.75, 0.5])
        ).all()
        and meta["all_sample_mass_range"][0] > 4
        and meta["all_sample_mass_range"][1] < 12
    )
    save(out, name, arrays, meta)
    budget.finish()
    print(
        json.dumps(
            {
                "case": name,
                "cpu": budget.sections[-1]["cpu_seconds"],
                "qualified": meta["qualified"],
            }
        ),
        flush=True,
    )
    return arrays, meta


def validate_split(dev=DEV, fresh=FRESH):
    if set(dev) & set(fresh) or len(set(fresh)) < 3:
        raise ValueError("Need three disjoint fresh preparation seeds")


def start(out, stage, budget):
    if out.parent.resolve() != ROOT.resolve() or any(
        p.is_symlink() for p in (out, *out.parents)
    ):
        raise ValueError("Unique nonsymlink output child required")
    if subprocess.check_output(
        ["git", "diff", "HEAD", "--", "experiments/mediated_patterns"], text=True
    ):
        raise RuntimeError("Commit scientific sources and contract before execution")
    out.mkdir(parents=True, exist_ok=False)
    sources = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
    # Reject an untracked scientific module instead of assigning it a Git revision.
    for name in sources:
        subprocess.run(
            [
                "git",
                "ls-files",
                "--error-unmatch",
                "experiments/mediated_patterns/" + name,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
    write_json(
        out / "protocol.json",
        {
            "stage": stage,
            "command": sys.argv,
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_hashes": sources,
            "report_sha256": digest(REPORT),
            "config": asdict(CFG),
            "python": sys.version,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "development_seeds": DEV,
            "fresh_seeds": FRESH,
            "budget_limit": budget.limit,
        },
    )


def pilot(out, budget):
    f = Field(CFG)
    seed = DEV[0]
    parts = isolated_components(f, seed, budget)
    initial = prepare(f, parts, seed, B, budget)
    np.savez_compressed(out / "preparation.npz", components=parts, initial=initial)
    budget.begin("pilot-write-continuations")
    arrays, meta = histories(
        CFG, initial, [("even", WRITE), ("odd", WRITE)], WAITS, budget
    )
    save(out, "pilot", arrays, meta)
    budget.finish()
    write_json(
        out / "pilot-cost.json",
        {
            "cpu_so_far": time.process_time() - budget.start_cpu,
            "maximum_section_cpu": max(x["cpu_seconds"] for x in budget.sections),
            "fresh_reserve": 600,
        },
    )
    print(
        json.dumps(
            {
                "qualified": meta["qualified"],
                "late_changes": arrays["changes"][[10, 50, 100, 180]].tolist(),
                "cpu": time.process_time() - budget.start_cpu,
            }
        ),
        flush=True,
    )


def screen(out, data, budget):
    """Cheap nonlinear late-response check before building the pathway panel."""
    pilot_arrays, _ = read_saved(data, "pilot")
    states = checkpoint(pilot_arrays, 100.0)[[0, 2]]
    f = Field(CFG)
    q = source_profile(f)
    starts, jumps = [], []
    for s in states:
        stimulated, jump = event(f, s, q, EPS)
        starts.extend([s.copy(), stimulated])
        jumps.append(jump)
    v = np.fft.rfft(np.asarray(starts))
    absolute, times = [], []
    budget.begin("late-full-field-screen")
    for k in range(grid_steps(80.0, CFG.dt) + 1):
        if k % grid_steps(0.5, CFG.dt) == 0:
            absolute.append(read(f, np.fft.irfft(v, n=CFG.n), B))
            times.append(100 + k * CFG.dt)
        if k == grid_steps(80.0, CFG.dt):
            break
        for j in range(len(v)):
            v[j], _ = f.step(v[j])
        if k % 100 == 0:
            budget.check()
    a = np.asarray(absolute)
    r0, rw = a[:, 1] - a[:, 0], a[:, 3] - a[:, 2]
    arrays = {
        "absolute": a,
        "absolute_time": np.asarray(times),
        "unwritten": r0,
        "written": rw,
        "delta": rw - r0,
        "initial": np.asarray(starts),
        "final": np.fft.irfft(v, n=CFG.n),
    }
    save(
        out,
        "screen",
        arrays,
        {"jumps": jumps, "probe_time": 100.0, "config": asdict(CFG)},
    )
    budget.finish()
    print(
        json.dumps(
            {
                "C_delta_rms": np.sqrt(np.mean((rw - r0)[:, 2] ** 2, axis=0)).tolist(),
                "C_unwritten_rms": np.sqrt(np.mean(r0[:, 2] ** 2, axis=0)).tolist(),
            }
        ),
        flush=True,
    )


def panel(out, stage, data, frozen, budget):
    if stage == "pilot":
        return pilot(out, budget)
    if stage == "screen":
        return screen(out, data, budget)
    fz = load(frozen / "freeze.json") if frozen else None
    if stage == "fresh":
        for name, h in fz["source_hashes"].items():
            if digest(Path(__file__).with_name(name)) != h:
                raise ValueError("Frozen scientific source changed: " + name)
        if fz["fresh_seeds"] != list(FRESH):
            raise ValueError("Frozen preparation split changed")
        (out / "freeze-copy.json").write_bytes((frozen / "freeze.json").read_bytes())
    seeds = FRESH if stage == "fresh" else DEV
    for seed in seeds:
        configs = (
            [("", CFG)]
            if stage != "refine"
            else [
                ("half-dt", replace(CFG, dt=CFG.dt / 2)),
                ("double-n", replace(CFG, n=2 * CFG.n)),
            ]
        )
        if stage == "fresh":
            f = Field(CFG)
            parts = isolated_components(f, seed, budget)
            initial = prepare(f, parts, seed, B, budget)
            np.savez_compressed(
                out / f"s{seed}-preparation.npz", components=parts, initial=initial
            )
        else:
            initial = np.load(data / "preparation.npz", allow_pickle=False)["initial"]
        for suffix, config in configs:
            state = (
                initial if config.n == CFG.n else resample(initial, config.n, axis=-1)
            )
            prefix = f"s{seed}" + ("-" + suffix if suffix else "")
            budget.begin(prefix + "-history")
            history, meta = histories(config, state, [("odd", WRITE)], WAITS, budget)
            save(out, prefix + "-history", history, meta)
            budget.finish()
            if not meta["qualified"]:
                raise RuntimeError("Three-patch history regime failed; data preserved")
            for wait in WAITS:
                for j, s in enumerate(checkpoint(history, wait)):
                    name = f"{prefix}-t{wait:g}-" + ("written" if j else "unwritten")
                    if stage == "fresh":
                        probe(out, name + "-prediction", config, s, [], budget, wait)
                    amplitudes = [EPS, EPS / 2] if stage == "develop" else [EPS]
                    probe(out, name, config, s, amplitudes, budget, wait)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["pilot", "screen", "develop", "refine", "fresh"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--freeze", type=Path)
    args = p.parse_args()
    validate_split()
    if args.stage in ("screen", "develop", "refine") and args.data is None:
        p.error("--data required")
    if args.stage == "fresh" and args.freeze is None:
        p.error("--freeze required")
    ROOT.mkdir(parents=True, exist_ok=True)
    lock = ROOT / ".active"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    budget = HistoryBudget(
        sum(load(x)["cpu_seconds"] for x in ROOT.glob("*/budget.json")),
        1800 if args.stage == "fresh" else 1200,
    )
    started = False
    status = "FAILED"
    try:
        start(args.output, args.stage, budget)
        started = True
        budget.check()
        panel(args.output, args.stage, args.data, args.freeze, budget)
        status = "COMPLETE"
    finally:
        if started:
            write_json(
                args.output / "budget.json", dict(**budget.receipt(), status=status)
            )
        lock.unlink()


if __name__ == "__main__":
    main()
