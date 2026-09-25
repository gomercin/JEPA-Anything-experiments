"""Bounded A-to-C relay diagnostic; full fields and own-sham ETDRK4 tangent."""

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.signal import resample

from .explore import write_json
from .measurements import describe, regions, shift
from .organization_response import (
    Budget,
    energy,
    evolve,
    isolated_components,
    rms,
    tangent_step,
)
from .simulator import Config, Field

ROOT = Path("work/mediated_patterns/source_receiver_relay")
CFG = Config(length=192, n=768, dt=0.00625)
A, C = -28.0, 28.0
POSITIONS = (0.0, -4.0)
DEV = (8101,)
FRESH = (9101, 9102, 9103)
EPS = 0.02
REPORT = Path(__file__).with_name("SOURCE_RECEIVER_RELAY.md")


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_profile(field):
    """New additive, unit-L2 amplitude direction at the fixed physical A anchor."""
    d = (field.x - A + field.c.length / 2) % field.c.length - field.c.length / 2
    q = np.zeros_like(d)
    inside = abs(d) < 8
    q[inside] = np.exp(1 - 1 / (1 - (d[inside] / 8) ** 2)) * np.cos(d[inside])
    return q / np.sqrt(np.sum(q * q) * field.c.length / field.c.n)


def cut_mask(field, b, translated=False):
    """Primary apparatus is the SAME union in both organizations; no A/C overlap."""
    centers = (b,) if translated else POSITIONS
    d = np.min(abs(field.x[None] - np.asarray(centers)[:, None]), axis=0)
    z = np.clip((d - 8) / 4, 0, 1)
    return 1 - (6 * z**5 - 15 * z**4 + 10 * z**3)


def weights(field, b):
    centers = (A, b, C)
    w = regions(field.x, centers)
    dx = field.c.length / field.c.n
    # M, signed moment P, and mean mediator; A/C never track true centers.
    return (
        w * dx,
        w * (field.x[None] - np.asarray(centers)[:, None]) / 8 * dx,
        w / w.sum(axis=1)[:, None],
    )


def read(field, state, b):
    w, p, m = weights(field, b)
    return np.stack(
        [
            np.einsum("jn,...n->...j", w, state[..., 0, :] ** 2),
            np.einsum("jn,...n->...j", p, state[..., 0, :] ** 2),
            np.einsum("jn,...n->...j", m, state[..., 1, :]),
        ],
        axis=-1,
    )


def read_tangent(field, base, z, b):
    w, p, m = weights(field, b)
    return np.stack(
        [
            2 * np.einsum("jn,...n->...j", w, base[0] * z[..., 0, :]),
            2 * np.einsum("jn,...n->...j", p, base[0] * z[..., 0, :]),
            np.einsum("jn,...n->...j", m, z[..., 1, :]),
        ],
        axis=-1,
    )


def descriptors(field, state, b):
    d = {k: v.tolist() for k, v in describe(field.x, state, (A, b, C)).items()}
    dx = field.c.length / field.c.n
    d.update(
        energy=energy(field, state),
        source_total=float(np.sum(state[0] ** 2) * dx * field.c.source),
        mediator_means=read(field, state, b)[:, 2].tolist(),
    )
    return d


def prepare(field, components, seed, b, budget):
    budget.begin(f"prepare-s{seed}-b{b:g}")
    centers = (A, b, C)
    joined = sum(shift(field, s, c) for s, c in zip(components, centers, strict=True))
    rng = np.random.default_rng(seed + 3000)
    for c in centers:
        d = field.x - c
        joined[0] += (
            0.02
            * np.exp(-((d / 5) ** 2))
            * sum(
                rng.normal() * np.cos(j * d / 2 + rng.uniform(0, 2 * np.pi))
                for j in range(1, 5)
            )
        )
    joined[1] += 0.02 * np.exp(-((field.x / 40) ** 2)) * np.cos(field.x / 7)
    initial = evolve(field, joined, 40, budget)
    budget.finish()
    return initial


def contrasts(absolute):
    full = absolute[:, 2::2] - absolute[:, 0, None]
    cut = absolute[:, 3::2] - absolute[:, 1, None]
    return full, cut, full - cut


def simulate(config, initial, b, amplitudes, budget, horizon=80.0, translated=False):
    f = Field(config)
    q, mask = source_profile(f), cut_mask(f, b, translated)
    v = np.fft.rfft(initial)
    cut_sham = v.copy()
    branches, jumps = [], []
    for amplitude in amplitudes:
        s = initial.copy()
        s[0] += amplitude * q
        dx = config.length / config.n
        jumps.append(
            {
                "amplitude": amplitude,
                "l2": float(np.linalg.norm(s[0] - initial[0]) * np.sqrt(dx)),
                "source_jump": float(
                    config.source * np.sum(s[0] ** 2 - initial[0] ** 2) * dx
                ),
                "energy_jump": energy(f, s) - energy(f, initial),
            }
        )
        branches += [np.fft.rfft(s), np.fft.rfft(s)]
    z0 = np.zeros((2, 2, config.n))
    z0[:, 0] = EPS * q
    z = np.fft.rfft(z0)
    tangent_weights = np.array([np.zeros_like(mask), mask])
    stride, steps = round(0.5 / config.dt), round(horizon / config.dt)
    if not np.isclose(steps * config.dt, horizon) or not np.isclose(
        stride * config.dt, 0.5
    ):
        raise ValueError("Time grid must include all samples")
    absolute, tangent, geometry, times, outgoing = [], [], [], [], []
    sham_error = 0.0
    # Fixed A exterior mediator sensor. This monitors late loop feedback too.
    exit_w = np.exp(-(((f.x - (A + 12)) / 2) ** 8))
    exit_w /= exit_w.sum()
    for k in range(steps + 1):
        if k % stride == 0:
            base = np.fft.irfft(v, n=config.n)
            states = np.array(
                [base, np.fft.irfft(cut_sham, n=config.n)]
                + [np.fft.irfft(s, n=config.n) for s in branches]
            )
            absolute.append(read(f, states, b))
            tangent.append(read_tangent(f, base, np.fft.irfft(z, n=config.n), b))
            outgoing.append(states[:, 1] @ exit_w)
            geometry.append(descriptors(f, base, b))
            times.append(k * config.dt)
            sham_error = max(sham_error, float(abs(states[0] - states[1]).max()))
        if k == steps:
            break
        v, z, stages = tangent_step(f, v, z, tangent_weights)
        cut_sham, _ = f.step(cut_sham, replay=stages, replay_weight=mask)
        for j, branch in enumerate(branches):
            branches[j], _ = f.step(
                branch,
                replay=stages if j % 2 else None,
                replay_weight=mask if j % 2 else None,
            )
        if k % 100 == 0:
            budget.check()
    absolute, tangent = np.asarray(absolute), np.asarray(tangent)
    full, cut, qb = contrasts(absolute)
    arrays = {
        "time": np.asarray(times),
        "absolute": absolute,
        "full": full,
        "cut": cut,
        "qb": qb,
        "tangent_full": tangent[:, 0],
        "tangent_cut": tangent[:, 1],
        "tangent_qb": tangent[:, 0] - tangent[:, 1],
        "outgoing": np.asarray(outgoing),
        "final_states": states,
        "mask": mask,
        "profile": q,
    }
    meta = {
        "b": b,
        "amplitudes": amplitudes,
        "jumps": jumps,
        "config": asdict(config),
        "translated_mask": translated,
        "sham_field_max": sham_error,
        "sham_geometry": geometry,
        "final_geometry": [descriptors(f, s, b) for s in states],
        "initial_sha256": hashlib.sha256(initial.tobytes()).hexdigest(),
    }
    return arrays, meta


def case(out, name, config, initial, seed, b, amplitudes, budget, **kwargs):
    if (out / f"{name}.npz").exists() or (out / f"{name}.json").exists():
        raise FileExistsError(name)
    budget.begin(name)
    arrays, meta = simulate(config, initial, b, amplitudes, budget, **kwargs)
    meta.update(seed=seed, preparation_id=f"s{seed}-b{b:g}")
    np.savez_compressed(out / f"{name}.npz", **arrays)
    write_json(out / f"{name}.json", meta)
    budget.finish()
    print(
        json.dumps(
            {
                "case": name,
                "cpu": budget.sections[-1]["cpu_seconds"],
                "c_tangent_rms": rms(arrays["tangent_full"][:, 2], axis=0).tolist(),
                "c_qb_rms": rms(arrays["tangent_qb"][:, 2], axis=0).tolist(),
            }
        ),
        flush=True,
    )


def validate_split():
    if set(DEV) & set(FRESH) or len(set(FRESH)) < 3:
        raise ValueError(
            "Three fresh whole preparations, disjoint from development, required"
        )


def start_output(out, stage):
    """Unique output only, including dangling symlinks and symlinked parents."""
    if out.parent.resolve() != ROOT.resolve() or any(
        p.is_symlink() for p in (out, *out.parents)
    ):
        raise ValueError("Use a direct, nonsymlink child of the fixed work root")
    if subprocess.check_output(
        ["git", "diff", "HEAD", "--", "experiments/mediated_patterns"], text=True
    ):
        raise RuntimeError(
            "Commit scientific source/report before outcome-bearing work"
        )
    out.mkdir(parents=True, exist_ok=False)
    write_json(
        out / "protocol.json",
        {
            "stage": stage,
            "command": sys.argv,
            "python": sys.version,
            "platform": platform.platform(),
            "revision": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "source_hashes": {
                p.name: digest(p) for p in Path(__file__).parent.glob("*.py")
            },
            "report_sha256": digest(REPORT),
            "config": asdict(CFG),
            "development_seeds": DEV,
            "fresh_seeds": FRESH,
        },
    )


def panel(out, stage, budget, data=None, frozen=None):
    f = Field(CFG)
    if stage in ("pilot", "fresh"):
        seeds = DEV if stage == "pilot" else FRESH
        if stage == "fresh":
            freeze = load(frozen / "freeze.json")
            if freeze["fresh_seeds"] != list(FRESH) or freeze[
                "development_seeds"
            ] != list(DEV):
                raise ValueError("Preparation split changed")
            (out / "freeze-copy.json").write_bytes(
                (frozen / "freeze.json").read_bytes()
            )
        for seed in seeds:
            components = isolated_components(f, seed, budget)
            np.savez_compressed(out / f"s{seed}-components.npz", components=components)
            for b in POSITIONS:
                initial = prepare(f, components, seed, b, budget)
                name = f"s{seed}-b{b:g}"
                np.savez_compressed(out / f"{name}-initial.npz", state=initial)
                # Pilot is calculation only; fresh prediction is saved BEFORE nonlinear references.
                case(out, name + "-prediction", CFG, initial, seed, b, [], budget)
                if stage == "fresh":
                    case(out, name, CFG, initial, seed, b, [EPS, 0.03], budget)
                elif b == POSITIONS[0]:
                    spent = time.process_time() - budget.start_cpu
                    write_json(
                        out / "pilot-cost.json",
                        {
                            "measured_cpu": spent,
                            "projected_cpu": 30 * spent,
                            "fresh_reserved_cpu": 600,
                        },
                    )
                    if 30 * spent > 1800:
                        raise RuntimeError("Pilot cost exceeds budget")
    elif stage in ("develop", "refine", "mask"):
        for b in POSITIONS:
            name = f"s{DEV[0]}-b{b:g}"
            initial = np.load(data / f"{name}-initial.npz", allow_pickle=False)["state"]
            if stage == "develop":
                case(
                    out,
                    name,
                    CFG,
                    initial,
                    DEV[0],
                    b,
                    [EPS, -EPS, EPS / 2, -EPS / 2],
                    budget,
                )
            elif stage == "mask":
                case(out, name, CFG, initial, DEV[0], b, [EPS], budget, translated=True)
            else:
                for label, config in [
                    ("half-dt", replace(CFG, dt=CFG.dt / 2)),
                    ("double-n", replace(CFG, n=CFG.n * 2)),
                ]:
                    state = (
                        initial
                        if config.n == CFG.n
                        else resample(initial, config.n, axis=-1)
                    )
                    case(
                        out, name + "-" + label, config, state, DEV[0], b, [EPS], budget
                    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["pilot", "develop", "refine", "mask", "fresh"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--freeze", type=Path)
    args = p.parse_args()
    validate_split()
    if args.stage in ("develop", "refine", "mask") and args.data is None:
        p.error("--data required")
    if args.stage == "fresh" and args.freeze is None:
        p.error("--freeze required")
    previous = sum(load(f)["cpu_seconds"] for f in ROOT.glob("*/budget.json"))
    budget = Budget(previous, 1800 if args.stage == "fresh" else 1200)
    start_output(args.output, args.stage)
    status = "FAILED"
    try:
        budget.check()
        panel(args.output, args.stage, budget, args.data, args.freeze)
        status = "COMPLETE"
    finally:
        write_json(args.output / "budget.json", dict(**budget.receipt(), status=status))


if __name__ == "__main__":
    main()
