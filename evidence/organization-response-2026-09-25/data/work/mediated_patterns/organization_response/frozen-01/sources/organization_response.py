"""Bounded full-field response investigation; expensive stages are opt-in CLI.

The tangent differentiates the implemented ETDRK4 stages along the own sham.
No frozen reduced model, fitted physical law, or historical panel is modified.
"""

import argparse
import hashlib
import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
from scipy.signal import resample

from .explore import start_panel, write_json
from .measurements import describe, regions, shift
from .simulator import Config, Field

ROOT = Path("work/mediated_patterns/organization_response")
CFG = Config(length=192, n=768, dt=0.00625)
ORGANIZATIONS = (-10.0, -18.0)
DEV_SEEDS = (6101, 6102)
FRESH_SEEDS = (7101, 7102, 7103)
EPS = 0.02
HORIZON = 80.0
KINDS = ("full", "cut", "return")
RECORD = Path(__file__).with_name("ORGANIZATION_RESPONSE.md")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def rms(a, axis=None):
    return np.sqrt(np.mean(np.asarray(a) ** 2, axis=axis))


class Budget:
    """Charge every CLI process and poll stops inside simulation loops."""

    def __init__(self, previous=0.0, limit=1800.0):
        self.previous = previous
        self.limit = limit
        self.start_cpu = time.process_time()
        self.start_wall = time.perf_counter()
        self.sections = []
        self.begin("startup")

    def begin(self, label):
        self.section_label = label
        self.section_cpu = time.process_time()
        self.section_wall = time.perf_counter()

    def check(self):
        cpu, wall = time.process_time(), time.perf_counter()
        if self.previous + cpu - self.start_cpu >= self.limit:
            raise RuntimeError("Aggregate CPU budget exhausted")
        if cpu - self.section_cpu >= 120 or wall - self.section_wall >= 180:
            raise RuntimeError(f"Per-task resource stop: {self.section_label}")

    def finish(self):
        self.check()
        self.sections.append(
            {
                "task": self.section_label,
                "cpu_seconds": time.process_time() - self.section_cpu,
                "wall_seconds": time.perf_counter() - self.section_wall,
            }
        )

    def receipt(self):
        return {
            "cpu_seconds": time.process_time() - self.start_cpu,
            "wall_seconds": time.perf_counter() - self.start_wall,
            "previous_cpu_seconds": self.previous,
            "limit_cpu_seconds": self.limit,
            "sections": self.sections,
        }


def probes(field):
    """Fixed physical input vectors; no state or measured center is an argument."""
    d = (field.x - 39 + field.c.length / 2) % field.c.length - field.c.length / 2
    bump = np.zeros_like(d)
    inside = abs(d) < 8
    bump[inside] = np.exp(1 - 1 / (1 - (d[inside] / 8) ** 2))
    q = np.array([bump * np.cos(d), bump * np.sin(d)])
    q /= np.sqrt((q * q).sum(axis=1) * field.c.length / field.c.n)[:, None]
    return q


def cut_weight(x):
    # One physical apparatus for both organizations; no C support.
    d = np.min(abs(x[None, :] - np.array([-18, -10, 14])[:, None]), axis=0)
    z = np.clip((d - 8) / 4, 0, 1)
    return 1 - (6 * z**5 - 15 * z**4 + 10 * z**3)


def sensors(field):
    w = regions(field.x, [39])[0]
    return np.array([w, w * (field.x - 39) / 8]) * field.c.length / field.c.n


def read(field, state):
    return np.einsum("jn,...n->...j", sensors(field), state[..., 0, :] ** 2)


def read_tangent(field, base, tangent):
    return 2 * np.einsum("jn,...n->...j", sensors(field), base[0] * tangent[..., 0, :])


def energy(field, state):
    """Existing coupled Lyapunov functional, in its original physical units."""
    c = field.c
    u, m = state
    uxx = np.fft.irfft(-(field.k**2) * np.fft.rfft(u), n=c.n)
    mx = np.fft.irfft(1j * field.k * np.fft.rfft(m), n=c.n)
    density = (
        -c.r * u**2 / 2
        + (u + uxx) ** 2 / 2
        - c.cubic * u**4 / 4
        + u**6 / 6
        - c.feedback * m * u**2 / 2
        + c.feedback * (m**2 + c.diffusion * mx**2) / (4 * c.source)
    )
    return float(density.sum() * c.length / c.n)


def descriptors(field, state, a_position):
    d = {
        k: v.tolist() for k, v in describe(field.x, state, [a_position, 14, 39]).items()
    }
    dx = field.c.length / field.c.n
    w = regions(field.x, [a_position, 14, 39])
    d.update(
        energy=energy(field, state),
        source_total=float(field.c.source * (state[0] ** 2).sum() * dx),
        source_in_windows=(
            field.c.source * (w * state[0] ** 2).sum(axis=1) * dx
        ).tolist(),
        mediator_total=float(state[1].sum() * dx),
        mediator_means=((w * state[1]).sum(axis=1) / w.sum(axis=1)).tolist(),
    )
    return d


def pulse(state, profile, amplitude):
    result = state.copy()
    result[0] += amplitude * profile
    return result


def tangent_nonlinear(field, base_v, z, weights):
    """N'(u0,m0) z; weights suppress only induced mediator feedback."""
    u, m = np.fft.irfft(base_v, n=field.c.n, axis=-1)
    physical = np.fft.irfft(z, n=field.c.n, axis=-1)
    du, dm = physical[:, 0], physical[:, 1]
    c = field.c
    return np.fft.rfft(
        np.stack(
            [
                (3 * c.cubic * u**2 - 5 * u**4 + c.feedback * m) * du
                + c.feedback * u * dm * (1 - weights),
                2 * c.source * u * du / c.tau,
            ],
            axis=1,
        ),
        axis=-1,
    )


def tangent_step(field, v, z, weights):
    """Differentiate each original ETDRK4 stage; return sham m for the cuts."""
    nv = field.nonlinear(v)
    nz = tangent_nonlinear(field, v, z, weights)
    a, za = field.e2 * v + field.q * nv, field.e2 * z + field.q * nz
    na, nza = field.nonlinear(a), tangent_nonlinear(field, a, za, weights)
    b, zb = field.e2 * v + field.q * na, field.e2 * z + field.q * nza
    nb, nzb = field.nonlinear(b), tangent_nonlinear(field, b, zb, weights)
    c = field.e2 * a + field.q * (2 * nb - nv)
    zc = field.e2 * za + field.q * (2 * nzb - nz)
    nc, nzc = field.nonlinear(c), tangent_nonlinear(field, c, zc, weights)
    out = field.e * v + field.f1 * nv + 2 * field.f2 * (na + nb) + field.f3 * nc
    zout = field.e * z + field.f1 * nz + 2 * field.f2 * (nza + nzb) + field.f3 * nzc
    if not np.isfinite(out).all() or not np.isfinite(zout).all():
        raise FloatingPointError("Nonfinite tangent/reference; no clipping")
    stages = np.array([np.fft.irfft(s[1], n=field.c.n) for s in (v, a, b, c)])
    return out, zout, stages


def evolve(field, initial, duration, budget):
    v = np.fft.rfft(initial, axis=-1)
    for k in range(round(duration / field.c.dt)):
        v, _ = field.step(v)
        if k % 100 == 0:
            budget.check()
    return np.fft.irfft(v, n=field.c.n, axis=-1)


def isolated_components(field, seed, budget):
    budget.begin(f"isolated-components-s{seed}")
    states = [
        evolve(field, field.seed(seed=seed + 100 * j, variation=0.01), 150, budget)
        for j in range(3)
    ]
    budget.finish()
    return states


def prepare(field, components, seed, a_position, budget):
    """Same recipe as prepare_triple, with only A's translation generalized."""
    budget.begin(f"joint-preparation-s{seed}-a{a_position:g}")
    centers = (a_position, 14.0, 39.0)
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


def simulate(config, initial, a_position, inputs, budget, horizon=HORIZON):
    """inputs=(name, q-even coefficient, q-odd coefficient, amplitude).

    Full and cut input trajectories never receive tangent predictions. Their
    own sham is advanced independently under the same recorded-stage replay.
    """
    field = Field(config)
    q = probes(field)
    mask = cut_weight(field.x)
    v = np.fft.rfft(initial, axis=-1)
    cut_sham = v.copy()
    stimulated, jumps = [], []
    for name, even, odd, amplitude in inputs:
        profile = even * q[0] + odd * q[1]
        p = pulse(initial, profile, amplitude)
        dx = config.length / config.n
        jumps.append(
            {
                "name": name,
                "coefficients": [even, odd],
                "amplitude": amplitude,
                "l2": float(np.sqrt(((p[0] - initial[0]) ** 2).sum() * dx)),
                "max_absolute": float(abs(p[0] - initial[0]).max()),
                "energy_jump": energy(field, p) - energy(field, initial),
                "source_jump": float(
                    config.source * ((p[0] ** 2 - initial[0] ** 2).sum()) * dx
                ),
            }
        )
        pv = np.fft.rfft(p, axis=-1)
        stimulated.append([pv.copy(), pv.copy()])
    z0 = np.zeros((4, 2, config.n))
    z0[:, 0] = EPS * q[[0, 1, 0, 1]]
    z = np.fft.rfft(z0, axis=-1)
    weights = np.array([np.zeros_like(mask), np.zeros_like(mask), mask, mask])
    absolute, tangent, times, geometry = [], [], [], []
    sham_error = 0.0
    stride, steps = round(0.5 / config.dt), round(horizon / config.dt)
    if (
        abs(steps * config.dt - horizon) > 1e-12
        or abs(stride * config.dt - 0.5) > 1e-12
    ):
        raise ValueError("Horizon and sampling must lie on the time grid")
    start = time.process_time()
    for k in range(steps + 1):
        if k % stride == 0:
            base = np.fft.irfft(v, n=config.n, axis=-1)
            sham = np.fft.irfft(cut_sham, n=config.n, axis=-1)
            branches = [read(field, base), read(field, sham)]
            for full, cut in stimulated:
                branches += [
                    read(field, np.fft.irfft(full, n=config.n, axis=-1)),
                    read(field, np.fft.irfft(cut, n=config.n, axis=-1)),
                ]
            absolute.append(branches)
            tangent.append(
                read_tangent(field, base, np.fft.irfft(z, n=config.n, axis=-1))
            )
            times.append(k * config.dt)
            sham_error = max(sham_error, float(abs(base - sham).max()))
            geometry.append(descriptors(field, base, a_position))
        if k == steps:
            break
        v, z, stages = tangent_step(field, v, z, weights)
        cut_sham, _ = field.step(cut_sham, replay=stages, replay_weight=mask)
        for branch in stimulated:
            branch[0], _ = field.step(branch[0])
            branch[1], _ = field.step(branch[1], replay=stages, replay_weight=mask)
        if k % 100 == 0:
            budget.check()
    a = np.array(absolute)
    full = a[:, 2::2] - a[:, 0, None]
    cut = a[:, 3::2] - a[:, 1, None]
    t = np.array(tangent)
    final = [
        descriptors(field, np.fft.irfft(s[0], n=config.n, axis=-1), a_position)
        for s in stimulated
    ]
    return {
        "time": times,
        "inputs": jumps,
        "absolute": a.tolist(),
        "full": full.tolist(),
        "cut": cut.tolist(),
        "return": (full - cut).tolist(),
        "tangent_full": t[:, :2].tolist(),
        "tangent_cut": t[:, 2:].tolist(),
        "tangent_return": (t[:, :2] - t[:, 2:]).tolist(),
        "sham_field_max": sham_error,
        "sham_geometry": geometry,
        "final_stimulated_geometry": final,
        "sham_field_rms_drift": float(rms(base - initial)),
        "cpu_seconds": time.process_time() - start,
        "config": asdict(config),
        "a_position": a_position,
    }


def standard_inputs():
    return [("even", 1, 0, EPS), ("odd", 0, 1, EPS)]


def execute_case(out, name, config, initial, seed, a_position, inputs, budget):
    path = out / f"{name}.json"
    if path.exists():
        raise FileExistsError(path)
    budget.begin(name)
    result = simulate(config, initial, a_position, inputs, budget)
    result["preparation_id"] = f"s{seed}-a{a_position:g}"
    result["seed"] = seed
    result["initial_sha256"] = hashlib.sha256(initial.tobytes()).hexdigest()
    write_json(path, result)
    budget.finish()
    return result


def development(out, budget):
    field = Field(CFG)
    for seed in DEV_SEEDS:
        components = isolated_components(field, seed, budget)
        for a_position in ORGANIZATIONS:
            initial = prepare(field, components, seed, a_position, budget)
            name = f"s{seed}-a{a_position:g}"
            np.savez_compressed(out / f"{name}-initial.npz", state=initial, x=field.x)
            inputs = standard_inputs()
            if seed == DEV_SEEDS[0]:
                inputs += [("even-negative", 1, 0, -EPS), ("odd-negative", 0, 1, -EPS)]
                inputs += [("even-half", 1, 0, EPS / 2), ("odd-half", 0, 1, EPS / 2)]
                inputs += [
                    ("even-negative-half", 1, 0, -EPS / 2),
                    ("odd-negative-half", 0, 1, -EPS / 2),
                ]
            result = execute_case(
                out, name, CFG, initial, seed, a_position, inputs, budget
            )
            print(
                json.dumps(
                    {
                        "case": name,
                        "response_cpu": result["cpu_seconds"],
                        "sham_error": result["sham_field_max"],
                    }
                ),
                flush=True,
            )
            if seed == DEV_SEEDS[0] and a_position == ORGANIZATIONS[0]:
                # Measured pilot extrapolation, deliberately conservative.
                spent = time.process_time() - budget.start_cpu
                projected = 15 * spent
                write_json(
                    out / "pilot-cost.json",
                    {
                        "measured_cpu": spent,
                        "projected_total_cpu": projected,
                        "fresh_reserved_cpu": 600,
                    },
                )
                if projected > 1800:
                    raise RuntimeError("Pilot projected cost exceeds 30 CPU-minutes")


def qualification(out, data, budget):
    for a_position in ORGANIZATIONS:
        name = f"s{DEV_SEEDS[0]}-a{a_position:g}"
        initial = np.load(data / f"{name}-initial.npz")["state"]
        for label, config in [
            ("half-dt", replace(CFG, dt=CFG.dt / 2)),
            ("double-n", replace(CFG, n=CFG.n * 2)),
        ]:
            state = (
                initial if config.n == CFG.n else resample(initial, config.n, axis=-1)
            )
            execute_case(
                out,
                f"{name}-{label}",
                config,
                state,
                DEV_SEEDS[0],
                a_position,
                standard_inputs(),
                budget,
            )
            print(f"Completed refinement {name}-{label}", flush=True)


def metric(truth, prediction, scales, floors):
    truth, prediction = np.asarray(truth), np.asarray(prediction)
    err = prediction - truth
    axes = tuple(range(err.ndim - 1))
    error = rms(err, axis=axes)
    signal = rms(truth, axis=axes)
    normalized = float(rms(err / scales) / max(float(rms(truth / scales)), 1e-15))
    return {
        "weighted_nrmse": normalized,
        "rmse": error.tolist(),
        "max_absolute": abs(err).max(axis=axes).tolist(),
        "signal_rms": signal.tolist(),
        "resolved": (signal > floors).tolist(),
        "practical_failure": bool(
            normalized > 0.05 and (error > 5 * np.asarray(floors)).any()
        ),
    }


def shifted(array, times, delay):
    array = np.asarray(array)
    flat = array.reshape((len(times), -1))
    return np.stack(
        [
            np.interp(np.asarray(times) - delay, times, v, left=0, right=v[-1])
            for v in flat.T
        ],
        axis=-1,
    ).reshape(array.shape)


def gain_delay(pairs, times, scales, delays):
    """One signed gain and one common delay across preparations/channels."""
    best = None
    y = np.stack([b for _, b in pairs]) / scales
    for delay in delays:
        x = np.stack([shifted(a, times, delay) for a, _ in pairs]) / scales
        denom = float(np.sum(x * x))
        gain = float(np.sum(x * y) / denom) if denom else 0.0
        objective = float(np.sum((gain * x - y) ** 2))
        if best is None or objective < best["objective"]:
            best = {"gain": gain, "delay": float(delay), "objective": objective}
    return best


def freeze(out, data, refinement, budget):
    budget.begin("freeze-and-fit")
    dev = [load(data / f"s{s}-a{a:g}.json") for s in DEV_SEEDS for a in ORGANIZATIONS]
    floors = np.full(2, 1e-10)
    refinement_rows = []
    for a in ORGANIZATIONS:
        base = next(
            r for r in dev if r["seed"] == DEV_SEEDS[0] and r["a_position"] == a
        )
        for label in ("half-dt", "double-n"):
            fine = load(refinement / f"s{DEV_SEEDS[0]}-a{a:g}-{label}.json")
            f = np.asarray(fine["full"]) - np.asarray(base["full"])[:, :2]
            c = np.asarray(fine["cut"]) - np.asarray(base["cut"])[:, :2]
            conservative = (rms(f, axis=0) + rms(c, axis=0)).max(axis=0)
            floors = np.maximum(floors, 5 * conservative)
            refinement_rows.append(
                {
                    "organization": a,
                    "refinement": label,
                    "full_rmse": rms(f, axis=(0, 1)).tolist(),
                    "conservative_return_bound": conservative.tolist(),
                    "direct_return_rmse": rms(f - c, axis=(0, 1)).tolist(),
                }
            )
    models, scores, checks = {}, {}, []
    candidate = False
    for kind in KINDS:
        stack = np.stack([np.asarray(d[kind])[:, :2] for d in dev])
        scales = np.maximum(rms(stack, axis=(0, 1, 2)), floors)
        pairs = [
            (np.asarray(dev[i][kind])[:, :2], np.asarray(dev[i + 1][kind])[:, :2])
            for i in (0, 2)
        ]
        fit = gain_delay(pairs, dev[0]["time"], scales, np.linspace(-8, 8, 641))
        gain = gain_delay(pairs, dev[0]["time"], scales, [0])
        models[kind] = {"scales": scales.tolist(), "gain_delay": fit, "gain_only": gain}
        prediction = np.stack(
            [fit["gain"] * shifted(a, dev[0]["time"], fit["delay"]) for a, _ in pairs]
        )
        truth = np.stack([b for _, b in pairs])
        scores[kind] = metric(truth, prediction, scales, floors)
        candidate |= kind in ("full", "return") and scores[kind]["practical_failure"]
        for d in dev:
            for j, info in enumerate(d["inputs"]):
                coeff = np.asarray(info["coefficients"]) * info["amplitude"] / EPS
                pred = np.einsum("tpj,p->tj", np.asarray(d[f"tangent_{kind}"]), coeff)
                score = metric(np.asarray(d[kind])[:, j], pred, scales, floors)
                checks.append(
                    {
                        "preparation": d["preparation_id"],
                        "probe": info["name"],
                        "kind": kind,
                        **score,
                    }
                )
    # Total-response linearization is a prerequisite, not a selectivity gate.
    invalid = [
        r
        for r in checks
        if r["kind"] == "full"
        and r["weighted_nrmse"] > 0.02
        and np.any(np.asarray(r["rmse"]) > floors)
    ]
    if invalid:
        write_json(out / "linearization-failure.json", invalid)
        raise RuntimeError(
            "Declared small-signal tangent check failed; preserve and diagnose"
        )
    frozen = {
        "development_seeds": list(DEV_SEEDS),
        "fresh_seeds": list(FRESH_SEEDS),
        "organizations": list(ORGANIZATIONS),
        "config": asdict(CFG),
        "epsilon": EPS,
        "horizon": HORIZON,
        "floors": floors.tolist(),
        "models": models,
        "development_scores": scores,
        "refinement": refinement_rows,
        "tangent_checks": checks,
        "candidate_selectivity": bool(candidate),
        "fresh_mixture": bool(candidate),
        "criteria": {
            "selectivity_nrmse": 0.05,
            "absolute_floor_multiple": 5,
            "total_prediction_nrmse": 0.02,
            "return_prediction_nrmse": 0.10,
        },
        "source_hashes": {
            p.name: digest(p)
            for p in [
                Path(__file__),
                Path(__file__).with_name("simulator.py"),
                Path(__file__).with_name("measurements.py"),
            ]
        },
        "contract_sha256": digest(RECORD),
        "development_files": {p.name: digest(p) for p in data.glob("*.json")},
        "refinement_files": {p.name: digest(p) for p in refinement.glob("*.json")},
        "prediction_access": "own unperturbed trajectory is diagnostic input; no autonomous reduction claim",
    }
    write_json(out / "freeze.json", frozen)
    budget.finish()
    print(
        json.dumps(
            {
                "floors": floors.tolist(),
                "scores": scores,
                "fresh_mixture": bool(candidate),
            }
        ),
        flush=True,
    )


def validate_split(frozen):
    if set(frozen["development_seeds"]) & set(frozen["fresh_seeds"]):
        raise ValueError("Preparations cross the development/fresh boundary")
    if len(set(frozen["fresh_seeds"])) < 3:
        raise ValueError("At least three fresh seeds are required")


def fresh(out, frozen_path, budget):
    frozen = load(frozen_path / "freeze.json")
    validate_split(frozen)
    for name, expected in frozen["source_hashes"].items():
        if digest(Path(__file__).with_name(name)) != expected:
            raise ValueError(f"Frozen source identity changed: {name}")
    write_json(out / "freeze-copy.json", frozen)
    for seed in frozen["fresh_seeds"]:
        field = Field(Config(**frozen["config"]))
        components = isolated_components(field, seed, budget)
        for a in frozen["organizations"]:
            initial = prepare(field, components, seed, a, budget)
            name = f"s{seed}-a{a:g}"
            np.savez_compressed(out / f"{name}-initial.npz", state=initial, x=field.x)
            # Save basis-probe tangent prediction before any mixture run.
            result = execute_case(
                out, name, field.c, initial, seed, a, standard_inputs(), budget
            )
            if frozen["fresh_mixture"]:
                prediction = {
                    k: (
                        np.asarray(result[f"tangent_{k}"]).sum(axis=1) / np.sqrt(2)
                    ).tolist()
                    for k in KINDS
                }
                write_json(
                    out / f"{name}-mixture-prediction.json",
                    {
                        "prediction": prediction,
                        "basis_result_sha256": digest(out / f"{name}.json"),
                        "freeze_sha256": digest(frozen_path / "freeze.json"),
                    },
                )
                inputs = [("mixture", 1 / np.sqrt(2), 1 / np.sqrt(2), EPS)]
                execute_case(
                    out, f"{name}-mixture", field.c, initial, seed, a, inputs, budget
                )
            print(f"Completed fresh preparation {name}", flush=True)


def analyze(out, data, frozen_path, budget):
    budget.begin("analysis")
    frozen = load(frozen_path / "freeze.json")
    floors = np.asarray(frozen["floors"])
    rows, predictions = [], []
    for seed in frozen["fresh_seeds"]:
        cases = [load(data / f"s{seed}-a{a:g}.json") for a in ORGANIZATIONS]
        for kind in KINDS:
            model = frozen["models"][kind]
            scales = np.asarray(model["scales"])
            source, target = [np.asarray(d[kind]) for d in cases]
            fit = model["gain_delay"]
            pred = fit["gain"] * shifted(source, cases[0]["time"], fit["delay"])
            row = {"seed": seed, "kind": kind, **metric(target, pred, scales, floors)}
            row["identity"] = metric(target, source, scales, floors)
            row["gain_only"] = metric(
                target, model["gain_only"]["gain"] * source, scales, floors
            )
            rows.append(row)
            for case in cases:
                predictions.append(
                    {
                        "seed": seed,
                        "organization": case["a_position"],
                        "kind": kind,
                        "prediction": "tangent-basis",
                        **metric(case[kind], case[f"tangent_{kind}"], scales, floors),
                    }
                )
                if frozen["fresh_mixture"]:
                    name = f"s{seed}-a{case['a_position']:g}"
                    truth = np.asarray(load(data / f"{name}-mixture.json")[kind])[:, 0]
                    prediction = load(data / f"{name}-mixture-prediction.json")[
                        "prediction"
                    ][kind]
                    predictions.append(
                        {
                            "seed": seed,
                            "organization": case["a_position"],
                            "kind": kind,
                            "prediction": "withheld-mixture",
                            **metric(truth, prediction, scales, floors),
                        }
                    )
    summary = {
        "fresh_comparator": rows,
        "fresh_predictions": predictions,
        "floors": floors.tolist(),
        "freeze_sha256": digest(frozen_path / "freeze.json"),
        "data": str(data),
    }
    write_json(out / "summary.json", summary)
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/organization-response-mpl")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Predesignated first fresh seed, all readouts/probes, no selected crossing.
    cases = [load(data / f"s{FRESH_SEEDS[0]}-a{a:g}.json") for a in ORGANIZATIONS]
    times = cases[0]["time"]
    for kind in ("full", "return"):
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
        scales = np.asarray(frozen["models"][kind]["scales"])
        fit = frozen["models"][kind]["gain_delay"]
        prediction = fit["gain"] * shifted(cases[0][kind], times, fit["delay"])
        for j, ax in enumerate(axes.flat):
            probe, sensor = divmod(j, 2)
            for case, color in zip(cases, ("#16697a", "#ce7e00"), strict=True):
                ax.plot(
                    times,
                    np.asarray(case[kind])[:, probe, sensor],
                    color=color,
                    label=f"AB gap {14 - case['a_position']:g}",
                )
            ax.plot(
                times,
                prediction[:, probe, sensor],
                "--",
                color="#8648ac",
                label="Frozen common gain/delay",
            )
            ax.axhspan(
                -floors[sensor],
                floors[sensor],
                color="gray",
                alpha=0.25,
                label="Numerical RMS floor",
            )
            ax.set_title(
                f"{('Amplitude', 'Displacement')[probe]} probe → {('C mass', 'C signed moment')[sensor]}"
            )
            ax.set_ylabel(
                "Signed stimulus − own sham"
                if kind == "full"
                else "Full response − cut response"
            )
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            ax.grid(alpha=0.18)
        for ax in axes[-1]:
            ax.set_xlabel("Time after event")
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(
            f"{('Total C response' if kind == 'full' else 'Selected AB-feedback return contrast')} · fresh seed {FRESH_SEEDS[0]}"
        )
        fig.tight_layout()
        fig.savefig(out / f"{kind}.png", dpi=160)
        plt.close(fig)
    budget.finish()
    print(
        json.dumps(
            {
                "fresh_scores": [
                    (r["seed"], r["kind"], r["weighted_nrmse"], r["practical_failure"])
                    for r in rows
                ]
            }
        ),
        flush=True,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "stage", choices=("development", "qualify", "freeze", "fresh", "analyze")
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--refinement", type=Path)
    p.add_argument("--freeze", type=Path)
    args = p.parse_args()
    if args.output.parent.resolve() != ROOT.resolve():
        p.error(f"Use a new immediate child directory of {ROOT}")
    if args.stage in ("qualify", "freeze", "analyze") and args.data is None:
        p.error("--data required")
    if args.stage == "freeze" and args.refinement is None:
        p.error("--refinement required")
    if args.stage in ("fresh", "analyze") and args.freeze is None:
        p.error("--freeze required")
    previous = sum(load(path)["cpu_seconds"] for path in ROOT.glob("*/budget.json"))
    limit = 1200 if args.stage in ("development", "qualify", "freeze") else 1800
    budget = Budget(previous, limit)
    out = start_panel(
        args.output,
        {
            "purpose": "organization-conditioned-response",
            "stage": args.stage,
            "new_contract": RECORD.read_text(),
            "config": asdict(CFG),
        },
    )
    (out / "contract.md").write_bytes(RECORD.read_bytes())
    status = "FAILED"
    try:
        budget.check()
        if args.stage == "development":
            development(out, budget)
        elif args.stage == "qualify":
            qualification(out, args.data, budget)
        elif args.stage == "freeze":
            freeze(out, args.data, args.refinement, budget)
        elif args.stage == "fresh":
            fresh(out, args.freeze, budget)
        else:
            analyze(out, args.data, args.freeze, budget)
        status = "COMPLETE"
    finally:
        write_json(out / "budget.json", {**budget.receipt(), "status": status})


if __name__ == "__main__":
    main()
