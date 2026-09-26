"""Opt-in evaluator and calibration runner; never imported by reduced inference."""

import argparse
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

from . import geometry_model as gm
from . import present_state_model as pm
from .history_conditioned_transmission import HistoryBudget, regime
from .hybrid_pair import save_npz_exclusive
from .measurements import regions
from .present_state_transmission import OLD, digest, load, old_data, reference
from .simulator import Field
from .source_receiver_relay import CFG

ROOT = Path("work/mediated_patterns/geometry_evolution")
PREVIOUS = Path(
    "evidence/present-state-transmission-2026-09-26/data/work/mediated_patterns/present_state_transmission"
)
F_FILE = PREVIOUS / "frozen-01/models.json"
F_SHA = "f0bc4ced6914bc43148d615912619370465ce48332a140dd41af52b7b85bb74e"
DEV = (8101, 10101, 10102, 10103, 12101, 12102, 12103)
FRESH = (14101, 14102, 14103)
SAMPLES = np.arange(50.0, 101.0, 5.0)


def frozen_response():
    if digest(F_FILE) != F_SHA:
        raise ValueError("Frozen response artifact changed")
    return load(F_FILE)["geometry"]


def center_rates(state, config=CFG):
    """Evaluator diagnostic: exact chain rule for the actual fixed-window extractor.

    The periodic spectral derivative acts on u. Local ell=x-anchor is unwrapped,
    exactly as in E; its nonzero windows do not cross the domain boundary.
    This computes an instantaneous RHS, never an integration or response probe.
    """
    state = np.asarray(state, float)
    z = pm.extract(state, config.length, "geometry")
    f = Field(config)
    v = np.fft.rfft(state)
    ut = np.fft.irfft(f.linear * v + f.nonlinear(v), n=config.n)[0]
    w = regions(f.x, pm.ANCHORS)
    ell = f.x[None] - pm.ANCHORS[:, None]
    dx = config.length / config.n
    mass = (w * state[0] ** 2).sum(axis=1) * dx
    return 2 * (w * (ell - z[:, None]) * state[0] * ut).sum(axis=1) * dx / mass


def development_sources():
    """Current checkpoints and metadata stay separate; file order is explicit."""
    result = []
    for seed in DEV[:4]:
        path = (
            OLD
            / ("develop-01" if seed == 8101 else "fresh-01")
            / f"s{seed}-history.npz"
        )
        for index, history in enumerate(["none", "odd04"]):
            result.append(({"seed": seed, "history": history}, path, index))
    for stage in ["extension-01", "fresh-01"]:
        rows = load(PREVIOUS / stage / "rows.json")
        for seed in dict.fromkeys(r["seed"] for r in rows):
            histories = list(
                dict.fromkeys(r["history"] for r in rows if r["seed"] == seed)
            )
            for index, history in enumerate(histories):
                result.append(
                    (
                        {"seed": seed, "history": history},
                        PREVIOUS / stage / f"s{seed}-history.npz",
                        index,
                    )
                )
    return result


def saved_targets():
    _, y, rows, inputs = old_data()
    lookup = {
        (r["seed"], r["history"], r["wait"]): a for r, a in zip(rows, y, strict=True)
    }
    for stage in ["extension-01", "fresh-01"]:
        path = PREVIOUS / stage / "targets.npz"
        inputs[str(path)] = digest(path)
        inputs[str(PREVIOUS / stage / "rows.json")] = digest(
            PREVIOUS / stage / "rows.json"
        )
        with np.load(path, allow_pickle=False) as f:
            for row, a in zip(
                load(PREVIOUS / stage / "rows.json"), f["y"], strict=True
            ):
                lookup[row["seed"], row["history"], row["wait"]] = a.copy()
    return lookup, inputs


def pilot(out, budget):
    budget.begin("saved-motion-rate-and-persistence-pilot")
    F = frozen_response()
    z, y, rows, inputs = old_data()
    predictions, later, meta, pairs = [], [], [], []
    motion, checks = [], []
    for seed in DEV[:4]:
        root = OLD / ("develop-01" if seed == 8101 else "fresh-01")
        path = root / f"s{seed}-history.json"
        inputs[str(path)] = digest(path)
        geometry = np.asarray(
            [[[*h["center"]] for h in row] for row in load(path)["geometry"]]
        )
        motion.append(geometry[100] - geometry[50])
        for index, history in enumerate(["none", "odd04"]):
            old = rows.index({"seed": seed, "wait": 50.0, "history": history})
            new = rows.index({"seed": seed, "wait": 100.0, "history": history})
            predictions.append(pm.predict(F, z[old : old + 1, :3])[0])
            later.append(y[new])
            meta.append({"seed": seed, "history": history, "wait": 100.0})
            state = pm.load_checkpoint(root / f"s{seed}-history.npz", 50.0, index)
            rate = center_rates(state)
            f = Field(CFG)
            v, _ = f.step(np.fft.rfft(state))
            fd = (
                pm.extract(np.fft.irfft(v, n=CFG.n), feature_set="geometry")
                - z[old, :3]
            ) / CFG.dt
            checks.append(
                {
                    "seed": seed,
                    "history": history,
                    "rate": rate.tolist(),
                    "one_step_rate": fd.tolist(),
                    "difference": (fd - rate).tolist(),
                }
            )
        pairs.append([len(meta) - 2, len(meta) - 1])
    scores = pm.scores(np.asarray(later), np.asarray(predictions), pairs, meta)
    pm.save_json(
        out / "pilot.json",
        {
            "inputs": inputs,
            "drift_50_to_100": np.asarray(motion).tolist(),
            "rates": checks,
            "scores": scores,
            "max_drift": np.max(abs(np.asarray(motion)), axis=(0, 1)).tolist(),
            "worst": {
                kind: max(r["relative_rms"] for r in scores if r["kind"] == kind)
                for kind in ["R", "Delta_R"]
            },
        },
    )
    budget.finish()


def unforced(state, config, sample_times, budget):
    """Evaluator only; times are elapsed from the supplied state, not model inputs."""
    f = Field(config)
    times = np.asarray(sample_times)
    ticks = np.rint(times / config.dt).astype(int)
    if not np.allclose(ticks * config.dt, times, atol=1e-10, rtol=0) or ticks[0] != 0:
        raise ValueError("Invalid reference sampling grid")
    wanted = dict(zip(ticks, range(len(ticks)), strict=True))
    v = np.fft.rfft(state)
    states, screens = [], []
    for k in range(ticks[-1] + 1):
        if k % round(1 / config.dt) == 0:
            s = np.fft.irfft(v, n=config.n)
            _, screen = regime(f, s[None])
            screens.append(screen)
            if not screen["qualified"]:
                raise RuntimeError("Unforced history left qualified regime")
        if k in wanted:
            states.append(np.fft.irfft(v, n=config.n))
        if k == ticks[-1]:
            break
        v, _ = f.step(v)
        if k % 100 == 0:
            budget.check()
    return np.asarray(states), screens


def development(out, budget):
    sources = development_sources()
    targets, inputs = saved_targets()
    rows, states_all, z_all, rates_all, features_all = [], [], [], [], []
    y, response_rows = [], []
    for i, (row, path, index) in enumerate(sources):
        budget.begin(f"unforced-{i}")
        state = pm.load_checkpoint(path, 50.0, index)
        inputs[str(path)] = digest(path)
        states, screens = unforced(state, CFG, SAMPLES - 50, budget)
        z = np.array([pm.extract(s, feature_set="geometry") for s in states])
        rates = np.array([center_rates(s) for s in states])
        features = np.array([pm.extract(s) for s in states])
        later = pm.load_checkpoint(path, 100.0, index)
        pm.save_json(
            out / f"trajectory-{i}.json",
            dict(
                **row,
                regime=screens,
                endpoint_field_max_discrepancy=float(abs(states[-1] - later).max()),
            ),
        )
        # Only new intermediate full states: do not duplicate inherited checkpoints.
        save_npz_exclusive(
            out / f"intermediate-{i}.npz", time=[60.0, 75.0], states=states[[2, 5]]
        )
        rows.append(row)
        states_all.append(states)
        z_all.append(z)
        rates_all.append(rates)
        features_all.append(features)
        budget.finish()
    save_npz_exclusive(
        out / "trajectories.npz",
        time=SAMPLES,
        z=z_all,
        rates=rates_all,
        features=features_all,
    )
    pm.save_json(out / "rows.json", rows)
    for i, (row, states) in enumerate(zip(rows, states_all, strict=True)):
        for t in [50.0, 60.0, 75.0, 100.0]:
            response_rows.append(dict(**row, wait=t))
            if t in (50.0, 100.0):
                y.append(targets[row["seed"], row["history"], t])
            else:
                budget.begin(f"reference-{i}-{t:g}")
                arrays, screens = reference(states[list(SAMPLES).index(t)], CFG, budget)
                save_npz_exclusive(out / f"reference-{i}-{t:g}.npz", **arrays)
                pm.save_json(
                    out / f"reference-{i}-{t:g}.json",
                    {"qualified": all(s["qualified"] for s in screens)},
                )
                y.append(arrays["response"])
                budget.finish()
    save_npz_exclusive(out / "targets.npz", y=y)
    pm.save_json(out / "response-rows.json", response_rows)
    pm.save_json(out / "inputs.json", inputs)


def response_pairs(rows):
    pairs = []
    for j, r in enumerate(rows):
        if r["history"] == "none":
            continue
        matching = [
            i
            for i, a in enumerate(rows)
            if all(a[k] == r[k] for k in r if k != "history") and a["history"] == "none"
        ]
        if len(matching) != 1:
            raise ValueError("Ambiguous or absent independent unwritten origin")
        pairs.append([matching[0], j])
    return pairs


def fit_panel(out, data, budget):
    budget.begin("grouped-free-rollout-selection")
    rows = load(data / "rows.json")
    with np.load(data / "trajectories.npz", allow_pickle=False) as a:
        z, rates, features = a["z"], a["rates"], a["features"]
    selected_times = (SAMPLES <= 75) | (SAMPLES == 100)
    groups = np.array([r["seed"] for r in rows])
    folds = pm.grouped_folds(groups)
    F = frozen_response()
    candidates = [
        ("persistence", 0.0),
        ("drift", 0.0),
        ("affine", 1e-6),
        ("affine", 1e-3),
    ]
    models, summaries = {}, {}
    predicted = {}
    for kind, ridge in candidates:
        name = kind if kind != "affine" else f"affine-{ridge:g}"
        oof = np.empty((len(z), 2, len(SAMPLES), 3))
        oof[:] = np.nan
        fold_info = []
        for train, test in folds:
            g = gm.fit(
                z[train][:, selected_times].reshape(-1, 3),
                rates[train][:, selected_times].reshape(-1, 3),
                kind,
                ridge,
            )
            fold_info.append(
                {
                    "train_groups": np.unique(groups[train]).tolist(),
                    "test_groups": np.unique(groups[test]).tolist(),
                    "model": g,
                }
            )
            for i in test:
                for b, start in enumerate([0, 2]):
                    oof[i, b, start:] = gm.rollout(
                        g, z[i, start], SAMPLES[start:] - SAMPLES[start]
                    )
        # Hold the interior 75-to-100 interval out of fitting and selection.
        residual = (oof - z[:, None])[:, :, selected_times]
        summaries[name] = {
            "max_error": np.nanmax(abs(residual), axis=(0, 1, 2)).tolist(),
            "rms": np.sqrt(np.nanmean(residual**2, axis=(0, 1, 2))).tolist(),
            "score": float(np.sqrt(np.nanmean(residual**2))),
            "folds": fold_info,
        }
        models[name] = gm.fit(
            z[:, selected_times].reshape(-1, 3),
            rates[:, selected_times].reshape(-1, 3),
            kind,
            ridge,
        )
        predicted[name] = np.nan_to_num(
            oof, nan=0.0
        )  # start=60 has no predictions before60; mask documented.
    selected = min(summaries, key=lambda k: summaries[k]["score"])
    response_rows = load(data / "response-rows.json")
    with np.load(data / "targets.npz", allow_pickle=False) as a:
        truth = a["y"]
    snapshot, forecasts, meta = [], {k: [] for k in models}, []
    truth_selected = []
    for r, y in zip(response_rows, truth, strict=True):
        i = rows.index({"seed": r["seed"], "history": r["history"]})
        ti = list(SAMPLES).index(r["wait"])
        for b, start in enumerate([0, 2]):
            if ti <= start:
                continue
            meta.append(dict(**r, origin=float(SAMPLES[start])))
            truth_selected.append(y)
            snapshot.append(pm.predict(F, z[i, ti : ti + 1])[0])
            for name, values in forecasts.items():
                values.append(pm.predict(F, predicted[name][i, b, ti : ti + 1])[0])
    pairs = response_pairs(meta)
    scores = {
        name: pm.scores(np.array(truth_selected), np.array(p), pairs, meta)
        for name, p in dict(snapshot=snapshot, **forecasts).items()
    }
    # F sensitivity to a physical center perturbation, independent of target fitting.
    flat = z[:, selected_times].reshape(-1, 3)
    base = pm.predict(F, flat)
    jac = np.stack(
        [
            (
                pm.predict(F, flat + np.eye(3)[j] * 1e-5)
                - pm.predict(F, flat - np.eye(3)[j] * 1e-5)
            )
            / 2e-5
            for j in range(3)
        ],
        axis=-1,
    )
    sensitivity = (
        np.sqrt(np.mean(jac**2, axis=1)) / np.sqrt(np.mean(base**2, axis=1))[:, :, None]
    )
    pm.save_json(
        out / "fit.json",
        {
            "selected": selected,
            "candidates": summaries,
            "scores": scores,
            "sensitivity_max": sensitivity.max(axis=0).tolist(),
            "geometry_correlation": np.corrcoef(flat.T).tolist(),
            "feature_age_correlation": np.corrcoef(
                np.column_stack(
                    [
                        features[:, selected_times].reshape(-1, 15),
                        np.tile(SAMPLES[selected_times], len(z)),
                    ]
                ).T
            ).tolist(),
            "selection_times": SAMPLES[selected_times].tolist(),
            "n_preparations": len(set(groups)),
            "n_trajectories": len(z),
            "conditioning": models[selected]["condition"],
        },
    )
    pm.save_json(out / "models.json", models)
    save_npz_exclusive(out / "rollouts.npz", **predicted)
    budget.finish()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stage", choices=["pilot", "development", "fit"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path)
    args = p.parse_args()
    if args.output.parent.resolve() != ROOT.resolve() or any(
        x.is_symlink() for x in [args.output, *args.output.parents]
    ):
        raise ValueError("Unique nonsymlink direct output child required")
    if subprocess.check_output(
        ["git", "diff", "HEAD", "--", "experiments/mediated_patterns"], text=True
    ):
        raise RuntimeError("Commit scientific sources and living note before running")
    ROOT.mkdir(parents=True, exist_ok=True)
    fd = os.open(ROOT / ".active", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    previous = 30 + sum(load(x)["cpu_seconds"] for x in ROOT.glob("*/budget.json"))
    budget = HistoryBudget(previous, 1200)
    started, status = False, "FAILED"
    try:
        args.output.mkdir(exist_ok=False)
        started = True
        pm.save_json(
            args.output / "protocol.json",
            {
                "command": sys.argv,
                "revision": subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                "sources": {
                    p.name: digest(p) for p in Path(__file__).parent.glob("*.py")
                },
                "report_sha256": digest(
                    Path(__file__).with_name("GEOMETRY_EVOLUTION.md")
                ),
                "response_model_sha256": digest(F_FILE),
                "previous_cpu": previous,
                "seeds": {"development": DEV, "fresh": FRESH},
            },
        )
        budget.check()
        if args.stage == "pilot":
            pilot(args.output, budget)
        elif args.stage == "development":
            development(args.output, budget)
        elif args.stage == "fit":
            fit_panel(args.output, args.data, budget)
        status = "COMPLETE"
    finally:
        if started:
            pm.save_json(
                args.output / "budget.json", dict(**budget.receipt(), status=status)
            )
        (ROOT / ".active").unlink()


if __name__ == "__main__":
    main()
