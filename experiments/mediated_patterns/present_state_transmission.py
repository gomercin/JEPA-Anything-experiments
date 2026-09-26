"""Opt-in, budgeted snapshot-response experiment; evaluator isolated from predictor."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from dataclasses import replace

import numpy as np
from scipy.signal import resample

from . import present_state_model as model
from .history_conditioned_transmission import HistoryBudget, regime, write_profile
from .hybrid_pair import save_npz_exclusive
from .organization_response import isolated_components
from .simulator import Field
from .source_receiver_relay import CFG, prepare, read, source_profile

ROOT = Path("work/mediated_patterns/present_state_transmission")
OLD = Path("evidence/history-conditioned-transmission-2026-09-26/data/work/mediated_patterns/history_conditioned_transmission")
DEV = (8101, 10101, 10102, 10103)
FRESH = (12101, 12102, 12103)
WAITS = (50.0, 100.0)


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def npz(path):
    with np.load(path, allow_pickle=False) as z:
        return dict(z)


def paired_indices(rows):
    result = []
    for j, row in enumerate(rows):
        if row["history"] == "none":
            continue
        candidates = [i for i, other in enumerate(rows) if other["seed"] == row["seed"]
                      and other["wait"] == row["wait"] and other["history"] == "none"]
        if len(candidates) != 1:
            raise ValueError("Exactly one matched unwritten state required")
        result.append([candidates[0], j])
    return result


def old_data():
    z, y, rows, inputs = [], [], [], {}
    for seed in DEV:
        root = OLD / ("develop-01" if seed == 8101 else "fresh-01")
        path = root / f"s{seed}-history.npz"
        inputs[str(path)] = digest(path)
        for wait in WAITS:
            for index, history in enumerate(["none", "odd04"]):
                z.append(model.extract(model.load_checkpoint(path, wait, index)))
                target = root / f"s{seed}-t{wait:g}-{'written' if index else 'unwritten'}.npz"
                # Targets deliberately loaded by evaluator, never checkpoint extractor.
                with np.load(target, allow_pickle=False) as data:
                    if not np.array_equal(data["time"], model.TIMES):
                        raise ValueError("Old target timeline differs")
                    y.append(data["full"][:, 0, 2, :2])
                inputs[str(target)] = digest(target)
                rows.append(dict(seed=seed, wait=wait, history=history))
    return np.asarray(z), np.asarray(y), rows, inputs


def old_initial(seed):
    path = OLD / ("pilot-01/preparation.npz" if seed == 8101 else f"fresh-01/s{seed}-preparation.npz")
    with np.load(path, allow_pickle=False) as z:
        return z["initial"].copy()


def advance_histories(initial, writes, config, budget):
    """Natural writes only, sampled regime screen, current checkpoints at 50/100."""
    f = Field(config)
    starts = []
    for odd, even in writes:
        state = initial.copy()
        pulse = odd * write_profile(f, "odd") + even * write_profile(f, "even")
        if np.linalg.norm(pulse) * np.sqrt(config.length / config.n) > 0.40000000001:
            raise ValueError("Write outside initial norm envelope")
        state[0] += pulse
        starts.append(state)
    v = np.fft.rfft(starts)
    checkpoints, screens = [], []
    for k in range(round(100 / config.dt) + 1):
        if k % round(1 / config.dt) == 0:
            states = np.fft.irfft(v, n=config.n)
            _, screen = regime(f, states)
            screens.append(screen)
            if not screen["qualified"]:
                raise RuntimeError("History left three-pattern regime")
            if k in [round(w / config.dt) for w in WAITS]:
                checkpoints.append(states.copy())
        if k == round(100 / config.dt):
            break
        for j in range(len(v)):
            v[j], _ = f.step(v[j])
        if k % 100 == 0:
            budget.check()
    return np.asarray(checkpoints), screens


def reference(state, config, budget):
    """Evaluator-only: two nonlinear branches from the identical current X."""
    f = Field(config)
    stimulated = state.copy()
    stimulated[0] += 0.02 * source_profile(f)
    v = np.fft.rfft([state, stimulated])
    absolute, screens = [], []
    for k in range(round(80 / config.dt) + 1):
        if k % round(0.5 / config.dt) == 0:
            states = np.fft.irfft(v, n=config.n)
            absolute.append(read(f, states, -4.0))
            _, screen = regime(f, states)
            screens.append(screen)
            if not screen["qualified"]:
                raise RuntimeError("Probe left three-pattern regime")
        if k == round(80 / config.dt):
            break
        for j in range(2):
            v[j], _ = f.step(v[j])
        if k % 100 == 0:
            budget.check()
    absolute = np.asarray(absolute)
    return {"absolute": absolute, "response": absolute[:, 1, 2, :2] - absolute[:, 0, 2, :2],
            "time": model.TIMES}, screens


def run_states(out, seeds, families, budget, fresh=False, frozen=None):
    rows, descriptors, states = [], [], []
    for seed in seeds:
        if fresh:
            f = Field(CFG)
            initial = prepare(f, isolated_components(f, seed, budget), seed, -4, budget)
        else:
            initial = old_initial(seed)
        save_npz_exclusive(out / f"s{seed}-initial.npz", initial=initial)
        budget.begin(f"s{seed}-histories")
        cp, screens = advance_histories(initial, list(families.values()), CFG, budget)
        budget.finish()
        save_npz_exclusive(out / f"s{seed}-history.npz", checkpoints=cp, checkpoint_times=WAITS)
        model.save_json(out / f"s{seed}-history.json", {"families": families, "regime": screens})
        for wi, wait in enumerate(WAITS):
            for hi, history in enumerate(families):
                rows.append(dict(seed=seed, wait=wait, history=history))
                states.append(cp[wi, hi])
                descriptors.append(model.extract(cp[wi, hi]))
    descriptors = np.asarray(descriptors)
    model.save_json(out / "rows.json", rows)
    save_npz_exclusive(out / "descriptors.npz", z=descriptors)
    if fresh:
        # Every fresh prediction exists before ANY fresh response is evaluated.
        predictors = load(frozen / "models.json")
        save_npz_exclusive(out / "predictions.npz", **{
            name: model.predict(m, descriptors[:, m["columns"]]) for name, m in predictors.items()})
        model.save_json(out / "prediction-seal.json", {
            "model_sha256": digest(frozen / "models.json"),
            "descriptors_sha256": digest(out / "descriptors.npz"),
            "predictions_sha256": digest(out / "predictions.npz"),
            "before_reference": True})
    targets = []
    for i, (row, state) in enumerate(zip(rows, states, strict=True)):
        budget.begin(f"reference-{i}")
        a, screens = reference(state, CFG, budget)
        save_npz_exclusive(out / f"reference-{i}.npz", **a)
        model.save_json(out / f"reference-{i}.json", dict(**row, regime=screens,
                        current_sha256=hashlib.sha256(state.tobytes()).hexdigest()))
        targets.append(a["response"])
        budget.finish()
        print(json.dumps({"reference": row, "cpu": budget.sections[-1]["cpu_seconds"]}), flush=True)
    save_npz_exclusive(out / "targets.npz", y=targets)


def dataset(extension=None):
    z, y, rows, inputs = old_data()
    if extension:
        more = load(extension / "rows.json")
        z = np.concatenate([z, npz(extension / "descriptors.npz")["z"]])
        y = np.concatenate([y, npz(extension / "targets.npz")["y"]])
        rows += more
        for name in ("rows.json", "descriptors.npz", "targets.npz"):
            inputs[str(extension / name)] = digest(extension / name)
    return z, y, rows, inputs


def fit_panel(out, extension, budget):
    z, y, rows, inputs = dataset(extension)
    pairs = paired_indices(rows)
    groups = [r["seed"] for r in rows]
    models, candidates = {}, []
    for feature_set in model.SETS:
        options = [(1, 1e-6)] if feature_set == "blind" else [(d, r) for d in (1, 2) for r in (1e-6, 1e-3)]
        for degree, ridge in options:
            cv = np.zeros_like(y)
            conditions = []
            for train, test in model.grouped_folds(groups):
                lookup = {j: i for i, j in enumerate(train)}
                train_pairs = [[lookup[i], lookup[j]] for i, j in pairs if i in lookup and j in lookup]
                m = model.fit(z[train], y[train], train_pairs, feature_set, degree, ridge)
                cv[test] = model.predict(m, z[test][:, m["columns"]])
                conditions.append(m["condition"])
            s = model.scores(y, cv, pairs, rows)
            objective = max(r["relative_rms"] / (0.02 if r["kind"] == "R" else 0.1) for r in s)
            name = f"{feature_set}-d{degree}-r{ridge:g}"
            fitted = model.fit(z, y, pairs, feature_set, degree, ridge)
            models[name] = fitted
            candidates.append(dict(name=name, feature_set=feature_set, objective=objective,
                                   fold_conditions=conditions, scores=s))
            save_npz_exclusive(out / f"{name}-cv.npz", prediction=cv)
            budget.check()
    best = {s: min((c for c in candidates if c["feature_set"] == s), key=lambda c: c["objective"])["name"]
            for s in model.SETS}
    summary = {"candidates": candidates, "best": best, "inputs": inputs,
               "preparations": sorted(set(groups)), "rows": rows,
               "feature_min": z.min(axis=0).tolist(), "feature_max": z.max(axis=0).tolist(),
               "feature_correlation": np.corrcoef(z.T).tolist(),
               "best_objectives": {s: next(c["objective"] for c in candidates if c["name"] == n)
                                   for s, n in best.items()}}
    model.save_json(out / "models.json", models)
    model.save_json(out / "fit.json", summary)
    save_npz_exclusive(out / "development.npz", z=z, y=y)
    print(json.dumps({"best": best, "objectives": summary["best_objectives"]}), flush=True)


def refine(out, data, seed, history, budget, fresh=False):
    rows = load(data / "rows.json")
    meta = load(data / f"s{seed}-history.json")
    family = meta["families"][history]
    initial = npz(data / f"s{seed}-initial.npz")["initial"]
    base_targets = npz(data / "targets.npz")["y"]
    if fresh:
        base_rows = rows
    else:
        _, old_y, old_rows, _ = old_data()
        base_rows = old_rows + rows
        base_targets = np.concatenate([old_y, base_targets])
    results = []
    for suffix, config in [("half-dt", replace(CFG, dt=CFG.dt / 2)),
                           ("double-n", replace(CFG, n=2 * CFG.n))]:
        budget.begin(suffix + "-history")
        start = initial if config.n == CFG.n else resample(initial, config.n, axis=-1)
        cp, screen = advance_histories(start, [(0, 0), family], config, budget)
        budget.finish()
        save_npz_exclusive(out / f"{suffix}-checkpoints.npz", checkpoints=cp, checkpoint_times=WAITS)
        model.save_json(out / f"{suffix}-regime.json", screen)
        for wi, wait in enumerate(WAITS):
            errors = []
            for hi, name in enumerate(["none", history]):
                budget.begin(f"{suffix}-{wait}-{name}")
                actual, screen = reference(cp[wi, hi], config, budget)
                save_npz_exclusive(out / f"{suffix}-{wait:g}-{name}.npz", **actual)
                model.save_json(out / f"{suffix}-{wait:g}-{name}.json", screen)
                index = next(i for i, r in enumerate(base_rows) if r == dict(seed=seed, wait=wait, history=name))
                diff = actual["response"] - base_targets[index]
                errors.append([np.sqrt(np.mean(diff[mask] ** 2, axis=0)) for mask in
                               [model.TIMES >= 0, model.TIMES >= 40]])
                budget.finish()
            results.append(dict(refinement=suffix, wait=wait, branch_errors=np.asarray(errors).tolist(),
                                sum_errors=np.sum(errors, axis=0).tolist()))
    model.save_json(out / "refinement.json", {"seed": seed, "history": history, "checks": results,
                    "floors": np.maximum(1e-12, 5 * np.max([r["sum_errors"] for r in results], axis=0)).tolist()})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stage", choices=["fit", "extension", "fresh", "refine"])
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--freeze", type=Path)
    p.add_argument("--seed", type=int, default=8101)
    p.add_argument("--history", default="mixed")
    p.add_argument("--fresh-refinement", action="store_true")
    args = p.parse_args()
    if args.output.parent.resolve() != ROOT.resolve() or any(x.is_symlink() for x in [args.output, *args.output.parents]):
        raise ValueError("Unique nonsymlink direct output child required")
    if subprocess.check_output(["git", "diff", "HEAD", "--", "experiments/mediated_patterns"], text=True):
        raise RuntimeError("Commit scientific sources and living note before running")
    ROOT.mkdir(parents=True, exist_ok=True)
    fd = os.open(ROOT / ".active", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    previous = 30 + sum(load(x)["cpu_seconds"] for x in ROOT.glob("*/budget.json"))
    budget = HistoryBudget(previous, 1800 if args.stage == "fresh" or args.fresh_refinement else 1200)
    started, status = False, "FAILED"
    try:
        args.output.mkdir(exist_ok=False)
        started = True
        hashes = {p.name: digest(p) for p in Path(__file__).parent.glob("*.py")}
        model.save_json(args.output / "protocol.json", {
            "command": sys.argv, "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
            "sources": hashes, "report_sha256": digest(Path(__file__).with_name("PRESENT_STATE_TRANSMISSION.md")),
            "previous_cpu": previous, "seed_split": {"development": DEV, "fresh": FRESH}})
        budget.check()
        if args.stage == "fit":
            fit_panel(args.output, args.data, budget)
        elif args.stage == "extension":
            run_states(args.output, [8101, 10101], {"negative": [-0.4, 0], "mixed": [0.28, 0.28]}, budget)
        elif args.stage == "fresh":
            contract = load(args.freeze / "freeze.json")
            if contract["sources"] != hashes or contract["fresh_seeds"] != list(FRESH):
                raise ValueError("Frozen source/split changed")
            if digest(args.freeze / "models.json") != contract["models_sha256"]:
                raise ValueError("Frozen coefficients changed")
            run_states(args.output, FRESH, contract["families"], budget, True, args.freeze)
        else:
            refine(args.output, args.data, args.seed, args.history, budget, args.fresh_refinement)
        status = "COMPLETE"
    finally:
        if started:
            model.save_json(args.output / "budget.json", dict(**budget.receipt(), status=status))
        (ROOT / ".active").unlink()


if __name__ == "__main__":
    main()
