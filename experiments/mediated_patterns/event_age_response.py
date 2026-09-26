"""Bounded single-event timing evaluator; never imported by causal inference."""

import argparse
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import event_age_model as em
from . import geometry_model as gm
from . import intervention_model as im
from . import present_state_model as pm
from . import repeated_intervention_model as rm
from .geometry_assay import initialize_written
from .geometry_evolution import unforced
from .history_conditioned_transmission import HistoryBudget
from .hybrid_pair import save_npz_exclusive as save_npz
from .present_state_transmission import digest, load
from .repeated_intervention_state import model, prefix_reference, source_hashes
from .simulator import Field
from .source_receiver_relay import CFG

ROOT = Path("work/mediated_patterns/event_age_response")
DEV, FRESH = [18101, 18102, 18103], [20101, 20102, 20103]
AGES, FRESH_AGES = [10, 15, 20, 30], [10, 18, 30]
WINDOWS = {"whole": slice(None), "late": slice(80, None)}


def slots(age, a):
    return ([10, 30], [a, 0.0]) if age == 30 else ([10, 40 - age], [0.0, a])


def arrays(path):
    with np.load(path, allow_pickle=False) as d:
        return dict(d)


def paired(path):
    d = arrays(path)
    y = d["absolute"][:, 1, 2, :2] - d["absolute"][:, 0, 2, :2]
    if not np.array_equal(y, d["response"]):
        raise ValueError("Matched subtraction mismatch")
    return d


def record(meta, z0, age, a, paths):
    d = [paired(p) for p in paths]
    return dict(
        **meta,
        z0=np.asarray(z0),
        age=age,
        a=a,
        y=np.array([x["response"] for x in d]),
        z=np.array([x["z"] for x in d]),
        absolute=np.array([x["absolute"] for x in d]),
    )


def records(inherited, data):
    base = inherited / "fresh-01"
    z = arrays(base / "initial-descriptors.npz")["z"]
    rows = load(base / "rows.json")
    cases = load(base / "cases.json")
    result = []
    for i, meta in enumerate(rows):
        for age in AGES:
            for a in [-0.02, 0.02]:
                if age == 20 or (age == 15 and a > 0):
                    names = [f"reference-{i}-{age}-{v:g}" for v in [0.0, a]]
                    paths = [data / (n + ".npz") for n in names]
                    if age == 15:
                        c = next(
                            c
                            for c in cases
                            if c["index"] == i and c["schedule"]["name"] == "timing"
                        )
                        paths[0] = base / (c["prefixes"][0] + ".npz")
                else:
                    schedule = (
                        "timing"
                        if age == 15
                        else ("cancel" if (age == 30) == (a > 0) else "reverse")
                    )
                    c = next(
                        c
                        for c in cases
                        if c["index"] == i and c["schedule"]["name"] == schedule
                    )
                    j = 1 if age == 30 else 2
                    paths = [base / (c["prefixes"][k] + ".npz") for k in [0, j]]
                result.append(record(meta, z[i], age, a, paths))
    return result


def pilot(out, args, budget):
    base = args.inherited / "fresh-01"
    initial = arrays(base / "boundary-50.npz")["states"][1]
    for a in [0.0, -0.02, 0.02]:
        t, aa = slots(20, a)
        prefix_reference(out, f"reference-1-20-{a:g}", initial, t, aa, budget)


def develop(out, args, budget):
    base = args.inherited / "fresh-01"
    starts = arrays(base / "boundary-50.npz")["states"]
    inputs = {}
    for i, initial in enumerate(starts):
        for age, aa in [(15, [0.02]), (20, [0.0, -0.02, 0.02])]:
            for a in aa:
                name = f"reference-{i}-{age}-{a:g}"
                old = ROOT / "pilot-01" / (name + ".npz")
                if old.exists():
                    # Copy only this task's pilot, retaining the original immutable bytes.
                    for suffix in [".npz", ".json"]:
                        src = old.with_suffix(suffix)
                        dest = out / src.name
                        with dest.open("xb") as f:
                            f.write(src.read_bytes())
                        inputs[str(src)] = digest(src)
                else:
                    t, amps = slots(age, a)
                    prefix_reference(out, name, initial, t, amps, budget)
    inputs.update(
        {
            str(p): digest(p)
            for p in [
                base / "boundary-50.npz",
                args.inherited / "frozen-01/original.json",
                args.inherited / "frozen-01/model.json",
            ]
        }
    )
    pm.save_json(out / "inputs.json", inputs)


def rms(x, axis=0):
    return np.sqrt(np.mean(np.asarray(x) ** 2, axis=axis))


def score(y, p, floors=None):
    floors = {} if floors is None else floors
    result = []
    for kind, t, v in [
        ("R_without", y[0], p[0]),
        ("R_after", y[1], p[1]),
        ("D1", y[1] - y[0], p[1] - p[0]),
    ]:
        for window, sl in WINDOWS.items():
            magnitude, error = rms(t[sl]), rms((v - t)[sl])
            for o, name in enumerate(["mass", "moment"]):
                floor = floors.get(kind + "/" + window, [0.0, 0.0])[o]
                resolved = bool(magnitude[o] > floor)
                relative = float(error[o] / magnitude[o]) if magnitude[o] else None
                h = pm.TIMES[sl]
                peak = int(np.argmax(abs(t[sl, o])))
                result.append(
                    dict(
                        kind=kind,
                        window=window,
                        output=name,
                        magnitude=float(magnitude[o]),
                        error_rms=float(error[o]),
                        error_max=float(abs(v - t)[sl, o].max()),
                        relative=relative,
                        floor=float(floor),
                        resolved=resolved,
                        passed=bool(relative <= (0.1 if kind == "D1" else 0.02))
                        if resolved
                        else None,
                        true_peak=float(t[sl, o][peak]),
                        true_peak_h=float(h[peak]),
                        predicted_at_true_peak=float(v[sl, o][peak]),
                        predicted_peak_h=float(h[np.argmax(abs(v[sl, o]))]),
                    )
                )
    return result


def refine(out, args, budget):
    from scipy.signal import resample

    initial = arrays(args.inherited / "fresh-01/s18101-initial.npz")["initial"]
    base = args.data
    y = np.array(
        [paired(base / f"reference-1-20-{a:g}.npz")["response"] for a in [0.0, 0.02]]
    )
    absolute = np.array(
        [paired(base / f"reference-1-20-{a:g}.npz")["absolute"] for a in [0.0, 0.02]]
    )
    z = np.array([paired(base / f"reference-1-20-{a:g}.npz")["z"] for a in [0.0, 0.02]])
    floors = {}
    details = []
    coord = []
    for label, cfg in [
        ("halfdt", replace(CFG, dt=CFG.dt / 2)),
        ("doubleN", replace(CFG, n=CFG.n * 2)),
    ]:
        sub = out / label
        sub.mkdir()
        budget.begin(label + "-prepare")
        init = initial if cfg.n == CFG.n else resample(initial, cfg.n, axis=-1)
        state = unforced(
            initialize_written(init, "odd04", cfg), cfg, [0.0, 50.0], budget
        )[0][-1]
        budget.finish()
        for a in [0.0, 0.02]:
            t, aa = slots(20, a)
            prefix_reference(sub, f"reference-{a:g}", state, t, aa, budget, cfg)
        rr = [paired(sub / f"reference-{a:g}.npz") for a in [0.0, 0.02]]
        r = np.array([d["response"] for d in rr])
        ab = np.array([d["absolute"] for d in rr])
        coord.append(5 * abs(np.array([d["z"] for d in rr]) - z).max(axis=(0, 1)))
        for window, sl in WINDOWS.items():
            roundoff = (
                64
                * np.finfo(float).eps
                * abs(absolute[:, :, :, 2, :2]).max(axis=(0, 1, 2))
            )
            pair = np.maximum(5 * rms((r - y)[:, sl], axis=1), roundoff)
            branches = 5 * rms((ab - absolute)[:, sl, :, 2, :2], axis=1)
            for kind, w in [
                ("R_without", [1, 0]),
                ("R_after", [0, 1]),
                ("D1", [-1, 1]),
            ]:
                w = np.array(w)
                bound = abs(w) @ pair
                key = kind + "/" + window
                floors[key] = np.maximum(floors.get(key, np.zeros(2)), bound)
                details.append(
                    dict(
                        refinement=label,
                        window=window,
                        kind=kind,
                        pair=pair.tolist(),
                        floor=bound.tolist(),
                        direct=(5 * rms(np.einsum("i,ito->to", w, r - y)[sl])).tolist(),
                        absolute_branch=branches.tolist(),
                        uncorrelated_absolute_bound=(
                            abs(w) @ branches.sum(axis=1)
                        ).tolist(),
                    )
                )
    pm.save_json(
        out / "refinement.json",
        dict(
            floors={k: v.tolist() for k, v in floors.items()},
            records=details,
            coordinate_floors=np.max(coord, axis=0).tolist(),
            rule="max halfdt/doubleN; 5x matched R differences and 64eps absolute scale; triangle propagated D1",
        ),
    )


def predict_original(z, age, a, original):
    yy = []
    for aa in [0.0, a]:
        s = im.State(original, z)
        s.advance(40 - age)
        s.event(aa)
        s.advance(age)
        yy.append(s.response())
    return np.array(yy)


def predict(candidate, z, age, a, checkpoint=None):
    yy = []
    tr = []
    xx = []
    for aa in [0.0, a]:
        s = em.State(candidate, z)
        trajectory, y = em.continue_state(s, [(40 - age, aa)], 40, checkpoint)
        yy.append(y)
        tr.append(trajectory)
        xx.append(s.response_state.copy())
    return np.array(yy), np.array(tr), np.array(xx)


def fresh(out, args, budget):
    from .organization_response import isolated_components
    from .source_receiver_relay import prepare
    from .event_age_analysis import privileged

    frozen = load(args.freeze / "freeze.json")
    candidate = load(args.freeze / "model.json")
    diagnostic = load(args.freeze / "age-diagnostic.json")
    if (
        frozen["sources"] != source_hashes()
        or gm.identity(candidate) != frozen["model_sha256"]
        or set(DEV) & set(FRESH)
    ):
        raise ValueError("Frozen sources/model/groups changed")
    starts = []
    rows = []
    original = model()
    for seed in FRESH:
        f = Field(CFG)
        initial = prepare(f, isolated_components(f, seed, budget), seed, -4.0, budget)
        save_npz(out / f"s{seed}-initial.npz", initial=initial)
        for history in ["none", "odd04"]:
            budget.begin(f"prepare-{seed}-{history}")
            starts.append(
                unforced(
                    initialize_written(initial, history, CFG), CFG, [0.0, 50.0], budget
                )[0][-1]
            )
            rows.append(dict(seed=seed, history=history))
            budget.finish()
    save_npz(out / "boundary-50.npz", states=starts)
    z0 = np.array([pm.extract(s, feature_set="geometry") for s in starts])
    save_npz(out / "initial-descriptors.npz", z=z0)
    pm.save_json(out / "rows.json", rows)
    cases = []
    budget.begin("seal-all-predictions")
    for i, (row, z) in enumerate(zip(rows, z0, strict=True)):
        for age in FRESH_AGES:
            for a in [-0.02, 0.02]:
                name = f"prediction-{i}-{age}-{a:g}"

                def checkpoint(s, i=i, age=age, a=a):
                    suffix = "event" if s.response_state.any() else "zero"
                    s.checkpoint(out / f"checkpoint-{i}-{age}-{a:g}-{suffix}.json")

                y, tr, x = predict(candidate, z, age, a, checkpoint)
                diagnostic_y = privileged(diagnostic, z, age, a)
                save_npz(
                    out / (name + ".npz"),
                    y=y,
                    trajectory=tr,
                    response_state=x,
                    original=predict_original(z, age, a, original),
                    privileged=diagnostic_y,
                )
                cases.append(
                    dict(
                        **row,
                        index=i,
                        age=age,
                        a=a,
                        prediction=name,
                        prefixes=[f"reference-{i}-{age}-{aa:g}" for aa in [0.0, a]],
                    )
                )
    pm.save_json(
        out / "seal.json",
        dict(
            predictions={p.name: digest(p) for p in out.glob("prediction-*.npz")},
            checkpoints={p.name: digest(p) for p in out.glob("checkpoint-*.json")},
            initialization_sha256=digest(out / "initial-descriptors.npz"),
            model_sha256=gm.identity(candidate),
            freeze_sha256=digest(args.freeze / "freeze.json"),
            order="all forecasts and retained states before all future reference outcomes",
        ),
    )
    budget.finish()
    for i, initial in enumerate(starts):
        for age in FRESH_AGES:
            for a in [0.0, -0.02, 0.02]:
                t, aa = slots(age, a)
                prefix_reference(
                    out, f"reference-{i}-{age}-{a:g}", initial, t, aa, budget
                )
    pm.save_json(out / "cases.json", cases)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--stage",
        required=True,
        choices=[
            "pilot",
            "develop",
            "refine",
            "fit",
            "freeze",
            "fresh",
            "analyze",
            "repeated",
        ],
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--inherited", type=Path, required=True)
    p.add_argument("--data", type=Path)
    p.add_argument("--freeze", type=Path)
    p.add_argument("--extra-mode", action="store_true")
    args = p.parse_args()
    if args.output.parent.resolve() != ROOT.resolve() or any(
        x.is_symlink() for x in [args.output, *args.output.parents]
    ):
        raise ValueError("Unique nonsymlink direct output child required")
    if subprocess.check_output(
        ["git", "diff", "HEAD", "--", "experiments/mediated_patterns"], text=True
    ):
        raise RuntimeError("Commit code and plan before science")
    ROOT.mkdir(parents=True, exist_ok=True)
    fd = os.open(ROOT / ".active", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    previous = 30 + sum(load(p)["cpu_seconds"] for p in ROOT.glob("*/budget.json"))
    budget = HistoryBudget(
        previous, 1800 if args.stage in ["fresh", "analyze", "repeated"] else 1200
    )
    started = False
    status = "FAILED"
    try:
        args.output.mkdir(exist_ok=False)
        started = True
        pm.save_json(
            args.output / "protocol.json",
            dict(
                command=sys.argv,
                revision=subprocess.check_output(
                    ["git", "rev-parse", "HEAD"], text=True
                ).strip(),
                sources=source_hashes(),
                note_sha256=digest(Path(__file__).with_name("EVENT_AGE_RESPONSE.md")),
            ),
        )
        model()
        budget.check()
        if args.stage in ["fit", "freeze", "analyze", "repeated"]:
            from . import event_age_analysis as ea

            getattr(ea, args.stage)(args.output, args, budget)
        else:
            globals()[args.stage](args.output, args, budget)
        status = "COMPLETE"
    finally:
        if started:
            pm.save_json(
                args.output / "budget.json", dict(budget.receipt(), status=status)
            )
        (ROOT / ".active").unlink()


if __name__ == "__main__":
    main()
