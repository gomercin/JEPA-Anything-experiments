"""Grouped development fits and post-seal evaluation; privileged age stays here."""

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import least_squares

from . import event_age_model as em
from . import geometry_model as gm
from . import intervention_model as im
from . import present_state_model as pm
from . import repeated_intervention_model as rm
from .event_age_response import (
    AGES,
    DEV,
    FRESH,
    FRESH_AGES,
    WINDOWS,
    arrays,
    paired,
    predict,
    predict_original,
    records,
    rms,
    score,
)
from .hybrid_pair import save_npz_exclusive as save_npz
from .present_state_transmission import digest, load
from .repeated_intervention_state import model, source_hashes


def geometry_model():
    original = model()
    return {k: original[k] for k in ["G", "F", "J", "transient"]}


def geometry(z, age, a):
    s = im.State(geometry_model(), z)
    s.advance(40 - age)
    s.event(a)
    s.advance(age)
    return s.flow.z.copy(), s.response()


def prepare(rows, prior=False):
    zz = []
    target = []
    for r in rows:
        z, y = geometry(r["z0"], r["age"], r["a"])
        _, base = geometry(r["z0"], r["age"], 0.0)
        zz.append(z)
        if prior:
            yy = predict_original(r["z0"], r["age"], r["a"], model())
            target.append(r["y"][1] - r["y"][0] - (yy[1] - yy[0]))
        else:
            target.append(r["y"][1] - r["y"][0] - (y - base))
    zz = np.array(zz)
    target = np.array(target)
    sy = np.maximum(rms(target, axis=(0, 1)), 1e-18)
    norm = target / sy
    _, _, vt = np.linalg.svd(
        norm.transpose(0, 2, 1).reshape(-1, 161), full_matrices=False
    )
    basis = vt[:4]
    weights = np.einsum("nto,kt->nok", norm, basis).reshape(len(rows), -1)
    mean = zz.mean(axis=0)
    scale = np.maximum(zz.std(axis=0), 1e-14)
    standard = (zz - mean) / scale
    _, singular, axes = np.linalg.svd(standard, full_matrices=False)
    projection = axes[:1].T
    centers = np.c_[np.ones(len(rows)), standard @ projection]
    readout = {
        "mean_z": mean.tolist(),
        "scale_z": scale.tolist(),
        "scale_y": sy.tolist(),
        "basis": basis.tolist(),
        "center_projection": projection.tolist(),
        "center_singular_values": singular.tolist(),
    }
    return centers, weights, readout


def features(rows, center, taus):
    age = np.array([r["age"] for r in rows])
    b = np.array([r["a"] / 0.02 for r in rows])
    modes = np.c_[np.ones(len(rows)), np.exp(-age[:, None] / np.asarray(taus))]
    x = (np.stack([b, b * b], axis=1)[:, :, None] * modes[:, None, :]).reshape(
        len(rows), -1
    )
    return np.einsum("ni,nj->nij", x, center).reshape(len(rows), -1)


def regression(x, y):
    return np.linalg.solve(x.T @ x + 1e-7 * np.eye(x.shape[1]), x.T @ y)


def fit_model(rows, extra=False, anchored=False):
    if {r["age"] for r in rows} != set(AGES) or any(r["seed"] not in DEV for r in rows):
        raise ValueError("Development groups and ages only")
    center, w, readout = prepare(rows, prior=anchored)
    if anchored:
        # Endpoint-preserving stable realization, selected before withheld age access.
        taus = [20.0, 5.0, 50.0]
        anchor_modes = np.c_[
            np.ones(2), np.exp(-np.array([10.0, 30.0])[:, None] / taus)
        ]
        _, _, vh = np.linalg.svd(anchor_modes, full_matrices=True)
        null = vh[2:].T
        transform = np.kron(np.kron(np.eye(2), null), np.eye(center.shape[1]))
        x = features(rows, center, taus)
        readout["coefficients"] = (transform @ regression(x @ transform, w)).tolist()
        m = {
            "schema": 1,
            "geometry": geometry_model(),
            "decays": [1.0, *np.exp(-1 / np.array(taus)).tolist()],
            "readout": readout,
            "prior": model()["readout"],
        }
        details = {
            "taus": taus,
            "training_seeds": sorted({r["seed"] for r in rows}),
            "training_ages": AGES,
            "anchored": True,
            "anchors": [10, 30],
            "response_scalars": 8,
            "nullspace_residual": float(abs(anchor_modes @ null).max()),
            "training_projection_relative": float(
                np.linalg.norm(x @ np.array(readout["coefficients"]) - w)
                / np.linalg.norm(w)
            ),
            "ridge": 1e-7,
        }
        return m, details

    def residual(logtau):
        x = features(rows, center, np.exp(logtau))
        return (x @ regression(x, w) - w).ravel()

    opt = least_squares(
        residual,
        np.log([5.0, 50.0]),
        bounds=(np.log(2.0), np.log(120.0)),
        max_nfev=40,
        ftol=1e-8,
        xtol=1e-8,
        gtol=1e-8,
    )
    taus = sorted(np.exp(opt.x).tolist())
    if extra:
        taus = sorted([*taus, float(np.sqrt(np.prod(taus)))])
    x = features(rows, center, taus)
    readout["coefficients"] = regression(x, w).tolist()
    m = {
        "schema": 1,
        "geometry": geometry_model(),
        "decays": [1.0, *np.exp(-1 / np.asarray(taus)).tolist()],
        "readout": readout,
    }
    details = {
        "taus": taus,
        "optimizer_nfev": opt.nfev,
        "optimizer_success": bool(opt.success),
        "training_seeds": sorted({r["seed"] for r in rows}),
        "training_ages": AGES,
        "training_projection_relative": float(
            np.linalg.norm(x @ np.array(readout["coefficients"]) - w)
            / np.linalg.norm(w)
        ),
        "ridge": 1e-7,
        "extra_mode": extra,
    }
    return m, details


def fit_privileged(rows, anchored=False):
    center, w, readout = prepare(rows, prior=anchored)
    b = np.array([r["a"] / 0.02 for r in rows])
    x = np.einsum("ni,nj->nij", np.c_[b, b * b], center).reshape(len(rows), -1)
    coefficients = []
    for age in AGES:
        mask = np.array([r["age"] == age for r in rows])
        coefficients.append(regression(x[mask], w[mask]).tolist())
    readout["age_coefficients"] = coefficients
    return {
        "geometry": geometry_model(),
        "ages": AGES,
        "readout": readout,
        "inherited_prior": anchored,
    }


def privileged(m, z0, age, a):
    r = m["readout"]
    z, y = geometry(z0, age, a)
    _, base = geometry(z0, age, 0.0)
    q = ((z - r["mean_z"]) / r["scale_z"]) @ np.asarray(r["center_projection"])
    b = a / 0.02
    x = np.outer([b, b * b], np.r_[1.0, q]).ravel()
    coefficient = CubicSpline(m["ages"], r["age_coefficients"], axis=0)(age)
    correction = (
        np.einsum("ok,kt->to", (x @ coefficient).reshape(2, -1), r["basis"])
        * r["scale_y"]
    )
    if m.get("inherited_prior"):
        yy = predict_original(z0, age, a, model())
        return yy + np.array([np.zeros_like(correction), correction])
    return np.array([base, y + correction])


def metadata(r):
    return {k: r[k] for k in ["seed", "history", "age", "a"]}


def evaluate_row(r, m, diagnostic):
    y, tr, x = predict(m, r["z0"], r["age"], r["a"])
    exact = np.array([em.readout(m, r["z"][i, -1], x[i]) for i in range(2)])
    return (
        {
            "causal": y,
            "exact_centers": exact,
            "explicit_age": privileged(diagnostic, r["z0"], r["age"], r["a"]),
            "original9": predict_original(r["z0"], r["age"], r["a"], model()),
        },
        tr,
        x,
    )


def summarize(scores):
    summary = {}
    for name in sorted({r["model"] for r in scores}):
        summary[name] = {}
        for kind in ["R_without", "R_after", "D1"]:
            rows = [r for r in scores if r["model"] == name and r["kind"] == kind]
            resolved = [r for r in rows if r["resolved"]]
            summary[name][kind] = {
                "count": len(rows),
                "resolved": len(resolved),
                "failed": sum(r["passed"] is False for r in rows),
                "worst_relative": max((r["relative"] for r in resolved), default=None),
                "worst": max(resolved, key=lambda r: r["relative"])
                if resolved
                else None,
            }
    return summary


def fit(out, args, budget):
    budget.begin("grouped-single-event-fit")
    rows = records(args.inherited, args.data)
    scores = []
    identity = []
    corrected = load(args.inherited / "frozen-01/model.json")
    for seed in DEV:
        train = [r for r in rows if r["seed"] != seed]
        test = [r for r in rows if r["seed"] == seed]
        m, details = fit_model(train, args.extra_mode, args.anchored)
        diag = fit_privileged(train, args.anchored)
        pm.save_json(out / f"fold-{seed}-model.json", m)
        pm.save_json(out / f"fold-{seed}-fit.json", details)
        pm.save_json(out / f"fold-{seed}-age.json", diag)
        for j, r in enumerate(test):
            candidates, tr, x = evaluate_row(r, m, diag)
            # Exactly the same isolated response for the composition-informed 11-state control.
            yy = []
            for aa in [0.0, r["a"]]:
                s = rm.state_class(corrected)(corrected, r["z0"])
                s.advance(40 - r["age"])
                s.event(aa)
                s.advance(r["age"])
                yy.append(s.response())
            error = float(abs(np.array(yy) - candidates["original9"]).max())
            if error > 1e-18:
                raise ValueError("Single-event collapse violated")
            identity.append(dict(**metadata(r), max_9_vs_11=error))
            for name, p in candidates.items():
                scores.extend(
                    dict(**metadata(r), model=name, **v) for v in score(r["y"], p)
                )
            save_npz(
                out / f"fold-{seed}-case-{j}.npz",
                truth=r["y"],
                true_geometry=r["z"],
                predicted_geometry=tr[:, :, :3],
                response_state=x,
                **candidates,
            )
    m, details = fit_model(rows, args.extra_mode, args.anchored)
    diag = fit_privileged(rows, args.anchored)
    pm.save_json(out / "model.json", m)
    pm.save_json(out / "fit.json", details)
    pm.save_json(out / "age-diagnostic.json", diag)
    pm.save_json(out / "scores.json", scores)
    pm.save_json(out / "summary.json", summarize(scores))
    pm.save_json(out / "one-event-identity.json", identity)
    scales = {
        window: np.max(
            [rms((r["y"][1] - r["y"][0])[sl]) for r in rows], axis=0
        ).tolist()
        for window, sl in WINDOWS.items()
    }
    pm.save_json(out / "development-scales.json", scales)
    budget.finish()


def freeze(out, args, budget):
    budget.begin("freeze-before-fresh")
    m = load(args.data / "model.json")
    diag = load(args.data / "age-diagnostic.json")
    pm.save_json(out / "model.json", m)
    pm.save_json(out / "age-diagnostic.json", diag)
    pm.save_json(
        out / "freeze.json",
        {
            "model_sha256": gm.identity(m),
            "diagnostic_sha256": gm.identity(diag),
            "sources": source_hashes(),
            "development_seeds": DEV,
            "development_ages": AGES,
            "fresh_seeds": FRESH,
            "fresh_ages": FRESH_AGES,
            "amplitudes": [-0.02, 0.02],
            "withheld_age": 18,
            "initial_boundary": 50,
            "probe_boundary": 90,
            "acquisition": "three centers; auxiliary zero means no new event since acquisition",
            "floors": load(args.freeze / "refinement.json"),
            "R_limit": 0.02,
            "D1_limit": 0.1,
            "development_scales": load(args.data / "development-scales.json"),
            "selection_sha256": digest(args.data / "summary.json"),
            "conditional_repeated": "only if every resolved fresh single-event gate passes; no two-event fitting",
        },
    )
    budget.finish()


def analyze(out, args, budget):
    budget.begin("score-sealed-fresh")
    data = args.data
    f = load(args.freeze / "freeze.json")
    m = load(args.freeze / "model.json")
    seal = load(data / "seal.json")
    if gm.identity(m) != seal["model_sha256"] or any(
        digest(data / name) != sha for name, sha in seal["predictions"].items()
    ):
        raise ValueError("Seal mismatch")
    scores = []
    geometry_scores = []
    for c in load(data / "cases.json"):
        ds = [paired(data / (name + ".npz")) for name in c["prefixes"]]
        truth = np.array([d["response"] for d in ds])
        z = np.array([d["z"] for d in ds])
        saved = arrays(data / (c["prediction"] + ".npz"))
        exact = np.array(
            [em.readout(m, z[i, -1], saved["response_state"][i]) for i in range(2)]
        )
        for name, p in [
            ("causal", saved["y"]),
            ("original9", saved["original"]),
            ("original11", saved["original"]),
            ("explicit_age", saved["privileged"]),
            ("exact_centers", exact),
        ]:
            scores.extend(
                dict(**metadata(c), model=name, **v)
                for v in score(truth, p, f["floors"]["floors"])
            )
        dz = saved["trajectory"][:, :, :3] - z
        induced = z[1] - z[0]
        predind = saved["trajectory"][1, :, :3] - saved["trajectory"][0, :, :3]
        geometry_scores.append(
            dict(
                **metadata(c),
                absolute_max=abs(dz).max(axis=(0, 1)).tolist(),
                event_motion_max=abs(induced).max(axis=0).tolist(),
                event_motion_error_max=abs(predind - induced).max(axis=0).tolist(),
            )
        )
    pm.save_json(out / "scores.json", scores)
    pm.save_json(out / "summary.json", summarize(scores))
    pm.save_json(out / "geometry.json", geometry_scores)
    selected = [r for r in scores if r["model"] == "causal"]
    pm.save_json(
        out / "decision.json",
        {
            "single_event_pass": all(r["passed"] is True for r in selected),
            "conditional_repeated_authorized": all(
                r["passed"] is True for r in selected
            ),
            "failed": sum(r["passed"] is False for r in selected),
            "unresolved": sum(r["passed"] is None for r in selected),
        },
    )
    figures(out, data, args.freeze)
    budget.finish()


def repeated(out, args, budget):
    # This stage cannot run on endpoint-only success or exposed development scores.
    if not load(args.data / "decision.json")["single_event_pass"]:
        raise ValueError("Fresh single-event timing contract did not pass")
    from .repeated_intervention_analysis import score as repeated_score
    from .repeated_intervention_analysis import truth

    m = load(args.freeze / "model.json")
    base = args.inherited / "fresh-01"
    cases = load(base / "cases.json")
    z = arrays(base / "initial-descriptors.npz")["z"]
    budget.begin("unchanged-exposed-repeated-reuse")
    # Save all predictions before opening even the exposed repeated responses.
    for i, c in enumerate(cases):
        yy = []
        tr = []
        for flags in rm.PREFIXES:
            events = [
                (t, a * b)
                for t, a, b in zip(
                    c["schedule"]["times"],
                    c["schedule"]["amplitudes"],
                    flags,
                    strict=True,
                )
            ]
            trajectory, y = em.forecast(m, z[c["index"]], events)
            yy.append(y)
            tr.append(trajectory)
        save_npz(out / f"prediction-{i}.npz", y=yy, trajectory=tr)
    pm.save_json(
        out / "seal.json", {p.name: digest(p) for p in out.glob("prediction-*.npz")}
    )
    floors = load(args.inherited / "frozen-01/freeze.json")["floors"]["floors"]
    scores = []
    for i, c in enumerate(cases):
        y, _, _ = truth(base, c)
        p = arrays(out / f"prediction-{i}.npz")["y"]
        for name, v in [
            ("causal", p),
            ("independent_addition", np.array([p[0], p[1], p[2], p[1] + p[2] - p[0]])),
        ]:
            scores.extend(
                dict(
                    seed=c["seed"],
                    history=c["history"],
                    schedule=c["schedule"]["name"],
                    model=name,
                    **v,
                )
                for v in repeated_score(y, v, floors)
            )
    pm.save_json(out / "scores.json", scores)
    pm.save_json(
        out / "decision.json",
        {
            "exposed_repeated_pass": all(
                r["pass"] is True
                for r in scores
                if r["model"] == "causal" and r["kind"] != "K12"
            ),
            "no_composition_fit": True,
        },
    )
    budget.finish()


def figures(out, data, frozen):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cases = load(data / "cases.json")
    # Fixed first written preparation, withheld age and positive event; no sensor selection.
    c = next(c for c in cases if c["index"] == 1 and c["age"] == 18 and c["a"] == 0.02)
    ds = [paired(data / (n + ".npz")) for n in c["prefixes"]]
    y = np.array([d["response"] for d in ds])
    z = np.array([d["z"] for d in ds])
    p = arrays(data / (c["prediction"] + ".npz"))
    m = load(frozen / "model.json")
    exact = np.array(
        [em.readout(m, z[i, -1], p["response_state"][i]) for i in range(2)]
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for o, name in enumerate(["C mass", "C signed moment"]):
        for j, kind in enumerate(["R_after", "D1"]):
            ax = axes[j, o]
            for label, a in [
                ("truth", y),
                ("causal", p["y"]),
                ("original9/11", p["original"]),
                ("explicit age", p["privileged"]),
                ("exact centers", exact),
            ]:
                value = a[1] if j == 0 else a[1] - a[0]
                ax.plot(
                    pm.TIMES,
                    value[:, o],
                    label=label,
                    lw=1.4,
                    ls="-" if label == "truth" else "--",
                )
            ax.set(title=f"{kind}: {name}", xlabel="Time since diagnostic probe")
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Fresh 20101 odd04; event at72, probe90; +.02 conditioning")
    fig.savefig(out / "response-age18.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), constrained_layout=True)
    for k, name in enumerate("ABC"):
        axes[0, k].plot(np.arange(41) + 50, z[1, :, k], label="true post-event history")
        axes[0, k].plot(
            np.arange(41) + 50, p["trajectory"][1, :, k], ls="--", label="predicted"
        )
        axes[1, k].plot(np.arange(41) + 50, (z[1] - z[0])[:, k])
        axes[1, k].plot(
            np.arange(41) + 50, (p["trajectory"][1] - p["trajectory"][0])[:, k], ls="--"
        )
        for j in [0, 1]:
            axes[j, k].axvline(72, color="gray", lw=0.7)
            axes[j, k].set(
                title=f"{name}: "
                + ("center offset" if j == 0 else "event-induced motion"),
                xlabel="Absolute time",
            )
    axes[0, 0].legend(fontsize=8)
    fig.savefig(out / "geometry-age18.png", dpi=160)
    plt.close(fig)


def repeated_fresh(out, args, budget):
    from .geometry_assay import initialize_written
    from .geometry_evolution import unforced
    from .organization_response import isolated_components
    from .repeated_intervention_state import references
    from .simulator import Field
    from .source_receiver_relay import CFG, prepare

    if not load(args.data / "decision.json")["exposed_repeated_pass"]:
        raise ValueError("Exposed repeated contract failed")
    frozen = load(args.freeze / "freeze.json")
    m = load(args.freeze / "model.json")
    if frozen["sources"] != source_hashes() or frozen["model_sha256"] != gm.identity(m):
        raise ValueError("Frozen construction changed")
    starts = []
    rows = []
    for seed in [22101, 22102, 22103]:
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
            rows.append({"seed": seed, "history": history})
            budget.finish()
    save_npz(out / "boundary-50.npz", states=starts)
    z0 = np.array([pm.extract(s, feature_set="geometry") for s in starts])
    save_npz(out / "initial-descriptors.npz", z=z0)
    schedule = {"name": "timing", "times": [10, 25], "amplitudes": [0.02, -0.02]}
    budget.begin("seal-all-repeated-forecasts")
    for i, z in enumerate(z0):
        yy = []
        tr = []
        for j, flags in enumerate(rm.PREFIXES):
            events = [
                (t, a * b)
                for t, a, b in zip(
                    schedule["times"], schedule["amplitudes"], flags, strict=True
                )
            ]

            def checkpoint(s, i=i, j=j):
                s.checkpoint(out / f"checkpoint-{i}-{j}-t{s.geometry.flow.steps}.json")

            trajectory, y = em.forecast(m, z, events, checkpoint=checkpoint)
            yy.append(y)
            tr.append(trajectory)
        save_npz(out / f"prediction-{i}.npz", y=yy, trajectory=tr)
    pm.save_json(
        out / "seal.json",
        {
            "model_sha256": gm.identity(m),
            "predictions": {p.name: digest(p) for p in out.glob("prediction-*.npz")},
            "checkpoints": {p.name: digest(p) for p in out.glob("checkpoint-*.json")},
            "order": "all repeated forecasts before future references",
        },
    )
    budget.finish()
    references(out, starts, rows, [schedule], budget)
    from .repeated_intervention_analysis import score as repeated_score
    from .repeated_intervention_analysis import truth

    floors = load(args.inherited / "frozen-01/freeze.json")["floors"]["floors"]
    scores = []
    for i, c in enumerate(load(out / "cases.json")):
        y, _, _ = truth(out, c)
        p = arrays(out / f"prediction-{i}.npz")["y"]
        scores.extend(
            dict(seed=c["seed"], history=c["history"], model="causal", **v)
            for v in repeated_score(y, p, floors)
        )
    pm.save_json(out / "scores.json", scores)
    pm.save_json(
        out / "decision.json",
        {
            "fresh_repeated_pass": all(
                r["pass"] is True for r in scores if r["kind"] != "K12"
            ),
            "no_composition_fit": True,
        },
    )
