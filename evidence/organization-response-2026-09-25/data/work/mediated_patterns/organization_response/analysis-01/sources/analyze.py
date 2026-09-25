"""Read saved fields, make comparable plots, and audit the compact response."""

import argparse
import hashlib
import itertools
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/mediated-mpl")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .explore import start_panel, write_json
from .measurements import describe, outward
from .reduced import ResponseModel, extract_inputs
from .simulator import Config, Field


def quick_plot(directory):
    a = np.load(directory / "case0.npz")
    r = json.loads((directory / "results.json").read_text())["records"][0]
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
    use = abs(a["x"]) < 32
    extent = [a["x"][use][0], a["x"][use][-1], 0, 50]
    for ax, idx, label, cmap, lo, hi in zip(
        axes[:2],
        [0, 1],
        ["Pattern u", "Mediator m"],
        ["RdBu_r", "viridis"],
        [-1.4, 0],
        [1.4, 0.45],
        strict=True,
    ):
        im = ax.imshow(
            a["full"][:, idx, use],
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=lo,
            vmax=hi,
        )
        ax.set(xlabel="Position", ylabel="Time", title=label)
        fig.colorbar(im, ax=ax)
    axes[2].plot(a["time"], np.array(r["response"])[:, 1], label="Full feedback")
    axes[2].plot(a["time"], np.array(r["openloop_response"])[:, 1], label="Sham-mediator replay")
    axes[2].set(
        xlabel="Time after pulse", ylabel="Receiver mass change", title="Pulse minus own sham"
    )
    axes[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(directory / "demonstration.png", dpi=170)
    plt.close(fig)


def energy(a, c):
    k = 2 * np.pi * np.fft.rfftfreq(c.n, c.length / c.n)
    u, m = a[..., 0, :], a[..., 1, :]
    uxx = np.fft.irfft(-k * k * np.fft.rfft(u, axis=-1), n=c.n, axis=-1)
    mx = np.fft.irfft(1j * k * np.fft.rfft(m, axis=-1), n=c.n, axis=-1)
    density = (
        -c.r * u * u / 2
        + (u + uxx) ** 2 / 2
        - c.cubic * u**4 / 4
        + u**6 / 6
        - c.feedback * m * u * u / 2
        + c.feedback * (m * m + c.diffusion * mx * mx) / (4 * c.source)
    )
    return density.sum(axis=-1) * c.length / c.n


def analyze(root, output):
    out = start_panel(
        output,
        {"purpose": "read-only evidence aggregation, plots, extraction cost", "root": str(root)},
    )

    def load(path):
        return json.loads((root / path).read_text())

    qualified = load("qualification-03/results.json")
    fresh = load("fresh-01/results.json")["records"]
    exterior = load("outward-fresh-01/results.json")
    control = load("receiver-control-01/results.json")
    q = np.load(root / "qualification-03/case0.npz")
    causal = np.load(root / "receiver-control-01/fields.npz")
    c = Config(**qualified["records"][0]["config"])
    e = energy(q["long_fields"], c)
    refinements = []
    base_signal = outward(q["x"], q["full"], 14) - outward(q["x"], q["base"], 14)
    for i in (1, 2, 3):
        fine = np.load(root / f"qualification-03/case{i}.npz")
        signal = outward(fine["x"], fine["full"], 14) - outward(fine["x"], fine["base"], 14)
        refinements.append(float(np.max(abs(signal - base_signal))))
    micro = []
    for d in (25, 27, 29, 31):
        for a, b in itertools.combinations((101, 202, 303), 2):
            aa = np.load(root / f"fresh-01/seed{a}-d{d}-initial.npz")["state"]
            bb = np.load(root / f"fresh-01/seed{b}-d{d}-initial.npz")["state"]
            ra = next(r for r in fresh if r["seed"] == a and r["separation"] == d)
            rb = next(r for r in fresh if r["seed"] == b and r["separation"] == d)
            sep_difference = float(abs(extract_inputs(ra)[0] - extract_inputs(rb)[0]))
            relative = float(np.linalg.norm(aa - bb) / np.linalg.norm(aa))
            micro.append(
                {
                    "separation": d,
                    "seeds": [a, b],
                    "coarse_difference": sep_difference,
                    "relative_field_difference": relative,
                    "near_coarse_distinct_micro": sep_difference <= 0.05 and relative > 1e-5,
                    "h20_response_difference": abs(ra["response"][20][1] - rb["response"][20][1]),
                }
            )
    model = ResponseModel.load(root / "development-01/response_model.json")
    state = q["base"][0]
    n = 1000
    start = time.perf_counter()
    for _ in range(n):
        describe(q["x"], state, [-14, 14])
    extract_seconds = (time.perf_counter() - start) / n
    inputs = extract_inputs(fresh[0])
    start = time.perf_counter()
    for _ in range(n):
        model.predict_inputs(inputs)
    model_seconds = (time.perf_counter() - start) / n
    start = time.perf_counter()
    Field(c).evolve(state, 20)
    field_seconds = time.perf_counter() - start
    protected = load("protected.json")
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in protected.items())
    frozen = load("fresh-freeze.json")
    assert all(
        hashlib.sha256(Path(p).read_bytes()).hexdigest() == h for p, h in frozen["files"].items()
    )
    measurement_controls = []
    for radius in (7, 8, 9):
        full = describe(causal["x"], causal["full"], [-14, 14], radius)["mass"]
        base = describe(causal["x"], causal["base"], [-14, 14], radius)["mass"]
        ctrl = describe(causal["x"], causal["control"], [-14, 14], radius)["mass"]
        measurement_controls.append(
            {
                "radius": radius,
                "full_h20": float((full - base)[20, 1]),
                "receiver_control_h20": float((ctrl - base)[20, 1]),
            }
        )
    write_json(
        out / "diagnostics.json",
        {
            "protected_files_verified": len(protected),
            "frozen_files_verified": len(frozen["files"]),
            "energy_initial_final": [float(e[0]), float(e[-1])],
            "max_positive_energy_increment": float(max(0, np.max(np.diff(e)))),
            "outward_refinement_max_errors": refinements,
            "micro_variation": micro,
            "measurement_window_controls": measurement_controls,
            "cost_seconds": {
                "descriptor_extraction": extract_seconds,
                "terminal_map": model_seconds,
                "field_h20": field_seconds,
            },
            "state_size": {
                "full_field_scalars": 2 * c.n,
                "retained_state": 1,
                "known_input": 1,
                "coefficients_each_map": 30,
            },
            "recursive_closure_tested": False,
        },
    )

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    use = abs(causal["x"]) < 34
    ext = [causal["x"][use][0], causal["x"][use][-1], 0, 50]
    im = axes[0, 0].imshow(
        causal["full"][:, 0, use],
        origin="lower",
        extent=ext,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-1.4,
        vmax=1.4,
    )
    axes[0, 0].set(title="Persistent pattern field u", xlabel="Position", ylabel="Time")
    fig.colorbar(im, ax=axes[0, 0])
    dm = causal["full"][:, 1] - causal["base"][:, 1]
    im = axes[0, 1].imshow(
        dm[:, use],
        origin="lower",
        extent=ext,
        aspect="auto",
        cmap="RdBu_r",
        vmin=-0.002,
        vmax=0.002,
    )
    axes[0, 1].set(title="Distributed mediator response Δm", xlabel="Position", ylabel="Time")
    fig.colorbar(im, ax=axes[0, 1])
    times = causal["time"]
    axes[0, 2].plot(times, 1e4 * np.array(control["full_response"])[:, 1], label="Full")
    axes[0, 2].plot(
        times,
        1e4 * np.array(control["receiver_control_response"])[:, 1],
        label="Receiver feedback replay",
    )
    axes[0, 2].set(
        title="Mediator changes the receiver response",
        xlabel="Time after pulse",
        ylabel="Δ receiver mass × 10⁴",
    )
    axes[0, 2].legend(fontsize=8)
    for key, label in (("full", "Full"), ("control", "Receiver feedback replay")):
        delta = outward(causal["x"], causal[key], 14) - outward(causal["x"], causal["base"], 14)
        axes[1, 0].plot(times, 1e6 * delta, label=label)
    axes[1, 0].set(
        title="Outward mediator, beyond both patches",
        xlabel="Time after pulse",
        ylabel="Δ exterior m × 10⁶",
    )
    axes[1, 0].legend(fontsize=8)
    truth = np.array([r["response"][20][1] for r in fresh])
    pred = np.array([r["compact_prediction"][1] for r in fresh])
    for ax, t, p, factor, title in (
        (axes[1, 1], truth, pred, 1e4, "Fresh receiver response"),
        (
            axes[1, 2],
            np.array(exterior["truth"])[:, 1],
            np.array(exterior["prediction"])[:, 1],
            1e6,
            "Fresh outward response",
        ),
    ):
        ax.scatter(t * factor, p * factor, c=[r["separation"] for r in fresh], cmap="viridis", s=24)
        bounds = np.array([min(t.min(), p.min()), max(t.max(), p.max())]) * factor
        ax.plot(bounds, bounds, "k--", lw=1)
        ax.set(
            title=title,
            xlabel=f"Full-field response × {factor:g}",
            ylabel=f"Compact prediction × {factor:g}",
        )
    fig.suptitle(
        "Local field interaction; terminal response prediction on fresh preparations", fontsize=13
    )
    fig.tight_layout()
    fig.savefig(out / "mechanism-and-response.png", dpi=180)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    analyze(a.root, a.output)
