"""Read-only report calculations; no simulation or fitted response update.

The extra organization-difference tangent score is descriptive, not a new gate.
Run from the repository root. Refuse to overwrite this report's output folder.
"""

import json
import os
import platform
import sys
from pathlib import Path

import numpy as np

from experiments.mediated_patterns.organization_response import Budget, metric

ROOT = Path("work/mediated_patterns/organization_response")
out = ROOT / "review-01"
out.mkdir(exist_ok=False)
budget = Budget(sum(json.loads(p.read_text())["cpu_seconds"] for p in ROOT.glob("*/budget.json")))
budget.begin("read-only-report-calculations-and-plot")
frozen = json.loads((ROOT / "frozen-01/freeze.json").read_text())
floors = np.asarray(frozen["floors"])
cases = {
    (seed, a): json.loads((ROOT / f"fresh-01/s{seed}-a{a}.json").read_text())
    for seed in frozen["fresh_seeds"]
    for a in (-10, -18)
}
details = {"geometry": {}, "paired_differences": [], "inputs": {}, "runtime": {"python": sys.version, "numpy": np.__version__, "platform": platform.platform()}}
for a in (-10, -18):
    rs = [cases[(seed, a)] for seed in frozen["fresh_seeds"]]
    details["geometry"][str(a)] = {
        key: {
            "mean": np.mean([r["sham_geometry"][0][key] for r in rs], axis=0).tolist(),
            "min": np.min([r["sham_geometry"][0][key] for r in rs], axis=0).tolist(),
            "max": np.max([r["sham_geometry"][0][key] for r in rs], axis=0).tolist(),
        }
        for key in ("mass", "center", "width", "peak", "mediator_means", "source_total", "energy")
    }
    details["geometry"][str(a)]["max_sham_center_drift"] = max(
        float(abs(np.asarray([g["center"] for g in r["sham_geometry"]]) - r["sham_geometry"][0]["center"]).max()) for r in rs
    )
    details["geometry"][str(a)]["max_sham_width_drift"] = max(
        float(abs(np.asarray([g["width"] for g in r["sham_geometry"]]) - r["sham_geometry"][0]["width"]).max()) for r in rs
    )
    details["inputs"][str(a)] = {
        name: {key: [min(r["inputs"][i][key] for r in rs), max(r["inputs"][i][key] for r in rs)] for key in ("l2", "energy_jump", "source_jump", "max_absolute")}
        for i, name in enumerate(("even", "odd"))
    }
for seed in frozen["fresh_seeds"]:
    left, right = cases[(seed, -10)], cases[(seed, -18)]
    for kind in ("full", "cut", "return"):
        delta = np.asarray(right[kind]) - np.asarray(left[kind])
        prediction = np.asarray(right[f"tangent_{kind}"]) - np.asarray(left[f"tangent_{kind}"])
        scores = metric(delta, prediction, np.asarray(frozen["models"][kind]["scales"]), floors)
        details["paired_differences"].append({"seed": seed, "kind": kind, "tangent_difference": scores})
details["max_sham_field_mismatch"] = max(r["sham_field_max"] for r in cases.values())
details["max_final_stimulated_center_drift"] = max(
    float(abs(np.asarray(g["center"]) - r["sham_geometry"][0]["center"]).max())
    for r in cases.values() for g in r["final_stimulated_geometry"]
)
details["max_final_stimulated_width_drift"] = max(
    float(abs(np.asarray(g["width"]) - r["sham_geometry"][0]["width"]).max())
    for r in cases.values() for g in r["final_stimulated_geometry"]
)
os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/organization-response-mpl")
os.environ.setdefault("XDG_CACHE_HOME", "/private/tmp/organization-response-cache")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

left, right = cases[(7101, -10)], cases[(7101, -18)]
times = left["time"]
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True)
for j, ax in enumerate(axes.flat):
    probe, sensor = divmod(j, 2)
    for kind, label, color in (("full", "Total response difference", "#16697a"), ("cut", "Difference with AB feedback cut", "#ce7e00")):
        d = (np.asarray(right[kind]) - np.asarray(left[kind]))[:, probe, sensor]
        ax.plot(times, d, color=color, label=label, linestyle="-" if kind == "full" else "--")
    d = (np.asarray(right["tangent_full"]) - np.asarray(left["tangent_full"]))[:, probe, sensor]
    ax.plot(times, d, ":", color="#8648ac", label="Equation-derived tangent difference")
    ax.axhspan(-floors[sensor], floors[sensor], alpha=0.3, color="gray", label="Numerical RMS floor")
    ax.set_title(f"{('Amplitude', 'Displacement')[probe]} probe → {('C mass', 'C signed moment')[sensor]}")
    ax.set_ylabel("Response at gap 32 − response at gap 24")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.18)
for ax in axes[-1]:
    ax.set_xlabel("Time after event")
axes[0, 0].legend(fontsize=7.8)
fig.suptitle("Small organization effects survive the selected feedback cut · fresh seed 7101")
fig.tight_layout()
fig.savefig(out / "organization_difference.png", dpi=160)
plt.close(fig)
(out / "details.json").write_text(json.dumps(details, indent=2, allow_nan=False) + "\n")
budget.finish()
(out / "budget.json").write_text(json.dumps({**budget.receipt(), "status": "COMPLETE"}, indent=2) + "\n")
print(json.dumps(details, indent=2))
