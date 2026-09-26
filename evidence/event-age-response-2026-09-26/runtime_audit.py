"""Replay sealed states in an isolated process without solver or field files."""

import argparse
import json
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.process_time()
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    source = Path(__file__).resolve().parents[2] / "experiments/mediated_patterns"
    cases = json.loads((args.data / "cases.json").read_text())
    jobs = []
    expected = []
    for case in cases:
        stem = f"checkpoint-{case['index']}-{case['age']}-{case['a']:g}"
        saved = np.load(args.data / (case["prediction"] + ".npz"))
        for index, label in enumerate(["zero", "event"]):
            cp = json.loads((args.data / (stem + "-" + label + ".json")).read_text())
            jobs.append(
                {
                    "checkpoint": cp,
                    "events": [[40 - case["age"], 0.0 if index == 0 else case["a"]]],
                }
            )
            expected.append(saved["y"][index])
    with tempfile.TemporaryDirectory(
        prefix="event-age-runtime-", dir=Path(tempfile.gettempdir()).resolve()
    ) as temp:
        root = Path(temp)
        package = root / "runtime"
        package.mkdir()
        (package / "__init__.py").write_text("")
        for name in [
            "event_age_model",
            "geometry_model",
            "intervention_model",
            "present_state_model",
            "measurements",
        ]:
            shutil.copyfile(source / (name + ".py"), package / (name + ".py"))
        shutil.copyfile(args.freeze / "model.json", root / "model.json")
        (root / "jobs.json").write_text(json.dumps(jobs))
        script = """
import builtins,json,sys,time
sys.path.insert(0,'.')
original=builtins.__import__
def guard(name,*args,**kwargs):
    if name.startswith(('scipy','experiments')) or 'simulator' in name:raise RuntimeError('scientific imports forbidden')
    return original(name,*args,**kwargs)
builtins.__import__=guard
import numpy as np
def forbidden(*args,**kwargs):raise RuntimeError('array files forbidden')
np.load=forbidden
from runtime.event_age_model import State,continue_state
m=json.load(open('model.json'));jobs=json.load(open('jobs.json'));results=[]
for job in jobs:
    state=State.restore(m,job['checkpoint'])
    _,y=continue_state(state,job['events'],40)
    results.append(y.tolist())
print(json.dumps(results))
"""
        (root / "resume.py").write_text(script)
        output = subprocess.check_output(
            [sys.executable, "-I", "resume.py"], cwd=root, text=True, timeout=120
        )
    actual = np.asarray(json.loads(output))
    target = np.asarray(expected)
    np.testing.assert_array_equal(actual, target)
    later = resource.getrusage(resource.RUSAGE_CHILDREN)
    receipt = {
        "checkpoints": len(jobs),
        "exact_equal": True,
        "max_absolute_error": float(abs(actual - target).max()),
        "fields_available": False,
        "scientific_solvers_available": False,
        "cpu_seconds": time.process_time()
        - start
        + later.ru_utime
        + later.ru_stime
        - child.ru_utime
        - child.ru_stime,
    }
    with args.output.open("x") as f:
        json.dump(receipt, f, indent=2)
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
