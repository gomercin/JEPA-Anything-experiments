"""Fresh-process continuation from both checkpoints, with solvers/arrays denied."""

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    own = time.process_time()
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    freeze = json.loads((args.data / "frozen-01/freeze.json").read_text())
    cases = json.loads((args.data / "fresh-01/cases.json").read_text())
    tasks = []
    for suffix in ["", "-selected"]:
        for case in cases:
            schedule = case["schedule"]
            for j, flags in enumerate([(0, 0), (1, 0), (0, 1), (1, 1)]):
                for t in [schedule["times"][0] + 2, schedule["times"][1] + 2]:
                    path = (
                        args.data
                        / f"fresh-01/checkpoint-{case['index']}-{schedule['name']}-{j}{suffix}-t{t}.json"
                    )
                    tasks.append(
                        {
                            "name": path.name,
                            "suffix": suffix,
                            "index": case["index"],
                            "schedule": schedule["name"],
                            "prefix": j,
                            "times": schedule["times"],
                            "amplitudes": [
                                a * b
                                for a, b in zip(
                                    schedule["amplitudes"], flags, strict=True
                                )
                            ],
                            "checkpoint": json.loads(path.read_text()),
                        }
                    )
    with tempfile.TemporaryDirectory(prefix="repeated-solver-free-") as td:
        td = Path(td)
        package = td / "runtime"
        package.mkdir()
        (package / "__init__.py").write_text("")
        for name in [
            "intervention_model",
            "geometry_model",
            "present_state_model",
            "measurements",
            "repeated_intervention_model",
            "repeated_intervention_extension",
        ]:
            shutil.copyfile(
                Path("experiments/mediated_patterns") / f"{name}.py",
                package / f"{name}.py",
            )
        shutil.copyfile(args.data / "frozen-01/model.json", td / "model.json")
        shutil.copyfile(args.data / "frozen-01/original.json", td / "original.json")
        (td / "tasks.json").write_text(json.dumps(tasks))
        script = """import sys,json,importlib.abc
sys.path.insert(0,'.')
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.startswith(('scipy','experiments')) or 'simulator' in fullname: raise RuntimeError('scientific solver unavailable')
sys.meta_path.insert(0,Block())
import numpy as np
def audit(event,args):
 if event=='open' and isinstance(args[0],str) and args[0].endswith(('.npz','.npy')): raise RuntimeError('field arrays unavailable')
sys.addaudithook(audit)
from runtime.intervention_model import State
from runtime.repeated_intervention_model import continue_state,vector,state_class
models={'-selected':json.load(open('model.json')),'':json.load(open('original.json'))}; results=[]
for task in json.load(open('tasks.json')):
 model=models[task['suffix']]
 state=state_class(model).restore(model,task['checkpoint'])
 _,y=continue_state(state,task['times'],task['amplitudes'],40)
 results.append(dict(name=task['name'],state=vector(state).tolist(),y=y.tolist()))
print(json.dumps(results))
"""
        (td / "resume.py").write_text(script)
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        env.pop("PYTHONPATH", None)
        results = json.loads(
            subprocess.check_output(
                [sys.executable, "-I", "resume.py"],
                cwd=td,
                env=env,
                text=True,
                timeout=60,
            )
        )
    checks = []
    for task, result in zip(tasks, results, strict=True):
        saved = np.load(
            args.data
            / f"fresh-01/prediction-{task['index']}-{task['schedule']}{task['suffix']}.npz"
        )
        ze = float(
            abs(
                np.array(result["state"]) - saved["trajectory"][task["prefix"], -1]
            ).max()
        )
        ye = float(abs(np.array(result["y"]) - saved["y"][task["prefix"]]).max())
        if ze or ye:
            raise AssertionError(
                "Isolated continuation differs from sealed uninterrupted forecast"
            )
        checks.append({"checkpoint": task["name"], "state_max": ze, "response_max": ye})
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = {
        "checks": checks,
        "selected_sha256": freeze["selected_sha256"],
        "own_cpu_seconds": time.process_time() - own,
        "child_cpu_seconds": after.ru_utime
        + after.ru_stime
        - child.ru_utime
        - child.ru_stime,
        "access": "Only inference modules, static model, retained states and future schedule slots; solver imports and all npz/npy reads denied",
    }
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    main()
