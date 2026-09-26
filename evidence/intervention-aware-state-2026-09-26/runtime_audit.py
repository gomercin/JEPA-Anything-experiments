"""Replay saved reduced checkpoints in isolated fresh processes, without fields."""

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
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    own = time.process_time()
    child = resource.getrusage(resource.RUSAGE_CHILDREN)
    models = json.loads((a.data / "frozen-01/models.json").read_text())
    freeze = json.loads((a.data / "frozen-01/freeze.json").read_text())
    model = models[freeze["selected"]]
    checks = []
    source = Path("experiments/mediated_patterns")
    with tempfile.TemporaryDirectory(prefix="intervention-solver-free-") as t:
        t = Path(t)
        package = t / "runtime"
        package.mkdir()
        (package / "__init__.py").write_text("")
        for name in [
            "intervention_model.py",
            "geometry_model.py",
            "present_state_model.py",
            "measurements.py",
        ]:
            shutil.copyfile(source / name, package / name)
        (t / "model.json").write_text(json.dumps(model))
        code = """import sys,json,importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.startswith(('scipy','experiments')) or 'simulator' in fullname: raise RuntimeError('scientific solver unavailable')
sys.meta_path.insert(0,Block())
import numpy as np
def audit(event,args):
 if event=='open' and isinstance(args[0],str) and args[0].endswith(('.npz','.npy')): raise RuntimeError('field arrays unavailable')
sys.addaudithook(audit)
from runtime.intervention_model import State
m=json.load(open('model.json'));s=State.restore(m,json.load(open('checkpoint.json')))
z1=s.advance(8);y1=s.response();z2=s.advance(20);y2=s.response()
print(json.dumps(dict(z=[z1.tolist(),z2.tolist()],y=[y1.tolist(),y2.tolist()])))
"""
        env = dict(os.environ, OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
        env.pop("PYTHONPATH", None)
        for path in sorted(
            (a.data / "fresh-01").glob(f"checkpoint-*-{freeze['selected']}-a*.json")
        ):
            shutil.copyfile(path, t / "checkpoint.json")
            result = json.loads(
                subprocess.check_output(
                    [sys.executable, "-c", code], cwd=t, env=env, text=True
                )
            )
            stem = path.stem
            i = int(stem.split("-")[1])
            amp = float(stem.split("-a")[-1])
            ai = [0.0, *freeze["amplitudes"]].index(amp)
            with np.load(
                a.data / f"fresh-01/prediction-{i}-{freeze['selected']}.npz"
            ) as d:
                ze = float(abs(np.asarray(result["z"]) - d["z"][ai]).max())
                ye = float(abs(np.asarray(result["y"]) - d["y"][ai]).max())
            if ze or ye:
                raise AssertionError("Restart differs from uninterrupted prediction")
            checks.append(
                {"checkpoint": path.name, "geometry_max": ze, "response_max": ye}
            )
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    report = {
        "checks": checks,
        "own_cpu_seconds": time.process_time() - own,
        "child_cpu_seconds": (
            after.ru_utime + after.ru_stime - child.ru_utime - child.ru_stime
        ),
        "access": "Only copied runtime modules, model.json and checkpoint.json; solver imports and all npz/npy field reads denied",
    }
    with a.output.open("x") as f:
        json.dump(report, f, indent=2)
        f.write("\n")


if __name__ == "__main__":
    main()
