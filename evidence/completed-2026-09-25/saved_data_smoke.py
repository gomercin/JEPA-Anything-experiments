"""Read only saved arrays and a recorded table. Never import or step a simulator."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    field = root / "work/mediated_patterns"
    dual = field / "dual_reduction"
    freeze = json.loads((dual / "frozen-02/freeze.json").read_text())
    files = {
        "AB": field / "two_way_hybrid/frozen-01/model.npz",
        "C": dual / "frozen-02/model.npz",
    }
    shapes = {"AB": (248, 20), "C": (72, 32)}
    hashes = {}
    for label, path in files.items():
        hashes[label] = hashlib.sha256(path.read_bytes()).hexdigest()
        if hashes[label] != freeze[label + "_model_sha256"]:
            raise ValueError(f"{label} model identity mismatch")
        with np.load(path, allow_pickle=False) as data:
            if data["basis"].shape != shapes[label] or not np.isfinite(data["basis"]).all():
                raise ValueError(f"{label} basis invalid")
    with np.load(dual / "checkpoint-02/state.npz", allow_pickle=False) as state:
        expected = {
            "config",
            "c_position",
            "ab_basis",
            "c_basis",
            "template",
            "spectral_coordinates",
            "mediator",
            "clock",
        }
        if set(state.files) != expected:
            raise ValueError("Unexpected checkpoint state")
        config = json.loads(str(state["config"]))
        if config != freeze["config"]:
            raise ValueError("Checkpoint configuration mismatch")
        sizes = {k: list(state[k].shape) for k in sorted(expected)}
        if sizes["spectral_coordinates"] != [500] or sizes["mediator"] != [385]:
            raise ValueError("Unexpected checkpoint dimensions")
        for label, key in [("AB", "ab_basis"), ("C", "c_basis")]:
            with np.load(files[label], allow_pickle=False) as model:
                np.testing.assert_array_equal(state[key], model["basis"])
        clock = float(state["clock"])
    table = json.loads((dual / "analysis-02/results.json").read_text())
    rr = table["aggregate"]["RR"]
    # Report recorded outcomes; this is not independent scientific validation.
    print(
        json.dumps(
            {
                "script": str(Path(__file__).resolve()),
                "root": str(root),
                "model_hashes": hashes,
                "checkpoint_shapes": sizes,
                "checkpoint_clock": clock,
                "saved_RR_table": rr,
                "source_imports": "standard library plus numpy only",
                "experiments_executed": False,
            },
            indent=2,
            allow_nan=False,
        )
    )


if __name__ == "__main__":
    main()
