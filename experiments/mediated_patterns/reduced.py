"""A terminal response map, not an autonomous reduced simulator.

Only measured separation and the declared relative pulse amplitude enter prediction.
Equation-informed decays permit both mediator transport and direct SH tail effects.
No fields, labels used by the solver, future samples, or history enter this map.
"""

import json
from dataclasses import dataclass

import numpy as np

HORIZONS = (5, 20, 50)


def extract_inputs(record):
    centers = record["initial_descriptors"]["center"]
    return np.array([centers[1] - centers[0], record["amplitude"]])


def features(inputs):
    s, a = np.asarray(inputs).T
    xi = (s - 28) / 4
    # Decay/oscillation rates of stationary SH tails at r=-.67.
    kappa = np.sqrt((np.sqrt(1.67) - 1) / 2)
    wave = np.sqrt((np.sqrt(1.67) + 1) / 2)
    med = np.exp(-(s - 28) / 8)
    tail = np.exp(-kappa * (s - 28))
    shape = np.stack(
        [med, med * xi, med * xi * xi, tail * np.cos(wave * s), tail * np.sin(wave * s)], axis=-1
    )
    return np.concatenate([a[..., None] * shape, a[..., None] ** 2 * shape], axis=-1)


@dataclass
class ResponseModel:
    coefficients: np.ndarray
    training_preparations: list
    design_condition: float

    @classmethod
    def fit(cls, records):
        inputs = np.array([extract_inputs(r) for r in records])
        design = features(inputs)
        target = np.array([[r["response"][h][1] for h in HORIZONS] for r in records])
        coeff, _, rank, sv = np.linalg.lstsq(design, target, rcond=None)
        if rank != design.shape[1]:
            raise ValueError("Response design is rank deficient")
        return cls(coeff, sorted({r["preparation_id"] for r in records}), float(sv[0] / sv[-1]))

    def predict_inputs(self, inputs):
        return features(inputs) @ self.coefficients

    def predict(self, record):
        return self.predict_inputs(extract_inputs(record))

    def score(self, records):
        truth = np.array([[r["response"][h][1] for h in HORIZONS] for r in records])
        predictions = np.array([self.predict(r) for r in records])
        independent = np.array(
            [[r["independent_response"][h][1] for h in HORIZONS] for r in records]
        )
        score = {"horizons": HORIZONS, "signal_rms": np.sqrt(np.mean(truth**2, axis=0)).tolist()}
        for name, p in (
            ("interacting", predictions),
            ("independent", independent),
            ("zero_response", np.zeros_like(truth)),
        ):
            error = p - truth
            score[name] = {
                "rmse": np.sqrt(np.mean(error**2, axis=0)).tolist(),
                "nrmse": np.sqrt(np.mean(error**2, axis=0) / np.mean(truth**2, axis=0)).tolist(),
                "max_abs": abs(error).max(axis=0).tolist(),
            }
        score["fresh_preparations_disjoint"] = not set(self.training_preparations) & {
            r["preparation_id"] for r in records
        }
        score["h20_within_frozen_tolerance"] = bool(
            score["interacting"]["max_abs"][1] <= 2e-5 and score["interacting"]["nrmse"][1] <= 0.15
        )
        return score

    def save(self, path):
        path.write_text(
            json.dumps(
                {
                    "kind": "terminal_response_only",
                    "inputs": ["measured_separation", "declared_relative_pulse_amplitude"],
                    "horizons": HORIZONS,
                    "coefficients": self.coefficients.tolist(),
                    "training_preparations": self.training_preparations,
                    "design_condition": self.design_condition,
                },
                indent=2,
                allow_nan=False,
            )
            + "\n"
        )

    @classmethod
    def load(cls, path):
        data = json.loads(path.read_text())
        return cls(
            np.array(data["coefficients"]), data["training_preparations"], data["design_condition"]
        )
