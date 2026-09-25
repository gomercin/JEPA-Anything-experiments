"""Two disjoint spatial interiors in the existing periodic Galerkin solver.

AB's basis/self operator is unmodified. Only the resolved exterior identity is
replaced by an independent C basis on C's rows. No joint basis is fitted.
"""

import json
from dataclasses import asdict

import numpy as np
from scipy.linalg import eigh

from .hybrid_pair import (
    PairHybrid,
    etd_coefficients,
    pair_indices,
    pulse_window,
    save_npz_exclusive,
)
from .simulator import Config


def c_indices(x, center):
    return np.flatnonzero((x >= center - 9) & (x < center + 9))


class DualHybrid(PairHybrid):
    """Reuse unchanged ETD stepping, total-field nonlinearity and source rule."""

    def __init__(self, config, c_position, ab_basis, c_basis, template):
        self.c = config
        self.c_position = float(c_position)
        self.x = np.arange(config.n) * config.length / config.n - config.length / 2
        self.k = 2 * np.pi * np.fft.rfftfreq(config.n, config.length / config.n)
        self.ab_idx = (
            pair_indices(self.x) if ab_basis is not None else np.array([], int)
        )
        self.c_idx = (
            c_indices(self.x, c_position) if c_basis is not None else np.array([], int)
        )
        if np.intersect1d(self.ab_idx, self.c_idx).size:
            raise ValueError("Disjoint interiors required")
        self.ab_basis = None if ab_basis is None else np.array(ab_basis, copy=True)
        self.c_basis = None if c_basis is None else np.array(c_basis, copy=True)
        self.rab = 0 if ab_basis is None else ab_basis.shape[1]
        self.rc = 0 if c_basis is None else c_basis.shape[1]
        self.inside = np.r_[self.ab_idx, self.c_idx]
        self.outside = np.setdiff1d(np.arange(config.n), self.inside)
        self.size = self.rab + self.rc + len(self.outside)
        p = np.zeros((config.n, self.size))
        start = 0
        for idx, b in [(self.ab_idx, self.ab_basis), (self.c_idx, self.c_basis)]:
            if b is not None:
                if b.shape[0] != len(idx) or not np.allclose(
                    b.T @ b, np.eye(b.shape[1]), atol=1e-10
                ):
                    raise ValueError("Invalid local orthonormal basis")
                p[idx, start : start + b.shape[1]] = b
                start += b.shape[1]
        p[self.outside, start + np.arange(len(self.outside))] = 1
        self.template = np.array(template, copy=True)
        if self.template.shape != (len(self.inside),):
            raise ValueError("Static interior template only")
        self.offset = np.zeros(config.n)
        self.offset[self.inside] = self.template
        lu = config.r - (1 - self.k**2) ** 2
        lp = np.fft.irfft(lu[:, None] * np.fft.rfft(p, axis=0), n=config.n, axis=0)
        lr = p.T @ lp
        self.eigenvalues, self.rotation = eigh((lr + lr.T) / 2)
        self.lift = p @ self.rotation
        self.constant = self.lift.T @ np.fft.irfft(
            lu * np.fft.rfft(self.offset), n=config.n
        )
        self.u_coeff = etd_coefficients(self.eigenvalues, config.dt)
        self.m_coeff = etd_coefficients(
            (-1 - config.diffusion * self.k**2) / config.tau, config.dt
        )
        self.q = np.zeros(self.size)
        self.m = np.zeros(len(self.k), complex)
        self.time = 0.0

    @classmethod
    def initialize(cls, config, c_position, ab_basis, c_basis, initial):
        x = np.arange(config.n) * config.length / config.n - config.length / 2
        idx = np.r_[
            pair_indices(x) if ab_basis is not None else np.array([], int),
            c_indices(x, c_position) if c_basis is not None else np.array([], int),
        ]
        obj = cls(config, c_position, ab_basis, c_basis, initial[0, idx])
        obj.q = (
            obj.rotation.T @ np.r_[np.zeros(obj.rab + obj.rc), initial[0, obj.outside]]
        )
        obj.m = np.fft.rfft(initial[1])
        return obj

    def pulse(self, center, amplitude):
        u = self.offset + self.lift @ self.q
        inc = amplitude * pulse_window(self.x, center, self.c.length) * u
        projected = self.lift @ (self.lift.T @ inc)
        self.q += self.lift.T @ inc
        return {
            "increment_l2_error": float(np.linalg.norm(projected - inc)),
            "increment_l2": float(np.linalg.norm(inc)),
            "max_absolute": float(abs(projected - inc).max()),
        }

    def block_ports(self, which):
        idx, b = (
            (self.ab_idx, self.ab_basis)
            if which == "AB"
            else (self.c_idx, self.c_basis)
        )
        if b is None:
            return []
        u, m = self.fields()
        other = u.copy()
        other[idx] = 0
        lu = self.c.r - (1 - self.k**2) ** 2
        force = np.fft.irfft(lu * np.fft.rfft(other), n=self.c.n)[idx]
        return np.r_[self.c.feedback * b.T @ (m[idx] * u[idx]), b.T @ force]

    def save(self, path):
        save_npz_exclusive(
            path,
            config=json.dumps(asdict(self.c)),
            c_position=self.c_position,
            ab_basis=np.empty((0, 0)) if self.ab_basis is None else self.ab_basis,
            c_basis=np.empty((0, 0)) if self.c_basis is None else self.c_basis,
            template=self.template,
            spectral_coordinates=self.q,
            mediator=self.m,
            clock=self.time,
        )

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=False)
        obj = cls(
            Config(**json.loads(str(d["config"]))),
            float(d["c_position"]),
            d["ab_basis"] if d["ab_basis"].size else None,
            d["c_basis"] if d["c_basis"].size else None,
            d["template"],
        )
        obj.q = d["spectral_coordinates"].copy()
        obj.m = d["mediator"].copy()
        obj.time = float(d["clock"])
        return obj
