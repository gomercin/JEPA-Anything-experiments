"""Affine spatial Galerkin AB coupled to resolved exterior u and full mediator.

No reference solver, measurement, or response-replay imports. A periodic global
trial space retains the SH fourth-order operator, including exterior cross terms.
Only coordinates in that trial space advance, even at intermediate RK stages.
"""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from scipy.linalg import eigh

from .simulator import Config


def etd_coefficients(linear, dt):
    roots = np.exp(1j * np.pi * (np.arange(32) + 0.5) / 32)
    z = dt * linear[..., None] + roots
    return (
        np.exp(dt * linear),
        np.exp(dt * linear / 2),
        dt * np.mean((np.exp(z / 2) - 1) / z, axis=-1).real,
        dt * np.mean((-4 - z + np.exp(z) * (4 - 3 * z + z * z)) / z**3, axis=-1).real,
        dt * np.mean((2 + z + np.exp(z) * (-2 + z)) / z**3, axis=-1).real,
        dt * np.mean((-4 - 3 * z - z * z + np.exp(z) * (4 - z)) / z**3, axis=-1).real,
    )


def pair_indices(x):
    """Fixed AB region; A/B nominal centers -14,+14, C at >=38."""
    return np.flatnonzero((x >= -34) & (x < 28))


def pulse_window(x, center, length, radius=8):
    d = (x - center + length / 2) % length - length / 2
    w = np.zeros_like(x)
    inside = abs(d) < radius
    w[inside] = np.exp(1 - 1 / (1 - (d[inside] / radius) ** 2))
    return w


def save_npz_exclusive(path, **arrays):
    """Honor NumPy's suffix convention without overwriting files or symlinks."""
    path = Path(path)
    if not str(path).endswith(".npz"):
        path = Path(str(path) + ".npz")
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


class PairHybrid:
    """Equation-derived projection; no fitted transition or scalar feedback gain.

    q=(z_AB, u_exterior). We internally diagonalize the projected linear SH
    operator for ETDRK4; checkpoints explicitly store q, not a hidden AB grid.
    The template contains only the initial, static AB profile. Each live AB
    profile is reconstructed from it plus rank-r coefficients, never refreshed.
    """

    def __init__(self, config, basis, template):
        self.c = config
        self.x = np.arange(config.n) * config.length / config.n - config.length / 2
        self.k = 2 * np.pi * np.fft.rfftfreq(config.n, config.length / config.n)
        self.inside = pair_indices(self.x)
        self.outside = np.setdiff1d(np.arange(config.n), self.inside)
        self.basis = np.array(basis, dtype=float, copy=True)
        self.template = np.array(template, dtype=float, copy=True)
        self.rank = self.basis.shape[1]
        if self.basis.shape[0] != len(self.inside) or self.template.shape != (
            len(self.inside),
        ):
            raise ValueError("Pair-only basis/template shapes required")
        if not np.allclose(self.basis.T @ self.basis, np.eye(self.rank), atol=1e-10):
            raise ValueError("Orthonormal spatial basis required")
        self.size = self.rank + len(self.outside)
        p = np.zeros((config.n, self.size))
        p[self.inside, : self.rank] = self.basis
        p[self.outside, self.rank + np.arange(len(self.outside))] = 1
        lu = config.r - (1 - self.k**2) ** 2
        lp = np.fft.irfft(lu[:, None] * np.fft.rfft(p, axis=0), n=config.n, axis=0)
        lr = p.T @ lp
        self.eigenvalues, self.rotation = eigh((lr + lr.T) / 2)
        self.lift = p @ self.rotation
        template_full = np.zeros(config.n)
        template_full[self.inside] = self.template
        self.offset = template_full
        self.constant = self.lift.T @ np.fft.irfft(
            lu * np.fft.rfft(template_full), n=config.n
        )
        lm = (-1 - config.diffusion * self.k**2) / config.tau
        self.u_coeff = etd_coefficients(self.eigenvalues, config.dt)
        self.m_coeff = etd_coefficients(lm, config.dt)
        self.q = np.zeros(self.size)
        self.m = np.zeros(len(self.k), dtype=complex)
        self.time = 0.0

    @classmethod
    def initialize(cls, config, basis, initial):
        x = np.arange(config.n) * config.length / config.n - config.length / 2
        idx = pair_indices(x)
        obj = cls(config, basis, initial[0, idx])
        q = np.r_[np.zeros(obj.rank), initial[0, obj.outside]]
        obj.q = obj.rotation.T @ q
        obj.m = np.fft.rfft(initial[1])
        return obj

    def fields(self, q=None, m=None):
        q = self.q if q is None else q
        m = self.m if m is None else m
        return np.array([self.offset + self.lift @ q, np.fft.irfft(m, n=self.c.n)])

    def nonlinear(self, q, m, replay_m=None, replay_weight=None):
        u, mediator = self.fields(q, m)
        if replay_m is not None:
            mediator = mediator + (replay_m - mediator) * replay_weight
        with np.errstate(over="raise", invalid="raise"):
            f = self.constant + self.lift.T @ (
                self.c.cubic * u**3 - u**5 + self.c.feedback * mediator * u
            )
            # Exactly one source. AB and exterior occupy disjoint spatial rows.
            source = np.fft.rfft(self.c.source * u * u / self.c.tau)
        return f, source

    def step(self, replay=None, replay_weight=None, record=False):
        u, m = self.q, self.m
        e, e2, q, f1, f2, f3 = self.u_coeff
        me, me2, mq, mf1, mf2, mf3 = self.m_coeff

        def nl(a, b, j):
            return self.nonlinear(
                a, b, None if replay is None else replay[j], replay_weight
            )

        nu, nm = nl(u, m, 0)
        a = e2 * u + q * nu
        am = me2 * m + mq * nm
        au, an = nl(a, am, 1)
        b = e2 * u + q * au
        bm = me2 * m + mq * an
        bu, bn = nl(b, bm, 2)
        c = e2 * a + q * (2 * bu - nu)
        cm = me2 * am + mq * (2 * bn - nm)
        cu, cn = nl(c, cm, 3)
        self.q = e * u + f1 * nu + 2 * f2 * (au + bu) + f3 * cu
        self.m = me * m + mf1 * nm + 2 * mf2 * (an + bn) + mf3 * cn
        self.time += self.c.dt
        if not np.isfinite(self.q).all() or not np.isfinite(self.m).all():
            raise FloatingPointError("Divergent hybrid; not clipped")
        return (
            np.array([np.fft.irfft(v, n=self.c.n) for v in (m, am, bm, cm)])
            if record
            else None
        )

    def pulse(self, center, amplitude):
        u = self.offset + self.lift @ self.q
        increment = amplitude * pulse_window(self.x, center, self.c.length) * u
        self.q += self.lift.T @ increment

    def ports(self):
        """Equivalent r+r generalized-force inputs, r outgoing coordinates.

        The coupling integrator evaluates these same terms jointly at RK stages.
        These are equation-defined state-dependent projections, not fitted gains.
        Environment owns m and exterior u; templates reconstruct sources from z.
        """
        u, m = self.fields()
        exterior = np.zeros(self.c.n)
        exterior[self.outside] = u[self.outside]
        lu = self.c.r - (1 - self.k**2) ** 2
        force = np.fft.irfft(lu * np.fft.rfft(exterior), n=self.c.n)
        return {
            "incident_mediator_force": self.c.feedback
            * self.basis.T
            @ (m[self.inside] * u[self.inside]),
            "incident_pattern_force": self.basis.T @ force[self.inside],
            "outgoing_coordinates": (self.rotation @ self.q)[: self.rank],
        }

    def save(self, path):
        # Eigen-coordinates are an orthogonal transform of (z_AB,u_exterior),
        # not additional state. The same static operator reconstructs the
        # transform on load; exact restart is qualified in the same environment.
        save_npz_exclusive(
            path,
            config=json.dumps(asdict(self.c)),
            basis=self.basis,
            template=self.template,
            spectral_coordinates=self.q,
            mediator=self.m,
            clock=np.array(self.time),
        )

    @classmethod
    def load(cls, path):
        data = np.load(Path(path), allow_pickle=False)
        obj = cls(
            Config(**json.loads(str(data["config"]))), data["basis"], data["template"]
        )
        obj.q = data["spectral_coordinates"].copy()
        obj.m = data["mediator"].copy()
        obj.time = float(data["clock"])
        return obj
