"""Multiblock explicit finite-volume driver.

State = tuple of arrays, one per block, each (5, Ni, Nj, Nk) incl. ghost cells.
All heavy functions take `geoms` explicitly (not via closure) so that the metrics
are jit *arguments* instead of huge compile-time constants.
"""
import numpy as np

from .backend import xp, jit, set_at
from .mesh import Interface, exchange, sl
from .bc import WallModel
from .sgs import make_sgs
from .viscous import green_gauss, viscous_flux, ducros


class ConstantMassFlux:
    """Post-step constraint: shift the volume-averaged rho*u_dir to `target`."""

    def __init__(self, direction, target):
        self.dir, self.target = direction, target

    def __call__(self, Q, geoms, blocks):
        num, den = 0.0, 0.0
        for q, g, b in zip(Q, geoms, blocks):
            vol = b.interior(g.vol)
            num = num + xp.sum(b.interior(q[1 + self.dir]) * vol)
            den = den + xp.sum(vol)
        delta = self.target - num / den
        out = []
        for q in Q:
            z = xp.zeros_like(q[0])
            mom = [z, z, z]
            mom[self.dir] = z + delta
            dE = (2.0 * q[1 + self.dir] * delta + delta ** 2) / (2.0 * q[0])
            out.append(q + xp.stack([z] + mom + [dE]))
        return tuple(out)


class Solver:
    def __init__(self, mesh, gas, scheme, sgs="vreman", cfl=0.8, sgs_kwargs=None,
                 constraints=(), sources=(), bc_passes=2):
        self.mesh, self.blocks, self.gas = mesh, mesh.blocks, gas
        self.geoms = mesh.geoms
        self.scheme, self.cfl = scheme, cfl
        self.sgs = make_sgs(sgs, **(sgs_kwargs or {}))
        self.constraints, self.sources, self.bc_passes = tuple(constraints), tuple(sources), bc_passes
        self._rhs_j, self._step_j, self._dt_j = jit(self._rhs), jit(self._step), jit(self._dt)

    # ------------------------------------------------------------------ init
    def init_state(self, prim_fn):
        """prim_fn(xc) -> primitives [rho,u,v,w,p] (5, ...) evaluated at cell centres."""
        return tuple(self.gas.prim_to_cons(xp.asarray(prim_fn(g.xc))) for g in self.geoms)

    # ------------------------------------------------------------------ ghost cells
    def apply_bcs(self, Q, geoms):
        Q = list(Q)
        for _ in range(self.bc_passes):
            for a in range(3):
                for ib, blk in enumerate(self.blocks):
                    q = Q[ib]
                    for s in (0, 1):
                        f = blk.faces[(a, s)]
                        if isinstance(f, Interface):
                            src = q if f.nb == ib else Q[f.nb]
                            q = exchange(q, src, blk, (a, s), self.blocks[f.nb], f, "cell")
                        else:
                            q = f.fill(q, blk, geoms[ib], self.gas, a, s)
                    Q[ib] = q
        return tuple(Q)

    # ------------------------------------------------------------------ RHS
    def _block_rhs(self, ib, q, geom):
        blk, gas, ng = self.blocks[ib], self.gas, self.mesh.ng
        w, T = gas.cons_to_prim(q)
        mu = gas.mu(T)
        phi = xp.concatenate([w[1:4], T[None]], axis=0)
        g = green_gauss(phi, geom)
        mut = self.sgs(g[:3], geom.vol ** (1.0 / 3.0), w[0])
        mu_eff = mu + mut
        k_eff = gas.cp * (mu / gas.Pr + mut / gas.Pr_t)
        theta = ducros(g[:3])
        rhs = 0.0
        for d in range(3):
            F = self.scheme(blk, geom, w, gas, d, theta) + viscous_flux(blk, geom, d, phi, g, mu_eff, k_eff)
            for s in (0, 1):
                bc = blk.faces[(d, s)]
                if getattr(bc, "flux_override", False):
                    Fw, _, _ = bc.wall_flux(w, T, blk, geom, gas, d, s)
                    F = set_at(F, sl(F.ndim, d, 0 if s == 0 else blk.n[d]), Fw)
            rhs = rhs - (F[sl(F.ndim, d, slice(1, None))] - F[sl(F.ndim, d, slice(None, -1))])
        return rhs / blk.interior(geom.vol)

    def _rhs(self, Q, geoms):
        Q = self.apply_bcs(Q, geoms)
        ng = self.mesh.ng
        out = []
        for ib, (q, g) in enumerate(zip(Q, geoms)):
            r = self._block_rhs(ib, q, g)
            for src in self.sources:
                r = r + src(self.blocks[ib], q, g, self.gas)
            out.append(xp.pad(r, [(0, 0)] + [(ng, ng)] * 3))
        return tuple(out)

    # ------------------------------------------------------------------ time stepping
    def _step(self, Q, geoms, dt):
        ax = lambda a, x, b, y: tuple(a * u + b * v for u, v in zip(x, y))
        R0 = self._rhs(Q, geoms)
        Q1 = ax(1.0, Q, dt, R0)
        R1 = self._rhs(Q1, geoms)
        Q2 = ax(0.75, Q, 0.25, ax(1.0, Q1, dt, R1))
        R2 = self._rhs(Q2, geoms)
        Q3 = ax(1.0 / 3.0, Q, 2.0 / 3.0, ax(1.0, Q2, dt, R2))
        for c in self.constraints:
            Q3 = c(Q3, geoms, self.blocks)
        return Q3

    def _dt(self, Q, geoms):
        gas = self.gas
        Q = self.apply_bcs(Q, geoms)
        dts = []
        for blk, q, g in zip(self.blocks, Q, geoms):
            w, T = gas.cons_to_prim(q)
            c = gas.sound_speed(T)
            nu = max(4.0 / 3.0, gas.gamma / gas.Pr) * gas.mu(T) / w[0]
            lam, lamv = 0.0, 0.0
            for d in range(3):
                Sd = g.S[d]
                Sb = 0.5 * (Sd[sl(4, d, slice(0, -1))] + Sd[sl(4, d, slice(1, None))])
                Sm = xp.sqrt(xp.sum(Sb * Sb, axis=0))
                lam = lam + xp.abs(xp.sum(w[1:4] * Sb, axis=0)) + c * Sm
                lamv = lamv + nu * Sm ** 2 / g.vol
            dts.append(xp.min(blk.interior(self.cfl / ((lam + 4.0 * lamv) / g.vol))))
        return xp.min(xp.stack(dts))

    def max_dt(self, Q):
        return self._dt_j(Q, self.geoms)

    def advance(self, Q, dt):
        return self._step_j(Q, self.geoms, dt)

    def rhs(self, Q):
        return self._rhs_j(Q, self.geoms)

    # ------------------------------------------------------------------ diagnostics
    def wall_stats(self, Q):
        """{(block, face): (tau_w, q_w)} for all wall-modelled faces."""
        Q = self.apply_bcs(Q, self.geoms)
        out = {}
        for ib, (blk, q, g) in enumerate(zip(self.blocks, Q, self.geoms)):
            w, T = self.gas.cons_to_prim(q)
            for (a, s), bc in blk.faces.items():
                if isinstance(bc, WallModel):
                    _, tau, qw = bc.wall_flux(w, T, blk, g, self.gas, a, s)
                    out[(blk.name, "ijk"[a] + ("min", "max")[s])] = (np.asarray(tau), np.asarray(qw))
        return out
