"""Multiblock explicit finite-volume driver.

State = tuple of arrays, one per block, each (5, Ni, Nj, Nk) incl. ghost cells.
All heavy functions take `geoms` explicitly (not via closure) so that the metrics
are jit *arguments* instead of huge compile-time constants.
"""
import numpy as np

from .backend import xp, asarray, jit, set_at
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
        self._rhs_j, self._dt_j = jit(self._rhs), jit(self._dt)
        self._rk1_j, self._rk2_j, self._rk3_j = (
            jit(self._rk1), jit(self._rk2), jit(self._rk3)
        )
        self._wall_stats_j = jit(self._wall_stats)

    # ------------------------------------------------------------------ init
    def init_state(self, prim_fn):
        """prim_fn(xc) -> primitives [rho,u,v,w,p] (5, ...) evaluated at cell centres."""
        return tuple(self.gas.prim_to_cons(asarray(prim_fn(g.xc))) for g in self.geoms)

    # ------------------------------------------------------------------ ghost cells
    def apply_bcs(self, Q, geoms, time=0.0):
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
                            q = f.fill_at_time(q, blk, geoms[ib], self.gas, a, s, time)
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

    def _rhs(self, Q, geoms, time=0.0):
        Q = self.apply_bcs(Q, geoms, time)
        ng = self.mesh.ng
        out = []
        for ib, (q, g) in enumerate(zip(Q, geoms)):
            r = self._block_rhs(ib, q, g)
            for src in self.sources:
                r = r + src(self.blocks[ib], q, g, self.gas)
            out.append(xp.pad(r, [(0, 0)] + [(ng, ng)] * 3))
        return tuple(out)

    # ------------------------------------------------------------------ time stepping
    @staticmethod
    def _rk1(Q, R, dt):
        return tuple(q + dt * r for q, r in zip(Q, R))

    @staticmethod
    def _rk2(Q0, Q1, R1, dt):
        return tuple(0.75 * q0 + 0.25 * (q1 + dt * r1)
                     for q0, q1, r1 in zip(Q0, Q1, R1))

    @staticmethod
    def _rk3(Q0, Q2, R2, dt):
        return tuple((1.0 / 3.0) * q0 + (2.0 / 3.0) * (q2 + dt * r2)
                     for q0, q2, r2 in zip(Q0, Q2, R2))

    def _advance_staged(self, Q, dt, time):
        """Apply SSP-RK3 with separate compiled RHS and update stages."""
        R0 = self._rhs_j(Q, self.geoms, time)
        Q1 = self._rk1_j(Q, R0, dt)
        R1 = self._rhs_j(Q1, self.geoms, time + dt)
        Q2 = self._rk2_j(Q, Q1, R1, dt)
        R2 = self._rhs_j(Q2, self.geoms, time + 0.5 * dt)
        Q3 = self._rk3_j(Q, Q2, R2, dt)
        for constraint in self.constraints:
            Q3 = constraint(Q3, self.geoms, self.blocks)
        return Q3

    def _dt(self, Q, geoms, time=0.0):
        gas = self.gas
        Q = self.apply_bcs(Q, geoms, time)
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

    def max_dt(self, Q, time=0.0):
        return self._dt_j(Q, self.geoms, time)

    def advance(self, Q, dt, time=0.0):
        return self._advance_staged(Q, dt, time)

    def rhs(self, Q, time=0.0):
        return self._rhs_j(Q, self.geoms, time)

    # ------------------------------------------------------------------ diagnostics
    def wall_stats(self, Q):
        """{(block, face): (tau_w, q_w)} for all wall-modelled faces."""
        return {
            key: (np.asarray(tau), np.asarray(qw))
            for key, (tau, qw) in self._wall_stats_j(Q, self.geoms).items()
        }

    def _wall_stats(self, Q, geoms):
        Q = self.apply_bcs(Q, geoms)
        out = {}
        for ib, (blk, q, g) in enumerate(zip(self.blocks, Q, geoms)):
            w, T = self.gas.cons_to_prim(q)
            for (a, s), bc in blk.faces.items():
                if isinstance(bc, WallModel):
                    _, tau, qw = bc.wall_flux(w, T, blk, g, self.gas, a, s)
                    out[(blk.name, "ijk"[a] + ("min", "max")[s])] = (tau, qw)
        return out

    def wall_loads(self, Q, moment_center=(0.0, 0.0, 0.0)):
        """Integrate fluid-on-wall force and moment for each WallModel face.

        Returns ``{(block, face): (force, moment)}``, with Cartesian force in N
        and moment in N m. Moments are taken about ``moment_center``.
        """
        center = np.asarray(moment_center, dtype=float)
        if center.shape != (3,) or not np.all(np.isfinite(center)):
            raise ValueError("moment_center must contain three finite coordinates")

        Q = self.apply_bcs(Q, self.geoms)
        out = {}
        for blk, q, g in zip(self.blocks, Q, self.geoms):
            w, T = self.gas.cons_to_prim(q)
            for (axis, side), bc in blk.faces.items():
                if not isinstance(bc, WallModel):
                    continue

                ng, n = blk.ng, blk.n[axis]
                face_index = ng if side == 0 else ng + n
                match_index = ng + bc.n_match - 1 if side == 0 else ng + n - bc.n_match
                cell_index = ng if side == 0 else ng + n - 1

                area_vector = blk.take(g.S[axis], axis, face_index)
                area = xp.sqrt(xp.sum(area_vector * area_vector, axis=0))
                normal = area_vector / area
                if side == 0:
                    normal = -normal

                velocity = blk.take(w[1:4], axis, match_index)
                tangential_velocity = velocity - xp.sum(
                    velocity * normal, axis=0
                )[None] * normal
                speed = xp.sqrt(xp.sum(tangential_velocity ** 2, axis=0) + 1e-20)
                tangent = tangential_velocity / speed
                _, tau, _ = bc.wall_flux(w, T, blk, g, self.gas, axis, side)
                wall_pressure = blk.take(w[4], axis, cell_index)
                traction = (wall_pressure[None] * normal + tau[None] * tangent) * area[None]

                face_center = blk.take(g.xf[axis], axis, face_index)
                arm = face_center - center.reshape((3,) + (1,) * (face_center.ndim - 1))
                moment_density = xp.stack([
                    arm[1] * traction[2] - arm[2] * traction[1],
                    arm[2] * traction[0] - arm[0] * traction[2],
                    arm[0] * traction[1] - arm[1] * traction[0],
                ])
                tangential_axes = tuple(range(1, traction.ndim))
                force = xp.sum(traction, axis=tangential_axes)
                moment = xp.sum(moment_density, axis=tangential_axes)
                face = "ijk"[axis] + ("min", "max")[side]
                out[(blk.name, face)] = (np.asarray(force), np.asarray(moment))
        return out
