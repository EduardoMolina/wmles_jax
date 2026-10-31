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
from .schemes import _albada
from .viscous import green_gauss, viscous_flux, scalar_diffusion_flux, ducros


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
            out.append(q + xp.stack([z] + mom + [dE] + [z] * (q.shape[0] - 5)))
        return tuple(out)


class Solver:
    def __init__(self, mesh, gas, scheme, sgs="vreman", cfl=0.8, sgs_kwargs=None,
                 constraints=(), sources=(), bc_passes=2, turbulence=None):
        """`turbulence`: a `turbulence.SpalartAllmaras`.  It adds rho*nu_tilde as the 6th row of
        every block state and replaces the LES closure `sgs`.  Without it the state keeps
        its 5 rows and nothing below changes."""
        self.turbulence = turbulence
        if turbulence is not None:
            if any(g.dist is None for g in mesh.geoms):
                from .wall_distance import attach_wall_distance
                attach_wall_distance(mesh)
            for blk in mesh.blocks:
                for bc in blk.faces.values():
                    if hasattr(bc, "chi_inf"):
                        bc.chi_inf = turbulence.nut_inf_chi
            sgs = None
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
        self._local_dt_j = jit(self._local_dt)
        self._step_local_j = jit(self._step_local)

    # ------------------------------------------------------------------ init
    def _init_block(self, prim_fn, geom, nu_tilde_fn=None):
        q = self.gas.prim_to_cons(asarray(prim_fn(geom.xc)))
        if self.turbulence is None:
            return q
        w, T = self.gas.cons_to_prim(q)
        if nu_tilde_fn is None:
            nu_tilde = self.turbulence.init_nu_tilde(self.gas.mu(T) / w[0])
        else:
            nu_tilde = asarray(nu_tilde_fn(geom.xc))
        return xp.concatenate([q, (w[0] * nu_tilde)[None]], axis=0)

    def init_state(self, prim_fn, nu_tilde_fn=None):
        """prim_fn(xc) -> primitives [rho,u,v,w,p] (5, ...) evaluated at cell centres.
        With a turbulence model, nu_tilde starts at its free-stream value unless
        `nu_tilde_fn(xc)` -> nu_tilde (...) is given."""
        return tuple(self._init_block(prim_fn, g, nu_tilde_fn) for g in self.geoms)

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
    def _turb_state(self, w, T, q, g):
        """(nu_tilde, mu_t, omega, |grad u|^2) from primitives and the gradient of [u,v,w,T,nu_tilde]."""
        turb = self.turbulence
        nu_tilde = q[5] / w[0]
        mut = turb.mu_t(w[0], nu_tilde, self.gas.mu(T) / w[0])
        gu = g[:3]
        omega = xp.sqrt((gu[2, 1] - gu[1, 2]) ** 2 + (gu[0, 2] - gu[2, 0]) ** 2
                        + (gu[1, 0] - gu[0, 1]) ** 2)
        return nu_tilde, mut, omega, xp.sum(gu * gu, axis=(0, 1))

    def _scalar_source(self, w, T, q, g, geom):
        """Production, destruction and cb2 terms of the nu_tilde equation (per volume)."""
        turb = self.turbulence
        nu_tilde, mut, omega, gu2 = self._turb_state(w, T, q, g)
        nu = self.gas.mu(T) / w[0]
        d_len = turb.length_scale(geom.dist, geom.delta_max, nu, mut / w[0], gu2)
        gnt2 = xp.sum(g[4] * g[4], axis=0)
        return turb.source(w[0], nu_tilde, nu, omega, d_len, gnt2)

    def _scalar_flux(self, blk, geom, d, mass_flux, nu_tilde, gnt, coef):
        """Convective (limited 2nd-order upwind on the mass flux) + diffusive flux of rho*nu_tilde."""
        L2, L = blk.slab(nu_tilde, d, -2, -1), blk.slab(nu_tilde, d, -1, 0)
        R, R2 = blk.slab(nu_tilde, d, 0, 1), blk.slab(nu_tilde, d, 1, 2)
        nl = L + 0.5 * _albada(L - L2, R - L)
        nr = R - 0.5 * _albada(R - L, R2 - R)
        conv = mass_flux * xp.where(mass_flux >= 0.0, nl, nr)
        return conv + scalar_diffusion_flux(blk, geom, d, nu_tilde, gnt, coef)

    def _block_rhs(self, ib, q, geom):
        blk, gas, ng = self.blocks[ib], self.gas, self.mesh.ng
        turb = self.turbulence
        w, T = gas.cons_to_prim(q)
        mu = gas.mu(T)
        phi = xp.concatenate([w[1:4], T[None]], axis=0)
        if turb is not None:
            phi = xp.concatenate([phi, (q[5] / w[0])[None]], axis=0)
        g = green_gauss(phi, geom)
        if turb is None:
            mut = self.sgs(g[:3], geom.vol ** (1.0 / 3.0), w[0])
        else:
            nu_tilde, mut, _, _ = self._turb_state(w, T, q, g)
            coef = turb.diffusivity(w[0], nu_tilde, mu)
        mu_eff = mu + mut
        k_eff = gas.cp * (mu / gas.Pr + mut / gas.Pr_t)
        theta = ducros(g[:3])
        rhs = 0.0
        for d in range(3):
            F = self.scheme(blk, geom, w, gas, d, theta) + viscous_flux(
                blk, geom, d, phi[:4], g[:4], mu_eff, k_eff)
            if turb is not None:
                Fs = self._scalar_flux(blk, geom, d, F[0], nu_tilde, g[4], coef)
                F = xp.concatenate([F, Fs[None]], axis=0)
            for s in (0, 1):
                bc = blk.faces[(d, s)]
                if getattr(bc, "flux_override", False):
                    Fw, _, _ = bc.wall_flux(w, T, blk, geom, gas, d, s)
                    if turb is not None:                     # no nu_tilde flux through a wall-modelled wall
                        Fw = xp.concatenate([Fw, xp.zeros_like(Fw[:1])], axis=0)
                    F = set_at(F, sl(F.ndim, d, 0 if s == 0 else blk.n[d]), Fw)
            rhs = rhs - (F[sl(F.ndim, d, slice(1, None))] - F[sl(F.ndim, d, slice(None, -1))])
        rhs = rhs / blk.interior(geom.vol)
        if turb is not None:
            src = blk.interior(self._scalar_source(w, T, q, g, geom))
            rhs = rhs + xp.concatenate([xp.zeros_like(rhs[:5]), src[None]], axis=0)
        return rhs

    def _rhs(self, Q, geoms, time=0.0):
        Q = self.apply_bcs(Q, geoms, time)
        ng = self.mesh.ng
        out = []
        for ib, (q, g) in enumerate(zip(Q, geoms)):
            r = self._block_rhs(ib, q, g)
            for src in self.sources:
                s = src(self.blocks[ib], q, g, self.gas)
                if s.shape[0] < r.shape[0]:
                    s = xp.concatenate([s, xp.zeros_like(r[s.shape[0]:])], axis=0)
                r = r + s
            out.append(xp.pad(r, [(0, 0)] + [(ng, ng)] * 3))
        return tuple(out)

    # ------------------------------------------------------------------ time stepping
    @staticmethod
    def _per_block(dt, n):
        """`dt` is a scalar (all blocks) or a per-block tuple of (ghost-padded) per-cell arrays."""
        return tuple(dt) if isinstance(dt, (tuple, list)) else (dt,) * n

    @staticmethod
    def _rk1(Q, R, dt):
        dts = Solver._per_block(dt, len(Q))
        return tuple(q + d * r for q, r, d in zip(Q, R, dts))

    @staticmethod
    def _rk2(Q0, Q1, R1, dt):
        dts = Solver._per_block(dt, len(Q0))
        return tuple(0.75 * q0 + 0.25 * (q1 + d * r1)
                     for q0, q1, r1, d in zip(Q0, Q1, R1, dts))

    @staticmethod
    def _rk3(Q0, Q2, R2, dt):
        dts = Solver._per_block(dt, len(Q0))
        return tuple((1.0 / 3.0) * q0 + (2.0 / 3.0) * (q2 + d * r2)
                     for q0, q2, r2, d in zip(Q0, Q2, R2, dts))

    def _advance_staged(self, Q, dt, time, local=False):
        """Apply SSP-RK3 with separate compiled RHS and update stages.  `local`: `dt` is a
        tuple of per-cell arrays (no constraints, time offsets use 0)."""
        t1, t2 = (time, time) if local else (time + dt, time + 0.5 * dt)
        R0 = self._rhs_j(Q, self.geoms, time)
        Q1 = self._rk1_j(Q, R0, dt)
        R1 = self._rhs_j(Q1, self.geoms, t1)
        Q2 = self._rk2_j(Q, Q1, R1, dt)
        R2 = self._rhs_j(Q2, self.geoms, t2)
        Q3 = self._rk3_j(Q, Q2, R2, dt)
        for constraint in self.constraints:
            Q3 = constraint(Q3, self.geoms, self.blocks)
        return Q3

    def _block_dt_field(self, blk, q, g):
        """Largest stable dt of every interior cell of one block (ghosts filled)."""
        gas = self.gas
        w, T = gas.cons_to_prim(q)
        c = gas.sound_speed(T)
        nu_mol = gas.mu(T) / w[0]
        if self.turbulence is not None:
            nu_mol = nu_mol + xp.maximum(q[5] / w[0], 0.0)      # nu + nu_tilde bounds the eddy diffusion
        nu = max(4.0 / 3.0, gas.gamma / gas.Pr) * nu_mol
        lam, lamv = 0.0, 0.0
        for d in range(3):
            Sd = g.S[d]
            Sb = 0.5 * (Sd[sl(4, d, slice(0, -1))] + Sd[sl(4, d, slice(1, None))])
            Sm = xp.sqrt(xp.sum(Sb * Sb, axis=0))
            lam = lam + xp.abs(xp.sum(w[1:4] * Sb, axis=0)) + c * Sm
            lamv = lamv + nu * Sm ** 2 / g.vol
        return blk.interior(self.cfl / ((lam + 4.0 * lamv) / g.vol))

    def _dt(self, Q, geoms, time=0.0):
        Q = self.apply_bcs(Q, geoms, time)
        return xp.min(xp.stack([xp.min(self._block_dt_field(blk, q, g))
                                for blk, q, g in zip(self.blocks, Q, geoms)]))

    def _local_dt(self, Q, geoms, time=0.0):
        ng = self.mesh.ng
        Q = self.apply_bcs(Q, geoms, time)
        return tuple(xp.pad(self._block_dt_field(blk, q, g), [(ng, ng)] * 3)
                     for blk, q, g in zip(self.blocks, Q, geoms))

    def max_dt(self, Q, time=0.0):
        return self._dt_j(Q, self.geoms, time)

    def local_dt(self, Q, time=0.0):
        """Per-cell stable time steps (ghost-padded arrays, one per block) for local time stepping
        toward a steady state: pass them as `dt` to `advance_local`."""
        return self._local_dt_j(Q, self.geoms, time)

    def advance(self, Q, dt, time=0.0):
        return self._advance_staged(Q, dt, time)

    def _step_local(self, Q, geoms):
        """Local-dt SSP-RK3 step fused into one compiled function (dt from the state at the start)."""
        dts = self._local_dt(Q, geoms)
        R0 = self._rhs(Q, geoms)
        Q1 = self._rk1(Q, R0, dts)
        Q2 = self._rk2(Q, Q1, self._rhs(Q1, geoms), dts)
        return self._rk3(Q, Q2, self._rhs(Q2, geoms), dts)

    def step_local(self, Q):
        """One steady-state iteration: SSP-RK3 with each cell at its own stable dt (`cfl`)."""
        Q = self._step_local_j(Q, self.geoms)
        for constraint in self.constraints:
            Q = constraint(Q, self.geoms, self.blocks)
        return Q

    def advance_local(self, Q, dts, time=0.0):
        """One SSP-RK3 step with a different time step in every cell (not time accurate)."""
        return self._advance_staged(Q, dts, time, local=True)

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
