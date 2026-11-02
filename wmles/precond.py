"""Block-local preconditioners for the matrix-free Newton-GMRES integrator (plan B3).

A preconditioner exposes ``build(Q, dt, time)`` (refresh from the Newton iterate, ``dt`` as
prepared by ``implicit._per_block_dt``) and ``apply(v)`` (``M^-1 v`` on interior block vectors in
the integrator's scaled variables). Everything is local to a block: no halo exchange and no
global reduction, so the same objects work per device on a sharded solver.
"""

import numpy as np

from .backend import xp, jit, set_at, HAS_JAX
from .mesh import sl
from .viscous import green_gauss


class PointJacobi:
    """Point (cell-diagonal) Jacobi from a first-order upwind Jacobian.

    For a Rusanov-type first-order flux the convective diagonal is ``0.5 * sum_f lambda_f A_f / V``
    times the identity (the ``dF/dQ`` parts cancel over a closed cell), and the viscous diagonal
    is of the order of ``sum_f nu A_f^2 / V^2``. The nu_tilde row adds the destruction part of the
    Spalart-Allmaras source Jacobian (production is left out so the diagonal stays dominant).
    With the integrator's scaling ``E J D`` the diagonal is ``1 + dt_i * sigma_i``, so
    ``M^-1 = 1 / (1 + dt_i * sigma_i)`` per cell and variable.
    """

    def __init__(self, solver, *, convective=0.5, viscous=2.0, source=True, source_eps=1e-4):
        self.solver = solver
        self.convective, self.viscous = convective, viscous
        self.source, self.source_eps = source and solver.turbulence is not None, source_eps
        self.inverse = None

    def rates(self, blk, q, geom):
        """First-order diagonal rate sigma (1/s), shape (nvar, interior), of one block."""
        solver, gas = self.solver, self.solver.gas
        w, T = gas.cons_to_prim(q)
        c = gas.sound_speed(T)
        nu_mol = gas.mu(T) / w[0]
        if solver.turbulence is not None:
            nu_mol = nu_mol + xp.maximum(q[5] / w[0], 0.0)
        nu = max(4.0 / 3.0, gas.gamma / gas.Pr) * nu_mol
        lam, lamv = 0.0, 0.0
        for d in range(3):
            Sd = geom.S[d]
            Sb = 0.5 * (Sd[sl(4, d, slice(0, -1))] + Sd[sl(4, d, slice(1, None))])
            Sm = xp.sqrt(xp.sum(Sb * Sb, axis=0))
            lam = lam + xp.abs(xp.sum(w[1:4] * Sb, axis=0)) + c * Sm
            lamv = lamv + nu * Sm ** 2 / geom.vol
        sigma = (self.convective * lam + self.viscous * lamv) / geom.vol
        rows = [sigma] * q.shape[0]
        if self.source:
            rows[5] = sigma + self._destruction_rate(q, w, T, geom)
        return xp.stack([blk.interior(r) for r in rows])

    def _destruction_rate(self, q, w, T, geom):
        """max(-dS/d(rho nu_tilde), 0) by a per-cell finite difference with frozen gradients."""
        solver = self.solver
        phi = xp.concatenate([w[1:4], T[None], (q[5] / w[0])[None]], axis=0)
        g = green_gauss(phi, geom)
        delta = self.source_eps * xp.abs(q[5]) + 1e-10 * w[0]
        q_pert = xp.concatenate([q[:5], (q[5] + delta)[None]], axis=0)
        base = solver._scalar_source(w, T, q, g, geom)
        pert = solver._scalar_source(w, T, q_pert, g, geom)
        return xp.maximum(-(pert - base) / delta, 0.0)

    def build(self, Q, dt, time=0.0, scale=None):
        solver = self.solver
        Qf = solver.apply_bcs(Q, solver.geoms, time)
        ng = solver.mesh.ng
        inverse = []
        for blk, q, geom, d in zip(solver.blocks, Qf, solver.geoms, dt):
            sigma = self.rates(blk, q, geom)
            step = blk.interior(d) if hasattr(d, "shape") and np.ndim(d) else d
            inv = 1.0 / (1.0 + step * sigma)
            inverse.append(xp.pad(inv, [(0, 0)] + [(ng, ng)] * 3, constant_values=1.0))
        self.inverse = tuple(inverse)

    def apply(self, v):
        return tuple(m * x for m, x in zip(self.inverse, v))


def _block_thomas_factor(lower, diag, upper):
    """Forward sweep of the block Thomas algorithm along axis 0 (trailing line axes independent).

    Arrays are (n, ..., nv, nv). Returns (C, inv_pivot, lower) with C_i = B_i^-1 U_i and
    B_i = D_i - L_i C_{i-1}.
    """
    if HAS_JAX:
        from jax import lax

        def step(c_prev, args):
            l, d, u = args
            inv = xp.linalg.inv(d - l @ c_prev)
            c = inv @ u
            return c, (c, inv)

        _, (c, inv) = lax.scan(step, xp.zeros_like(diag[0]), (lower, diag, upper))
        return c, inv, lower
    c, inv = np.empty_like(diag), np.empty_like(diag)
    prev = np.zeros_like(diag[0])
    for i in range(diag.shape[0]):
        inv[i] = np.linalg.inv(diag[i] - lower[i] @ prev)
        prev = c[i] = inv[i] @ upper[i]
    return c, inv, lower


def _block_thomas_solve(factor, rhs):
    """Solve with a factorisation from ``_block_thomas_factor``; ``rhs`` is (n, ..., nv)."""
    c, inv, lower = factor
    mv = lambda m, x: (m @ x[..., None])[..., 0]
    if HAS_JAX:
        from jax import lax

        def fwd(d_prev, args):
            r, l, iv = args
            d = mv(iv, r - mv(l, d_prev))
            return d, d

        _, dp = lax.scan(fwd, xp.zeros_like(rhs[0]), (rhs, lower, inv))

        def bwd(x_next, args):
            d, cc = args
            x = d - mv(cc, x_next)
            return x, x

        _, x = lax.scan(bwd, xp.zeros_like(rhs[0]), (dp, c), reverse=True)
        return x
    dp, x = np.empty_like(rhs), np.empty_like(rhs)
    prev = np.zeros_like(rhs[0])
    for i in range(rhs.shape[0]):
        prev = dp[i] = mv(inv[i], rhs[i] - mv(lower[i], prev))
    nxt = np.zeros_like(rhs[0])
    for i in range(rhs.shape[0] - 1, -1, -1):
        nxt = x[i] = dp[i] - mv(c[i], nxt)
    return x


_block_thomas_factor_j = jit(_block_thomas_factor)
_block_thomas_solve_j = jit(_block_thomas_solve)


def _flux_jacobian(gas, nv):
    """f(q, S) -> d(S . F_euler)/dq, mapped over the leading axes of q (..., nv) and S (..., 3).

    The row nv=6 (rho nu_tilde) is transported with the mass flux.
    """
    def flux(q, S):
        rho = q[0]
        u = q[1:4] / rho
        p = (gas.gamma - 1.0) * (q[4] - 0.5 * rho * xp.sum(u * u))
        un = xp.sum(u * S)
        parts = [rho * un, q[1:4] * un + p * S, (q[4] + p) * un]
        if nv == 6:
            parts.append(q[5] * un)
        return xp.concatenate([xp.reshape(x, (-1,)) for x in parts])

    import jax
    jac = jax.jacfwd(flux)
    return lambda q, S: jax.vmap(jac)(q.reshape(-1, nv), S.reshape(-1, 3)).reshape(q.shape[:-1] + (nv, nv))


class LineImplicit(PointJacobi):
    """Line-implicit block preconditioner: a block-tridiagonal solve along one grid axis.

    The matrix is the first-order Jacobian of a Rusanov flux with a scalar viscous term, scaled
    like the integrator's ``E J D`` system (``scale`` conjugates the blocks, ``dt_i/V_i`` multiplies
    row i). With ``A = d(S.F)/dq`` from ``jacfwd`` of the Euler flux, per face
    ``cL = 0.5 A(q_L) + (0.5 lam + a) I`` and ``cR = 0.5 A(q_R) - (0.5 lam + a) I``,
    ``lam = |u_n| + c|S|``, ``a = nu |S|^2 / V_f``, and per cell
    ``D_i = I + dt_i/V_i sum_d (cL_{d,i+1/2} - cR_{d,i-1/2})`` over the active axes (more than one
    cell and a state that varies along it), ``L_i = -dt_i/V_i cL_{i-1/2}``, ``U_i = dt_i/V_i cR_{i+1/2}`` along the line axis. The
    nu_tilde diagonal also gets the Spalart-Allmaras destruction rate. Couplings through block
    faces are dropped (additive Schwarz), so the solve is local to a block (no halo exchange).

    ``axis``: ``"auto"`` picks, per block, the active axis with the largest mean diagonal
    contribution (the wall-normal one on a stretched boundary-layer grid), an int 0..2,
    ``"all"``: every active axis, stiffest first, as an approximate factorisation.
    Memory: three nv x nv matrices per cell.
    """

    def __init__(self, solver, *, axis="auto", source=True, source_eps=1e-4):
        super().__init__(solver, source=source, source_eps=source_eps)
        self.axis_set = None
        if isinstance(axis, (tuple, list)):
            if not axis or any(a not in (0, 1, 2) for a in axis):
                raise ValueError("axis tuple must contain axes 0..2")
            self.axis_set, axis = tuple(axis), "all"
        elif axis not in ("auto", "all", 0, 1, 2):
            raise ValueError("axis must be 'auto', 'all', 0, 1, 2 or a tuple of axes")
        self.axis = axis
        self.axes = None
        self.lines = None

    def _face_coeffs(self, blk, geom, q, w, T, d):
        """(cL, cR) as (n0, n1, n2, nv, nv) blocks on the n_d+1 interior faces of axis d."""
        gas, ng, nv = self.solver.gas, blk.ng, q.shape[0]
        c = gas.sound_speed(T)
        nu_mol = gas.mu(T) / w[0]
        if self.solver.turbulence is not None:
            nu_mol = nu_mol + xp.maximum(q[5] / w[0], 0.0)
        nu_mol = max(4.0 / 3.0, gas.gamma / gas.Pr) * nu_mol
        left, right = (lambda a: blk.slab(a, d, -1, 0)), (lambda a: blk.slab(a, d, 0, 1))
        idx = [slice(ng, ng + n) for n in blk.n]
        idx[d] = slice(ng, ng + blk.n[d] + 1)
        Sf = geom.S[d][(slice(None),) + tuple(idx)]
        un = xp.sum(0.5 * (left(w[1:4]) + right(w[1:4])) * Sf, axis=0)
        Sm = xp.sqrt(xp.sum(Sf * Sf, axis=0))
        lam = xp.abs(un) + 0.5 * (left(c) + right(c)) * Sm
        a = 0.5 * (left(nu_mol) + right(nu_mol)) * Sm ** 2 / (0.5 * (left(geom.vol) + right(geom.vol)))
        S_last = xp.moveaxis(Sf, 0, -1)
        jac = self._jacobian
        AL = jac(xp.moveaxis(left(q), 0, -1), S_last)
        AR = jac(xp.moveaxis(right(q), 0, -1), S_last)
        eye = xp.eye(nv)
        dissipation = (0.5 * lam + a)[..., None, None] * eye
        return 0.5 * AL + dissipation, 0.5 * AR - dissipation

    def _blocks(self, blk, geom, q, step, ratio):
        """Active-axis blocks: {d: (cL, cR, diagonal contribution)} with the dt/V scaling applied."""
        w, T = self.solver.gas.cons_to_prim(q)
        scale = (step / blk.interior(geom.vol))
        scale = scale[..., None, None] if np.ndim(scale) else scale
        inner = blk.interior(q)
        qmax = float(xp.max(xp.abs(inner)))
        axes = [d for d in range(3) if blk.n[d] >= 2 and (self.axis_set is None or d in self.axis_set)]
        # a state that is uniform along d (2D case on a thin slab) has no flux gradient there;
        # if every axis is uniform (free-stream block) all of them are kept
        varying = [d for d in axes
                   if float(xp.max(xp.abs(inner - xp.take(inner, xp.asarray([0]), axis=d + 1)))) > 1e-12 * qmax]
        parts = {}
        for d in (varying or axes):
            cL, cR = self._face_coeffs(blk, geom, q, w, T, d)
            cL, cR = cL * ratio, cR * ratio
            hi, lo = [slice(None)] * 3, [slice(None)] * 3
            hi[d], lo[d] = slice(1, None), slice(None, -1)
            parts[d] = (cL, cR, (cL[tuple(hi)] - cR[tuple(lo)]) * scale)
        return parts, scale, w, T

    def build(self, Q, dt, time=0.0, scale=None):
        solver = self.solver
        Qf = solver.apply_bcs(Q, solver.geoms, time)
        nv = Qf[0].shape[0]
        if not hasattr(self, "_jacobian"):
            self._jacobian = jit(_flux_jacobian(solver.gas, nv))
        eye = xp.eye(nv)
        lines = []
        for ib, (blk, q, geom, dtb) in enumerate(zip(solver.blocks, Qf, solver.geoms, dt)):
            step = blk.interior(dtb) if np.ndim(dtb) else dtb
            s = xp.ones(nv) if scale is None else scale[ib][:, 0, 0, 0]
            ratio = s[None, :] / s[:, None]            # (S^-1 A S)_kl = A_kl s_l / s_k
            parts, dt_over_v, w, T = self._blocks(blk, geom, q, step, ratio)
            if not parts:
                raise ValueError("line-implicit preconditioner needs a block with at least 2 cells in some axis")
            stiff = {k: float(xp.mean(xp.trace(p[2], axis1=-2, axis2=-1))) for k, p in parts.items()}
            ranked = sorted(parts, key=stiff.get, reverse=True)
            if self.axis == "auto":
                chosen = ranked[:1]
            elif self.axis == "all":
                chosen = ranked
            elif self.axis in parts:
                chosen = [self.axis]
            else:
                raise ValueError(f"block {blk.name}: axis {self.axis} has fewer than 2 cells")
            extra = 0.0
            if self.source:
                rate = step * blk.interior(self._destruction_rate(q, w, T, geom))
                extra = rate[..., None, None] * _unit(nv, 5)
            entries = []
            for n, d in enumerate(chosen):
                # one axis: the other axes sit on the diagonal; several axes: approximate
                # factorisation (I + A_x)(I + A_z), each factor carries only its own axis
                others = sum(p[2] for k, p in parts.items()) if len(chosen) == 1 else parts[d][2]
                diag = eye + others + (extra if n == 0 else 0.0)
                cL, cR, _ = parts[d]
                hi, lo = [slice(None)] * 3, [slice(None)] * 3
                hi[d], lo[d] = slice(1, None), slice(None, -1)
                lower = -dt_over_v * cL[tuple(lo)]
                upper = dt_over_v * cR[tuple(hi)]
                first, last = [slice(None)] * 3, [slice(None)] * 3
                first[d], last[d] = slice(0, 1), slice(-1, None)
                lower = set_at(lower, tuple(first), 0.0)
                upper = set_at(upper, tuple(last), 0.0)
                move = lambda a, d=d: xp.moveaxis(a, d, 0)
                entries.append((d, _block_thomas_factor_j(move(lower), move(diag), move(upper))))
            lines.append(tuple(entries))
        self.lines = tuple(lines)
        self.axes = tuple(tuple(d for d, _ in entries) for entries in lines)

    def apply(self, v):
        ng = self.solver.mesh.ng
        out = []
        for blk, x, entries in zip(self.solver.blocks, v, self.lines):
            sol = blk.interior(x)
            for d, factor in entries:
                inner = xp.moveaxis(xp.moveaxis(sol, 0, -1), d, 0)
                sol = xp.moveaxis(xp.moveaxis(_block_thomas_solve_j(factor, inner), 0, d), -1, 0)
            out.append(xp.pad(sol, [(0, 0)] + [(ng, ng)] * 3))
        return tuple(out)


def _unit(nv, k):
    """nv x nv matrix with a single 1 at (k, k)."""
    m = np.zeros((nv, nv))
    m[k, k] = 1.0
    return xp.asarray(m)
