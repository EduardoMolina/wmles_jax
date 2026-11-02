"""Matrix-free implicit time integration for steady finite-volume RANS."""

import numpy as np

from .backend import xp


def _interior_vector(solver, vector):
    """Keep only cell-centred unknowns; ghost values are supplied by the BCs."""
    ng = solver.mesh.ng
    return tuple(
        xp.pad(blk.interior(v), [(0, 0)] + [(ng, ng)] * 3)
        for blk, v in zip(solver.blocks, vector)
    )


def _per_block_dt(solver, dt):
    """Scalar dt, or a per-block tuple of per-cell arrays (e.g. ``Solver.local_dt``).

    Returns a tuple with one entry per block. Array entries keep their interior values and
    replicate them into the ghost layers (``Solver.local_dt`` pads with zeros, which would
    divide by zero in the pseudo-time term).
    """
    if not isinstance(dt, (tuple, list)):
        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        return (dt,) * len(solver.blocks)
    if len(dt) != len(solver.blocks):
        raise ValueError("per-cell dt needs one array per block")
    ng, out = solver.mesh.ng, []
    for blk, d in zip(solver.blocks, dt):
        inner = blk.interior(d)
        if not np.all(np.isfinite(np.asarray(inner))) or float(np.min(np.asarray(inner))) <= 0.0:
            raise ValueError("per-cell dt must be positive and finite")
        out.append(xp.pad(inner, [(ng, ng)] * 3, mode="edge"))
    return tuple(out)


def _add(a, b, scale=1.0):
    return tuple(x + scale * y for x, y in zip(a, b))


def _inner(solver, a, b):
    value = sum(xp.sum(blk.interior(x) * blk.interior(y))
                for blk, x, y in zip(solver.blocks, a, b))
    return float(value)


def _norm(solver, a):
    return np.sqrt(max(_inner(solver, a, a), 0.0))


def _scale(a, value):
    return tuple(value * x for x in a)


def _scale_rows(a, rows):
    """Multiply every block array by per-variable factors of shape (nvar, 1, 1, 1)."""
    return tuple(rows * x for x in a)


def _mul(a, b):
    return tuple(x * y for x, y in zip(a, b))


class ImplicitIntegrator:
    """Backward-Euler/BDF2 Newton-GMRES integrator and steady pseudo-time driver.

    Unknowns and residuals use the solver's conserved, per-volume state convention.
    Krylov vectors contain interior cells only; every residual evaluation refreshes
    ghost cells through ``Solver.rhs`` and its boundary-condition machinery.
    """

    def __init__(self, solver, *, scheme="bdf1", newton_rtol=1e-6, newton_atol=1e-10,
                 gmres_rtol=1e-3, gmres_maxiter=30, gmres_restart=20,
                 max_newton=8, fd_epsilon=1e-7, max_backtracks=10):
        if scheme not in ("bdf1", "bdf2"):
            raise ValueError("scheme must be 'bdf1' or 'bdf2'")
        if gmres_maxiter < 1 or gmres_restart < 1 or max_newton < 1:
            raise ValueError("iteration limits must be positive")
        if fd_epsilon <= 0.0:
            raise ValueError("fd_epsilon must be positive")
        self.solver = solver
        self.scheme = scheme
        self.newton_rtol, self.newton_atol = newton_rtol, newton_atol
        self.gmres_rtol = gmres_rtol
        self.gmres_maxiter, self.gmres_restart = gmres_maxiter, gmres_restart
        self.max_newton, self.fd_epsilon = max_newton, fd_epsilon
        self.max_backtracks = max_backtracks
        self.last_newton_iterations = 0
        self.last_residual_norm = np.inf
        self.last_gmres_iterations = 0

    def residual(self, Q, Qn, dt, *, previous=None, time=0.0):
        """Return the backward-Euler or fixed-step BDF2 residual.

        ``dt`` is a scalar or, for steady local pseudo-time stepping, a per-block tuple of
        per-cell arrays (backward Euler only).
        BDF2 requires ``previous`` to be the state one step before ``Qn``. The
        returned residual is nonzero only in block interiors.
        """
        if self.scheme == "bdf2" and previous is None:
            raise ValueError("BDF2 requires the state from the previous time step")
        if self.scheme == "bdf2" and isinstance(dt, (tuple, list)):
            raise ValueError("BDF2 does not support local (per-cell) time steps")
        return self._residual(Q, Qn, _per_block_dt(self.solver, dt), previous, time)

    def _residual(self, Q, Qn, dts, previous, time):
        """Residual with ``dts`` already prepared by ``_per_block_dt``."""
        rhs = self.solver.rhs(Q, time)
        if self.scheme == "bdf1":
            coeffs = (1.0, -1.0)
            states = (Q, Qn)
        else:
            coeffs = (1.5, -2.0, 0.5)
            states = (Q, Qn, previous)
        residual = tuple(sum(c * state[i] for c, state in zip(coeffs, states)) / d - r
                         for i, (r, d) in enumerate(zip(rhs, dts)))
        return _interior_vector(self.solver, residual)

    def _row_scales(self, Q):
        """Per-variable reference magnitudes, shape (nvar, 1, 1, 1), for diagonal scaling.

        The conserved variables differ by many orders of magnitude (rho*E ~ 1e5 against
        rho*nu_tilde ~ 1e-3), so unscaled L2 norms and finite-difference perturbations would
        ignore the small rows. Scales: rho, a common momentum scale sqrt(rms(rho) rms(rho E)),
        rho*E, and rms of any extra row (floored). Frozen over one Newton solve.
        """
        nv = Q[0].shape[0]
        count = sum(int(np.prod(blk.n)) for blk in self.solver.blocks)
        rms = np.array([np.sqrt(sum(float(np.sum(np.asarray(blk.interior(q[k])) ** 2))
                                    for blk, q in zip(self.solver.blocks, Q)) / count)
                        for k in range(nv)])
        scale = np.empty(nv)
        scale[0], scale[4] = rms[0], rms[4]
        scale[1:4] = np.sqrt(rms[0] * rms[4])
        floor = 1e-8 * rms[0]
        scale[5:] = np.maximum(rms[5:], floor)
        if not np.all(np.isfinite(scale)) or np.any(scale <= 0.0):
            scale = np.ones(nv)
        return xp.asarray(scale)[:, None, None, None]

    def _gmres(self, matvec, b):
        """Restarted GMRES on tuples of block arrays.

        Inexact: when the relative tolerance is not reached within ``gmres_maxiter`` the best
        iterate is returned (``last_linear_converged`` is False); the Newton line search then
        decides whether the step is usable.
        """
        solver = self.solver
        x = tuple(xp.zeros_like(v) for v in b)
        bnorm = _norm(solver, b)
        self.last_linear_converged = True
        if bnorm == 0.0:
            return x, 0
        total = 0
        while total < self.gmres_maxiter:
            r = _add(b, matvec(x), -1.0)
            beta = _norm(solver, r)
            if beta <= self.gmres_rtol * bnorm:
                return x, total
            basis = [_scale(r, 1.0 / beta)]
            h = np.zeros((self.gmres_restart + 1, self.gmres_restart))
            e = np.zeros(self.gmres_restart + 1)
            e[0] = beta
            for j in range(min(self.gmres_restart, self.gmres_maxiter - total)):
                v = _interior_vector(solver, matvec(basis[j]))
                for i in range(j + 1):
                    h[i, j] = _inner(solver, basis[i], v)
                    v = _add(v, basis[i], -h[i, j])
                h[j + 1, j] = _norm(solver, v)
                if h[j + 1, j] > 0.0:
                    basis.append(_scale(v, 1.0 / h[j + 1, j]))
                y, *_ = np.linalg.lstsq(h[:j + 2, :j + 1], e[:j + 2], rcond=None)
                candidate = x
                for k, coeff in enumerate(y):
                    candidate = _add(candidate, basis[k], coeff)
                total += 1
                estimate = np.linalg.norm(e[:j + 2] - h[:j + 2, :j + 1] @ y)
                if estimate <= self.gmres_rtol * bnorm or h[j + 1, j] == 0.0:
                    true_norm = _norm(solver, _add(b, matvec(candidate), -1.0))
                    if true_norm <= self.gmres_rtol * bnorm:
                        return candidate, total
                    x = candidate
                    break
                if total >= self.gmres_maxiter:
                    x = candidate
                    break
            else:
                x = candidate
        true_norm = _norm(solver, _add(b, matvec(x), -1.0))
        self.last_linear_converged = true_norm <= self.gmres_rtol * bnorm
        return x, total

    def _admissible(self, Q):
        for blk, q in zip(self.solver.blocks, Q):
            w, _ = self.solver.gas.cons_to_prim(blk.interior(q))
            if not np.all(np.isfinite(np.asarray(w))):
                return False
            if np.any(np.asarray(w[0]) <= 0.0) or np.any(np.asarray(w[4]) <= 0.0):
                return False
        return True

    def advance(self, Qn, dt, *, previous=None, time=0.0):
        """Advance one implicit step, using BDF2 when requested and available.

        ``dt``: scalar, or a per-block tuple of per-cell arrays for steady local pseudo-time
        stepping (backward Euler only). The field is fixed during the Newton solve so the
        finite-difference J*v differentiates the same residual that is being solved.
        Newton, GMRES and the line search work on the scaled system
        ``E J D y = -E R`` with ``D = diag(scale)`` (``_row_scales``: every conserved variable
        carries weight) and ``E = diag(dt / scale)`` (equations multiplied by the local pseudo-time
        step, which removes the 1/dt_i spread of the diagonal and keeps GMRES well conditioned).
        """
        if self.scheme == "bdf2" and previous is None:
            raise ValueError("BDF2 requires the state from the previous time step")
        if self.scheme == "bdf2" and isinstance(dt, (tuple, list)):
            raise ValueError("BDF2 does not support local (per-cell) time steps")
        dt = _per_block_dt(self.solver, dt)
        scale = self._row_scales(Qn)
        eq = tuple(d / scale for d in dt)
        snorm = lambda a: _norm(self.solver, _mul(eq, a))
        Q = Qn
        residual = self._residual(Q, Qn, dt, previous, time)
        initial_norm = snorm(residual)
        tolerance = self.newton_atol + self.newton_rtol * initial_norm
        self.last_gmres_iterations = 0
        for iteration in range(self.max_newton + 1):
            norm = snorm(residual)
            self.last_newton_iterations = iteration
            self.last_residual_norm = norm
            if norm <= tolerance:
                return self.solver.apply_bcs(Q, self.solver.geoms, time)
            if iteration == self.max_newton:
                break

            def matvec(y):
                ymax = max(float(np.max(np.abs(np.asarray(blk.interior(a)))))
                           for blk, a in zip(self.solver.blocks, y))
                if ymax == 0.0:
                    return tuple(xp.zeros_like(a) for a in y)
                qmax = max(max(float(np.max(np.abs(np.asarray(blk.interior(q / scale)))))
                               for blk, q in zip(self.solver.blocks, Q)), 1.0)
                eps = self.fd_epsilon * qmax / ymax
                perturbed = tuple(q + eps * s_y for q, s_y in zip(Q, _scale_rows(y, scale)))
                rp = self._residual(perturbed, Qn, dt, previous, time)
                return _mul(eq, _scale(_add(rp, residual, -1.0), 1.0 / eps))

            y, nlinear = self._gmres(matvec, _scale(_mul(eq, residual), -1.0))
            delta = _scale_rows(y, scale)
            self.last_gmres_iterations += nlinear
            accepted = False
            alpha = 1.0
            for _ in range(self.max_backtracks + 1):
                candidate = tuple(q + alpha * dq for q, dq in zip(Q, delta))
                if self._admissible(candidate):
                    candidate_residual = self._residual(candidate, Qn, dt, previous, time)
                    if snorm(candidate_residual) < norm:
                        Q, residual = candidate, candidate_residual
                        accepted = True
                        break
                alpha *= 0.5
            if not accepted:
                raise RuntimeError(
                    f"implicit Newton line search failed at iteration {iteration}; residual={norm:.6e}")
        raise RuntimeError(
            f"implicit Newton did not converge in {self.max_newton} iterations; "
            f"residual={self.last_residual_norm:.6e}")

    def run_steady(self, Q, steps, dt, *, growth=1.5, max_dt=None, callback=None, time=0.0):
        """March toward steady state with backward Euler and a geometric pseudo-time ramp."""
        if self.scheme != "bdf1":
            raise ValueError("steady pseudo-time continuation currently requires scheme='bdf1'")
        if steps < 0 or dt <= 0.0 or growth < 1.0 or (max_dt is not None and max_dt <= 0.0):
            raise ValueError("steps must be nonnegative, dt and max_dt positive, and growth at least 1")
        history = []
        current_dt = dt
        for step in range(1, steps + 1):
            Q = self.advance(Q, current_dt, time=time)
            residual = self.solver.rhs(Q, time)
            norm = _norm(self.solver, _interior_vector(self.solver, residual))
            history.append((step, current_dt, norm))
            if callback is not None:
                callback(step, Q, history[-1])
            current_dt = min(current_dt * growth, max_dt) if max_dt is not None else current_dt * growth
        return Q, history

    def _pseudo_dt(self, Q, cfl, local, max_dt, time):
        """Pseudo-time step at ``cfl``: a per-cell field (local) or the global minimum.

        ``Solver.local_dt`` / ``max_dt`` already include ``solver.cfl``, so they are rescaled.
        Recomputed from the current state once per step, then frozen for the Newton solve.
        """
        ratio = cfl / float(self.solver.cfl)
        if not local:
            dt = float(self.solver.max_dt(Q, time)) * ratio
            return dt if max_dt is None else min(dt, max_dt)
        dts = tuple(d * ratio for d in self.solver.local_dt(Q, time))
        if max_dt is not None:
            dts = tuple(xp.minimum(d, max_dt) for d in dts)
        return dts

    def _min_dt(self, dt):
        if not isinstance(dt, (tuple, list)):
            return dt
        return min(float(np.min(np.asarray(blk.interior(d)))) for blk, d in zip(self.solver.blocks, dt))

    def run_steady_cfl(self, Q, steps, cfl, *, growth=1.5, max_cfl=None, local=True,
                       ramp="ser", ser_exponent=1.0, max_dt=None, max_retries=5,
                       callback=None, time=0.0):
        """Steady backward-Euler pseudo-time continuation driven by a CFL ramp.

        ``local=True`` (default) gives every cell its own pseudo-time step
        ``dt_i = cfl * dt_i(explicit, cfl=1)``; it is not time accurate and only the steady
        state is meaningful. ``local=False`` uses one global step from the smallest cell.

        ``ramp="ser"``: switched evolution relaxation, ``cfl = cfl0 (|R0|/|R|)^ser_exponent``
        limited to ``growth`` per step and ``max_cfl``. ``ramp="geometric"``: ``cfl *= growth``.
        When a Newton solve fails the step is retried from the same state with half the CFL, at
        most ``max_retries`` times. ``max_dt`` optionally caps the pseudo-time step.
        History rows: (step, cfl, smallest dt, L2 norm of the rhs).
        """
        if self.scheme != "bdf1":
            raise ValueError("steady pseudo-time continuation currently requires scheme='bdf1'")
        if steps < 0 or cfl <= 0.0 or growth < 1.0:
            raise ValueError("steps must be nonnegative, cfl positive, and growth at least 1")
        if max_cfl is not None and (max_cfl <= 0.0 or max_cfl < cfl):
            raise ValueError("max_cfl must be positive and at least the initial cfl")
        if ramp not in ("ser", "geometric"):
            raise ValueError("ramp must be 'ser' or 'geometric'")
        if max_dt is not None and max_dt <= 0.0:
            raise ValueError("max_dt must be positive")
        solver_cfl = float(self.solver.cfl)
        if not np.isfinite(solver_cfl) or solver_cfl <= 0.0:
            raise ValueError("solver.cfl must be finite and positive to derive a CFL-based dt")

        def rhs_norm(state):
            return _norm(self.solver, _interior_vector(self.solver, self.solver.rhs(state, time)))

        history, cfl0, current_cfl = [], cfl, cfl
        norm0 = rhs_norm(Q) if ramp == "ser" else None
        for step in range(1, steps + 1):
            for attempt in range(max_retries + 1):
                dt = self._pseudo_dt(Q, current_cfl, local, max_dt, time)
                try:
                    Qnew = self.advance(Q, dt, time=time)
                    break
                except RuntimeError:
                    if attempt == max_retries:
                        raise
                    current_cfl *= 0.5
            Q = Qnew
            norm = rhs_norm(Q)
            history.append((step, current_cfl, self._min_dt(dt), norm))
            if callback is not None:
                callback(step, Q, history[-1])
            limit = current_cfl * growth if max_cfl is None else min(current_cfl * growth, max_cfl)
            if ramp == "ser":
                target = cfl0 * (norm0 / max(norm, 1e-300)) ** ser_exponent
                current_cfl = min(max(target, cfl0), limit)
            else:
                current_cfl = limit
        return Q, history
