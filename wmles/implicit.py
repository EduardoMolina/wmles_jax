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

        BDF2 requires ``previous`` to be the state one step before ``Qn``. The
        returned residual is nonzero only in block interiors.
        """
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.scheme == "bdf2" and previous is None:
            raise ValueError("BDF2 requires the state from the previous time step")
        rhs = self.solver.rhs(Q, time)
        if self.scheme == "bdf1":
            coeffs = (1.0, -1.0)
            states = (Q, Qn)
        else:
            coeffs = (1.5, -2.0, 0.5)
            states = (Q, Qn, previous)
        residual = tuple(sum(c * state[i] for c, state in zip(coeffs, states)) / dt - r
                         for i, r in enumerate(rhs))
        return _interior_vector(self.solver, residual)

    def _gmres(self, matvec, b):
        """Restarted GMRES on tuples of block arrays."""
        solver = self.solver
        x = tuple(xp.zeros_like(v) for v in b)
        bnorm = _norm(solver, b)
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
        if true_norm > self.gmres_rtol * bnorm:
            raise RuntimeError(
                f"GMRES did not converge in {total} iterations; "
                f"relative linear residual={true_norm / bnorm:.6e}")
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
        """Advance one implicit step, using BDF2 when requested and available."""
        if dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.scheme == "bdf2" and previous is None:
            raise ValueError("BDF2 requires the state from the previous time step")
        Q = Qn
        residual = self.residual(Q, Qn, dt, previous=previous, time=time)
        initial_norm = _norm(self.solver, residual)
        tolerance = self.newton_atol + self.newton_rtol * initial_norm
        self.last_gmres_iterations = 0
        for iteration in range(self.max_newton + 1):
            norm = _norm(self.solver, residual)
            self.last_newton_iterations = iteration
            self.last_residual_norm = norm
            if norm <= tolerance:
                return self.solver.apply_bcs(Q, self.solver.geoms, time)
            if iteration == self.max_newton:
                break

            def matvec(v):
                vnorm = _norm(self.solver, v)
                if vnorm == 0.0:
                    return tuple(xp.zeros_like(x) for x in v)
                qscale = max(max(float(np.max(np.abs(np.asarray(blk.interior(q)))))
                                 for blk, q in zip(self.solver.blocks, Q)), 1.0)
                vscale = max(float(np.max(np.abs(np.asarray(blk.interior(x)))))
                             for blk, x in zip(self.solver.blocks, v))
                eps = self.fd_epsilon * qscale / max(vscale, 1e-300)
                perturbed = tuple(q + eps * dv for q, dv in zip(Q, v))
                rp = self.residual(perturbed, Qn, dt, previous=previous, time=time)
                return _scale(_add(rp, residual, -1.0), 1.0 / eps)

            delta, nlinear = self._gmres(matvec, _scale(residual, -1.0))
            self.last_gmres_iterations += nlinear
            accepted = False
            alpha = 1.0
            for _ in range(self.max_backtracks + 1):
                candidate = tuple(q + alpha * dq for q, dq in zip(Q, delta))
                if self._admissible(candidate):
                    candidate_residual = self.residual(
                        candidate, Qn, dt, previous=previous, time=time)
                    if _norm(self.solver, candidate_residual) < norm:
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

    def run_steady_cfl(self, Q, steps, cfl, *, growth=1.5, max_cfl=None,
                       callback=None, time=0.0):
        """Ramp pseudo-time CFL, deriving each dt from the solver's explicit stability limit.

        ``solver.max_dt`` includes ``solver.cfl`` in its result, so the returned
        step is rescaled to the requested CFL. The limit is recomputed from the
        current state at every pseudo-time step.
        """
        if self.scheme != "bdf1":
            raise ValueError("steady pseudo-time continuation currently requires scheme='bdf1'")
        if steps < 0 or cfl <= 0.0 or growth < 1.0:
            raise ValueError("steps must be nonnegative, cfl positive, and growth at least 1")
        if max_cfl is not None and (max_cfl <= 0.0 or max_cfl < cfl):
            raise ValueError("max_cfl must be positive and at least the initial cfl")
        solver_cfl = float(self.solver.cfl)
        if not np.isfinite(solver_cfl) or solver_cfl <= 0.0:
            raise ValueError("solver.cfl must be finite and positive to derive a CFL-based dt")

        history = []
        current_cfl = cfl
        for step in range(1, steps + 1):
            dt = float(self.solver.max_dt(Q)) * current_cfl / solver_cfl
            if not np.isfinite(dt) or dt <= 0.0:
                raise RuntimeError(f"solver returned an invalid CFL-based time step: {dt}")
            Q = self.advance(Q, dt, time=time)
            residual = self.solver.rhs(Q, time)
            norm = _norm(self.solver, _interior_vector(self.solver, residual))
            history.append((step, current_cfl, dt, norm))
            if callback is not None:
                callback(step, Q, history[-1])
            next_cfl = current_cfl * growth
            current_cfl = min(next_cfl, max_cfl) if max_cfl is not None else next_cfl
        return Q, history
