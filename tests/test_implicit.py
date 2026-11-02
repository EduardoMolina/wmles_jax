"""Steady implicit time-integration formulation. Run: python -m unittest tests.test_implicit"""
import unittest

import numpy as np

from wmles import implicit
from wmles.gas import Gas
from wmles.implicit import ImplicitIntegrator
from wmles import precond
from wmles.precond import LineImplicit, PointJacobi
from wmles.rans_case import build_rans_case, initial_state
from wmles.schemes import MUSCLHLLC


class _Block:
    ng = 1
    n = (1, 1, 1)

    @staticmethod
    def interior(a):
        return a[(Ellipsis,) + (slice(1, -1),) * 3]


class _Mesh:
    ng = 1


class _LinearSolver:
    def __init__(self, coefficient=1.0):
        self.blocks = (_Block(),)
        self.mesh = _Mesh()
        self.geoms = (None,)
        self.gas = Gas()
        self.coefficient = coefficient
        self.cfl = 0.8

    def rhs(self, Q, time=0.0):
        return tuple(-self.coefficient * q for q in Q)

    def max_dt(self, Q, time=0.0):
        return 0.08

    def local_dt(self, Q, time=0.0):
        """Per-cell dt (zero ghost padding like ``Solver.local_dt``); cell-dependent."""
        out = []
        for q in Q:
            dt = np.zeros(q.shape[1:])
            dt[1:-1, 1:-1, 1:-1] = 0.08 * (1.0 + np.arange(q[0, 1:-1, 1:-1, 1:-1].size)
                                           .reshape(q[0, 1:-1, 1:-1, 1:-1].shape))
            out.append(dt)
        return tuple(out)

    @staticmethod
    def apply_bcs(Q, geoms, time=0.0):
        return Q


def _state():
    q = np.zeros((5, 3, 3, 3))
    q[0] = 1.0
    q[4] = 250_000.0
    return (q,)


class ImplicitFormulationTests(unittest.TestCase):
    def test_backward_euler_residual_uses_per_volume_rhs(self):
        solver = _LinearSolver(coefficient=2.0)
        integrator = ImplicitIntegrator(solver)
        qn = _state()
        q = (qn[0] * 1.2,)
        residual = integrator.residual(q, qn, 0.5)[0]
        factor = (1.2 - 1.0) / 0.5 + 2.0 * 1.2
        np.testing.assert_allclose(residual[:, 1, 1, 1], qn[0][:, 1, 1, 1] * factor)
        self.assertEqual(float(np.abs(residual[:, 0]).max()), 0.0)

    def test_bdf2_residual_and_previous_state_requirement(self):
        solver = _LinearSolver(coefficient=2.0)
        integrator = ImplicitIntegrator(solver, scheme="bdf2")
        qn = _state()
        previous = (qn[0] * 0.8,)
        q = (qn[0] * 1.2,)
        with self.assertRaisesRegex(ValueError, "previous"):
            integrator.residual(q, qn, 0.5)
        residual = integrator.residual(q, qn, 0.5, previous=previous)[0]
        factor = (1.5 * 1.2 - 2.0 + 0.5 * 0.8) / 0.5 + 2.0 * 1.2
        np.testing.assert_allclose(residual[:, 1, 1, 1], qn[0][:, 1, 1, 1] * factor)

    def test_backward_euler_newton_gmres_step(self):
        solver = _LinearSolver(coefficient=2.0)
        integrator = ImplicitIntegrator(solver, newton_rtol=1e-10, gmres_rtol=1e-10)
        qn = _state()
        q = integrator.advance(qn, 0.25)
        np.testing.assert_allclose(q[0][:, 1, 1, 1], qn[0][:, 1, 1, 1] / 1.5, rtol=1e-8)
        self.assertLess(integrator.last_residual_norm, 1e-7)

    def test_steady_driver_ramps_cfl_and_scales_dt_from_solver_limit(self):
        solver = _LinearSolver(coefficient=2.0)
        integrator = ImplicitIntegrator(
            solver, newton_rtol=1e-10, gmres_rtol=1e-10)
        _, history = integrator.run_steady_cfl(
            _state(), 3, 0.1, growth=2.0, max_cfl=0.3, local=False, ramp="geometric")
        np.testing.assert_allclose([row[1] for row in history], [0.1, 0.2, 0.3])
        np.testing.assert_allclose([row[2] for row in history], [0.01, 0.02, 0.03])

    def test_uniform_local_dt_equals_scalar_dt(self):
        solver = _LinearSolver(coefficient=2.0)
        integrator = ImplicitIntegrator(solver, newton_rtol=1e-10, gmres_rtol=1e-10)
        qn = _state()
        uniform = (0.25 + 0.0 * qn[0][0],)
        a, b = integrator.advance(qn, uniform), integrator.advance(qn, 0.25)
        np.testing.assert_allclose(a[0], b[0], rtol=1e-12, atol=0.0)
        r1 = integrator.residual((qn[0] * 1.1,), qn, uniform)[0]
        r2 = integrator.residual((qn[0] * 1.1,), qn, 0.25)[0]
        np.testing.assert_allclose(r1, r2, rtol=1e-13)

    def test_local_dt_step_solves_each_cell_with_its_own_dt(self):
        solver = _LinearSolver(coefficient=2.0)
        integrator = ImplicitIntegrator(solver, newton_rtol=1e-10, gmres_rtol=1e-10)
        qn = _state()
        dts = solver.local_dt(qn)
        q = integrator.advance(qn, dts)[0]
        expected = qn[0][:, 1:-1, 1:-1, 1:-1] / (1.0 + 2.0 * dts[0][1:-1, 1:-1, 1:-1])
        np.testing.assert_allclose(q[:, 1:-1, 1:-1, 1:-1], expected, rtol=1e-8)

    def test_local_dt_validation_and_bdf2_rejection(self):
        solver = _LinearSolver()
        qn = _state()
        bad = (np.zeros_like(qn[0][0]),)
        with self.assertRaises(ValueError):
            ImplicitIntegrator(solver).residual(qn, qn, bad)
        with self.assertRaisesRegex(ValueError, "local"):
            ImplicitIntegrator(solver, scheme="bdf2").residual(qn, qn, solver.local_dt(qn), previous=qn)

    def test_local_ser_ramp_grows_cfl_and_caps_it(self):
        solver = _LinearSolver(coefficient=2.0)
        integrator = ImplicitIntegrator(solver, newton_rtol=1e-10, gmres_rtol=1e-10)
        _, history = integrator.run_steady_cfl(_state(), 4, 0.1, growth=2.0, max_cfl=0.5)
        cfls = [row[1] for row in history]
        self.assertEqual(cfls[0], 0.1)
        self.assertTrue(all(b >= a - 1e-15 for a, b in zip(cfls, cfls[1:])))
        self.assertLessEqual(max(cfls), 0.5)
        self.assertGreater(cfls[-1], cfls[0])

    def test_reductions_go_through_solver_reduce_and_match_local_sum(self):
        class Sharded(_LinearSolver):
            calls = 0

            def _reduce(self, values, op):
                Sharded.calls += 1
                return op(np.stack(values), axis=0)

        rng = np.random.default_rng(0)
        a, b = (rng.normal(size=(5, 3, 3, 3)),), (rng.normal(size=(5, 3, 3, 3)),)
        plain, sharded = _LinearSolver(), Sharded()
        self.assertAlmostEqual(implicit._inner(plain, a, b), implicit._inner(sharded, a, b))
        many = implicit._inner_many(sharded, [a, b], a)
        np.testing.assert_allclose(many, [implicit._inner(plain, a, a), implicit._inner(plain, b, a)])
        self.assertGreater(Sharded.calls, 0)
        x = np.zeros((5, 3, 3, 3))
        x[2, 1, 1, 1] = -7.0
        self.assertEqual(implicit._max_abs(sharded, (x,)), 7.0)

    def test_gmres_right_preconditioner_reduces_iterations(self):
        solver = _LinearSolver()
        diag = np.zeros((5, 3, 3, 3))
        diag[:, 1:-1, 1:-1, 1:-1] = np.logspace(0, 4, 5)[:, None, None, None]
        matvec = lambda v: (diag * v[0],)
        b = (np.zeros((5, 3, 3, 3)),)
        b[0][:, 1, 1, 1] = 1.0
        inv = np.where(diag > 0.0, 1.0 / np.where(diag > 0.0, diag, 1.0), 0.0)
        integrator = ImplicitIntegrator(solver, gmres_rtol=1e-10, gmres_maxiter=50)
        x0, n0 = integrator._gmres(matvec, b)
        x1, n1 = integrator._gmres(matvec, b, lambda v: (inv * v[0],))
        for x in (x0, x1):
            np.testing.assert_allclose(diag * x[0], b[0], atol=1e-8)
        self.assertLess(n1, n0)
        self.assertEqual(n1, 1)

    def test_point_jacobi_is_identity_at_zero_dt_and_contracts_with_dt(self):
        _, solver, props = build_rans_case(
            "tests/data/2DZP/flatplate_clust2_3levelsdown_69x49", mach=0.2, reynolds=5e6,
            scheme=MUSCLHLLC())
        Q = initial_state(solver, props, profile="flatplate")
        pc = PointJacobi(solver)
        tiny = implicit._per_block_dt(solver, 1e-30)
        pc.build(Q, tiny)
        for m in pc.inverse:
            np.testing.assert_allclose(np.asarray(m), 1.0)
        big = implicit._per_block_dt(solver, 1e-4)
        pc.build(Q, big)
        for blk, m in zip(solver.blocks, pc.inverse):
            inner = np.asarray(blk.interior(m))
            self.assertTrue(np.all(np.isfinite(inner)))
            self.assertTrue(np.all((inner > 0.0) & (inner < 1.0)))


    def test_block_thomas_matches_dense_solve(self):
        rng = np.random.default_rng(3)
        n, nv, lines = 6, 3, (2, 2)
        shape = (n,) + lines + (nv, nv)
        lower, upper = rng.normal(size=shape) * 0.3, rng.normal(size=shape) * 0.3
        diag = rng.normal(size=shape) * 0.2 + 3.0 * np.eye(nv)
        lower[0] = 0.0
        upper[-1] = 0.0
        rhs = rng.normal(size=(n,) + lines + (nv,))
        x = np.asarray(precond._block_thomas_solve(precond._block_thomas_factor(lower, diag, upper), rhs))
        for a in range(lines[0]):
            for b in range(lines[1]):
                dense = np.zeros((n * nv, n * nv))
                for i in range(n):
                    dense[i * nv:(i + 1) * nv, i * nv:(i + 1) * nv] = diag[i, a, b]
                    if i > 0:
                        dense[i * nv:(i + 1) * nv, (i - 1) * nv:i * nv] = lower[i, a, b]
                    if i < n - 1:
                        dense[i * nv:(i + 1) * nv, (i + 1) * nv:(i + 2) * nv] = upper[i, a, b]
                ref = np.linalg.solve(dense, rhs[:, a, b].reshape(-1)).reshape(n, nv)
                np.testing.assert_allclose(x[:, a, b], ref, rtol=1e-9, atol=1e-10)

    def test_line_implicit_identity_at_zero_dt_and_finite_at_large_dt(self):
        _, solver, props = build_rans_case(
            "tests/data/2DZP/flatplate_clust2_3levelsdown_69x49", mach=0.2, reynolds=5e6,
            scheme=MUSCLHLLC())
        Q = initial_state(solver, props, profile="flatplate")
        rng = np.random.default_rng(0)
        v = implicit._interior_vector(solver, tuple(rng.normal(size=q.shape) for q in Q))
        pc = LineImplicit(solver, axis=(2, 1))
        pc.build(Q, implicit._per_block_dt(solver, 1e-30))
        for x, y in zip(pc.apply(v), v):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-9)
        pc.build(Q, implicit._per_block_dt(solver, 1e-4))
        out = pc.apply(v)
        self.assertEqual(pc.axes[1], (2, 1))
        for blk, x in zip(solver.blocks, out):
            self.assertTrue(np.all(np.isfinite(np.asarray(blk.interior(x)))))
        with self.assertRaises(ValueError):
            LineImplicit(solver, axis=3)


if __name__ == "__main__":
    unittest.main()
