"""Steady implicit time-integration formulation. Run: python -m unittest tests.test_implicit"""
import unittest

import numpy as np

from wmles.gas import Gas
from wmles.implicit import ImplicitIntegrator


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


if __name__ == "__main__":
    unittest.main()
