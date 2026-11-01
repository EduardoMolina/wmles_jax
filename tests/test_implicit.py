"""Steady implicit time-integration formulation. Run: python -m unittest tests.test_implicit"""
import unittest

import numpy as np

from wmles.gas import Gas
from wmles.implicit import ImplicitIntegrator


class _Block:
    ng = 1

    @staticmethod
    def interior(a):
        return a[(slice(None),) + (slice(1, -1),) * 3]


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
            _state(), 3, 0.1, growth=2.0, max_cfl=0.3)
        np.testing.assert_allclose([row[1] for row in history], [0.1, 0.2, 0.3])
        np.testing.assert_allclose([row[2] for row in history], [0.01, 0.02, 0.03])


if __name__ == "__main__":
    unittest.main()
