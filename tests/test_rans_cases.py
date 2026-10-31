"""RANS case builders and post-processing.  Run:  python -m unittest tests.test_rans_cases"""
import unittest

import numpy as np

from wmles.backend import xp
from wmles.rans_case import build_rans_case, initial_state, run_steady
from wmles import rans_post as rp

PLATE = "tests/data/2DZP/flatplate_clust2_3levelsdown_69x49"
WAKE = "tests/data/2DANW/nak_a_ff_500c_2zn_L5"


class FlatPlateCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh, cls.solver, cls.props = build_rans_case(PLATE, mach=0.2, reynolds=5e6)

    def test_topology_and_wall(self):
        walls = [(b.name, f) for b in self.mesh.blocks for f, bc in b.faces.items()
                 if getattr(bc, "is_wall", False)]
        self.assertEqual(walls, [("b1_010", (2, 0))])                 # plate = kmin of the downstream block
        self.assertEqual(sum(int(np.prod(b.n)) for b in self.mesh.blocks), 4 * (12 + 56) * 48)
        self.assertAlmostEqual(self.props["rho_inf"] * self.props["U_inf"] * self.props["length"]
                               / self.props["mu_inf"], 5e6, delta=5.0)

    def test_wall_distance_is_height_above_the_plate(self):
        ib = [b.name for b in self.mesh.blocks].index("b1_010")
        blk, g = self.mesh.blocks[ib], self.mesh.geoms[ib]
        z = np.asarray(blk.interior(g.xc))[2]
        np.testing.assert_allclose(np.asarray(blk.interior(g.dist)), z, rtol=1e-10, atol=1e-14)

    def test_short_run_is_finite(self):
        Q = initial_state(self.solver, self.props)
        Q, hist = run_steady(self.solver, Q, 30, report_every=30)
        for q in Q:
            self.assertTrue(np.all(np.isfinite(np.asarray(q))))
            self.assertGreater(float(np.asarray(q[0]).min()), 0.0)
        self.assertEqual(len(hist[0]), 3)

    def test_local_dt_step_equals_global_dt_step_when_dt_is_uniform(self):
        s = self.solver
        Q = initial_state(s, self.props)
        Q, _ = run_steady(s, Q, 5, report_every=100)
        dt = 1e-8
        uniform = tuple(dt + 0.0 * q[0] for q in Q)
        a, b = s.advance_local(Q, uniform), s.advance(Q, dt)
        for qa, qb in zip(a, b):
            np.testing.assert_allclose(np.asarray(qa), np.asarray(qb), rtol=1e-12, atol=0.0)

    def test_skin_friction_of_a_linear_profile(self):
        s, props = self.solver, self.props
        G = 2.0e4
        Q = s.init_state(lambda xc: xp.stack([props["rho_inf"] + 0 * xc[0], G * xc[2], 0 * xc[0],
                                              0 * xc[0], props["p_inf"] + 0 * xc[0]]))
        sf = rp.skin_friction(s, Q, props)[("b1_010", "kmin")]
        mu = float(s.gas.mu(props["p_inf"] / (props["rho_inf"] * s.gas.R)))
        np.testing.assert_allclose(sf["tau"], mu * G, rtol=1e-8)
        np.testing.assert_allclose(sf["cf"], mu * G / (0.5 * props["rho_inf"] * props["U_inf"] ** 2), rtol=1e-8)


class PostProcessing(unittest.TestCase):
    def test_karman_schoenherr_reference(self):
        self.assertAlmostEqual(float(rp.ks_cf(1.0e4)), 1.0 / (17.08 * 16 + 25.11 * 4 + 6.012), places=15)
        self.assertAlmostEqual(float(rp.ks_cf(1.0e4)), 0.002633, places=5)

    def test_momentum_thickness_of_a_linear_profile(self):
        props = dict(U_inf=10.0, rho_inf=1.2)
        delta, n = 0.01, 4000
        y = (np.arange(n) + 0.5) * delta / n
        cols = dict(y=y[None], ut=(props["U_inf"] * y / delta)[None], rho=np.full((1, n), 1.2))
        np.testing.assert_allclose(rp.momentum_thickness(cols, props), delta / 6.0, rtol=1e-3)


class AirfoilWakeCase(unittest.TestCase):
    def test_topology_and_short_run(self):
        mesh, solver, props = build_rans_case(WAKE, mach=0.088, reynolds=1.2e6, periodic_type="symmetry_y")
        walls = [(b.name, f) for b in mesh.blocks for f, bc in b.faces.items() if getattr(bc, "is_wall", False)]
        self.assertEqual(walls, [("b2", (2, 0))])
        self.assertEqual(sum(int(np.prod(b.n)) for b in mesh.blocks), 4 * (42 * 56 + 64 * 28))
        Q, hist = run_steady(solver, initial_state(solver, props), 20, report_every=20)
        self.assertTrue(all(np.all(np.isfinite(np.asarray(q))) for q in Q))
        forces = rp.wall_forces(solver, Q, props)
        self.assertTrue(np.all(np.isfinite(forces["force"])))


if __name__ == "__main__":
    unittest.main()
