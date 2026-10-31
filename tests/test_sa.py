"""SA coupling in the solver.  Run:  python -m unittest tests.test_sa"""
import unittest

import numpy as np

from wmles import Block, Mesh, Solver, KEP, Hybrid, Gas, NoSlipWall, Extrapolate
from wmles.backend import xp
from wmles.turbulence import SpalartAllmaras
from wmles.viscous import green_gauss, scalar_diffusion_flux
from tests.test_multiblock import build as build_multiblock, prim_fn

gas = Gas()


def nu_tilde_fn(xc):
    x, y, z = xc
    return 4.0e-5 * (1.0 + 0.4 * xp.sin(2 * np.pi * x) * xp.cos(2 * np.pi * z)) * (0.2 + y)


def with_nu_tilde(solver, Q, fn):
    return tuple(xp.concatenate([q[:5], (q[0] * fn(g.xc))[None]]) for q, g in zip(Q, solver.geoms))


def gather(solver, R, nrows):
    out = {}
    for blk, r, g in zip(solver.blocks, R, solver.geoms):
        xc = np.asarray(blk.interior(g.xc)).reshape(3, -1).T
        rr = np.asarray(blk.interior(r)).reshape(nrows, -1).T
        for c, v in zip(xc, rr):
            out[tuple(np.round(c, 8))] = v
    return out


def cartesian_block(n=(10, 6, 6), L=(1.0, 0.6, 0.6)):
    x, y, z = np.meshgrid(*(np.linspace(0, l, k + 1) for l, k in zip(L, n)), indexing="ij")
    A = Block("box", np.stack([x, y, z]))
    m = Mesh([A])
    m.connect(A, "jmin", A, "jmax", shift=(0, -L[1], 0))
    m.connect(A, "kmin", A, "kmax", shift=(0, 0, -L[2]))
    A.set_bc("imin", NoSlipWall())          # a wall is required for the wall distance
    A.set_bc("imax", Extrapolate())
    return m.finalize(), A


def prim_fn_uniform(xc):
    return xp.stack([1.2 + 0 * xc[0], 50 + 0 * xc[0], 0 * xc[0], 0 * xc[0], 1e5 + 0 * xc[0]])


class SolverSATests(unittest.TestCase):
    def test_state_has_six_rows_and_free_stream_nu_tilde(self):
        m, _ = build_multiblock("single", wm=True)
        s = Solver(m, gas, KEP(4), turbulence=SpalartAllmaras())
        Q = s.init_state(prim_fn)
        self.assertEqual(Q[0].shape[0], 6)
        w, T = gas.cons_to_prim(Q[0])
        np.testing.assert_allclose(np.asarray(Q[0][5] / Q[0][0]), np.asarray(3.0 * gas.mu(T) / w[0]), rtol=1e-13)
        R = s.rhs(Q)
        self.assertEqual(R[0].shape, Q[0].shape)
        self.assertTrue(np.all(np.isfinite(np.asarray(R[0]))))

    def test_zero_nu_tilde_reproduces_laminar_solver(self):
        m, _ = build_multiblock("single", wm=True)
        sa, ref = Solver(m, gas, KEP(4), turbulence=SpalartAllmaras()), Solver(m, gas, KEP(4), sgs=None)
        Q = with_nu_tilde(sa, sa.init_state(prim_fn), lambda xc: 0.0 * xc[0])
        R, R0 = sa.rhs(Q), ref.rhs(tuple(q[:5] for q in Q))
        scale = np.abs(np.asarray(R0[0])).max()
        err = max(np.abs(np.asarray(r[:5]) - np.asarray(r0)).max() for r, r0 in zip(R, R0)) / scale
        self.assertLess(err, 1e-13)
        self.assertEqual(float(np.abs(np.asarray(R[0][5])).max()), 0.0)

    def test_multiblock_equals_single_block(self):
        def rhs(kind, scheme):
            m, _ = build_multiblock(kind, wm=True)
            s = Solver(m, gas, scheme, turbulence=SpalartAllmaras())
            return gather(s, s.rhs(with_nu_tilde(s, s.init_state(prim_fn), nu_tilde_fn)), 6)

        for scheme in (KEP(4), Hybrid()):
            ref = rhs("single", scheme)
            scale = np.abs(np.array(list(ref.values()))).max(axis=0)
            for kind in ("split", "flip", "rot_cyc"):
                got = rhs(kind, scheme)
                self.assertEqual(ref.keys(), got.keys())
                err = max(np.max(np.abs(ref[k] - got[k]) / scale) for k in ref)
                self.assertLess(err, 1e-9, (kind, type(scheme).__name__))

    def test_scalar_diffusion_is_exact_for_quadratics(self):
        m, A = cartesian_block()
        g = m.geoms[0]
        coef = 2.5
        s = g.xc[0] ** 2
        gs = green_gauss(s[None], g)[0]
        F = scalar_diffusion_flux(A, g, 0, s, gs, coef + 0.0 * s)
        div = -(F[1:] - F[:-1]) / A.interior(g.vol)
        np.testing.assert_allclose(np.asarray(div), 2.0 * coef, rtol=1e-12)

    def test_scalar_upwind_flux_is_exact_for_linear_profiles(self):
        m, A = cartesian_block()
        g = m.geoms[0]
        s = Solver(m, gas, KEP(4), turbulence=SpalartAllmaras())
        nt = 1e-4 * (1.0 + 2.0 * g.xc[0])
        for sign in (1.0, -1.0):                      # both upwind directions
            mass = sign * 3.0 * A.slab(g.S[0], 0, 0, 1)[0]
            F = s._scalar_flux(A, g, 0, mass, nt, 0.0 * g.S[0][:, :-1], 0.0 * nt)
            div = -(F[1:] - F[:-1]) / A.interior(g.vol)
            np.testing.assert_allclose(np.asarray(div), -sign * 3.0 * 2e-4, rtol=1e-10)

    def test_no_slip_wall_mirrors_nu_tilde_with_opposite_sign(self):
        x, y, z = np.meshgrid(np.linspace(0, 1, 7), np.linspace(0, 0.3, 9), np.linspace(0, 0.5, 7),
                              indexing="ij")
        A = Block("w", np.stack([x, y, z]))
        m = Mesh([A])
        m.connect(A, "imin", A, "imax", shift=(-1.0, 0, 0))
        m.connect(A, "kmin", A, "kmax", shift=(0, 0, -0.5))
        A.set_bc("jmin", NoSlipWall())
        A.set_bc("jmax", Extrapolate())
        m.finalize()
        s = Solver(m, gas, KEP(2), turbulence=SpalartAllmaras())
        Q = s.apply_bcs(s.init_state(prim_fn_uniform), s.geoms)
        nt = np.asarray(Q[0][5] / Q[0][0])
        np.testing.assert_allclose(nt[:, 0:3, :][:, ::-1], -nt[:, 3:6, :], rtol=1e-13)


class TurbulenceSANegTests(unittest.TestCase):
    def test_negative_nu_tilde_is_driven_back_to_zero_and_source_is_continuous(self):
        sa = SpalartAllmaras()
        nu, rho, d, omega = 1.5e-5, 1.2, 1e-3, 200.0
        src = lambda nt: float(sa.source(rho, nt, nu, omega, d, 0.0))
        self.assertGreater(src(-4e-5), 0.0)               # negative nu_tilde grows toward zero
        self.assertAlmostEqual(src(1e-14), src(-1e-14), delta=1e-9 * abs(src(1e-5)))
        self.assertEqual(float(sa.mu_t(rho, -4e-5, nu)), 0.0)

    def test_diffusivity_branches(self):
        sa = SpalartAllmaras()
        nu, rho = 1.5e-5, 1.2
        mu = rho * nu
        self.assertAlmostEqual(float(sa.diffusivity(rho, 3e-5, mu)), (mu + rho * 3e-5) / sa.sigma, places=15)
        neg = float(sa.diffusivity(rho, -3e-5, mu))
        self.assertLess(neg, mu / sa.sigma)               # negative nu_tilde reduces the diffusivity
        self.assertGreater(neg, 0.0)


if __name__ == "__main__":
    unittest.main()
