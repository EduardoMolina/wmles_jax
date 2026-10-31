"""SA closure functions.  Run:  python -m unittest tests.test_turbulence"""
import unittest

import numpy as np

from wmles.turbulence import SpalartAllmaras


class SpalartAllmarasTests(unittest.TestCase):
    def setUp(self):
        self.sa = SpalartAllmaras()
        self.ddes = SpalartAllmaras(des="ddes")

    def test_constants(self):
        self.assertAlmostEqual(self.sa.cw1, 3.2390678, places=6)

    def test_fv1_limits_and_negative_nu_tilde_gives_zero_mu_t(self):
        self.assertEqual(float(self.sa.fv1(0.0)), 0.0)
        self.assertAlmostEqual(float(self.sa.fv1(1e4)), 1.0, places=6)
        self.assertEqual(float(self.sa.mu_t(1.2, -3e-5, 1.5e-5)), 0.0)
        self.assertAlmostEqual(float(self.sa.mu_t(1.2, 3e-5, 1.5e-5)),
                               1.2 * 3e-5 * float(self.sa.fv1(2.0)))

    def test_fw_is_one_at_r_one(self):
        # r = 1  <=>  nu_tilde = S_tilde (kappa d)^2
        d, s = 0.01, 50.0
        nu_tilde = s * (self.sa.kappa * d) ** 2
        self.assertAlmostEqual(float(self.sa.fw(s, nu_tilde, d)), 1.0, places=12)

    def test_s_tilde_is_continuous_and_positive_at_limiter_switch(self):
        nu_tilde, nu, d = 3e-5, 1.5e-5, 1e-3
        s_bar = nu_tilde / (self.sa.kappa * d) ** 2 * float(self.sa.fv2(nu_tilde / nu))
        self.assertLess(s_bar, 0.0)                       # fv2 < 0 here, so the limiter can engage
        w_switch = -s_bar / self.sa.cv2                   # omega where S_bar = -cv2 * omega
        a = float(self.sa.s_tilde(w_switch * (1 - 1e-9), nu_tilde, nu, d))
        b = float(self.sa.s_tilde(w_switch * (1 + 1e-9), nu_tilde, nu, d))
        self.assertAlmostEqual(a, b, delta=1e-6 * abs(a))
        self.assertAlmostEqual(a, w_switch * (1.0 - self.sa.cv2), delta=1e-6 * abs(a))
        for w in (0.1 * w_switch, 0.5 * w_switch, 2.0 * w_switch, 20.0 * w_switch):
            self.assertGreater(float(self.sa.s_tilde(w, nu_tilde, nu, d)), 0.0)

    def test_rans_length_scale_is_wall_distance(self):
        d = np.array([1e-3, 0.1])
        np.testing.assert_array_equal(self.sa.length_scale(d, 0.05, 1.5e-5, 1e-4, 1e4), d)

    def test_ddes_switches_between_rans_and_les(self):
        nu, kap = 1.5e-5, self.ddes.kappa
        d, delta = 0.1, 0.01
        # attached log layer: r_d = 1 -> RANS (d_tilde = d)
        gradsq = (nu * 1.0 / (kap * d) ** 2) ** 2     # sqrt(gradsq) = nu_t+nu / (kappa d)^2 with nu_t+nu=nu
        near = float(self.ddes.length_scale(d, delta, nu, 0.0, gradsq))
        self.assertAlmostEqual(near, d, places=12)
        # separated / free flow: tiny nu_t, large gradients -> r_d -> 0 -> d_tilde = C_DES * Delta
        les = float(self.ddes.length_scale(d, delta, nu, 0.0, 1e12))
        self.assertAlmostEqual(les, self.ddes.c_des * delta, places=10)
        # grid finer than d/C_DES never increases the length scale (d_tilde <= d)
        self.assertLessEqual(float(self.ddes.length_scale(d, 1.0, nu, 0.0, 1e12)), d)

    def test_invalid_des_option(self):
        with self.assertRaises(ValueError):
            SpalartAllmaras(des="iddes")


if __name__ == "__main__":
    unittest.main()
