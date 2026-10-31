"""Spalart-Allmaras (SA) model: constants and local (point-wise) closure functions.

Only algebraic pieces live here (no transport, no solver coupling): they take arrays of
any shape and are written with the backend `xp`, so they are jit/AD friendly.

* SA-noft2 constants, `S_tilde` with the Allmaras et al. (2012) limiter (negative-S̃ safe).
* ν̃ < 0 is treated as the SA-neg model: nu_t = 0 there (see `mu_t`); the negative-ν̃
  transport terms come with the solver coupling.
* DDES (Spalart et al. 2006): d_tilde = d - f_d max(0, d - C_DES Delta), with
  f_d = 1 - tanh((8 r_d)^3): f_d -> 0 (RANS, d_tilde = d) inside attached boundary layers,
  f_d -> 1 (LES) away from walls.  Delta is `Geom.delta_max` (Delta_max).
"""
from dataclasses import dataclass

from .backend import xp


@dataclass(frozen=True)
class SpalartAllmaras:
    des: str = None              # None (RANS) | "ddes"
    cb1: float = 0.1355
    cb2: float = 0.622
    sigma: float = 2.0 / 3.0
    kappa: float = 0.41
    cv1: float = 7.1
    cv2: float = 0.7
    cv3: float = 0.9
    cw2: float = 0.3
    cw3: float = 2.0
    c_des: float = 0.65
    nut_inf_chi: float = 3.0     # free-stream nu_tilde / nu

    def __post_init__(self):
        if self.des not in (None, "ddes"):
            raise ValueError("des must be None or 'ddes'")

    @property
    def cw1(self):
        return self.cb1 / self.kappa ** 2 + (1.0 + self.cb2) / self.sigma

    @staticmethod
    def chi(nu_tilde, nu):
        return nu_tilde / nu

    def fv1(self, chi):
        chi3 = chi ** 3
        return chi3 / (chi3 + self.cv1 ** 3)

    def fv2(self, chi):
        return 1.0 - chi / (1.0 + chi * self.fv1(chi))

    def mu_t(self, rho, nu_tilde, nu):
        """Eddy viscosity; zero for negative nu_tilde (SA-neg)."""
        chi = self.chi(xp.maximum(nu_tilde, 0.0), nu)
        return rho * xp.maximum(nu_tilde, 0.0) * self.fv1(chi)

    def s_tilde(self, omega, nu_tilde, nu, d):
        """Modified vorticity, bounded below as in Allmaras, Johnson & Spalart (2012)."""
        s_bar = nu_tilde / (self.kappa * d) ** 2 * self.fv2(self.chi(nu_tilde, nu))
        lo = omega + s_bar
        alt = omega + omega * (self.cv2 ** 2 * omega + self.cv3 * s_bar) \
            / ((self.cv3 - 2.0 * self.cv2) * omega - s_bar)
        return xp.where(s_bar >= -self.cv2 * omega, lo, alt)

    def fw(self, s_tilde, nu_tilde, d_len):
        """Destruction function; `d_len` is the length scale (d, or d_tilde for DES)."""
        r = xp.minimum(nu_tilde / (xp.maximum(s_tilde, 1e-30) * (self.kappa * d_len) ** 2), 10.0)
        g = r + self.cw2 * (r ** 6 - r)
        return g * ((1.0 + self.cw3 ** 6) / (g ** 6 + self.cw3 ** 6)) ** (1.0 / 6.0)

    def length_scale(self, d, delta, nu, nu_t, grad_u_sq):
        """Length scale in the SA destruction term: d (RANS) or the DDES d_tilde.
        grad_u_sq = sum_ij (du_i/dx_j)^2."""
        if self.des is None:
            return d
        r_d = (nu_t + nu) / (xp.sqrt(xp.maximum(grad_u_sq, 1e-20)) * (self.kappa * d) ** 2)
        f_d = 1.0 - xp.tanh((8.0 * r_d) ** 3)
        return d - f_d * xp.maximum(0.0, d - self.c_des * delta)
