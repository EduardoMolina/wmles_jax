"""Wall models.  A wall model maps (matching-point state) -> (tau_w, q_w).

Interface (all arguments are arrays over the wall face):
    model(gas, h, U, T_h, p_h, T_wall) -> tau_w [Pa], q_w [W/m^2, positive INTO the fluid]

`EquilibriumWallModel` = Spalding's law of the wall (algebraic, Newton solve for
u_tau) + Kader's temperature law (isothermal wall) or a recovery-temperature
adiabatic wall.  Add other models (TBLE, ODE-based, data-driven...) by writing
another callable with the same signature.
"""
import numpy as np
from .backend import xp, fori_loop


class EquilibriumWallModel:
    def __init__(self, kappa=0.41, B=5.2, n_iter=30):
        self.kappa, self.B, self.n_iter = kappa, B, n_iter

    def utau(self, h, U, rho_w, mu_w):
        k = self.kappa
        ekb = np.exp(-k * self.B)
        Reh = h * U * rho_w / mu_w                 # h*U/nu_w

        def E(up):
            ku = k * up
            return ekb * (xp.exp(ku) - 1.0 - ku - 0.5 * ku ** 2 - ku ** 3 / 6.0)

        def F(up):
            return up + E(up) - Reh / up

        def dF(up):
            ku = k * up
            return 1.0 + ekb * k * (xp.exp(ku) - 1.0 - ku - 0.5 * ku ** 2) + Reh / up ** 2

        def body(_, up):
            new = up - F(up) / dF(up)
            return xp.minimum(xp.maximum(new, 0.5 * up), 100.0)   # safeguarded Newton

        up = fori_loop(0, self.n_iter, body, xp.clip(xp.sqrt(Reh), 1.0, 25.0))
        return U / up                                                # u_tau

    def __call__(self, gas, h, U, T_h, p_h, T_wall):
        Pr = gas.Pr
        U = xp.maximum(U, 1e-8)
        if T_wall is None:      # adiabatic: recovery temperature
            T_w = T_h + Pr ** (1.0 / 3.0) * U ** 2 / (2.0 * gas.cp)
        else:
            T_w = T_wall + 0.0 * T_h
        rho_w = p_h / (gas.R * T_w)
        mu_w = gas.mu(T_w)
        ut = self.utau(h, U, rho_w, mu_w)
        tau_w = rho_w * ut ** 2
        if T_wall is None:
            return tau_w, 0.0 * tau_w
        # Kader (1981) temperature profile
        yp = xp.maximum(h * ut * rho_w / mu_w, 1e-6)
        Gam = xp.maximum(0.01 * (Pr * yp) ** 4 / (1.0 + 5.0 * Pr ** 3 * yp), 1e-20)
        beta = (3.85 * Pr ** (1 / 3) - 1.3) ** 2 + 2.12 * np.log(Pr)
        Tp = Pr * yp * xp.exp(-Gam) + (2.12 * xp.log(1.0 + yp) + beta) * xp.exp(-1.0 / Gam)
        q_w = rho_w * gas.cp * ut * (T_w - T_h) / Tp
        return tau_w, q_w
