"""Calorically perfect gas with Sutherland viscosity.

Conservative layout everywhere: q = [rho, rho*u, rho*v, rho*w, rho*E]
Primitive layout:               w = [rho, u, v, w, p]
"""
from dataclasses import dataclass
from .backend import xp


@dataclass(frozen=True)
class Gas:
    gamma: float = 1.4
    R: float = 287.05
    Pr: float = 0.72
    Pr_t: float = 0.9
    mu_ref: float = 1.716e-5
    T_ref: float = 273.15
    S_suth: float = 110.4

    @property
    def cp(self):
        return self.gamma * self.R / (self.gamma - 1.0)

    def mu(self, T):
        return (self.mu_ref * (T / self.T_ref) ** 1.5
                * (self.T_ref + self.S_suth) / (T + self.S_suth))

    def cons_to_prim(self, q):
        rho = q[0]
        u = q[1:4] / rho
        p = (self.gamma - 1.0) * (q[4] - 0.5 * rho * xp.sum(u * u, axis=0))
        T = p / (rho * self.R)
        return xp.concatenate([rho[None], u, p[None]], axis=0), T

    def prim_to_cons(self, w):
        rho, u, p = w[0], w[1:4], w[4]
        E = p / (self.gamma - 1.0) + 0.5 * rho * xp.sum(u * u, axis=0)
        return xp.concatenate([rho[None], rho * u, E[None]], axis=0)

    def sound_speed(self, T):
        return xp.sqrt(self.gamma * self.R * T)
