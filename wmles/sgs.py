"""Eddy-viscosity SGS models.  All take the velocity-gradient tensor g[i,j]=du_i/dx_j
(3,3,...), the filter width delta = V^(1/3) and return nu_t."""
from .backend import xp


def _eye():
    return xp.eye(3).reshape(3, 3, 1, 1, 1)


def _sym(g):
    return 0.5 * (g + xp.swapaxes(g, 0, 1))


def smagorinsky(Cs=0.16):
    def nut(g, delta):
        S = _sym(g)
        return (Cs * delta) ** 2 * xp.sqrt(2.0 * xp.sum(S * S, axis=(0, 1)))
    return nut


def wale(Cw=0.325):
    def nut(g, delta):
        S = _sym(g)
        SS = xp.sum(S * S, axis=(0, 1))
        g2 = xp.einsum('ik...,kj...->ij...', g, g)
        tr = g2[0, 0] + g2[1, 1] + g2[2, 2]
        Sd = _sym(g2) - _eye() * tr / 3.0
        SdSd = xp.sum(Sd * Sd, axis=(0, 1))
        return (Cw * delta) ** 2 * SdSd ** 1.5 / (SS ** 2.5 + SdSd ** 1.25 + 1e-30)
    return nut


def vreman(Cs=0.17):
    c = 2.5 * Cs ** 2

    def nut(g, delta):
        beta = delta ** 2 * xp.einsum('im...,jm...->ij...', g, g)
        B = (beta[0, 0] * beta[1, 1] - beta[0, 1] ** 2 + beta[0, 0] * beta[2, 2]
             - beta[0, 2] ** 2 + beta[1, 1] * beta[2, 2] - beta[1, 2] ** 2)
        aa = xp.sum(g * g, axis=(0, 1))
        return c * xp.sqrt(xp.maximum(B, 0.0) / (aa + 1e-30))
    return nut


def make_sgs(name="vreman", **kw):
    """Returns f(g, delta, rho) -> mu_t.  name=None/'none' -> implicit LES (mu_t=0)."""
    if name in (None, "none"):
        return lambda g, delta, rho: 0.0 * rho
    nut = {"smagorinsky": smagorinsky, "wale": wale, "vreman": vreman}[name](**kw)
    return lambda g, delta, rho: rho * nut(g, delta)
