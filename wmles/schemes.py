"""Inviscid face-flux schemes.

Signature:  scheme(blk, geom, w, gas, d, theta) -> F  with shape (5, ..n_d+1.., n_t1, n_t2)
where w = primitives [rho,u,v,w,p] on the ghosted block and theta the cell-centred
shock sensor.  F already includes the face area, i.e. F = flux . S.

* KEP     : kinetic-energy-preserving split-form central flux (Kennedy-Gruber two-point
            flux, Pirozzoli-type order 2/4 extension).  Non-dissipative -> LES bulk.
* MUSCLHLLC : MUSCL (van Albada) reconstruction of primitives + HLLC.  Shock capturing.
* Hybrid  : blends the two with a Ducros-based switch.
"""
from .backend import xp


def _kg(wa, wb, S, gamma):
    rho = 0.5 * (wa[0] + wb[0])
    u = 0.5 * (wa[1:4] + wb[1:4])
    p = 0.5 * (wa[4] + wb[4])
    ea = 0.5 * (wa[4] / ((gamma - 1.0) * wa[0]) + 0.5 * xp.sum(wa[1:4] ** 2, axis=0)
                + wb[4] / ((gamma - 1.0) * wb[0]) + 0.5 * xp.sum(wb[1:4] ** 2, axis=0))
    un = xp.sum(u * S, axis=0)
    m = rho * un
    return xp.concatenate([m[None], m * u + p * S, (m * ea + p * un)[None]], axis=0)


class KEP:
    def __init__(self, order=4):
        assert order in (2, 4)
        self.order = order

    def __call__(self, blk, geom, w, gas, d, theta=None):
        S = blk.slab(geom.S[d], d, 0, 1)
        L, R = blk.slab(w, d, -1, 0), blk.slab(w, d, 0, 1)
        g = gas.gamma
        F = _kg(L, R, S, g)
        if self.order == 2:
            return F
        L2, R2 = blk.slab(w, d, -2, -1), blk.slab(w, d, 1, 2)
        return 2.0 * ((2.0 / 3.0) * F - (1.0 / 12.0) * (_kg(L, R2, S, g) + _kg(L2, R, S, g)))


def _albada(a, b, eps=1e-12):
    return (a * (b * b + eps) + b * (a * a + eps)) / (a * a + b * b + 2.0 * eps)


def _minmod(a, b):
    return 0.5 * (xp.sign(a) + xp.sign(b)) * xp.minimum(xp.abs(a), xp.abs(b))


def hllc(wl, wr, S, gamma):
    Sm = xp.sqrt(xp.sum(S * S, axis=0))
    n = S / Sm
    rl, ul, pl = wl[0], wl[1:4], wl[4]
    rr, ur, pr = wr[0], wr[1:4], wr[4]
    unl, unr = xp.sum(ul * n, axis=0), xp.sum(ur * n, axis=0)
    cl, cr = xp.sqrt(gamma * pl / rl), xp.sqrt(gamma * pr / rr)
    sl_, sr_ = xp.minimum(unl - cl, unr - cr), xp.maximum(unl + cl, unr + cr)
    ss = ((pr - pl + rl * unl * (sl_ - unl) - rr * unr * (sr_ - unr))
          / (rl * (sl_ - unl) - rr * (sr_ - unr)))
    El = pl / (gamma - 1.0) + 0.5 * rl * xp.sum(ul * ul, axis=0)
    Er = pr / (gamma - 1.0) + 0.5 * rr * xp.sum(ur * ur, axis=0)

    def phys(r, u, p, un, E):
        return xp.concatenate([(r * un)[None], r * un * u + p * n, ((E + p) * un)[None]], axis=0)

    def cons(r, u, E):
        return xp.concatenate([r[None], r * u, E[None]], axis=0)

    def star(r, u, p, un, E, s):
        f = r * (s - un) / (s - ss)
        e = E / r + (ss - un) * (ss + p / (r * (s - un)))
        return xp.concatenate([f[None], f * (u + (ss - un) * n), (f * e)[None]], axis=0)

    Fl, Fr = phys(rl, ul, pl, unl, El), phys(rr, ur, pr, unr, Er)
    Ul, Ur = cons(rl, ul, El), cons(rr, ur, Er)
    Fsl = Fl + sl_ * (star(rl, ul, pl, unl, El, sl_) - Ul)
    Fsr = Fr + sr_ * (star(rr, ur, pr, unr, Er, sr_) - Ur)
    F = xp.where(sl_ >= 0, Fl, xp.where(ss >= 0, Fsl, xp.where(sr_ > 0, Fsr, Fr)))
    return F * Sm


class MUSCLHLLC:
    def __init__(self, limiter="albada"):
        self.lim = {"albada": _albada, "minmod": _minmod}[limiter]

    def __call__(self, blk, geom, w, gas, d, theta=None):
        S = blk.slab(geom.S[d], d, 0, 1)
        L2, L = blk.slab(w, d, -2, -1), blk.slab(w, d, -1, 0)
        R, R2 = blk.slab(w, d, 0, 1), blk.slab(w, d, 1, 2)
        wl = L + 0.5 * self.lim(L - L2, R - L)
        wr = R - 0.5 * self.lim(R - L, R2 - R)
        bad = (wl[0] <= 0) | (wl[4] <= 0) | (wr[0] <= 0) | (wr[4] <= 0)   # positivity fallback
        wl, wr = xp.where(bad, L, wl), xp.where(bad, R, wr)
        return hllc(wl, wr, S, gas.gamma)


class Hybrid:
    """F = (1-s) F_central + s F_upwind, s = ramp(max(theta_L, theta_R))."""

    def __init__(self, central=None, upwind=None, th0=0.6, th1=0.9):
        self.c = central or KEP(4)
        self.u = upwind or MUSCLHLLC()
        self.th0, self.th1 = th0, th1

    def __call__(self, blk, geom, w, gas, d, theta):
        th = xp.maximum(blk.slab(theta, d, -1, 0), blk.slab(theta, d, 0, 1))
        s = xp.clip((th - self.th0) / (self.th1 - self.th0), 0.0, 1.0)
        return (1.0 - s) * self.c(blk, geom, w, gas, d) + s * self.u(blk, geom, w, gas, d)
