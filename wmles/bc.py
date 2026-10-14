"""Boundary conditions.

Every BC implements `fill(q, blk, geom, gas, axis, side)` which returns `q` with the
ghost layers of that face filled (full tangential extent, so edge ghosts are
consistent when faces are processed axis by axis).  A BC may additionally set
`flux_override = True` and implement `wall_flux(...)`, in which case the solver
replaces the numerical flux on that boundary face (used by the wall model).
"""
import numpy as np
from .backend import xp, set_at
from .mesh import sl


def _fill_mirror(q, blk, geom, axis, side, fn):
    """ghost layer m <- fn(cons slab of interior layer m, unit +axis normal n, m)."""
    ng, n = blk.ng, blk.n[axis]
    fb = ng if side == 0 else ng + n                      # boundary face index
    S = geom.S[axis][sl(4, axis, slice(fb, fb + 1))]      # (3, ..1..) full tangential extent
    nrm = S / xp.sqrt(xp.sum(S * S, axis=0))
    for m in range(ng):
        gi = ng - 1 - m if side == 0 else ng + n + m
        ii = ng + m if side == 0 else ng + n - 1 - m
        src = q[sl(q.ndim, axis, slice(ii, ii + 1))]
        q = set_at(q, sl(q.ndim, axis, slice(gi, gi + 1)), fn(src, nrm, m))
    return q


class BC:
    flux_override = False

    def fill(self, q, blk, geom, gas, axis, side):
        raise NotImplementedError

    def fill_at_time(self, q, blk, geom, gas, axis, side, time):
        return self.fill(q, blk, geom, gas, axis, side)


class SlipWall(BC):
    """Symmetry / inviscid wall: reflect the normal velocity."""

    def fill(self, q, blk, geom, gas, axis, side):
        def fn(src, nrm, m):
            w, _ = gas.cons_to_prim(src)
            u = w[1:4]
            un = xp.sum(u * nrm, axis=0)
            return gas.prim_to_cons(xp.concatenate([w[:1], u - 2.0 * un * nrm, w[4:5]], axis=0))
        return _fill_mirror(q, blk, geom, axis, side, fn)


class NoSlipWall(BC):
    """Resolved (wall-resolved LES) no-slip wall.  T_wall=None -> adiabatic."""

    def __init__(self, T_wall=None):
        self.T_wall = T_wall

    def fill(self, q, blk, geom, gas, axis, side):
        Tw = self.T_wall

        def fn(src, nrm, m):
            w, T = gas.cons_to_prim(src)
            Tg = T if Tw is None else xp.maximum(2.0 * Tw - T, 0.1 * Tw)
            rho = w[4] / (gas.R * Tg)
            return gas.prim_to_cons(xp.concatenate([rho[None], -w[1:4], w[4:5]], axis=0))
        return _fill_mirror(q, blk, geom, axis, side, fn)


class Dirichlet(BC):
    """Fixed primitive state [rho,u,v,w,p] (supersonic inflow / far field)."""

    def __init__(self, prim):
        self.prim = np.asarray(prim, float)

    def fill(self, q, blk, geom, gas, axis, side):
        def fn(src, nrm, m):
            w = xp.asarray(self.prim).reshape((5,) + (1,) * (src.ndim - 1)) * xp.ones_like(src[:1])
            return gas.prim_to_cons(w)
        return _fill_mirror(q, blk, geom, axis, side, fn)


class CharacteristicFarfield(BC):
    """Linearized characteristic far-field condition for primitive [rho,u,v,w,p]."""

    def __init__(self, prim):
        self.prim = np.asarray(prim, float)

    def fill(self, q, blk, geom, gas, axis, side):
        def fn(src, nrm, m):
            w, T = gas.cons_to_prim(src)
            w_inf = xp.asarray(self.prim).reshape((5,) + (1,) * (src.ndim - 1)) * xp.ones_like(src[:1])
            rho_inf, u_inf, p_inf = w_inf[0], w_inf[1:4], w_inf[4]
            c_inf = xp.sqrt(gas.gamma * p_inf / rho_inf)

            n_out = nrm if side == 1 else -nrm
            un = xp.sum(w[1:4] * n_out, axis=0)
            un_inf = xp.sum(u_inf * n_out, axis=0)
            c = gas.sound_speed(T)
            drho, dp = w[0] - rho_inf, w[4] - p_inf
            dun = un - un_inf

            # Retain outgoing acoustic modes and impose incoming modes from infinity.
            a_plus = xp.where(un + c > 0.0, dp + rho_inf * c_inf * dun, 0.0)
            a_minus = xp.where(un - c > 0.0, dp - rho_inf * c_inf * dun, 0.0)
            dp_bc = 0.5 * (a_plus + a_minus)
            dun_bc = (a_plus - a_minus) / (2.0 * rho_inf * c_inf)

            outgoing_convective = un > 0.0
            entropy = xp.where(outgoing_convective, drho - dp / c_inf ** 2, 0.0)
            du_t = w[1:4] - u_inf - dun * n_out
            du_t = xp.where(outgoing_convective[None], du_t, 0.0)

            rho_bc = rho_inf + dp_bc / c_inf ** 2 + entropy
            u_bc = u_inf + dun_bc * n_out + du_t
            p_bc = p_inf + dp_bc
            return gas.prim_to_cons(xp.concatenate([rho_bc[None], u_bc, p_bc[None]], axis=0))
        return _fill_mirror(q, blk, geom, axis, side, fn)


class Extrapolate(BC):
    """Zero-gradient (supersonic outflow)."""

    def fill(self, q, blk, geom, gas, axis, side):
        ng, n = blk.ng, blk.n[axis]
        ii = ng if side == 0 else ng + n - 1
        edge = q[sl(q.ndim, axis, slice(ii, ii + 1))]
        return _fill_mirror(q, blk, geom, axis, side, lambda src, nrm, m: edge)


class WallModel(BC):
    """Wall-modelled LES wall: slip ghost cells for the numerics + the numerical
    flux on the wall face replaced by  [0, p n + tau_w t, -q_w] * |S|.

    T_wall=None -> adiabatic.  n_match = number of cells off the wall at which the
    LES state is sampled (1 = first cell centre; 2-3 is often more robust for
    low-order schemes).
    """
    flux_override = True

    def __init__(self, model, T_wall=None, n_match=1):
        self.model, self.T_wall, self.n_match = model, T_wall, n_match
        self._slip = SlipWall()

    def fill(self, q, blk, geom, gas, axis, side):
        return self._slip.fill(q, blk, geom, gas, axis, side)

    def wall_flux(self, w, T, blk, geom, gas, axis, side):
        ng, n = blk.ng, blk.n[axis]
        fb = ng if side == 0 else ng + n
        mi = ng + self.n_match - 1 if side == 0 else ng + n - self.n_match
        ci = ng if side == 0 else ng + n - 1
        S = blk.take(geom.S[axis], axis, fb)                    # (3, t1, t2)
        Sm = xp.sqrt(xp.sum(S * S, axis=0))
        n_out = S / Sm if side == 1 else -S / Sm
        xf = blk.take(geom.xf[axis], axis, fb)
        xcm = blk.take(geom.xc, axis, mi)
        h = xp.sum((xcm - xf) * (-n_out), axis=0)               # wall distance of sample point
        u = blk.take(w[1:4], axis, mi)
        u_par = u - xp.sum(u * n_out, axis=0) * n_out
        U = xp.sqrt(xp.sum(u_par * u_par, axis=0) + 1e-20)
        T_h, p_h = blk.take(T, axis, mi), blk.take(w[4], axis, mi)
        p_b = blk.take(w[4], axis, ci)
        tau, qw = self.model(gas, h, U, T_h, p_h, self.T_wall)
        t = u_par / U
        Fout = xp.concatenate([xp.zeros_like(tau)[None],
                               (p_b * n_out + tau * t) * Sm,
                               (-qw * Sm)[None]], axis=0)
        return (Fout if side == 1 else -Fout), tau, qw
