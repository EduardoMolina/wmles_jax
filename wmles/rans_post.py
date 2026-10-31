"""Post-processing of RANS / DDES results on no-slip walls (2-D cases extruded along axis 0).

* `skin_friction`: C_f and the first-cell y+ of every `NoSlipWall` face (2nd-order one-sided
  estimate of mu_w du/dn from the first two cell centres), span-averaged.
* `wall_columns`: wall-normal columns of a wall face (distance, tangential velocity, density),
  span-averaged, plus `momentum_thickness` / `re_theta` for zero-pressure-gradient plates.
* `ks_cf`: Karman-Schoenherr correlation, the TMR 2DZP reference.

Wall-normal axis = the face axis; the spanwise axis is axis 0 of the block, so a face whose
normal is axis 2 has tangential axes (span, streamwise).
"""
import numpy as np

from .bc import NoSlipWall


def ks_cf(re_theta):
    """Karman-Schoenherr skin friction vs Re_theta (TMR 2DZP validation)."""
    lg = np.log10(re_theta)
    return 1.0 / (17.08 * lg ** 2 + 25.11 * lg + 6.012)


def wall_faces(solver):
    for ib, blk in enumerate(solver.blocks):
        for (axis, side), bc in blk.faces.items():
            if isinstance(bc, NoSlipWall):
                yield ib, blk, axis, side


def _layers(solver, Q, ib, axis, side, count):
    """First `count` cell layers off the wall: dict of arrays (count, t1, t2) + face data."""
    blk, gas, g = solver.blocks[ib], solver.gas, solver.geoms[ib]
    q = solver.apply_bcs(Q, solver.geoms)[ib]
    w, T = gas.cons_to_prim(q)
    ng, n = blk.ng, blk.n[axis]
    idx = [ng + m if side == 0 else ng + n - 1 - m for m in range(count)]
    fb = ng if side == 0 else ng + n
    S = np.asarray(blk.take(g.S[axis], axis, fb))
    n_in = S / np.sqrt(np.sum(S * S, axis=0))
    if side == 1:
        n_in = -n_in
    xf = np.asarray(blk.take(g.xf[axis], axis, fb))
    out = {"x_face": xf, "y": [], "ut": [], "rho": [], "T": [], "p": []}
    for i in idx:
        xc = np.asarray(blk.take(g.xc, axis, i))
        u = np.asarray(blk.take(w[1:4], axis, i))
        out["y"].append(np.sum((xc - xf) * n_in, axis=0))
        out["ut"].append(np.sqrt(np.sum((u - np.sum(u * n_in, axis=0) * n_in) ** 2, axis=0)))
        for key, arr in (("rho", w[0]), ("T", T), ("p", w[4])):
            out[key].append(np.asarray(blk.take(arr, axis, i)))
    for key in ("y", "ut", "rho", "T", "p"):
        out[key] = np.stack(out[key])
    return out


def skin_friction(solver, Q, props):
    """{(block, face): dict(x, cf, tau, yplus1, u_tau)} along the streamwise axis (span-averaged);
    x is the x-coordinate of the face centres in grid units."""
    out = {}
    q_inf = 0.5 * props["rho_inf"] * props["U_inf"] ** 2
    for ib, blk, axis, side in wall_faces(solver):
        L = _layers(solver, Q, ib, axis, side, 2)
        y1, y2, u1, u2 = L["y"][0], L["y"][1], L["ut"][0], L["ut"][1]
        mu_w = np.asarray(solver.gas.mu(L["T"][0]))
        tau = mu_w * (u1 * y2 ** 2 - u2 * y1 ** 2) / (y1 * y2 * (y2 - y1))
        rho_w = L["rho"][0]
        u_tau = np.sqrt(np.abs(tau) / rho_w)
        mean = lambda a: a.mean(axis=0)
        out[(blk.name, "ijk"[axis] + ("min", "max")[side])] = dict(
            x=mean(L["x_face"][0]) / props["length"], tau=mean(tau), cf=mean(tau) / q_inf,
            yplus1=mean(y1 * u_tau * rho_w / mu_w), u_tau=mean(u_tau))
    return out


def wall_columns(solver, Q, props, ib, axis, side):
    """Span-averaged wall-normal columns of a block's wall face, ordered from the wall:
    dict(x (grid units), y, ut, rho, T, p) with arrays (t2, layers) for the latter five."""
    blk = solver.blocks[ib]
    L = _layers(solver, Q, ib, axis, side, blk.n[axis])
    cols = {k: np.moveaxis(L[k].mean(axis=1), 0, -1) for k in ("y", "ut", "rho", "T", "p")}
    cols["x"] = L["x_face"][0].mean(axis=0) / props["length"]
    return cols


def momentum_thickness(cols, props):
    """theta(x) = int rho u/(rho_inf U_inf) (1 - u/U_inf) dy over the cell-centre column
    (trapezoid, zero at the wall), up to the first cell with u >= 0.99 U_inf so that outer-flow
    disturbances do not contribute.  Valid for zero-pressure-gradient plates only."""
    ratio = cols["ut"] / props["U_inf"]
    crossed = np.maximum.accumulate(ratio >= 0.99, axis=-1)
    inside = np.cumsum(crossed, axis=-1) <= 1                 # layers up to and including the first crossing
    f = cols["rho"] / props["rho_inf"] * ratio * (1.0 - ratio) * inside
    zero = np.zeros_like(cols["y"][..., :1])
    y0, f0 = np.concatenate([zero, cols["y"]], axis=-1), np.concatenate([zero, f], axis=-1)
    return np.sum(0.5 * (f0[..., 1:] + f0[..., :-1]) * np.diff(y0, axis=-1), axis=-1)


def re_theta(cols, props):
    return momentum_thickness(cols, props) * props["U_inf"] / props["nu_inf"]


def wall_forces(solver, Q, props):
    """Force per unit span on all no-slip walls (pressure + skin friction), in N/m, and the
    lift/drag coefficients for the free-stream direction of `props` with the grid unit as the
    reference length.  The 2-D TMR grids lie in the (x, z) plane and extrude along y, so the
    force is the (Fx, Fz) pair and the incidence is measured in that plane.
    Returns dict(force=(Fx, Fz), cl, cd, cd_pressure, cd_friction)."""
    gas = solver.gas
    alpha = np.deg2rad(props["alpha_deg"])
    drag_dir, lift_dir = np.array([np.cos(alpha), np.sin(alpha)]), np.array([-np.sin(alpha), np.cos(alpha)])
    fp, fv = np.zeros(2), np.zeros(2)
    for ib, blk, axis, side in wall_faces(solver):
        g = solver.geoms[ib]
        q = solver.apply_bcs(Q, solver.geoms)[ib]
        w, T = gas.cons_to_prim(q)
        ng, n = blk.ng, blk.n[axis]
        fb = ng if side == 0 else ng + n
        S = np.asarray(blk.take(g.S[axis], axis, fb))
        area = np.sqrt(np.sum(S * S, axis=0))
        n_in = S / area if side == 0 else -S / area
        L = _layers(solver, Q, ib, axis, side, 2)
        y1, y2, u1, u2 = L["y"][0], L["y"][1], L["ut"][0], L["ut"][1]
        tau = np.asarray(gas.mu(L["T"][0])) * (u1 * y2 ** 2 - u2 * y1 ** 2) / (y1 * y2 * (y2 - y1))
        u = np.asarray(blk.take(w[1:4], axis, ng if side == 0 else ng + n - 1))
        ut = u - np.sum(u * n_in, axis=0) * n_in
        t = ut / np.maximum(np.sqrt(np.sum(ut * ut, axis=0)), 1e-30)
        span = float(np.ptp(np.asarray(blk.x_int)[1]))            # the 2-D TMR grids extrude along y
        pf = -L["p"][0] * area * n_in[[0, 2]]                     # in-plane components (x, z)
        vf = tau * area * t[[0, 2]]
        fp += pf.reshape(2, -1).sum(axis=1) / span
        fv += vf.reshape(2, -1).sum(axis=1) / span
    q_inf = 0.5 * props["rho_inf"] * props["U_inf"] ** 2 * props["length"]
    total = fp + fv
    return dict(force=total, cl=float(total @ lift_dir) / q_inf, cd=float(total @ drag_dir) / q_inf,
                cd_pressure=float(fp @ drag_dir) / q_inf, cd_friction=float(fv @ drag_dir) / q_inf)
