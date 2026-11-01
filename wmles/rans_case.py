"""RANS / DDES validation cases on TMBWG Plot3D + NMF grids (2DZP flat plate, 2DANW airfoil wake, 2DCJ coflowing jet).

    mesh, solver, props = build_rans_case("tests/data/2DZP/flatplate_clust2_3levelsdown_69x49",
                                           mach=0.2, reynolds=5e6)
    Q, history = run_steady(solver, initial_state(solver, props), iterations=20000)

The grid unit length is mapped to `props["length"]` metres so that Re (per grid unit) is exact.
The 2-D grids are respanned to `ncells` periodic spanwise cells (the solver is 3-D).
"""
import numpy as np

from .airfoil_case import flow_properties
from .backend import xp
from .bc import BackPressure, CharacteristicFarfield, TotalConditionsInflow
from .nmf import build_mesh, read_nmf, respan
from .plot3d import read_plot3d
from .schemes import MUSCLHLLC
from .solver import Solver
from .turbulence import SpalartAllmaras

FARFIELD_TYPES = ("farfield_riem", "subsonic_inflow_pt", "back_pressure")


def build_rans_case(base, *, mach, reynolds, alpha_deg=0.0, T_inf=300.0, p_inf=101_325.0,
                    ncells=4, span=0.04, periodic_type="symmetry_y_strong", scheme=None,
                    turbulence="sa", cfl=0.8, bc_map=None):
    """`base`: path without extension of the .p3dfmt/.nmf pair.  `reynolds`: per grid unit.
    `turbulence`: 'sa', None (laminar) or a configured turbulence model."""
    grid, nmf = read_plot3d(base + ".p3dfmt"), read_nmf(base + ".nmf")
    props = flow_properties(mach=mach, reynolds=reynolds, alpha_deg=alpha_deg, T_inf=T_inf, p_inf=p_inf)
    props["length"] = props.pop("chord")
    nodes, nmf = respan(grid.nodes, nmf, ncells=ncells, length=span)
    nodes = [X * props["length"] for X in nodes]
    prim = props["prim_inf"]
    bcs = {t: CharacteristicFarfield(prim) for t in FARFIELD_TYPES}
    bcs["back_pressure"] = BackPressure(prim[4])
    bcs.update(bc_map or {})
    mesh = build_mesh(nodes, nmf, bc_map=bcs, periodic=((periodic_type, 0),))
    if turbulence == "sa":
        turbulence = SpalartAllmaras()
    solver = Solver(mesh, props["gas"], scheme or MUSCLHLLC(), cfl=cfl, turbulence=turbulence)
    props["base"] = base
    return mesh, solver, props


def initial_state(solver, props, profile=None):
    """Free-stream state, or `profile="flatplate"`: an analytic turbulent boundary layer along a plate
    starting at x = 0 with the wall at z = 0 (a convergence aid only: the steady solution does not
    depend on the initial state, but the uniform start needs ~1e5 explicit pseudo-time steps)."""
    prim = props["prim_inf"]
    if profile is None:
        return solver.init_state(lambda xc: xp.stack([prim[i] + 0.0 * xc[0] for i in range(5)]))
    if profile != "flatplate":
        raise ValueError("profile must be None or 'flatplate'")
    return solver.init_state(*_flatplate_profile(props, solver.gas))


def _flatplate_profile(props, gas, kappa=0.41):
    """(prim_fn, nu_tilde_fn): 1/7th-power-law skin friction and thickness (Schlichting) with a
    Reichardt inner law plus a Coles wake, and nu_tilde = kappa u_tau y damped near the wall."""
    U, nu, prim, L = props["U_inf"], props["nu_inf"], props["prim_inf"], props["length"]

    def state(xc):
        x, z = xc[0], xc[2]
        xs = xp.maximum(x, 0.02 * L)
        re_x = U * xs / nu
        u_tau = U * xp.sqrt(0.5 * 0.0592 * re_x ** -0.2)
        delta = 0.37 * xs * re_x ** -0.2
        yp = u_tau * z / nu
        up = (xp.log(1.0 + kappa * yp) / kappa
              + 7.8 * (1.0 - xp.exp(-yp / 11.0) - yp / 11.0 * xp.exp(-yp / 3.0)))
        wake = 2.0 * 0.5 / kappa * xp.sin(0.5 * np.pi * xp.minimum(z / delta, 1.0)) ** 2
        u = xp.where((x > 0.0) & (z < delta), xp.minimum((up + wake) * u_tau, U), U)
        nut = kappa * u_tau * z * (1.0 - xp.exp(-yp / 17.0)) * xp.exp(-(z / (0.4 * delta)) ** 2)
        nut = xp.where((x > 0.0) & (z < delta), nut, 0.0)
        return u, nut

    def prim_fn(xc):
        u, _ = state(xc)
        zero = 0.0 * xc[0]
        return xp.stack([prim[0] + zero, u, zero, zero, prim[4] + zero])

    def nu_tilde_fn(xc):
        _, nut = state(xc)
        return xp.maximum(nut, 3.0 * nu)

    return prim_fn, nu_tilde_fn


def residual_norms(solver, Q):
    """Global RMS (over all interior cells) of the rhs of rho and, with SA, of rho*nu_tilde."""
    R = solver.rhs(Q)
    rows = (0,) if solver.turbulence is None else (0, 5)
    n = sum(int(np.prod(b.n)) for b in solver.blocks)
    return tuple(float(np.sqrt(sum(np.sum(np.asarray(b.interior(r[row])) ** 2)
                                   for b, r in zip(solver.blocks, R)) / n)) for row in rows)


def run_steady(solver, Q, iterations, *, report_every=500, tol=None, callback=None):
    """Local-time-stepping SSP-RK3 toward a steady state.  Returns (Q, history); history rows are
    (iteration, rms rho residual[, rms rho*nu_tilde residual]).  Stops early once the density
    residual has dropped by `tol` relative to the first report."""
    history = []
    for it in range(1, iterations + 1):
        if it % report_every == 0 or it == 1:
            history.append((it,) + residual_norms(solver, Q))
            if callback:
                callback(it, Q, history[-1])
            if tol is not None and history[-1][1] < tol * history[0][1]:
                break
        Q = solver.step_local(Q)
    return Q, history


JET_MACH = (0.5, 0.25)            # inner jet (y < 0.5), outer stream
JET_SPLIT = 0.5                   # height of the separating plate, grid units


def build_jet_case(base, *, reynolds=5.0e4, mach=JET_MACH, **kw):
    """TMR 2DCJ coflowing jet: uniform static p and T, inner/outer stream Mach numbers `mach`.
    As in the TMR (CFL3D/FUN3D inputs) the reference Mach number is that of the inner stream and
    `reynolds` = rho U_ref L / mu per grid unit, so props["U_inf"] = mach[0] a and force coefficients
    are normalised with q_ref = rho U_ref^2 / 2."""
    props_bc = flow_properties(mach=mach[0], reynolds=reynolds)
    gas, T, p = props_bc["gas"], props_bc["T_inf"], props_bc["p_inf"]
    g = gas.gamma
    t0 = [1.0 + 0.5 * (g - 1.0) * m * m for m in mach]
    length = props_bc["chord"]
    inflow = TotalConditionsInflow(((JET_SPLIT * length, p * t0[0] ** (g / (g - 1.0)), T * t0[0]),
                                    (np.inf, p * t0[1] ** (g / (g - 1.0)), T * t0[1])))
    bc_map = {"subsonic_inflow_pt": inflow, "back_pressure": BackPressure(p), **kw.pop("bc_map", {})}
    mesh, solver, props = build_rans_case(base, mach=mach[0], reynolds=reynolds, bc_map=bc_map, **kw)
    props["mach_streams"] = tuple(mach)
    props["a_inf"] = props["U_inf"] / mach[0]
    return mesh, solver, props


def jet_initial_state(solver, props):
    """Uniform p and T; inner/outer stream velocity M a_inf on either side of the plate height."""
    prim, a, L = props["prim_inf"], props["a_inf"], props["length"]
    m_in, m_out = props["mach_streams"]

    def prim_fn(xc):
        u = xp.where(xc[2] < JET_SPLIT * L, m_in * a, m_out * a)
        zero = 0.0 * xc[0]
        return xp.stack([prim[0] + zero, u + zero, zero, zero, prim[4] + zero])

    return solver.init_state(prim_fn)
