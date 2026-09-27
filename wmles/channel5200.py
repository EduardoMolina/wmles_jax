"""Compressible WMLES setup for the UMD wall-model validation case:
    https://wmles.umd.edu/validation/channel-flow-at-re5200/

Reference (incompressible DNS): Lee & Moser (2015), JFM 774, 395-415.
    Re_b = U_b h / nu = 125,000  ->  Re_tau ~ 5200   (h = channel half-height)
The DNS is incompressible; the UMD page reproduces it with compressible solvers at
bulk Mach 0.2, which is what this module sets up. Domain (12.8h, 2h, 4.8h), periodic
in x (streamwise) and z (spanwise), wall-modelled walls at y=0 and y=2h, adiabatic
(the incompressible reference has no heat transfer).

GRID_TABLE reproduces the UMD "wall-stress model" grids exactly (node counts,
Delta x = Delta z spacing). The wall-normal node distribution is NOT the published
one -- the site's ASCII y-files (channel_1dwm_*.txt) were unreachable from this
sandbox (network disabled / links returned 404) -- so `clustered_y` instead builds a
symmetric tanh-clustered grid tuned to the stated design ratio Delta y_w ~= 0.25 Delta
y_c. Swap in the real coordinates via `Block(..., nodes=...)` with your own y-array if
you have the official files.

Usage
-----
    from wmles.channel5200 import build_case, PlaneStats, load_lm_profile
    mesh, solver, props = build_case(level=1, ar=1)
    Q = solver.init_state(lambda xc: initial(xc, props))
    stats = PlaneStats(mesh, solver)
    for it in range(nsteps):
        Q = solver.advance(Q, solver.max_dt(Q))
        if it > n_warmup:
            stats.accumulate(Q)
    y, prof = stats.result()
"""
import numpy as np

from .backend import xp
from .gas import Gas
from .mesh import Block, Mesh
from .bc import WallModel
from .wall_model import EquilibriumWallModel
from .schemes import Hybrid, KEP
from .solver import Solver, ConstantMassFlux

# UMD "Class 1" (wall-stress model) grids: node counts (Nx,Ny,Nz), AR = dx/dz.
# Keyed by resolution level 1 (coarsest) .. 4 (finest); dz/h is the label used on the site.
GRID_TABLE = {
    1: {1: dict(Nx=129, Ny=33, Nz=49, dz_h=0.100), 2: dict(Nx=65, Ny=33, Nz=49, dz_h=0.100)},
    2: {1: dict(Nx=193, Ny=49, Nz=73, dz_h=0.067), 2: dict(Nx=97, Ny=49, Nz=73, dz_h=0.067)},
    3: {1: dict(Nx=257, Ny=65, Nz=97, dz_h=0.050), 2: dict(Nx=129, Ny=65, Nz=97, dz_h=0.050)},
    4: {1: dict(Nx=385, Ny=97, Nz=145, dz_h=0.033), 2: dict(Nx=193, Ny=97, Nz=145, dz_h=0.033)},
}
DOMAIN = dict(Lx_h=12.8, Ly_h=2.0, Lz_h=4.8)   # (Lx, Ly, Lz) / h
RE_B, RE_TAU_TARGET = 125_000.0, 5200.0        # Lee & Moser (2015)
H_WM_H = 0.1                                   # nominal wall-model layer thickness / h


def flow_properties(Ma_b=0.2, Re_b=RE_B, T0=300.0, p0=101_325.0, gas=None):
    """Pick (h, U_b) so that Re_b = U_b h / nu(T0) and U_b = Ma_b * c(T0), for real air."""
    gas = gas or Gas()
    rho0 = p0 / (gas.R * T0)
    nu0 = float(gas.mu(xp.asarray(T0))) / rho0
    Ub = Ma_b * float(gas.sound_speed(xp.asarray(T0)))
    h = Re_b * nu0 / Ub
    return dict(gas=gas, T0=T0, p0=p0, rho0=rho0, nu0=nu0, mu0=nu0 * rho0, Ub=Ub, h=h,
               u_tau_est=RE_TAU_TARGET * nu0 / h)


def clustered_y(ny, h=1.0, ratio=0.25, maxiter=80):
    """Ny+1 nodes on [0,2h], symmetric tanh clustering with Delta y_wall / Delta y_center
    tuned (by bisection on the clustering strength) to `ratio`."""
    def dyw_dyc(beta):
        eta = np.array([0.0, 1.0 / ny, 0.5, 0.5 + 1.0 / ny])
        y = h * (1 + np.tanh(beta * (2 * eta - 1)) / np.tanh(beta))
        return (y[1] - y[0]) / (y[3] - y[2])
    lo, hi = 1e-6, 8.0
    for _ in range(maxiter):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if dyw_dyc(mid) > ratio else (lo, mid)
    beta = 0.5 * (lo + hi)
    eta = np.linspace(0.0, 1.0, ny + 1)
    return h * (1 + np.tanh(beta * (2 * eta - 1)) / np.tanh(beta))


def build_case(level=1, ar=1, scheme=None, sgs="vreman", sgs_kwargs=None, cfl=0.7,
              n_match=1, props=None):
    """One periodic Cartesian block, wall-modelled adiabatic walls at y=0 and y=2h."""
    g = GRID_TABLE[level][ar]
    props = props or flow_properties()
    h = props["h"]
    Lx, Ly, Lz = DOMAIN["Lx_h"] * h, DOMAIN["Ly_h"] * h, DOMAIN["Lz_h"] * h
    x = np.linspace(0.0, Lx, g["Nx"])
    y = clustered_y(g["Ny"] - 1, h=h, ratio=0.25)
    z = np.linspace(0.0, Lz, g["Nz"])
    nodes = np.stack(np.meshgrid(x, y, z, indexing="ij"))
    blk = Block("channel", nodes)
    mesh = Mesh([blk])
    mesh.connect(blk, "imin", blk, "imax", shift=(-Lx, 0, 0))
    mesh.connect(blk, "kmin", blk, "kmax", shift=(0, 0, -Lz))
    wm = EquilibriumWallModel()
    blk.set_bc("jmin", WallModel(wm, T_wall=None, n_match=n_match))
    blk.set_bc("jmax", WallModel(wm, T_wall=None, n_match=n_match))
    mesh.finalize()
    scheme = scheme or Hybrid(KEP(4))
    solver = Solver(mesh, props["gas"], scheme, sgs=sgs, sgs_kwargs=sgs_kwargs, cfl=cfl,
                    constraints=[ConstantMassFlux(0, props["rho0"] * props["Ub"])])
    return mesh, solver, props


def initial(xc, props, seed=0):
    """1/7-power mean profile + divergence-free-ish sinusoidal perturbations."""
    x, y, z = xc
    h, Ub = props["h"], props["Ub"]
    Lx = DOMAIN["Lx_h"] * h
    Lz = DOMAIN["Lz_h"] * h
    yn = xp.clip(xp.minimum(y, 2 * h - y) / h, 1e-3, 1.0)
    u = 1.15 * Ub * yn ** (1.0 / 7.0)
    k = 2 * np.pi
    u = u + 0.09 * Ub * xp.sin(2 * k * z / Lz) * xp.sin(k * x / Lx) * xp.sin(np.pi * y / (2 * h))
    v = 0.06 * Ub * xp.sin(2 * k * x / Lx) * xp.sin(k * z / Lz) * xp.sin(np.pi * y / (2 * h))
    w = 0.06 * Ub * xp.cos(k * x / Lx) * xp.sin(np.pi * y / (2 * h))
    return xp.stack([props["rho0"] + 0 * x, u, v, w, props["p0"] + 0 * x])


class PlaneStats:
    """Streamwise/spanwise-plane- and time-averaged profiles (mesh must have Ny distinct
    y-planes shared by every (x,z), i.e. an unskewed channel block as `build_case` makes)."""

    def __init__(self, mesh, solver):
        self.mesh, self.solver = mesh, solver
        blk, geom = mesh.blocks[0], solver.geoms[0]
        self.y = np.asarray(blk.interior(geom.xc[1]))[0, :, 0]
        self.n = 0
        self.sum = {k: np.zeros_like(self.y) for k in
                   ("rho", "u", "v", "w", "uu", "vv", "ww", "uv", "T")}

    def accumulate(self, Q):
        blk, gas = self.mesh.blocks[0], self.solver.gas
        w, T = gas.cons_to_prim(Q[0])
        u, v, wc = (np.asarray(blk.interior(c)) for c in w[1:4])
        rho, Ta = np.asarray(blk.interior(w[0])), np.asarray(blk.interior(T))
        ax = (0, 2)
        self.sum["rho"] += rho.mean(axis=ax)
        self.sum["u"] += u.mean(axis=ax)
        self.sum["v"] += v.mean(axis=ax)
        self.sum["w"] += wc.mean(axis=ax)
        self.sum["uu"] += (u * u).mean(axis=ax)
        self.sum["vv"] += (v * v).mean(axis=ax)
        self.sum["ww"] += (wc * wc).mean(axis=ax)
        self.sum["uv"] += (u * v).mean(axis=ax)
        self.sum["T"] += Ta.mean(axis=ax)
        self.n += 1

    def result(self):
        assert self.n > 0, "call accumulate() at least once before result()"
        avg = {k: v / self.n for k, v in self.sum.items()}
        avg["uu"] -= avg["u"] ** 2
        avg["vv"] -= avg["v"] ** 2
        avg["ww"] -= avg["w"] ** 2
        avg["uv"] -= avg["u"] * avg["v"]
        return self.y, avg

    def wall_units(self, tau_w, rho_w, mu_w, h):
        """Inner-scaled profile for the lower half of the channel: y+, U+, u_i u_j+."""
        y, avg = self.result()
        u_tau = np.sqrt(tau_w / rho_w)
        nu_w = mu_w / rho_w
        mask = y <= h
        yp = y[mask] * u_tau / nu_w
        out = {"y+": yp, "U+": avg["u"][mask] / u_tau,
               "uu+": avg["uu"][mask] / u_tau ** 2, "vv+": avg["vv"][mask] / u_tau ** 2,
               "ww+": avg["ww"][mask] / u_tau ** 2, "uv+": avg["uv"][mask] / u_tau ** 2}
        return u_tau, out


def load_lm_profile(path):
    """Read a Lee & Moser `LM_Channel_*_mean_prof.txt` file (columns: y/delta, y+, U, dU/dy,
    W, P, in wall units). Download your own copy, e.g. from
    https://turbulence.oden.utexas.edu/channel2015/ or https://lee.me.uh.edu/ ."""
    data = np.loadtxt(path, comments="%")
    names = ["y_delta", "y+", "U+", "dUdy+", "W+", "P+"]
    return {n: data[:, i] for i, n in enumerate(names)}