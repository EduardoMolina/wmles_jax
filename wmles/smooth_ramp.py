"""WMLES setup for the UMD smooth-ramp validation case.

The grid uses Baurle's grid-6 streamwise and wall-normal spacing distributions.
The streamwise distribution is extended at constant end spacing to cover the
upstream boundary-layer development region and the downstream comparison region.
"""
from pathlib import Path

import numpy as np

from .backend import xp, set_at
from .bc import BC, CharacteristicFarfield, SlipWall, WallModel
from .gas import Gas
from .mesh import Block, Mesh, sl
from .schemes import Hybrid, KEP
from .solver import Solver
from .wall_model import EquilibriumWallModel

RE_L = 667_000.0
MACH_INF = 0.2
RAMP_HEIGHT_L = 0.22
TOP_HEIGHT_L = 0.74
SPAN_L = 0.288
DELTA_IN_L = 0.028

_DATA_DIR = Path(__file__).resolve().parent.parent / "tests" / "data"
_GRID_FILES = {
    "x_wall": "baurle2021wsm_grid6_x_deltaI_wall.txt",
    "y_upstream": "baurle2021wsm_grid6_y_deltaJ_upstream.txt",
    "y_downstream": "baurle2021wsm_grid6_y_deltaJ_downstream.txt",
}


def flow_properties(Ma_inf=MACH_INF, Re_L=RE_L, T_inf=300.0, p_inf=101_325.0, gas=None):
    """Set the physical scale from the requested free-stream Mach and Reynolds numbers."""
    if Ma_inf <= 0.0 or Re_L <= 0.0 or T_inf <= 0.0 or p_inf <= 0.0:
        raise ValueError("Ma_inf, Re_L, T_inf, and p_inf must all be positive")
    gas = gas or Gas()
    rho_inf = p_inf / (gas.R * T_inf)
    mu_inf = float(gas.mu(xp.asarray(T_inf)))
    nu_inf = mu_inf / rho_inf
    U_inf = Ma_inf * float(gas.sound_speed(xp.asarray(T_inf)))
    L = Re_L * nu_inf / U_inf
    return dict(gas=gas, Ma_inf=Ma_inf, Re_L=Re_L, T_inf=T_inf, p_inf=p_inf,
                rho_inf=rho_inf, mu_inf=mu_inf, nu_inf=nu_inf, U_inf=U_inf, L=L)


def load_grid6_distributions(data_dir=None):
    """Load and validate the three two-column Baurle grid-6 distribution files."""
    directory = Path(data_dir) if data_dir is not None else _DATA_DIR
    distributions = {}
    for key, filename in _GRID_FILES.items():
        data = np.loadtxt(directory / filename, dtype=float)
        if data.ndim != 2 or data.shape[1] != 2 or data.shape[0] < 2:
            raise ValueError(f"{filename} must contain at least two rows of coordinate, spacing")
        if not np.all(np.isfinite(data)) or np.any(np.diff(data[:, 0]) <= 0.0):
            raise ValueError(f"{filename} coordinates must be finite and strictly increasing")
        if np.any(data[:, 1] <= 0.0):
            raise ValueError(f"{filename} spacings must be positive")
        distributions[key] = data

    y_up, y_down = distributions["y_upstream"][:, 0], distributions["y_downstream"][:, 0]
    if len(y_up) != len(y_down):
        raise ValueError("upstream and downstream wall-normal distributions must have equal lengths")
    if not (np.isclose(y_up[0], RAMP_HEIGHT_L) and np.isclose(y_down[0], 0.0)
            and np.isclose(y_up[-1], TOP_HEIGHT_L) and np.isclose(y_down[-1], TOP_HEIGHT_L)):
        raise ValueError("wall-normal distributions do not match the expected grid-6 ramp bounds")
    return distributions


def _extend_streamwise_nodes(distribution, x_min, x_max):
    x, spacing = distribution.T
    if x_min > x[0] or x_max < x[-1]:
        raise ValueError(f"x_min and x_max must cover the supplied streamwise range [{x[0]}, {x[-1]}]")
    left_count = int(np.ceil((x[0] - x_min) / spacing[0]))
    right_count = int(np.ceil((x_max - x[-1]) / spacing[-1]))
    left = np.linspace(x_min, x[0], left_count + 1)[:-1] if left_count else np.empty(0)
    right = np.linspace(x[-1], x_max, right_count + 1)[1:] if right_count else np.empty(0)
    return np.concatenate((left, x, right))


def _ramp_fraction(x):
    xi = xp.clip(x, 0.0, 1.0)
    return 10.0 * xi**3 - 15.0 * xi**4 + 6.0 * xi**5


class SmoothRampInflow(BC):
    """Mean 1/7-power incoming boundary layer; this does not add turbulence."""

    def __init__(self, props):
        self.props = props

    def fill(self, q, blk, geom, gas, axis, side):
        if axis != 0 or side != 0:
            raise ValueError("SmoothRampInflow is only valid on the i-min face")
        ng = blk.ng
        for m in range(ng):
            gi = ng - 1 - m
            xc = geom.xc[sl(4, axis, slice(gi, gi + 1))]
            eta = xp.clip((xc[1] - RAMP_HEIGHT_L * self.props["L"])
                          / (DELTA_IN_L * self.props["L"]), 0.0, 1.0)
            u = self.props["U_inf"] * eta ** (1.0 / 7.0)
            zero = xp.zeros_like(u)
            prim = xp.concatenate([
                (self.props["rho_inf"] + zero)[None],
                u[None], zero[None], zero[None],
                (self.props["p_inf"] + zero)[None],
            ], axis=0)
            q = set_at(q, sl(q.ndim, axis, slice(gi, gi + 1)), gas.prim_to_cons(prim))
        return q


def build_case(data_dir=None, x_min=-1.0, x_max=4.0, nz=16, scheme=None,
               sgs="vreman", cfl=0.5, n_match=1, props=None):
    """Build the single-block, wall-modelled smooth-ramp mesh and solver.

    Coordinates are nondimensional in the distribution files and scaled to the
    physical length L implied by Re_L and the free-stream state.
    """
    if nz < 3:
        raise ValueError("nz must be at least 3 cells for the mesh's three ghost layers")
    props = props or flow_properties()
    distributions = load_grid6_distributions(data_dir)
    x = _extend_streamwise_nodes(distributions["x_wall"], x_min, x_max)
    y_up = distributions["y_upstream"][:, 0]
    y_down = distributions["y_downstream"][:, 0]
    ramp = 10.0 * np.clip(x, 0.0, 1.0) ** 3 - 15.0 * np.clip(x, 0.0, 1.0) ** 4 \
        + 6.0 * np.clip(x, 0.0, 1.0) ** 5
    y = (1.0 - ramp[:, None]) * y_up[None, :] + ramp[:, None] * y_down[None, :]
    z = np.linspace(0.0, SPAN_L, nz + 1)
    shape = (len(x), len(y_up), len(z))
    X = np.broadcast_to(x[:, None, None], shape)
    Y = np.broadcast_to(y[:, :, None], shape)
    Z = np.broadcast_to(z[None, None, :], shape)
    nodes = np.stack((X, Y, Z)) * props["L"]

    blk = Block("smooth-ramp", nodes)
    mesh = Mesh([blk])
    L = props["L"]
    Lz = SPAN_L * L
    mesh.connect(blk, "kmin", blk, "kmax", shift=(0.0, 0.0, -Lz))

    prim_inf = np.array([props["rho_inf"], props["U_inf"], 0.0, 0.0, props["p_inf"]])
    blk.set_bc("imin", SmoothRampInflow(props))
    blk.set_bc("imax", CharacteristicFarfield(prim_inf))
    blk.set_bc("jmin", WallModel(EquilibriumWallModel(), T_wall=None, n_match=n_match))
    blk.set_bc("jmax", SlipWall())
    mesh.finalize()

    solver = Solver(mesh, props["gas"], scheme or Hybrid(KEP(4)), sgs=sgs, cfl=cfl)
    return mesh, solver, props


def initial(xc, props):
    """Initialize a smooth 1/7-power boundary-layer mean profile along the ramp."""
    x, y, _ = xc
    x_l = x / props["L"]
    wall = RAMP_HEIGHT_L * props["L"] * (1.0 - _ramp_fraction(x_l))
    eta = xp.clip((y - wall) / (DELTA_IN_L * props["L"]), 0.0, 1.0)
    u = props["U_inf"] * eta ** (1.0 / 7.0)
    zero = xp.zeros_like(u)
    return xp.stack([
        props["rho_inf"] + zero, u, zero, zero, props["p_inf"] + zero,
    ])
