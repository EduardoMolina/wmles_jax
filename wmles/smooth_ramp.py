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
_DNS_DIR = _DATA_DIR / "smooth_ramp_dns"
_GRID_FILES = {
    "x_wall": "baurle2021wsm_grid6_x_deltaI_wall.txt",
    "y_upstream": "baurle2021wsm_grid6_y_deltaJ_upstream.txt",
    "y_downstream": "baurle2021wsm_grid6_y_deltaJ_downstream.txt",
}
_DNS_PROFILES = {
    -0.40: "dnsProfile_xm0p40.dat",
    -0.62: "dnsProfile_xm0p62.dat",
    -0.76: "dnsProfile_xm0p76.dat",
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


def load_dns_profile(data_dir=None, station_x_l=-0.76):
    """Load one DNS mean/Reynolds-stress profile normalized by Uref."""
    station = min(_DNS_PROFILES, key=lambda x: abs(x - station_x_l))
    if not np.isclose(station, station_x_l, rtol=0.0, atol=1e-12):
        raise ValueError(f"DNS profiles are available at x/L={tuple(_DNS_PROFILES)}")
    directory = Path(data_dir) if data_dir is not None else _DNS_DIR
    path = directory / _DNS_PROFILES[station]
    try:
        profile = np.loadtxt(path, dtype=float)
    except OSError as exc:
        raise OSError(f"could not load smooth-ramp DNS profile: {path}") from exc
    if profile.ndim != 2 or profile.shape[1] != 7 or profile.shape[0] < 2:
        raise ValueError(f"{path.name} must contain y, U, V, uu, vv, ww, and uv columns")
    if not np.all(np.isfinite(profile)) or np.any(np.diff(profile[:, 0]) <= 0.0):
        raise ValueError(f"{path.name} coordinates and values must be finite; y must increase")
    variances = profile[:, 3:6]
    if np.any(variances < -1e-12):
        raise ValueError(f"{path.name} contains negative Reynolds normal stresses")
    if np.any(profile[:, 6] ** 2 > profile[:, 3] * profile[:, 4] + 1e-12):
        raise ValueError(f"{path.name} contains a non-positive-semidefinite u-v stress")
    return profile


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
    """DNS-profile synthetic turbulent inflow with deterministic Fourier modes."""

    def __init__(self, props, profile, length_scales=(0.1, 0.02, 0.03),
                 seed=0, n_modes=64, span_cells=16):
        length_scales = np.asarray(length_scales, dtype=float)
        if (length_scales.shape != (3,) or not np.all(np.isfinite(length_scales))
                or np.any(length_scales <= 0.0)):
            raise ValueError("inflow length_scales must contain three positive finite values")
        if seed < 0:
            raise ValueError("inflow seed must be non-negative")
        if n_modes < 4 or span_cells < 3:
            raise ValueError("inflow synthesis needs at least four modes and three spanwise cells")
        self.props, self.profile = props, np.asarray(profile, dtype=float)
        self.length_scales = tuple(length_scales)
        self.seed, self.n_modes = int(seed), int(n_modes)
        self._modes = []
        max_span_mode = max(1, span_cells // 2)
        span_modes = np.arange(-max_span_mode, max_span_mode + 1)
        span_modes = span_modes[span_modes != 0]
        kz_values = 2.0 * np.pi * span_modes / SPAN_L
        span_weights = np.exp(-0.5 * (kz_values * length_scales[2]) ** 2)
        span_weights /= np.sum(span_weights)
        for component in range(3):
            rng = np.random.default_rng(self.seed + component)
            kx = rng.normal(0.0, 1.0 / length_scales[0], n_modes)
            ky = rng.normal(0.0, 1.0 / length_scales[1], n_modes)
            kz = rng.choice(kz_values, size=n_modes, p=span_weights)
            phase = rng.uniform(0.0, 2.0 * np.pi, n_modes)
            self._modes.append((kx, ky, kz, phase))

    def fill(self, q, blk, geom, gas, axis, side):
        return self.fill_at_time(q, blk, geom, gas, axis, side, 0.0)

    def fill_at_time(self, q, blk, geom, gas, axis, side, time):
        if axis != 0 or side != 0:
            raise ValueError("SmoothRampInflow is only valid on the i-min face")
        ng = blk.ng
        for m in range(ng):
            gi = ng - 1 - m
            xc = geom.xc[sl(4, axis, slice(gi, gi + 1))]
            y_l, z_l = xc[1, 0] / self.props["L"], xc[2, 0] / self.props["L"]
            noise = self._unit_noise(y_l, z_l, time, blk.n[2], ng)
            tab = xp.asarray(self.profile)
            interp = lambda column: xp.interp(y_l, tab[:, 0], tab[:, column])
            Uref = self.props["U_inf"]
            uu = xp.maximum(interp(3), 0.0) * Uref ** 2
            vv = xp.maximum(interp(4), 0.0) * Uref ** 2
            ww = xp.maximum(interp(5), 0.0) * Uref ** 2
            uv = interp(6) * Uref ** 2
            sqrt_uu = xp.sqrt(uu)
            l10 = xp.where(sqrt_uu > 1e-14, uv / xp.maximum(sqrt_uu, 1e-14), 0.0)
            l11 = xp.sqrt(xp.maximum(vv - l10 ** 2, 0.0))
            u = Uref * interp(1) + sqrt_uu * noise[0]
            v = Uref * interp(2) + l10 * noise[0] + l11 * noise[1]
            w = xp.sqrt(ww) * noise[2]
            zero = xp.zeros_like(u)
            prim = xp.stack([
                self.props["rho_inf"] + zero, u, v, w, self.props["p_inf"] + zero,
            ])[:, None, :, :]
            q = set_at(q, sl(q.ndim, axis, slice(gi, gi + 1)), gas.prim_to_cons(prim))
        return q

    def _unit_noise(self, y_l, z_l, time, span_cells, ng):
        coordinates = (self.props["U_inf"] * time / self.props["L"], y_l, z_l)
        fields = []
        z_int = slice(ng, ng + span_cells)
        for component, (kx, ky, kz, phase) in enumerate(self._modes):
            phase_x = xp.asarray(kx)[:, None, None] * coordinates[0]
            phase_y = xp.asarray(ky)[:, None, None] * coordinates[1][None, :, :]
            phase_z = xp.asarray(kz)[:, None, None] * coordinates[2][None, :, :]
            field = xp.sum(xp.cos(phase_x + phase_y + phase_z
                                  + xp.asarray(phase)[:, None, None]), axis=0)
            field = field - xp.mean(field[:, z_int], axis=1, keepdims=True)
            if component == 1:
                reference = fields[0]
                covariance = xp.mean(field[:, z_int] * reference[:, z_int],
                                     axis=1, keepdims=True)
                field = field - covariance * reference
            variance = xp.mean(field[:, z_int] ** 2, axis=1, keepdims=True)
            fields.append(field / xp.sqrt(xp.maximum(variance, 1e-20)))
        return xp.stack(fields)


class SmoothRampSponge:
    """Damp spanwise departures from the evolving spanwise-mean conservative state."""

    def __init__(self, props, x_start_l, x_end_l, strength=1.0):
        if not np.isfinite(strength) or strength < 0.0:
            raise ValueError("sponge strength must be finite and non-negative")
        if not np.isfinite(x_start_l + x_end_l) or x_start_l >= x_end_l:
            raise ValueError("sponge start must be finite and upstream of its end")
        self.props = props
        self.x_start_l, self.x_end_l = float(x_start_l), float(x_end_l)
        self.strength = float(strength)

    def __call__(self, blk, q, geom, gas):
        q_int = blk.interior(q)
        vol = blk.interior(geom.vol)
        mean = xp.sum(q_int * vol[None], axis=3, keepdims=True) \
            / xp.sum(vol, axis=2, keepdims=True)[None]
        x_l = blk.interior(geom.xc[0]) / self.props["L"]
        eta = xp.clip((x_l - self.x_start_l) / (self.x_end_l - self.x_start_l), 0.0, 1.0)
        ramp = eta ** 3 * (10.0 + eta * (-15.0 + 6.0 * eta))
        sigma = self.strength * self.props["U_inf"] / self.props["L"] * ramp
        return -sigma[None] * (q_int - mean)


def build_case(data_dir=None, x_min=-0.76, x_max=4.0, nz=16, scheme=None,
               sgs="vreman", cfl=0.5, n_match=1, props=None, dns_data_dir=None,
               inflow_scales=(0.1, 0.02, 0.03), inflow_seed=0, inflow_modes=64,
               sponge=True, sponge_start=None, sponge_strength=1.0):
    """Build the single-block, wall-modelled smooth-ramp mesh and solver.

    Coordinates are nondimensional in the distribution files and scaled to the
    physical length L implied by Re_L and the free-stream state.
    """
    if nz < 3:
        raise ValueError("nz must be at least 3 cells for the mesh's three ghost layers")
    props = props or flow_properties()
    distributions = load_grid6_distributions(data_dir)
    dns_profile = load_dns_profile(dns_data_dir)
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
    blk.set_bc("imin", SmoothRampInflow(
        props, dns_profile, inflow_scales, inflow_seed, inflow_modes, nz
    ))
    blk.set_bc("imax", CharacteristicFarfield(prim_inf))
    blk.set_bc("jmin", WallModel(EquilibriumWallModel(), T_wall=None, n_match=n_match))
    blk.set_bc("jmax", SlipWall())
    mesh.finalize()

    sources = ()
    if sponge:
        start = max(x_min, x_max - 1.0) if sponge_start is None else sponge_start
        if not x_min <= start < x_max:
            raise ValueError("sponge_start must lie inside the streamwise domain and precede x_max")
        sources = (SmoothRampSponge(props, start, x_max, sponge_strength),)
    solver = Solver(mesh, props["gas"], scheme or Hybrid(KEP(4)), sgs=sgs, cfl=cfl,
                    sources=sources)
    return mesh, solver, props


def initial(xc, props, dns_profile=None):
    """Initialize the DNS mean boundary-layer profile along the ramp."""
    dns_profile = load_dns_profile() if dns_profile is None else dns_profile
    x, y, _ = xc
    x_l = x / props["L"]
    wall = RAMP_HEIGHT_L * props["L"] * (1.0 - _ramp_fraction(x_l))
    y_dns_l = y / props["L"] - wall / props["L"] + RAMP_HEIGHT_L
    tab = xp.asarray(dns_profile)
    u = props["U_inf"] * xp.interp(y_dns_l, tab[:, 0], tab[:, 1])
    v = props["U_inf"] * xp.interp(y_dns_l, tab[:, 0], tab[:, 2])
    zero = xp.zeros_like(u)
    return xp.stack([
        props["rho_inf"] + zero, u, v, zero, props["p_inf"] + zero,
    ])
