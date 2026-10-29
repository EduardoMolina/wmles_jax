"""WMLES setup for the Tamaki-Kawai A-airfoil reference conditions.

Reference conditions: Mach 0.15, Re_c = 1e7, and 13.3 degree incidence.
The case uses a characteristic far-field, periodic span, and an equilibrium
wall model on the airfoil. T_inf and p_inf set the physical scale and default
to standard ambient air; they are not specified by the reference database.
"""
from pathlib import Path

import numpy as np

from .bc import CharacteristicFarfield, WallModel
from .gas import Gas
from .nmf import FACES, NMF, Patch, build_mesh, describe
from .plot3d_wm import read_plot3d
from .schemes import Hybrid
from .solver import Solver
from .wall_model import EquilibriumWallModel
from .backend import xp

MACH_INF = 0.15
RE_CHORD = 1.0e7
ALPHA_DEG = 13.3
_BLOCK_NAMES = ("airfoil", "trailing_edge", "tunnel", "wake")

# Boundary connections for the four-block grid emitted by airfoil_grid.py.
_INTERFACES = (
    (0, 3, 1, 5),  # airfoil upper TE edge <-> trailing-edge block
    (0, 4, 1, 5),  # airfoil lower TE edge <-> trailing-edge block
    (0, 6, 2, 5),  # airfoil outer layer <-> tunnel inner boundary
    (1, 3, 2, 5),  # lower TE wake edge <-> tunnel inner boundary
    (1, 4, 2, 5),  # upper TE wake edge <-> tunnel inner boundary
    (1, 6, 3, 3),  # TE wake <-> downstream wake
    (2, 3, 3, 3),  # upper tunnel edge <-> downstream wake
    (2, 4, 3, 3),  # lower tunnel edge <-> downstream wake
)


def flow_properties(gas=None, mach=MACH_INF, reynolds=RE_CHORD,
                    alpha_deg=ALPHA_DEG, T_inf=300.0, p_inf=101_325.0):
    """Return dimensional free-stream properties for the specified chord Re."""
    values = (mach, reynolds, alpha_deg, T_inf, p_inf)
    if not np.all(np.isfinite(values)) or mach <= 0.0 or reynolds <= 0.0 \
            or T_inf <= 0.0 or p_inf <= 0.0:
        raise ValueError("flow conditions must be finite; Mach, Re, T_inf, and p_inf must be positive")
    gas = gas or Gas()
    mu_inf = float(gas.mu(xp.asarray(T_inf)))
    U_inf = mach * float(gas.sound_speed(xp.asarray(T_inf)))
    chord = 1.0
    rho_inf = reynolds * mu_inf / (U_inf * chord)
    nu_inf = mu_inf / rho_inf
    p_inf = rho_inf * gas.R * T_inf
    alpha = np.deg2rad(alpha_deg)
    # Positive geometric incidence corresponds to a downward free-stream
    # velocity component in airfoil-fixed coordinates.
    prim_inf = np.array([
        rho_inf, U_inf * np.cos(alpha), -U_inf * np.sin(alpha), 0.0, p_inf
    ])
    return {
        "gas": gas, "mach": mach, "reynolds": reynolds,
        "alpha_deg": alpha_deg, "T_inf": T_inf, "p_inf": p_inf,
        "rho_inf": rho_inf, "mu_inf": mu_inf, "nu_inf": nu_inf,
        "U_inf": U_inf, "chord": chord, "prim_inf": prim_inf,
    }


def _boundary_curve(nodes, face):
    axis, side = FACES[face][:2]
    index = 0 if side == 0 else nodes.shape[axis + 1] - 1
    plane = np.take(nodes, index, axis=axis + 1)
    return plane[:, :, 0].T


def _find_shared_segment(curve_a, curve_b, tolerance):
    """Find the longest node-identical segment, preserving its orientation."""
    best = None
    for reverse in (False, True):
        candidate = curve_b[::-1] if reverse else curve_b
        for start_a in range(len(curve_a)):
            for start_b in range(len(candidate)):
                length = 0
                while (start_a + length < len(curve_a)
                       and start_b + length < len(candidate)
                       and np.allclose(
                           curve_a[start_a + length],
                           candidate[start_b + length],
                           rtol=0.0, atol=tolerance,
                       )):
                    length += 1
                if length >= 2 and (best is None or length > best[0]):
                    if reverse:
                        start_b_original = len(curve_b) - 1 - start_b
                        end_b = len(curve_b) - start_b - length
                    else:
                        start_b_original = start_b
                        end_b = start_b + length - 1
                    best = (
                        length, start_a, start_a + length - 1,
                        start_b_original, end_b,
                    )
    if best is None:
        raise ValueError("four-block grid is missing an expected matching interface")
    length, a0, a1, b0, b1 = best
    if length < 4:
        raise ValueError("matching interface needs at least three cells for ghost layers")
    return (a0, a1), (b0, b1)


def _face_ranges(face, curve_axis, curve_range, dims):
    _, _, tangent_axes = FACES[face]
    ranges = []
    for axis in tangent_axes:
        if axis == curve_axis:
            start, end = curve_range
            ranges.append((start + 1, end + 1))
        else:
            ranges.append((1, dims[axis]))
    return tuple(ranges)


def _full_face_ranges(face, dims):
    _, _, tangent_axes = FACES[face]
    return tuple((1, dims[axis]) for axis in tangent_axes)


def _orient_blocks(nodes):
    """Make all 2D blocks right-handed for the solver's positive-volume metric."""
    oriented, flipped = [], []
    for block in nodes:
        i_edges = block[0, 1:, :-1, 0] - block[0, :-1, :-1, 0]
        j_edges = block[1, :-1, 1:, 0] - block[1, :-1, :-1, 0]
        signed_area = (
            i_edges * j_edges
            - (block[1, 1:, :-1, 0] - block[1, :-1, :-1, 0])
            * (block[0, :-1, 1:, 0] - block[0, :-1, :-1, 0])
        )
        if np.all(signed_area < 0.0):
            oriented.append(np.flip(block, axis=1).copy())
            flipped.append(True)
        elif np.all(signed_area > 0.0):
            oriented.append(block)
            flipped.append(False)
        else:
            raise ValueError("grid contains zero-area or mixed-orientation cells")
    return oriented, flipped


def _build_grid_nmf(nodes, flipped):
    """Infer the generated grid's conformal interfaces from shared coordinates."""
    if len(nodes) != len(_BLOCK_NAMES):
        raise ValueError("A-airfoil WMLES requires the four-block airfoil_grid mesh")
    dims = [tuple(block.shape[1:]) for block in nodes]
    patches = []
    interface_faces = set()
    scale = max(float(np.max(np.abs(block))) for block in nodes)
    tolerance = 1e-10 * max(scale, 1.0)

    for block_a, original_face_a, block_b, original_face_b in _INTERFACES:
        face_a = (
            {3: 4, 4: 3}.get(original_face_a, original_face_a)
            if flipped[block_a] else original_face_a
        )
        face_b = (
            {3: 4, 4: 3}.get(original_face_b, original_face_b)
            if flipped[block_b] else original_face_b
        )
        curve_axis_a = 1 if face_a in (3, 4) else 0
        curve_axis_b = 1 if face_b in (3, 4) else 0
        range_a, range_b = _find_shared_segment(
            _boundary_curve(nodes[block_a], face_a),
            _boundary_curve(nodes[block_b], face_b),
            tolerance,
        )
        tang_a, tang_b = FACES[face_a][2], FACES[face_b][2]
        swap = (tang_a.index(curve_axis_a) != tang_b.index(curve_axis_b))
        patch = Patch(
            "one_to_one", block_a, face_a,
            _face_ranges(face_a, curve_axis_a, range_a, dims[block_a]),
            {
                "block": block_b,
                "face": face_b,
                "ranges": _face_ranges(
                    face_b, curve_axis_b, range_b, dims[block_b]
                ),
                "swap": swap,
            },
        )
        patches.append(patch)
        interface_faces.update(((block_a, face_a), (block_b, face_b)))

    for block, block_dims in enumerate(dims):
        for face in range(1, 7):
            if face in (1, 2):
                patches.append(Patch(
                    "periodic_z", block, face,
                    _full_face_ranges(face, block_dims),
                ))
            elif (block, face) not in interface_faces:
                boundary_type = (
                    "viscous_solid" if block == 0 and face == 5
                    else "farfield"
                )
                patches.append(Patch(
                    boundary_type, block, face,
                    _full_face_ranges(face, block_dims),
                ))
    te_block = _BLOCK_NAMES.index("trailing_edge")
    te_face = 5
    te_interfaces = sorted(
        (min(p.partner["ranges"][1]), max(p.partner["ranges"][1]))
        for p in patches
        if p.partner and p.partner["block"] == te_block
        and p.partner["face"] == te_face
    )
    for (_, previous_end), (next_start, _) in zip(te_interfaces, te_interfaces[1:]):
        if next_start > previous_end:
            patches.append(Patch(
                "viscous_solid", te_block, te_face,
                ((1, dims[te_block][2]), (previous_end, next_start)),
            ))
    return NMF(dims, patches)


def build_case(grid_path="out/a_airfoil.p3d", *, mach=MACH_INF,
               reynolds=RE_CHORD, alpha_deg=ALPHA_DEG, T_inf=300.0,
               p_inf=101_325.0, n_match=1, cfl=0.5, sgs="vreman", partitions=None):
    """Read the generated grid and build a wall-modelled A-airfoil solver case.

    `partitions` splits the assembled zones into at least that many work units (the
    same split as the METIS preprocessing) before the solver is built."""
    grid_path = Path(grid_path)
    grid = read_plot3d(grid_path)
    if grid.ndim != 3 or any(nodes.shape[-1] < 4 for nodes in grid.nodes):
        raise ValueError("grid must be a 3D four-block mesh with at least 3 span cells")
    props = flow_properties(
        mach=mach, reynolds=reynolds, alpha_deg=alpha_deg,
        T_inf=T_inf, p_inf=p_inf,
    )
    nodes, flipped = _orient_blocks(grid.nodes)
    nodes = [block * props["chord"] for block in nodes]
    nmf = _build_grid_nmf(nodes, flipped)
    prim = props["prim_inf"]
    mesh = build_mesh(
        nodes, nmf,
        bc_map={
            "farfield": CharacteristicFarfield(prim),
            "viscous_solid": lambda: WallModel(
                EquilibriumWallModel(), T_wall=None, n_match=n_match
            ),
        },
        periodic=(("periodic_z", 2),),
    )
    if partitions is not None:
        from .plot3d_util import split_mesh
        mesh = split_mesh(mesh, partitions)
    solver = Solver(
        mesh, props["gas"], Hybrid(), sgs=sgs, cfl=cfl,
    )
    props["grid_path"] = grid_path
    return mesh, solver, props


def initial_state(solver, props):
    """Uniform free-stream initialization in airfoil-fixed coordinates."""
    prim = props["prim_inf"]
    return solver.init_state(
        lambda xc: xp.stack([prim[i] + 0.0 * xc[0] for i in range(5)])
    )


def describe_case(mesh, props):
    """Format the reference conditions and the assembled block topology."""
    return (
        f"A-airfoil WMLES: Ma={props['mach']:.3f}, "
        f"Re_c={props['reynolds']:.3e}, alpha={props['alpha_deg']:.2f} deg\n"
        f"chord={props['chord']:.6g} m, U_inf={props['U_inf']:.6g} m/s, "
        f"T_inf={props['T_inf']:.1f} K, p_inf={props['p_inf']:.1f} Pa\n"
        f"rho_inf={props['rho_inf']:.6g} kg/m^3, grid={props['grid_path']}\n{describe(mesh)}"
    )
