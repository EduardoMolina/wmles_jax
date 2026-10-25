"""Standalone four-block structured mesh generator for an airfoil.

The module requires NumPy and SciPy; PyVista is needed only for VTK output.
Both output formats contain four 3D blocks:

    python -m wmles.airfoil_grid out/a_airfoil.p3d
    python -m wmles.airfoil_grid out/a_airfoil.vtm

The blocks follow PyAero's Windtunnel construction: airfoil contour,
trailing-edge wake, outer tunnel around the airfoil, and downstream wake.
All lengths are in chord lengths and the default contour is tests/data/a-airfoil.dat.
"""

import argparse
import math
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
from scipy import interpolate, ndimage

from .plot3d_wm import write_plot3d as _write_plot3d


def naca0012_surface(parameter: np.ndarray) -> np.ndarray:
    """Return cosine-spaced NACA 0012 points from upper to lower trailing edge."""
    if parameter.ndim != 1 or len(parameter) < 3:
        raise ValueError("parameter must be a one-dimensional array with at least 3 points")
    if (not np.all(np.isfinite(parameter))
            or np.any(parameter < 0.0) or np.any(parameter > 1.0)):
        raise ValueError("parameter values must be finite and within [0, 1]")

    x = np.clip(0.5 * (1.0 + np.cos(2.0 * np.pi * parameter)), 0.0, 1.0)
    thickness = 0.12 / 0.2 * (
        0.2969 * np.sqrt(x)
        - 0.1260 * x
        - 0.3516 * x**2
        + 0.2843 * x**3
        - 0.1036 * x**4
    )
    y = np.where(parameter <= 0.5, thickness, -thickness)
    return np.column_stack((x, y))


def read_airfoil_coordinates(filename: Union[str, Path]) -> np.ndarray:
    """Read, orient, and normalize two-column airfoil coordinates.

    Input points may follow an open trailing-edge-to-trailing-edge contour or a
    closed loop. Closed loops are rotated to run from the upper trailing edge,
    around the leading edge, to the lower trailing edge. Header and comment
    lines are ignored; coordinates are normalized to unit chord along x.
    """
    points = []
    with Path(filename).open("r", encoding="utf-8") as coordinate_file:
        for line_number, line in enumerate(coordinate_file, start=1):
            content = line.partition("#")[0].replace(",", " ").split()
            if not content:
                continue
            try:
                x, y = float(content[0]), float(content[1])
            except (ValueError, IndexError):
                if points:
                    try:
                        float(content[0])
                    except ValueError:
                        continue
                    raise ValueError(
                        f"invalid coordinate on line {line_number} of {filename}"
                    )
                continue
            if not math.isfinite(x) or not math.isfinite(y):
                raise ValueError(
                    f"non-finite coordinate on line {line_number} of {filename}"
                )
            points.append((x, y))

    if len(points) < 4:
        raise ValueError("airfoil coordinate file must contain at least 4 points")

    coordinates = np.asarray(points, dtype=float)
    keep = np.concatenate((
        [True],
        np.linalg.norm(np.diff(coordinates, axis=0), axis=1)
        > np.finfo(float).eps,
    ))
    coordinates = coordinates[keep]
    if len(coordinates) < 4:
        raise ValueError("airfoil contour has fewer than 4 distinct consecutive points")

    chord = float(np.ptp(coordinates[:, 0]))
    if chord <= np.finfo(float).eps:
        raise ValueError("airfoil contour must have a nonzero chord along x")
    coordinates[:, 0] = (coordinates[:, 0] - np.min(coordinates[:, 0])) / chord
    coordinates[:, 1] /= chord

    if np.linalg.norm(coordinates[0] - coordinates[-1]) <= 1e-12:
        coordinates = coordinates[:-1]
        te_x = float(np.max(coordinates[:, 0]))
        te_indices = np.flatnonzero(np.isclose(
            coordinates[:, 0], te_x, rtol=0.0, atol=1e-10
        ))
        upper_te = int(te_indices[np.argmax(coordinates[te_indices, 1])])
        lower_te = int(te_indices[np.argmin(coordinates[te_indices, 1])])
        point_count = len(coordinates)
        if upper_te == lower_te:
            indices = np.concatenate((
                np.arange(upper_te, point_count), np.arange(upper_te + 1)
            ))
        else:
            forward_distance = (lower_te - upper_te) % point_count
            direction = 1 if forward_distance > point_count / 2 else -1
            distance = (forward_distance if direction == 1
                        else (upper_te - lower_te) % point_count)
            indices = (upper_te + direction * np.arange(distance + 1)) % point_count
        coordinates = coordinates[indices]

    # PyAero's contour convention starts at the upper trailing edge.
    if coordinates[0, 1] < coordinates[-1, 1]:
        coordinates = coordinates[::-1].copy()
    elif math.isclose(coordinates[0, 1], coordinates[-1, 1], abs_tol=1e-12):
        if coordinates[1, 1] < coordinates[-2, 1]:
            coordinates = coordinates[::-1].copy()

    if (coordinates[0, 0] < 1.0 - 1e-3
            or coordinates[-1, 0] < 1.0 - 1e-3):
        raise ValueError(
            "airfoil coordinates must start and end at the trailing edge"
        )
    return coordinates


def _segment_angle(points: np.ndarray, index: int) -> float:
    incoming = points[index - 1] - points[index]
    outgoing = points[index + 1] - points[index]
    denominator = float(np.linalg.norm(incoming) * np.linalg.norm(outgoing))
    if denominator <= np.finfo(float).eps:
        return 180.0
    cosine = float(np.dot(incoming, outgoing) / denominator)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _refine_leading_edge(tck, parameters: np.ndarray,
                         tolerance: float) -> np.ndarray:
    """Recursively add B-spline samples at sharp contour turns."""
    for _ in range(30):
        points = np.column_stack(interpolate.splev(parameters, tck))
        insertions = []
        for index in range(1, len(parameters) - 1):
            if _segment_angle(points, index) < tolerance:
                insertions.extend((
                    0.5 * (parameters[index - 1] + parameters[index]),
                    0.5 * (parameters[index] + parameters[index + 1]),
                ))
        if not insertions:
            return parameters
        if len(parameters) + len(insertions) > 100000:
            raise ValueError("leading-edge refinement exceeded 100000 points")
        parameters = np.unique(np.concatenate((parameters, insertions)))
    raise ValueError("leading-edge refinement did not converge after 30 passes")


def spline_refine_contour(
    coordinates: np.ndarray,
    points: int = 257,
    tolerance: float = 172.0,
    te_old_segments: int = 3,
    te_new_segments: int = 6,
    te_ratio: float = 3.0,
) -> np.ndarray:
    """Apply PyAero-style cubic B-spline sampling and LE/TE refinements."""
    if coordinates.ndim != 2 or coordinates.shape[1] != 2:
        raise ValueError("coordinates must have shape (number_of_points, 2)")
    if len(coordinates) < 4 or not np.all(np.isfinite(coordinates)):
        raise ValueError("coordinates must contain at least 4 finite points")
    if isinstance(points, bool) or not isinstance(points, int) or points < 10:
        raise ValueError("points must be an integer of at least 10")
    if not math.isfinite(tolerance) or not 0.0 < tolerance < 180.0:
        raise ValueError("tolerance must be finite and between 0 and 180 degrees")
    if (isinstance(te_old_segments, bool)
            or not isinstance(te_old_segments, int) or te_old_segments < 1):
        raise ValueError("te_old_segments must be a positive integer")
    if (isinstance(te_new_segments, bool)
            or not isinstance(te_new_segments, int) or te_new_segments < 1):
        raise ValueError("te_new_segments must be a positive integer")
    if te_old_segments * 2 + 2 >= points:
        raise ValueError("te_old_segments is too large for the spline point count")
    if not math.isfinite(te_ratio) or te_ratio < 1.0:
        raise ValueError("te_ratio must be finite and at least 1")

    tck, _ = interpolate.splprep(coordinates.T, s=0.0, k=3)
    parameters = _refine_leading_edge(
        tck, np.linspace(0.0, 1.0, points), tolerance
    )

    refined_coordinates = np.column_stack(interpolate.splev(parameters, tck))
    refined_tck, parameters = interpolate.splprep(
        refined_coordinates.T, s=0.0, k=3
    )
    spline_points = np.column_stack(interpolate.splev(parameters, refined_tck))

    te_parameter = parameters[te_old_segments]
    if te_new_segments == 1:
        spacing = np.array([0.0, te_parameter])
    else:
        growth = te_ratio ** (1.0 / (te_new_segments - 1))
        widths = growth ** np.arange(te_new_segments + 1, dtype=float)
        spacing = np.concatenate(([0.0], np.cumsum(widths / widths.sum())))
        spacing *= te_parameter

    retained = spline_points[
        te_old_segments + 1:len(spline_points) - te_old_segments - 1
    ]
    upper = np.column_stack(
        interpolate.splev(spacing, refined_tck)
    )
    lower = np.column_stack(
        interpolate.splev(1.0 - spacing[::-1], refined_tck)
    )
    contour = np.vstack((upper, retained, lower))
    return contour


def _unit_normals(line: np.ndarray) -> np.ndarray:
    tangent = np.empty_like(line)
    tangent[0] = line[1] - line[0]
    tangent[-1] = line[-1] - line[-2]
    tangent[1:-1] = line[2:] - line[:-2]
    lengths = np.linalg.norm(tangent, axis=1)
    if np.any(lengths <= np.finfo(float).eps):
        raise ValueError("airfoil contour contains duplicate consecutive points")
    tangent /= lengths[:, np.newaxis]
    return np.column_stack((tangent[:, 1], -tangent[:, 0]))


def _cell_offsets(first_cell_height: float, divisions: int,
                  growth: float) -> np.ndarray:
    """Match BlockMesh.spacing_cell_thickness for the near-wall block."""
    widths = [first_cell_height]
    for _ in range(divisions - 1):
        widths.append(first_cell_height + widths[-1] * growth)
    return np.concatenate(([0.0], np.cumsum(widths)))


def _geometric_line(start: np.ndarray, end: np.ndarray, divisions: int,
                    last_to_first: float = 1.0) -> np.ndarray:
    if divisions < 1:
        raise ValueError("line divisions must be at least 1")
    if not math.isfinite(last_to_first) or last_to_first <= 0.0:
        raise ValueError("line spacing ratio must be finite and positive")
    if divisions == 1:
        fractions = np.array([0.0, 1.0])
    else:
        cell_growth = last_to_first ** (1.0 / (divisions - 1))
        widths = cell_growth ** np.arange(divisions, dtype=float)
        fractions = np.concatenate(([0.0], np.cumsum(widths / widths.sum())))
    return start + fractions[:, np.newaxis] * (end - start)


def _normalized_arclength(line: np.ndarray) -> np.ndarray:
    lengths = np.linalg.norm(np.diff(line, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    total = cumulative[-1]
    if total <= np.finfo(float).eps:
        return np.linspace(0.0, 1.0, len(line))
    return cumulative / total


def _resample_polyline(line: np.ndarray, count: int,
                       distribution: Optional[np.ndarray] = None) -> np.ndarray:
    parameter = _normalized_arclength(line)
    target = (np.linspace(0.0, 1.0, count)
              if distribution is None else distribution)
    return np.column_stack([
        np.interp(target, parameter, line[:, component])
        for component in range(line.shape[1])
    ])


def _transfinite(bottom: np.ndarray, top: np.ndarray, left: np.ndarray,
                 right: np.ndarray) -> np.ndarray:
    """Coons-patch interpolation with bottom/top along i and left/right along j."""
    if len(bottom) != len(top) or len(left) != len(right):
        raise ValueError("opposite transfinite boundaries must have matching point counts")
    if (not np.allclose(bottom[0], left[0])
            or not np.allclose(bottom[-1], right[0])
            or not np.allclose(top[0], left[-1])
            or not np.allclose(top[-1], right[-1])):
        raise ValueError("transfinite boundary endpoints do not coincide")

    xi = _normalized_arclength(bottom)
    eta = _normalized_arclength(left)
    grid = np.empty((len(bottom), len(left), 2), dtype=float)
    c00, c01 = bottom[0], top[0]
    c10, c11 = bottom[-1], top[-1]

    for i, u in enumerate(xi):
        for j, v in enumerate(eta):
            side_blend = (1.0 - u) * left[j] + u * right[j]
            chord_blend = (1.0 - v) * bottom[i] + v * top[i]
            corner_blend = (
                (1.0 - u) * (1.0 - v) * c00
                + (1.0 - u) * v * c01
                + u * (1.0 - v) * c10
                + u * v * c11
            )
            grid[i, j] = side_blend + chord_blend - corner_blend
    return grid


def _remap_i_region(grid: np.ndarray, first: int, last: int) -> None:
    if last - first < 2:
        return
    remapped = _transfinite(
        grid[first:last + 1, 0],
        grid[first:last + 1, -1],
        grid[first],
        grid[last],
    )
    grid[first:last + 1] = remapped


def _smooth_tunnel_spacing(
    tunnel: np.ndarray,
    smoothing_fraction: float,
) -> np.ndarray:
    """Smooth cross-tunnel node spacing along i while preserving all boundaries."""
    if smoothing_fraction == 0.0:
        return tunnel

    parameters = np.stack([_normalized_arclength(line) for line in tunnel])
    sigma = smoothing_fraction * (len(tunnel) - 1)
    smoothed = ndimage.gaussian_filter1d(
        parameters, sigma=sigma, axis=0, mode="nearest"
    )
    coordinate = np.linspace(0.0, 1.0, len(tunnel))
    fade = np.clip(
        np.minimum(coordinate, 1.0 - coordinate) / smoothing_fraction,
        0.0, 1.0,
    )
    weight = fade**2 * (3.0 - 2.0 * fade)
    smoothed = parameters + weight[:, np.newaxis] * (smoothed - parameters)

    result = tunnel.copy()
    for i in range(1, len(tunnel) - 1):
        result[i] = np.column_stack([
            np.interp(smoothed[i], parameters[i], tunnel[i, :, component])
            for component in range(tunnel.shape[2])
        ])
    return result


def _tunnel_distribution(count: int, bias: str) -> np.ndarray:
    limits = {
        "symmetric": (-1.3, 1.3),
        "lower": (-1.2, 1.5),
        "upper": (-1.5, 1.2),
    }
    if bias not in limits:
        raise ValueError("bias must be 'symmetric', 'lower', or 'upper'")
    lower, upper = limits[bias]
    values = np.linspace(lower, upper, count)
    return (np.tanh(values) + 1.0) / 2.0


def generate_four_block_mesh(
    ni: int = 257,
    airfoil_divisions: int = 15,
    airfoil_first_cell_height: float = 0.004,
    airfoil_growth: float = 1.05,
    te_divisions: int = 15,
    te_first_cell_height: float = 0.004,
    te_growth: float = 1.05,
    tunnel_height: float = 3.5,
    tunnel_height_divisions: int = 100,
    tunnel_height_ratio: float = 10.0,
    wake_length: float = 7.0,
    wake_divisions: int = 100,
    wake_ratio: float = 15.0,
    wake_equalize: float = 0.30,
    bias: str = "symmetric",
    airfoil_coordinates: Optional[np.ndarray] = None,
    blunt_te_divisions: int = 3,
    refinement_tolerance: float = 172.0,
    te_old_segments: int = 3,
    te_new_segments: int = 6,
    te_refinement_ratio: float = 3.0,
    tunnel_smoothing: float = 0.1,
) -> Dict[str, np.ndarray]:
    """Generate PyAero's four conformal NACA 0012 mesh blocks.

    Each returned array has shape ``(ni_block, nj_block, 2)``. The block
    keys are ``airfoil``, ``trailing_edge``, ``tunnel``, and ``wake``.
    """
    integer_settings = {
        "ni": ni,
        "airfoil_divisions": airfoil_divisions,
        "te_divisions": te_divisions,
        "blunt_te_divisions": blunt_te_divisions,
        "tunnel_height_divisions": tunnel_height_divisions,
        "wake_divisions": wake_divisions,
    }
    for name, value in integer_settings.items():
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
    if ni < 3 or ni % 2 == 0:
        raise ValueError("ni must be an odd integer of at least 3")
    if any(value < 1 for name, value in integer_settings.items()
           if name != "ni"):
        raise ValueError("all mesh division counts must be positive")
    positive_settings = {
        "airfoil_first_cell_height": airfoil_first_cell_height,
        "airfoil_growth": airfoil_growth,
        "te_first_cell_height": te_first_cell_height,
        "te_growth": te_growth,
        "tunnel_height": tunnel_height,
        "tunnel_height_ratio": tunnel_height_ratio,
        "wake_length": wake_length,
        "wake_ratio": wake_ratio,
    }
    for name, value in positive_settings.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if not 0.0 < wake_equalize < 1.0:
        raise ValueError("wake_equalize must be strictly between 0 and 1")
    if (not math.isfinite(tunnel_smoothing)
            or not 0.0 <= tunnel_smoothing <= 0.5):
        raise ValueError("tunnel_smoothing must be finite and between 0 and 0.5")
    if bias not in ("symmetric", "lower", "upper"):
        raise ValueError("bias must be 'symmetric', 'lower', or 'upper'")

    if airfoil_coordinates is None:
        surface = naca0012_surface(np.linspace(0.0, 1.0, ni))
    else:
        surface = spline_refine_contour(
            airfoil_coordinates,
            points=ni,
            tolerance=refinement_tolerance,
            te_old_segments=te_old_segments,
            te_new_segments=te_new_segments,
            te_ratio=te_refinement_ratio,
        )

    # Block 1: normal layers extruded from the airfoil contour.
    normals = _unit_normals(surface)
    wall_offsets = _cell_offsets(
        airfoil_first_cell_height, airfoil_divisions, airfoil_growth
    )
    airfoil = surface[:, np.newaxis, :] + (
        wall_offsets[np.newaxis, :, np.newaxis]
        * normals[:, np.newaxis, :]
    )

    # Block 2: wake extrusion across the upper/lower trailing-edge boundaries.
    lower_edge = airfoil[-1, ::-1]
    upper_edge = airfoil[0, 1:]
    te_gap = float(np.linalg.norm(surface[0] - surface[-1]))
    if te_gap > 1.0e-10:
        blunt_face = _geometric_line(
            surface[-1], surface[0], blunt_te_divisions
        )
        te_inner = np.vstack((lower_edge, blunt_face[1:], upper_edge))
    else:
        te_inner = np.vstack((lower_edge, upper_edge))
    te_direction = np.mean(_unit_normals(te_inner), axis=0)
    direction_length = float(np.linalg.norm(te_direction))
    if direction_length <= np.finfo(float).eps:
        raise ValueError("could not determine the trailing-edge wake direction")
    te_direction /= direction_length
    te_offsets = _cell_offsets(
        te_first_cell_height, te_divisions, te_growth
    )
    te_outer = te_inner + te_offsets[-1] * te_direction
    te_outer = _resample_polyline(te_outer, len(te_inner))
    trailing_edge = _transfinite(
        bottom=te_inner,
        top=te_outer,
        left=_geometric_line(
            te_inner[0], te_outer[0], te_divisions, te_growth
        ),
        right=_geometric_line(
            te_inner[-1], te_outer[-1], te_divisions, te_growth
        ),
    )

    # Block 3: outer C-shaped tunnel around the airfoil and TE blocks.
    upper_te_stream = trailing_edge[-1, ::-1]
    lower_te_stream = trailing_edge[0]
    inner_tunnel = np.vstack((
        upper_te_stream[:-1],
        airfoil[:, -1][:-1],
        lower_te_stream,
    ))
    upper_corner = np.array([inner_tunnel[0, 0], tunnel_height])
    lower_corner = np.array([inner_tunnel[-1, 0], -tunnel_height])
    outer_curve = [upper_corner]
    outer_curve.extend(
        np.column_stack((
            np.linspace(upper_corner[0], 0.0, 10)[1:],
            np.full(9, tunnel_height),
        ))
    )
    angles = np.deg2rad(np.linspace(90.0, 270.0, 200))
    circle = tunnel_height * np.column_stack((np.cos(angles), np.sin(angles)))
    outer_curve.extend(circle[1:-1])
    outer_curve.extend(
        np.column_stack((
            np.linspace(0.0, lower_corner[0], 10),
            np.full(10, -tunnel_height),
        ))
    )
    outer_curve = np.asarray(outer_curve)
    outer_boundary = _resample_polyline(
        outer_curve,
        len(inner_tunnel),
        _tunnel_distribution(len(inner_tunnel), bias),
    )
    tunnel = _transfinite(
        bottom=inner_tunnel,
        top=outer_boundary,
        left=_geometric_line(
            inner_tunnel[0], outer_boundary[0],
            tunnel_height_divisions, tunnel_height_ratio,
        ),
        right=_geometric_line(
            inner_tunnel[-1], outer_boundary[-1],
            tunnel_height_divisions, tunnel_height_ratio,
        ),
    )

    # Use PyAero's normal-projection blend near the inner C-shaped boundary.
    tunnel_normals = _unit_normals(inner_tunnel)
    for j in range(1, tunnel.shape[1] - 1):
        fraction = (j / len(tunnel[0])) ** 0.3
        displacement = tunnel[:, j] - inner_tunnel
        projected_distance = np.sum(
            displacement * tunnel_normals, axis=1
        )
        normal_points = inner_tunnel + (
            projected_distance[:, np.newaxis] * tunnel_normals
        )
        tunnel[:, j] = (
            (1.0 - fraction) * normal_points + fraction * tunnel[:, j]
        )

    # Keep the first/last 30 i-lines aligned as in Windtunnel.TunnelMesh.
    edge_patch = min(30, (len(tunnel) - 1) // 2)
    if edge_patch >= 2:
        _remap_i_region(tunnel, 0, edge_patch)
        _remap_i_region(tunnel, len(tunnel) - edge_patch - 1, len(tunnel) - 1)

    tunnel = _smooth_tunnel_spacing(tunnel, tunnel_smoothing)

    # Block 4: wake from the TE/tunnel join to the downstream outlet.
    lower_tunnel_edge = tunnel[-1, ::-1]
    upper_tunnel_edge = tunnel[0]
    wake_left = np.vstack((
        lower_tunnel_edge[:-1],
        trailing_edge[:, -1][:-1],
        upper_tunnel_edge,
    ))
    lower_outlet = np.array([1.0 + wake_length, -tunnel_height])
    upper_outlet = np.array([1.0 + wake_length, tunnel_height])
    lower_upstream = wake_left[0]
    upper_upstream = wake_left[-1]
    wake_bottom = _geometric_line(
        lower_upstream, lower_outlet, wake_divisions, wake_ratio
    )
    wake_top = _geometric_line(
        upper_upstream, upper_outlet, wake_divisions, wake_ratio
    )
    wake_right = np.column_stack((
        np.full(len(wake_left), 1.0 + wake_length),
        np.linspace(-tunnel_height, tunnel_height, len(wake_left)),
    ))
    wake = _transfinite(
        bottom=wake_bottom,
        top=wake_top,
        left=wake_left,
        right=wake_right,
    )

    threshold = 1.0 + wake_length * wake_equalize
    split = int(np.searchsorted(wake_bottom[:, 0], threshold, side="left"))
    split = min(max(split, 1), len(wake_bottom) - 2)
    wake[split] = _resample_polyline(wake[split], len(wake_left))
    _remap_i_region(wake, 0, split)
    _remap_i_region(wake, split, len(wake) - 1)

    return {
        "airfoil": airfoil,
        "trailing_edge": trailing_edge,
        "tunnel": tunnel,
        "wake": wake,
    }


def write_plot3d(
    filename: Union[str, Path],
    blocks: Union[np.ndarray, Sequence[np.ndarray]],
    span_length: float = 1.0,
    spanwise_cells: int = 1,
) -> None:
    """Extrude blocks uniformly in span and write ASCII multi-block Plot3D."""
    if isinstance(blocks, np.ndarray):
        block_list = [blocks]
    else:
        block_list = list(blocks)
    if not block_list:
        raise ValueError("at least one block is required")
    if not math.isfinite(span_length) or span_length <= 0.0:
        raise ValueError("span_length must be finite and positive")
    if (isinstance(spanwise_cells, bool) or not isinstance(spanwise_cells, int)
            or spanwise_cells < 1):
        raise ValueError("spanwise_cells must be a positive integer")

    plot3d_blocks = []
    for block in block_list:
        if block.ndim != 3 or block.shape[2] != 2:
            raise ValueError("each block must have shape (ni, nj, 2)")
        if block.shape[0] < 2 or block.shape[1] < 2:
            raise ValueError("each block needs at least 2 points in each direction")
        if not np.all(np.isfinite(block)):
            raise ValueError("block coordinates must all be finite")

        ni, nj, _ = block.shape
        x = np.repeat(
            block[:, :, 0, np.newaxis], spanwise_cells + 1, axis=2
        )
        y = np.repeat(
            block[:, :, 1, np.newaxis], spanwise_cells + 1, axis=2
        )
        z = np.broadcast_to(
            np.linspace(0.0, span_length, spanwise_cells + 1)
            [np.newaxis, np.newaxis, :],
            (ni, nj, spanwise_cells + 1),
        ).copy()
        plot3d_blocks.append(np.stack((x, y, z)))

    _write_plot3d(Path(filename), plot3d_blocks, formatted=True)


def write_vtk(filename: Union[str, Path],
              blocks: Union[np.ndarray, Sequence[np.ndarray]],
              span_length: float = 1.0,
              spanwise_cells: int = 1) -> None:
    """Write the 3D structured blocks to a VTK XML multiblock dataset."""
    import pyvista as pv

    if isinstance(blocks, np.ndarray):
        block_list = [blocks]
    else:
        block_list = list(blocks)
    if not block_list:
        raise ValueError("at least one block is required")
    if not math.isfinite(span_length) or span_length <= 0.0:
        raise ValueError("span_length must be finite and positive")
    if (isinstance(spanwise_cells, bool) or not isinstance(spanwise_cells, int)
            or spanwise_cells < 1):
        raise ValueError("spanwise_cells must be a positive integer")

    multiblock = pv.MultiBlock()
    for index, block in enumerate(block_list):
        if block.ndim != 3 or block.shape[2] != 2:
            raise ValueError("each block must have shape (ni, nj, 2)")
        if block.shape[0] < 2 or block.shape[1] < 2:
            raise ValueError("each block needs at least 2 points in each direction")
        if not np.all(np.isfinite(block)):
            raise ValueError("block coordinates must all be finite")

        ni, nj, _ = block.shape
        x = np.repeat(
            block[:, :, 0, np.newaxis], spanwise_cells + 1, axis=2
        )
        y = np.repeat(
            block[:, :, 1, np.newaxis], spanwise_cells + 1, axis=2
        )
        z = np.broadcast_to(
            np.linspace(0.0, span_length, spanwise_cells + 1)
            [np.newaxis, np.newaxis, :],
            (ni, nj, spanwise_cells + 1),
        )
        multiblock.append(pv.StructuredGrid(x, y, z), name=f"block_{index + 1}")

    multiblock.save(str(Path(filename)))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "output", type=Path,
        help="output file (.xyz/.p3d for Plot3D or .vtm for VTK multiblock)",
    )
    parser.add_argument(
        "--airfoil", type=Path,
        default=Path(__file__).resolve().parent.parent / "tests" / "data" / "a-airfoil.dat",
        help="two-column airfoil coordinate file (default: tests/data/a-airfoil.dat)",
    )
    parser.add_argument(
        "--ni", type=int, default=1025,
        help="initial contour spline point count (odd)",
    )
    parser.add_argument("--refinement-tolerance", type=float, default=175.0,
                        help="minimum included angle in degrees for leading-edge refinement")
    parser.add_argument("--te-old-segments", type=int, default=1,
                        help="trailing-edge segments on each side to replace")
    parser.add_argument("--te-new-segments", type=int, default=1,
                        help="new trailing-edge segments on each side")
    parser.add_argument("--te-refinement-ratio", type=float, default=2.0,
                        help="trailing-edge refinement spacing ratio")
    parser.add_argument(
        "--blunt-te-divisions", type=int, default=3,
        help="number of cells across a blunt trailing edge (detected from endpoint gap)",
    )
    parser.add_argument("--airfoil-divisions", type=int, default=35)
    parser.add_argument("--airfoil-first-cell-height", type=float, default=0.00015)
    parser.add_argument("--airfoil-growth", type=float, default=1.05)
    parser.add_argument("--te-divisions", type=int, default=15)
    parser.add_argument("--te-first-cell-height", type=float, default=0.0003)
    parser.add_argument("--te-growth", type=float, default=1.05)
    parser.add_argument("--tunnel-height", type=float, default=15.0)
    parser.add_argument("--tunnel-height-divisions", type=int, default=100)
    parser.add_argument("--tunnel-height-ratio", type=float, default=20.0)
    parser.add_argument(
        "--tunnel-smoothing", type=float, default=0.1,
        help="smooth cross-tunnel spacing over this fraction of the block's i-lines (0 disables)",
    )
    parser.add_argument("--wake-length", type=float, default=15.0)
    parser.add_argument("--wake-divisions", type=int, default=150)
    parser.add_argument("--wake-ratio", type=float, default=15.0)
    parser.add_argument("--span-length", type=float, default=0.0481,
                        help="periodic spanwise length in chord lengths (default: 0.2)")
    parser.add_argument("--spanwise-spacing", type=float, default=0.0012025,
                        help="uniform spanwise cell spacing in chord lengths (default: 0.01)")
    parser.add_argument("--wake-equalize", type=float, default=0.30,
                        help="wake equalization location as a fraction of wake length")
    parser.add_argument("--bias", choices=("symmetric", "lower", "upper"),
                        default="symmetric")
    args = parser.parse_args(argv)

    if (not math.isfinite(args.spanwise_spacing)
            or args.spanwise_spacing <= 0.0):
        parser.error("--spanwise-spacing must be finite and positive")
    if not math.isfinite(args.span_length) or args.span_length <= 0.0:
        parser.error("--span-length must be finite and positive")
    cell_count = args.span_length / args.spanwise_spacing
    spanwise_cells = round(cell_count)
    if (spanwise_cells < 1
            or not math.isclose(cell_count, spanwise_cells, rel_tol=1e-10, abs_tol=1e-10)):
        parser.error("--span-length must be an integer multiple of --spanwise-spacing")

    coordinates = read_airfoil_coordinates(args.airfoil)
    blocks = generate_four_block_mesh(
        ni=args.ni,
        airfoil_divisions=args.airfoil_divisions,
        airfoil_first_cell_height=args.airfoil_first_cell_height,
        airfoil_growth=args.airfoil_growth,
        te_divisions=args.te_divisions,
        te_first_cell_height=args.te_first_cell_height,
        te_growth=args.te_growth,
        tunnel_height=args.tunnel_height,
        tunnel_height_divisions=args.tunnel_height_divisions,
        tunnel_height_ratio=args.tunnel_height_ratio,
        wake_length=args.wake_length,
        wake_divisions=args.wake_divisions,
        wake_ratio=args.wake_ratio,
        wake_equalize=args.wake_equalize,
        bias=args.bias,
        airfoil_coordinates=coordinates,
        blunt_te_divisions=args.blunt_te_divisions,
        refinement_tolerance=args.refinement_tolerance,
        te_old_segments=args.te_old_segments,
        te_new_segments=args.te_new_segments,
        te_refinement_ratio=args.te_refinement_ratio,
        tunnel_smoothing=args.tunnel_smoothing,
    )
    if args.output.suffix.lower() in (".vtm", ".vtmb", ".vtkhdf"):
        write_vtk(
            args.output, list(blocks.values()), span_length=args.span_length,
            spanwise_cells=spanwise_cells,
        )
        output_format = "VTK multiblock"
    elif args.output.suffix.lower() in (".xyz", ".p3d"):
        write_plot3d(
            args.output, list(blocks.values()), span_length=args.span_length,
            spanwise_cells=spanwise_cells,
        )
        output_format = "Plot3D"
    else:
        parser.error("output extension must be .xyz, .p3d, .vtm, .vtmb, or .vtkhdf")
    print(f"Wrote four-block {output_format} mesh to {args.output}")
    print(f"  airfoil: {args.airfoil}")
    print(f"  span: {args.span_length:g} c, dz: {args.spanwise_spacing:g} c "
          f"({spanwise_cells} uniform cells)")
    print(f"  tunnel spacing smoothing: {args.tunnel_smoothing:g} of i-lines")
    for name, block in blocks.items():
        print(f"  {name}: {block.shape[0]} x {block.shape[1]} points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
