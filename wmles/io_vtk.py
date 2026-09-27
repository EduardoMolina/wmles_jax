"""VTU/PVD export via pyvista, for visualization in ParaView / pyvista itself.

Needs `pip install pyvista` (and vtk, which it pulls in). Nothing else in this
package imports pyvista, so the solver runs fine without it -- this module is
only touched when you actually export.

* `block_grid`     -> one `pyvista.UnstructuredGrid` (hexahedral) per block, no
                      ghost cells, with whatever cell fields you pass in.
* `wall_patch_grid`-> a 2D quad-mesh grid of one wall face, for tau_w / q_w.
* `derived_fields` -> {block index: {name: array}} of the usual flow variables
                      (+ vorticity magnitude, Q-criterion, mu_t) from a solver state.
* `write_case`     -> writes one .vtu per block (+ optional wall-patch .vtu's) for
                      one time step, and appends to a .pvd collection so the whole
                      run opens as a single time-aware object in ParaView.

Example
-------
    from wmles.io_vtk import write_case
    writer = None
    for it in range(nsteps):
        ...
        if it % 50 == 0:
            writer = write_case(mesh, solver, Q, it, t, "out/channel", writer=writer)
"""
import os
import numpy as np

try:
    import pyvista as pv
    HAS_PYVISTA = True
except ImportError:
    HAS_PYVISTA = False

from .viscous import green_gauss

_HEX_TYPE, _QUAD_TYPE = None, None


def _cell_types():
    global _HEX_TYPE, _QUAD_TYPE
    if _HEX_TYPE is None:
        if hasattr(pv, "CellType"):
            _HEX_TYPE, _QUAD_TYPE = int(pv.CellType.HEXAHEDRON), int(pv.CellType.QUAD)
        else:                                  # older pyvista: raw VTK ints
            _HEX_TYPE, _QUAD_TYPE = 12, 9
    return _HEX_TYPE, _QUAD_TYPE


def _require_pyvista():
    if not HAS_PYVISTA:
        raise ImportError("VTU export needs pyvista: pip install pyvista")


def _set_cell_data(grid, name, arr, ncell_shape):
    """arr: (ncell_shape...) scalar, or (ncomp, ncell_shape...) vector -- C order."""
    arr = np.asarray(arr)
    ndim_extra = arr.ndim - len(ncell_shape)
    assert arr.shape[ndim_extra:] == ncell_shape, \
        f"field '{name}' has shape {arr.shape}, expected (...,{ncell_shape})"
    if ndim_extra == 0:
        grid.cell_data[name] = arr.ravel(order="C")
    elif ndim_extra == 1:
        grid.cell_data[name] = np.stack([c.ravel(order="C") for c in arr], axis=1)
    else:
        raise ValueError(f"field '{name}': only scalar or single-vector fields are supported")


def _hex_connectivity(ni, nj, nk):
    """VTK_HEXAHEDRON connectivity over a (ni,nj,nk) cell block whose (ni+1,nj+1,nk+1)
    nodes are numbered in C order (k fastest, i slowest) -- matches `nodes.reshape(3,-1)`."""
    i, j, k = np.meshgrid(np.arange(ni), np.arange(nj), np.arange(nk), indexing="ij")

    def nid(di, dj, dk):
        return ((i + di) * (nj + 1) + (j + dj)) * (nk + 1) + (k + dk)

    order = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0),
             (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]
    conn = np.stack([nid(*o).ravel(order="C") for o in order], axis=1)         # (ncell,8)
    cells = np.concatenate([np.full((len(conn), 1), 8, dtype=np.int64), conn], axis=1).ravel()
    ctype, _ = _cell_types()
    return cells, np.full(len(conn), ctype, dtype=np.uint8)


def _quad_connectivity(n1, n2):
    """VTK_QUAD connectivity for an (n1,n2) face patch, nodes C-order ((n1+1)*(n2+1))."""
    i, j = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")

    def nid(di, dj):
        return (i + di) * (n2 + 1) + (j + dj)

    order = [(0, 0), (1, 0), (1, 1), (0, 1)]
    conn = np.stack([nid(*o).ravel(order="C") for o in order], axis=1)
    cells = np.concatenate([np.full((len(conn), 1), 4, dtype=np.int64), conn], axis=1).ravel()
    _, qtype = _cell_types()
    return cells, np.full(len(conn), qtype, dtype=np.uint8)


def block_grid(blk, fields=None):
    """Hexahedral `pyvista.UnstructuredGrid` of one block's physical cells (no ghosts).
    `fields`: {name: array}, array.shape[-3:] == blk.n (scalar) or (3,)+blk.n (vector)."""
    _require_pyvista()
    ni, nj, nk = blk.n
    pts = np.moveaxis(np.asarray(blk.x_int), 0, -1).reshape(-1, 3)
    cells, ctypes = _hex_connectivity(ni, nj, nk)
    grid = pv.UnstructuredGrid(cells, ctypes, pts)
    for name, arr in (fields or {}).items():
        _set_cell_data(grid, name, arr, (ni, nj, nk))
    return grid


def wall_patch_grid(blk, axis, side, fields=None):
    """Quad-mesh `pyvista.UnstructuredGrid` of one wall face (for tau_w, q_w, y+...).
    `fields`: {name: array}, array.shape == the two tangential cell counts of that face."""
    _require_pyvista()
    tang = [a for a in range(3) if a != axis]
    n1, n2 = blk.n[tang[0]], blk.n[tang[1]]
    layer = 0 if side == 0 else blk.n[axis]
    X = np.moveaxis(np.asarray(blk.x_int), 0, -1)                 # (ni+1,nj+1,nk+1,3)
    plane = np.take(X, layer, axis=axis).reshape(-1, 3)            # C order in the two tangential axes
    cells, ctypes = _quad_connectivity(n1, n2)
    grid = pv.UnstructuredGrid(cells, ctypes, plane)
    for name, arr in (fields or {}).items():
        _set_cell_data(grid, name, arr, (n1, n2))
    return grid


def derived_fields(solver, Q):
    """{block index: {name: array}} of primitives + T, Mach, mu_t, |omega|, Q-criterion.
    Arrays are ghost-free, shape == that block's blk.n (or (3,)+blk.n for vectors)."""
    gas = solver.gas
    Q = solver.apply_bcs(Q, solver.geoms)          # need one valid ghost layer for gradients
    out = {}
    for ib, (blk, q, geom) in enumerate(zip(solver.blocks, Q, solver.geoms)):
        w, T = gas.cons_to_prim(q)
        mu = gas.mu(T)
        g = green_gauss(w[1:4], geom)
        mut = solver.sgs(g, geom.vol ** (1.0 / 3.0), w[0])
        wxy, wxz, wyz = g[1, 0] - g[0, 1], g[2, 0] - g[0, 2], g[2, 1] - g[1, 2]
        omega_mag = np.sqrt(np.asarray(wxy) ** 2 + np.asarray(wxz) ** 2 + np.asarray(wyz) ** 2)
        S = 0.5 * (g + np.swapaxes(g, 0, 1))
        Om = 0.5 * (g - np.swapaxes(g, 0, 1))
        Qcrit = 0.5 * (np.sum(np.asarray(Om) ** 2, axis=(0, 1)) - np.sum(np.asarray(S) ** 2, axis=(0, 1)))
        M = np.asarray(np.sqrt(np.sum(w[1:4] ** 2, axis=0)) / gas.sound_speed(T))
        out[ib] = {
            "rho": blk.interior(w[0]), "velocity": blk.interior(w[1:4]),
            "pressure": blk.interior(w[4]), "temperature": blk.interior(T),
            "mach": blk.interior(M), "mu_t": blk.interior(mut), "mu": blk.interior(mu),
            "vorticity_mag": blk.interior(omega_mag), "q_criterion": blk.interior(Qcrit),
        }
        out[ib] = {k: np.asarray(v) for k, v in out[ib].items()}
    return out


class VTKWriter:
    """Holds the .pvd collection state across time steps for one case."""

    def __init__(self, outdir, name="case"):
        _require_pyvista()
        self.outdir, self.name = outdir, name
        os.makedirs(outdir, exist_ok=True)
        self.entries = []          # (time, relative_vtu_path)

    def _write_pvd(self):
        lines = ['<?xml version="1.0"?>',
                '<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">',
                '  <Collection>']
        for t, path in self.entries:
            lines.append(f'    <DataSet timestep="{t:.9g}" part="0" file="{path}"/>')
        lines += ['  </Collection>', '</VTKFile>', '']
        with open(os.path.join(self.outdir, f"{self.name}.pvd"), "w") as f:
            f.write("\n".join(lines))

    def write_step(self, mesh, solver, Q, step, t, extra_fields=None):
        fields = derived_fields(solver, Q)
        if extra_fields:
            for ib, fd in extra_fields.items():
                fields.setdefault(ib, {}).update(fd)
        for ib, blk in enumerate(mesh.blocks):
            grid = block_grid(blk, fields[ib])
            rel = f"{self.name}_block{ib}_{step:06d}.vtu"
            grid.save(os.path.join(self.outdir, rel), binary=True)
            self.entries.append((t, rel))
        self._write_pvd()
        return self


def write_wall_patches(mesh, solver, Q, step, outdir, name="wall"):
    """One .vtu per wall-modelled face, with tau_w and q_w as cell data."""
    _require_pyvista()
    os.makedirs(outdir, exist_ok=True)
    from .bc import WallModel
    Qb = solver.apply_bcs(Q, solver.geoms)
    paths = []
    for blk, q, geom in zip(solver.blocks, Qb, solver.geoms):
        w, T = solver.gas.cons_to_prim(q)
        for (axis, side), bc in blk.faces.items():
            if isinstance(bc, WallModel):
                _, tau, qw = bc.wall_flux(w, T, blk, geom, solver.gas, axis, side)
                grid = wall_patch_grid(blk, axis, side,
                                       {"tau_w": np.asarray(tau), "q_w": np.asarray(qw)})
                p = os.path.join(outdir, f"{name}_{blk.name}_{'ijk'[axis]}{side}_{step:06d}.vtu")
                grid.save(p, binary=True)
                paths.append(p)
    return paths


def write_case(mesh, solver, Q, step, t, outdir, name="case", writer=None, wall=True):
    """Convenience one-liner: writes block VTUs (+ pvd) and, if `wall`, wall patches.
    Pass back the returned `writer` on the next call to keep appending to the same .pvd."""
    if writer is None:
        writer = VTKWriter(outdir, name)
    writer.write_step(mesh, solver, Q, step, t)
    if wall:
        write_wall_patches(mesh, solver, Q, step, os.path.join(outdir, "wall"), name=name)
    return writer