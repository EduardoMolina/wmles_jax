"""Plot3D / NMF reader tests.   python -m tests.test_plot3d_nmf
(set WMLES_BACKEND=numpy to force NumPy)

The airfoil grid in tests/data is the coarsest (L5) NASA TMBWG 'Airfoil Near-Wake' grid,
https://tmbwg.github.io/turbmodels/airfoilwake500c_grids.html
"""
import os
import tempfile
import textwrap

import numpy as np

from wmles import *
from wmles.backend import xp
from wmles.mesh import sl
from tests.test_multiblock import nodes_from_index, prim_fn, gather, L

DATA = os.path.join(os.path.dirname(__file__), "data")
gas = Gas()


def _nmf(text):
    with tempfile.NamedTemporaryFile("w", suffix=".nmf", delete=False) as f:
        f.write(textwrap.dedent(text))
    try:
        return read_nmf(f.name)
    finally:
        os.unlink(f.name)


def scaled_freestream_residual(mesh, solver, prim):
    """max |rhs| relative to (flux scale) x (face-area / volume) -> round-off level if the metrics close."""
    w = np.asarray(prim, float)
    E = w[4] / (gas.gamma - 1) + 0.5 * w[0] * np.sum(w[1:4] ** 2)
    F0 = (E + w[4]) * np.linalg.norm(w[1:4]) + w[4]
    Q = solver.init_state(lambda xc: xp.stack([w[i] + 0 * xc[0] for i in range(5)]))
    R = solver.rhs(Q)
    worst = 0.0
    for blk, r, g in zip(mesh.blocks, R, solver.geoms):
        av = 0.0
        for d in range(3):
            S = np.asarray(blk.slab(g.S[d], d, 0, 1))
            lo, hi = S[sl(4, d, slice(None, -1))], S[sl(4, d, slice(1, None))]
            av = av + (np.linalg.norm(lo, axis=0) + np.linalg.norm(hi, axis=0))
        av = av / np.asarray(blk.interior(g.vol))
        res = np.abs(np.asarray(blk.interior(r))).max(axis=0)
        worst = max(worst, float((res / (F0 * av)).max()))
    return worst


# ----------------------------------------------------------------------------------- Plot3D
def test_plot3d_roundtrip():
    rng = np.random.default_rng(0)
    nodes = [rng.normal(size=(3, 4, 5, 6)), rng.normal(size=(3, 3, 7, 4))]
    ibl = [rng.integers(0, 2, size=X.shape[1:]) for X in nodes]
    cases = [(True, "f8", "<", False), (True, "f8", "<", True), (False, "f8", "<", False),
             (False, "f8", ">", True), (False, "f4", "<", False)]
    with tempfile.TemporaryDirectory() as d:
        for formatted, dt, en, use_ibl in cases:
            path = os.path.join(d, "g.p3d")
            write_plot3d(path, nodes, formatted=formatted, iblank=ibl if use_ibl else None, dtype=dt, endian=en)
            g = read_plot3d(path)
            assert g.formatted == formatted and g.ndim == 3
            tol = dict(rtol=1e-6, atol=1e-6) if dt == "f4" else dict(rtol=0, atol=1e-13)
            for a, b in zip(g.nodes, nodes):
                assert a.shape == b.shape and np.allclose(a, b, **tol)
            if use_ibl:
                assert all(np.array_equal(a, b) for a, b in zip(g.iblank, ibl))
            else:
                assert g.iblank is None
    print("  plot3d round trip (formatted/unformatted, LE/BE, f4/f8, iblank): ok")


def test_plot3d_2d():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "g2.p2dfmt")
        ni, nj = 4, 3
        x, y = np.meshgrid(np.linspace(0, 1, ni), np.linspace(0, 2, nj), indexing="ij")
        with open(path, "w") as f:
            f.write(f"1\n{ni} {nj}\n")
            f.write(" ".join(f"{v:.15E}" for v in np.concatenate([x.ravel(order="F"), y.ravel(order="F")])) + "\n")
        g = read_plot3d(path)
        assert g.ndim == 2 and g.nodes[0].shape == (3, ni, nj, 1)
        assert np.allclose(g.nodes[0][0, :, :, 0], x) and np.allclose(g.nodes[0][1, :, :, 0], y)
        ext = extrude_2d(g.nodes, 5, 0.3)
        assert ext[0].shape == (3, ni, nj, 6) and np.isclose(ext[0][2].max(), 0.3)
    print("  2-D plot3d read + extrude_2d: ok")


# ----------------------------------------------------------------------------------- NMF text
def test_nmf_parse():
    nmf = _nmf("""\
        # header \\
        # Block#   IDIM   JDIM   KDIM \\
        2 \\
        1 5 6 7 \\
        2 7 5 6 \\
        # Type B1 F1 S1 E1 S2 E2 B2 F2 S1 E1 S2 E2 Swap
        WALL      1 1  1 5  1 6
        Symmetry-Y 1 2  1 5  1 6
        ONE_TO_ONE 1 4  1 6  1 7   2 5  6 1  1 7  TRUE
        """)
    assert nmf.dims == [(5, 6, 7), (7, 5, 6)]
    assert [p.type for p in nmf.patches] == ["wall", "symmetry_y", "one_to_one"]
    o = nmf.patches[2]
    assert o.block == 0 and o.face == 4 and o.partner == dict(block=1, face=5, ranges=((6, 1), (1, 7)), swap=True)
    print("  nmf parsing (comments, continuation, case, swap flag): ok")


# ----------------------------------------------------------------------------------- topology
def test_swap_and_reversed_ranges_vs_single_block():
    """Two blocks split along x; block 2 is transposed AND flipped so the interface needs Swap=TRUE and
    a reversed range. The multiblock RHS must equal the single-block RHS."""
    n = (12, 10, 8)
    h = n[0] // 2
    ix, iy, iz = (np.arange(m + 1) for m in n)
    nodes = nodes_from_index(ix, iy, iz, n)
    prim0 = np.array([1.2, 30.0, 0.0, 0.0, 1e5])
    ff = Dirichlet(prim0)

    # reference: single block, Dirichlet in x, periodic in y and z
    ref = Block("ref", nodes)
    m0 = Mesh([ref])
    m0.connect(ref, "jmin", ref, "jmax", shift=(0, -L[1], 0))
    m0.connect(ref, "kmin", ref, "kmax", shift=(0, 0, -L[2]))
    ref.set_bc("imin", ff)
    ref.set_bc("imax", ff)
    m0.finalize()

    A = nodes[:, :h + 1]
    B = np.transpose(nodes[:, h:], (0, 2, 1, 3))[:, :, :, ::-1]         # (i',j',k') = (y, x, -z)
    ny, nz, nxb = n[1] + 1, n[2] + 1, n[0] - h + 1
    nmf = _nmf(f"""\
        2
        1 {h + 1} {ny} {nz}
        2 {ny} {nxb} {nz}
        one_to_one 1 4  1 {ny} 1 {nz}   2 5  {nz} 1 1 {ny}  true
        ffx 1 3  1 {ny} 1 {nz}
        ffx 2 6  1 {nz} 1 {ny}
        py_a 1 5  1 {nz} 1 {h + 1}
        py_a 1 6  1 {nz} 1 {h + 1}
        pz   1 1  1 {h + 1} 1 {ny}
        pz   1 2  1 {h + 1} 1 {ny}
        py_b 2 3  1 {nxb} 1 {nz}
        py_b 2 4  1 {nxb} 1 {nz}
        pz   2 1  1 {ny} 1 {nxb}
        pz   2 2  1 {ny} 1 {nxb}
        """)
    m1 = build_mesh([A, B], nmf, bc_map={"ffx": ff}, periodic=(("py_a", 1), ("pz", 2), ("py_b", 0)))
    assert len(m1.blocks) == 2
    s0, s1 = Solver(m0, gas, KEP(4), sgs="vreman"), Solver(m1, gas, KEP(4), sgs="vreman")
    R0 = gather(s0, s0.rhs(s0.init_state(prim_fn)))
    R1 = gather(s1, s1.rhs(s1.init_state(prim_fn)))
    assert R0.keys() == R1.keys()
    scale = np.abs(np.array(list(R0.values()))).max()
    err = max(np.abs(R0[k] - R1[k]).max() for k in R0) / scale
    print(f"  swap=TRUE + reversed range vs single block: max rel diff = {err:.2e}")
    assert err < 1e-9


def _box(x, y, z, nx, ny, nz):
    """Nodes of a box, smoothly perturbed by a function of global coordinates (so blocks conform)."""
    X = np.stack(np.meshgrid(np.linspace(*x, nx + 1), np.linspace(*y, ny + 1), np.linspace(*z, nz + 1),
                             indexing="ij"))
    P = X.copy()
    P[0] += 0.04 * np.sin(3 * X[1]) * np.sin(2 * X[2])
    P[1] += 0.03 * np.sin(2 * X[0]) * np.cos(X[2])
    P[2] += 0.03 * np.sin(2 * X[0]) * np.sin(3 * X[1])
    return P


def test_cascade_split():
    """A's jmax face is joined half to B and half to C -> A must be cut at k=5. A's imax face is joined
    whole to D, so the cut has to propagate into D (otherwise the partner sub-face cannot be found)."""
    A = _box((0, 1), (0, 1), (0, 2), 6, 6, 8)
    B = _box((0, 1), (1, 2), (0, 1), 6, 6, 4)
    C = _box((0, 1), (1, 2), (1, 2), 6, 6, 4)
    D = _box((1, 2), (0, 1), (0, 2), 6, 6, 8)
    lines = ["4", "1 7 7 9", "2 7 7 5", "3 7 7 5", "4 7 7 9",
             "one_to_one 1 6  1 5 1 7   2 5  1 5 1 7  false",
             "one_to_one 1 6  5 9 1 7   3 5  1 5 1 7  false",
             "one_to_one 2 2  1 7 1 7   3 1  1 7 1 7  false",
             "one_to_one 1 4  1 7 1 9   4 3  1 7 1 9  false"]
    ff = lambda b, f, r1, r2: lines.append(f"ff {b} {f} {r1[0]} {r1[1]} {r2[0]} {r2[1]}")
    full = (1, 7)
    for f, r in ((1, (full, full)), (2, (full, full)), (3, (full, (1, 9))), (5, ((1, 9), full))):
        ff(1, f, *r)
    for f, r in ((1, (full, full)), (3, (full, (1, 5))), (4, (full, (1, 5))), (6, ((1, 5), full))):
        ff(2, f, *r)
    for f, r in ((2, (full, full)), (3, (full, (1, 5))), (4, (full, (1, 5))), (6, ((1, 5), full))):
        ff(3, f, *r)
    for f, r in ((1, (full, full)), (2, (full, full)), (4, (full, (1, 9))), (5, ((1, 9), full)),
                 (6, ((1, 9), full))):
        ff(4, f, *r)
    nmf = _nmf("\n".join(lines) + "\n")
    prim0 = np.array([1.2, 40.0, -10.0, 20.0, 1e5])
    mesh = build_mesh([A, B, C, D], nmf, bc_map={"ff": Dirichlet(prim0)})
    names = sorted(b.name for b in mesh.blocks)
    assert names == ["b1_000", "b1_001", "b2", "b3", "b4_000", "b4_001"], names
    cells = sum(int(np.prod(b.n)) for b in mesh.blocks)
    assert cells == 6 * 6 * 8 * 2 + 6 * 6 * 4 * 2
    solver = Solver(mesh, gas, Hybrid(), sgs="vreman")
    r = scaled_freestream_residual(mesh, solver, prim0)
    print(f"  cascade split (6 sub-blocks) free-stream residual (scaled) = {r:.2e}")
    assert r < 1e-11


def test_airfoil_wake_grid():
    up = os.path.join(DATA, "nak_a_ff_500c_2zn_L5")
    grid = read_plot3d(up + ".p3dfmt")
    nmf = read_nmf(up + ".nmf")
    assert [X.shape[1:] for X in grid.nodes] == [(2, 43, 57), (2, 65, 29)] and grid.iblank is None
    nodes, nmf = respan(grid.nodes, nmf, ncells=4, length=0.1)
    prim0 = np.array([1.2, 60.0, 0.0, 15.0, 1e5])
    ff = Dirichlet(prim0)
    mesh = build_mesh(nodes, nmf, bc_map={"farfield_riem": ff, "viscous_solid": NoSlipWall()},
                      periodic=(("symmetry_y", 0),))
    assert sorted(b.name for b in mesh.blocks) == ["b1_000", "b1_001", "b2"]        # wake block cut at the TE line
    assert sum(int(np.prod(b.n)) for b in mesh.blocks) == 4 * (42 * 56 + 64 * 28)
    walls = [(b.name, f) for b in mesh.blocks for f, bc in b.faces.items() if isinstance(bc, NoSlipWall)]
    assert walls == [("b2", (2, 0))]                                              # airfoil = b2.kmin only
    # total volume must equal the volume of the original (uncut, un-respanned) cells x span
    from wmles.mesh import compute_geometry
    vol = sum(float(np.asarray(b.interior(b.geom.vol)).sum()) for b in mesh.blocks)
    vol_ref = sum(float(compute_geometry(X)[0].sum()) for X in nodes)
    assert abs(vol - vol_ref) < 1e-12 * vol_ref
    # free stream (airfoil surface treated as a Dirichlet face so uniform flow is a legitimate solution)
    mesh_fs = build_mesh(nodes, nmf, bc_map={"farfield_riem": ff, "viscous_solid": ff},
                         periodic=(("symmetry_y", 0),))
    for sc in (KEP(4), MUSCLHLLC(), Hybrid()):
        r = scaled_freestream_residual(mesh_fs, Solver(mesh_fs, gas, sc, sgs="vreman"), prim0)
        print(f"  airfoil wake grid L5, {type(sc).__name__:10s} free-stream residual (scaled) = {r:.2e}")
        assert r < 1e-11


if __name__ == "__main__":
    from wmles.backend import HAS_JAX
    print("backend:", "jax" if HAS_JAX else "numpy")
    test_plot3d_roundtrip()
    test_plot3d_2d()
    test_nmf_parse()
    test_swap_and_reversed_ranges_vs_single_block()
    test_cascade_split()
    test_airfoil_wake_grid()
    print("ALL PASSED")
