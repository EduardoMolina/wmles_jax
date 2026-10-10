"""Consistency tests.  Run:  python -m tests.test_multiblock   (from the project root)
   (set WMLES_BACKEND=numpy to force NumPy)"""
import csv
import tempfile
import numpy as np
from wmles import *
from wmles.backend import DEVICE, HAS_JAX, device_info, xp
from wmles.io_wall_loads import WallLoadCSVWriter

L = (1.0, 0.6, 0.5)
gas = Gas()


def nodes_from_index(ix, iy, iz, n, skew=0.04):
    """Skewed curvilinear (but periodic-compatible) nodes for global node indices."""
    X, Y, Z = np.meshgrid(ix / n[0] * L[0], iy / n[1] * L[1], iz / n[2] * L[2], indexing="ij")
    x = X + skew * np.sin(2 * np.pi * Y / L[1]) * np.sin(2 * np.pi * Z / L[2])
    y = Y + skew * 0.5 * np.sin(2 * np.pi * X / L[0])
    z = Z + skew * 0.5 * np.sin(2 * np.pi * X / L[0]) * np.cos(2 * np.pi * Y / L[1])
    return np.stack([x, y, z])


def prim_fn(xc):
    x, y, z = xc
    k = 2 * np.pi
    rho = 1.2 * (1 + 0.05 * xp.sin(k * x / L[0]) * xp.cos(k * z / L[2]))
    u = 60 * xp.sin(k * y / L[1]) + 20 * xp.cos(k * z / L[2])
    v = 15 * xp.sin(k * x / L[0]) * xp.sin(k * z / L[2])
    w = 10 * xp.cos(k * x / L[0]) + 5 * xp.sin(k * y / L[1])
    p = 1e5 * (1 + 0.03 * xp.cos(k * x / L[0]) * xp.sin(k * y / L[1]))
    return xp.stack([rho, u, v, w, p])


def build(kind, n=(12, 10, 8), scheme=None, wm=False, periodic_y=False):
    """kind: 'single' | 'split' | 'rot_cyc' | 'flip'.
    x,z periodic; y = slip wall / wall-model (or periodic if periodic_y)."""
    scheme = scheme or KEP(4)
    ybc = (lambda: WallModel(EquilibriumWallModel(), T_wall=300.0)) if wm else SlipWall
    ix, iy, iz = np.arange(n[0] + 1), np.arange(n[1] + 1), np.arange(n[2] + 1)
    if kind == "single":
        A = Block("all", nodes_from_index(ix, iy, iz, n))
        blocks = [A]
        m = Mesh(blocks)
        m.connect(A, "imin", A, "imax", shift=(-L[0], 0, 0))
        zpair = {A: ("kmin", "kmax")}
        yfaces = {A: ("jmin", "jmax")}
    else:
        h = n[0] // 2
        nodesA = nodes_from_index(ix[:h + 1], iy, iz, n)
        nodesB = nodes_from_index(ix[h:], iy, iz, n)
        if kind == "rot_cyc":      # (i,j,k)->(j,k,i): cyclic permutation, right-handed
            nodesB = np.transpose(nodesB, (0, 2, 3, 1))
        elif kind == "flip":       # 180 deg rotation about k: i and j reversed
            nodesB = nodesB[:, ::-1, ::-1, :]
        A, B = Block("A", nodesA), Block("B", nodesB)
        blocks = [A, B]
        m = Mesh(blocks)
        # B's face names (x-min, x-max, z-min, z-max, y-min, y-max) depend on orientation
        fx = {"split": ("imin", "imax"), "flip": ("imax", "imin"), "rot_cyc": ("kmin", "kmax")}[kind]
        fz = {"split": ("kmin", "kmax"), "flip": ("kmin", "kmax"), "rot_cyc": ("jmin", "jmax")}[kind]
        fy = {"split": ("jmin", "jmax"), "flip": ("jmax", "jmin"), "rot_cyc": ("imin", "imax")}[kind]
        m.connect(A, "imax", B, fx[0])
        m.connect(B, fx[1], A, "imin", shift=(L[0], 0, 0))
        zpair = {A: ("kmin", "kmax"), B: fz}
        yfaces = {A: ("jmin", "jmax"), B: fy}
    for b in blocks:
        m.connect(b, zpair[b][0], b, zpair[b][1], shift=(0, 0, -L[2]))
        if periodic_y:
            m.connect(b, yfaces[b][0], b, yfaces[b][1], shift=(0, -L[1], 0))
        else:
            for f in yfaces[b]:
                b.set_bc(f, ybc())
    m.finalize()
    return m, Solver(m, gas, scheme, sgs="vreman", cfl=0.5)


def gather(solver, R):
    """interior rhs values keyed by rounded cell-centre coordinates"""
    out = {}
    for blk, r, g in zip(solver.blocks, R, solver.geoms):
        xc = np.asarray(blk.interior(g.xc)).reshape(3, -1).T
        rr = np.asarray(blk.interior(r)).reshape(5, -1).T
        for c, v in zip(xc, rr):
            out[tuple(np.round(c, 8))] = v
    return out


def compare(kind, scheme=None, wm=False):
    m0, s0 = build("single", scheme=scheme, wm=wm)
    m1, s1 = build(kind, scheme=scheme, wm=wm)
    R0 = gather(s0, s0.rhs(s0.init_state(prim_fn)))
    R1 = gather(s1, s1.rhs(s1.init_state(prim_fn)))
    assert R0.keys() == R1.keys(), (len(R0), len(R1))
    scale = np.max(np.abs(np.array(list(R0.values())))) 
    err = max(np.max(np.abs(R0[k] - R1[k])) for k in R0) / scale
    print(f"  {kind:8s} scheme={type(s0.scheme).__name__:10s} wm={wm!s:5s}  max rel diff vs single block = {err:.2e}")
    return err


def freestream():
    prim = lambda xc: xp.stack([1.2 + 0 * xc[0], 50 + 0 * xc[0], -20 + 0 * xc[0], 30 + 0 * xc[0], 1e5 + 0 * xc[0]])
    for sc in (KEP(2), KEP(4), MUSCLHLLC(), Hybrid()):
        m, s = build("single", scheme=sc, periodic_y=True)
        R = s.rhs(s.init_state(prim))
        r = max(float(np.abs(np.asarray(m.blocks[0].interior(R[0]))[i]).max()) for i in range(5))
        print(f"  freestream residual {type(sc).__name__:10s}: {r:.2e}  (round-off level; energy flux scale ~ 1e7)")
        assert r < 1e-6 * 1e5


def characteristic_farfield():
    m, s = build("single")
    blk, geom = m.blocks[0], m.geoms[0]
    ng, n = blk.ng, blk.n[1]
    prim_inf = np.array([1.2, 10.0, 0.0, 0.0, 1e5])
    bc = CharacteristicFarfield(prim_inf)

    def state(prim):
        field = xp.asarray(prim).reshape(5, 1, 1, 1) * xp.ones((1,) + blk.N)
        return s.gas.prim_to_cons(field)

    cases = [
        (state([1.0, 0.0, 500.0, 0.0, 8e4]), 0, prim_inf),
        (state([1.0, 0.0, 1000.0, 0.0, 8e4]), 1, [1.0, 0.0, 1000.0, 0.0, 8e4]),
        (state(prim_inf), 0, prim_inf),
        (state(prim_inf), 1, prim_inf),
    ]
    for q, side, expected in cases:
        filled = bc.fill(q, blk, geom, s.gas, 1, side)
        ghost_j = ng - 1 if side == 0 else ng + n
        ghost = filled[:, ng, ghost_j, ng]
        np.testing.assert_allclose(np.asarray(s.gas.cons_to_prim(ghost)[0]), expected, rtol=1e-12)


def wall_model_selfconsistency():
    wmod = EquilibriumWallModel()
    kap, B = wmod.kappa, wmod.B
    nu, ut, h = 1.5e-5, 2.0, 0.003
    yp = h * ut / nu
    # Spalding: y+ = u+ + e^{-kB}(e^{k u+} - 1 - k u+ - (k u+)^2/2 - (k u+)^3/6): invert with bisection
    lo, hi = 1e-6, 60.0
    f = lambda up: up + np.exp(-kap * B) * (np.exp(kap * up) - 1 - kap * up - (kap * up) ** 2 / 2 - (kap * up) ** 3 / 6) - yp
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(mid) < 0 else (lo, mid)
    U = ut * 0.5 * (lo + hi)
    rho = 1.2; mu = nu * rho
    got = wmod.utau(xp.asarray([h]), xp.asarray([U]), xp.asarray([rho]), xp.asarray([mu]))
    print(f"  Spalding inversion: u_tau true={ut:.6f}  recovered={float(got[0]):.6f}")
    assert abs(float(got[0]) - ut) < 1e-6


def selected_device_placement():
    mesh, solver = build("single", n=(4, 4, 4))
    q = solver.init_state(prim_fn)[0]
    if HAS_JAX:
        assert mesh.geoms[0].vol.device == DEVICE
        assert q.device == DEVICE
    print("selected device:", device_info())


def wall_load_integration():
    axes = [np.linspace(0.0, 1.0, 5)] * 3
    nodes = np.stack(np.meshgrid(*axes, indexing="ij"))
    block = Block("wall", nodes)
    mesh = Mesh([block])
    wall = WallModel(EquilibriumWallModel(), T_wall=300.0)
    for face in ("imin", "imax", "jmin", "jmax", "kmin", "kmax"):
        block.set_bc(face, SlipWall())
    block.set_bc("jmin", wall)
    block.set_bc("jmax", wall)
    mesh.finalize()
    solver = Solver(mesh, gas, KEP(2), sgs=None)

    def uniform(xc):
        z = 0.0 * xc[0]
        return xp.stack([1.2 + z, 30.0 + z, z, z, 1e5 + z])

    Q = solver.init_state(uniform)
    center = (0.5, 0.5, 0.5)
    loads = solver.wall_loads(Q, center)
    stats = solver.wall_stats(Q)
    assert set(loads) == {("wall", "jmin"), ("wall", "jmax")}
    for name, normal_y in (("jmin", -1.0), ("jmax", 1.0)):
        force, moment = loads[("wall", name)]
        np.testing.assert_allclose(force[0], np.sum(stats[("wall", name)][0]) / 16.0)
        np.testing.assert_allclose(force[1], normal_y * 1e5)
        np.testing.assert_allclose(force[2], 0.0, atol=1e-10)
        np.testing.assert_allclose(moment[:2], 0.0, atol=1e-8)
        np.testing.assert_allclose(moment[2], -normal_y * 0.5 * force[0])
    np.testing.assert_allclose(sum(v[1] for v in loads.values()), 0.0, atol=1e-8)

    with tempfile.TemporaryDirectory() as tmp:
        path = f"{tmp}/wall_loads.csv"
        csv_writer = WallLoadCSVWriter(path, center)
        csv_writer.write(7, 0.25, loads)
        with open(path, newline="") as stream:
            rows = list(csv.reader(stream))
        assert len(rows) == 4  # header, two wall faces, and total
        assert rows[-1][0:4] == ["7", "0.25", "TOTAL", "total"]
        resumed_writer = WallLoadCSVWriter(path, center, append=True)
        resumed_writer.write(8, 0.5, loads)
        with open(path, newline="") as stream:
            rows = list(csv.reader(stream))
        assert len(rows) == 7
        assert rows[-1][0:4] == ["8", "0.5", "TOTAL", "total"]


if __name__ == "__main__":
    print("backend:", "jax" if HAS_JAX else "numpy")
    selected_device_placement()
    print("characteristic far-field:"); characteristic_farfield()
    print("wall model:"); wall_model_selfconsistency()
    print("wall load integration and CSV output:"); wall_load_integration()
    print("free-stream preservation on a skewed grid:"); freestream()
    print("multiblock vs single block RHS:")
    for kind in ("split", "flip", "rot_cyc"):
        e = compare(kind); assert e < 1e-9, e
    e = compare("split", scheme=Hybrid()); assert e < 1e-9
    e = compare("split", wm=True); assert e < 1e-9
    print("ALL PASSED")
