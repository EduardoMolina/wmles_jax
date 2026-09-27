"""Wall-modelled LES of a compressible turbulent channel (Ma_b ~ 0.3), two blocks in x.

    python -m examples.channel_wmles [n_steps]

Domain  x in [0, 2 pi d] (2 blocks, periodic), y in [0, 2 d] (wall-modelled walls),
z in [0, pi d] (periodic).  Constant mass flux is enforced after every step.
"""
import sys
import numpy as np
from wmles import (Gas, Block, Mesh, Solver, KEP, Hybrid, WallModel,
                   EquilibriumWallModel, ConstantMassFlux)
from wmles.backend import xp, HAS_JAX

gas = Gas()
d = 0.01                                   # half height [m]
Lx, Ly, Lz = 2 * np.pi * d, 2 * d, np.pi * d
nx, ny, nz = 24, 24, 16                    # cells per block in x is nx//2
T0, rho0 = 300.0, 1.2
Ub = 0.3 * float(gas.sound_speed(T0))
p0 = rho0 * gas.R * T0


def box(x0, x1, nxc, s=1.6):
    x = np.linspace(x0, x1, nxc + 1)
    eta = np.linspace(0, 1, ny + 1)
    y = d * (1 + np.tanh(s * (2 * eta - 1)) / np.tanh(s))          # wall-clustered
    z = np.linspace(0, Lz, nz + 1)
    return np.stack(np.meshgrid(x, y, z, indexing="ij"))


def build():
    A = Block("A", box(0, Lx / 2, nx // 2))
    B = Block("B", box(Lx / 2, Lx, nx // 2))
    mesh = Mesh([A, B])
    mesh.connect(A, "imax", B, "imin")
    mesh.connect(B, "imax", A, "imin", shift=(Lx, 0, 0))         # periodic in x
    wm = EquilibriumWallModel()
    for b in (A, B):
        mesh.connect(b, "kmin", b, "kmax", shift=(0, 0, -Lz))    # periodic in z (self-connection)
        b.set_bc("jmin", WallModel(wm, T_wall=T0, n_match=1))
        b.set_bc("jmax", WallModel(wm, T_wall=T0, n_match=1))
    mesh.finalize()
    solver = Solver(mesh, gas, Hybrid(KEP(4)), sgs="vreman", cfl=0.7,
                    constraints=[ConstantMassFlux(0, rho0 * Ub)])
    return mesh, solver


def initial(xc):
    x, y, z = xc
    yn = xp.clip(xp.minimum(y, 2 * d - y) / d, 1e-3, 1.0)
    u = 1.15 * Ub * yn ** (1 / 7)
    k = 2 * np.pi
    u = u + 0.08 * Ub * xp.sin(k * z / Lz * 2) * xp.sin(k * x / Lx)
    v = 0.05 * Ub * xp.sin(k * x / Lx * 2) * xp.sin(k * z / Lz)
    w = 0.05 * Ub * xp.cos(k * x / Lx) * xp.sin(k * y / Ly)
    return xp.stack([rho0 + 0 * x, u, v, w, p0 + 0 * x])


if __name__ == "__main__":
    nsteps = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    out_every = int(sys.argv[2]) if len(sys.argv) > 2 else 0     # 0 = no VTU output
    mesh, solver = build()
    Q = solver.init_state(initial)
    print(f"backend={'jax' if HAS_JAX else 'numpy'}  Ub={Ub:.1f} m/s  blocks={[b.n for b in mesh.blocks]}")
    writer = None
    t = 0.0
    for it in range(1, nsteps + 1):
        dt = solver.max_dt(Q)
        Q = solver.advance(Q, dt)
        t += float(dt)
        if out_every and it % out_every == 0:
            from wmles.io_vtk import write_case
            writer = write_case(mesh, solver, Q, it, t, "out/channel", writer=writer)
        if it % max(1, nsteps // 10) == 0 or it == 1:
            st = solver.wall_stats(Q)
            tau = np.mean([np.mean(v[0]) for v in st.values()])
            qw = np.mean([np.mean(v[1]) for v in st.values()])
            rmin = min(float(xp.min(b.interior(q[0]))) for b, q in zip(mesh.blocks, Q))
            ke = sum(float(xp.sum(b.interior(0.5 * xp.sum(q[1:4] ** 2, axis=0) / q[0] * g.vol)))
                     for b, q, g in zip(mesh.blocks, Q, solver.geoms))
            print(f"it={it:5d} t={t:.3e} dt={float(dt):.2e} <tau_w>={tau:8.3f} Pa "
                  f"<q_w>={qw:9.1f} W/m2 rho_min={rmin:.4f} KE={ke:.4e}")