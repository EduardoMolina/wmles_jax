"""Wall-modelled LES of a compressible turbulent channel (Ma_b ~ 0.3), two blocks in x.

    python -m examples.channel_wmles [n_steps] [vtu_every]

Domain  x in [0, 2 pi d] (2 blocks, periodic), y in [0, 2 d] (wall-modelled walls),
z in [0, pi d] (periodic).  Constant mass flux is enforced after every step.
"""
import argparse
from pathlib import Path
import re
import time
import numpy as np
from wmles import (Gas, Block, Mesh, Solver, KEP, Hybrid, WallModel,
                   EquilibriumWallModel, ConstantMassFlux)
from wmles.backend import xp, HAS_JAX, device_info

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


def _parse_steps(value):
    try:
        steps = tuple(int(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("steps must be comma-separated integers") from exc
    if not steps or any(step <= 0 for step in steps):
        raise argparse.ArgumentTypeError("steps must be positive integers")
    if len(set(steps)) != len(steps):
        raise argparse.ArgumentTypeError("steps must not contain duplicates")
    return tuple(sorted(steps))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("n_steps", nargs="?", type=int, default=200)
    parser.add_argument("vtu_steps", nargs="?", type=int, default=0,
                        help="VTU output stride; 0 disables VTU output")
    parser.add_argument("--force_steps", type=int, default=100,
                        help="Step frequency to write wall loads")
    parser.add_argument("--moment_center", type=float, nargs=3, default=(0.0, 0.0, 0.0),
                        metavar=("X", "Y", "Z"), help="moment reference point in metres")
    parser.add_argument("--force-file", default="out/channel_wall_loads.csv",
                        help="CSV output path for wall loads")
    parser.add_argument("--restart", help="restart from a saved .pvd collection or block .vtu")
    args = parser.parse_args()
    nsteps, vtusteps, forcesteps = args.n_steps, args.vtu_steps, args.force_steps
    if nsteps < 1:
        parser.error("n_steps must be positive")

    mesh, solver = build()
    if args.restart:
        from wmles.io_vtk import read_restart
        Q, restart_step, t = read_restart(args.restart, solver)
    else:
        Q, restart_step, t = solver.init_state(initial), 0, 0.0
    print(f"backend={'jax' if HAS_JAX else 'numpy'} device={device_info()}  Ub={Ub:.1f} m/s  "
          f"blocks={[b.n for b in mesh.blocks]}")
    wall_writer = None
    if args.force_steps:
        from wmles.io_wall_loads import WallLoadCSVWriter
        wall_writer = WallLoadCSVWriter(
            args.force_file, args.moment_center, append=bool(args.restart)
        )

    output_dir, output_name, resume_output = "out/channel", "case", False
    if args.restart:
        restart_path = Path(args.restart)
        match = re.fullmatch(r"(.+)_block\d+_\d+\.vtu", restart_path.name)
        if restart_path.suffix.lower() == ".pvd":
            output_dir, output_name, resume_output = str(restart_path.parent), restart_path.stem, True
        elif match:
            output_dir, output_name = str(restart_path.parent), match.group(1)
            resume_output = (restart_path.parent / f"{output_name}.pvd").is_file()

    writer = None
    if vtusteps and resume_output:
        from wmles.io_vtk import VTKWriter
        writer = VTKWriter(output_dir, output_name, resume=True)
    step_times = []
    for local_step in range(1, nsteps + 1):
        it = restart_step + local_step
        step_start = time.perf_counter()
        dt = solver.max_dt(Q)
        Q = solver.advance(Q, dt)
        t += float(dt)
        if HAS_JAX:
            for q in Q:
                q.block_until_ready()
        step_times.append(time.perf_counter() - step_start)
        if vtusteps and it % vtusteps == 0:
            from wmles.io_vtk import write_case
            writer = write_case(mesh, solver, Q, it, t, output_dir,
                                name=output_name, writer=writer)
        if forcesteps and it % forcesteps == 0:
            wall_writer.write(it, t, solver.wall_loads(Q, args.moment_center))
        if local_step % max(1, nsteps // 10) == 0 or local_step == 1:
            st = solver.wall_stats(Q)
            tau = np.mean([np.mean(v[0]) for v in st.values()])
            qw = np.mean([np.mean(v[1]) for v in st.values()])
            rmin = min(float(xp.min(b.interior(q[0]))) for b, q in zip(mesh.blocks, Q))
            ke = sum(float(xp.sum(b.interior(0.5 * xp.sum(q[1:4] ** 2, axis=0) / q[0] * g.vol)))
                     for b, q, g in zip(mesh.blocks, Q, solver.geoms))
            print(f"it={it:5d} t={t:.3e} dt={float(dt):.2e} <tau_w>={tau:8.3f} Pa "
                  f"<q_w>={qw:9.1f} W/m2 rho_min={rmin:.4f} KE={ke:.4e}")
    warm = step_times[1:]
    cells = sum(np.prod(b.n) for b in mesh.blocks)
    first_label = "compile+run" if HAS_JAX else "run"
    if warm:
        warm_mean = np.mean(warm)
        print(f"step timing: first_{first_label}={step_times[0]:.4f}s  "
              f"warm_mean={warm_mean:.4f}s/step  cells_per_second={cells / warm_mean:.3g}")
    elif step_times:
        print(f"step timing: first_{first_label}={step_times[0]:.4f}s  warm_mean=n/a")