"""Run a short inviscid ONERA M6 simulation on the bundled Plot3D/NMF mesh.

    WMLES_DEVICE=cpu conda run -n pyvista python -m examples.oneram6_euler --steps 100

This is an integration example, not a converged aerodynamic validation case.
"""
import argparse
import os

import numpy as np

from wmles import (CharacteristicFarfield, Extrapolate, Gas, MUSCLHLLC, Solver, SlipWall,
                   build_mesh, read_nmf, read_plot3d)
from wmles.backend import HAS_JAX, xp


HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_BASE = os.path.join(HERE, "..", "tests", "data", "oneram6")


def _check_state(Q, mesh, gas):
    q = np.asarray(mesh.blocks[0].interior(Q[0]))
    pressure = (gas.gamma - 1.0) * (q[4] - 0.5 * np.sum(q[1:4] ** 2, axis=0) / q[0])
    if not np.isfinite(q).all() or not np.isfinite(pressure).all():
        raise FloatingPointError("ONERA M6 Euler state became non-finite")
    if np.any(q[0] <= 0.0) or np.any(pressure <= 0.0):
        raise FloatingPointError("ONERA M6 Euler state lost positive density or pressure")
    return float(q[0].min()), float(pressure.min())


def run(steps=100, report_every=10, write_vtk=0, vtk_dir="out/oneram6"):
    if steps < 0:
        raise ValueError("steps must be non-negative")
    if write_vtk < 0:
        raise ValueError("write_vtk must be non-negative")

    gas = Gas(mu_ref=0.0)
    temperature, pressure = 288.15, 101_325.0
    mach, alpha = 0.84, np.deg2rad(3.06)
    rho = pressure / (gas.R * temperature)
    speed = mach * np.sqrt(gas.gamma * gas.R * temperature)
    prim = np.array([rho, speed * np.cos(alpha), 0.0, speed * np.sin(alpha), pressure])

    grid = read_plot3d(DEFAULT_BASE + ".p3d")
    mesh = build_mesh(
        grid.nodes,
        read_nmf(DEFAULT_BASE + ".nmf"),
        bc_map={
            "pole": Extrapolate(),
            "farfield": CharacteristicFarfield(prim),
            "viscous_solid": SlipWall(),
        },
    )
    #solver = Solver(mesh, gas, KEP(2), sgs=None, cfl=0.05)
    solver = Solver(mesh, gas, MUSCLHLLC(), sgs=None, cfl=0.75)
    Q = solver.init_state(lambda xc: xp.stack([prim[i] + 0.0 * xc[0] for i in range(5)]))
    report_every = max(1, report_every)
    time = 0.0
    writer = None

    print(f"backend={'jax' if HAS_JAX else 'numpy'} mesh=ONERA-M6 cells={mesh.blocks[0].n} "
          f"Mach={mach:.2f} alpha={np.rad2deg(alpha):.2f} deg steps={steps}")
    if write_vtk:
        print(f"VTK output: every {write_vtk} steps in {vtk_dir}")
    for step in range(1, steps + 1):
        dt = solver.max_dt(Q)
        dt_value = float(dt)
        if not np.isfinite(dt_value) or dt_value <= 0.0:
            raise FloatingPointError(f"invalid time step at iteration {step}: {dt_value}")
        Q = solver.advance(Q, dt)
        time += dt_value
        if write_vtk and step % write_vtk == 0:
            from wmles.io_vtk import write_case
            writer = write_case(mesh, solver, Q, step, time, vtk_dir,
                                name="oneram6", writer=writer, wall=False)
        if step % report_every == 0 or step == 1 or step == steps:
            rho_min, p_min = _check_state(Q, mesh, gas)
            print(f"step={step:4d} time={time:.6e} dt={dt_value:.3e} "
                  f"rho_min={rho_min:.6e} p_min={p_min:.6e}")
    return Q


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=100, help="number of explicit Euler time steps")
    parser.add_argument("--report-every", type=int, default=10, help="state report interval")
    parser.add_argument("--writeVTK", type=int, default=0, metavar="NSTEPS",
                        help="write VTK output every NSTEPS steps (0 disables output)")
    args = parser.parse_args()
    run(args.steps, args.report_every, args.writeVTK)


if __name__ == "__main__":
    main()