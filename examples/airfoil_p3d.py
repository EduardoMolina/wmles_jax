"""Load a Plot3D + NMF multiblock airfoil grid (e.g. the NASA TMBWG 'Airfoil Near-Wake' grids,
https://tmbwg.github.io/turbmodels/airfoilwake500c_grids.html), build the solver mesh and, optionally,
advance a few steps.

    python -m examples.airfoil_p3d [base] [--ncells 8] [--span 0.1] [--mach 0.2] [--alpha 0]
                                   [--re 1.2e6] [--steps 0] [--vtu 0]

`base` is the path without extension (<base>.p3dfmt + <base>.nmf); default: tests/data L5 grid.
Coordinates are in chords; --span (chords) and --ncells re-discretize the single spanwise cell of the
TMBWG grids (span = index i), periodic in the span.  --re scales the chord so that Re_c = U c / nu.

CAVEATS (this is the mesh-reader milestone, not a validated airfoil case):
* far-field ('farfield_riem') is a fixed free-stream Dirichlet state -> reflects acoustics; a
  characteristic far-field BC is the next step.
* the TMBWG grids are RANS grids (wall spacing y+ ~ 1 at the finest level; tiny explicit dt). For WMLES
  use a grid whose first cell sits in the log layer; the airfoil surface here gets the wall model
  (adiabatic, n_match=1) only to show the wiring.
"""
import argparse
import os
import numpy as np

from wmles import (Gas, Dirichlet, WallModel, EquilibriumWallModel, Solver, Hybrid,
                   read_plot3d, read_nmf, respan, build_mesh, describe)
from wmles.backend import xp, HAS_JAX

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "..", "tests", "data", "nak_a_ff_500c_2zn_L5")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base", nargs="?", default=DEFAULT)
    ap.add_argument("--ncells", type=int, default=8, help="spanwise cells")
    ap.add_argument("--span", type=float, default=0.1, help="span in chords")
    ap.add_argument("--mach", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=0.0, help="angle of attack [deg] (x-z plane)")
    ap.add_argument("--re", type=float, default=1.2e6, help="chord Reynolds number")
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--vtu", type=int, default=0, help="write VTU every N steps (needs pyvista)")
    a = ap.parse_args()

    gas = Gas()
    T0, p0 = 300.0, 101_325.0
    rho0 = p0 / (gas.R * T0)
    U = a.mach * float(gas.sound_speed(xp.asarray(T0)))
    nu0 = float(gas.mu(xp.asarray(T0))) / rho0
    chord = a.re * nu0 / U                                       # physical chord [m]
    al = np.radians(a.alpha)
    prim = np.array([rho0, U * np.cos(al), 0.0, U * np.sin(al), p0])   # airfoil in the x-z plane

    grid = read_plot3d(a.base + ".p3dfmt")
    nodes, nmf = respan(grid.nodes, read_nmf(a.base + ".nmf"), ncells=a.ncells, length=a.span)
    nodes = [X * chord for X in nodes]
    mesh = build_mesh(nodes, nmf,
                      bc_map={"farfield_riem": Dirichlet(prim),
                              "viscous_solid": lambda: WallModel(EquilibriumWallModel(), T_wall=None, n_match=1)},
                      periodic=(("symmetry_y", 0),))
    print(f"backend={'jax' if HAS_JAX else 'numpy'}  chord={chord:.4f} m  U={U:.1f} m/s  Re_c={a.re:.3g}")
    print(describe(mesh))
    wall = next(b for b in mesh.blocks if any(isinstance(f, WallModel) for f in b.faces.values()))
    ax, side = next(k for k, f in wall.faces.items() if isinstance(f, WallModel))
    h1 = np.linalg.norm(np.asarray(wall.take(wall.geom.xc, ax, wall.ng if side == 0 else wall.ng + wall.n[ax] - 1))
                        - np.asarray(wall.take(wall.geom.xf[ax], ax, wall.ng if side == 0 else wall.ng + wall.n[ax])), axis=0)
    print(f"matching-point (first cell centre) wall distance: min {h1.min() / chord:.3e} c, max {h1.max() / chord:.3e} c")

    solver = Solver(mesh, gas, Hybrid(), sgs="vreman", cfl=0.5)
    Q = solver.init_state(lambda xc: xp.stack([prim[i] + 0 * xc[0] for i in range(5)]))
    print(f"dt_max = {float(solver.max_dt(Q)):.3e} s   (chord flow-through = {chord / U:.3e} s)")
    writer, t = None, 0.0
    for it in range(1, a.steps + 1):
        dt = solver.max_dt(Q)
        Q = solver.advance(Q, dt)
        t += float(dt)
        if a.vtu and it % a.vtu == 0:
            from wmles.io_vtk import write_case
            writer = write_case(mesh, solver, Q, it, t, "out/airfoil", writer=writer)
        if it % max(1, a.steps // 10) == 0 or it == 1:
            tau = np.mean([v[0].mean() for v in solver.wall_stats(Q).values()])
            print(f"it={it:5d} t={t:.3e} dt={float(dt):.2e} <tau_w>={tau:.3f} Pa")


if __name__ == "__main__":
    main()
