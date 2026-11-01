"""TMR 2DANW (Nakayama A-airfoil near wake) with Spalart-Allmaras.

    python -m examples.airfoil_wake_sa --iterations 30000 --out out/airfoil_wake_sa

Writes forces.txt (cl, cd, split into pressure/friction), cf_x.csv (airfoil skin friction) and
wake_x*.csv (z/c, u/U_inf profiles at x/c = 1.01, 1.05, 1.20, 1.40, 1.80, 2.19, interpolated from the
cell-centre mid-span plane) and state.npz (restart).
"""
import argparse
import os
import time

import numpy as np
from scipy.interpolate import LinearNDInterpolator

from wmles import rans_post as rp
from wmles.backend import asarray
from wmles.rans_case import build_rans_case, initial_state, run_steady
from wmles.schemes import Hybrid, MUSCLHLLC

GRID = "tests/data/2DANW/nak_a_ff_500c_2zn_L5"
STATIONS = (1.01, 1.05, 1.20, 1.40, 1.80, 2.19)


def save_state(path, Q):
    np.savez(path, **{f"q{i}": np.asarray(q) for i, q in enumerate(Q)})


def load_state(path, Q):
    data = np.load(path)
    return tuple(asarray(data[f"q{i}"]) for i in range(len(Q)))


def midplane_points(solver, Q):
    """Scattered (x, z) cell centres, u and nu_tilde of the mid-span plane of all blocks
    (the grids extrude along y, axis 0 of every block)."""
    Qb = solver.apply_bcs(Q, solver.geoms)
    pts, u, nut = [], [], []
    for blk, q, g in zip(solver.blocks, Qb, solver.geoms):
        w, _ = solver.gas.cons_to_prim(q)
        m = blk.n[0] // 2
        xc = np.asarray(blk.interior(g.xc))[:, m]
        pts.append(xc[[0, 2]].reshape(2, -1).T)
        u.append(np.asarray(blk.interior(w[1]))[m].reshape(-1))
        nut.append(np.asarray(blk.interior(q[5] / q[0]))[m].reshape(-1))
    return np.concatenate(pts), np.concatenate(u), np.concatenate(nut)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=GRID)
    ap.add_argument("--mach", type=float, default=0.088)
    ap.add_argument("--reynolds", type=float, default=1.2e6, help="per chord (grid unit)")
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--cfl", type=float, default=0.8)
    ap.add_argument("--scheme", choices=("muscl", "muscl-lm", "hybrid"), default="muscl")
    ap.add_argument("--report", type=int, default=1000)
    ap.add_argument("--out", default="out/airfoil_wake_sa")
    ap.add_argument("--load", default=None, help="restart from a state.npz")
    args = ap.parse_args(argv)

    scheme = {"muscl": MUSCLHLLC, "muscl-lm": lambda: MUSCLHLLC(low_mach=True), "hybrid": Hybrid}[args.scheme]()
    mesh, solver, props = build_rans_case(args.grid, mach=args.mach, reynolds=args.reynolds,
                                          periodic_type="symmetry_y", scheme=scheme, cfl=args.cfl)
    os.makedirs(args.out, exist_ok=True)
    Q = initial_state(solver, props)
    if args.load:
        Q = load_state(args.load, Q)
    t0 = time.time()

    def log(it, Q, row):
        print(f"{it:7d} {time.time() - t0:7.0f}s  res(rho)={row[1]:.3e}  res(rho nu~)={row[2]:.3e}", flush=True)
        save_state(os.path.join(args.out, "state.npz"), Q)

    Q, history = run_steady(solver, Q, args.iterations, report_every=args.report, callback=log)
    save_state(os.path.join(args.out, "state.npz"), Q)

    forces = rp.wall_forces(solver, Q, props)
    with open(os.path.join(args.out, "forces.txt"), "w") as f:
        f.write("".join(f"{k} = {v}\n" for k, v in forces.items()))
    print("forces:", {k: (np.round(v, 6) if not np.isscalar(v) else round(v, 6)) for k, v in forces.items()})

    sf = rp.skin_friction(solver, Q, props)
    with open(os.path.join(args.out, "cf_x.csv"), "w") as f:
        f.write("x,cf,yplus1\n")
        for key, v in sf.items():
            for row in zip(v["x"], v["cf"], v["yplus1"]):
                f.write(",".join(f"{a:.8e}" for a in row) + "\n")
    print("max first-cell y+:", max(float(v["yplus1"].max()) for v in sf.values()))

    pts, u, _ = midplane_points(solver, Q)
    interp = LinearNDInterpolator(pts / props["length"], u / props["U_inf"])
    ys = np.linspace(-0.12, 0.12, 241)
    for x in STATIONS:
        prof = interp(np.full_like(ys, x), ys)
        np.savetxt(os.path.join(args.out, f"wake_x{x:.2f}.csv"), np.column_stack([ys, prof]),
                   header="z/c,u/U_inf", delimiter=",", comments="")
        print(f"x/c={x:4.2f}: min u/U_inf = {np.nanmin(prof):.4f}")


if __name__ == "__main__":
    main()
