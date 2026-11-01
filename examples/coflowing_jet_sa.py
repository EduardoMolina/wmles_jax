"""TMR 2DCJ coflowing jet (verification) with Spalart-Allmaras: centreline u/a_inf vs CFL3D and FUN3D.

    python -m examples.coflowing_jet_sa --iterations 20000 --out out/coflowing_jet_sa

Writes centerline.csv (x, u/a_inf of the symmetry line y=0, linearly extrapolated from the first two
cell layers; CFL3D and FUN3D references interpolated to the same x), drag.txt (skin-friction drag
coefficient of the plate, both sides, reference area 10; compare with convergence.dat) and state.npz (restart).
"""
import argparse
import os
import time

import numpy as np

from wmles import rans_post as rp
from wmles.backend import asarray
from wmles.rans_case import build_jet_case, jet_initial_state, run_steady
from wmles.schemes import Hybrid, MUSCLHLLC

GRID = "tests/data/2DCJ/shear_3levelsdown"
REFERENCE = "tests/data/2DCJ/uvel_y_0.dat"


def read_reference(path=REFERENCE):
    """{'CFL3D': (x, u), 'FUN3D': (x, u)} from the two-zone Tecplot file."""
    out, cur = {}, None
    for line in open(path):
        if line.lower().startswith("zone"):
            cur = line.split('"')[1]
            out[cur] = []
        elif cur and line.strip() and line[0] in " -0123456789":
            v = line.split()
            out[cur].append((float(v[0]), float(v[2])))
    return {k: tuple(np.array(v).T) for k, v in out.items()}


def centerline(solver, Q):
    """(x, u) on the symmetry line z = 0, linearly extrapolated from the first two cell layers of the
    blocks adjacent to it (mid-span plane)."""
    xs, us = [], []
    for blk, q, g in zip(solver.blocks, Q, solver.geoms):
        if np.asarray(blk.x_int)[2].min() > 1e-9:
            continue
        w, _ = solver.gas.cons_to_prim(q)
        xc, u = np.asarray(blk.interior(g.xc)), np.asarray(blk.interior(w[1]))
        m = u.shape[0] // 2
        z0, z1 = xc[2][m, :, 0], xc[2][m, :, 1]
        xs.append(xc[0][m, :, 0])
        us.append(u[m, :, 0] - z0 * (u[m, :, 1] - u[m, :, 0]) / (z1 - z0))
    x, u = np.concatenate(xs), np.concatenate(us)
    i = np.argsort(x)
    return x[i], u[i]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=GRID)
    ap.add_argument("--reynolds", type=float, default=5.0e4, help="per grid unit, based on the inner-stream (reference) velocity")
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--cfl", type=float, default=0.8)
    ap.add_argument("--scheme", choices=("muscl", "muscl-lm", "hybrid"), default="muscl")
    ap.add_argument("--report", type=int, default=1000)
    ap.add_argument("--out", default="out/coflowing_jet_sa")
    ap.add_argument("--load", default=None, help="restart from a state.npz")
    args = ap.parse_args(argv)

    scheme = {"muscl": MUSCLHLLC, "muscl-lm": lambda: MUSCLHLLC(low_mach=True), "hybrid": Hybrid}[args.scheme]()
    mesh, solver, props = build_jet_case(args.grid, reynolds=args.reynolds, scheme=scheme, cfl=args.cfl)
    os.makedirs(args.out, exist_ok=True)
    Q = jet_initial_state(solver, props)
    if args.load:
        data = np.load(args.load)
        Q = tuple(asarray(data[f"q{i}"]) for i in range(len(Q)))
    t0 = time.time()

    def save(Q):
        np.savez(os.path.join(args.out, "state.npz"), **{f"q{i}": np.asarray(q) for i, q in enumerate(Q)})

    def log(it, Q, row):
        print(f"{it:7d} {time.time() - t0:7.0f}s  res(rho)={row[1]:.3e}  res(rho nu~)={row[2]:.3e}", flush=True)
        save(Q)

    Q, _ = run_steady(solver, Q, args.iterations, report_every=args.report, callback=log)
    save(Q)

    a = props["a_inf"]
    x, u = centerline(solver, Q)
    u = u / a
    ref = read_reference()
    cols = {k: np.interp(x / props["length"], *v) for k, v in ref.items()}
    with open(os.path.join(args.out, "centerline.csv"), "w") as f:
        f.write("x,u_a," + ",".join(f"u_a_{k}" for k in cols) + "\n")
        for i in range(len(x)):
            f.write(",".join(f"{v:.8e}" for v in (x[i] / props["length"], u[i], *(c[i] for c in cols.values()))) + "\n")
    xn = x / props["length"]
    sel = xn >= 0.0
    for k, c in cols.items():
        d = u[sel] - c[sel]
        print(f"centerline u/a_inf vs {k}: max|diff| = {np.abs(d).max():.4e}  rms = {np.sqrt(np.mean(d * d)):.4e}")

    forces = rp.wall_forces(solver, Q, props)
    cd = forces["cd_friction"] / 10.0                    # reference area 10 (plate length)
    with open(os.path.join(args.out, "drag.txt"), "w") as f:
        f.write(f"cd_friction_refarea10 = {cd}\n")
    print(f"plate skin-friction drag (ref. area 10, q = rho U_ref^2 / 2, M_ref = 0.5): {cd:.6e}")


if __name__ == "__main__":
    main()
