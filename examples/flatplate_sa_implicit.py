"""Prototype steady implicit SA run on the TMR 2DZP flat plate.

    python -m examples.flatplate_sa_implicit --steps 20 --out out/flatplate_sa_implicit

The current matrix-free Newton-GMRES implementation has no preconditioner yet. By default
each cell uses its own pseudo-time step (local time stepping, steady state only) and the CFL
follows an SER ramp; use --global-dt for one step set by the smallest cell.
"""
import argparse
import os

import numpy as np

from wmles.backend import asarray
from wmles.implicit import ImplicitIntegrator
from wmles.rans_case import build_rans_case, initial_state
from wmles.schemes import MUSCLHLLC
from wmles import rans_post as rp

GRID = "tests/data/2DZP/flatplate_clust2_3levelsdown_69x49"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", default=GRID)
    parser.add_argument("--mach", type=float, default=0.2)
    parser.add_argument("--reynolds", type=float, default=5.0e6)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfl", type=float, default=1.0, help="initial pseudo-time CFL")
    parser.add_argument("--cfl-growth", type=float, default=1.5, help="max CFL growth per step")
    parser.add_argument("--max-cfl", type=float, default=1.0e3)
    parser.add_argument("--ramp", choices=("ser", "geometric"), default="ser")
    parser.add_argument("--ser-exponent", type=float, default=1.0)
    parser.add_argument("--local-dt", action=argparse.BooleanOptionalAction, default=True,
                        help="per-cell pseudo-time step (default) or one global step")
    parser.add_argument("--global-dt", dest="local_dt", action="store_false",
                        help="one global pseudo-time step from the smallest cell (same as --no-local-dt)")
    parser.add_argument("--newton-rtol", type=float, default=1.0e-6)
    parser.add_argument("--gmres-maxiter", type=int, default=30)
    parser.add_argument("--profile", choices=("flatplate", "none"), default="flatplate")
    parser.add_argument("--load", default=None, help="restart from a state.npz")
    parser.add_argument("--out", default="out/flatplate_sa_implicit")
    parser.add_argument("--x-max", type=float, default=1.5,
                        help="evaluate the C_f error only for x <= x_max: the last ~25%% of the plate is "
                             "contaminated by the characteristic outflow boundary (reported separately)")
    args = parser.parse_args(argv)

    _, solver, props = build_rans_case(
        args.grid, mach=args.mach, reynolds=args.reynolds, scheme=MUSCLHLLC())
    Q = initial_state(solver, props, profile=None if args.profile == "none" else args.profile)
    if args.load:
        with np.load(args.load) as data:
            Q = tuple(asarray(data[f"q{i}"]) for i in range(len(Q)))
    os.makedirs(args.out, exist_ok=True)

    integrator = ImplicitIntegrator(
        solver, newton_rtol=args.newton_rtol, gmres_maxiter=args.gmres_maxiter)

    def report(step, state, row):
        print(f"{step:5d} CFL={row[1]:.3e} dt_min={row[2]:.3e} rhs_l2={row[3]:.3e} "
              f"Newton={integrator.last_newton_iterations} "
              f"GMRES={integrator.last_gmres_iterations}", flush=True)
#        np.savez(os.path.join(args.out, "state.npz"),
#                 **{f"q{i}": np.asarray(q) for i, q in enumerate(state)})

    Q, _ = integrator.run_steady_cfl(
        Q, args.steps, args.cfl, growth=args.cfl_growth, max_cfl=args.max_cfl,
        local=args.local_dt, ramp=args.ramp, ser_exponent=args.ser_exponent, callback=report)
    np.savez(os.path.join(args.out, "state.npz"),
             **{f"q{i}": np.asarray(q) for i, q in enumerate(Q)})

    (ib, blk, axis, side) = next(rp.wall_faces(solver))
    sf = rp.skin_friction(solver, Q, props)[(blk.name, "kmin")]
    cols = rp.wall_columns(solver, Q, props, ib, axis, side)
    ret = rp.re_theta(cols, props)
    cf_ks = rp.ks_cf(np.maximum(ret, 1.0))
    with open(os.path.join(args.out, "cf_x.csv"), "w") as f:
        f.write("x,re_theta,cf,cf_ks,yplus1\n")
        for row in zip(sf["x"], ret, sf["cf"], cf_ks, sf["yplus1"]):
            f.write(",".join(f"{v:.8e}" for v in row) + "\n")

    sel = (ret > 4000.0) & (ret < 13000.0) & (sf["x"] > 0.0)
    for label, mask in ((f"x <= {args.x_max}", sel & (sf["x"] <= args.x_max)), ("all stations", sel)):
        if mask.any():
            err = (sf["cf"][mask] - cf_ks[mask]) / cf_ks[mask]
            print(f"Re_theta in [4000, 13000], {label}: cf/cf_KS - 1 = {100 * err.min():+.2f}% .. "
                  f"{100 * err.max():+.2f}% (mean {100 * err.mean():+.2f}%, {mask.sum()} stations)")
        else:
            print(f"{label}: no station with Re_theta in [4000, 13000]")
    near = np.where((sf["x"] > 0.0) & (sf["x"] <= args.x_max), np.abs(ret - 1.0e4), np.inf)
    i = int(np.argmin(near))
    u_tau, nu_w = sf["u_tau"][i], props["nu_inf"]
    y, u = cols["y"][i], cols["ut"][i]
    with open(os.path.join(args.out, "profile_Re10000.csv"), "w") as f:
        f.write("yplus,uplus\n")
        for yp, up in zip(y * u_tau / nu_w, u / u_tau):
            f.write(f"{yp:.8e},{up:.8e}\n")
    print(f"profile station x={sf['x'][i]:.3f}  Re_theta={ret[i]:.0f}  cf={sf['cf'][i]:.5f}  "
          f"y+(first cell)={sf['yplus1'][i]:.2f}")


if __name__ == "__main__":
    main()
