"""TMR 2DZP flat plate with Spalart-Allmaras: C_f vs Re_theta against Karman-Schoenherr.

    python -m examples.flatplate_sa --iterations 30000 --scheme muscl --out out/flatplate_sa
    python -m examples.flatplate_sa --iterations 30000 --load out/flatplate_sa/state.npz  # continue

Writes cf_x.csv (x, Re_theta, cf, cf_KS, y+ of the first cell), profile_Re10000.csv (u+ vs y+ at the
station where Re_theta is closest to 10000) and state.npz (restart).
"""
import argparse
import os
import time

import numpy as np

from wmles import rans_post as rp
from wmles.backend import asarray
from wmles.rans_case import build_rans_case, initial_state, run_steady
from wmles.schemes import Hybrid, MUSCLHLLC

GRID = "tests/data/2DZP/flatplate_clust2_3levelsdown_69x49"


def save_state(path, Q):
    np.savez(path, **{f"q{i}": np.asarray(q) for i, q in enumerate(Q)})


def load_state(path, Q):
    data = np.load(path)
    return tuple(asarray(data[f"q{i}"]) for i in range(len(Q)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default=GRID)
    ap.add_argument("--mach", type=float, default=0.2)
    ap.add_argument("--reynolds", type=float, default=5.0e6, help="per grid unit")
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--cfl", type=float, default=0.8)
    ap.add_argument("--scheme", choices=("muscl", "muscl-lm", "hybrid"), default="muscl")
    ap.add_argument("--report", type=int, default=1000)
    ap.add_argument("--out", default="out/flatplate_sa")
    ap.add_argument("--load", default=None, help="restart from a state.npz")
    ap.add_argument("--x-max", type=float, default=1.5,
                    help="evaluate the C_f error only for x <= x_max: the last ~25%% of the plate is "
                         "contaminated by the characteristic outflow boundary (reported separately)")
    ap.add_argument("--profile", choices=("none", "flatplate"), default="flatplate",
                    help="initial state: uniform free stream or an analytic turbulent boundary layer")
    args = ap.parse_args(argv)

    scheme = {"muscl": MUSCLHLLC, "muscl-lm": lambda: MUSCLHLLC(low_mach=True), "hybrid": Hybrid}[args.scheme]()
    mesh, solver, props = build_rans_case(args.grid, mach=args.mach, reynolds=args.reynolds,
                                          scheme=scheme, cfl=args.cfl)
    os.makedirs(args.out, exist_ok=True)
    Q = initial_state(solver, props, profile=None if args.profile == "none" else args.profile)
    if args.load:
        Q = load_state(args.load, Q)
    t0 = time.time()

    def log(it, Q, row):
        print(f"{it:7d} {time.time() - t0:7.0f}s  res(rho)={row[1]:.3e}  res(rho nu~)={row[2]:.3e}", flush=True)
        save_state(os.path.join(args.out, "state.npz"), Q)

    Q, history = run_steady(solver, Q, args.iterations, report_every=args.report, callback=log)
    save_state(os.path.join(args.out, "state.npz"), Q)

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
