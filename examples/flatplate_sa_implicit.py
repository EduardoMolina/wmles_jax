"""Prototype steady implicit SA run on the TMR 2DZP flat plate.

    python -m examples.flatplate_sa_implicit --steps 20 --out out/flatplate_sa_implicit

The current matrix-free Newton-GMRES implementation has no preconditioner yet; start
with a conservative CFL and ramp gradually. Pseudo-time dt is recomputed from the
current state's explicit stability limit at each step.
"""
import argparse
import os

import numpy as np

from wmles.backend import asarray
from wmles.implicit import ImplicitIntegrator
from wmles.rans_case import build_rans_case, initial_state
from wmles.schemes import MUSCLHLLC

GRID = "tests/data/2DZP/flatplate_clust2_3levelsdown_69x49"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", default=GRID)
    parser.add_argument("--mach", type=float, default=0.2)
    parser.add_argument("--reynolds", type=float, default=5.0e6)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--cfl", type=float, default=0.1, help="initial pseudo-time CFL")
    parser.add_argument("--cfl-growth", type=float, default=1.25)
    parser.add_argument("--max-cfl", type=float, default=10.0)
    parser.add_argument("--newton-rtol", type=float, default=1.0e-6)
    parser.add_argument("--gmres-maxiter", type=int, default=30)
    parser.add_argument("--profile", choices=("flatplate", "none"), default="flatplate")
    parser.add_argument("--load", default=None, help="restart from a state.npz")
    parser.add_argument("--out", default="out/flatplate_sa_implicit")
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
        print(f"{step:5d} CFL={row[1]:.3e} dt={row[2]:.3e} rhs_l2={row[3]:.3e} "
              f"Newton={integrator.last_newton_iterations} "
              f"GMRES={integrator.last_gmres_iterations}", flush=True)
        np.savez(os.path.join(args.out, "state.npz"),
                 **{f"q{i}": np.asarray(q) for i, q in enumerate(state)})

    Q, _ = integrator.run_steady_cfl(
        Q, args.steps, args.cfl, growth=args.cfl_growth,
        max_cfl=args.max_cfl, callback=report)
    np.savez(os.path.join(args.out, "state.npz"),
             **{f"q{i}": np.asarray(q) for i, q in enumerate(Q)})


if __name__ == "__main__":
    main()
