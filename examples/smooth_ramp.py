"""Run the UMD smooth-ramp WMLES setup using Baurle's grid-6 distributions.

    python -m examples.smooth_ramp --steps 10
    python -m examples.smooth_ramp --steps 10 --nz 8 --vtu 5

The default mesh spans x/L=-1..4, uses the provided 120-cell wall-normal
distributions and a uniform 16-cell periodic span. A run with zero steps builds
and reports the setup without advancing the solver.
"""
import argparse
import time

from wmles.backend import HAS_JAX, device_info, xp
from wmles.smooth_ramp import build_case, initial


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--nz", type=int, default=16, help="uniform spanwise cell count")
    parser.add_argument("--x-min", type=float, default=-1.0, help="upstream extent in x/L")
    parser.add_argument("--x-max", type=float, default=4.0, help="downstream extent in x/L")
    parser.add_argument("--vtu", type=int, default=0, help="write VTU every N steps (needs pyvista)")
    args = parser.parse_args()
    if args.steps < 0 or args.vtu < 0:
        parser.error("--steps and --vtu must be non-negative")

    mesh, solver, props = build_case(x_min=args.x_min, x_max=args.x_max, nz=args.nz)
    blk = mesh.blocks[0]
    print(f"backend={'jax' if HAS_JAX else 'numpy'} device={device_info()}  "
          f"cells={blk.n}  L={props['L']:.5g} m  U_inf={props['U_inf']:.3f} m/s  "
          f"Re_L={props['Re_L']:.0f}  Ma_inf={props['Ma_inf']:.3f}")
    print(f"domain x/L=[{blk.x_int[0, 0, 0, 0] / props['L']:.3f}, "
          f"{blk.x_int[0, -1, 0, 0] / props['L']:.3f}], "
          f"span/L={blk.x_int[2, 0, 0, -1] / props['L']:.3f}")
    if args.steps == 0:
        return

    Q = solver.init_state(lambda xc: initial(xc, props))
    t, writer = 0.0, None
    for it in range(1, args.steps + 1):
        start = time.perf_counter()
        dt = solver.max_dt(Q)
        Q = solver.advance(Q, dt)
        t += float(dt)
        if HAS_JAX:
            for q in Q:
                q.block_until_ready()
        if args.vtu and it % args.vtu == 0:
            from wmles.io_vtk import write_case
            writer = write_case(mesh, solver, Q, it, t, "out/smooth-ramp", writer=writer)
        if it % max(1, args.steps // 10) == 0 or it == 1:
            rho_min = min(float(xp.min(b.interior(q[0]))) for b, q in zip(mesh.blocks, Q))
            print(f"it={it:5d} t={t:.3e} dt={float(dt):.2e} rho_min={rho_min:.5g} "
                  f"step={time.perf_counter() - start:.3f}s")


if __name__ == "__main__":
    main()
