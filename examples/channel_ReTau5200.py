"""UMD wall-model validation case: compressible WMLES channel at Re_tau ~ 5200.
    https://wmles.umd.edu/validation/channel-flow-at-re5200/

    python -m examples.channel_ReTau5200 [level 1-4] [ar 1|2] [nsteps] [n_warmup] [out_every]
    python -m examples.channel_ReTau5200 --scheme kep2 --n-match 2

Prints the running friction Reynolds number (target ~5200) and, at the end, writes the
time- and plane-averaged mean and velocity-fluctuation profiles to
`out/channel_ReTau5200/`, using the same variables as the included Lee & Moser (2015)
DNS profiles. A positive `out_every` also writes VTU/PVD snapshots and a forces/moments CSV.
Choose `hybrid` (default), `kep2`, `kep4`, or `muscl-hllc` with `--scheme`.
`--n-match` selects the wall-normal cell sampled by the wall model, counted from each wall.

NOTE: this reproduces the UMD grids' node counts and Delta x = Delta z spacing exactly; the
wall-normal distribution is an engineering approximation (see channel5200.py docstring) since
the official ASCII y-files could not be reached from this sandbox. A real Re_tau=5200 run
needs the JAX/GPU backend and is far too expensive for a quick check -- use `--tiny` (a small
Re_tau to sanity-check the mechanics in seconds) if you just want to see the pipeline run.
"""
import argparse
import os
import subprocess
import time
from pathlib import Path
import numpy as np

from wmles.backend import xp, HAS_JAX, device_info
from wmles.channel5200 import (build_case, initial, flow_properties, PlaneStats,
                               GRID_TABLE, RE_TAU_TARGET)
from wmles.schemes import Hybrid, KEP, MUSCLHLLC


def _positive_int(value):
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be an integer") from exc
    if result < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return result


def _make_scheme(name):
    if name == "kep2":
        return KEP(2)
    if name == "kep4":
        return KEP(4)
    if name == "muscl-hllc":
        return MUSCLHLLC()
    return Hybrid(KEP(4))


def process_rss_mib():
    """Return this process's current resident set size in MiB."""
    result = subprocess.run(
        ("ps", "-o", "rss=", "-p", str(os.getpid())),
        check=True,
        capture_output=True,
        text=True,
    )
    return int(result.stdout.strip()) / 1024.0


def run(level, ar, nsteps, n_warmup, out_every, props=None, scheme=None, n_match=1):
    mesh, solver, props = build_case(
        level=level, ar=ar, props=props, scheme=scheme, n_match=n_match
    )
    Q = solver.init_state(lambda xc: initial(xc, props))
    stats = PlaneStats(mesh, solver)
    outdir = Path("out/channel_ReTau5200")
    from wmles.io_wall_loads import WallLoadCSVWriter
    wall_writer = WallLoadCSVWriter(outdir / "forces_moments.csv", (0.0, 0.0, 0.0))
    vtu_writer = None
    last_output_step = 0
    print(f"backend={'jax' if HAS_JAX else 'numpy'} device={device_info()}  grid={mesh.blocks[0].n}  "
         f"h={props['h']:.4e} m  Ub={props['Ub']:.2f} m/s  nu={props['nu0']:.3e} m2/s  "
         f"target Re_tau~{RE_TAU_TARGET:.0f} (u_tau_est={props['u_tau_est']:.3f} m/s)")
    t = 0.0
    first_step_time = None
    warm_step_time_sum = 0.0
    warm_step_count = 0
    for it in range(1, nsteps + 1):
        step_start = time.perf_counter()
        dt = solver.max_dt(Q)
        Q = solver.advance(Q, dt)
        t += float(dt)
        if HAS_JAX:
            for q in Q:
                q.block_until_ready()
        step_time = time.perf_counter() - step_start
        if it == 1:
            first_step_time = step_time
        else:
            warm_step_time_sum += step_time
            warm_step_count += 1
        if it > n_warmup:
            stats.accumulate(Q)
        if out_every > 0 and it % out_every == 0:
            from wmles.io_vtk import write_case
            vtu_writer = write_case(mesh, solver, Q, it, t, outdir,
                                    name="channel_ReTau5200", writer=vtu_writer)
            wall_writer.write(it, t, solver.wall_loads(Q))
            last_output_step = it
        if it % max(1, nsteps // 20) == 0 or it == 1:
            st = solver.wall_stats(Q)
            tau_w = np.mean([np.mean(v[0]) for v in st.values()])
            u_tau = float(np.sqrt(tau_w / props["rho0"]))
            re_tau = u_tau * props["h"] / props["nu0"]
            rss_mib = process_rss_mib()
            print(f"it={it:6d} t={t:.4e} dt={float(dt):.2e}  tau_w={tau_w:7.3f} Pa  "
                 f"u_tau={u_tau:.4f}  Re_tau={re_tau:8.1f} (target {RE_TAU_TARGET:.0f}, "
                 f"RSS={rss_mib:.1f} MiB)")
    if nsteps > last_output_step:
        wall_writer.write(nsteps, t, solver.wall_loads(Q))
        if out_every > 0:
            from wmles.io_vtk import write_case
            vtu_writer = write_case(mesh, solver, Q, nsteps, t, outdir,
                                    name="channel_ReTau5200", writer=vtu_writer)
    cells = sum(np.prod(b.n) for b in mesh.blocks)
    first_label = "compile+run" if HAS_JAX else "run"
    if warm_step_count:
        warm_mean = warm_step_time_sum / warm_step_count
        print(f"step timing: first_{first_label}={first_step_time:.4f}s  "
              f"warm_mean={warm_mean:.4f}s/step  cells_per_second={cells / warm_mean:.3g}")
    elif first_step_time is not None:
        print(f"step timing: first_{first_label}={first_step_time:.4f}s  warm_mean=n/a")
    return mesh, solver, props, stats, t, Q


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("level", nargs="?", type=int, choices=range(1, 5), default=1)
    parser.add_argument("ar", nargs="?", type=int, choices=(1, 2), default=1)
    parser.add_argument("nsteps", nargs="?", type=int, default=200)
    parser.add_argument("n_warmup", nargs="?", type=int, default=None)
    parser.add_argument("out_every", nargs="?", type=int, default=0)
    parser.add_argument(
        "--scheme", choices=("hybrid", "kep2", "kep4", "muscl-hllc"), default="hybrid",
        help="inviscid numerical flux scheme (default: hybrid)",
    )
    parser.add_argument(
        "--n-match", type=_positive_int, default=1,
        help="wall-normal cell sampled by the wall model, counted from each wall (default: 1)",
    )
    parser.add_argument(
        "--tiny", action="store_true",
        help="use a much lower-Re flow for a cheap CPU smoke test",
    )
    args = parser.parse_args()
    level, ar, nsteps, out_every = args.level, args.ar, args.nsteps, args.out_every
    n_warmup = args.n_warmup if args.n_warmup is not None else nsteps // 2
    if nsteps < 0 or n_warmup < 0 or out_every < 0:
        parser.error("nsteps, n_warmup, and out_every must be non-negative")
    if args.n_match >= GRID_TABLE[level][ar]["Ny"]:
        parser.error(
            f"n_match must not exceed the number of wall-normal cells "
            f"({GRID_TABLE[level][ar]['Ny'] - 1} for this grid)"
        )

    props = None
    if args.tiny:
        # A small Re_tau so the whole pipeline (incl. transients) is cheap to check on CPU;
        # NOT the validation case itself -- see the module docstring.
        props = flow_properties(Ma_b=0.2, Re_b=4_000.0)

    mesh, solver, props, stats, t, Q = run(
        level, ar, nsteps, n_warmup, out_every, props=props,
        scheme=_make_scheme(args.scheme), n_match=args.n_match,
    )
    if stats.n > 0:
        y, avg = stats.result()
        u_tau, wu = stats.wall_units(
            stats.mean_tau_w, props["rho0"], props["mu0"], props["h"]
        )
        outdir = Path("out/channel_ReTau5200")
        mean_columns = ("y/delta", "y+", "U+", "dU/dy+", "W+", "P+")
        fluctuation_columns = ("y/delta", "y+", "uu+", "vv+", "ww+", "uv+", "uw+", "vw+", "k+")
        np.savetxt(
            outdir / "mean_profile.dat",
            np.column_stack([wu[key] for key in mean_columns]),
            header="y/delta  y+  U+  dU/dy+  W+  P+",
            comments="% ",
        )
        np.savetxt(
            outdir / "velocity_fluctuations.dat",
            np.column_stack([wu[key] for key in fluctuation_columns]),
            header="y/delta  y+  u'u'  v'v'  w'w'  u'v'  u'w'  v'w'  k",
            comments="% ",
        )
        print(f"\ncollected {stats.n} samples over the last "
             f"{nsteps - n_warmup} steps ({t:.3e} s of the run)")
        print(f"{'y/h':>8s} {'U (m/s)':>10s} {'u_rms':>8s} {'v_rms':>8s} {'w_rms':>8s} {'-uv':>10s}")
        for j in range(0, len(y), max(1, len(y) // 12)):
            print(f"{y[j] / props['h']:8.4f} {avg['u'][j]:10.3f} "
                 f"{np.sqrt(max(avg['uu'][j], 0)):8.3f} {np.sqrt(max(avg['vv'][j], 0)):8.3f} "
                 f"{np.sqrt(max(avg['ww'][j], 0)):8.3f} {-avg['uv'][j]:10.4f}")