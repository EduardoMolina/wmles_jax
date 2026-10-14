"""UMD wall-model validation case: compressible WMLES channel at Re_tau ~ 5200.
    https://wmles.umd.edu/validation/channel-flow-at-re5200/

    python -m examples.channel_ReTau5200 [level 1-4] [ar 1|2] [nsteps] [n_warmup] [out_every]

Prints the running friction Reynolds number (target ~5200) and, at the end, writes the
time- and plane-averaged mean and velocity-fluctuation profiles to
`out/channel_ReTau5200/`, using the same variables as the included Lee & Moser (2015)
DNS profiles. A positive `out_every` also writes VTU/PVD snapshots and a forces/moments CSV.

NOTE: this reproduces the UMD grids' node counts and Delta x = Delta z spacing exactly; the
wall-normal distribution is an engineering approximation (see channel5200.py docstring) since
the official ASCII y-files could not be reached from this sandbox. A real Re_tau=5200 run
needs the JAX/GPU backend and is far too expensive for a quick check -- use `--tiny` (a small
Re_tau to sanity-check the mechanics in seconds) if you just want to see the pipeline run.
"""
import sys
import time
from pathlib import Path
import numpy as np

from wmles.backend import xp, HAS_JAX, device_info
from wmles.channel5200 import (build_case, initial, flow_properties, PlaneStats,
                               RE_TAU_TARGET, H_WM_H)


def run(level, ar, nsteps, n_warmup, out_every, props=None):
    mesh, solver, props = build_case(level=level, ar=ar, props=props)
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
    step_times = []
    for it in range(1, nsteps + 1):
        step_start = time.perf_counter()
        dt = solver.max_dt(Q)
        Q = solver.advance(Q, dt)
        t += float(dt)
        if HAS_JAX:
            for q in Q:
                q.block_until_ready()
        step_times.append(time.perf_counter() - step_start)
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
            print(f"it={it:6d} t={t:.4e} dt={float(dt):.2e}  tau_w={tau_w:7.3f} Pa  "
                 f"u_tau={u_tau:.4f}  Re_tau={re_tau:8.1f} (target {RE_TAU_TARGET:.0f})")
    if nsteps > last_output_step:
        wall_writer.write(nsteps, t, solver.wall_loads(Q))
        if out_every > 0:
            from wmles.io_vtk import write_case
            vtu_writer = write_case(mesh, solver, Q, nsteps, t, outdir,
                                    name="channel_ReTau5200", writer=vtu_writer)
    warm = step_times[1:]
    cells = sum(np.prod(b.n) for b in mesh.blocks)
    first_label = "compile+run" if HAS_JAX else "run"
    if warm:
        warm_mean = np.mean(warm)
        print(f"step timing: first_{first_label}={step_times[0]:.4f}s  "
              f"warm_mean={warm_mean:.4f}s/step  cells_per_second={cells / warm_mean:.3g}")
    elif step_times:
        print(f"step timing: first_{first_label}={step_times[0]:.4f}s  warm_mean=n/a")
    return mesh, solver, props, stats, t, Q


if __name__ == "__main__":
    argv = sys.argv[1:]
    tiny = "--tiny" in argv
    argv = [a for a in argv if a != "--tiny"]
    level = int(argv[0]) if len(argv) > 0 else 1
    ar = int(argv[1]) if len(argv) > 1 else 1
    nsteps = int(argv[2]) if len(argv) > 2 else 200
    n_warmup = int(argv[3]) if len(argv) > 3 else nsteps // 2
    out_every = int(argv[4]) if len(argv) > 4 else 0

    props = None
    if tiny:
        # A small Re_tau so the whole pipeline (incl. transients) is cheap to check on CPU;
        # NOT the validation case itself -- see the module docstring.
        props = flow_properties(Ma_b=0.2, Re_b=4_000.0)

    mesh, solver, props, stats, t, Q = run(level, ar, nsteps, n_warmup, out_every, props=props)
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