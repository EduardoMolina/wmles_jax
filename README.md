# wmles-jax — compressible wall-modelled LES, multiblock explicit finite volume (Python/JAX)

```
wmles/
  backend.py     JAX (float64, selected device) or NumPy fallback (WMLES_BACKEND=numpy)
  gas.py         perfect gas + Sutherland; conservative <-> primitive
  mesh.py        Block, Mesh.connect (auto orientation + periodic shift), ghost exchange, metrics
  bc.py          SlipWall, NoSlipWall, Dirichlet, CharacteristicFarfield, Extrapolate, WallModel
  wall_model.py  EquilibriumWallModel: Spalding (u_tau, Newton) + Kader (heat flux) / adiabatic recovery
  sgs.py         Smagorinsky, WALE, Vreman (or None = implicit LES)
  schemes.py     KEP (order 2/4 split-form central), MUSCL-HLLC, Hybrid (Ducros switch)
  viscous.py     Green-Gauss gradients, compact-corrected face gradients, viscous fluxes
  solver.py      apply_bcs -> RHS -> SSP-RK3, CFL dt, constant-mass-flux constraint, wall stats/loads
  io_vtk.py      VTU/PVD export and restart loading via pyvista (optional dependency)
  io_wall_loads.py CSV export of integrated wall forces and moments
  channel5200.py UMD wall-model validation case setup: grid table, flow properties, IC, plane/time stats
  smooth_ramp.py UMD smooth-ramp grid-6 WMLES mesh and solver setup
  plot3d.py      Plot3D multiblock reader/writer (formatted + unformatted, endianness/precision/iblank auto-detect)
  nmf.py         Neutral Map File parser, span re-discretization, topology builder with automatic block splitting
examples/channel_wmles.py   Ma_b=0.3 channel, 2 blocks in x, wall-modelled walls; 2nd CLI arg = VTU output stride
examples/channel_ReTau5200.py  UMD Re_tau~5200 channel validation case driver (see below)
examples/smooth_ramp.py        UMD smooth-ramp case driver (see below)
examples/airfoil_p3d.py     load Plot3D + NMF airfoil grid, build mesh, optional short run / VTU output
tests/data/                 NASA TMBWG airfoil near-wake grid L5 (nak_a_ff_500c_2zn_L5 .p3dfmt/.nmf)
tests/                      multiblock/orientation, free-stream, wall model, Sod tube
```

## Smooth-ramp setup
The UMD smooth-ramp validation case is described at
https://wmles.umd.edu/smooth-ramp/. `wmles/smooth_ramp.py` builds a single-block
WMLES setup using the three `baurle2021wsm_grid6_*.txt` distributions in `tests/data/`:
the streamwise spacing at the wall and the upstream/downstream wall-normal distributions.
It maps the upstream wall at `y/L=0.22` smoothly to the downstream wall at `y/L=0`
using the published quintic ramp, with the supplied `y/L=0.74` top boundary. The
default domain is `x/L=-0.76..4`, the periodic span is `0.288L`, and the physical scale
is set from `Re_L=667,000` and `Ma_inf=0.2`. The provided streamwise grid is extended
outside `x/L=-0.05..1.05` using constant end spacing; the spanwise distribution is
uniform because no spanwise grid file is included.

The bottom wall uses the equilibrium wall model, the top uses slip, the span is periodic,
and the outflow is characteristic. The inlet and initial state use the supplied DNS
profile at `x/L=-0.76` in `tests/data/smooth_ramp_dns/`. A deterministic Fourier-mode
synthetic field applies the profile's `uu`, `vv`, `ww`, and `uv` stresses; its default
correlation scales are `(0.1, 0.02, 0.03)L` in streamwise, wall-normal, and spanwise
directions. Override these and the phase seed with `--inflow-scales LX LY LZ` and
`--inflow-seed`. The DNS data contain only means and second moments, not time histories
or correlation lengths, so this generator matches those local moments statistically;
it cannot reproduce DNS temporal coherence. The missing `uw` and `vw` covariances are
assumed zero.

An optional-by-CLI sponge source is enabled by default over the last `1L` before
outflow (or from `--sponge-start x/L`), with maximum rate `U_inf/L`. It smoothly damps
departures from each instantaneous spanwise mean and therefore preserves that mean
profile. Spanwise-uniform disturbances are intentionally left to the characteristic
outflow condition. Use `--sponge-strength` to change the maximum rate or `--no-sponge`
to disable it. Physical time is passed to inflow boundary fills at the SSP-RK stage
times, keeping the generated field deterministic and JIT-compatible. The gas uses the
project's Sutherland viscosity model; the site notes constant viscosity is also
reasonable at this low Mach.

```bash
python -m examples.smooth_ramp                         # build/report grid only
python -m examples.smooth_ramp --steps 10 --nz 16     # short solver smoke run
python -m examples.smooth_ramp --x-min -0.05 --x-max 1.05 --nz 4  # smaller setup check
python -m examples.smooth_ramp --inflow-scales 0.1 0.02 0.03 --sponge-start 3.0
python -m examples.smooth_ramp --steps 100 --vtu 20   # optional ParaView output
```

## A-airfoil grid
`wmles/airfoil_grid.py` generates a four-block Plot3D or VTK mesh from the
provided A-airfoil contour in `tests/data/a-airfoil.dat`. The default periodic
span is `0.0481c`, with 40 uniformly spaced spanwise cells
(`Delta z = 0.0012025c`).
The span is deliberately an integer multiple of the requested spacing so every
spanwise cell has the same width.
The tunnel block defaults to cross-stream spacing smoothing over 10% of its
streamwise index range. This blends each line's normalized arclength spacing
with a Gaussian-smoothed distribution along the block, fading to zero at the
block ends to retain exact interface nodes. A gentler normal-projection blend
also avoids folded cells at the tunnel corners. Use `--tunnel-smoothing 0` to
disable spacing smoothing.

```bash
python -m wmles.airfoil_grid out/a_airfoil.p3d
python -m wmles.airfoil_grid out/a_airfoil.vtm
python -m wmles.airfoil_grid out/a_airfoil.p3d --tunnel-smoothing 0.15
python -m wmles.airfoil_grid out/a_airfoil.p3d --tunnel-smoothing 0
```

`wmles/airfoil_case.py` sets up the published A-airfoil reference conditions
from the [Tamaki-Kawai database README](https://www.klab.mech.tohoku.ac.jp/database/2022TamakiAIAAJ/README.txt):
Mach 0.15, `Re_c = 1.0e7`, and 13.3-degree incidence. It applies a characteristic
far-field, spanwise periodicity, and an adiabatic equilibrium wall model on the airfoil.
Because the database README does not specify dimensional pressure or temperature,
the default SI scale assumes standard ambient air at 300 K and 101325 Pa; override
these using `--temperature` and `--pressure`. The adiabatic wall treatment is a
WMLES modeling assumption, not a thermal condition stated in the database README.

```bash
python -m examples.airfoil_wmles out/a_airfoil.p3d          # assemble/report setup
python -m examples.airfoil_wmles out/a_airfoil.p3d --steps 10
python -m examples.airfoil_wmles out/a_airfoil.p3d --steps 100 --vtu 20
```

The example reconstructs the four-block interfaces from matching grid faces.
The default generated grid has 40 spanwise cells; the solver requires at least
three spanwise cells for its ghost layers. The cited database is a wall-resolved
LES reference; this driver applies WMLES boundary treatment at those published
flow conditions and is not itself a reproduction of the database solution.

## Re_tau = 5200 channel validation case
`wmles/channel5200.py` sets up the UMD wall-model validation case
(https://wmles.umd.edu/validation/channel-flow-at-re5200/), reproducing incompressible DNS
by Lee & Moser (2015) with a compressible solver at bulk Mach 0.2:
* Domain `(12.8h, 2h, 4.8h)`, periodic in x/z, wall-modelled (adiabatic) walls at y=0,2h.
* `GRID_TABLE` reproduces the site's 4 resolution levels x 2 aspect ratios (AR=1, AR=2)
  exactly in node count and Delta x = Delta z spacing (`Nx,Ny,Nz` from their table).
* `flow_properties()` picks the channel half-height `h` and bulk velocity so that, with
  real air viscosity, `Re_b = U_b h / nu = 125,000` (Ma_b = 0.2), matching their target
  `Re_tau ~ 5200`.
* **Caveat:** the site's own wall-normal grid coordinates (`channel_1dwm_*.txt`) were not
  reachable from this sandbox (network disabled here / the direct links 404'd when tried
  from a normal fetch) -- `clustered_y()` instead builds a symmetric tanh-clustered grid
  tuned to their stated `Delta y_wall ~= 0.25 Delta y_center` design ratio. Swap in the
  official y-coordinates yourself (pass your own `nodes` array to `Block(...)`) for an
  exact reproduction.
* `PlaneStats` accumulates plane- (x,z-) and time-averaged mean velocity, temperature and
  the resolved Reynolds stresses; `.wall_units(...)` rescales the lower half of the channel
  into `y+`, `U+` for comparison with a log law or DNS data.
* `load_lm_profile(path)` reads a Lee & Moser mean profile. The Re_tau=5200 DNS reference
  profiles are included in `tests/data/` as `LM_Channel_5200_mean_prof.dat` and
  `LM_Channel_5200_vel_fluc_prof.dat`. Each run writes comparison profiles with the same
  variables to `out/channel_ReTau5200/mean_profile.dat` and
  `out/channel_ReTau5200/velocity_fluctuations.dat`.

```bash
python -m examples.channel_ReTau5200 [level 1-4] [ar 1|2] [nsteps] [n_warmup] [out_every]
python -m examples.channel_ReTau5200 --scheme kep2 --n-match 2
python -m examples.channel_ReTau5200 --tiny         # cheap CPU smoke test at a much lower Re
```
Choose `--scheme hybrid` (default), `kep2`, `kep4`, or `muscl-hllc`. `--n-match`
selects the wall-normal cell sampled by the wall model, counted from each wall
(default: first cell centre, `1`).
When `out_every` is positive, the driver writes VTU/PVD snapshots and a
`forces_moments.csv` sample at that stride (plus the final step when it falls between
strides). The CSV is still written at the final step when `out_every` is zero; zero
disables VTU output.
**Realistic cost note:** the real case (grid level 1 = 128x32x48 ~ 2e5 cells) needs on the
order of 1e5 explicit substeps to flush several flow-throughs and gather converged
statistics -- far too slow on the NumPy fallback used for testing here; this needs the
JAX backend. CPU/JAX integration checks are available, but GPU execution and actual
Re_tau=5200 agreement with DNS have not been validated here.

## JAX device selection
JAX uses its default device when `WMLES_DEVICE` is unset or `auto`. Set it to `cpu` to
force CPU execution, or `gpu` to require an available accelerator. The GPU setting fails
at startup if JAX cannot initialize a GPU, preventing an unnoticed CPU fallback. Solver
startup logs report the selected device. `WMLES_BACKEND=numpy` continues to force NumPy
and cannot be combined with `WMLES_DEVICE=gpu`.

Run the solver checks locally with JAX on CPU:

```bash
WMLES_DEVICE=cpu python -m tests.test_multiblock
WMLES_DEVICE=cpu python -m tests.test_sod
```

On the provisioned offline Ubuntu host, first confirm its existing JAX installation can
see the H200, then require GPU placement during checks and runs:

```bash
WMLES_DEVICE=gpu python -c 'import jax; print(jax.devices())'
WMLES_DEVICE=gpu python -m tests.test_multiblock
WMLES_DEVICE=gpu python -m tests.test_sod
WMLES_DEVICE=gpu python -m examples.channel_wmles 1
WMLES_DEVICE=gpu python -m examples.channel_ReTau5200 1 1 10 5 0
```

## Visualization (pyvista, .vtu/.pvd)
`pip install pyvista` (needs the `vtk` package too, which pyvista pulls in). Not required to
run the solver -- only `wmles/io_vtk.py` imports it, and only when you call it.

```python
from wmles.io_vtk import write_case
writer = None
for it in range(nsteps):
    ...
    if it % 50 == 0:
        writer = write_case(mesh, solver, Q, it, t, "out/channel", writer=writer)
```
This writes one hexahedral `.vtu` per block per output step (cell data: rho, velocity,
pressure, temperature, Mach, mu_t, mu, vorticity magnitude, Q-criterion), a `.pvd`
collection tying the whole run together as one time-aware object in ParaView, and, for
every `WallModel` face, a 2D quad-mesh `.vtu` under `out/channel/wall/` with tau_w and q_w
as cell data. `examples/channel_wmles.py N M` runs N steps and writes VTU every M steps.
Lower-level building blocks (`block_grid`, `wall_patch_grid`, `derived_fields`) are there
if you want a custom field list or your own output cadence/naming.
The connectivity/index math (which node goes with which cell, field ordering) is unit-tested
in the sandbox with a lightweight pyvista stub, since pyvista itself isn't installed there --
worth a quick smoke test on your machine with the real package before a long run.

### Restarting from a VTU/PVD checkpoint

`examples.channel_wmles` can continue from the latest checkpoint in a `.pvd` collection,
or from one of its per-block `.vtu` files. Pass the number of additional steps to run as
the first positional argument; step and simulation time continue from the saved checkpoint:

```bash
python -m examples.channel_wmles 200 50 --restart out/channel/case.pvd
python -m examples.channel_wmles 200 50 \
  --restart out/channel/case_block0_000100.vtu
```

When restarting, VTU output continues in the checkpoint's directory and appends to its PVD
collection. If force output is enabled, rows append to the existing wall-load CSV after its
header is checked. Restart loads all block files for the selected step and checks that their
mesh and required primitive fields match the configured solver. New VTUs store the step and
time as metadata; older files can be restarted from their PVD collection, which supplies the
time metadata. PyVista is required to read or write VTU files.

Integrated fluid-on-wall forces and moments for `WallModel` faces can be written at selected
steps independently of VTU output:

```bash
python -m examples.channel_wmles 200 0 --force_steps 100 \
  --moment-center 0.01 0.02 0.0 --force-file out/channel_wall_loads.csv
```

`--force_steps` is a positive step number no greater than `N`.
The CSV contains one row per wall face and a total row per selected step; forces are in N,
moments in N m, and moments are taken about the supplied reference point (default origin).
`Solver.wall_loads(Q, moment_center=...)` provides the same integrated values as
`{(block, face): (force, moment)}` for Python callers.

## Plot3D + NMF multiblock meshes (airfoil grids)
```python
from wmles import *
grid = read_plot3d("nak_a_ff_500c_2zn_L5.p3dfmt")                 # Plot3DGrid(nodes, iblank, ...)
nmf  = read_nmf("nak_a_ff_500c_2zn_L5.nmf")
nodes, nmf = respan(grid.nodes, nmf, ncells=8, length=0.1)         # 1 spanwise cell -> 8 cells, 0.1 c span
mesh = build_mesh(nodes, nmf,
                  bc_map={"viscous_solid": lambda: WallModel(EquilibriumWallModel(), T_wall=None),
                          "farfield_riem": CharacteristicFarfield(prim)},
                  periodic=(("symmetry_y", 0),))                   # spanwise periodic
print(describe(mesh))
```
* **Plot3D:** whole-format multiblock 3-D files (also 2-D, returned with a singleton k axis;
  `extrude_2d` turns those into 3-D blocks). Formatted or Fortran-unformatted, little/big endian,
  4/8-byte reals and optional iblank are auto-detected. `write_plot3d` writes the same formats.
* **NMF:** parsed with the NASA face convention (1=kmin, 2=kmax, 3=imin, 4=imax, 5=jmin, 6=jmax; tangential
  order i,j / j,k / k,i), including reversed ranges and the Swap flag. Supported entries: `one_to_one` and
  boundary types (`viscous_solid`, `symmetry_y`, `farfield_riem`, ... -> map them to BC objects with
  `bc_map`; `wall`/`viscous_solid` -> `NoSlipWall` and `symmetry_*` -> `SlipWall` by default).
* **Sub-face patches -> block splitting.** `Mesh.connect` joins whole faces, but the wake block of the
  TMBWG airfoil grid is joined half to the airfoil block's `jmax` face and half to its `jmin` face.
  `build_mesh` cuts blocks along every patch boundary (propagating the cuts through interfaces until
  nothing changes), so each sub-block face is exactly one patch; here the wake block becomes two blocks
  (`b1_000`, `b1_001`) cut along the trailing-edge line. Every interface's orientation is re-derived
  from node coordinates by `Mesh.connect`, so an inconsistent NMF fails loudly instead of silently.
* **Span:** the TMBWG grids have one spanwise cell (`y` in [-1,0], span = index i); the solver needs
  >= `ng` (=3) cells per direction, so use `respan`. `periodic=(type, axis)` turns min/max faces of
  that BC type into a periodic pair (translation taken from the nodes).

**Verified (NumPy backend, in the sandbox), on the real L5 grid and synthetic cases**
* Plot3D round trips: formatted/unformatted, LE/BE, float32/64, with/without iblank; 2-D read.
* Interface handling: two blocks with a transposed + flipped neighbour (Swap=TRUE, reversed range) reproduce
  the single-block RHS to 2e-12; a 4-block case where one block's cut must propagate into a neighbour builds
  and preserves a uniform stream at round-off (2e-16 scaled residual). Mutation checks confirm both tests
  fail if cut propagation or the Swap flag is broken.
* TMBWG L5 airfoil wake grid (3 sub-blocks, 16.6k cells with 4 span cells): volumes conserved, ghost-cell
  volumes match across every interface (5e-13), uniform free stream preserved at round-off (~4e-16 scaled
  residual) for KEP, MUSCL-HLLC and Hybrid; short runs with no-slip and wall-model airfoil surface stay finite.

**Not covered / caveats**
* Only the L5 file was available here; L1-L4 share the topology but were not run.
* Unsupported NMF entries (raise `NotImplementedError`): `patched`, `collapsed`, `pole_dir1/2`.
  Periodicity across *different* blocks (translation shift) can't be expressed in an NMF; `periodic=`
  handles a block's own min/max faces only.
* Points where 3 or 5 blocks meet (extraordinary vertices) and their diagonal ghost cells are not stress-tested;
  the stencils only use face neighbours and the L5 case (a valence-4 trailing-edge point) works.
* `CharacteristicFarfield` retains outgoing linearized Euler characteristics and imposes incoming
  characteristics from the supplied free stream. The TMBWG grids are RANS-type (y+ ~ 1) so explicit
  dt is tiny - use a wall-modelled-LES grid for real runs.

## Design
* State: tuple of `(5, Ni, Nj, Nk)` arrays (one per block, ghosts included), `q = [rho, rho*u_vec, rho*E]`
  with **physical Cartesian** momentum -> block halo exchange is a pure index remap, no vector rotation.
* Geometry: face area vectors + divergence-theorem volumes from (ghost-extended) nodes, so a uniform
  stream is preserved exactly on curvilinear grids. Metrics are passed to jitted functions as arguments.
* Blocks: `mesh.connect(A, "imax", B, "kmin", shift=...)`. Orientation (axis permutation / flips) is
  detected by matching interface node planes. Periodicity = connection with `shift` (self-connection allowed).
* Wall model BC: ghost cells are slip-mirrored for the numerics; the wall-face flux is replaced by
  `[0, p n + tau_w t, -q_w] |S|`, with (tau_w, q_w) from the wall model sampled `n_match` cells off the wall.
* Adding a piece = writing one callable: a scheme `(blk, geom, w, gas, d, theta) -> F`, an SGS model,
  a wall model `(gas, h, U, T_h, p_h, T_wall) -> (tau_w, q_w)`, a BC with `fill(...)`, a constraint or source.

## Verified (NumPy backend, in the sandbox)
* 2-block split, 180-degree-flipped block and cyclically-permuted block reproduce the single-block RHS to ~1e-15
  (skewed curvilinear grid, periodic in x/z, wall-model and slip walls, KEP and Hybrid).
* Free-stream preservation at round-off on a skewed grid (KEP2/4, MUSCL-HLLC, Hybrid).
* Spalding solve recovers a prescribed u_tau; Sod tube plateau values match exact solution;
  channel run conserves mass to 1e-15 and holds the target mass flux.
* The multiblock solver checks run under JAX CPU; GPU execution remains to be verified on
  the target H200 host.

## Known limitations / next steps
* KEP is non-dissipative; for high-Re production runs add a small dissipation or entropy-stable flux.
* Near-wall SGS: Green-Gauss gradient in the first cell sees a slip ghost -> low mu_t there (usual WMLES caveat).
* MUSCL reconstructs in index space (assumes smooth stretching).
* Only Cartesian-orientation matching interfaces (conforming, 1:1 nodes); no rotational periodicity / hanging nodes.
* Ideas: TBLE/ODE wall model, dynamic Smagorinsky, WENO5, `jax.pmap`/`shard_map` over blocks, checkpointing (orbax),
  differentiable wall model calibration (everything is JAX-differentiable except the fori_loop Newton safeguards).

## Developer
```bash
JAX_LOG_COMPILES=1  python -m examples.channel_ReTau5200
```