# wmles-jax — compressible wall-modelled LES, multiblock explicit finite volume (Python/JAX)

```
wmles/
  backend.py     JAX (float64) or NumPy fallback (WMLES_BACKEND=numpy) - same source for both
  gas.py         perfect gas + Sutherland; conservative <-> primitive
  mesh.py        Block, Mesh.connect (auto orientation + periodic shift), ghost exchange, metrics
  bc.py          SlipWall, NoSlipWall, Dirichlet, Extrapolate, WallModel (flux-override BC)
  wall_model.py  EquilibriumWallModel: Spalding (u_tau, Newton) + Kader (heat flux) / adiabatic recovery
  sgs.py         Smagorinsky, WALE, Vreman (or None = implicit LES)
  schemes.py     KEP (order 2/4 split-form central), MUSCL-HLLC, Hybrid (Ducros switch)
  viscous.py     Green-Gauss gradients, compact-corrected face gradients, viscous fluxes
  solver.py      apply_bcs -> RHS -> SSP-RK3, CFL dt, constant-mass-flux constraint, wall stats
  io_vtk.py      VTU/PVD export via pyvista (optional dependency, pip install pyvista)
  channel5200.py UMD wall-model validation case setup: grid table, flow properties, IC, plane/time stats
  plot3d.py      Plot3D multiblock reader/writer (formatted + unformatted, endianness/precision/iblank auto-detect)
  nmf.py         Neutral Map File parser, span re-discretization, topology builder with automatic block splitting
examples/channel_wmles.py   Ma_b=0.3 channel, 2 blocks in x, wall-modelled walls; 2nd CLI arg = VTU output stride
examples/re5200_channel.py  UMD Re_tau~5200 channel validation case driver (see below)
examples/airfoil_p3d.py     load Plot3D + NMF airfoil grid, build mesh, optional short run / VTU output
tests/data/                 NASA TMBWG airfoil near-wake grid L5 (nak_a_ff_500c_2zn_L5 .p3dfmt/.nmf)
tests/                      multiblock/orientation, free-stream, wall model, Sod tube
```

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
* `load_lm_profile(path)` reads a Lee & Moser `LM_Channel_*_mean_prof.txt` file if you
  download your own copy (e.g. from https://turbulence.oden.utexas.edu/channel2015/ or
  https://lee.me.uh.edu/) -- I could not fetch `LM_Channel_5200_mean_prof.txt` itself from
  here to hard-code the reference numbers, but the file format (comment lines starting
  with `%`, columns `y/delta, y+, U, dU/dy, W, P` in wall units) is confirmed against the
  Re_tau=180 file at that URL, so the loader should read the 5200 file as-is.

```
python -m examples.re5200_channel [level 1-4] [ar 1|2] [nsteps] [n_warmup] [out_every]
python -m examples.re5200_channel --tiny         # cheap CPU smoke test at a much lower Re
```
**Realistic cost note:** the real case (grid level 1 = 128x32x48 ~ 2e5 cells) needs on the
order of 1e5 explicit substeps to flush several flow-throughs and gather converged
statistics -- far too slow on the NumPy fallback used for testing here; this needs the
JAX/GPU backend. I only ran the `--tiny` smoke test (reduced Re, 40 steps, NumPy) in the
sandbox to confirm the case builds, runs, and the statistics/wall-unit pipeline works
end-to-end; I have not validated actual Re_tau=5200 numbers against DNS.

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

## Plot3D + NMF multiblock meshes (airfoil grids)
```python
from wmles import *
grid = read_plot3d("nak_a_ff_500c_2zn_L5.p3dfmt")                 # Plot3DGrid(nodes, iblank, ...)
nmf  = read_nmf("nak_a_ff_500c_2zn_L5.nmf")
nodes, nmf = respan(grid.nodes, nmf, ncells=8, length=0.1)         # 1 spanwise cell -> 8 cells, 0.1 c span
mesh = build_mesh(nodes, nmf,
                  bc_map={"viscous_solid": lambda: WallModel(EquilibriumWallModel(), T_wall=None),
                          "farfield_riem": Dirichlet(prim)},
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
* `farfield_riem` is only a fixed free-stream Dirichlet placeholder for now (reflective); the TMBWG
  grids are RANS-type (y+ ~ 1) so explicit dt is tiny - use a wall-modelled-LES grid for real runs.

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
* **Not yet run under JAX/GPU** (no JAX in the sandbox): the code uses only jit-safe, functional array ops,
  but expect to fix small things on first `jax.jit` run.

## Known limitations / next steps
* KEP is non-dissipative; for high-Re production runs add a small dissipation or entropy-stable flux.
* Near-wall SGS: Green-Gauss gradient in the first cell sees a slip ghost -> low mu_t there (usual WMLES caveat).
* MUSCL reconstructs in index space (assumes smooth stretching).
* Only Cartesian-orientation matching interfaces (conforming, 1:1 nodes); no rotational periodicity / hanging nodes.
* Ideas: TBLE/ODE wall model, dynamic Smagorinsky, WENO5, `jax.pmap`/`shard_map` over blocks, checkpointing (orbax),
  differentiable wall model calibration (everything is JAX-differentiable except the fori_loop Newton safeguards).
