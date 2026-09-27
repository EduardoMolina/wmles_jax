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
examples/channel_wmles.py   Ma_b=0.3 channel, 2 blocks in x, wall-modelled walls; 2nd CLI arg = VTU output stride
tests/                      multiblock/orientation, free-stream, wall model, Sod tube
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
