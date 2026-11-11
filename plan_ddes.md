I read the solver, SGS, viscous, BC, wall-model, scheme, mesh, backend and sharding modules, plus `plan.md` and the README. This is a plan only; nothing was changed.

## What the code looks like today (relevant to both features)

- **State.** `Q` is a tuple of per-block `(5, Ni, Nj, Nk)` arrays that include ghost cells. Halo exchange, RK stages and block sharding are all generic in the leading axis. `solver.py` is the reference path and `sharding.py` is a per-block twin that reuses its hooks (`_block_rhs_padded`, `_block_dt`, `apply_bcs`).
- **Turbulence closure.** `make_sgs` in `sgs.py` returns `f(grad_u, delta, rho) -> mu_t`, with `delta = V^(1/3)`. It is purely algebraic, so SA does not fit that signature. SA needs a transported variable, wall distance, Δ_max and a source term.
- **Viscous flux.** `viscous_flux` in `viscous.py` already takes `mu_eff` and `k_eff`, so SA's μ_t plugs in without changes. It is hard-wired to the four variables `[u,v,w,T]`, so scalar diffusion needs its own function.
- **Hard-coded 5-variable assumptions.** `Gas.cons_to_prim` and `prim_to_cons` build 5-row arrays. The BC fills in `bc.py` use `w[:1]` and `w[4:5]`, and the `Dirichlet` and `CharacteristicFarfield` fills reshape with `(5,)`. `ConstantMassFlux`, `io_vtk` and the `init_state` callers also assume five rows.
- **Wall BCs.** `WallModel` uses slip ghost cells plus a flux override. `NoSlipWall` mirrors the state. Neither has any wall-distance field; `WallModel.wall_flux` only computes `h` at the match point.
- **Time stepping.** It is explicit SSP-RK3 with a CFL from convective and viscous spectral radii. The smoke run in `plan.md` gives dt ≈ 1e-7 s, so wall-normal stiffness is the bottleneck.
- **Enablers for JFNK.**
  - float64 is on.
  - `_rhs(Q, geoms, time)` is a pure function, and the RHS is already divided by volume.
  - `Solver.rhs` and `DistributedSolver.rhs` share one signature, so a matrix-free operator can be written against either.
- **JFNK obstacles.**
  - The hybrid scheme's non-smooth pieces are `where`-based HLLC branches, limiters, the Ducros sensor and the positivity fallback.
  - The wall-model `fori_loop` Newton uses `clip` safeguards.
  - `DistributedSolver` orchestrates halos in Python, so a single `jax.jit` or `lax` GMRES cannot wrap it.
  - The central KEP scheme is non-dissipative, which makes it a poor basis for a preconditioner.

---

## Part A — Spalart–Allmaras and SA-DDES

### A0. Design decisions (assumed unless you object)

1. **Variable layout:** append `ρν̃` as `q[5]`, so `Q` becomes `(6, …)` when SA is on. Halo exchange, RK, sharding and restart then work unchanged. The 5-variable path stays bit-identical when SA is off.
   - `cons_to_prim` and `prim_to_cons` keep operating on `q[:5]`, and the solver splits off `q[5:]`.
   - The alternative, a second pytree field, would duplicate halo and sharding code.
2. **Turbulence abstraction:** new `wmles/rans.py` (or `turbulence.py`) with a `SpalartAllmaras(variant="noft2", des=None | "ddes")` object. It exposes `n_scalars`, `mu_t(...)`, `source(...)`, `diffusion_flux(...)`, `scalar_bc(...)` and `init(...)`. `Solver(..., turbulence=...)` takes it, and the existing `sgs=` argument stays as the LES closure.
3. **Model variants:**
   - SA-noft2 with the Edwards–Chandra Ŝ limiter (S̃ ≥ 0.3 Ω).
   - **SA-neg** (Allmaras 2012) negative-ν̃ treatment. This is needed for robustness and is mandatory for the JFNK phase.
   - DDES (Spalart 2006): `d̃ = d − f_d·max(0, d − C_DES·Δ)`, `f_d = 1 − tanh[(8 r_d)³]`, C_DES = 0.65.
   - Δ options: Δ_max (default), Δ_ω (vorticity-aligned, better for the spanwise-periodic airfoil), and the SLA option later.

### A1. Geometry: wall distance and Δ_max
- Add wall distance `d` (cell-centred, ghosts included) and `delta_max` to `Geom` in `mesh.py`, as optional fields. `Geom` is a `NamedTuple`, so `device_put` in sharding still works.
- New `wmles/wall_distance.py`, computed on the host once, before sharding:
  - Collect wall faces from all blocks whose BC is a wall (`NoSlipWall` or `WallModel`).
  - Use a `scipy.spatial.cKDTree` (scipy is already used in `airfoil_grid.py`) on face centres.
  - Refine candidates with point-to-quad distance, so coarse tangential spacing doesn't bias `d` near the wall.
  - Add periodic-shift copies for periodic spans.
- Tests: flat-plate `d = y` exactly, and a multi-block split gives the same `d` as the unsplit mesh.

### A2. Scalar transport and BCs
- **Convection:** use the mass flux from the existing scheme. After `self.scheme(...)`, set `F[5] = F[0]·ν̃_upwind` with a limited, second-order upwind face value. This works with KEP, MUSCL or Hybrid without touching them.
- **Diffusion:** a new `scalar_diffusion_flux` in `viscous.py`, reusing the compact face-gradient correction. It computes `(1/σ)(μ+ρν̃)∇ν̃·S`. The `cb2 ρ|∇ν̃|²/σ` term goes in the source.
- **BC hooks:** add `scalar_ghost(...)` to the `BC` base class, with these behaviours.

  | BC | ν̃ ghost value |
  |---|---|
  | `NoSlipWall` | `−ν̃_interior`, so ν̃_face = 0 |
  | `SlipWall`, `Extrapolate` | copy |
  | `CharacteristicFarfield` / `Dirichlet` | ν̃∞ (χ ≈ 3–5) for inflow, extrapolate for outflow |
  | `Interface` | free via the generic halo exchange |

- Fix the `(5,)` reshapes and the `w[:1]` / `w[4:5]` slicing in `bc.py` so the 6th row passes through.
- **Wall-model case (WMLES):**
  - Keep `WallModel`'s flux override for τ_w and q_w.
  - Impose the near-wall ν̃ from the wall-model u_τ, for example ν̃ = κ u_τ y at the match cell. This is separate and optional (A6), because the DDES buffet grids may be wall-resolved.

### A3. Solver integration
- In `_block_rhs`:
  1. Compute ν̃ and `fv1` from `q[5]/ρ`.
  2. Set `mut = ρ ν̃ fv1`, replacing the SGS `mut` when SA is on.
  3. Add the SA source and the scalar flux rows to `rhs`.
- In `_block_dt`, add the SA source and diffusion stiffness limit.
- Positivity: clip or SA-neg after each RK stage. Add a per-stage hook in `_rk1/2/3` and the sharding `_rk*_blk` kernels.
- Move SGS-versus-SA selection and the Ducros-style shield into the hybrid scheme (A5).
- Update `sharding.py`: `init_state`, `_block_totals` and `conserved_totals`.
- Update `io_vtk.py` to write `nut_tilde`, `mu_t`, `d` and `f_d`.
- Restart: the 6th component should round-trip through the VTU/PVD files.
- Add the SA initial condition to `airfoil_case.py` and `examples/airfoil_wmles.py` (`--turbulence sa|sa-ddes`).

### A4. Validation ladder
1. **Unit tests.** SA functions and limits, `f_d → 0` in the boundary layer and `f_d → 1` in separated flow, and equivalence with the 5-variable path when `ν̃ ≡ 0`.
2. **Regression.** 2-block split versus single-block RHS (extend `test_multiblock`), and free-stream preservation including ν̃. Also `test_sharding` bit-equivalence with 6 variables, and `d` and `Δ_max` agreement across splits.
3. **Physics.**
   - Zero-pressure-gradient flat plate: Cf against NASA TMR SA references.
   - NACA0012 with the existing `test_naca0012_cgrid.py` grids: Cl and Cd.
   - Channel DDES: `examples/channel_wmles.py`.
   - Then the OAT15a buffet.
4. **Numerics for DDES.** Add a switch that blends central and upwind using `f_d` and the Ducros sensor, because the KEP scheme is non-dissipative in RANS regions. This is the main risk for the grey zone.

### A5/A6 (later)
- IDDES, SLA Δ, and WM-DDES wall treatment.

---

## Part B — Implicit JFNK operator

### B0a. Formulation for Steady-State
- Solve `R(Q) = (Q − Qⁿ)/Δt − rhs(Q) = 0`, matching the per-volume residual the solver already computes.
- Time integration: backward Euler and BDF2 for time-accurate steps. Steady RANS uses backward-Euler pseudo-time continuation with **local time stepping** (below).
- Where the speedup comes from: wall-normal stiffness in wall-resolved cells. Implicit time steps are limited only by the convective CFL in the LES region, which could be about 100× larger than today's dt of 1e-7 s. In isotropic LES regions the benefit is smaller.
- First implementation: `wmles/implicit.py` provides backward-Euler and fixed-step BDF2 residuals, finite-difference Newton-GMRES, and a geometric CFL-ramped backward-Euler pseudo-time driver. `examples/flatplate_sa_implicit.py` targets the wall-resolved 2DZP SA case.
- Status: B0a-LTS implemented (per-cell Δt, SER ramp, half-CFL retry, variable and Δt row scaling of the Newton/GMRES system). 2DZP, CFL 1 → ~7 in 150 steps, residual 1.6e10 → 1.1e9; GMRES iterations per step grow from ~40 to ~200 as CFL rises. Remaining: B3 preconditioner, and the C_f cross-check against explicit `step_local` at convergence.
- Row scaling matters: without it the ρν̃ row was ignored by the norm and FD step, and global-Δt GMRES needed 53 iterations at CFL 1 (now 8). Local Δt additionally needs the `dt/scale` left scaling (61 vs failure).

#### B0a-LTS. Local (pseudo-)time stepping for the implicit steady solver
Steady-state only (not time accurate); the unsteady/DDES path in B0b keeps a uniform physical Δt.
- **Residual.** Replace the scalar Δt by a per-cell field `Δtᵢ = CFLₚₛ · Δtᵢ^explicit(1.0)` and solve
  `R(Q) = (Q − Qⁿ)/Δtᵢ − rhs(Q) = 0`. `Δtᵢ` is an array per block (ghost-padded, same layout as `Solver.local_dt`), so the residual is `(q − qⁿ) / dt[None] − rhs`, and the pseudo-time diagonal is `V/Δtᵢ` in B3.
- **Δt field.** Reuse `Solver.local_dt(Q)` (it already includes SA diffusion stiffness through ν + ν̃). Because `solver.cfl` is baked in, rescale by `CFLₚₛ / solver.cfl`. Add an optional per-cell cap `Δtᵢ ≤ Δt_max` for robustness.
- **Freezing.** Compute `Δtᵢ` once per Newton solve (from `Qⁿ`) and keep it fixed during the Newton and Krylov iterations. Otherwise J·v would differentiate through `Δt(Q)` and the FD operator would be inconsistent with the residual.
- **API.** `ImplicitIntegrator.advance(Qn, dt, ...)` accepts `dt` as a scalar or a per-block tuple of arrays (as `Solver._per_block` does). Add `run_steady_cfl(..., local=True)` as the default for steady runs, and keep `local=False` (global Δt) for comparison and tests. The example gets `--local-dt/--global-dt` (default local).
- **CFL ramp.** Ramp `CFLₚₛ` by SER on the residual: `CFLₚₛ ← min(CFL_max, CFLₚₛ · (‖R⁰‖/‖Rⁿ‖)^p)`, with the geometric ramp as fallback. Start at 1–10 in the first iterations and grow to 10³–10⁶ as the residual drops. Reduce it (and restart the step from `Qⁿ`) when the Newton line search fails.
- **Consistency checks.**
  - With a uniform Δt field, the local-dt residual and the step must equal the scalar-Δt result to round-off (mirrors `test_local_dt_step_equals_global_dt_step_when_dt_is_uniform`).
  - As `CFLₚₛ → 0`, one implicit step should agree with one explicit `advance_local` step.
  - The converged steady state must not depend on the Δt field. Compare 2DZP C_f with the global-Δt and the explicit-local results.
- **Caveats.**
  - Local Δt makes the transient meaningless (not time accurate), so BDF2 and the unsteady path never use it.
  - With very large Δtᵢ the diagonal `V/Δtᵢ` vanishes and the system tends to the steady Jacobian, so the preconditioner (B3) is what keeps GMRES converging. Expect the iteration count to grow with CFLₚₛ.
  - The cell-to-cell Δt jumps across block interfaces are harmless, because ghost cells are exchanged as state and Δt is not communicated.
  - In sharded runs `Δtᵢ` stays per block, so no extra reduction is needed (a global-Δt run needs one `min` reduction).

### B0b. Formulation for Unsteady
- Dual-Time Stepping for DDES.

### B1. Matrix-free operator (`wmles/implicit.py`)
- Start with **finite-difference J·v** `(R(Q+εv) − R(Q))/ε`, since float64 is on.
  - It works unchanged with both `Solver.rhs` and `DistributedSolver.rhs`, and with the existing halo and BC machinery.
  - It has no dependence on JAX AD through non-smooth parts.
- Later: an exact **`jax.jvp`** per block kernel, with frozen limiter, sensor and wall-model derivatives (`stop_gradient`) to avoid noise from non-smooth pieces. For the wall-model solve, add a `custom_jvp` using the implicit-function theorem rather than differentiating through `fori_loop`.
- API: `ImplicitIntegrator(solver, scheme="bdf1"|"bdf2", ...)` with `.advance(Q, dt, time)`, so the existing drivers don't change. `dt` is a scalar or a per-block tuple of per-cell arrays (local pseudo-time, B0a-LTS); the field is frozen within one Newton solve.

### B2. Krylov solver (`wmles/krylov.py`)
- Pytree GMRES(m), restarted, with vectors as tuples of per-block arrays.
- Use a Python-level loop for the distributed path, because the halo exchange is orchestrated in Python and can't live inside one `jit`. Dot products and norms reduce per-block partials on device 0, mirroring `_reduce` in `sharding.py`.
- Single-device path: optionally `jax.scipy.sparse.linalg.gmres` under `jit`.
- Inexact Newton with Eisenstat–Walker forcing, so each Newton iteration solves the linear system only as accurately as it needs.
- Memory is manageable: about 160 MB per Krylov vector on a 3.4M-cell, 6-variable case, so m = 20–30 fits when split across 8 devices.

### B3. Preconditioner (the real work)
Status: B3.1 implemented as `wmles/precond.py::PointJacobi` (opt-in `--precond jacobi`, right preconditioner). Measured on 2DZP (3 steps, GMRES iterations in the last step, unpreconditioned → Jacobi): global dt CFL 1: 14 → 20; CFL 10: 742 → 627; local dt CFL 1: 28 → 29; CFL 10: 492 → 614. As expected, with the integrator's `dt/scale` row scaling the Rusanov diagonal is `1 + CFLᵢ·const`, nearly uniform for local Δt, so point Jacobi cannot fix the wall-normal stiffness; B3.2 (line solve) is the real step. The `PointJacobi.rates` split (conv/visc/SA destruction) provides the diagonal for B3.2 and its off-diagonals follow from the same first-order flux.
Status B3.2: `wmles/precond.py::LineImplicit` (`--precond line|line-all`) is a block-tridiagonal (nv×nv, exact Euler flux Jacobian via `jacfwd`, Rusanov dissipation, scalar viscous term, SA destruction on the ν̃ diagonal) solve along the stiffest axis, or an approximate factorisation over several axes (`axis="all"` or a tuple). Blocks are conjugated by the integrator's variable scaling, and couplings across block faces are dropped (additive Schwarz, no halo exchange). A per-variable scalar tridiagonal was tried first and did not help: the stiff wall-normal modes are acoustic and couple p and u. 2DZP, 3 steps, GMRES iterations in the last step (unpreconditioned → line (wall-normal [+ streamwise for the tuple])): global dt CFL 10: 742 → 245 (216); CFL 100: 1328 → 346 (219). Local dt is **not** yet improved (CFL 10: 492 → 444; CFL 100: 1400 → 764–1200, results scatter), so the next step is to find out why before adding more machinery: candidate causes are the second-order/limited operator versus the first-order model in far-field cells that get very large Δtᵢ, and FD J·v noise (≈6e-4 relative at ε=1e-7). Spanwise slab axes of 2D cases must be excluded (`axis=(2, 1)`), since the auto detection of uniform axes is not reliable once round-off breaks the symmetry.
Diagnosis of the local-Δt stall (2DZP, state after 5 steps): (1) a global Δt equal to the local median is *harder* than local Δt at the same cells, so local Δt itself is not the problem; the preconditioner error `‖A M⁻¹v − v‖/‖v‖` grows with the effective CFL (smooth v: 0.08 at local CFL 1, 0.75 at CFL 10). (2) FD J·v error was round-off noise: the central difference with ε from 1e-2 to 1e-8 differs from ε=1e-3 by 11%, 1.5%, 5%, 18%, 28%, 29% (growing as ε shrinks), so the old default ε=1e-7 was pure noise. (3) Fix: default `fd_epsilon=1e-4`. Local Δt + `LineImplicit(axis=(2, 1))`, 3 steps, GMRES iterations in the last step: CFL 10: 420 → 135 (ε=1e-4) → 30 (ε=1e-3); CFL 100: 881 → 52 (ε=1e-4); ε=1e-3 at CFL 100 failed the Newton tolerance (truncation error), so 1e-4 is the default. (4) Steady pseudo-time steps need only `newton_rtol≈1e-2`: with 1e-6 and a larger ε the Newton loop could not converge and the CFL was halved repeatedly. 150-step ramped run (CFL 1 → 8, residual 1.6e10 → 1.1e9, identical history to the unpreconditioned run): one Newton iteration and 5–48 GMRES iterations per step versus 3–8 Newton and 40–210 GMRES before. Open: an adaptive ε or the exact `jvp` (B1) would remove the ε trade-off; CFL beyond ~10 not yet exercised in a full ramp.
Reduction abstraction (for sharding): all global quantities in `implicit.py` (dot products, norms, max, row scales) go through `solver._reduce` when present (`DistributedSolver` on `fix_xla`) and a local stack otherwise; per-variable scales are placed per block device; GMRES uses classical Gram-Schmidt with one reorthogonalisation (two reductions per Arnoldi step).
Build the preconditioner from a **first-order upwind Jacobian**, not the central or KEP scheme.
1. **Point-block Jacobi.** The diagonal block is `V/Δtᵢ + Σ|A_f|` (`Δtᵢ` is the per-cell local pseudo-time step in steady runs), obtained via `jax.jacfwd` of a Rusanov or HLLC-type flux, plus an analytic or `jacfwd` SA source Jacobian. It is local to each block, so there are no halo needs. Start here for correctness.
2. **Line-implicit block-tridiagonal solve** along the wall-normal axis, using `lax.scan` or an associative scan, with `vmap` over lines. This is what actually handles y+ ≈ 1 stretching.
   - Treat block interfaces as additive Schwarz (Jacobi coupling), which parallelises naturally over METIS blocks.
   - The wall-normal axis comes from each block's wall-BC face.
3. **Lagged preconditioner:** refresh every N Newton iterations.

### B4. Robustness
- Positivity-preserving Newton update: cap relative changes in ρ and p, and use SA-neg for ν̃.
- Pseudo-time CFL ramp (SER) with local Δtᵢ (B0a-LTS) and backtracking on `‖R‖`. On a failed line search, halve `CFLₚₛ`, recompute `Δtᵢ` from `Qⁿ` and retry the step.
- Constraints like `ConstantMassFlux` stay as post-step operations at first.
- Make the wall-model `utau` Newton safeguards differentiable-friendly.

### B5. Validation and performance
- Linear check: matrix-free J·v against a dense `jacfwd` on a tiny grid.
- Check that backward Euler with a tiny Δt reproduces explicit RK3.
- Local-Δt checks (B0a-LTS): a uniform per-cell Δt field equals the scalar-Δt step; the converged 2DZP C_f is independent of local versus global Δt and matches the explicit `step_local` result; report iterations to a 4-order residual drop for global versus local Δt.
- Check that a steady SA flat plate or NACA0012 converges by about 4 orders in residual.
- `DistributedSolver` equivalence: the implicit step on 1 device versus 8 virtual CPU devices (extend `tests/test_sharding.py`).
- Performance gate on L1: wall-time per physical time versus explicit, with the compile-time cost reported separately. Include GMRES iterations per Newton step and communication fraction on 8 GPUs.

---

## Suggested order
1. A0 and A1: wall distance, `Geom` fields, tests.
2. A2 and A3: SA-RANS on one device, then flat plate and NACA0012.
3. DDES (`d̃`, Δ_max, shielding), then channel DDES.
4. 6-variable sharding, restart and VTU, then multi-block and sharding regression.
5. B1 and B2: FD matrix-free GMRES with a point-block preconditioner on the explicit-equivalent test. Add local pseudo-time stepping (B0a-LTS) to the steady 2DZP driver before tuning the preconditioner.
6. B3 line-implicit preconditioner, B4 robustness, and steady SA convergence.
7. Dual-time DDES on OAT15a, plus the exact `jvp` variant if FD proves noisy or slow.

## Open questions that change the plan
- Should the SA/DDES OAT15a runs be **wall-resolved** (y+ ≈ 1, `NoSlipWall`) or **wall-modelled** on the current L1 grid? This decides whether A6 is needed. It should support first support y+=1, DDES with wall funcition can be implemented later.  
- Is SA mainly for steady RANS initialisation or for unsteady DDES? Both.
- Is the `(6, …)` state layout acceptable, or must the 5-variable layout stay untouched? The 5-variable layout stay untouched since this will be a WMLES / DDES solver.
- Should DDES numerics use the existing hybrid scheme, or should I add an `f_d`-blended central/upwind scheme from the start? Start with the existing scheme.

To implement any of this, switch to an agent that can edit files.

## Wall distance issue.
The current default, exact=True in wall_distance.py, is the expensive path. It evaluates 24 candidates × 2 triangles per cell through a long chain of np.where calls, and re-searches with 4× more candidates for uncertified points.

### Poisson wall distance (Tucker)
Solve ∇²φ = −1 with φ = 0 on walls and zero gradient elsewhere. Then:

d = −|∇φ| + √(|∇φ|² + 2φ)

This is exact for a planar wall. In 1D, φ = c·y − y²/2 gives |∇φ|² + 2φ = c², so d = y. For curved walls and corners the error is a few percent. It fits this code well:

- Same machinery: it is a finite-volume Laplacian built from geom.S and geom.vol.
- Interfaces: block interfaces, periodic spans and sharding reuse the existing halo exchange.
- Solver: it needs a CG/Krylov solver, which the JFNK work needs anyway.

Its weak points:

- Stretched grids. At y+≈1 the grid is very anisotropic, and plain Jacobi-preconditioned CG converges slowly (hundreds to thousands of iterations). It needs a wall-normal line-implicit preconditioner or multigrid. Each iteration is cheap on GPU, and the tolerance can be loose.
- Ghost cells. They need a mirrored, unsigned value at wall faces.
- Accuracy. The error is largest near the leading edge, the blunt trailing edge and concave corners.
- Reduction. A cross-device dot product is needed for the multi-GPU solve.

### What matters for SA/DDES
Exactness only matters in a thin band near the wall. That is where f_v2, f_w, r_d and the RANS side of the DDES switch live. Far from the wall, d_tilde is set by C_DES·Δ_max, so a few percent error in d is irrelevant. So I would not use one method for every cell.

### Proposed order
1. Band-limited exact search (no solver needed):
    - Compute a cheap approximate distance for all cells with cKDTree.query(k=1), which takes seconds for millions of points.
    - Use the nearest-centre value as an upper bound.
    - Refine exactly (small k, with certification) only for cells within a band of the wall, for example d_approx < 10 × reach or the first N wall-normal cells. That is about 10% of the cells.
    - Far cells keep the approximate value, whose error is bounded by reach.
2. Poisson/Tucker as a second backend, selectable via method="poisson". It is worth adding once the Krylov solver from the implicit work exists, and it is the only option that scales to a CRM-sized mesh on the GPUs. I would validate it against the exact search on the flat plate, the bump and the sphere, and report its error as a function of d/δ.
3. A Poisson-plus-exact-band hybrid if the Poisson accuracy near the wall proves insufficient.
Step 1 is probably enough for the OAT15a grid and removes the cost problem without a new solver. Poisson is a larger project that overlaps with the JFNK work. If you want me to implement step 1 first, switch to an agent that can edit files and run tests.

Step 1 is probably enough for the OAT15a grid and removes the cost problem without a new solver. Poisson is a larger project that overlaps with the JFNK work. Wrote this a reminder to implement later.

## Status (A2/A3, wall-resolved SA)
- [x] A0/A1: `Geom.delta_max` / `Geom.dist`, `BC.is_wall`, `wall_distance.py`, `turbulence.py` closure functions.
- [x] A2: rho*nu_tilde as 6th state row (only with `turbulence=`), limited upwind convection on the scheme's mass flux, compact-gradient diffusion (`viscous.scalar_diffusion_flux`), SA-neg source/diffusivity, nu_tilde ghost values in `NoSlipWall`/`Dirichlet`/`CharacteristicFarfield` (copy elsewhere).
- [x] A3 (reference `Solver` only; this branch has no `sharding.py`): SA coupling in `_block_rhs`, SA-aware dt, per-cell local time stepping (`local_dt`, `advance_local`, fused `step_local`), `init_state(..., nu_tilde_fn)`.
- [x] Tests: `tests/test_sa.py` (nu_tilde=0 equals the laminar solver, split/flip/rot_cyc blocks equal single block, operator exactness, SA-neg), `tests/test_rans_cases.py`.
- [x] 2DZP flat plate (`examples/flatplate_sa.py`, MUSCL-HLLC, 30k local-dt steps from the analytic profile): C_f/C_f,KS - 1 between -1.8% and 0% for Re_theta 4000-10500 (x <= 1.5). Not fully converged (interior C_f still drifts ~1% per 10k steps, rho residual down ~2 orders). The last ~25% of the plate (x > 1.6) is contaminated by the characteristic outflow/corner (C_f up to +90% at the last cell).
- [~] 2DANW airfoil near wake (`examples/airfoil_wake_sa.py`): runs and is attached (C_f > 0 everywhere, first-cell y+ <= 1.9), but NOT validated: C_d = 0.028 with C_d,pressure = 0.0186 is far too high and the wake recovers too quickly. Suspects: low-Mach dissipation of MUSCL-HLLC at M = 0.088 and the coarsest (L5) grid, incomplete convergence. The Hybrid scheme (central KEP4) diverges slowly for steady RANS on this grid (rho residual 3e2 -> 1.4e3), so a low-dissipation upwind/central blend is needed before DDES. To do: low-Mach-corrected HLLC or M = 0.2 test, finer 2DANW level, TMR SA-noft2 reference values.
- [ ] Outflow BC / corner treatment at the end of the plate; second finer plate grid for a convergence check.
- [ ] Band-limited wall distance (cost), then DDES (d_tilde, Delta_max, shielding), 6-row sharding, VTU output of nu_tilde, A5/A6, then Part B (JFNK). The JAX sharding part is currently addressed in another branch. Once it is ready, I will merge to main.
- Explicit pseudo-time is stiff: from a uniform start the plate boundary layer needs ~1e5 steps, so the plate case starts from an analytic turbulent profile (`initial_state(profile="flatplate")`). This is the motivation for the implicit solver.
