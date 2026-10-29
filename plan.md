## Plan: Eight-GPU DPW Buffet WMLES

Build on the now-working OAT15a case to deliver verified single-node, eight-GPU WMLES for OAT15a first and NASA CRM in the JAXA WT second. First make the OAT grid recipe reproducible, then integrate NASA Plot3D Utilities/METIS preprocessing, and only then add JAX multi-device execution. Do not treat mesh partitioning as solver sharding or the existing 10-step CPU smoke run as buffet-solution validation.

**Steps**

### Phase 1 — Freeze and validate the OAT15a baseline
1. [x] Reconcile the README recipe: it now writes and reads `out/oat15a_L1.p3d`, records the L1 generator options (including contour resolution, span, and blunt-TE divisions), and runs both commands in the `pyvista` conda environment.
2. [x] Add/retain case gates in `wmles/airfoil_case.py`, `wmles/airfoil_grid.py`, and `tests/test_airfoil_case.py`: expected block dimensions, positive interior volumes, valid interface coverage, periodic span, farfield and wall-model labels including blunt TE, finite initialization, and a short run. OAT15a blunt-TE small-grid assembly, positive volumes, periodic connectivity, BC coverage, and a finite timestep are tested; the focused airfoil/grid/Plot3D suite passes (9 tests).
3. [~] Preliminary smoke baseline captured in `out/oat15a_L1.log`: JAX CPU, Ma=0.73, Re_c=3e6, alpha=3.5 deg, T_inf=273 K, CFL=0.5; 3,362,400 cells across 12 assembled blocks, 40 spanwise cells. The run used the smoke-grid command `conda run -n pyvista python -m wmles.airfoil_grid out/oat15a_L1.p3d --airfoil tests/data/oat15a.dat --ni 257 --span-length 0.125 --spanwise-spacing 0.003125 --blunt-te-divisions 3 --wake-equalize 0.3`, then `conda run -n pyvista python -m examples.airfoil_wmles out/oat15a_L1.p3d --steps 10 --mach 0.73 --re 3e6 --temperature 273.0 --alpha 3.5 --cfl 0.5`. This is not the README's documented L1 grid recipe (`ni=513`, span=0.065, blunt-TE divisions=12, wake-equalize=0.8); use a distinct filename or regenerate the documented grid before calling it the L1 baseline. Smoke-run dt=1.067e-7 to 1.225e-7 s over 10 steps, t=1.1101e-6 s / 2.68e-4 ctu; rho_min stayed positive (0.205325 at step 1, 0.172506 at step 10); first-step time 33.076 s (includes compilation), steps 2-10 6.058-6.608 s (mean ~6.23 s); sampled RSS 1,643-2,051 MiB (not peak RSS). `out/a_airfoil_wmles/forces_moments.csv` records total step-10 loads about (0.25,0,0): Fx=249.746 N, Fy=-311.856 N, Mx=19.491 Nm, My=15.609 Nm, Mz=-100.499 Nm. These are transient smoke-run values, not buffet statistics. Still needed: baseline the documented L1 grid separately, distinguish sampled RSS from actual peak memory, and capture a longer warmed single-device benchmark. Establish physical-time and experimental-validation criteria separately.

### Phase 2 — NASA Plot3D Utilities and METIS preprocessing
4. [x] Turn `wmles/plot3d_util.py` from an exploratory hardcoded script into an importable/CLI preprocessing path: read the selected Plot3D file, discover and validate connectivity, preserve block/face orientation and BC labels, and emit a solver-consumable mesh plus a versioned partition manifest. Keep `wmles/plot3d_wm.py` as the project's independent reader/reference.
5. [x] Make the METIS dependency explicit and reproducible in `pyvista` and on AWS. Document the working macOS `METIS_DLL` invocation; on Linux configure the native library path appropriate to the host. Verify a real partition call, not only `import metis` or connectivity discovery.
6. [x] Split oversized OAT zones before METIS so there are enough work units for eight partitions. Preserve periodic faces and the OAT blunt-wall/farfield tags; enforce the solver's minimum per-axis extent (`ng=3`) and validate volumes and all subface links after conversion.
7. [x] Check the manifest: every solver block assigned exactly once, all eight partitions nonempty where feasible, estimated loads and inter-partition interface counts reported. Begin with cell-count vertex weights and interface communication weights; calibrate using measured per-block costs, including wall-model work.

### Phase 3 — JAX multi-GPU architecture and numerical equivalence
8. [~] Add explicit 8-device selection and placement in `wmles/backend.py`; current placement uses the first device. `WMLES_NUM_DEVICES` + `select_devices` (raises instead of falling back) and `python -m wmles.backend --require 8` are implemented and tested on CPU. Still needed: run the gate on the AWS g6e.48xlarge.
9. [x] Prototype block-state layout and ownership around `wmles/solver.py`. Resolve heterogeneous block shapes with a measured design (shape buckets/padding or block-local compiled kernels); do not assume all blocks can be stacked directly. Keep the existing single-device solver as the reference path. Decision: block-local compiled kernels with per-block state committed to the owner device (`DistributedSolver` in `wmles/sharding.py`); `shape_report` shows stacking/padding would add 120% cells on the 24-block OAT15a split, and wall-model/BC indices are block-static. Reference `Solver` numerics are unchanged (per-block hooks factored out).
10. [x] Implement per-device RHS/BC work and cross-device halo exchange using the existing `Interface` orientation and `exchange` contracts in `wmles/mesh.py`. Cover ghost layers plus edge/corner sweeps; add global minimum CFL reduction and global sum reductions for constraints/diagnostics as needed. Halo exchange replays the reference (pass, axis, block) order using `mesh.exchange_slab`/`insert_slab`, so edge/corner ghosts match; CFL, constraint sums and `conserved_totals` are device-0 reductions replicated without host sync. Measured on CPU with 8 virtual devices (24-block OAT15a split, perturbed state, 3 RK3 steps): RHS, dt and state are bit-identical to the reference. GPU timing/dispatch overhead is deferred to Phase 4.
11. [~] Add focused distributed tests in a new `tests/test_sharding.py`: partition/ownership coverage, halo exchange for same- and cross-device neighbors, free-stream preservation, global CFL, and agreement of RHS plus one SSP-RK3 step against the unsharded solver. Run CPU/JAX tests first, then 2-GPU and 8-GPU hardware gates. `tests/test_sharding.py` (16 tests, CPU virtual devices) covers ownership/manifest validation, same- and cross-device halos for split/flip/rot_cyc orientations, free stream, global CFL with the limiting block off device 0, constraint/totals, wall stats/loads, and RHS + RK3 vs reference on a 4-device wall-modelled airfoil. 2-GPU and 8-GPU gates are written but skipped until run on AWS.

### Phase 4 — Eight-GPU OAT15a and performance gates
12. Run the fixed OAT15a L1 case on one GPU and eight GPUs with identical mesh, options, precision, and initial state. Compare RHS/state tolerance, conservation, wall loads, stability, and device placement. Measure compile time, steady-state step time, memory, communication fraction, and partition imbalance.
13. Require no silent CPU fallback, no missing BCs/halos, finite physical state, and a demonstrated multi-GPU speedup before calling the eight-GPU path usable. Optimize partition weights/halo schedule only against these measurements.
14. Establish buffet validation requirements beyond the smoke run: sufficient flow-through/physical sampling time, force and pressure statistics, restart/output strategy, and comparison quantities/data for the DPW OAT15a case.

### Phase 5 — NASA CRM / JAXA WT
15. Obtain and freeze the CRM Plot3D grid, connectivity, periodicity, boundary labels, tunnel/test conditions, and requested CRM configuration. No CRM grid or case setup is currently present in the workspace.
16. Add a dedicated CRM case module and driver rather than reusing OAT15a-specific surface assumptions. Reuse the proven preprocessing, METIS manifest, eight-GPU solver, and validation gates; verify force/moment reference axes and wind-tunnel boundaries before long runs.

**Relevant files**
- `README.md` — correct the `.vtm`/`.p3d` OAT recipe; document pyvista-env and METIS/AWS invocation, smoke vs production validation, and results.
- `wmles/airfoil_grid.py`, `wmles/airfoil_case.py`, `examples/airfoil_wmles.py` — current OAT mesh/case path and CLI.
- `wmles/plot3d_util.py`, `wmles/plot3d_wm.py`, `wmles/nmf.py` — NASA adapter, reference I/O, split/connectivity/BC translation.
- `wmles/backend.py`, `wmles/mesh.py`, `wmles/solver.py` — device placement, interfaces/halos/metrics, per-block RHS/RK/CFL.
- `tests/test_airfoil_grid.py`, `tests/test_airfoil_case.py`, `tests/test_plot3d_nmf.py`, new `tests/test_sharding.py` — baseline, preprocessing, and distributed behavior.
- New CRM case module/driver and CRM test data — add only after grid and case requirements are supplied.

**Verification**
1. Run focused tests in the `pyvista` environment: `conda run -n pyvista python -m unittest tests.test_airfoil_grid tests.test_airfoil_case tests.test_plot3d_nmf`.
2. Rebuild and assemble the exact documented OAT15a `.p3d` file; run a short CPU/JAX smoke test and verify the full mesh report.
3. On AWS, run `conda run -n pyvista env WMLES_DEVICE=gpu python -c 'import jax; print(jax.devices())'`; require eight devices before multi-GPU gates.
4. Compare one-GPU/eight-GPU RHS and one-step outputs, then run a fixed timed benchmark and longer OAT buffet validation. Repeat equivalent gates for CRM once its grid/setup are available.

**Decisions**
- Target is one g6e-48xlarge host with eight GPUs; distributed multi-node MPI is outside this plan.
- NASA Plot3D Utilities/METIS is preprocessing and partition metadata; JAX must still implement runtime device placement, halo transfers, and global reductions.
- Keep `pyvista` as the conda environment for commands and validation.
- OAT15a is the first actual DPW case; the generated A-airfoil test case is not a substitute. NASA CRM follows OAT15a.
- Do not claim buffet prediction validity from mesh assembly or a 10-step smoke run.
