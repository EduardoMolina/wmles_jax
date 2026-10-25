I’ll check the repository for any existing METIS or NASA utility hooks, then revise the order so mesh preprocessing and partitioning are a prerequisite to solver sharding.

Searched for regex `metis|pymetis|parmetis|plot3d_util|partition|shard_map`, 7 results

There’s no source-level METIS or NASA-utility integration in the repo; the search only found output artifacts and the existing note about future sharding. I’ll verify the utility’s intended role before treating “merge blocks” and METIS partitioning as one operation, since they affect different parts of the mesh pipeline.



Searched nasa/plot3d_utilities for "Find README and source support for merging multiblock Plot3D grids and METIS-based graph partitioning/workload balancing. Report exact tools/commands, input/output formats, and whether METIS partitions blocks or cells.", 64 results

Agreed. **NASA Plot3D Utilities and its METIS-backed partitioning should come before the JAX sharding implementation.** One important detail from the utility’s API: METIS partitions a graph of **blocks**. The current `a_airfoil.p3d` starts as four blocks, so to use eight GPUs effectively, we’ll need to split those blocks into more solver-sized pieces first. Merging is available too, but it may reduce parallelism; I’d treat merge as an optional preprocessing operation, not the first step.

The NASA repository documents `partition_from_face_matches(...)`, which uses its `metis-rs` integration and block-size weights. Its Plot3D tools also include block splitting and merging. That gives us a better starting point than building our own partitioner or assuming a plain `shard_map` wrapper will balance the workload.

## Revised sequence

1. **Add an optional NASA utilities adapter**
   - New module: `wmles/plot3d_utilities.py` (or similar).
   - Keep `plot3d_wm.py` as the project’s existing reader/writer. Normalize NASA utility `Block` objects and connectivity metadata into the repo’s node arrays and mesh conventions.
   - Pin or document the NASA utility and METIS dependency in the project’s setup instructions. Keep the solver usable when the optional dependency is absent.
   - First check: read `a_airfoil.p3d` with both paths and compare block shapes and coordinates; discover and verify the same interfaces.

2. **Prepare enough zones for eight-way partitioning**
   - Use the NASA splitting utilities on the four input blocks before partitioning.
   - Preserve interface matching, span periodicity, and the airfoil wall / far-field labels.
   - Ensure resulting pieces meet this solver’s minimum cell extent for its ghost layers (`ng=3`, see `mesh.py`).
   - Don’t merge by default: splitting creates partitionable work; merging is useful only when preprocessing shows a reason to reduce or reshape the zone layout.

3. **Run METIS and save a partition manifest**
   - Use NASA connectivity and `partition_from_face_matches` for the split mesh.
   - Start with cell-count vertex weights and connectivity-derived edge weights. Later, adjust costs using measured block timings and wall-model work.
   - Store the mapping from each prepared zone to a partition, plus its source block, BC labels, neighbors, and estimated load. METIS’s assignment is preprocessing output, not yet a JAX device layout.
   - Check: all zones assigned once, eight partitions populated when there are enough zones, and report load imbalance and cut interfaces.

4. **Convert the prepared mesh into the repo’s mesh model**
   - Adapt the split blocks and connectivity into `Block`, `Mesh.connect`, and BC setup in `airfoil_case.py` and `nmf.py`.
   - Keep existing `Mesh.connect` orientation checks as a validation gate; NASA-discovered connectivity must not bypass them.
   - Compare total cell volume and verify boundary labels, periodic faces, and interface coverage.

5. **Validate preprocessing before touching the solver**
   - Add tests for the adapter, splits, BC preservation, and METIS manifest in `test_airfoil_case.py` or a focused new test module.
   - On `a_airfoil.p3d`, verify constant free-stream preservation and finite initialization using the resulting split mesh.
   - Keep the unsplit case as a reference.

6. **Implement JAX multi-GPU execution**
   - Only after the split-mesh and manifest path is stable, use the partition mapping in a new sharding layer around `solver.py`.
   - Update `backend.py` for explicit multi-device placement; currently `asarray` targets the first selected device.
   - Handle unequal block shapes and cross-device halo exchange explicitly. METIS assigns blocks to partitions; it does not perform runtime JAX array sharding or halo communication.
   - Check the sharded RHS and one-step result against the existing single-device path before performance work.

7. **Then progress to DPW cases**
   - First use `a_airfoil.p3d` to exercise the full preprocessing-to-eight-GPU pipeline.
   - Next add the distinct OAT15a grid and case configuration, then the CRM grid and case. The generated A-airfoil case is a pipeline test, not a substitute for either DPW test case.

For any commands I run for these steps, I’ll use the `pyvista` conda environment.