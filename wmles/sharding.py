"""Multi-device block ownership and the distributed explicit solver.

Design (single node, one Python controller, N JAX devices):

* Ownership.  Every block lives on exactly one device (`Ownership`).  Blocks are
  heterogeneous, so they are *not* stacked or padded into a sharded array; each block
  keeps its own shape and its kernels are compiled for that shape and committed to the
  owner device (`shape_report` quantifies what padding into one shape would cost).
* Halo exchange.  `Solver.apply_bcs` visits (pass, axis, block) in a fixed order and
  every interface copy reads the *current* state of the neighbour.  `DistributedSolver`
  dispatches exactly the same sequence: the neighbour's ghost-layer slab is extracted
  on its device (`mesh.exchange_slab`, which applies the interface orientation), moved
  with `device_put`, and inserted on the owner device in a per-block jitted sweep that
  also applies the physical BCs.  Same order => same edge/corner ghost values as the
  single-device reference.  JAX dispatch is asynchronous, so devices overlap wherever
  the data dependencies allow; nothing here blocks the host.
* Reductions.  The CFL time step and the constraint sums are reduced on device 0 and
  the result is replicated back to every device, still without host synchronisation.

The single-device `Solver` is the reference; this module never changes its numerics.
"""
import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np

from . import backend
from .backend import xp, jit, device_put, select_devices
from .bc import WallModel
from .mesh import Interface, exchange, exchange_slab, insert_slab
from .solver import Solver

MANIFEST_VERSION = 1
_BYTES_PER_VALUE = 8
_NVAR = 5


# --------------------------------------------------------------------------- ownership
class Ownership:
    """Block index -> device slot (0..n_devices-1)."""

    def __init__(self, owner, n_devices):
        owner = tuple(int(o) for o in owner)
        if n_devices < 1:
            raise ValueError("n_devices must be positive")
        bad = [(i, o) for i, o in enumerate(owner) if not 0 <= o < n_devices]
        if bad:
            raise ValueError(f"block owners outside 0..{n_devices - 1}: {bad}")
        self.owner, self.n_devices = owner, int(n_devices)

    def __len__(self):
        return len(self.owner)

    def blocks_of(self, device):
        return tuple(i for i, o in enumerate(self.owner) if o == device)

    @classmethod
    def balanced(cls, mesh, n_devices):
        """Greedy largest-first assignment by cell count (no communication weighting)."""
        cells = [int(np.prod(b.n)) for b in mesh.blocks]
        load = [0] * n_devices
        owner = [0] * len(cells)
        for i in sorted(range(len(cells)), key=lambda j: (-cells[j], j)):
            dev = min(range(n_devices), key=lambda d: (load[d], d))
            owner[i] = dev
            load[dev] += cells[i]
        return cls(owner, n_devices)

    @classmethod
    def from_manifest(cls, manifest, mesh):
        """Read a `plot3d_util` partition manifest (path, dict or PartitionManifest) and
        check that it describes exactly the blocks of `mesh`."""
        if isinstance(manifest, (str, Path)):
            with open(manifest) as stream:
                manifest = json.load(stream)
        elif hasattr(manifest, "as_dict"):
            manifest = manifest.as_dict()
        if manifest.get("version") != MANIFEST_VERSION:
            raise ValueError(f"unsupported manifest version {manifest.get('version')!r}")
        entries = sorted(manifest["blocks"], key=lambda e: e["index"])
        if [e["index"] for e in entries] != list(range(len(entries))):
            raise ValueError("manifest block indices must be 0..N-1, each exactly once")
        if len(entries) != len(mesh.blocks):
            raise ValueError(f"manifest has {len(entries)} blocks, mesh has {len(mesh.blocks)}")
        for entry, blk in zip(entries, mesh.blocks):
            if entry["name"] != blk.name or tuple(entry["shape"]) != tuple(blk.n):
                raise ValueError(
                    f"manifest block {entry['index']} ({entry['name']}, {entry['shape']}) does "
                    f"not match mesh block {blk.name} {list(blk.n)}; re-run the preprocessing"
                )
        return cls([e["partition"] for e in entries], manifest["partitions"])

    def report(self, mesh):
        """Per-device cell loads, imbalance and cross-device halo traffic."""
        cells = [int(np.prod(b.n)) for b in mesh.blocks]
        loads = [0] * self.n_devices
        for c, o in zip(cells, self.owner):
            loads[o] += c
        interfaces = cross = halo_values = 0
        seen = set()
        for ib, blk in enumerate(mesh.blocks):
            for (axis, side), f in blk.faces.items():
                if not isinstance(f, Interface) or f.nb == ib:
                    continue
                t = [blk.n[a] + 2 * blk.ng for a in range(3) if a != axis]
                if self.owner[ib] != self.owner[f.nb]:
                    halo_values += blk.ng * t[0] * t[1] * _NVAR
                key = frozenset(((ib, axis, side), (f.nb,) + tuple(f.src_face)))
                if key in seen:
                    continue
                seen.add(key)
                interfaces += 1
                cross += self.owner[ib] != self.owner[f.nb]
        mean = sum(loads) / self.n_devices
        return {
            "blocks": len(cells),
            "device_cells": loads,
            "empty_devices": [d for d, c in enumerate(loads) if c == 0],
            "imbalance": max(loads) / mean,
            "interfaces": interfaces,
            "cross_device_interfaces": cross,
            "halo_bytes_per_exchange": halo_values * _BYTES_PER_VALUE,
        }


def shape_report(mesh):
    """Block-shape statistics behind the block-local-kernel decision: how many distinct
    kernels are compiled and how many cells padding every block to the per-axis maximum
    (the cost of stacking them into one array) would add."""
    shapes = [tuple(b.n) for b in mesh.blocks]
    cells = sum(int(np.prod(s)) for s in shapes)
    pad = tuple(max(s[a] for s in shapes) for a in range(3))
    return {
        "blocks": len(shapes),
        "distinct_shapes": len(set(shapes)),
        "cells": cells,
        "padded_shape": pad,
        "padding_overhead": len(shapes) * int(np.prod(pad)) / cells - 1.0,
    }


# --------------------------------------------------------------------------- solver
class DistributedSolver(Solver):
    """`Solver` with block ownership on several devices.

    State and geometry of block i live on `devices[ownership.owner[i]]`.  Use
    `place_state` to move a reference state in and `gather_state` to bring it back.
    Public methods keep the `Solver` signatures.
    """

    def __init__(self, mesh, gas, scheme, *, ownership=None, devices=None,
                 release_mesh_geometry=False, **kwargs):
        if not backend.HAS_JAX:
            raise RuntimeError("DistributedSolver requires the JAX backend")
        super().__init__(mesh, gas, scheme, **kwargs)
        if devices is None:
            devices = select_devices(ownership.n_devices if ownership else None)
        self.devices = tuple(devices)
        self.ownership = ownership or Ownership.balanced(mesh, len(self.devices))
        if self.ownership.n_devices != len(self.devices):
            raise ValueError(f"ownership has {self.ownership.n_devices} partitions for "
                             f"{len(self.devices)} devices")
        if len(self.ownership) != len(self.blocks):
            raise ValueError("ownership must assign every block")
        for c in self.constraints:
            if not all(hasattr(c, n) for n in ("partial_sums", "correction", "shift")):
                raise NotImplementedError(
                    f"constraint {type(c).__name__} lacks partial_sums/correction/shift "
                    f"and cannot be reduced across devices")
        self.block_device = tuple(self.devices[o] for o in self.ownership.owner)
        self.geoms = tuple(device_put(g, d) for g, d in zip(mesh.geoms, self.block_device))
        if release_mesh_geometry:
            mesh.geoms = self.geoms
            for blk, g in zip(self.blocks, self.geoms):
                blk.geom = g
        self._build_kernels()

    @classmethod
    def from_solver(cls, solver, ownership=None, devices=None, **kwargs):
        """Distributed twin of a reference `Solver` (same mesh, gas, scheme and options)."""
        return cls(solver.mesh, solver.gas, solver.scheme, ownership=ownership,
                   devices=devices, **solver.config, **kwargs)

    # ------------------------------------------------------------------ compiled kernels
    def _build_kernels(self):
        n = len(self.blocks)
        self._plans = [[self._side_plan(ib, a) for a in range(3)] for ib in range(n)]
        self._sweeps = [[self._make_sweep(ib, a) for a in range(3)] for ib in range(n)]
        self._extractors, self._extractor_cache = {}, {}
        for ib, blk in enumerate(self.blocks):
            for a in range(3):
                for kind, s, f in self._plans[ib][a]:
                    if kind == "remote":
                        self._extractors[(ib, a, s)] = self._extractor(blk, (a, s), f)
        self._rhs_blk = [jit(partial(self._block_rhs_padded, ib)) for ib in range(n)]
        self._dt_blk = [jit(partial(self._block_dt, ib)) for ib in range(n)]
        self._totals_blk = [jit(partial(self._block_totals, ib)) for ib in range(n)]
        self._stats_blk = [
            jit(partial(self._block_wall_stats, ib))
            if any(isinstance(bc, WallModel) for bc in blk.faces.values()) else None
            for ib, blk in enumerate(self.blocks)
        ]
        self._rk1_blk = jit(lambda q, r, dt: q + dt * r)
        self._rk2_blk = jit(lambda q0, q1, r, dt: 0.75 * q0 + 0.25 * (q1 + dt * r))
        self._rk3_blk = jit(lambda q0, q2, r, dt: (1.0 / 3.0) * q0 + (2.0 / 3.0) * (q2 + dt * r))
        self._partial_blk = [[jit(partial(self._constraint_partial, c, ib)) for ib in range(n)]
                             for c in self.constraints]
        self._shift_blk = [jit(c.shift) for c in self.constraints]

    def _side_plan(self, ib, axis):
        plan = []
        for s in (0, 1):
            f = self.blocks[ib].faces[(axis, s)]
            if isinstance(f, Interface):
                plan.append(("self" if f.nb == ib else "remote", s, f))
            else:
                plan.append(("bc", s, f))
        return plan

    def _make_sweep(self, ib, axis):
        blk, gas, plan = self.blocks[ib], self.gas, self._plans[ib][axis]

        def sweep(q, geom, slabs, time):
            received = iter(slabs)
            for kind, s, f in plan:
                if kind == "remote":
                    q = insert_slab(q, next(received), blk, (axis, s), "cell")
                elif kind == "self":
                    q = exchange(q, q, blk, (axis, s), blk, f, "cell")
                else:
                    q = f.fill_at_time(q, blk, geom, gas, axis, s, time)
            return q

        return jit(sweep)

    def _extractor(self, blk, face, itf):
        src = self.blocks[itf.nb]
        key = (face, tuple(itf.src_face), src.n[itf.src_face[0]], blk.ng,
               tuple(itf.mapping), tuple(itf.flip))
        cache = self._extractor_cache
        if key not in cache:
            cache[key] = jit(lambda q: exchange_slab(q, blk, face, src, itf, "cell"))
        return cache[key]

    def _block_totals(self, ib, q, g):
        blk = self.blocks[ib]
        return xp.sum(blk.interior(q) * blk.interior(g.vol), axis=(1, 2, 3))

    def _constraint_partial(self, constraint, ib, q, vol):
        return constraint.partial_sums(q, vol, self.blocks[ib])

    # ------------------------------------------------------------------ placement / reductions
    def place_state(self, Q):
        """Commit a state (e.g. from the reference solver) to the owner devices."""
        return tuple(device_put(q, d) for q, d in zip(Q, self.block_device))

    @staticmethod
    def gather_state(Q):
        """Host copy of a distributed state."""
        return tuple(np.asarray(q) for q in Q)

    def init_state(self, prim_fn):
        return tuple(
            device_put(self.gas.prim_to_cons(xp.asarray(prim_fn(g.xc))), d)
            for g, d in zip(self.geoms, self.block_device)
        )

    def _replicate(self, x):
        """One copy of scalar `x` per device (a list is taken as already replicated)."""
        if isinstance(x, list):
            return x
        return [device_put(x, d) for d in self.devices]

    def _reduce(self, values, op):
        """Combine per-block device scalars/arrays on device 0 (no host round trip)."""
        return op(xp.stack([device_put(v, self.devices[0]) for v in values]), axis=0)

    # ------------------------------------------------------------------ halo exchange / BCs
    def apply_bcs(self, Q, geoms, time=0.0):
        times = self._replicate(time)
        Q = list(Q)
        owner = self.ownership.owner
        for _ in range(self.bc_passes):
            for a in range(3):
                for ib in range(len(self.blocks)):
                    slabs = tuple(
                        device_put(self._extractors[(ib, a, s)](Q[f.nb]), self.block_device[ib])
                        for kind, s, f in self._plans[ib][a] if kind == "remote"
                    )
                    Q[ib] = self._sweeps[ib][a](Q[ib], geoms[ib], slabs, times[owner[ib]])
        return tuple(Q)

    # ------------------------------------------------------------------ Solver API
    def _rhs_from_filled(self, Qb):
        return tuple(fn(q, g) for fn, q, g in zip(self._rhs_blk, Qb, self.geoms))

    def rhs(self, Q, time=0.0):
        return self._rhs_from_filled(self.apply_bcs(Q, self.geoms, self._replicate(time)))

    def max_dt(self, Q, time=0.0):
        Qb = self.apply_bcs(Q, self.geoms, self._replicate(time))
        return self._reduce([fn(q, g) for fn, q, g in zip(self._dt_blk, Qb, self.geoms)], xp.min)

    def advance(self, Q, dt, time=0.0):
        owner = self.ownership.owner
        dts, t0 = self._replicate(dt), self._replicate(time)
        t1 = [t + d for t, d in zip(t0, dts)]
        t2 = [t + 0.5 * d for t, d in zip(t0, dts)]
        R0 = self.rhs(Q, t0)
        Q1 = tuple(self._rk1_blk(q, r, dts[owner[i]]) for i, (q, r) in enumerate(zip(Q, R0)))
        R1 = self.rhs(Q1, t1)
        Q2 = tuple(self._rk2_blk(q, q1, r, dts[owner[i]])
                   for i, (q, q1, r) in enumerate(zip(Q, Q1, R1)))
        R2 = self.rhs(Q2, t2)
        Q3 = tuple(self._rk3_blk(q, q2, r, dts[owner[i]])
                   for i, (q, q2, r) in enumerate(zip(Q, Q2, R2)))
        for ic, constraint in enumerate(self.constraints):
            parts = [self._partial_blk[ic][ib](q, g.vol)
                     for ib, (q, g) in enumerate(zip(Q3, self.geoms))]
            num = self._reduce([p[0] for p in parts], xp.sum)
            den = self._reduce([p[1] for p in parts], xp.sum)
            delta = self._replicate(constraint.correction(num, den))
            Q3 = tuple(self._shift_blk[ic](q, delta[owner[i]]) for i, q in enumerate(Q3))
        return Q3

    # ------------------------------------------------------------------ diagnostics
    def wall_stats(self, Q):
        Qb = self.apply_bcs(Q, self.geoms)
        out = {}
        for fn, q, g in zip(self._stats_blk, Qb, self.geoms):
            if fn is not None:
                out.update({k: (np.asarray(t), np.asarray(h)) for k, (t, h) in fn(q, g).items()})
        return out

    def conserved_totals(self, Q):
        parts = [fn(q, g) for fn, q, g in zip(self._totals_blk, Q, self.geoms)]
        return np.asarray(self._reduce(parts, xp.sum))

    def describe(self):
        """Device placement summary, e.g. for run logs."""
        rep = self.ownership.report(self.mesh)
        lines = [f"{len(self.blocks)} blocks on {len(self.devices)} devices "
                 f"(imbalance {rep['imbalance']:.3f}, {rep['cross_device_interfaces']}/"
                 f"{rep['interfaces']} interfaces cross devices, "
                 f"{rep['halo_bytes_per_exchange'] / 1e6:.2f} MB halo per exchange)"]
        for d, dev in enumerate(self.devices):
            owned = self.ownership.blocks_of(d)
            lines.append(f"  {dev}: {len(owned)} blocks, {rep['device_cells'][d]} cells")
        return "\n".join(lines)


# --------------------------------------------------------------------------- equivalence check
def _max_rel(a, b):
    scale = max(float(np.max(np.abs(b))), 1e-300)
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) / scale


def compare_with_reference(reference, distributed, Q0, steps=1):
    """Run `steps` SSP-RK3 steps with both solvers from the same state.  Returns the
    worst per-block relative differences of RHS, dt and final state, plus device checks."""
    Qd = distributed.place_state(Q0)
    rhs_ref, rhs_dist = reference.rhs(Q0), distributed.rhs(Qd)
    out = {"rhs": max(_max_rel(a, b) for a, b in zip(rhs_dist, rhs_ref))}
    Qr, t = Q0, 0.0
    dt_err = 0.0
    for _ in range(steps):
        dt_r, dt_d = reference.max_dt(Qr, t), distributed.max_dt(Qd, t)
        dt_err = max(dt_err, abs(float(dt_d) - float(dt_r)) / float(dt_r))
        Qr, Qd = reference.advance(Qr, dt_r, t), distributed.advance(Qd, dt_r, t)
        t += float(dt_r)
    out["dt"] = dt_err
    out["state"] = max(_max_rel(a, b) for a, b in zip(Qd, Qr))
    out["finite"] = all(bool(np.all(np.isfinite(np.asarray(q)))) for q in Qd)
    out["placement_ok"] = all(q.devices() == {dev} for q, dev in zip(Qd, distributed.block_device))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compare the multi-device solver with the single-device reference on an "
                    "OAT15a Plot3D grid (split into blocks as in the METIS preprocessing).")
    parser.add_argument("grid", type=Path)
    parser.add_argument("--devices", type=int, default=None,
                        help="device count (default WMLES_NUM_DEVICES)")
    parser.add_argument("--manifest", type=Path, default=None,
                        help="partition manifest; default is a cell-balanced assignment")
    parser.add_argument("--steps", type=int, default=1)
    args = parser.parse_args(argv)

    from .airfoil_case import build_case, initial_state
    devices = select_devices(args.devices)
    mesh, reference, props = build_case(args.grid, partitions=len(devices))
    ownership = (Ownership.from_manifest(args.manifest, mesh) if args.manifest
                 else Ownership.balanced(mesh, len(devices)))
    distributed = DistributedSolver.from_solver(reference, ownership, devices)
    print(distributed.describe())
    print("shapes:", shape_report(mesh))
    print("compare:", compare_with_reference(
        reference, distributed, initial_state(reference, props), args.steps))


if __name__ == "__main__":
    main()
