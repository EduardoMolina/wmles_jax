"""Multi-device solver tests.  Run:  python -m unittest tests.test_sharding

CPU runs use virtual devices (see tests/__init__.py).  The hardware gates at the end
run only on hosts that expose enough GPUs, e.g. on the AWS g6e.48xlarge:
    WMLES_DEVICE=gpu python -m unittest tests.test_sharding
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

import tests  # noqa: F401  (sets XLA_FLAGS before JAX starts)
from wmles import ConstantMassFlux, Solver, KEP
from wmles.airfoil_case import build_case, initial_state
from wmles.airfoil_grid import generate_four_block_mesh, read_airfoil_coordinates, write_plot3d
from wmles.backend import HAS_JAX, select_devices, xp, requested_device_count
from wmles.sharding import (DistributedSolver, Ownership, compare_with_reference,
                            shape_report)
from tests.test_multiblock import build, prim_fn

from wmles import backend

TOL = 1e-11
UNIFORM = lambda xc: xp.stack([1.2 + 0 * xc[0], 50 + 0 * xc[0], -20 + 0 * xc[0],
                               30 + 0 * xc[0], 1e5 + 0 * xc[0]])
ndev = len(backend._DEVICES)
needs_two = unittest.skipUnless(ndev >= 2, "needs at least 2 devices")
needs_four = unittest.skipUnless(ndev >= 4, "needs at least 4 devices")


def small_airfoil(partitions):
    blocks = generate_four_block_mesh(
        ni=65, airfoil_divisions=5, te_divisions=4, tunnel_height_divisions=12,
        wake_divisions=16,
        airfoil_coordinates=read_airfoil_coordinates("tests/data/oat15a.dat"),
    )
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "oat15a.p3d"
        write_plot3d(path, list(blocks.values()), span_length=0.2, spanwise_cells=8)
        return build_case(path, partitions=partitions)


def twin(reference, owner, devices):
    return DistributedSolver.from_solver(
        reference, Ownership(owner, len(devices)), devices=devices)


class OwnershipTests(unittest.TestCase):
    def setUp(self):
        self.mesh, self.solver = build("rot_cyc")

    def test_balanced_covers_every_block_once(self):
        mesh, _, _ = small_airfoil(4)
        for count in (1, 3, 4):
            own = Ownership.balanced(mesh, count)
            self.assertEqual(len(own), len(mesh.blocks))
            self.assertEqual(sorted(sum((own.blocks_of(d) for d in range(count)), ())),
                             list(range(len(mesh.blocks))))
            rep = own.report(mesh)
            self.assertEqual(sum(rep["device_cells"]), sum(int(np.prod(b.n)) for b in mesh.blocks))
            self.assertLess(rep["imbalance"], 1.5)

    def test_report_counts_cross_device_interfaces(self):
        same = Ownership([0, 0], 2).report(self.mesh)
        split = Ownership([0, 1], 2).report(self.mesh)
        self.assertEqual(same["cross_device_interfaces"], 0)
        self.assertEqual(same["halo_bytes_per_exchange"], 0)
        self.assertEqual(same["empty_devices"], [1])
        self.assertEqual(split["cross_device_interfaces"], 2)
        self.assertGreater(split["halo_bytes_per_exchange"], 0)

    def test_manifest_roundtrip_and_mismatch(self):
        blocks = [{"index": i, "name": b.name, "shape": list(b.n), "partition": i % 2}
                  for i, b in enumerate(self.mesh.blocks)]
        manifest = {"version": 1, "partitions": 2, "blocks": blocks}
        self.assertEqual(Ownership.from_manifest(manifest, self.mesh).owner, (0, 1))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "m.json"
            path.write_text(json.dumps(manifest))
            self.assertEqual(Ownership.from_manifest(path, self.mesh).owner, (0, 1))
        for broken in (
            {**manifest, "version": 2},
            {**manifest, "blocks": blocks[:1]},
            {**manifest, "blocks": [dict(blocks[0], shape=[1, 1, 1])] + blocks[1:]},
            {**manifest, "blocks": [dict(blocks[0], name="other")] + blocks[1:]},
        ):
            with self.assertRaises(ValueError):
                Ownership.from_manifest(broken, self.mesh)
        with self.assertRaises(ValueError):
            Ownership([0, 2], 2)

    def test_shape_report(self):
        rep = shape_report(self.mesh)
        self.assertEqual(rep["blocks"], 2)
        self.assertGreaterEqual(rep["padding_overhead"], 0.0)

    def test_device_selection_never_falls_back(self):
        self.assertEqual(len(select_devices(1)), 1)
        with self.assertRaises(RuntimeError):
            select_devices(ndev + 1)
        old = os.environ.get("WMLES_NUM_DEVICES")
        try:
            os.environ["WMLES_NUM_DEVICES"] = "0"
            with self.assertRaises(ValueError):
                requested_device_count()
            os.environ["WMLES_NUM_DEVICES"] = "2"
            self.assertEqual(requested_device_count(), 2)
        finally:
            if old is None:
                os.environ.pop("WMLES_NUM_DEVICES", None)
            else:
                os.environ["WMLES_NUM_DEVICES"] = old


@unittest.skipUnless(HAS_JAX, "requires JAX")
@needs_two
class HaloAndEquivalenceTests(unittest.TestCase):
    devices = None

    @classmethod
    def setUpClass(cls):
        cls.devices = select_devices(2)

    def test_halo_exchange_same_and_cross_device(self):
        # Both orientations that permute and flip index axes, plus periodic wrap-around.
        for kind in ("split", "flip", "rot_cyc"):
            mesh, ref = build(kind, wm=True)
            Q = ref.init_state(prim_fn)
            expected = ref.apply_bcs(Q, ref.geoms)
            for owner in ([0, 0], [0, 1], [1, 0]):
                dist = twin(ref, owner, self.devices)
                got = dist.apply_bcs(dist.place_state(Q), dist.geoms)
                for g, e in zip(got, expected):
                    np.testing.assert_allclose(np.asarray(g), np.asarray(e), rtol=1e-13,
                                               atol=0, err_msg=f"{kind} {owner}")

    def test_rhs_and_rk3_step_match_reference(self):
        for kind, wm in (("split", False), ("flip", True), ("rot_cyc", True)):
            mesh, ref = build(kind, wm=wm)
            dist = twin(ref, [0, 1], self.devices)
            result = compare_with_reference(ref, dist, ref.init_state(prim_fn), steps=2)
            self.assertLess(result["rhs"], TOL, (kind, result))
            self.assertLess(result["dt"], TOL, (kind, result))
            self.assertLess(result["state"], TOL, (kind, result))
            self.assertTrue(result["finite"] and result["placement_ok"])

    def test_single_self_periodic_block_with_idle_device(self):
        mesh, ref = build("single", wm=True)
        dist = twin(ref, [1], self.devices)
        result = compare_with_reference(ref, dist, ref.init_state(prim_fn))
        self.assertLess(result["rhs"], TOL)
        self.assertLess(result["state"], TOL)

    def test_free_stream_preserved(self):
        mesh, ref = build("rot_cyc", periodic_y=True)
        dist = twin(ref, [0, 1], self.devices)
        Q = dist.init_state(UNIFORM)
        for r in dist.rhs(Q):
            self.assertLess(float(np.abs(np.asarray(r)).max()), 1e-6 * 1e5)
        Q1 = dist.advance(Q, dist.max_dt(Q))
        for q0, q1 in zip(Q, Q1):
            np.testing.assert_allclose(np.asarray(q1), np.asarray(q0), rtol=1e-12)

    def test_time_dependent_arguments_follow_the_reference(self):
        mesh, ref = build("split", wm=True)
        dist = twin(ref, [0, 1], self.devices)
        Q = ref.init_state(prim_fn)
        dt = ref.max_dt(Q, 0.5)
        got = dist.advance(dist.place_state(Q), dt, 0.5)
        want = ref.advance(Q, dt, 0.5)
        for g, w in zip(got, want):
            np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=TOL, atol=0)

    def test_constraint_and_totals_are_global(self):
        mesh, ref0 = build("split")
        constraint = ConstantMassFlux(0, 1.2 * 40.0)
        ref = Solver(mesh, ref0.gas, KEP(4), cfl=0.5, constraints=[constraint])
        dist = twin(ref, [0, 1], self.devices)
        Q = ref.init_state(prim_fn)
        dt = ref.max_dt(Q)
        want, got = ref.advance(Q, dt), dist.advance(dist.place_state(Q), dt)
        for g, w in zip(got, want):
            np.testing.assert_allclose(np.asarray(g), np.asarray(w), rtol=TOL, atol=0)
        # The mean mass flux over *both* devices' blocks hits the target.
        flux = sum(float(xp.sum(b.interior(q[1]) * b.interior(g_.vol)))
                   for b, q, g_ in zip(mesh.blocks, got, dist.geoms))
        volume = sum(float(xp.sum(b.interior(g_.vol))) for b, g_ in zip(mesh.blocks, ref.geoms))
        self.assertAlmostEqual(flux / volume, 1.2 * 40.0, delta=1e-9 * 48.0)
        totals = ref.conserved_totals(want)
        np.testing.assert_allclose(dist.conserved_totals(got), totals,
                                   rtol=1e-12, atol=1e-12 * np.abs(totals).max())

    def test_diagnostics_across_devices(self):
        mesh, ref = build("split", wm=True)
        dist = twin(ref, [0, 1], self.devices)
        Q = ref.init_state(prim_fn)
        stats_ref, stats = ref.wall_stats(Q), dist.wall_stats(dist.place_state(Q))
        self.assertEqual(stats_ref.keys(), stats.keys())
        for key in stats:
            for a, b in zip(stats[key], stats_ref[key]):
                np.testing.assert_allclose(a, b, rtol=TOL)
        loads_ref, loads = ref.wall_loads(Q), dist.wall_loads(dist.place_state(Q))
        for key in loads:
            for a, b in zip(loads[key], loads_ref[key]):
                np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9)


@unittest.skipUnless(HAS_JAX, "requires JAX")
@needs_four
class AirfoilDistributionTests(unittest.TestCase):
    def test_wall_modelled_airfoil_matches_reference_on_four_devices(self):
        devices = select_devices(4)
        mesh, ref, props = small_airfoil(4)
        self.assertGreaterEqual(len(mesh.blocks), 4)
        own = Ownership.balanced(mesh, 4)
        self.assertEqual(own.report(mesh)["empty_devices"], [])
        dist = DistributedSolver.from_solver(ref, own, devices)
        result = compare_with_reference(ref, dist, initial_state(ref, props), steps=2)
        self.assertLess(result["rhs"], TOL, result)
        self.assertLess(result["state"], TOL, result)
        self.assertTrue(result["finite"] and result["placement_ok"])
        self.assertGreater(own.report(mesh)["cross_device_interfaces"], 0)

    def test_global_cfl_when_the_limiting_block_is_not_on_device_zero(self):
        devices = select_devices(4)
        mesh, ref, props = small_airfoil(4)
        Q = initial_state(ref, props)
        Qb = ref.apply_bcs(Q, ref.geoms)
        per_block = [float(ref._block_dt(ib, q, g)) for ib, (q, g) in enumerate(zip(Qb, ref.geoms))]
        limiting = int(np.argmin(per_block))
        owner = [(ib + 1) % 4 for ib in range(len(mesh.blocks))]
        owner[limiting] = 3
        dist = twin(ref, owner, devices)
        dt = dist.max_dt(dist.place_state(Q))
        self.assertAlmostEqual(float(dt), min(per_block), delta=1e-14 * min(per_block))
        self.assertEqual(float(dt), float(ref.max_dt(Q)))


def _gpu_count():
    return ndev if ndev and backend._DEVICES[0].platform == "gpu" else 0


@unittest.skipUnless(HAS_JAX, "requires JAX")
class GpuHardwareGates(unittest.TestCase):
    def _gate(self, count):
        devices = select_devices(count)
        self.assertTrue(all(d.platform == "gpu" for d in devices))
        mesh, ref, props = small_airfoil(count)
        dist = DistributedSolver.from_solver(ref, Ownership.balanced(mesh, count), devices)
        result = compare_with_reference(ref, dist, initial_state(ref, props), steps=2)
        self.assertLess(result["rhs"], 1e-10, result)
        self.assertLess(result["state"], 1e-10, result)
        self.assertTrue(result["finite"] and result["placement_ok"])
        used = {next(iter(q.devices())) for q in dist.init_state(UNIFORM)}
        self.assertEqual(len(used), len(set(dist.block_device)))

    @unittest.skipUnless(_gpu_count() >= 2, "needs 2 GPUs")
    def test_two_gpus(self):
        self._gate(2)

    @unittest.skipUnless(_gpu_count() >= 8, "needs 8 GPUs")
    def test_eight_gpus(self):
        self._gate(8)


if __name__ == "__main__":
    unittest.main()
