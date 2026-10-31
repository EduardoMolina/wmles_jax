"""Wall distance and Delta_max.  Run:  python -m unittest tests.test_wall_distance"""
import unittest

import numpy as np

from wmles import Block, Mesh, NoSlipWall, SlipWall, Extrapolate
from wmles.wall_distance import (attach_wall_distance, distance_to_patches,
                                 point_triangle_distance2, wall_distance)
from tests.test_multiblock import build as build_multiblock


def sphere_patch(radius=1.0, n_lat=65, n_lon=129):
    """Lat-lon sphere surface (3, n_lat, n_lon) incl. both poles (collapsed, degenerate faces)."""
    th = np.linspace(0.0, np.pi, n_lat)[:, None]
    ph = np.linspace(0.0, 2.0 * np.pi, n_lon)[None, :]
    return radius * np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph),
                              np.cos(th) + 0.0 * ph])


def plate(n=(8, 12, 6), L=(1.0, 0.5, 0.4), wall=True, periodic_xz=True):
    """One block, stretched in y, wall on jmin.  x/z periodic so ghosts see a full plate."""
    sx = np.linspace(0.0, 1.0, n[1] + 1) ** 1.7 * L[1]
    X, Y, Z = np.meshgrid(np.linspace(0, L[0], n[0] + 1), sx, np.linspace(0, L[2], n[2] + 1),
                          indexing="ij")
    A = Block("plate", np.stack([X, Y, Z]))
    m = Mesh([A])
    if periodic_xz:
        m.connect(A, "imin", A, "imax", shift=(-L[0], 0, 0))
        m.connect(A, "kmin", A, "kmax", shift=(0, 0, -L[2]))
    else:
        for f in ("imin", "imax", "kmin", "kmax"):
            A.set_bc(f, Extrapolate())
    A.set_bc("jmin", NoSlipWall() if wall else SlipWall())
    A.set_bc("jmax", Extrapolate())
    return m.finalize(), A, L


def tri_reference(p, a, b, c, m=400):
    """Dense barycentric sampling of the triangle (independent of the Ericson regions)."""
    u, v = np.meshgrid(np.linspace(0, 1, m), np.linspace(0, 1, m), indexing="ij")
    ok = u + v <= 1.0
    u, v = u[ok], v[ok]
    pts = a + u[:, None] * (b - a) + v[:, None] * (c - a)
    return np.min(np.sum((pts - p) ** 2, axis=1))


class WallDistanceTests(unittest.TestCase):
    def test_point_triangle_regions(self):
        rng = np.random.default_rng(1)
        a, b, c = np.array([0.0, 0, 0]), np.array([1.0, 0.2, 0]), np.array([0.3, 1.1, 0.4])
        for _ in range(40):
            p = rng.uniform(-1.0, 2.0, 3)
            exact = float(point_triangle_distance2(p, a, b, c))
            ref = tri_reference(p, a, b, c)
            self.assertLessEqual(exact, ref + 1e-12)          # exact is never worse than sampling
            self.assertAlmostEqual(exact, ref, delta=2e-3)
        degenerate = float(point_triangle_distance2(np.array([0.0, 1.0, 0]), a, a, a))
        self.assertAlmostEqual(degenerate, 1.0)

    def test_flat_plate_is_height_including_ghosts_and_periodic_images(self):
        mesh, blk, _ = plate()
        d = wall_distance(mesh)[0]
        y = np.asarray(mesh.geoms[0].xc)[1]
        np.testing.assert_allclose(d, np.abs(y), rtol=0.0, atol=1e-12)

    def test_attach_sets_geom_and_mesh(self):
        mesh, blk, _ = plate()
        self.assertIsNone(mesh.geoms[0].dist)
        attach_wall_distance(mesh)
        self.assertIs(mesh.geoms[0], blk.geom)
        np.testing.assert_allclose(np.asarray(blk.geom.dist), np.abs(np.asarray(blk.geom.xc)[1]), atol=1e-12)

    def test_no_wall_raises(self):
        mesh, _, _ = plate(wall=False)
        with self.assertRaises(ValueError):
            wall_distance(mesh)

    def test_delta_max_of_box_cells(self):
        mesh, blk, L = plate(n=(8, 12, 6), L=(1.0, 0.5, 0.4))
        dm = np.asarray(mesh.geoms[0].delta_max)
        y = np.asarray(blk.X)[1, 0, :, 0]
        dy = np.diff(y)
        expect = np.maximum(np.maximum(L[0] / 8, L[2] / 6), dy)[None, :, None]
        np.testing.assert_allclose(dm, np.broadcast_to(expect, dm.shape), rtol=1e-12)

    def test_candidate_search_matches_brute_force_on_skewed_grid(self):
        mesh, _ = build_multiblock("single", wm=True)
        fast = wall_distance(mesh, k=24)[0]
        full = wall_distance(mesh, k=10 ** 6)[0]
        np.testing.assert_allclose(fast, full, rtol=0.0, atol=1e-12)

    def test_sphere_matches_analytic_distance(self):
        rng = np.random.default_rng(3)
        v = rng.normal(size=(4000, 3))
        v /= np.linalg.norm(v, axis=1)[:, None]
        r = rng.uniform(0.02, 3.0, 4000)
        pts = v * r[:, None]
        pts[:20] = np.outer(rng.uniform(0.3, 2.0, 20), [0.0, 0.0, 1.0])     # on the pole axis
        r[:20] = pts[:20, 2]
        d = distance_to_patches(pts, [sphere_patch()])
        self.assertLess(np.max(np.abs(d - np.abs(r - 1.0))), 1e-3)     # facet sag only
        # facets lie inside the sphere: points outside see a distance >= analytic - sag
        outside = r > 1.0
        self.assertTrue(np.all(d[outside] >= (r - 1.0)[outside] - 1e-12))

    def test_exact_search_recovers_unequal_faces(self):
        # one 10 x 10 face at z=0 and 400 tiny faces nearer to the query point than the big
        # face's centre: the plain k=24 search picks the wrong (tiny) faces
        big = np.stack(np.meshgrid([0.0, 10.0], [0.0, 10.0], [0.0], indexing="ij")).reshape(3, 2, 2)
        gx, gy = np.meshgrid(np.linspace(12.0, 13.0, 21), np.linspace(9.0, 10.0, 21), indexing="ij")
        tiny = np.stack([gx, gy, 0.3 + 0.0 * gx])
        p = np.array([[9.9, 9.9, 0.5]])
        stats = {}
        got = distance_to_patches(p, [big, tiny], k=24, stats=stats)
        coarse = distance_to_patches(p, [big, tiny], k=24, exact=False)
        self.assertAlmostEqual(got[0], 0.5, places=12)
        self.assertGreater(coarse[0], 1.0)                 # the uncertified search is wrong here
        self.assertEqual(stats["researched"] > 0, True)
        brute = distance_to_patches(p, [big, tiny], k=10 ** 6, exact=False)
        self.assertAlmostEqual(got[0], brute[0], places=12)

    def test_doubly_curved_bump_wall_in_a_mesh(self):
        n, L = (10, 14, 10), (1.0, 0.6, 1.0)
        x, y, z = np.meshgrid(np.linspace(0, L[0], n[0] + 1), np.linspace(0, 1, n[1] + 1) ** 1.5 * L[1],
                              np.linspace(0, L[2], n[2] + 1), indexing="ij")
        bump = 0.08 * np.sin(np.pi * x) ** 2 * np.sin(np.pi * z) ** 2
        A = Block("bump", np.stack([x, y + bump * (1 - y / L[1]), z]))
        m = Mesh([A])
        m.connect(A, "imin", A, "imax", shift=(-L[0], 0, 0))
        m.connect(A, "kmin", A, "kmax", shift=(0, 0, -L[2]))
        A.set_bc("jmin", NoSlipWall())
        A.set_bc("jmax", Extrapolate())
        m.finalize()
        stats = {}
        d = wall_distance(m, stats=stats)[0]
        full = wall_distance(m, k=10 ** 6, exact=False)[0]
        np.testing.assert_allclose(d, full, rtol=0.0, atol=1e-12)
        xc = np.asarray(m.geoms[0].xc)
        wall_y = np.asarray(A.x_int)[1, :, 0, :]                     # (ni+1, nk+1) wall heights
        self.assertGreater(wall_y.max() - wall_y.min(), 0.05)        # genuinely 3D wall
        self.assertTrue(np.all(d > 0.0))
        # never farther than the vertical distance to the wall surface above/below the cell
        self.assertTrue(np.all(d[:, 4:-4, :] <= np.abs(xc[1, :, 4:-4, :]) + 0.08 + 1e-12))

    def test_split_mesh_matches_single_block(self):
        def by_coord(mesh):
            out = {}
            for blk, d in zip(mesh.blocks, wall_distance(mesh)):
                xc = np.asarray(blk.interior(blk.geom.xc)).reshape(3, -1).T
                for c, v in zip(xc, blk.interior(d).reshape(-1)):
                    out[tuple(np.round(c, 8))] = v
            return out

        ref = by_coord(build_multiblock("single", wm=True)[0])
        for kind in ("split", "flip", "rot_cyc"):
            got = by_coord(build_multiblock(kind, wm=True)[0])
            self.assertEqual(ref.keys(), got.keys())
            err = max(abs(ref[k] - got[k]) for k in ref)
            self.assertLess(err, 1e-12, kind)


if __name__ == "__main__":
    unittest.main()
