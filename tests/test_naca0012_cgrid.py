import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv
from plot3d import read_plot3D

from wmles.airfoil_grid import (
    generate_four_block_mesh,
    naca0012_surface,
    read_airfoil_coordinates,
    spline_refine_contour,
    write_plot3d,
    write_vtk,
)


class Naca0012CGridTests(unittest.TestCase):
    def setUp(self):
        self.blocks = generate_four_block_mesh(
            ni=65,
            airfoil_divisions=5,
            airfoil_first_cell_height=2.0e-4,
            te_divisions=4,
            te_first_cell_height=2.0e-4,
            tunnel_height=3.5,
            tunnel_height_divisions=12,
            wake_length=7.0,
            wake_divisions=16,
        )

    def test_four_blocks_are_finite_and_have_expected_shape(self):
        self.assertEqual(
            list(self.blocks),
            ["airfoil", "trailing_edge", "tunnel", "wake"],
        )
        self.assertEqual(self.blocks["airfoil"].shape, (65, 6, 2))
        self.assertTrue(all(
            block.ndim == 3 and block.shape[2] == 2
            and np.all(np.isfinite(block))
            for block in self.blocks.values()
        ))

    def test_first_wall_spacing_and_airfoil_surface(self):
        first_cell_height = 2.0e-4
        airfoil = self.blocks["airfoil"]
        spacing = np.linalg.norm(airfoil[:, 1] - airfoil[:, 0], axis=1)
        np.testing.assert_allclose(spacing, first_cell_height, rtol=1e-10)
        self.assertAlmostEqual(airfoil[0, 0, 0], 1.0)
        self.assertAlmostEqual(airfoil[len(airfoil) // 2, 0, 0], 0.0)
        self.assertAlmostEqual(airfoil[-1, 0, 0], 1.0)

    def test_coordinate_file_is_normalized_and_oriented(self):
        coordinates = naca0012_surface(np.linspace(0.0, 1.0, 81))
        coordinates = coordinates[::-1] * np.array([2.0, 2.0]) + [3.0, 4.0]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "profile.dat"
            with path.open("w", encoding="utf-8") as output:
                output.write("NACA test profile\n# coordinates\n")
                for x, y in coordinates:
                    output.write(f"{x:.12f}, {y:.12f}\n")
            loaded = read_airfoil_coordinates(path)

        self.assertEqual(loaded.shape, (81, 2))
        self.assertAlmostEqual(loaded[0, 0], 1.0)
        self.assertAlmostEqual(loaded[-1, 0], 1.0)
        self.assertGreater(loaded[1, 1], loaded[-2, 1])
        self.assertAlmostEqual(float(np.ptp(loaded[:, 0])), 1.0)

    def test_bspline_refines_le_and_te(self):
        raw = naca0012_surface(np.linspace(0.0, 1.0, 81))
        contour = spline_refine_contour(
            raw,
            points=41,
            tolerance=172.0,
            te_old_segments=2,
            te_new_segments=4,
            te_ratio=3.0,
        )
        self.assertGreater(len(contour), 41 + 2 * (4 - 2))
        self.assertTrue(np.all(np.isfinite(contour)))
        self.assertAlmostEqual(contour[0, 0], 1.0, places=6)
        self.assertAlmostEqual(contour[-1, 0], 1.0, places=6)
        self.assertLess(np.min(contour[:, 0]), 1.0e-3)

    def test_le_refinement_points_survive_respline(self):
        raw = read_airfoil_coordinates("data/Airfoils/a-airfoil.dat")
        initial_count = 257
        contour = spline_refine_contour(
            raw,
            points=initial_count,
            tolerance=175.0,
            te_old_segments=1,
            te_new_segments=1,
        )

        self.assertGreater(len(contour), initial_count)
        le_index = int(np.argmin(contour[:, 0]))
        nearby = contour[le_index - 5:le_index + 6]
        self.assertEqual(len(nearby), 11)
        local_spacing = np.linalg.norm(np.diff(nearby, axis=0), axis=1)
        self.assertLess(float(np.max(local_spacing)), 0.005)

    def test_blunt_trailing_edge_gets_configured_face_divisions(self):
        parameter = np.linspace(0.0, 1.0, 65)
        x = 0.5 * (1.0 + np.cos(2.0 * np.pi * parameter))
        thickness = 0.12 / 0.2 * (
            0.2969 * np.sqrt(x)
            - 0.1260 * x
            - 0.3516 * x**2
            + 0.2843 * x**3
            - 0.1015 * x**4
        )
        blunt = np.column_stack((
            x, np.where(parameter <= 0.5, thickness, -thickness)
        ))
        divisions = 4

        blocks = generate_four_block_mesh(
            ni=65,
            airfoil_divisions=5,
            te_divisions=4,
            blunt_te_divisions=divisions,
            tunnel_height_divisions=12,
            wake_divisions=16,
            airfoil_coordinates=blunt,
            te_old_segments=2,
            te_new_segments=4,
        )
        expected_te_i_points = 2 * (5 + 1) - 1 + divisions
        self.assertEqual(
            blocks["trailing_edge"].shape[0], expected_te_i_points
        )
        self.assertTrue(np.all(np.isfinite(blocks["trailing_edge"])))
        airfoil = blocks["airfoil"]
        blunt_face = np.linspace(
            airfoil[-1, 0], airfoil[0, 0], divisions + 1
        )
        np.testing.assert_allclose(
            blocks["trailing_edge"][5:10, 0], blunt_face
        )
        for name, block in blocks.items():
            i_edges = block[1:, :-1] - block[:-1, :-1]
            j_edges = block[:-1, 1:] - block[:-1, :-1]
            areas = (
                i_edges[:, :, 0] * j_edges[:, :, 1]
                - i_edges[:, :, 1] * j_edges[:, :, 0]
            )
            self.assertTrue(
                np.all(areas > 0.0) or np.all(areas < 0.0),
                f"blunt {name} block contains inverted cells",
            )

    def test_blocks_share_matching_interfaces(self):
        airfoil = self.blocks["airfoil"]
        trailing_edge = self.blocks["trailing_edge"]
        tunnel = self.blocks["tunnel"]
        wake = self.blocks["wake"]

        np.testing.assert_allclose(trailing_edge[:, 0], np.vstack((
            airfoil[-1, ::-1],
            airfoil[0, 1:],
        )))
        np.testing.assert_allclose(
            tunnel[:, 0],
            np.vstack((
                trailing_edge[-1, ::-1][:-1],
                airfoil[:, -1][:-1],
                trailing_edge[0],
            )),
        )
        np.testing.assert_allclose(
            wake[0],
            np.vstack((
                tunnel[-1, ::-1][:-1],
                trailing_edge[:, -1][:-1],
                tunnel[0],
            )),
        )

    def test_blocks_have_consistent_cell_orientation(self):
        for name, block in self.blocks.items():
            i_edges = block[1:, :-1] - block[:-1, :-1]
            j_edges = block[:-1, 1:] - block[:-1, :-1]
            areas = (
                i_edges[:, :, 0] * j_edges[:, :, 1]
                - i_edges[:, :, 1] * j_edges[:, :, 0]
            )
            self.assertTrue(
                np.all(areas > 0.0) or np.all(areas < 0.0),
                f"{name} contains inverted or degenerate cells",
            )

    def test_default_mesh_has_consistent_cell_orientation(self):
        blocks = generate_four_block_mesh()
        for name, block in blocks.items():
            i_edges = block[1:, :-1] - block[:-1, :-1]
            j_edges = block[:-1, 1:] - block[:-1, :-1]
            areas = (
                i_edges[:, :, 0] * j_edges[:, :, 1]
                - i_edges[:, :, 1] * j_edges[:, :, 0]
            )
            self.assertTrue(
                np.all(areas > 0.0) or np.all(areas < 0.0),
                f"default {name} block contains inverted or degenerate cells",
            )

    def test_plot3d_writer_creates_one_spanwise_cell_per_block(self):
        span_length = 0.75
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "naca0012.xyz"
            write_plot3d(
                path, list(self.blocks.values()), span_length=span_length
            )
            written_blocks = read_plot3D(str(path), binary=False)

        self.assertEqual(len(written_blocks), 4)
        for source, written in zip(self.blocks.values(), written_blocks):
            self.assertEqual(written.X.shape, (*source.shape[:2], 2))
            self.assertEqual(written.Y.shape, (*source.shape[:2], 2))
            self.assertEqual(written.Z.shape, (*source.shape[:2], 2))
            np.testing.assert_allclose(
                written.X[:, :, 0], source[:, :, 0], atol=1e-14
            )
            np.testing.assert_allclose(
                written.X[:, :, 1], source[:, :, 0], atol=1e-14
            )
            np.testing.assert_allclose(
                written.Y[:, :, 0], source[:, :, 1], atol=1e-14
            )
            np.testing.assert_allclose(
                written.Y[:, :, 1], source[:, :, 1], atol=1e-14
            )
            np.testing.assert_allclose(written.Z[:, :, 0], 0.0)
            np.testing.assert_allclose(written.Z[:, :, 1], span_length)

    def test_vtk_writer_round_trips_four_structured_blocks(self):
        span_length = 0.75
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "naca0012.vtm"
            write_vtk(
                path, list(self.blocks.values()), span_length=span_length
            )
            written = pv.read(path)

        self.assertIsInstance(written, pv.MultiBlock)
        self.assertEqual(written.n_blocks, 4)
        for source, mesh in zip(self.blocks.values(), written):
            self.assertIsInstance(mesh, pv.StructuredGrid)
            self.assertEqual(mesh.dimensions, (*source.shape[:2], 2))
            points = mesh.points.reshape((*source.shape[:2], 2, 3), order="F")
            np.testing.assert_allclose(
                points[:, :, :, 0],
                np.repeat(source[:, :, 0, None], 2, axis=2),
            )
            np.testing.assert_allclose(
                points[:, :, :, 1],
                np.repeat(source[:, :, 1, None], 2, axis=2),
            )
            self.assertTrue(np.all(points[:, :, 0, 2] == 0.0))
            self.assertTrue(np.all(points[:, :, 1, 2] == span_length))

    def test_rejects_invalid_settings(self):
        with self.assertRaises(ValueError):
            generate_four_block_mesh(ni=64)
        with self.assertRaises(ValueError):
            generate_four_block_mesh(wake_equalize=1.0)
        with self.assertRaises(ValueError):
            generate_four_block_mesh(airfoil_first_cell_height=float("nan"))
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                write_plot3d(
                    Path(directory) / "bad.xyz",
                    list(self.blocks.values()),
                    span_length=0.0,
                )
        with self.assertRaises(ValueError):
            spline_refine_contour(
                naca0012_surface(np.linspace(0.0, 1.0, 21)),
                points=10,
                te_old_segments=5,
            )


if __name__ == "__main__":
    unittest.main()
