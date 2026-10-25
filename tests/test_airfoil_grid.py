import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv

from wmles.airfoil_grid import generate_four_block_mesh, main, read_airfoil_coordinates
from wmles.plot3d_wm import read_plot3d


class AirfoilGridTests(unittest.TestCase):
    def test_reads_closed_oat15a_contour_in_trailing_edge_order(self):
        coordinates = read_airfoil_coordinates("tests/data/oat15a.dat")

        self.assertAlmostEqual(coordinates[0, 0], 1.0)
        self.assertAlmostEqual(coordinates[-1, 0], 1.0)
        self.assertGreater(coordinates[0, 1], coordinates[-1, 1])
        self.assertAlmostEqual(float(coordinates[:, 0].min()), 0.0)
        self.assertGreater(len(coordinates), 400)

    def test_full_resolution_a_airfoil_mesh_has_no_folded_cells(self):
        blocks = generate_four_block_mesh(
            ni=1025,
            airfoil_divisions=35,
            airfoil_first_cell_height=0.00015,
            te_divisions=15,
            te_first_cell_height=0.0003,
            tunnel_height=15.0,
            tunnel_height_divisions=100,
            tunnel_height_ratio=20.0,
            wake_length=15.0,
            wake_divisions=150,
            te_old_segments=1,
            te_new_segments=1,
            te_refinement_ratio=2.0,
            airfoil_coordinates=read_airfoil_coordinates(
                "tests/data/a-airfoil.dat"
            ),
        )
        for name, block in blocks.items():
            edge_i = block[1:, :-1] - block[:-1, :-1]
            edge_j = block[:-1, 1:] - block[:-1, :-1]
            signed_areas = (
                edge_i[:, :, 0] * edge_j[:, :, 1]
                - edge_i[:, :, 1] * edge_j[:, :, 0]
            )
            self.assertTrue(
                np.all(signed_areas > 0.0) or np.all(signed_areas < 0.0),
                f"{name} contains folded or degenerate cells",
            )

    def test_tunnel_smoothing_improves_skew_and_preserves_interfaces(self):
        settings = {
            "ni": 65,
            "airfoil_divisions": 5,
            "te_divisions": 4,
            "tunnel_height_divisions": 12,
            "tunnel_height_ratio": 20.0,
            "wake_divisions": 16,
            "airfoil_coordinates": read_airfoil_coordinates(
                "tests/data/a-airfoil.dat"
            ),
        }
        raw = generate_four_block_mesh(
            **settings, tunnel_smoothing=0.0
        )["tunnel"]
        smoothed = generate_four_block_mesh(
            **settings, tunnel_smoothing=0.1
        )["tunnel"]

        np.testing.assert_allclose(smoothed[[0, -1]], raw[[0, -1]])
        np.testing.assert_allclose(smoothed[:, [0, -1]], raw[:, [0, -1]])

        def metrics(block):
            edge_i = block[1:, :-1] - block[:-1, :-1]
            edge_j = block[:-1, 1:] - block[:-1, :-1]
            cosine = np.sum(edge_i * edge_j, axis=2) / (
                np.linalg.norm(edge_i, axis=2)
                * np.linalg.norm(edge_j, axis=2)
            )
            return (
                np.median(np.abs(cosine)),
                np.min(np.linalg.norm(edge_j, axis=2)),
            )

        i_edges = smoothed[1:, :-1] - smoothed[:-1, :-1]
        j_edges = smoothed[:-1, 1:] - smoothed[:-1, :-1]
        signed_areas = (
            i_edges[:, :, 0] * j_edges[:, :, 1]
            - i_edges[:, :, 1] * j_edges[:, :, 0]
        )
        self.assertTrue(np.all(signed_areas < 0.0))
        raw_median_skew, raw_min_spacing = metrics(raw)
        smooth_median_skew, smooth_min_spacing = metrics(smoothed)
        self.assertLess(smooth_median_skew, raw_median_skew)
        self.assertGreater(smooth_min_spacing, raw_min_spacing)

    def test_cli_uses_a_airfoil_profile_and_uniform_span_spacing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "a_airfoil.p3d"
            main([
                str(output),
                "--ni", "65",
                "--airfoil-divisions", "4",
                "--te-divisions", "4",
                "--tunnel-height-divisions", "8",
                "--wake-divisions", "12",
                "--span-length", "0.2",
                "--spanwise-spacing", "0.01",
            ])
            blocks = read_plot3d(output).nodes
            try:
                from plot3d import read_plot3D
            except ImportError:
                read_plot3D = None
            if read_plot3D is not None:
                nasa_blocks = read_plot3D(str(output))

        self.assertEqual(len(blocks), 4)
        for block in blocks:
            self.assertEqual(block.shape[-1], 21)
            np.testing.assert_allclose(block[2, 0, 0], np.linspace(0.0, 0.2, 21))
            np.testing.assert_allclose(np.diff(block[2, 0, 0]), 0.01)
        if read_plot3D is not None:
            self.assertEqual(len(nasa_blocks), len(blocks))
            for nasa_block, block in zip(nasa_blocks, blocks):
                np.testing.assert_allclose(
                    np.stack((nasa_block.X, nasa_block.Y, nasa_block.Z)), block
                )

    def test_vtk_output_has_uniform_span_spacing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "a_airfoil.vtm"
            main([
                str(output),
                "--ni", "65",
                "--airfoil-divisions", "4",
                "--te-divisions", "4",
                "--tunnel-height-divisions", "8",
                "--wake-divisions", "12",
                "--span-length", "0.2",
                "--spanwise-spacing", "0.01",
            ])
            blocks = pv.read(output)

        self.assertEqual(blocks.n_blocks, 4)
        for block in blocks:
            self.assertEqual(block.dimensions[2], 21)
            levels = np.unique(np.round(block.points[:, 2], 10))
            np.testing.assert_allclose(np.diff(levels), 0.01)

    def test_span_must_be_an_integer_multiple_of_spacing(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "bad.p3d"
            with self.assertRaises(SystemExit):
                main([
                    str(output),
                    "--span-length", "0.2",
                    "--spanwise-spacing", "0.03",
                ])
        with self.assertRaises(ValueError):
            generate_four_block_mesh(tunnel_smoothing=0.6)


if __name__ == "__main__":
    unittest.main()
