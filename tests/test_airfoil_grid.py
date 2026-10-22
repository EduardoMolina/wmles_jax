import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyvista as pv

from wmles.airfoil_grid import main
from wmles.plot3d import read_plot3d


class AirfoilGridTests(unittest.TestCase):
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
            ])
            blocks = read_plot3d(output).nodes

        self.assertEqual(len(blocks), 4)
        for block in blocks:
            self.assertEqual(block.shape[-1], 21)
            np.testing.assert_allclose(block[2, 0, 0], np.linspace(0.0, 0.2, 21))
            np.testing.assert_allclose(np.diff(block[2, 0, 0]), 0.01)

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


if __name__ == "__main__":
    unittest.main()
