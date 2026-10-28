import tempfile
import unittest
from pathlib import Path

import numpy as np

from wmles.airfoil_case import (
    ALPHA_DEG,
    MACH_INF,
    RE_CHORD,
    build_case,
    flow_properties,
    initial_state,
)
from wmles.airfoil_grid import (
    generate_four_block_mesh,
    read_airfoil_coordinates,
    write_plot3d,
)
from wmles.bc import CharacteristicFarfield, WallModel
from wmles.mesh import Interface


class AirfoilCaseTests(unittest.TestCase):
    def test_builds_wall_modelled_oat15a_blunt_trailing_edge(self):
        blocks = generate_four_block_mesh(
            ni=65,
            airfoil_divisions=5,
            te_divisions=4,
            tunnel_height_divisions=12,
            wake_divisions=16,
            airfoil_coordinates=read_airfoil_coordinates(
                "tests/data/oat15a.dat"
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            grid_path = Path(directory) / "oat15a.p3d"
            write_plot3d(
                grid_path, list(blocks.values()),
                span_length=0.2, spanwise_cells=8,
            )
            mesh, solver, props = build_case(grid_path)

        te_wall_faces = [
            bc for block in mesh.blocks
            if block.origin["block"] == 1
            for face, bc in block.faces.items()
            if face == (1, 0) and isinstance(bc, WallModel)
        ]
        self.assertTrue(te_wall_faces)
        wall_source_blocks = {
            block.origin["block"] for block in mesh.blocks
            if any(isinstance(bc, WallModel) for bc in block.faces.values())
        }
        self.assertTrue({0, 1}.issubset(wall_source_blocks))
        self.assertTrue(any(
            isinstance(bc, CharacteristicFarfield)
            for block in mesh.blocks for bc in block.faces.values()
        ))

        expected_cells = sum(
            (block.shape[0] - 1) * (block.shape[1] - 1) * 8
            for block in blocks.values()
        )
        self.assertEqual(sum(int(np.prod(block.n)) for block in mesh.blocks), expected_cells)
        for index, block in enumerate(mesh.blocks):
            self.assertGreater(float(np.asarray(block.interior(block.geom.vol)).min()), 0.0)
            for side in (0, 1):
                periodic = block.faces[(2, side)]
                self.assertIsInstance(periodic, Interface)
                self.assertEqual(periodic.nb, index)

        state = initial_state(solver, props)
        self.assertTrue(all(np.all(np.isfinite(np.asarray(q))) for q in state))
        dt = solver.max_dt(state)
        self.assertGreater(float(dt), 0.0)
        advanced = solver.advance(state, dt)
        self.assertTrue(all(np.all(np.isfinite(np.asarray(q))) for q in advanced))

    def test_reference_flow_properties(self):
        props = flow_properties()
        self.assertEqual(props["mach"], MACH_INF)
        self.assertEqual(props["reynolds"], RE_CHORD)
        self.assertEqual(props["alpha_deg"], ALPHA_DEG)
        self.assertGreater(props["chord"], 0.0)
        self.assertGreater(props["prim_inf"][1], 0.0)
        self.assertLess(props["prim_inf"][2], 0.0)

    def test_builds_wall_modelled_periodic_four_block_case(self):
        blocks = generate_four_block_mesh(
            ni=65,
            airfoil_divisions=5,
            te_divisions=4,
            tunnel_height_divisions=12,
            wake_divisions=16,
            airfoil_coordinates=read_airfoil_coordinates(
                "tests/data/a-airfoil.dat"
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            grid_path = Path(directory) / "small_airfoil.p3d"
            write_plot3d(
                grid_path, list(blocks.values()),
                span_length=0.2, spanwise_cells=8,
            )
            mesh, solver, props = build_case(grid_path)

        self.assertGreaterEqual(len(mesh.blocks), 4)
        self.assertEqual(props["grid_path"], grid_path)
        self.assertTrue(any(
            isinstance(bc, WallModel)
            for block in mesh.blocks for bc in block.faces.values()
        ))
        self.assertTrue(any(
            isinstance(bc, CharacteristicFarfield)
            for block in mesh.blocks for bc in block.faces.values()
        ))
        self.assertTrue(any(
            isinstance(bc, Interface) and bc.nb == index
            for index, block in enumerate(mesh.blocks)
            for face, bc in block.faces.items()
            if face[0] == 2
        ))
        state = initial_state(solver, props)
        self.assertTrue(all(np.all(np.isfinite(np.asarray(x))) for x in state))
        dt = solver.max_dt(state)
        advanced = solver.advance(state, dt)
        self.assertTrue(all(np.all(np.isfinite(np.asarray(x))) for x in advanced))


if __name__ == "__main__":
    unittest.main()
