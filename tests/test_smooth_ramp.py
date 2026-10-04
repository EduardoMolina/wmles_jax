"""Grid and setup checks for the UMD smooth-ramp case."""
import numpy as np

from wmles.bc import CharacteristicFarfield, SlipWall, WallModel
from wmles.mesh import Interface
from wmles.smooth_ramp import (
    RAMP_HEIGHT_L,
    SPAN_L,
    TOP_HEIGHT_L,
    SmoothRampInflow,
    build_case,
    flow_properties,
    initial,
    load_grid6_distributions,
)


def test_grid6_distributions_and_ramp_mesh():
    distributions = load_grid6_distributions()
    assert distributions["x_wall"].shape == (293, 2)
    assert distributions["y_upstream"].shape == (121, 2)
    assert distributions["y_downstream"].shape == (121, 2)

    mesh, solver, props = build_case(x_min=-0.05, x_max=1.05, nz=4)
    blk = mesh.blocks[0]
    x_l = blk.x_int[0, :, 0, 0] / props["L"]
    y_l = blk.x_int[1, :, :, 0] / props["L"]
    ramp = 10.0 * np.clip(x_l, 0.0, 1.0) ** 3 - 15.0 * np.clip(x_l, 0.0, 1.0) ** 4 \
        + 6.0 * np.clip(x_l, 0.0, 1.0) ** 5
    expected_wall = RAMP_HEIGHT_L * (1.0 - ramp)

    assert blk.n == (292, 120, 4)
    assert np.allclose(y_l[:, 0], expected_wall)
    assert np.allclose(y_l[:, -1], TOP_HEIGHT_L)
    assert np.all(np.asarray(mesh.geoms[0].vol) > 0.0)
    assert np.isclose(blk.x_int[2, 0, 0, -1] / props["L"], SPAN_L)
    assert isinstance(blk.faces[(0, 0)], SmoothRampInflow)
    assert isinstance(blk.faces[(0, 1)], CharacteristicFarfield)
    assert isinstance(blk.faces[(1, 0)], WallModel)
    assert isinstance(blk.faces[(1, 1)], SlipWall)
    assert isinstance(blk.faces[(2, 0)], Interface)
    assert np.isclose(props["U_inf"] / props["gas"].sound_speed(props["T_inf"]), 0.2)
    state = solver.init_state(lambda xc: initial(xc, props))
    state = solver.apply_bcs(state, solver.geoms)
    primitive, _ = solver.gas.cons_to_prim(state[0])
    assert np.all(np.isfinite(np.asarray(primitive)))
    assert np.all(np.asarray(blk.interior(primitive[0])) > 0.0)


def test_smooth_ramp_flow_properties_match_reynolds_number():
    props = flow_properties()
    assert np.isclose(props["U_inf"] * props["L"] / props["nu_inf"], props["Re_L"])


if __name__ == "__main__":
    test_grid6_distributions_and_ramp_mesh()
    test_smooth_ramp_flow_properties_match_reynolds_number()
    print("smooth-ramp grid and setup: ok")
