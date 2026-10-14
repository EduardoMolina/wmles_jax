"""Grid and setup checks for the UMD smooth-ramp case."""
import numpy as np

from wmles.bc import CharacteristicFarfield, SlipWall, WallModel
from wmles.mesh import Interface
from wmles.smooth_ramp import (
    RAMP_HEIGHT_L,
    SPAN_L,
    TOP_HEIGHT_L,
    SmoothRampInflow,
    SmoothRampSponge,
    build_case,
    flow_properties,
    initial,
    load_dns_profile,
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
    inflow = blk.faces[(0, 0)]
    assert np.allclose(inflow.profile, load_dns_profile())
    state_t0 = solver.apply_bcs(state, solver.geoms)
    state_t1 = solver.apply_bcs(state, solver.geoms, 0.1 * props["L"] / props["U_inf"])
    inlet_t0 = np.asarray(state_t0[0][:, :blk.ng, blk.ng:-blk.ng, blk.ng:-blk.ng])
    inlet_t1 = np.asarray(state_t1[0][:, :blk.ng, blk.ng:-blk.ng, blk.ng:-blk.ng])
    assert not np.allclose(inlet_t0, inlet_t1)
    state = state_t0
    primitive, _ = solver.gas.cons_to_prim(state[0])
    assert np.all(np.isfinite(np.asarray(primitive)))
    assert np.all(np.asarray(blk.interior(primitive[0])) > 0.0)
    inlet = np.asarray(primitive[:, blk.ng - 1, blk.ng:-blk.ng, blk.ng:-blk.ng])
    y_inlet_l = np.asarray(
        solver.geoms[0].xc[1, blk.ng - 1, blk.ng:-blk.ng, blk.ng]
    ) / props["L"]
    expected = np.stack([np.interp(y_inlet_l, inflow.profile[:, 0], inflow.profile[:, col])
                         for col in range(1, 7)], axis=0)
    u_fluc = inlet[1] / props["U_inf"] - expected[0, :, None]
    v_fluc = inlet[2] / props["U_inf"] - expected[1, :, None]
    assert np.allclose(np.mean(inlet[1] / props["U_inf"], axis=1), expected[0], atol=1e-12)
    assert np.allclose(np.mean(inlet[2] / props["U_inf"], axis=1), expected[1], atol=1e-12)
    assert np.allclose(np.mean(u_fluc ** 2, axis=1), expected[2], atol=1e-10)
    assert np.allclose(np.mean(v_fluc ** 2, axis=1), expected[3], atol=1e-10)
    assert np.allclose(np.mean(u_fluc * v_fluc, axis=1), expected[5], atol=1e-10)
    assert len(solver.sources) == 1
    assert isinstance(solver.sources[0], SmoothRampSponge)


def test_dns_profile_validation_and_sponge_mean_preservation():
    profile = load_dns_profile()
    assert profile.ndim == 2 and profile.shape[1] == 7 and profile.shape[0] > 121
    assert profile[0, 0] == 0.22

    mesh, solver, props = build_case(
        x_min=-0.05, x_max=1.05, nz=4, sponge_start=0.6
    )
    blk, geom = mesh.blocks[0], mesh.geoms[0]
    Q = solver.init_state(lambda xc: initial(xc, props))
    q = np.asarray(Q[0]).copy()
    z = blk.interior(geom.xc[2])[0, 0]
    q[1, blk.ng:-blk.ng, blk.ng:-blk.ng, blk.ng:-blk.ng] += (
        0.01 * np.sin(2.0 * np.pi * z / (SPAN_L * props["L"]))
    )[None, None, :]
    source = np.asarray(solver.sources[0](blk, q, geom, props["gas"]))
    x_l = blk.interior(geom.xc[0])[:, 0, 0] / props["L"]
    before_sponge = np.where(x_l < 0.6)[0]
    after_sponge = np.where(x_l > 0.6)[0]
    assert np.all(source[:, before_sponge] == 0.0)
    assert np.any(np.abs(source[1, after_sponge]) > 0.0)
    vol = blk.interior(geom.vol)
    assert np.allclose(np.sum(source[1] * vol, axis=2), 0.0, atol=1e-12)


def test_smooth_ramp_flow_properties_match_reynolds_number():
    props = flow_properties()
    assert np.isclose(props["U_inf"] * props["L"] / props["nu_inf"], props["Re_L"])


if __name__ == "__main__":
    test_grid6_distributions_and_ramp_mesh()
    test_dns_profile_validation_and_sponge_mean_preservation()
    test_smooth_ramp_flow_properties_match_reynolds_number()
    print("smooth-ramp grid and setup: ok")
