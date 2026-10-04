"""Named frames survive physical topology compaction on CPU and CUDA."""

import os
from pathlib import Path

import numpy as np
import pytest
import torch

import kangengine as ke
from kangengine.state import KangWorldState


def chain(tmp_path, order="DFS"):
    path = tmp_path / "frames.urdf"
    inertia = '''<inertial><mass value="1"/>
      <inertia ixx="1" iyy="1" izz="1" ixy="0" ixz="0" iyz="0"/></inertial>'''
    path.write_text(f'''<robot name="frames">
      <link name="root">{inertia}</link>
      <link name="mount"/>
      <joint name="mount_fixed" type="fixed"><parent link="root"/><child link="mount"/>
        <origin xyz="1 0 0" rpy="0 0 1.5707963267948966"/></joint>
      <link name="arm">{inertia}</link>
      <joint name="hinge" type="revolute"><parent link="mount"/><child link="arm"/>
        <origin xyz="1 0 0"/><axis xyz="0 0 1"/>
        <limit lower="-3" upper="3" effort="10" velocity="10"/></joint>
      <link name="tip"/>
      <joint name="tip_fixed" type="fixed"><parent link="arm"/><child link="tip"/>
        <origin xyz="1 0 0"/></joint>
      <link name="tip_end"/>
      <joint name="tip_end_fixed" type="fixed"><parent link="tip"/><child link="tip_end"/>
        <origin xyz="0.2 0 0"/></joint>
      <link name="sibling">{inertia}</link>
      <joint name="sibling_fixed" type="fixed"><parent link="root"/><child link="sibling"/>
        <origin xyz="0 0 1"/></joint>
    </robot>''')
    return ke.asset.URDFLoader.load(str(path), order=order)


@pytest.mark.parametrize("order", ["DFS", "BFS"])
@pytest.mark.parametrize("gpu,device", [(False, "cpu"), (True, "cuda"), (True, "cpu")])
def test_fixed_chain_pose_and_indices(tmp_path, order, gpu, device, monkeypatch):
    if gpu and os.environ.get("KANGENGINE_TEST_PHYSX_GPU") != "1":
        pytest.skip("set KANGENGINE_TEST_PHYSX_GPU=1")
    data = chain(tmp_path, order)
    names = data.skeleton_tree.node_names()
    data.add_fixed_frame(ke.asset.FixedFrameDesc(
        "sensor", body_index=names.index("tip_end"), pos=[0, .3, 0]))
    config = ke.physics.PhysicsConfig.z_up()
    config.enable_gpu = gpu
    world = ke.physics.PhysicsWorld(config)
    template = ke.physics.ArticulationTemplate.create(data)
    assert list(template.body_names) == (
        ["root", "arm", "sibling"] if order == "DFS" else ["root", "sibling", "arm"])
    assert data.skeleton_tree.num_joints() == 6
    assert template.num_dofs() == 1
    robots = [ke.physics.Articulation.build_from_template(world, template) for _ in range(2)]
    system = None
    state = KangWorldState(num_envs=2, device=device,
                          canonical_source="gpu" if gpu else "cpu",
                          gpu_system_provider=lambda: system)
    try:
        for e, robot in enumerate(robots):
            robot.set_dof_state([.4], [0.])
            state.add_articulation(robot, env_id=e, obj_id=0, physics=world)
        if gpu:
            system = ke.physics.PhysicsGpuSystem(world, ke.physics.GpuPhysicsConfig())
            system.init()
        world.step()
        if not gpu:
            state.refresh()
        selected = ["tip_end", "mount", "sensor"]
        def reject_host(*args, **kwargs):
            raise AssertionError("Frame query downloaded a CUDA tensor")
        with monkeypatch.context() as guard:
            if gpu:
                guard.setattr(torch.Tensor, "cpu", reject_host)
                guard.setattr(torch.Tensor, "numpy", reject_host)
            positions = state.get_frame_pos(0, frame_names=selected)
            rotations = state.get_frame_rot(0, frame_names=selected)
            state.get_frame_pos(0, frame_names=selected, fetch=False)
        angle = np.pi/2 + .4
        arm = np.array([1., 1., 0.])
        end = arm + 1.2 * np.array([np.cos(angle), np.sin(angle), 0])
        sensor = end + .3 * np.array([-np.sin(angle), np.cos(angle), 0])
        expected = np.stack([end, [1, 0, 0], sensor])
        np.testing.assert_allclose(positions.cpu(), np.stack([expected]*2), atol=2e-5)
        layout = ke.animation.ArticulationCoordinateLayout.from_data(data)
        fk = ke.animation.ArticulationMotionMapper(layout).to_skeleton_state(
            np.array([.4], dtype=np.float32)).compute_global_positions()
        tip = fk[names.index("tip_end")]
        np.testing.assert_allclose([tip.x, tip.y, tip.z], end, atol=2e-5)
        np.testing.assert_allclose(rotations.cpu()[0, 0],
                                   [0, 0, np.sin(angle/2), np.cos(angle/2)], atol=2e-5)
        assert positions.is_cuda == gpu
        assert state.get_frame_pos(0, frame_names=[]).shape == (2, 0, 3)
        with pytest.raises(KeyError):
            state.get_frame_pos(0, frame_names=["missing"])
        if not gpu:
            visual = ke.visual.ArticulationVisualAsset.from_data(data)
            scene = ke.scene.create_backend(ke.scene.BackendType.NATIVE)
            view = visual.instantiate(scene, "/robot", "/.Resources/frames", True,
                                      hierarchical=True)
            marker = view.body_prim(names.index("tip_end"))
            assert marker.get_parent() == view.body_prim(names.index("tip"))
            marker_version = marker.get_transform_component().version
            bridge = ke.physics.PhysicsBridge()
            bridge.add(robots[0], view)
            bridge.sync()
            matrix = np.asarray(view.body_prim(names.index("tip_end")).compute_world_matrix())
            np.testing.assert_allclose(matrix.reshape(4, 4, order="F")[:3, 3], end, atol=2e-5)
            assert marker.get_transform_component().version == marker_version
            # FK updates still use local transforms in a hierarchical visual.
            view.set_joint_rotation(names.index("arm"), [0., 0., 0., 1.])
            view.apply_pose()
            zero_end = np.asarray(marker.compute_world_matrix()).reshape(4, 4, order="F")[:3, 3]
            np.testing.assert_allclose(zero_end, [1, 2.2, 0], atol=2e-5)
    finally:
        state.release()
        if system is not None:
            system.invalidate()
        for robot in robots:
            robot.release()


def test_site_and_authored_frame_validation(tmp_path):
    path = tmp_path / "sites.xml"
    path.write_text('''<mujoco><worldbody><body name="root">
      <inertial mass="1" diaginertia="1 1 1"/>
      <body name="marker" pos="1 0 0">
        <site name="imu" pos="0 2 0" rgba="1 0 0 1" size=".01"/>
      </body></body></worldbody></mujoco>''')
    data = ke.asset.MJCFLoader.load(str(path))
    template = ke.physics.ArticulationTemplate.create(data)
    frame = next(f for f in template.fixed_frames if f.name == "imu")
    assert isinstance(data.sites["imu"], ke.asset.FixedFrameDesc)
    assert frame.body_index == 0
    np.testing.assert_allclose([frame.pos.x, frame.pos.y, frame.pos.z], [1, 2, 0])
    with pytest.raises(ValueError):
        data.add_fixed_frame(ke.asset.FixedFrameDesc("imu", body_index=0))
    with pytest.raises(ValueError):
        data.add_fixed_frame(ke.asset.FixedFrameDesc("bad", body_index=12))
    with pytest.raises(ValueError):
        ke.asset.FixedFrameDesc("bad", body_index=0, quat_xyzw=[0, 0, 0, 0])
    data.add_fixed_frame(ke.asset.FixedFrameDesc("later", body_index=0))
    assert "later" not in [f.name for f in template.fixed_frames]


def test_g1_massless_feet_and_d6(tmp_path):
    path = Path(__file__).resolve().parents[2] / "assets/characters/g1/g1_29dof.urdf"
    if not path.exists():
        pytest.skip("G1 asset unavailable")
    data = ke.asset.URDFLoader.load(str(path))
    config = ke.physics.PhysicsConfig.z_up()
    world = ke.physics.PhysicsWorld(config)
    robot = ke.physics.Articulation.build(world, data)
    batch = None
    try:
        frames = {f.name: f for f in robot.template.fixed_frames}
        assert "left_foot_link" not in robot.body_names
        assert "left_foot_link" in robot.frame_names
        frame = frames["left_foot_link"]
        assert robot.body_names[frame.body_index] == "left_ankle_roll_link"
        assert robot.get_body_id("left_ankle_roll_link") == frame.body_index
        for lookup in (robot.get_body_id, robot.link):
            with pytest.raises(KeyError, match="fixed frame attached to 'left_ankle_roll_link'.*get_frame_pos"):
                lookup("left_foot_link")
        np.testing.assert_allclose([frame.pos.x, frame.pos.y, frame.pos.z], [.15, 0, 0], atol=1e-6)
        assert robot.num_dofs() == 29
        expected_mass = sum(i.mass for i in data.inertials.values())
        assert sum(robot.get_link_masses()) == pytest.approx(expected_mass)
        batch = ke.physics.D6Batch.create_world_frames(
            world, [robot], ["left_foot_link"], world_frames=[[0, 0, 0, 0, 0, 0, 1]])
        assert len(batch.joints) == 1
        np.testing.assert_allclose(batch.joints[0].frame1[:3], [.15, 0, 0], atol=1e-6)
        # The original source hierarchy is still available for retargeting.
        assert "left_foot_link" in data.skeleton_tree.node_names()
    finally:
        if batch is not None:
            batch.release()
        robot.release()


def test_body_and_frame_names_on_simulation_handles(tmp_path):
    from kangengine.sim.world import SimArticulationBatch

    data = chain(tmp_path)
    # A body and an authored frame may have the same name; body lookup wins.
    data.add_fixed_frame(ke.asset.FixedFrameDesc("arm", body_index=0))
    world = ke.sim.KangSimWorld(num_envs=1)
    try:
        record = world.add_articulation(data, env_id=0, obj_id=0)
        batch = SimArticulationBatch(world, obj_id=0, env_ids=(0,))
        for handle in (record, batch, record.articulation,
                       ke.physics.Articulation(record.articulation)):
            assert handle.body_names == ["root", "arm", "sibling"]
            assert set(handle.frame_names) == {"mount", "tip", "tip_end", "arm"}
            assert handle.get_body_id("arm") == 1
            with pytest.raises(KeyError, match="fixed frame attached to 'arm'.*get_frame_pos"):
                handle.get_body_id("tip")
            with pytest.raises(KeyError, match="body 'unknown' not found"):
                handle.get_body_id("unknown")
            names = handle.frame_names
            names.clear()
            assert "tip" in handle.frame_names
    finally:
        world.release()


def test_frame_axes_query_only_while_enabled(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from kangengine.visual.sim.world_visualizer import SimWorldVisualizer

    world = ke.sim.KangSimWorld(num_envs=1)
    draws, cleared, queries = [], [], []
    app = SimpleNamespace(
        get_native_scene=lambda: None,
        debug_overlay=SimpleNamespace(
            axes=lambda path, p, q, **kw: draws.append((path, p.copy(), q.copy())),
            clear=cleared.append,
        ),
    )
    visual = SimWorldVisualizer(app, world)
    try:
        world.add_articulation(chain(tmp_path), env_id=0, obj_id=0,
                               config=ke.physics.ArticulationConfig.fixed_base())
        world.state.refresh()
        original = world.state.get_frame_pos
        def read_frames(*args, **kwargs):
            queries.append(kwargs["frame_names"])
            return original(*args, **kwargs)
        monkeypatch.setattr(world.state, "get_frame_pos", read_frames)
        visual.sync()
        assert not queries
        visual.set_frames_visible(0, True, frame_names=["tip_end"])
        visual.sync()
        assert queries == [("tip_end",)]
        np.testing.assert_allclose(draws[0][1], [1, 2.2, 0], atol=2e-5)
        np.testing.assert_allclose(draws[0][2], [2**-.5, 0, 0, 2**-.5], atol=2e-5)
        visual.set_frames_visible(0, False)
        visual.sync()
        assert len(queries) == 1
        assert cleared == [draws[0][0]]
        visual.set_frames_visible(0, True, frame_names=["mount"])
        visual.sync()
        visual.release()
        assert cleared[-1] == draws[-1][0]
    finally:
        visual.release()
        world.release()
