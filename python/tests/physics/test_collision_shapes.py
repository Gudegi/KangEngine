"""Physical body state stays separate from per-instance collision shape poses."""

import math

import pytest
import torch

import kangengine as ke
from kangengine.utils.batched_rotations import quat_xyzw_rotate


@pytest.fixture(params=["cpu", "cuda"])
def world(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    result = ke.sim.KangSimWorld(
        num_envs=2, sim_device=request.param, state_device="cpu", add_ground=False,
    )
    try:
        yield result
    finally:
        result.release()


def rigid_source(name="cup", offset=1.0):
    geoms = []
    for x in (0.0, offset):
        geom = ke.asset.CollisionGeomDesc(
            type=ke.asset.CollisionGeomDescType.BOX, size=(0.2, 0.2, 0.2),
            position=(x, 0, 0), quat_xyzw=(0, 0, 0, 1),
        )
        geoms.append(geom)
    return ke.asset.RigidDesc(name=name, collision_geoms=geoms)


def initialize(world):
    if world.sim_device.type == "cuda":
        world.init_gpu_system()


@pytest.mark.parametrize("mode", ["dynamic", "kinematic", "static"])
def test_rigid_body_and_shape_slots(world, mode, monkeypatch):
    sources = [rigid_source(), rigid_source("box", 2.0)]
    batch = ke.sim.GridCloner(world, spacing=5).add_rigid(
        sources, static=mode == "static", kinematic=mode == "kinematic",
    )
    assert batch.num_bodies == 1
    assert batch.num_collision_shapes == 2
    assert [record.body_names for record in batch.records] == [["cup"], ["box"]]
    assert batch.collision_shape_names == [["cup:geom_0", "cup:geom_1"], ["box:geom_0", "box:geom_1"]]
    assert batch.collision_shape_body_ids.tolist() == [[0, 0], [0, 0]]
    assert batch.collision_shape_geom_ids.tolist() == [[0, 1], [0, 1]]

    # Rotate only one actor; the collider offset must rotate with it.
    device = world.sim_device
    pos = torch.tensor([[3., 4., 5.]], device=device)
    quat = torch.tensor([[0., 0., math.sqrt(.5), math.sqrt(.5)]], device=device)
    if mode == "static":
        batch.records[1].rigid.set_root_pose(pos[0].cpu().tolist(), quat[0].cpu().tolist())
    initialize(world)
    cuda_reset = device.type == "cuda" and mode != "static"
    if mode != "static":
        batch.set_root_state(
            [1], pos if cuda_reset else pos.cpu(), quat if cuda_reset else quat.cpu(),
            immediate=not cuda_reset,
        )
    if cuda_reset:
        world.apply_resets()

    def reject_download(*args, **kwargs):
        raise AssertionError("shape query downloaded CUDA data")

    with monkeypatch.context() as guard:
        if device.type == "cuda":
            guard.setattr(torch.Tensor, "cpu", reject_download)
            guard.setattr(torch.Tensor, "numpy", reject_download)
        poses = batch.get_collision_shape_pose(env_ids=[1])
        single = batch.records[1].get_collision_shape_pose()
    torch.testing.assert_close(poses[0, :, :3].cpu(), torch.tensor([[3., 4., 5.], [3., 6., 5.]]))
    torch.testing.assert_close(single, poses[0])
    assert batch.get_collision_shape_pose([]).shape == (0, 2, 7)
    subset = world.get_rigid_batch(env_ids=[1], obj_id=0)
    torch.testing.assert_close(subset.get_collision_shape_pose(), poses)
    with pytest.raises(KeyError, match="outside this view"):
        subset.get_collision_shape_pose([0])

    if mode != "static":
        world.refresh()
        state = batch.data
        assert state.body_pos.shape == (2, 1, 3)
        torch.testing.assert_close(state.body_pos[:, 0], state.root_pos)
        torch.testing.assert_close(state.body_quat_xyzw[:, 0], state.root_quat_xyzw)
        assert world.body_forces[(0, 0)].shape == (1, 3)
        if device.type == "cuda":
            assert batch.add_contact_sensor().body_ids == (0,)
            with pytest.raises(IndexError):
                batch.add_contact_sensor(body_ids=[1], name="invalid")
    assert all(record.source_data is source for record, source in zip(batch.records, sources))


def articulation_source(counts):
    builder = ke.physics.ArticulationBuilder()
    shape = ke.physics.ArticulationBuilder.BoxShape((.1, .1, .1), position=(1, 0, 0))
    builder.add_link("root", inertia_diagonal=(1, 1, 1), shapes=[shape] * counts[0])
    builder.add_link("arm", parent="root", position=(0, 2, 0),
                     inertia_diagonal=(1, 1, 1), shapes=[shape] * counts[1],
                     joint=ke.physics.ArticulationBuilder.RevoluteJoint())
    return builder.build()


def test_articulation_environment_specific_mapping_and_joint_pose(world, monkeypatch):
    for env_id, counts in enumerate(((2, 1), (1, 2))):
        record = world.add_articulation(
            articulation_source(counts), env_id=env_id,
            config=ke.physics.ArticulationConfig.fixed_base(),
        )
        record.articulation.set_dof_state([math.pi / 2], [0.])
    batch = world.get_articulation_batch(env_ids=[1, 0])
    initialize(world)
    assert batch.num_bodies == 2
    assert batch.num_collision_shapes == 3
    assert batch.collision_shape_body_ids.tolist() == [[0, 1, 1], [0, 0, 1]]
    assert batch.collision_shape_geom_ids.tolist() == [[0, 0, 1], [0, 1, 0]]

    with monkeypatch.context() as guard:
        if world.sim_device.type == "cuda":
            def reject_download(*args, **kwargs):
                raise AssertionError("shape query downloaded CUDA data")
            guard.setattr(torch.Tensor, "cpu", reject_download)
            guard.setattr(torch.Tensor, "numpy", reject_download)
        poses = batch.get_collision_shape_pose()
        if world.sim_device.type == "cuda":
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                streamed = batch.get_collision_shape_pose()
                # A second fetch must not overwrite the first result.
                reordered = batch.get_collision_shape_pose([0, 1])
            torch.cuda.current_stream().wait_stream(stream)
            torch.testing.assert_close(streamed, poses)
            torch.testing.assert_close(reordered, poses.flip(0))
    expected = torch.tensor([[[1., 0., 0.], [0., 3., 0.], [0., 3., 0.]],
                             [[1., 0., 0.], [1., 0., 0.], [0., 3., 0.]]])
    torch.testing.assert_close(poses[..., :3].cpu(), expected, atol=1e-6, rtol=0)
    torch.testing.assert_close(batch.first.get_collision_shape_pose(), poses[0])
    assert batch.first.get_collision_shape_pose(env_ids=[1]).shape == (1, 3, 7)
    assert batch.get_collision_shape_pose(env_ids=[]).shape == (0, 3, 7)
    copied_ids = batch.collision_shape_body_ids
    copied_ids.zero_()
    assert batch.collision_shape_body_ids.tolist() == [[0, 1, 1], [0, 0, 1]]


@pytest.mark.parametrize("static", [False, True])
def test_topology_mismatch_rejected_before_registration(world, static):
    add = world.add_static_rigid if static else world.add_rigid
    add(rigid_source(), env_id=0)
    with pytest.raises(ValueError, match="collision shapes"):
        add(ke.asset.RigidDesc(), env_id=1)
    with pytest.raises(KeyError):
        world.get_rigid(env_id=1)
    # A rejected actor must not occupy the registration slot.
    add(rigid_source(), env_id=1)
    assert world.get_rigid_batch().num_collision_shapes == 2


def test_generated_colliders_have_no_authored_geom(world):
    rigid = world.add_rigid(ke.asset.RigidDesc(), obj_id=0)
    builder = ke.physics.ArticulationBuilder()
    builder.add_link("root", inertia_diagonal=(1, 1, 1))
    source = builder.build()
    robot = world.add_articulation(ke.asset.ArticulationDesc(
        source.skeleton_tree, joints={}, inertials=source.inertials,
        collision_geoms={0: []},
    ), obj_id=1)
    no_collision = world.add_articulation(source, obj_id=2)
    assert rigid.collision_shape_geom_ids.tolist() == [-1]
    assert robot.collision_shape_geom_ids.tolist() == [-1]
    assert rigid.collision_shape_body_ids.tolist() == [0]
    assert robot.collision_shape_body_ids.tolist() == [0]
    assert no_collision.num_collision_shapes == 0
    initialize(world)
    assert no_collision.get_collision_shape_pose().shape == (0, 7)
    assert no_collision.collision_shape_geom_ids.shape == (0,)


@pytest.mark.parametrize("from_to", [False, True])
def test_capsule_query_uses_physx_shape_frame(world, tmp_path, from_to):
    pose = 'fromto="0 0 1.6 0 0 2.4"' if from_to else 'pos="0 0 2"'
    mjcf_path = tmp_path / "capsule.xml"
    mjcf_path.write_text(
        f'<mujoco><worldbody><body name="rod"><geom name="shaft" '
        f'type="capsule" size=".1 .4" {pose}/></body></worldbody></mujoco>'
    )
    record = world.add_rigid(world.load_mjcf(str(mjcf_path)), kinematic=True)
    assert record.collision_shape_names == ["rod:shaft"]
    initialize(world)
    pose = record.get_collision_shape_pose().cpu()
    torch.testing.assert_close(pose[0, :3], torch.tensor([0., 0., 2.]))
    # PhysX capsules extend along local X; the authored capsule extends along Z.
    axis = quat_xyzw_rotate(pose[0, 3:], torch.tensor([1., 0., 0.]))
    torch.testing.assert_close(axis, torch.tensor([0., 0., 1.]), atol=1e-6, rtol=0)


@pytest.mark.parametrize("static", [False, True])
def test_mimickit_compound_body_state_matches_actor(tmp_path, static):
    from kangengine.mimickit_engine import KangEngineEngine, MimicObjType

    mjcf_path = tmp_path / "compound.xml"
    mjcf_path.write_text(
        '<mujoco><worldbody><body name="cup">'
        '<geom type="box" size=".2 .2 .2" pos="-1 0 0"/>'
        '<geom type="box" size=".2 .2 .2" pos="1 0 0"/>'
        '</body></worldbody></mujoco>'
    )
    engine = KangEngineEngine(
        {"sim_device": "cpu", "add_ground": False, "control_mode": "none"},
        num_envs=1, device="cpu", visualize=False,
    )
    try:
        engine.create_env()
        engine.create_obj(0, MimicObjType.rigid, str(mjcf_path), "cup",
                          fix_root=static, start_pos=[2., 3., 4.])
        engine.initialize_sim()
        assert engine.get_obj_num_bodies(0) == 1
        assert engine.get_obj_body_names(0) == ["cup"]
        torch.testing.assert_close(engine.get_body_pos(0)[:, 0], engine.get_root_pos(0))
        torch.testing.assert_close(engine.get_body_rot(0)[:, 0], engine.get_root_rot(0))
    finally:
        engine.release()
