"""Nontrivial poses survive physics, animation, and scene API boundaries."""

import numpy as np
import pytest
import torch

import kangengine as ke
from kangengine.state import KangWorldState
from kangengine.utils import batched_rotations


@pytest.fixture
def articulation_data():
    builder = ke.physics.ArticulationBuilder()
    builder.add_link("root", inertia_diagonal=(1, 1, 1))
    builder.add_link(
        "child", parent="root", position=(1, 0, 0),
        inertia_diagonal=(1, 1, 1),
        joint=ke.physics.ArticulationBuilder.RevoluteJoint(),
    )
    return builder.build()


@pytest.mark.parametrize("input_kind", ["list", "numpy", "torch", "quat"])
def test_physics_root_and_snapshot_use_xyzw(articulation_data, input_kind):
    quaternion = np.array([1, 2, 3, 4], dtype=np.float32)
    quaternion /= np.linalg.norm(quaternion)
    value = {
        "list": quaternion.tolist(), "numpy": quaternion,
        "torch": torch.from_numpy(quaternion),
        "quat": ke.Quat.from_xyzw(quaternion),
    }[input_kind]
    world = ke.physics.PhysicsWorld(ke.physics.PhysicsConfig.z_up())
    robot = ke.physics.Articulation.build(world, articulation_data)
    state = KangWorldState(num_envs=1, device="cpu")
    try:
        robot.reset_root(pos=(2, 3, 4), quat_xyzw=value)
        np.testing.assert_allclose(robot.get_root_quat_xyzw(), quaternion, atol=1e-6)
        np.testing.assert_allclose(
            robot.get_link_quats_xyzw().reshape(-1, 4)[0], quaternion, atol=1e-6,
        )
        state.add_articulation(robot, env_id=0, obj_id=0, physics=world)
        state.refresh()
        np.testing.assert_allclose(
            state.get_root_quat_xyzw(0)[0], quaternion, atol=1e-6,
        )
        np.testing.assert_allclose(
            state.get_body_quat_xyzw(0)[0, 0], quaternion, atol=1e-6,
        )
        robot.reset_root()
        np.testing.assert_allclose(robot.get_root_quat_xyzw(), [0, 0, 0, 1], atol=1e-6)
    finally:
        state.release()
        robot.release()


def test_visual_joint_wxyz_composes_parent_pose(articulation_data):
    asset = ke.visual.ArticulationVisualAsset.from_data(articulation_data)
    scene = ke.scene.create_backend(ke.scene.BackendType.NATIVE)
    visual = asset.instantiate(scene, "/robot", "/.Resources/test", True, hierarchical=True)
    half = 2 ** -0.5
    visual.set_joint_quat_wxyz(index=0, quat_wxyz=[half, 0, 0, half])
    visual.set_joint_quat_wxyz(index=1, quat_wxyz=np.array([half, half, 0, 0]))
    visual.apply_pose()
    matrix = np.asarray(visual.body_prim(1).compute_world_matrix())
    np.testing.assert_allclose(matrix[:3, 3], [0, 1, 0], atol=1e-6)
    np.testing.assert_allclose(
        matrix[:3, :3], [[0, 0, 1], [1, 0, 0], [0, 1, 0]], atol=1e-6,
    )


def test_cloned_simulation_exposes_xyzw_fields(articulation_data):
    world = ke.sim.KangSimWorld(num_envs=2, state_device="cpu")
    quaternion = np.array([0, 0, 2 ** -0.5, 2 ** -0.5], dtype=np.float32)
    try:
        cloner = ke.sim.GridCloner(world)
        robots = cloner.add_articulation(
            articulation_data, initial_root_quat_xyzw=quaternion,
        )
        world.state.refresh()
        np.testing.assert_allclose(
            robots.get_root_quat_xyzw(), np.tile(quaternion, (2, 1)), atol=1e-6,
        )
        np.testing.assert_allclose(robots.first.data.root_quat_xyzw, quaternion, atol=1e-6)
        np.testing.assert_allclose(robots.first.data.body_quat_xyzw[0], quaternion, atol=1e-6)
    finally:
        world.release()


@pytest.mark.parametrize("component", [False, True])
def test_scene_local_and_world_quaternion_frames(component):
    scene = ke.scene.create_backend(ke.scene.BackendType.NATIVE)
    parent = scene.define_prim("/parent", ke.scene.PrimType.XFORM)
    child = scene.define_prim("/parent/child", ke.scene.PrimType.XFORM)
    half = 2 ** -0.5
    parent.set_local_quat_wxyz(quat_wxyz=[half, 0, 0, half])
    target = child.get_transform_component() if component else child
    target.set_local_quat_wxyz(quat_wxyz=np.array([half, half, 0, 0]))
    np.testing.assert_allclose(target.get_world_quat().to_wxyz(), [.5, .5, .5, .5], atol=1e-6)

    target.set_world_quat_wxyz(quat_wxyz=ke.Quat(1, 0, 0, 0))
    np.testing.assert_allclose(target.get_world_quat().to_wxyz(), [1, 0, 0, 0], atol=1e-6)
    np.testing.assert_allclose(target.get_local_quat().to_wxyz(), [half, 0, 0, -half], atol=1e-6)
    assert not hasattr(target, "set_local_rotation")
    assert not hasattr(child, "add_rotate_quaternion_op")


def test_tensor_order_conversion_keeps_dtype_device_and_gradients():
    quaternion = torch.tensor([[1., 2., 3., 4.]], dtype=torch.float64, requires_grad=True)
    converted = batched_rotations.quat_wxyz_to_xyzw(quaternion)
    assert converted.device == quaternion.device
    assert converted.dtype == quaternion.dtype
    torch.testing.assert_close(converted, torch.tensor([[2., 3., 4., 1.]], dtype=torch.float64))
    torch.testing.assert_close(batched_rotations.quat_xyzw_to_wxyz(converted), quaternion)
    (converted * torch.tensor([10., 20., 30., 40.])).sum().backward()
    torch.testing.assert_close(quaternion.grad, torch.tensor([[40., 10., 20., 30.]], dtype=torch.float64))
