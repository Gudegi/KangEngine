"""Rigid descriptions share appearance and colliders without skeleton metadata."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch

import kangengine as ke


@pytest.fixture(params=["cpu", "cuda"])
def world(request):
    pytest.importorskip("warp")
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    result = ke.sim.KangSimWorld(
        num_envs=3,
        sim_device=request.param,
        state_device=request.param,
        sim_dt=1.0 / 120.0,
        add_ground=False,
    )
    try:
        yield result
    finally:
        result.release()


def _plane(height=0.0):
    vertices = np.array(
        [[-2, -1, height], [2, -1, height], [2, 1, height], [-2, 1, height]],
        dtype=np.float32,
    )
    return ke.terrain.TerrainMesh(vertices, [[0, 1, 2], [0, 2, 3]])


def _batch(world, *, spacing=5.0, heights=(0.0, 0.75)):
    first, second = (_plane(height) for height in heights)
    # Render chunks must not change the rigid state's single-body topology.
    second = ke.terrain.TerrainMesh(
        second.vertices, second.indices, max_triangles_per_chunk=1
    )
    assets = tuple(ke.asset.rigid_from_terrain_mesh(mesh) for mesh in (first, second))
    cloner = ke.sim.GridCloner(world, spacing=spacing, columns=3)
    batch = cloner.add_rigid(
        [assets[0], assets[1], assets[0]], obj_id=1, kinematic=True
    )
    return assets, batch


def _initialize_gpu(world):
    if world.sim_device.type == "cuda":
        world.init_gpu_system()


def _box_collision(*, compound=False, convex=False):
    positions = (-0.5, 0.5) if compound else (0.0,)
    if convex:
        vertices = (
            np.array(
                [
                    [-1, -1, -1],
                    [1, -1, -1],
                    [1, 1, -1],
                    [-1, 1, -1],
                    [-1, -1, 1],
                    [1, -1, 1],
                    [1, 1, 1],
                    [-1, 1, 1],
                ],
                dtype=np.float32,
            )
            * 0.25
        )
        indices = np.array(
            [
                [0, 2, 1],
                [0, 3, 2],
                [4, 5, 6],
                [4, 6, 7],
                [0, 1, 5],
                [0, 5, 4],
                [3, 7, 6],
                [3, 6, 2],
                [0, 4, 7],
                [0, 7, 3],
                [1, 2, 6],
                [1, 6, 5],
            ],
            dtype=np.int32,
        )
        geometry = dict(
            type=ke.asset.CollisionGeomDescType.CONVEX_MESH,
            mesh=ke.scene.MeshData.from_arrays(vertices, indices),
        )
    else:
        geometry = dict(
            type=ke.asset.CollisionGeomDescType.BOX,
            size=np.array([0.25, 0.25, 0.25], dtype=np.float32),
        )

    shapes = [
        ke.asset.CollisionGeomDesc(
            **geometry,
            position=[x, 0.0, 0.0]
            if convex
            else np.array([x, 0.0, 0.0], dtype=np.float32),
            quat_xyzw=[0.0, 0.0, 0.0, 1.0],
        )
        for x in positions
    ]
    return shapes


def test_terrain_descriptor_shares_geometry_without_skeleton_metadata():
    plane = _plane()
    mesh = ke.terrain.TerrainMesh(
        plane.vertices, plane.indices, max_triangles_per_chunk=1
    )
    descriptor = ke.asset.rigid_from_terrain_mesh(mesh)
    assert isinstance(descriptor, ke.asset.RigidDesc)
    assert not hasattr(descriptor, "skeleton_tree")
    assert not hasattr(descriptor, "traversal_order")
    assert len(descriptor.visual_geoms) == 2
    assert all(
        visual.mesh_data is source
        for visual, source in zip(descriptor.visual_geoms, mesh._meshes)
    )

    # Render chunks stay shared while rigid instances use one combined shape.
    shape = descriptor.collision_geoms[0]
    assert len(descriptor.collision_geoms) == 1
    assert len(shape.mesh_data.indices) == 6


def test_instances_share_sources_but_keep_independent_actors(world):
    assets, batch = _batch(world)
    _initialize_gpu(world)

    first = world.get_rigid(env_id=0, obj_id=1)
    second = world.get_rigid(env_id=1, obj_id=1)
    third = world.get_rigid(env_id=2, obj_id=1)
    assert first.source_data is third.source_data is assets[0]
    assert second.source_data is assets[1]
    assert first.rigid is not third.rigid
    assert world.physics.num_body_actors() == 3
    assert world.physics.num_cached_triangle_meshes() == 2
    assert isinstance(batch, ke.sim.SimRigidBatch)
    assert batch.num_bodies == 1

    positions, quaternions = batch.get_root_pose()
    expected_positions = torch.tensor(
        [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [10.0, 0.0, 0.0]],
        device=positions.device,
    )
    expected_quaternions = torch.tensor(
        [[0.0, 0.0, 0.0, 1.0]], device=positions.device
    ).expand(3, -1)
    torch.testing.assert_close(positions, expected_positions)
    torch.testing.assert_close(quaternions, expected_quaternions)

    world.step(substeps=120, refresh=world.sim_device.type == "cpu")
    positions, quaternions = batch.get_root_pose()
    torch.testing.assert_close(positions, expected_positions)
    torch.testing.assert_close(quaternions, expected_quaternions)

    starts = expected_positions[:, None, :] + torch.tensor(
        [0.0, 0.0, 3.0], device=positions.device
    )
    direction = torch.tensor([0.0, 0.0, -1.0], device=positions.device)
    hits, distances = batch.raycast(starts, direction)
    torch.testing.assert_close(
        hits[:, 0, 2], torch.tensor([0.0, 0.75, 0.0], device=positions.device)
    )
    torch.testing.assert_close(
        distances[:, 0], torch.tensor([3.0, 2.25, 3.0], device=positions.device)
    )


def test_selected_reset_rotates_queries_in_world_space(world):
    _, batch = _batch(world)
    _initialize_gpu(world)
    device = world.sim_device
    original_positions, original_quaternions = (
        value.clone() for value in batch.get_root_pose()
    )
    target = torch.tensor([[3.0, -2.0, 1.5]], device=device)

    # Rotate the plane by 30 degrees about Y, then 90 degrees about world Z.
    sine, cosine = math.sin(math.pi / 12), math.cos(math.pi / 12)
    half = math.sqrt(0.5)
    quaternion = torch.tensor(
        [[-half * sine, half * sine, half * cosine, half * cosine]], device=device
    )
    reset_ids = (
        torch.tensor([1], dtype=torch.int32, device=device)
        if device.type == "cuda"
        else [1]
    )
    batch.set_root_state(reset_ids, target, quaternion, immediate=device.type == "cpu")
    if device.type == "cuda":
        world.step(substeps=0, refresh=False, apply_commands=False)
    positions, quaternions = batch.get_root_pose()
    torch.testing.assert_close(positions[[0, 2]], original_positions[[0, 2]])
    torch.testing.assert_close(quaternions[[0, 2]], original_quaternions[[0, 2]])
    torch.testing.assert_close(positions[1:2], target)
    torch.testing.assert_close(quaternions[1:2], quaternion)

    normal = torch.tensor([0.0, 0.5, math.sqrt(0.75)], device=device)
    expected_hit = target[0] + torch.tensor(
        [0.0, 0.5 * math.sqrt(0.75) + 0.375, -0.25 + 0.75 * math.sqrt(0.75)],
        device=device,
    )
    hits, distances = batch.raycast(
        (expected_hit + 2.0 * normal).reshape(1, 1, 3),
        -4.0 * normal,
        env_ids=[1],
    )
    torch.testing.assert_close(hits[0, 0], expected_hit, atol=1e-5, rtol=0)
    torch.testing.assert_close(
        distances, torch.tensor([[2.0]], device=device), atol=1e-5, rtol=0
    )

    heights = batch.sample_height(target[:, None, :], env_ids=[1], ray_start_height=5.0)
    torch.testing.assert_close(
        heights,
        torch.tensor([[1.5 + 0.75 / math.sqrt(0.75)]], device=device),
        atol=1e-5,
        rtol=0,
    )
    torch.testing.assert_close(
        batch.sample_height(target[:, None, :], env_ids=[1]), heights
    )
    unchanged = batch.sample_height(
        original_positions[[2, 0], None, :], env_ids=[2, 0], ray_start_height=5.0
    )
    torch.testing.assert_close(unchanged, torch.zeros((2, 1), device=device))

    missed_hits, missed_distances = batch.raycast(
        (expected_hit + 2.0 * normal).reshape(1, 1, 3),
        -normal,
        env_ids=[1],
        max_distance=1.0,
    )
    assert missed_hits.isposinf().all()
    assert missed_distances.isposinf().all()


def test_overlapping_environments_only_collide_with_their_rigid_asset(world):
    _, batch = _batch(world, spacing=0.0, heights=(0.0, 1.0))
    ball_path = Path(ke.__file__).parent / "assets" / "objects" / "ball.xml"
    sphere_data = world.load_mjcf(str(ball_path))
    spheres = ke.sim.GridCloner(world, spacing=0.0).add_rigid(
        sphere_data, obj_id=2, initial_root_pos=(0.0, 0.0, 2.5)
    )
    _initialize_gpu(world)

    world.step(substeps=360, refresh=True)

    torch.testing.assert_close(
        spheres.get_root_pos()[:, 2],
        torch.tensor([0.12, 1.12, 0.12], device=world.device),
        atol=0.02,
        rtol=0,
    )
    positions, _ = batch.get_root_pose()
    torch.testing.assert_close(positions, torch.zeros_like(positions))


def test_triangle_mesh_rejects_dynamic_mode_before_registering_actor():
    world = ke.sim.KangSimWorld(num_envs=1, state_device="cpu", add_ground=False)
    try:
        with pytest.raises(ValueError, match="kinematic"):
            asset = ke.asset.rigid_from_terrain_mesh(_plane())
            world.add_rigid(asset, kinematic=False)
        assert world.physics.num_body_actors() == 0
        assert not world.rigids
    finally:
        world.release()


@pytest.mark.parametrize("collider", ["box", "box_compound", "convex_compound"])
def test_dynamic_collision_override_keeps_queries_on_authored_surface(world, collider):
    surface = _plane(height=1.25)
    compound = collider != "box"
    colliders = _box_collision(compound=compound, convex=collider == "convex_compound")
    asset = ke.asset.rigid_from_terrain_mesh(surface, collision_geoms=colliders)
    cloner = ke.sim.GridCloner(world, spacing=5.0, columns=3)
    cloner.add_rigid(ke.asset.rigid_from_terrain_mesh(_plane()), obj_id=1, static=True)
    boxes = cloner.add_rigid(
        asset, obj_id=2, initial_root_pos=(0.0, 0.0, 2.5), density=600.0
    )
    _initialize_gpu(world)

    assert len(asset.collision_geoms) == len(colliders)
    assert all(record.source_data is asset for record in boxes.records)
    assert boxes.num_bodies == 1
    assert boxes.num_collision_shapes == (2 if compound else 1)

    # Collision follows the authored box colliders, not the elevated visual plane.
    world.step(substeps=360, refresh=True)
    positions, _ = boxes.get_root_pose()
    torch.testing.assert_close(
        positions[:, 2],
        torch.full((3,), 0.25, device=positions.device),
        atol=0.02,
        rtol=0,
    )
    heights = boxes.sample_height(positions[:, None, :])
    torch.testing.assert_close(heights[:, 0], positions[:, 2] + 1.25, atol=0.02, rtol=0)

    # Single handles expose the same surface query without a separate asset batch.
    single_heights = boxes.first.sample_height(positions[0:1, None, :])
    torch.testing.assert_close(single_heights, heights[0:1], atol=1e-5, rtol=0)


def test_static_instances_query_current_world_pose(world):
    asset = ke.asset.rigid_from_terrain_mesh(_plane(height=0.75))
    cloner = ke.sim.GridCloner(world, spacing=5.0, columns=3)
    instances = cloner.add_rigid(
        asset, obj_id=1, static=True, initial_root_pos=(0.0, 0.0, 1.0)
    )
    _initialize_gpu(world)

    assert world.physics.num_cached_triangle_meshes() == 1
    assert all(record.source_data is asset for record in instances.records)
    positions, quaternions = instances.get_root_pose()
    expected_positions = torch.tensor(
        [[0.0, 0.0, 1.0], [5.0, 0.0, 1.0], [10.0, 0.0, 1.0]],
        device=world.device,
    )
    torch.testing.assert_close(positions, expected_positions)
    torch.testing.assert_close(
        quaternions,
        torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=world.device).expand(3, 4),
    )
    heights = instances.sample_height(positions[[2, 0], None, :], env_ids=[2, 0])
    torch.testing.assert_close(heights, torch.full((2, 1), 1.75, device=world.device))
    single_positions, _ = instances.first.get_root_pose()
    torch.testing.assert_close(single_positions, positions[0:1])


def test_selected_environments_keep_their_source_assignment(world):
    low = ke.asset.rigid_from_terrain_mesh(_plane(height=0.25))
    high = ke.asset.rigid_from_terrain_mesh(_plane(height=1.0))
    cloner = ke.sim.GridCloner(world, spacing=5.0, columns=3)
    instances = cloner.add_rigid([high, low], obj_id=1, env_ids=[2, 0], kinematic=True)
    _initialize_gpu(world)

    positions, _ = instances.get_root_pose()
    torch.testing.assert_close(
        instances.sample_height(positions[:, None, :]),
        torch.tensor([[1.0], [0.25]], device=world.device),
    )
    torch.testing.assert_close(
        instances.sample_height(positions[[1, 0], None, :], env_ids=[0, 2]),
        torch.tensor([[0.25], [1.0]], device=world.device),
    )


@pytest.mark.parametrize("sequence", [list, tuple])
def test_mesh_descriptor_accepts_sequence_local_pose(sequence):
    source = ke.asset.rigid_from_terrain_mesh(_plane())
    local_position = [1.0, -2.0, 0.5]
    local_quaternion = [0.0, math.sqrt(0.5), 0.0, math.sqrt(0.5)]
    shape = ke.asset.CollisionGeomDesc(
        type=ke.asset.CollisionGeomDescType.TRIANGLE_MESH,
        mesh=source.collision_geoms[0].mesh_data,
        position=sequence(local_position),
        quat_xyzw=sequence(local_quaternion),
    )
    descriptor = ke.asset.RigidDesc(collision_geoms=[shape])
    world = ke.sim.KangSimWorld(num_envs=1, state_device="cpu", add_ground=False)
    try:
        instance = world.add_rigid(descriptor, kinematic=True)
        np.testing.assert_allclose(instance.data.body_pos[0], (0, 0, 0), atol=1e-6)
        np.testing.assert_allclose(
            instance.data.body_quat_xyzw[0], (0, 0, 0, 1), atol=1e-6
        )
        pose = instance.get_collision_shape_pose()
        np.testing.assert_allclose(pose[0, :3], local_position, atol=1e-6)
        np.testing.assert_allclose(pose[0, 3:], local_quaternion, atol=1e-6)
    finally:
        world.release()


def test_obj_keeps_uvs_material_subsets_and_collision_coordinates(tmp_path):
    (tmp_path / "floor.mtl").write_text(
        "newmtl Brick\nKd 0.8 0.3 0.1\nmap_Kd textures/brick.png\n"
        "newmtl Paint\nKd 0.1 0.2 0.9\n",
        encoding="utf-8",
    )
    obj_path = tmp_path / "floor.obj"
    obj_path.write_text(
        "mtllib floor.mtl\n"
        "v -2 -1 0.75\nv 2 -1 0.75\nv 2 1 0.75\nv -2 1 0.75\n"
        "vt 0 0\nvt 1 0\nvt 1 1\nvt 0 1\nvn 0 0 1\n"
        "usemtl Brick\nf 1/1/1 2/2/1 3/3/1\n"
        "usemtl Paint\nf 1/1/1 3/3/1 4/4/1\n",
        encoding="utf-8",
    )

    descriptor = ke.asset.load_obj_rigid(obj_path)
    assert descriptor.asset_dir == str(tmp_path)
    assert len(descriptor.visual_geoms) == 2
    brick, paint = descriptor.visual_geoms
    assert isinstance(brick.material, ke.material.NativePhongMaterial)
    assert brick.material.texture_paths["diffuse_map"] == str(
        tmp_path / "textures/brick.png"
    )
    assert brick.material.diffuse_map is None
    np.testing.assert_allclose(paint.material.diffuse, [0.1, 0.2, 0.9])
    assert all(
        len(part.mesh_data.uvs) == len(part.mesh_data.vertices)
        for part in descriptor.visual_geoms
    )
    assert {
        (uv.x, uv.y) for part in descriptor.visual_geoms for uv in part.mesh_data.uvs
    } == {
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 1),
    }
    assert sum(len(part.mesh_data.indices) for part in descriptor.visual_geoms) == 6
    shape = descriptor.collision_geoms[0]
    assert shape.type == ke.asset.CollisionGeomDescType.TRIANGLE_MESH
    np.testing.assert_allclose([vertex.z for vertex in shape.mesh_data.vertices], 0.75)

    dynamic = ke.asset.load_obj_rigid(
        obj_path, collision_geoms=_box_collision(compound=True)
    )
    assert len(dynamic.collision_geoms) == 2
    assert all(
        geom.type == ke.asset.CollisionGeomDescType.BOX
        for geom in dynamic.collision_geoms
    )
    assert len(dynamic.visual_geoms) == 2
    np.testing.assert_allclose(
        [
            vertex.z
            for part in dynamic.visual_geoms
            for vertex in part.mesh_data.vertices
        ],
        0.75,
    )


def test_authored_mass_properties_override_density(world):
    descriptor = ke.asset.RigidDesc(
        name="weighted_box",
        collision_geoms=_box_collision(),
        inertial=ke.asset.InertialDesc(
            mass=7.5,
            diag_inertia=[1.0, 1.5, 2.0],
            com=[0.1, 0.0, 0.0],
            quat_xyzw=[0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
        ),
    )
    batch = ke.sim.GridCloner(world).add_rigid(descriptor, density=600.0)
    # Read authored native properties before Direct GPU initialization.
    for record in batch.records:
        assert record.rigid.get_mass() == pytest.approx(7.5)
        np.testing.assert_allclose(
            record.rigid.get_mass_space_inertia(), [1.0, 1.5, 2.0]
        )
        np.testing.assert_allclose(
            record.rigid.get_local_com_position(), [0.1, 0.0, 0.0]
        )
        np.testing.assert_allclose(
            record.rigid.get_local_com_quat_xyzw(),
            [0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)],
            atol=1e-6,
        )


def test_visual_local_pose_and_scale_affect_surface_queries(world):
    source = ke.asset.rigid_from_terrain_mesh(_plane())
    half = math.sqrt(0.5)
    descriptor = ke.asset.RigidDesc(
        visual_geoms=[
            ke.asset.VisualGeomDesc(
                mesh_data=source.visual_geoms[0].mesh_data,
                position=[1.0, -2.0, 0.75],
                quat_xyzw=[0.0, 0.0, half, half],
                scale=[2.0, 1.0, 1.0],
            )
        ],
        collision_geoms=_box_collision(),
    )
    cloner = ke.sim.GridCloner(world, spacing=10.0)
    batch = cloner.add_rigid(
        descriptor, initial_root_pos=(1.0, 0.0, 1.0), kinematic=True
    )
    _initialize_gpu(world)
    starts = cloner.env_origins.to(world.device)[:, None, :] + torch.tensor(
        [[[2.0, 1.0, 3.0], [4.0, -2.0, 3.0]]], device=world.device
    )
    hits, _ = batch.raycast(starts, torch.tensor([0.0, 0.0, -1.0], device=world.device))
    torch.testing.assert_close(
        hits[:, 0, 2], torch.full((3,), 1.75, device=world.device)
    )
    assert hits[:, 1].isposinf().all()


def test_explicit_rigid_conversion_keeps_shared_visual_and_mass():
    source = ke.asset.rigid_from_terrain_mesh(_plane())
    tree = ke.animation.SkeletonTree(
        ["prop"], [-1], np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0, 0.0]])
    )
    inertial = ke.asset.InertialDesc(
        mass=2.0, diag_inertia=[0.2, 0.3, 0.4], com=[0.0, 0.0, 0.0]
    )
    articulated = ke.asset.ArticulationDesc(
        tree, joints={}, collision_geoms={0: _box_collision()}, inertials={0: inertial}
    )
    articulated.visual_geoms = source.visual_geoms
    rigid = ke.asset.RigidDesc.from_articulation(articulated)
    assert not hasattr(rigid, "skeleton_tree")
    assert rigid.visual_geoms[0].mesh_data is source.visual_geoms[0].mesh_data
    assert len(rigid.collision_geoms) == 1
    assert rigid.inertial.mass == pytest.approx(2.0)
