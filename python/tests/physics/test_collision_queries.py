"""Prescribed-pose collision queries are independent of simulation state."""

import numpy as np
import pytest

import kangengine as ke


@pytest.fixture
def scene():
    builder = ke.physics.ArticulationBuilder()
    for name, parent in (("root", None), ("arm", "root"), ("tip", "arm")):
        builder.add_link(name, parent=parent, position=(0, 0, 3),
                         inertia_diagonal=(1, 1, 1),
                         shapes=[builder.SphereShape(.2)],
                         joint=builder.RevoluteJoint() if parent else None)
    world = ke.physics.PhysicsWorld()
    robot = ke.physics.Articulation.build(world, builder.build())
    yield world, robot
    robot.release()


def poses(xs):
    result = np.zeros((len(xs), 7), np.float32)
    result[:, 0] = xs
    result[:, 6] = 1
    return result


def test_self_pairs_and_metadata(scene):
    world, robot = scene
    geometry = world.get_collision_geometry(robot)
    assert geometry["body_names"] == ["root", "arm", "tip"]
    assert len(geometry["robot"]) == 3
    assert all(s["type"] == "sphere" and s["radius"] == pytest.approx(.2) for s in geometry["robot"])
    pose = poses([0, .15, .3])
    rows = world.query_collisions(robot, pose)
    assert len(rows) == 1
    assert (rows[0]["body_a"], rows[0]["body_b"]) == (0, 2)
    assert rows[0]["depth"] == pytest.approx(.1, abs=1e-6)
    np.testing.assert_allclose(rows[0]["direction"], [-1, 0, 0], atol=1e-6)
    separated = pose.copy()
    separated[0, :3] += np.asarray(rows[0]["direction"]) * (rows[0]["depth"] + 1e-4)
    assert world.query_collisions(robot, separated) == []
    assert len(world.query_collisions(robot, pose, exclude_adjacent=False)) == 3
    assert world.query_collisions(robot, pose, body_pairs=[]) == []
    assert len(world.query_collisions(robot, pose, body_pairs=[(1, 0)])) == 1
    assert world.query_collisions(robot, pose, excluded_body_pairs=[(2, 0)]) == []
    assert world.query_collisions(robot, pose, self_collision=False) == []


def test_batch_environment_and_no_state_mutation(scene):
    world, robot = scene
    wall = world.add_static_box([.1, 2, 2], [1, 0, 0], register_as_ground=False)
    before = np.asarray(robot.get_link_positions()).copy()
    batch = np.stack([poses([4, 6, 8]), poses([.85, 6, 8])])
    rows = world.query_collisions(robot, batch, environment=[wall], self_collision=False)
    assert len(rows) == 1 and rows[0]["frame_idx"] == 1
    assert rows[0]["environment_idx"] == 0 and rows[0]["body_b"] == -1
    assert rows[0]["depth"] == pytest.approx(.15, abs=1e-5)
    np.testing.assert_allclose(rows[0]["direction"], [-1, 0, 0], atol=1e-5)
    assert rows[0]["status"] == "overlap_mtd"
    assert world.query_collisions(robot, batch, environment=[wall], self_collision=False,
                                  excluded_environment_pairs=[(0, 0)]) == []
    np.testing.assert_array_equal(robot.get_link_positions(), before)
    assert world.num_contacts() == 0
    assert world.query_collisions(robot, np.empty((0, 3, 7), np.float32)) == []
    singles = world.query_collisions(robot, batch[1], environment=[wall], self_collision=False)
    assert singles[0]["depth"] == rows[0]["depth"]


def test_batch_reuses_shapes_across_self_and_environment_pairs(scene):
    world, robot = scene
    walls = [world.add_static_box([.1, 2, 2], [x, 0, 0], register_as_ground=False)
             for x in (.05, .25)]
    batch = np.stack([poses([0, .15, .3]), poses([4, 6, 8]), poses([.1, .25, .4])])
    kwargs = {"environment": walls, "exclude_adjacent": False}
    rows = world.query_collisions(robot, batch, **kwargs)
    expected = []
    for frame_idx, pose in enumerate(batch):
        for row in world.query_collisions(robot, pose, **kwargs):
            expected.append(dict(row, frame_idx=frame_idx))
    assert rows == expected
    assert len([row for row in rows if row["frame_idx"] == 0]) == 9
    assert not any(row["frame_idx"] == 1 for row in rows)
    assert {row["environment_idx"] for row in rows} == {-1, 0, 1}


def test_triangle_wall_mtd_and_mesh_metadata(scene):
    world, robot = scene
    wall = world.add_triangle_mesh(ke.geometry.create_box_data(.2, 4, 4),
                                   position=(1, 0, 0), register_as_ground=False)
    geometry = world.get_collision_geometry(robot, environment=[wall], include_mesh_data=True)
    shape = geometry["environment"][0]
    assert shape["type"] == "triangle_mesh"
    assert len(shape["triangles"]) == 12
    assert np.isfinite(shape["vertices"]).all()
    rows = world.query_collisions(robot, poses([.85, 6, 8]), environment=[wall], self_collision=False)
    assert rows and rows[0]["status"] in ("overlap_mtd_approximate", "overlap_depth_unavailable")
    if rows[0]["status"] == "overlap_mtd_approximate":
        assert rows[0]["depth"] > 0
        assert np.isfinite(rows[0]["direction"]).all()
    assert world.query_collisions(robot, poses([.4, 6, 8]), environment=[wall], self_collision=False) == []


def test_local_shape_offset_rotates_with_link():
    builder = ke.physics.ArticulationBuilder()
    builder.add_link("root", inertia_diagonal=(1, 1, 1),
                     shapes=[builder.SphereShape(.1, position=(1, 0, 0))])
    world = ke.physics.PhysicsWorld()
    robot = ke.physics.Articulation.build(world, builder.build())
    wall = world.add_static_box([.1, .1, .1], [0, 1, 0])
    pose = poses([0])
    assert world.query_collisions(robot, pose, environment=[wall]) == []
    pose[0, 3:] = [0, 0, np.sqrt(.5), np.sqrt(.5)]
    assert len(world.query_collisions(robot, pose, environment=[wall])) == 1
    robot.release()


def test_convex_environment_geometry_and_mtd(scene):
    world, robot = scene
    part = ke.physics.ConvexMeshPart.from_mesh_data(ke.geometry.create_box_data(.2, 4, 4))
    wall = world.create_static_convex_compound([part], pos=[1, 0, 0])
    shape = world.get_collision_geometry(robot, environment=[wall], include_mesh_data=True)["environment"][0]
    assert shape["type"] == "convex_mesh"
    assert len(shape["vertices"]) == 8 and len(shape["triangles"]) == 12
    rows = world.query_collisions(robot, poses([.85, 6, 8]), environment=[wall], self_collision=False)
    assert len(rows) == 1 and rows[0]["status"] == "overlap_mtd"
    assert rows[0]["depth"] == pytest.approx(.15, abs=1e-4)


def test_invalid_inputs_and_removed_actor(scene):
    world, robot = scene
    for invalid in (np.zeros((3, 6)), poses([0, 1]), np.zeros((3, 7)), poses([0, float("nan"), 1])):
        with pytest.raises(ValueError):
            world.query_collisions(robot, invalid)
    for kwargs in ({"body_pairs": [(0, 9)]}, {"excluded_body_pairs": [(0, 0)]},
                   {"excluded_environment_pairs": [(0, 0)]}):
        with pytest.raises(ValueError):
            world.query_collisions(robot, poses([0, 1, 2]), **kwargs)
    wall = world.add_static_box([1, 1, 1], [0, 0, 0])
    world.remove_static_actor(wall)
    with pytest.raises(ValueError, match="live static actor"):
        world.query_collisions(robot, poses([0, 1, 2]), environment=[wall])


def test_released_articulation_is_rejected():
    builder = ke.physics.ArticulationBuilder()
    builder.add_link("root", inertia_diagonal=(1, 1, 1), shapes=[builder.SphereShape(.2)])
    world = ke.physics.PhysicsWorld()
    robot = ke.physics.Articulation.build(world, builder.build())
    robot.release()
    with pytest.raises(RuntimeError, match="live articulation"):
        world.get_collision_geometry(robot)
    with pytest.raises(RuntimeError, match="live articulation"):
        world.query_collisions(robot, poses([0]))
