"""Check actual collider geometry without a renderer or simulation step."""

import numpy as np
import pytest
import kangengine as ke
from types import SimpleNamespace

from kangengine.visual.physics_inspector import _shape_lines


def test_static_snapshot_pose_and_lifetime():
    world = ke.physics.PhysicsWorld()
    actor = world.add_static_box(
        half_extents=[1, 2, 3], pos=[4, 5, 6],
        quat_xyzw=[0, 0, np.sqrt(0.5), np.sqrt(0.5)],
    )
    world.create_dynamic_sphere(radius=1, pos=[20, 0, 0])
    shapes = world.get_static_collision_geometry()
    assert len(shapes) == 1
    starts, ends = _shape_lines(shapes[0], 10)
    points = np.concatenate([starts, ends])
    np.testing.assert_allclose(points.min(axis=0), [2, 4, 3], atol=1e-5)
    np.testing.assert_allclose(points.max(axis=0), [6, 6, 9], atol=1e-5)
    reference = world.joint_body(actor)
    info = world.native._get_collision_body_geometry(reference)
    assert info["kind"] == "static" and info["available"]
    np.testing.assert_allclose(info["shapes"][0]["half_extents"], [1, 2, 3])
    world.remove_static_actor(actor)
    assert world.native._get_collision_body_geometry(reference) is None
    assert world.get_static_collision_geometry() == []
    np.testing.assert_array_equal(_shape_lines(shapes[0], 10)[0], starts)


def test_collider_pick_filters_and_lifetime():
    world = ke.physics.PhysicsWorld()
    static = world.add_static_box(half_extents=[.5, .5, .5], pos=[0, 0, 0])
    dynamic = world.create_dynamic_sphere(radius=.5, pos=[0, 0, 2])
    pick = world.native._pick_collision_body
    body, index = pick([0, 0, 5], [0, 0, -1], 10, True, True)
    assert index == 0
    assert world.native._get_collision_body_geometry(body)["kind"] == "dynamic"
    static_body, index = pick([0, 0, 5], [0, 0, -1], 10, True, False)
    assert index == 0
    assert world.native._get_collision_body_geometry(static_body)["kind"] == "static"
    assert pick([0, 0, 5], [0, 0, 1], 10, True, True) is None
    assert pick([0, 0, 5], [0, 0, -1], 10, False, False) is None
    dynamic.release()
    world.remove_static_actor(static)
    assert not body.valid and not static_body.valid


def test_cooked_convex_local_pose():
    world = ke.physics.PhysicsWorld()
    mesh = ke.geometry.create_cube_data(1)
    part = ke.physics.ConvexMeshPart(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        indices=np.asarray(mesh.indices, dtype=np.uint32),
        local_position=[2, 0, 0],
    )
    world.create_static_convex_compound(parts=[part], pos=[0, 3, 0])
    shape, = world.get_static_collision_geometry()
    assert shape["type"] == "convex_mesh"
    starts, ends = _shape_lines(shape, 10)
    points = np.concatenate([starts, ends])
    np.testing.assert_allclose(points.min(axis=0), [1.5, 2.5, -0.5], atol=1e-5)
    np.testing.assert_allclose(points.max(axis=0), [2.5, 3.5, 0.5], atol=1e-5)


def test_compound_pick_identifies_shape():
    world = ke.physics.PhysicsWorld()
    mesh = ke.geometry.create_cube_data(1)
    parts = [ke.physics.ConvexMeshPart(
        vertices=np.asarray(mesh.vertices, dtype=np.float32),
        indices=np.asarray(mesh.indices, dtype=np.uint32), local_position=[x, 0, 0],
    ) for x in (-1, 1)]
    world.create_static_convex_compound(parts=parts)
    body, index = world.native._pick_collision_body([1, 0, 5], [0, 0, -1], 10, True, False)
    assert index == 1
    info = world.native._get_collision_body_geometry(body)
    assert len(info["shapes"]) == 2
    np.testing.assert_allclose(info["shapes"][index]["local_pose"][:3], [1, 0, 0])


def test_plane_preview_and_mesh_scale():
    world = ke.physics.PhysicsWorld()
    world.add_default_ground()
    shape, = world.get_static_collision_geometry()
    assert shape["type"] == "plane"
    starts, ends = _shape_lines(shape, 8)
    points = np.concatenate([starts, ends])
    np.testing.assert_allclose(points[:, 1], 0, atol=1e-5)
    np.testing.assert_allclose(np.ptp(points, axis=0), [8, 0, 8], atol=1e-5)

    shape.update(type="convex_mesh", vertices=[[0, 0, 0], [1, 0, 0], [0, 1, 0]],
                 triangles=[[0, 1, 2]], mesh_scale=[2, 3, 4],
                 mesh_scale_rotation_xyzw=[0, 0, np.sqrt(0.5), np.sqrt(0.5)],
                 local_pose=[0, 0, 0, 0, 0, 0, 1], actor_pose=[0, 0, 0, 0, 0, 0, 1])
    starts, ends = _shape_lines(shape, 8)
    np.testing.assert_allclose(np.concatenate([starts, ends]).max(axis=0), [3, 2, 0], atol=1e-5)


def test_disabled_inspector_does_not_query_physics():
    class World:
        def get_static_collision_geometry(self):
            pytest.fail("Disabled inspector must not query physics")

    class App:
        def _set_physics_inspector_callback(self, callback):
            self.callback = callback

    inspector = ke.visual.PhysicsInspector(app=App(), world=World())
    inspector.refresh()
    inspector.sync()
    assert not inspector.static_visible
    assert not inspector.dynamic_visible


def test_triangle_mesh_snapshot():
    world = ke.physics.PhysicsWorld()
    mesh = ke.geometry.create_plane_data(4)
    world.add_triangle_mesh(mesh, position=[1, 2, 3])
    shape, = world.get_static_collision_geometry()
    assert shape["type"] == "triangle_mesh"
    starts, ends = _shape_lines(shape, 10)
    points = np.concatenate([starts, ends])
    np.testing.assert_allclose(points.min(axis=0), [-1, 2, 1], atol=1e-5)
    np.testing.assert_allclose(points.max(axis=0), [3, 2, 5], atol=1e-5)


@pytest.mark.parametrize("up_axis", [ke.UpAxis.Y, ke.UpAxis.Z])
def test_heightfield_cooked_snapshot(up_axis):
    world = ke.physics.PhysicsWorld()
    heights = np.array([[0, .5, 1, 1.5], [.5, 1, 1.5, 2], [1, 1.5, 2, 2.5]], dtype=np.float32)
    assert world.add_heightfield(heights.ravel(), 3, 4, horizontal_scale=.25,
                                 up_axis=up_axis, center=False)
    shape, = world.get_static_collision_geometry()
    assert shape["type"] == "heightfield" and shape["mesh_data_available"]
    assert len(shape["triangles"]) == 12
    points = np.concatenate(_shape_lines(shape, 10, world_space=False))
    np.testing.assert_allclose(np.ptp(points, axis=0), [.75, 2.5, .5], atol=1e-4)
    world_points = np.concatenate(_shape_lines(shape, 10))
    height_axis = 1 if up_axis == ke.UpAxis.Y else 2
    np.testing.assert_allclose(np.ptp(world_points[:, height_axis]), 2.5, atol=1e-4)


class RecordingOverlay:
    def __init__(self):
        self.lines_by_path = {}
        self.uploads = 0

    def lines(self, path, starts, ends, colors, **kwargs):
        self.lines_by_path[path] = np.concatenate([starts, ends])
        self.uploads += 1

    def remove(self, path):
        self.lines_by_path.pop(path, None)


class RecordingApp:
    def __init__(self):
        self.debug_overlay = RecordingOverlay()
        self.sync_callback = None
        self.selection = SimpleNamespace(
            hit=False, prim=None, transform_source=ke.render.TransformSource.SCENE_GRAPH,
        )

    def get_selection(self):
        return self.selection

    def _set_physics_inspector_callback(self, callback):
        self.callback = callback

    def _set_physics_inspector_sync_callback(self, callback):
        self.sync_callback = callback


def test_custom_inspection_source_and_disconnect():
    class Source:
        supports_dynamic = True
        dynamic_unavailable_reason = ""

        def __init__(self):
            self.body = SimpleNamespace(valid=True)
            self.queries = 0
            self.pose = np.array([[0, 0, 0, 0, 0, 0, 1]], dtype=np.float32)

        def dynamic_shapes(self):
            self.queries += 1
            return [dict(type="box", body_idx=0, shape_idx=0, half_extents=[1, 1, 1],
                         local_pose=[0, 0, 0, 0, 0, 0, 1])], [self.body]

        def body_poses(self, bodies):
            self.queries += 1
            return self.pose.copy(), np.array([self.body.valid])

    app = RecordingApp()
    source = Source()
    inspector = ke.visual.PhysicsInspector(app, source=source)
    inspector.sync()
    assert source.queries == 0 and app.sync_callback is None
    inspector.set_dynamic_visible(True)
    source.pose[0, 0] = 3
    app.sync_callback()
    points = app.debug_overlay.lines_by_path[inspector._dynamic_path]
    np.testing.assert_allclose(points.min(axis=0), [2, -1, -1])
    source.body.valid = False
    app.sync_callback()
    assert not app.debug_overlay.lines_by_path
    inspector.set_source(None)
    assert app.callback is None and app.sync_callback is None
    assert not inspector.dynamic_visible and not inspector._dynamic_bodies
    queries = source.queries
    inspector.sync()
    assert source.queries == queries
    with pytest.raises(RuntimeError, match="source"):
        inspector.set_dynamic_visible(True)
    inspector.release()
    assert app._physics_inspector is None


def test_dynamic_pose_cache_and_deletion(monkeypatch):
    import kangengine.visual.physics_inspector as module

    world = ke.physics.PhysicsWorld()
    body = world.create_dynamic_box(half_extents=[0.5, 1, 1.5], pos=[0, 3, 0])
    app = RecordingApp()
    inspector = ke.visual.PhysicsInspector(app=app, world=world)
    inspector.set_dynamic_visible(True)
    path, = app.debug_overlay.lines_by_path
    initial = app.debug_overlay.lines_by_path[path].copy()
    uploads = app.debug_overlay.uploads
    inspector.sync()
    assert app.debug_overlay.uploads == uploads

    def unexpected_geometry(*args, **kwargs):
        pytest.fail("Pose sync must not rebuild geometry")

    with monkeypatch.context() as patch:
        patch.setattr(module, "_shape_lines", unexpected_geometry)
        body.set_root_state(pos=[2, 4, 0], quat_xyzw=[0, 0, np.sqrt(.5), np.sqrt(.5)],
                            linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
        inspector.sync()
        points = app.debug_overlay.lines_by_path[path]
        np.testing.assert_allclose(points.min(axis=0), [1, 3.5, -1.5], atol=1e-5)
        np.testing.assert_allclose(points.max(axis=0), [3, 4.5, 1.5], atol=1e-5)
        assert not np.array_equal(points, initial)
        body.release()
        inspector.sync()
        assert path not in app.debug_overlay.lines_by_path

    world.create_dynamic_sphere(radius=.3, pos=[10, 0, 0])
    inspector.sync()
    assert path not in app.debug_overlay.lines_by_path
    inspector.refresh()
    assert path in app.debug_overlay.lines_by_path
    inspector.set_dynamic_visible(False)
    assert app.debug_overlay.lines_by_path == {}
    inspector.sync()
    assert app.debug_overlay.lines_by_path == {}
    inspector.release()


def test_dynamic_snapshot_kinematic_and_articulation_lifetime():
    world = ke.physics.PhysicsWorld()
    body = world.create_dynamic_sphere(radius=.2, pos=[0, 2, 0])
    body.set_kinematic(True)
    builder = ke.physics.ArticulationBuilder()
    builder.add_link("root", position=(0, 0, 3), inertia_diagonal=(1, 1, 1),
                     shapes=[builder.SphereShape(.2)])
    robot = ke.physics.Articulation.build(world, builder.build())
    shapes, bodies = world.native._get_dynamic_collision_geometry()
    assert len(shapes) == len(bodies) == 2
    poses, valid = world.native._get_collision_body_poses(bodies)
    assert valid.all()
    assert world.native._get_collision_body_geometry(bodies[0])["kind"] == "kinematic"
    assert world.native._get_collision_body_geometry(bodies[1])["kind"] == "articulation"
    np.testing.assert_allclose(poses[0, :3], [0, 2, 0])
    robot.release()
    _, valid = world.native._get_collision_body_poses(bodies)
    np.testing.assert_array_equal(valid, [True, False])
    del robot, body, world
    assert not any(handle.valid for handle in bodies)
