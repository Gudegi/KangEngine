"""Verify collider snapshots, refresh, and visibility through each renderer."""

import ctypes
import tempfile
from pathlib import Path

import numpy as np
import kangengine as ke
from kangengine._core import _ke

from kangengine.utils._example_args import render_example_parser


class InspectorApp(ke.App):
    before_render = None

    def pre_render(self):
        if self.before_render is not None:
            self.before_render()


def run(renderer):
    app = InspectorApp()
    try:
        app.initialize(width=320, height=240, hide_ui=True, renderer=renderer)
        app.set_frame_capture_active(True)
        app.set_camera_view(position=[3, 2, 5], target=[0, 0, 0])
        sim_world = ke.sim.KangSimWorld(num_envs=2, add_ground=False)
        world = sim_world.physics
        actor = world.add_static_box(half_extents=[0.8, 0.5, 0.4], pos=[0, 0, 0])
        inspector = app.physics_inspector
        assert inspector.source is None
        inspector.set_source(ke.physics.PhysXInspectionSource(world))
        try:
            ke.visual.PhysicsInspector(app=app, world=world)
        except RuntimeError as error:
            assert "only one physics inspector" in str(error)
        else:
            raise AssertionError("A second inspector must not replace the active menu")

        def capture():
            for _ in range(4):
                app.draw_frame()
            return app.read_rgb_pixels().astype(float)

        empty = capture()
        inspector.set_static_visible(True)
        visible = capture()
        assert np.count_nonzero(np.abs(visible - empty).max(axis=2) > 10) > 30
        inspector.set_static_visible(False)
        np.testing.assert_allclose(capture(), empty, atol=1)
        inspector.set_static_visible(True)
        np.testing.assert_allclose(capture(), visible, atol=1)

        world.remove_static_actor(actor)
        inspector.refresh()
        np.testing.assert_allclose(capture(), empty, atol=1)

        ball = world.create_dynamic_sphere(radius=0.5, pos=[-1, 0, 0])
        inspector.set_dynamic_visible(True)
        left = capture()
        app.before_render = lambda: ball.set_root_state(
            pos=[1, 0, 0], quat_xyzw=[0, 0, 0, 1],
            linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0],
        )
        right = capture()
        app.before_render = None
        assert np.count_nonzero(np.abs(left - right).max(axis=2) > 10) > 30
        inspector.set_dynamic_visible(False)
        np.testing.assert_allclose(capture(), empty, atol=1)
        inspector.set_dynamic_visible(True)
        np.testing.assert_allclose(capture(), right, atol=1)
        ball.release()
        inspector.sync()
        np.testing.assert_allclose(capture(), empty, atol=1)

        # The same selection drives inspection and a separate collider highlight.
        material = app.create_pbr_material()
        view = app.scene.add_mesh("/selected", ke.geometry.create_cube_data(1), material)
        body = world.create_dynamic_box(half_extents=[.5, .5, .5], pos=[0, 0, 0])
        inspector.bind_visual(view, body)
        inspector.refresh()
        app.select_prim(view.prim)
        inspector.sync()
        assert inspector._selected_info["kind"] == "dynamic"
        np.testing.assert_allclose(inspector._selected_pose[:3], [0, 0, 0])
        highlighted = capture()
        app.clear_selection()
        inspector.sync()
        assert inspector._selected_info is None and inspector._selected_pose is None
        assert np.count_nonzero(np.abs(highlighted - capture()).max(axis=2) > 10) > 20
        app.select_prim(view.prim)
        inspector.sync()
        body.set_root_state(pos=[1, 0, 0], quat_xyzw=[0, 0, 0, 1],
                            linear_velocity=[0, 0, 0], angular_velocity=[0, 0, 0])
        inspector.sync()
        np.testing.assert_allclose(inspector._selected_pose[:3], [1, 0, 0])
        body.release()
        inspector.sync()
        assert inspector._selected_info is None and inspector._selected_pose is None
        view.remove()
        replacement_view = app.scene.add_mesh("/selected", ke.geometry.create_cube_data(1), material)
        app.select_prim(replacement_view.prim)
        inspector.sync()
        assert inspector._selected_info is None
        replacement_view.remove()

        check_external_selection(app, world, inspector, material, capture)
        check_collider_selection(app, world, inspector, capture)
        inspector.release()
        inspector.release()
        replacement = ke.visual.PhysicsInspector(app=app, world=world)
        replacement.release()
        check_visualizer_selection(app, sim_world, capture)
        app.physics_inspector.set_source(ke.physics.PhysXInspectionSource(world))
        app.physics_inspector.set_static_visible(True)
        app.physics_inspector.set_collider_picking_enabled(True)
        print(f"physics_inspector_smoke: {renderer} PASS")
    finally:
        owned_inspector = app._physics_inspector
        app.close()
        if owned_inspector is not None:
            assert owned_inspector._released and owned_inspector.source is None


def check_collider_selection(app, world, inspector, capture):
    app.set_camera_view(position=[0, 0, 5], target=[0, 0, 0])
    body = world.create_dynamic_sphere(radius=.5, pos=[0, 0, 0])
    inspector.refresh()
    inspector.set_collider_picking_enabled(True)
    capture()
    selection = click_center(app, capture)
    assert selection.hit and selection.prim.get_path() == inspector._selection_path
    inspector.sync()
    assert inspector._selected_info["kind"] == "dynamic"
    assert inspector._selected_info["picked_shape_idx"] == 0
    assert inspector._selected_pose is not None
    body.release()
    inspector.sync()
    assert not app.get_selection().hit
    assert inspector._selected_info is None
    inspector.set_collider_picking_enabled(False)
    assert app.scene.get_prim_at_path(inspector._selection_path) is None


def check_external_selection(app, world, inspector, material, capture):
    app.set_camera_view(position=[0, 0, 5], target=[0, 0, 0])
    view = app.scene.add_mesh(
        "/batch", ke.geometry.create_cube_data(.6), material,
        transform_source=ke.render.TransformSource.EXTERNAL_BUFFER,
    )
    matrices = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    matrices[0, 3, 0] = 1.5
    app.get_scene_render_system().update_instances(view.component, matrices)
    bodies = [world.create_dynamic_sphere(radius=radius, pos=pos)
              for radius, pos in ((.2, [1.5, 0, 0]), (.3, [0, 0, 0]))]
    for idx, body in enumerate(bodies):
        inspector.bind_visual(view, body, instance_index=idx)
    capture()

    # Exercise actual viewport picking, including the ExternalBuffer instance index.
    selection = click_center(app, capture)
    inspector.sync()
    assert selection.instance_index == 1
    assert inspector._selected_info["instance_index"] == 1
    assert abs(inspector._selected_info["shapes"][0]["radius"] - .3) < 1e-6
    np.testing.assert_allclose(inspector._selected_pose[:3], [0, 0, 0])
    view.remove()
    inspector.sync()
    assert inspector._selected_info is None and inspector._selected_pose is None
    for body in bodies:
        body.release()


def click_center(app, capture):
    glfw = ctypes.CDLL(_ke.__file__)
    glfw.glfwGetCurrentContext.restype = ctypes.c_void_p
    window = glfw.glfwGetCurrentContext()
    glfw.glfwGetWindowSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
    width, height = ctypes.c_int(), ctypes.c_int()
    glfw.glfwGetWindowSize(window, ctypes.byref(width), ctypes.byref(height))
    mouse_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int)
    cursor_type = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_double, ctypes.c_double)
    glfw.glfwSetMouseButtonCallback.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    glfw.glfwSetMouseButtonCallback.restype = ctypes.c_void_p
    glfw.glfwSetCursorPosCallback.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    glfw.glfwSetCursorPosCallback.restype = ctypes.c_void_p
    mouse = glfw.glfwSetMouseButtonCallback(window, None)
    cursor = glfw.glfwSetCursorPosCallback(window, None)
    glfw.glfwSetMouseButtonCallback(window, mouse)
    glfw.glfwSetCursorPosCallback(window, cursor)
    cursor_type(cursor)(window, width.value / 2, height.value / 2)
    mouse_type(mouse)(window, 0, 1, 0)
    mouse_type(mouse)(window, 0, 0, 0)
    capture()
    return app.get_selection()


def check_visualizer_selection(app, world, capture):
    visualizer = ke.visual.sim.SimWorldVisualizer(app, world)
    inspector = app.physics_inspector
    assert inspector.source is visualizer._inspection_source
    other_visualizer = ke.visual.sim.SimWorldVisualizer(app, world)
    assert inspector.source is visualizer._inspection_source
    other_visualizer.release()
    assert inspector.source is visualizer._inspection_source
    try:
        mesh = ke.geometry.create_cube_data(.4)
        data = ke.asset.RigidDesc(
            collision_geoms=[ke.asset.CollisionGeomDesc(
                type=ke.asset.CollisionGeomDescType.BOX, size=(.2, .2, .2),
                position=(0, 0, 0), quat_xyzw=(0, 0, 0, 1))],
            visual_geoms=[ke.asset.VisualGeomDesc(mesh_data=mesh)],
        )
        rigids = [world.add_rigid(data, env_id=idx, obj_id=0, kinematic=True, pos=[idx, 0, 0])
                  for idx in range(2)]
        scene_visuals = [visualizer.add_scene_graph(rigid, path=f"/sim_body{idx}")
                         for idx, rigid in enumerate(rigids)]
        visualizer.sync()
        capture()
        inspector.set_dynamic_visible(True)
        for idx, visual in enumerate(scene_visuals):
            app.select_prim(visual.body_prims[0])
            inspector.sync()
            info = inspector._selected_info
            assert (info["env_id"], info["obj_id"], info["body_id"]) == (idx, 0, 0)
            assert info["kind"] == "kinematic"
            visual.set_visible(False)

        visualizer.add(world.get_rigid_batch(obj_id=0), path="/sim_batch")
        visualizer.sync()
        for idx in range(2):
            app.set_camera_view(position=[idx, 0, 5], target=[idx, 0, 0])
            capture()
            assert click_center(app, capture).hit
            inspector.sync()
            info = inspector._selected_info
            assert (info["env_id"], info["obj_id"], info["body_id"]) == (idx, 0, 0)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "robot.xml"
            path.write_text('<mujoco><worldbody><body name="root"><geom type="box" size=".2 .2 .2"/>'
                            '<body name="link" pos="0 0 1"><joint type="hinge"/>'
                            '<geom type="sphere" size=".1"/></body></body></worldbody></mujoco>')
            robot = world.add_articulation(world.load_mjcf(str(path)), obj_id=1,
                                           config=ke.physics.ArticulationConfig.fixed_base())
        robot_visual = visualizer.add_scene_graph(robot, path="/robot")
        visualizer.sync()
        capture()
        app.select_prim(robot_visual.render_prims[1])
        inspector.sync()
        info = inspector._selected_info
        assert (info["kind"], info["body_id"]) == ("articulation", 1)
        assert info["shapes"][0]["type"] == "sphere"
        world.release()
        inspector.sync()
        assert inspector._selected_info is None
    finally:
        visualizer.release()
        assert inspector.source is None
        assert not inspector.static_visible and not inspector.dynamic_visible
        world.release()


if __name__ == "__main__":
    run(render_example_parser(description=__doc__).parse_args().renderer)
