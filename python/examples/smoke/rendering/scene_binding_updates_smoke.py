"""Check registered scene replacements, rendered pixels, and raycast geometry."""

import numpy as np

import kangengine as ke


def expect_rejected(change, message):
    try:
        change()
    except RuntimeError as error:
        assert message in str(error), str(error)
    else:
        raise AssertionError("unsupported registered replacement was accepted")


def main():
    app = ke.App()
    app.initialize(width=128, height=128, hide_ui=True, headless=True)
    try:
        app.set_frame_capture_active(True)
        app.set_camera_view(position=[0, 0, 6], target=[0, 0, 0])
        red = app.create_pbr_material(base_color=[1, 0.02, 0.02, 1])
        blue = app.create_pbr_material(base_color=[0.02, 0.02, 1, 1])
        red.unlit = blue.unlit = True
        mesh = ke.geometry.create_cube_data(2)
        first = app.scene.add_mesh("/first", mesh, red)
        second = app.scene.add_mesh("/second", mesh, red)
        second.set_local_translation(ke.Vec3(3, 0, 0))
        system = app.get_scene_render_system()
        resources = app.get_scene_resource_manager()

        def capture():
            app.draw_frame()
            return app.read_rgb_pixels().copy()

        initial = capture()
        assert initial[64, 64, 0] > initial[64, 64, 2] + 50
        assert system.shares_batch(first.component, second.component)

        # Direct authoring and the view API must use the same registration path.
        first.prim.set_material(blue.native)
        direct = capture()
        assert direct[64, 64, 2] > direct[64, 64, 0] + 50
        assert not system.shares_batch(first.component, second.component)
        first.set_material(blue)
        np.testing.assert_array_equal(capture(), direct)
        first.set_material(red)
        assert system.shares_batch(first.component, second.component)
        np.testing.assert_array_equal(capture(), initial)
        assert system.registration_count == 2

        # The smaller source must change both the image and CPU ray hits.
        ray_origin = ke.Vec3(0.6, 0, 5)
        ray_direction = ke.Vec3(0, 0, -1)
        assert app.raycast(ray_origin, ray_direction).prim is first.prim
        small = ke.geometry.create_cube_data(0.2)
        small_handle = resources.register_mesh("SmallReplacement", small)
        first.prim.set_mesh_data(small)
        first.prim.get_mesh_component().resource_handle = small_handle
        replaced = capture()
        assert np.abs(initial.astype(float) - replaced).mean() > 1
        assert not app.raycast(ray_origin, ray_direction).hit
        assert app.raycast(ke.Vec3(0, 0, 5), ray_direction).prim is first.prim
        assert app.raycast(ke.Vec3(3, 0, 5), ray_direction).prim is second.prim
        assert not system.shares_batch(first.component, second.component)

        # Invalid replacements leave the old scene binding and batch usable.
        expect_rejected(lambda: first.prim.set_material(None), "requires a material")
        expect_rejected(lambda: first.prim.set_mesh_data(ke.scene.MeshData()),
                        "failed to register replacement")
        assert first.prim.get_material() is red.native
        assert first.prim.get_mesh_data() is small
        np.testing.assert_array_equal(capture(), replaced)

        external = app.scene.add_mesh(
            "/external", mesh, red,
            transform_source=ke.render.TransformSource.EXTERNAL_BUFFER,
        )
        expect_rejected(lambda: external.prim.set_material(blue.native), "ExternalBuffer")
        expect_rejected(lambda: external.prim.set_mesh_data(small), "ExternalBuffer")
        assert external.prim.get_material() is red.native
        assert external.prim.get_mesh_data() is mesh
        external.remove()

        # A skinned source cannot be swapped without replacing its bone layout.
        skin_mesh = ke.geometry.create_cube_data(1)
        vertex_count = len(skin_mesh.vertices)
        skin = ke.scene.SkinnedMeshData(
            mesh=skin_mesh,
            bone_indices=np.zeros((vertex_count, 4), dtype=np.int32),
            bone_weights=np.tile([1, 0, 0, 0], (vertex_count, 1)),
            bone_node_indices=[0],
            inverse_bind_matrices=np.eye(4, dtype=np.float32)[None],
        )
        prim = app.scene.define_prim("/skin", ke.scene.PrimType.MESH)
        prim.set_mesh_data(skin_mesh)
        skinned = app.add_skinned_mesh(prim, red, skin)
        expect_rejected(lambda: prim.set_mesh_data(small), "skinned")
        assert prim.get_mesh_data() is skin_mesh
        prim.set_material(blue.native)
        skinned.update_skinning(np.eye(4, dtype=np.float32)[None])
        first.set_visible(False)
        assert capture()[64, 64, 2] > 100
        skinned.remove()
        first.remove()
        second.remove()
        assert system.registration_count == 0
        print("PASS: scene binding replacements, rollback, and ownership limits")
    finally:
        app.close()


if __name__ == "__main__":
    main()
