"""Check independent skinning and geometry on shared-source surface instances."""

import gc
import weakref

import numpy as np

import kangengine as ke


def main():
    app = ke.App()
    app.initialize(width=256, height=192, headless=True, hide_ui=True)
    try:
        app.set_frame_capture_active(True)
        app.set_camera_view(position=[0, 0, 5], target=[0, 0, 0])
        mesh = ke.geometry.create_cube_data(0.6)
        count = len(mesh.vertices)
        tree = ke.animation.SkeletonTree(
            ["root"], [-1], [[0, 0, 0]], [[1, 0, 0, 0]],
        )
        asset = ke.visual.SkinnedSurfaceAsset(
            tree, np.asarray(mesh.vertices, np.float32),
            np.asarray(mesh.normals, np.float32), np.asarray(mesh.indices, np.int32),
            np.zeros((count, 4), np.int32),
            np.tile([1, 0, 0, 0], (count, 1)), np.array([0], np.int32),
            np.eye(4, dtype=np.float32)[None],
        )

        # Direct registration must retain a caller-created material too.
        source, skin = asset.create_native()
        prim = app.scene.define_prim("/retained", ke.scene.PrimType.MESH)
        prim.set_mesh_data(source)
        material = ke.material.PBRMaterial()
        reference = weakref.ref(material)
        view = app.add_skinned_mesh(prim, material, skin)
        resource = app._existing_resource_handle(material)
        assert resource is not None
        assert app.resources.material(resource) is material.native
        del material
        gc.collect()
        assert reference() is not None
        view.remove()

        material = app.create_pbr_material(base_color=[1, 0.02, 0.02, 1])
        material.unlit = True
        first = ke.visual.SkinnedSurface.create(app, "/first", asset, material=material)
        second = first.create_instance("/second")
        assert first.views[0].get_base_color() is None
        assert second.views[0].get_base_color() is None
        first.prims[0].set_local_translation(ke.Vec3(-1, 0, 0))
        second.prims[0].set_local_translation(ke.Vec3(1, 0, 0))
        system = app.get_scene_render_system()
        assert not system.shares_batch(first.views[0].component, second.views[0].component)
        first_mesh = first.prims[0].get_mesh_component()
        second_mesh = second.prims[0].get_mesh_component()
        assert first_mesh.mesh_data is second_mesh.mesh_data
        assert first_mesh.resource_handle == second_mesh.resource_handle
        assert first.prims[0].get_material() is second.prims[0].get_material()
        source_vertices = np.asarray(first_mesh.mesh_data.vertices).copy()

        def capture():
            app.draw_frame()
            return app.read_rgb_pixels().copy()

        def center_y(image, right=False):
            rgb = image[:, 128:] if right else image[:, :128]
            rgb = rgb.astype(float)
            rows, _ = np.where((rgb[:, :, 0] > 150) &
                               (rgb[:, :, 0] > rgb[:, :, 1] + 80))
            assert len(rows) > 20
            return rows.mean()

        identity = np.eye(4, dtype=np.float32)[None]
        first.update_joint_matrices(identity)
        second.update_joint_matrices(identity)
        initial = capture()
        moved = identity.copy()
        moved[0, 1, 3] = 0.8
        first.update_joint_matrices(moved)
        posed = capture()
        assert center_y(posed) < center_y(initial) - 20
        np.testing.assert_array_equal(posed[:, 128:], initial[:, 128:])
        second.update_joint_matrices(moved)
        both_posed = capture()
        assert center_y(both_posed, True) < center_y(initial, True) - 20
        np.testing.assert_array_equal(both_posed[:, :128], posed[:, :128])

        # Policy changes retain the current pose and independent GPU geometry.
        first.views[0].set_double_sided(True)
        first.views[0].set_casts_shadow(False)
        np.testing.assert_array_equal(capture(), both_posed)

        # Per-instance pre-skinning edits must not change the shared source.
        first.update_joint_matrices(identity)
        second.update_joint_matrices(identity)
        deformed = asset.positions.copy()
        deformed[:, 1] += 0.8
        first.update_bind_geometry(deformed)
        changed = capture()
        assert center_y(changed) < center_y(initial) - 20
        np.testing.assert_array_equal(changed[:, 128:], initial[:, 128:])
        np.testing.assert_array_equal(first_mesh.mesh_data.vertices, source_vertices)
        first.reset_bind_geometry()
        np.testing.assert_array_equal(capture(), initial)

        # Removing and recreating an instance must not reuse its previous pose.
        second.update_joint_matrices(moved)
        survivor = capture()
        first.remove()
        np.testing.assert_array_equal(capture()[:, 128:], survivor[:, 128:])
        replacement = second.create_instance("/first")
        replacement.prims[0].set_local_translation(ke.Vec3(-1, 0, 0))
        recreated = capture()
        assert abs(center_y(recreated) - center_y(initial)) < 1
        np.testing.assert_array_equal(recreated[:, 128:], survivor[:, 128:])

        # Static objects still batch together, even with a skinned source mesh.
        static_a = app.scene.add_mesh("/static_a", second_mesh.mesh_data, material)
        static_b = app.scene.add_mesh("/static_b", second_mesh.mesh_data, material)
        assert system.shares_batch(static_a.component, static_b.component)
        assert not system.shares_batch(static_a.component, second.views[0].component)
        static_a.remove()
        static_b.remove()
        replacement.remove()
        second.remove()
        assert system.registration_count == 0
        print("PASS: skinned instance poses, geometry, resources, lifetime, and static batching")
    finally:
        app.close()


if __name__ == "__main__":
    main()
