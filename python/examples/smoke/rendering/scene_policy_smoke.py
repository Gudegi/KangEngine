"""Check object policies, batch reuse, and external transform preservation."""

import numpy as np

import kangengine as ke


def main():
    app = ke.App()
    app.initialize(width=256, height=192, headless=True, hide_ui=True)
    try:
        app.set_frame_capture_active(True)
        app.set_camera_view(position=[0, 0, 5], target=[0, 0, 0])
        mesh = ke.geometry.create_cube_data(0.6)
        material = app.create_pbr_material(base_color=[1, 0.02, 0.02, 0.4])
        material.unlit = True
        first = app.scene.add_mesh("/first", mesh, material)
        second = app.scene.add_mesh("/second", mesh, material)
        first.set_local_translation(ke.Vec3(-1, 0, 0))
        second.set_local_translation(ke.Vec3(1, 0, 0))
        system = app.get_scene_render_system()

        def capture():
            app.draw_frame()
            return app.read_rgb_pixels().copy()

        def shares_batch():
            return system.shares_batch(first.component, second.component)

        initial = capture()
        assert shares_batch()
        for setter, changed, restored in (
            (first.set_double_sided, True, False),
            (first.set_casts_shadow, False, True),
        ):
            setter(changed)
            assert not shares_batch()
            setter(restored)
            assert shares_batch()
        np.testing.assert_array_equal(capture(), initial)

        first.set_alpha_mode(ke.render.AlphaMode.MASK, cutoff=0.5)
        masked = capture()
        assert not shares_batch()
        assert np.count_nonzero(masked[:, :128] != initial[:, :128]) > 100
        np.testing.assert_array_equal(masked[:, 128:], initial[:, 128:])
        first.set_alpha_mode(ke.render.AlphaMode.MASK, cutoff=0.3)
        np.testing.assert_array_equal(capture(), initial)
        first.set_alpha_mode(ke.render.AlphaMode.OPAQUE)
        assert shares_batch()

        # A split must not silently restore source vertices over runtime edits.
        positions = np.asarray(mesh.vertices, dtype=np.float32).copy()
        positions[:, 1] += 0.5
        first.update_geometry(positions=positions)
        deformed = capture()
        try:
            first.set_double_sided(True)
        except RuntimeError as error:
            assert "shared dynamic geometry" in str(error)
        else:
            raise AssertionError("shared dynamic geometry was silently replaced")
        assert not first.component.double_sided
        assert shares_batch()
        np.testing.assert_array_equal(capture(), deformed)
        second.remove()
        first.set_double_sided(True)
        np.testing.assert_array_equal(capture()[:, :128], deformed[:, :128])
        first.remove()

        # ExternalBuffer policies affect every row while preserving their poses.
        external = app.scene.add_mesh(
            "/external", mesh, material,
            transform_source=ke.render.TransformSource.EXTERNAL_BUFFER,
        )
        transforms = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        transforms[:, 3, 0] = [-1, 1]
        external.set_transform_buffer(transforms)
        external_initial = capture()
        external.set_double_sided(True)
        external.set_casts_shadow(False)
        np.testing.assert_array_equal(capture(), external_initial)
        external.set_alpha_mode(ke.render.AlphaMode.MASK, cutoff=0.5)
        external_masked = capture()
        for half in (slice(None, 128), slice(128, None)):
            assert np.count_nonzero(external_masked[:, half] != external_initial[:, half]) > 100
        external.set_alpha_mode(ke.render.AlphaMode.OPAQUE)
        np.testing.assert_array_equal(capture(), external_initial)
        transforms[:, 3, 1] = 0.5
        external.set_transform_buffer(transforms)
        assert not np.array_equal(capture(), external_initial)
        external.remove()
        print("PASS: object policies, batch reuse, dynamic geometry, and external transforms")
    finally:
        app.close()


if __name__ == "__main__":
    main()
