"""Check Newton opacity changes against rendered pixels and retained transforms."""

import numpy as np
import newton
import warp as wp

import kangengine as ke
from kangengine.adapters.newton import ViewerKE


def main():
    builder = newton.ModelBuilder()
    body = builder.add_body(xform=wp.transform((0, 0, 1), wp.quat_identity()))
    builder.add_shape_sphere(body, radius=0.75)
    model = builder.finalize(device="cpu")
    state = model.state()
    viewer = ViewerKE(width=320, height=240, headless=True)
    try:
        viewer.app.set_vsync(enabled=False)
        viewer.set_model(model)
        viewer.set_camera(wp.vec3(0, -5, 1), pitch=0, yaw=90)
        viewer.log_state(state)
        viewer.end_frame()
        opaque = viewer.app.read_rgb_pixels().copy()
        key, batch = next(iter(viewer._instances.items()))
        transforms = batch.transform_buffer.copy()

        # Profile identical frames, changing only the appearance upload.
        renderer = viewer.app.get_renderer()
        renderer.set_profiler_enabled(enabled=True)
        colors = batch.colors[:, :3].copy()
        alpha = batch.colors[:, 3].copy()
        original_pre_render = viewer.app.pre_render
        viewer.app.pre_render = lambda: viewer.log_instances(
            key, batch.mesh_name, None, None, colors, None, opacities=alpha
        )
        viewer.end_frame()
        unchanged_bytes = renderer.latest_frame_profile().counters.buffer_upload_bytes
        cached = batch.colors
        viewer.end_frame()
        assert batch.colors is cached
        assert renderer.latest_frame_profile().counters.buffer_upload_bytes == unchanged_bytes
        colors[0, 0] = 0.0 if colors[0, 0] > 0.5 else 1.0
        viewer.end_frame()
        changed_bytes = renderer.latest_frame_profile().counters.buffer_upload_bytes
        assert changed_bytes == unchanged_bytes + batch.count * 4 * 4
        assert batch.colors is not cached
        viewer.end_frame()
        assert renderer.latest_frame_profile().counters.buffer_upload_bytes == unchanged_bytes
        print(f"PASS: unchanged appearance skips {changed_bytes - unchanged_bytes} upload bytes")
        viewer.app.pre_render = original_pre_render
        renderer.set_profiler_enabled(enabled=False)

        model.shape_opacity.assign(np.zeros(model.shape_count, dtype=np.float32))
        viewer.log_state(state)
        viewer.end_frame()
        transparent = viewer.app.read_rgb_pixels().copy()
        assert viewer._instances[key] is batch
        np.testing.assert_array_equal(batch.transform_buffer, transforms)
        assert np.all(batch.colors[:, 3] == 0)
        assert batch.view.component.alpha_mode == ke.render.AlphaMode.BLEND
        assert np.abs(opaque.astype(float) - transparent).mean() > 0.1

        # Appearance-only calls retain both transforms and the omitted RGB data.
        viewer.log_instances(key, batch.mesh_name, None, None, None, None,
                             opacities=np.ones(batch.count, dtype=np.float32))
        viewer.end_frame()
        restored = viewer.app.read_rgb_pixels()
        np.testing.assert_array_equal(restored, opaque)
        assert batch.view.component.alpha_mode == ke.render.AlphaMode.OPAQUE
        try:
            batch.view._render_system.set_instance_colors(
                batch.view.component, np.ones((batch.count + 1, 4), dtype=np.float32)
            )
        except ValueError:
            pass
        else:
            raise AssertionError("mismatched instance colors were accepted")
        print("PASS: opacity changes pixels; appearance-only updates retain transforms")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
