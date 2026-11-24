"""Check the interactive renderer samples and environment restoration."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import kangengine as ke

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rendering"))
from view_scene_renderer import SceneRendererApp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "ovrtx"), default="native")
    args = parser.parse_args()
    app = SceneRendererApp()
    try:
        app.initialize(width=960, height=640, up_axis=ke.UpAxis.Y, render_backend=args.backend)
        app.setup()
        app.set_vsync(enabled=False)
        app.update_and_draw()
        deadline = time.monotonic() + 0.3
        while time.monotonic() < deadline:
            app.update_and_draw()
            time.sleep(0.01)
        assert app.angle > 0, "Scheduled animation never ran"
        app.animate = False
        angle = app.angle
        for _ in range(4):
            time.sleep(0.02)
            app.update_and_draw()
        assert app.angle == angle, "Paused animation kept advancing"

        def capture():
            for _ in range(32):
                app.draw_frame()
            return app.read_rgb_pixels().astype(float)

        before = capture()
        settled = capture()
        noise = np.abs(settled - before).mean()
        for _ in range(2):
            app.environment.set_visible(False)
            hidden = capture()
            assert np.abs(hidden - settled).mean() > 1
            app.environment.set_visible(True)
            restored = capture()
            error = np.abs(restored - settled).mean()
            assert error < max(2, noise * 3), f"Environment toggle changed the scene: {error=}, {noise=}"

        app.set_camera_view(position=[-7, 4, 3], target=[-7, 1.5, -5])
        app.angle = 0
        app.samples_dirty = True
        before = capture()
        app.angle = 1.5
        app.samples_dirty = True
        after = capture()
        assert np.abs(after - before).mean() > 1, "Pose phase did not change the visible skins"
        source = app.skins[0].prims[0].resolve_mesh_data()
        assert all(skin.prims[0].resolve_mesh_data() is source for skin in app.skins)
        np.testing.assert_array_equal(app.deform_mesh.vertices, app.deform_positions)

        app.memory_enabled = False
        app.set_sample_textures()
        for material in app.sample_materials:
            material.unlit = True
        app.normal_material.normal_texture = None
        app.masked.set_alpha_mode(ke.render.AlphaMode.OPAQUE)
        capture()
        app.memory_enabled = True
        app.set_sample_textures()
        for material in app.sample_materials:
            material.unlit = False
        app.normal_material.normal_texture = app.normal_texture
        app.masked.set_alpha_mode(ke.render.AlphaMode.MASK)
        for light in app.local_lights.values():
            light.set_visible(True)
        capture()

        if args.backend == "ovrtx":
            app.set_camera_view(position=[8, 5, 13], target=[8, 0.5, 5])
            before = capture()
            app.set_cuda_enabled(True)
            assert app.cuda_enabled, app.cuda_status
            after = capture()
            assert np.abs(after - before).mean() > 1, "CUDA instances were not visible"
            app.set_cuda_enabled(False)
            capture()
            app.set_cuda_enabled(True)
            capture()
        app.set_checker_ground(True)
        assert not app.floor.prim.is_visible()
        app.set_light_ambient(ambient=[0.1, 0.1, 0.1])
        app.set_background_color(color=[0.1, 0.1, 0.1, 1])
        app.set_exposure(exposure=0.5)
        capture()
        app.set_checker_ground(False)
        assert app.floor.prim.is_visible()
        app.set_exposure(exposure=1)
        capture()
    finally:
        app.close()
    print(f"PASS: {args.backend} animation, shared skins, source mesh, material/light toggles, and CUDA when ovrtx")


if __name__ == "__main__":
    main()
