"""Check shadow exclusion, mesh debug geometry, and independent skeleton visibility."""

import argparse

import numpy as np
import kangengine as ke

from skinning_backend_smoke import cube_asset


def run(backend):
    app = ke.App()
    try:
        app.initialize(width=320, height=240, headless=True, hide_ui=True,
                       up_axis=ke.UpAxis.Y, renderer=backend)
        app.set_vsync(False)
        app.set_frame_capture_active(True)
        app.set_camera_view(position=[4, 5, 7], target=[0, 0, 0])
        app.set_light_direction([0.6, -1, 0.3])
        app.set_light_ambient([0, 0, 0])
        app.set_light_intensity(3)
        material = app.create_pbr_material(roughness=1)
        floor = app.scene.add_mesh(
            "/floor", ke.geometry.create_plane_data(10, ke.UpAxis.Y), material,
        )

        def capture():
            for _ in range(32 if backend == "ovrtx" else 2):
                app.draw_frame()
            return app.read_rgb_pixels().astype(float)

        # Each ownership path must preserve the caster while excluding its shadow.
        for kind in ("scene", "external", "skin"):
            if kind == "skin":
                surface = ke.visual.SkinnedSurface.create(
                    app, "/caster", cube_asset(), material=material,
                )
                surface.prims[0].set_local_translation(ke.Vec3(0, 1, 0))
                component = surface.prims[0].get_render_component()
                set_shadow = lambda enabled: setattr(component, "casts_shadow", enabled)
            else:
                surface = app.scene.add_mesh(
                    "/caster", ke.geometry.create_cube_data(1.5), material,
                    transform_source=(ke.render.TransformSource.EXTERNAL_BUFFER
                                      if kind == "external" else ke.render.TransformSource.SCENE_GRAPH),
                )
                if kind == "external":
                    transforms = np.eye(4, dtype=np.float32).reshape(1, 4, 4)
                    transforms[0, 3, 1] = 1
                    surface.set_transform_buffer(transforms)
                else:
                    surface.set_local_translation(ke.Vec3(0, 1, 0))
                set_shadow = surface.set_casts_shadow
            set_shadow(False)
            without = capture()
            set_shadow(True)
            with_shadow = capture()
            set_shadow(False)
            restored = capture()
            effect = (without - with_shadow).mean(axis=2)
            noise = np.abs(without - restored).mean()
            assert np.count_nonzero(effect > 10) > 20, (kind, effect.max())
            assert effect.mean() > noise, (kind, effect.mean(), noise)
            surface.set_visible(False)
            hidden = capture()
            assert np.count_nonzero(np.abs(restored - hidden).max(axis=2) > 10) > 20, kind
            surface.remove()
        floor.remove()

        app.set_camera_view(position=[0, 0, 5], target=[0, 0, 0])
        material.unlit = True
        empty = capture()
        for create in (app.debug.geometry.add_lines, app.scene.log_lines):
            view = create("/lines", material=material, starts=[[0, -1, 0]],
                          ends=[[0, 1, 0]], radius=0.08)
            assert view.component.transform_source == ke.render.TransformSource.EXTERNAL_BUFFER
            assert np.abs(capture() - empty).mean() > 1
            view.set_visible(False)
            assert np.abs(capture() - empty).mean() < 1
            view.set_visible(True)
            view.update_lines([[1, -1, 0]], [[1, 1, 0]])
            assert np.abs(capture() - empty).mean() > 1
            view.remove()

        tree = ke.animation.SkeletonTree(
            ["root", "tip"], [-1, 0], [[0, -0.6, 0], [0, 1.2, 0]],
            [[1, 0, 0, 0], [1, 0, 0, 0]],
        )
        state = ke.animation.SkeletonState.from_rotation_and_root_translation(
            tree, np.array([[1, 0, 0, 0], [1, 0, 0, 0]], np.float32),
            np.array([0, -0.6, 0], np.float32),
        )
        config = ke.visual.SkeletalVisualConfig()
        config.bone_radius = 0.05
        config.joint_radius = 0.15
        config.show_joints = False
        skeleton = ke.visual.SkeletalVisual.define(app, material, "/skeleton", state, config)
        bones = capture()
        skeleton.set_show_joints(True)
        joints = capture()
        assert np.abs(joints - bones).mean() > 0.2
        skeleton.set_visible(False)
        assert np.abs(capture() - empty).mean() < 1
        skeleton.set_show_joints(False)
        skeleton.apply_state(state)
        assert np.abs(capture() - empty).mean() < 1
        skeleton.set_visible(True)
        assert np.abs(capture() - bones).mean() < 1
        skeleton.remove()
        assert np.abs(capture() - empty).mean() < 1
        print(f"PASS: {backend} shadow exclusion, mesh lines, skeleton visibility and removal")
    finally:
        app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", choices=("native", "ovrtx"), default="native")
    run(parser.parse_args().renderer)
