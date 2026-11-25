"""Check shared skin sources, independent poses, and FBX rendering."""

import argparse
from pathlib import Path
from unittest.mock import patch

import numpy as np
import kangengine as ke


class RecordingAdapter:
    frame = None
    reject = False

    def render(self, frame):
        self.frame = frame
        if self.reject:
            raise RuntimeError("Rejected skin frame")
        return np.zeros((192, 256, 3), dtype=np.uint8)

    def close(self):
        pass


def cube_asset():
    mesh = ke.geometry.create_cube_data(0.6)
    tree = ke.animation.SkeletonTree(["root"], [-1], [[0, 0, 0]], [[1, 0, 0, 0]])
    count = len(mesh.vertices)
    return ke.visual.SkinnedSurfaceAsset(
        tree, np.asarray(mesh.vertices, np.float32), np.asarray(mesh.normals, np.float32),
        np.asarray(mesh.indices, np.int32), np.zeros((count, 4), np.int32),
        np.tile([1, 0, 0, 0], (count, 1)), np.array([0], np.int32),
        np.eye(4, dtype=np.float32)[None],
    )


def run(backend, fbx_path=None):
    app = ke.App()
    adapter = None
    spy = None
    writes = []
    try:
        if backend == "native":
            app.initialize(width=256, height=192, headless=True, hide_ui=True)
        else:
            if backend == "recording":
                adapter = RecordingAdapter()
            else:
                from kangengine.render._ovrtx_scene import OvrtxAppAdapter
                adapter = OvrtxAppAdapter(256, 192)
                write = adapter._stage.write_attribute

                def track(query, attribute, **kwargs):
                    if attribute in ("points", "normals", "restTransforms"):
                        writes.append(attribute)
                    return write(query, attribute, **kwargs)

                spy = patch.object(adapter._stage, "write_attribute", track)
                spy.start()
            app._initialize_scene_renderer(
                256, 192, ke.UpAxis.Y, ke.scene.BackendType.NATIVE,
                lambda width, height: adapter,
            )
        app.set_camera_view(position=[0, 1, 6] if fbx_path else [0, 0, 5],
                            target=[0, 1, 0] if fbx_path else [0, 0, 0])
        light = ke.scene.DirectionalLight()
        light.direction = ke.Vec3(0, 0, -1)
        light.ambient = ke.Vec3(0, 0, 0)
        light.intensity = 3
        app.scene.get_prim_at_path("/lights/default_directional").set_directional_light(light)
        material = app.create_pbr_material(base_color=[0.1, 0.8, 0.2, 1], roughness=1)
        if fbx_path:
            result = ke.asset.FBXLoader.parse(
                fbx_path=str(fbx_path), clip_index=0, fps=-1.0, scale=0.01,
            )
            first = ke.visual.SkinnedSurface.create_from_fbx_result(
                app, "/first", result, material=material, use_imported_materials=False,
            )
        else:
            first = ke.visual.SkinnedSurface.create(app, "/first", cube_asset(), material=material)
        second = first.create_instance("/second")
        for surface, x in ((first, -1), (second, 1)):
            for prim in surface.prims:
                prim.set_local_translation(ke.Vec3(x, 0, 0))
        for a, b in zip(first.prims, second.prims):
            assert a.resolve_mesh_data() is b.resolve_mesh_data()
        source = [np.asarray(asset.positions).copy() for asset in first.assets]

        def pose(surface, time):
            if fbx_path:
                state = result.motion.sample(time=time, loop=False)
                root = state.root_translation()
                state.set_root_translation([0, root.y, 0])
                surface.apply_state(state)
            else:
                matrices = np.eye(4, dtype=np.float32)[None]
                matrices[0, 1, 3] = time
                surface.update_joint_matrices(matrices)

        def capture():
            for _ in range(12 if backend == "ovrtx" else 1):
                app.draw_frame()
            return app.read_rgb_pixels().astype(float)

        pose(first, 0)
        pose(second, 0)
        before = capture()
        if backend == "recording":
            a, b = adapter.frame["batches"]
            assert a["skin"] is b["skin"] and a["mesh"] is b["mesh"]
            assert a["joints"] is not b["joints"]
            assert not a["joints"].flags.writeable
            assert not adapter.frame["objects"]
            retained = a["joints"]
        settled = capture()
        idle_writes = len(writes)
        pose(first, 0.8)
        moved = capture()
        if backend == "recording":
            a, b = adapter.frame["batches"]
            assert a["skin_revision"] > b["skin_revision"]
            np.testing.assert_array_equal(retained[0], np.eye(4))
            adapter.reject = True
            pose(first, 0.4)
            try:
                app.draw_frame()
            except RuntimeError as error:
                assert "Rejected skin frame" in str(error)
            else:
                raise AssertionError("Rejected pose was accepted")
            pending = adapter.frame["batches"][0]["joints"]
            adapter.reject = False
            app.draw_frame()
            assert np.shares_memory(pending, adapter.frame["batches"][0]["joints"])
        else:
            noise = np.abs(before - settled).mean()
            left = np.abs(moved[:, :128] - settled[:, :128]).mean()
            right = np.abs(moved[:, 128:] - settled[:, 128:]).mean()
            assert left > max(0.4, noise * 3), (left, noise)
            assert right < max(2, noise * 4), (right, noise)
            if backend == "ovrtx":
                assert writes[idle_writes:] == ["restTransforms"] * len(first.prims), writes
        if not fbx_path:
            first.update_bind_geometry(first.asset.positions + [0, 0.3, 0])
            capture()
            first.reset_bind_geometry()
            capture()
        for prim, original in zip(first.prims, source):
            np.testing.assert_array_equal(prim.resolve_mesh_data().vertices, original)
        first.set_visible(False)
        capture()
        first.set_visible(True)
        capture()
        replacement_path = first.views[0].path if len(first.views) == 1 else "/first"
        first.remove()
        capture()
        replacement = second.create_instance(replacement_path)
        pose(replacement, 0.4)
        capture()
        replacement.remove()
        second.remove()
        capture()
        assert app.get_scene_render_system().registration_count == 0
        if fbx_path:
            legacy = ke.visual.SkinVisual.from_fbx(
                app, material.native, str(fbx_path), path="/legacy",
                clip_index=0, fps=-1, scale=0.01, use_materials=False,
            )
            legacy.apply_time(0.8)
            shown = capture()
            legacy.set_visible(False)
            hidden = capture()
            assert np.abs(shown - hidden).mean() > 0.2
            legacy.set_visible(True)
            capture()
            legacy.remove()
            capture()
        if backend == "recording":
            assert not adapter.frame["batches"]
        elif backend == "ovrtx":
            assert not adapter._batches and not adapter._skin_layers
    finally:
        if spy is not None:
            spy.stop()
        app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", choices=("recording", "native", "ovrtx"), default="recording")
    parser.add_argument("--fbx-path", type=Path)
    args = parser.parse_args()
    run(args.renderer, args.fbx_path)
    print(f"PASS: {args.renderer} independent skinning")
