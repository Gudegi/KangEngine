"""Check CPU instance ownership, bulk updates, and shared ovrtx geometry."""

import argparse
from unittest.mock import patch

import numpy as np
import kangengine as ke


class RecordingAdapter:
    frame = None
    reject = False

    def render(self, frame):
        self.frame = frame
        if self.reject:
            raise RuntimeError("Rejected instance frame")
        return np.zeros((96, 128, 3), dtype=np.uint8)

    def close(self):
        pass


def add_batch(app, mesh, material, path="/batch"):
    return app.scene.add_mesh(
        path, mesh, material, transform_source=ke.render.TransformSource.EXTERNAL_BUFFER,
    )


def transforms():
    # update_instances accepts GLM column-major storage, including translation.
    matrices = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
    matrices[:, 3, 0] = [-0.65, 0.65]
    matrices[0, 1, 0] = 0.2
    return matrices


def check_delivery():
    adapter = RecordingAdapter()
    app = ke.App()
    try:
        app._initialize_scene_renderer(
            128, 96, ke.UpAxis.Y, ke.scene.BackendType.NATIVE,
            lambda width, height: adapter,
        )
        mesh = ke.geometry.create_cube_data(0.6)
        material = app.create_pbr_material()
        view = add_batch(app, mesh, material)
        system = app.get_scene_render_system()
        values = transforms()
        expected = values.swapaxes(1, 2).copy()
        system.update_instances(view.component, values)
        values[:] = 0
        ordinary = app.scene.add_mesh("/ordinary", mesh, material)
        app.draw_frame()
        item, = adapter.frame["batches"]
        assert len(adapter.frame["objects"]) == 1
        assert adapter.frame["objects"][0]["path"] == "/ordinary"
        np.testing.assert_array_equal(item["transforms"], expected)
        assert not item["transforms"].flags.writeable
        accepted = item

        view.prim.set_local_translation(ke.Vec3(100, 0, 0))
        app.draw_frame()
        current, = adapter.frame["batches"]
        assert current["revision"] == accepted["revision"]
        assert np.shares_memory(current["transforms"], accepted["transforms"])
        assert not adapter.frame["objects"]
        view.set_visible(False)
        app.draw_frame()
        assert not adapter.frame["batches"][0]["visible"]
        view.set_visible(True)
        view.prim.set_active(False)
        app.draw_frame()
        assert not adapter.frame["batches"][0]["visible"]
        view.prim.set_active(True)

        for operation in (
            lambda: system.update_instances(view.component, transforms(), [[1, 0, 0, 1]]),
            lambda: system.set_instance_colors(view.component, [[1, 0, 0, 1]]),
            lambda: system.set_external_buffer(view.component, ke.render.ExternalBufferDesc()),
        ):
            try:
                operation()
            except RuntimeError:
                pass
            else:
                raise AssertionError("Unsupported batch input was accepted")
        app.draw_frame()
        assert adapter.frame["batches"][0]["revision"] == accepted["revision"]

        updated = transforms()
        updated[0, 3, 1] = 0.5
        system.update_instances(view.component, updated)
        adapter.reject = True
        try:
            app.draw_frame()
        except RuntimeError as error:
            assert "Rejected instance frame" in str(error)
        else:
            raise AssertionError("Rejected batch was accepted")
        pending = adapter.frame["batches"][0]
        adapter.reject = False
        app.draw_frame()
        retry = adapter.frame["batches"][0]
        assert retry["revision"] == pending["revision"]
        assert np.shares_memory(retry["transforms"], pending["transforms"])
        np.testing.assert_array_equal(accepted["transforms"], expected)

        # One batch can have multiple owners; removing the last releases arrays.
        peer = add_batch(app, mesh, material, "/peer")
        view.remove()
        app.draw_frame()
        assert adapter.frame["batches"][0]["id"] == accepted["id"]
        peer.remove()
        app.draw_frame()
        assert not adapter.frame["batches"]
        np.testing.assert_array_equal(accepted["transforms"], expected)
        replacement = add_batch(app, mesh, material)
        app.draw_frame()
        item, = adapter.frame["batches"]
        assert item["id"] != accepted["id"] and len(item["transforms"]) == 0
        replacement.remove()
        ordinary.remove()
    finally:
        app.close()


def check_rendering(backend):
    app = ke.App()
    adapter = None
    spy = None
    writes = []
    try:
        if backend == "native":
            app.initialize(width=128, height=96, headless=True, hide_ui=True)
        else:
            from kangengine.render._ovrtx_scene import OvrtxAppAdapter
            import ovstage

            adapter = OvrtxAppAdapter(128, 96)
            app._initialize_scene_renderer(
                128, 96, ke.UpAxis.Y, ke.scene.BackendType.NATIVE,
                lambda width, height: adapter,
            )
            write = adapter._stage.write_attribute

            def track(query, attribute, **kwargs):
                values = kwargs["tensors"]
                if attribute == "omni:xform" and len(values) > 1:
                    writes.append(len(values))
                return write(query, attribute, **kwargs)

            spy = patch.object(adapter._stage, "write_attribute", track)
            spy.start()
        app.set_camera_view(position=[0, 0, 4], target=[0, 0, 0])
        light = ke.scene.DirectionalLight()
        light.direction = ke.Vec3(0, 0, -1)
        light.ambient = ke.Vec3(0, 0, 0)
        light.intensity = 3
        app.scene.get_prim_at_path("/lights/default_directional").set_directional_light(light)
        material = app.create_pbr_material(base_color=[0.1, 0.8, 0.2, 1], roughness=1)
        mesh = ke.geometry.create_cube_data(0.6)
        view = add_batch(app, mesh, material)
        system = app.get_scene_render_system()
        values = transforms()
        system.update_instances(view.component, values)
        if adapter:
            # A rejected scene must not leave a batch marked as populated.
            light.ambient = ke.Vec3(float("nan"), 0, 0)
            light_prim = app.scene.get_prim_at_path("/lights/default_directional")
            light_prim.set_directional_light(light)
            try:
                app.draw_frame()
            except ValueError as error:
                assert "ambient" in str(error)
            else:
                raise AssertionError("Unsupported light was accepted")
            light.ambient = ke.Vec3(0, 0, 0)
            light_prim.set_directional_light(light)

        def capture():
            for _ in range(12 if adapter else 1):
                app.draw_frame()
            return app.read_rgb_pixels().astype(float)

        before = capture()
        settled = capture()
        noise = np.abs(before - settled).mean()
        assert settled[:, :64, 1].mean() > 2 and settled[:, 64:, 1].mean() > 2
        if adapter:
            roots = ovstage.instancing.get_prototype_roots(adapter._stage)
            assert len(roots) == 1
            assert len(ovstage.instancing.get_instance_roots(adapter._stage, roots[0])) == 2
            assert writes == [2], writes
            references = adapter._batches.copy()
        view.prim.set_local_translation(ke.Vec3(100, 0, 0))
        unchanged = capture()
        assert np.abs(unchanged - settled).mean() < max(2, noise * 3)
        values[0, 1, 1] = 0.2
        system.update_instances(view.component, values)
        moved = capture()
        assert np.abs(moved[:, :64] - settled[:, :64]).mean() > max(1, noise * 3)
        assert np.abs(moved[:, 64:] - settled[:, 64:]).mean() < max(2, noise * 3)
        if adapter:
            assert writes == [2, 2], writes
            assert adapter._batches == references

        view.set_visible(False)
        hidden = capture()
        assert np.abs(hidden - moved).mean() > 1
        view.set_visible(True)
        visible = capture()
        assert np.abs(visible - moved).mean() < max(2, noise * 3)
        if adapter:
            assert writes == [2, 2], writes

        # SceneGraph population must retain existing external transforms.
        ordinary = app.scene.add_mesh("/ordinary", mesh, material)
        ordinary.set_local_translation(ke.Vec3(0, 5, 0))
        mixed = capture()
        assert np.abs(mixed - visible).mean() < max(2, noise * 3)
        ordinary.remove()
        capture()
        system.update_instances(view.component, transforms()[:1])
        reduced = capture()
        assert np.abs(reduced[:, 64:] - visible[:, 64:]).mean() > 1
        system.update_instances(view.component, np.empty((0, 4, 4), dtype=np.float32))
        empty = capture()
        empty_error = np.abs(empty - reduced).mean()
        assert empty_error > 1, f"Clearing instances did not change the image: {empty_error}"
        system.update_instances(view.component, transforms())
        restored = capture()
        assert np.abs(restored - empty).mean() > 1
        material.base_color = [0.8, 0.1, 0.2, 1]
        recolored = capture()
        for region in (slice(None, 64), slice(64, None)):
            assert np.abs(recolored[:, region] - restored[:, region]).mean() > 1
        view.remove()
        removed = capture()
        assert np.abs(removed - recolored).mean() > 1
        if adapter:
            assert not adapter._batches and not adapter._batch_queries
    finally:
        if spy is not None:
            spy.stop()
        app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("recording", "native", "ovrtx"), default="recording")
    backend = parser.parse_args().backend
    if backend == "recording":
        check_delivery()
    else:
        check_rendering(backend)
    print(f"PASS: {backend} CPU external instances")
