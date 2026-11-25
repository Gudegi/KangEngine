"""Check deformation delivery, source ownership, and optional rendered output."""

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
            raise RuntimeError("Rejected geometry frame")
        return np.zeros((96, 128, 3), dtype=np.uint8)

    def close(self):
        pass


def check_delivery():
    adapter = RecordingAdapter()
    app = ke.App()
    try:
        app._initialize_scene_renderer(
            128, 96, ke.UpAxis.Y, ke.scene.BackendType.NATIVE,
            lambda width, height: adapter,
        )
        mesh = ke.geometry.create_cube_data(1)
        positions = np.asarray(mesh.vertices, dtype=np.float32)
        normals = np.asarray(mesh.normals, dtype=np.float32)
        material = app.create_pbr_material()
        first = app.scene.add_mesh("/first", mesh, material)
        second = app.scene.add_mesh("/second", mesh, material)
        app.draw_frame()
        ids = {item["path"]: item["id"] for item in adapter.frame["objects"]}
        app.draw_frame()
        assert not adapter.frame["objects"]

        deformed = positions.copy()
        deformed[:, 0] *= 0.4
        first.update_geometry(deformed, normals)
        deformed[:] = 10
        app.draw_frame()
        item, = adapter.frame["objects"]
        assert item["id"] == ids["/first"]
        accepted = item["geometry"]
        np.testing.assert_allclose(accepted.positions[:, 0], positions[:, 0] * 0.4)
        assert not accepted.positions.flags.writeable
        np.testing.assert_array_equal(mesh.vertices, positions)
        np.testing.assert_array_equal(mesh.normals, normals)

        first.update_geometry(positions * 0.8)
        adapter.reject = True
        try:
            app.draw_frame()
        except RuntimeError as error:
            assert "Rejected geometry frame" in str(error)
        else:
            raise AssertionError("Rejected geometry was accepted")
        pending = adapter.frame["objects"][0]["geometry"]
        adapter.reject = False
        app.draw_frame()
        assert adapter.frame["objects"][0]["geometry"] is pending
        np.testing.assert_array_equal(pending.normals, normals)
        np.testing.assert_allclose(accepted.positions[:, 0], positions[:, 0] * 0.4)
        app.draw_frame()
        assert not adapter.frame["objects"]

        for points, normal_values in ((positions[:-1], normals), (positions, normals[:-1])):
            try:
                first.update_geometry(points, normal_values)
            except (ValueError, RuntimeError):
                pass
            else:
                raise AssertionError("Topology mismatch was accepted")
        app.draw_frame()
        assert not adapter.frame["objects"]

        replacement = ke.geometry.create_cube_data(0.5)
        resource = app.resources.register_mesh("Replacement", replacement)
        first.prim.set_mesh_data(replacement)
        first.prim.get_mesh_component().resource_handle = resource
        app.draw_frame()
        item, = adapter.frame["objects"]
        assert item["mesh"] is replacement and item["geometry"] is None

        # Re-registration must not replay a pending update from the old identity.
        first.update_geometry(positions * 0.2)
        system = app.get_scene_render_system()
        component = first.component
        first.prim.remove_render_component()
        first = app.scene.add_renderable(first.prim, material)
        app.draw_frame()
        item, = adapter.frame["objects"]
        assert item["id"] != ids["/first"] and item["geometry"] is None
        assert ids["/first"] in adapter.frame["removed_objects"]
        first.update_geometry(positions * 0.5)
        first.remove()
        try:
            system.update_geometry(component, positions)
        except RuntimeError:
            pass
        else:
            raise AssertionError("Removed registration accepted an update")
        app.draw_frame()
        assert not adapter.frame["objects"]
        assert second.prim.resolve_mesh_data() is mesh
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

            adapter = OvrtxAppAdapter(128, 96)
            app._initialize_scene_renderer(
                128, 96, ke.UpAxis.Y, ke.scene.BackendType.NATIVE,
                lambda width, height: adapter,
            )
            write = adapter._stage.write_attribute

            def tracked_write(query, attribute, **kwargs):
                if attribute in ("points", "normals"):
                    writes.append(attribute)
                return write(query, attribute, **kwargs)

            spy = patch.object(adapter._stage, "write_attribute", tracked_write)
            spy.start()
        app.set_camera_view(position=[0, 0, 4], target=[0, 0, 0])
        light = ke.scene.DirectionalLight()
        light.direction = ke.Vec3(0, 0, -1)
        light.intensity = 3
        light.ambient = ke.Vec3(0, 0, 0)
        app.scene.get_prim_at_path("/lights/default_directional").set_directional_light(light)
        material = app.create_pbr_material(base_color=[0.1, 0.8, 0.2, 1], roughness=1)
        mesh = ke.geometry.create_cube_data(0.8)
        positions = np.asarray(mesh.vertices, dtype=np.float32)
        normals = np.asarray(mesh.normals, dtype=np.float32)
        first = app.scene.add_mesh("/first", mesh, material)
        # Native geometry updates retain their existing batch-wide scope.
        peer_material = material.clone() if backend == "native" else material
        second = app.scene.add_mesh("/second", mesh, peer_material)
        first.set_local_translation(ke.Vec3(-0.65, 0, 0))
        second.set_local_translation(ke.Vec3(0.65, 0, 0))

        def capture():
            for _ in range(12 if adapter else 1):
                app.draw_frame()
            return app.read_rgb_pixels().astype(float)

        before = capture()
        settled = capture()
        noise = np.abs(before - settled).mean()
        deformed = positions.copy()
        deformed[:, 1] *= 0.2
        first.update_geometry(deformed, normals)
        after = capture()
        assert np.abs(after[:, :64] - settled[:, :64]).mean() > max(1, noise * 3)
        assert np.abs(after[:, 64:] - settled[:, 64:]).mean() < max(2, noise * 3)
        np.testing.assert_array_equal(mesh.vertices, positions)
        if adapter:
            assert writes == ["points", "normals"], writes
            references = adapter._references.copy()
            first.set_local_translation(ke.Vec3(-0.65, 0.1, 0))
            capture()
            assert writes == ["points", "normals"]
            assert adapter._references == references
            # Population changes must not reset an existing deformation.
            third = app.scene.add_mesh("/third", mesh, material)
            third.set_local_translation(ke.Vec3(0, 5, 0))
            capture()
            assert len(writes) == 4
            third.remove()
            capture()
            assert len(writes) == 6
        first.update_geometry(positions)
        restored = capture()
        assert np.abs(restored[:, :64] - after[:, :64]).mean() > 1
        tilted = normals.copy()
        angle = np.deg2rad(60)
        tilted[:, 0] = normals[:, 0] * np.cos(angle) + normals[:, 2] * np.sin(angle)
        tilted[:, 2] = -normals[:, 0] * np.sin(angle) + normals[:, 2] * np.cos(angle)
        first.update_geometry(positions, tilted)
        relit = capture()
        normal_error = np.abs(relit[:, :64] - restored[:, :64]).mean()
        assert normal_error > 1, f"Normal update did not change lighting: {normal_error}"
        np.testing.assert_array_equal(mesh.normals, normals)
    finally:
        if spy is not None:
            spy.stop()
        app.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--renderer", choices=("recording", "native", "ovrtx"), default="recording")
    backend = parser.parse_args().renderer
    if backend == "recording":
        check_delivery()
    else:
        check_rendering(backend)
    print(f"PASS: {backend} dynamic geometry")
