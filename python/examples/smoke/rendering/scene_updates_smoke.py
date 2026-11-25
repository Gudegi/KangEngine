"""Check incremental scene delivery, identity, and retry without a renderer."""

from pathlib import Path
import sys

import numpy as np
import kangengine as ke
from kangengine._core import _ke

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "rendering"))
from static_scene_renderer import build_scene


def check_app_retry():
    class Adapter:
        reject = False
        frame = None

        def render(self, frame):
            self.frame = frame
            if self.reject:
                raise RuntimeError("Rejected test frame")
            return np.zeros((8, 8, 3), dtype=np.uint8)

        def close(self):
            pass

    adapter = Adapter()
    app = ke.App()
    try:
        app._initialize_scene_renderer(8, 8, ke.UpAxis.Y, ke.scene.BackendType.NATIVE,
                                      lambda width, height: adapter)
        material = app.create_pbr_material()
        cube = app.scene.add_mesh(path="/cube", mesh_data=ke.geometry.create_cube_data(1),
                                  material=material)
        app.draw_frame()
        instance_id = adapter.frame["objects"][0]["id"]
        app.draw_frame()
        assert not adapter.frame["objects"]
        cube.set_local_translation(ke.Vec3(1, 0, 0))
        adapter.reject = True
        try:
            app.draw_frame()
        except RuntimeError as error:
            assert "Rejected test frame" in str(error)
        else:
            raise AssertionError("Consumer failure did not propagate")
        adapter.reject = False
        app.draw_frame()
        assert [item["id"] for item in adapter.frame["objects"]] == [instance_id]
        app.draw_frame()
        assert not adapter.frame["objects"]
    finally:
        app.close()
    print("PASS: App retries rejected updates and accepts successful submissions")


def main():
    scene, resources, material, add_cube = build_scene(
        Path(ke.__file__).parent / "assets/textures/awesomeface.png",
    )
    second = add_cube("/group/second")
    collector = _ke._RenderSceneDataCollector()

    def collect():
        return collector.collect(scene, resources, "/camera")

    first = collect()
    ids = {item["path"]: item["id"] for item in first["objects"]}
    assert len(ids) == 2 and len(first["lights"]) == 1
    collector.accept()
    idle = collect()
    assert idle["objects"] == [] and idle["removed_objects"] == [] and idle["lights"] is None

    # Shared material values can change without a component version change.
    versions = [scene.get_prim_at_path(path).get_material_binding_component().version
                for path in ids]
    for field, value, exported, expected in (
        ("base_color", ke.Vec4(0.3, 0.5, 0.7, 1), "base_color", [0.3, 0.5, 0.7, 1]),
        ("roughness", 0.25, "roughness", 0.25),
        ("emissive_color", ke.Vec3(0.1, 0.2, 0.3), "emissive", [0, 0, 0]),
        ("emissive_strength", 2.0, "emissive", [0.2, 0.4, 0.6]),
    ):
        setattr(material.native, field, value)
        changed = collect()["objects"]
        assert {item["id"] for item in changed} == set(ids.values()), field
        for item in changed:
            np.testing.assert_allclose(item["material"][exported], expected)
        collector.accept()
        setattr(material.native, field, value)
        assert not collect()["objects"], field
    assert versions == [scene.get_prim_at_path(path).get_material_binding_component().version
                        for path in ids]

    for enabled in (False, True):
        second.get_render_component().casts_shadow = enabled
        changed = collect()["objects"]
        assert len(changed) == 1 and changed[0]["id"] == ids["/group/second"]
        assert changed[0]["casts_shadow"] is enabled
        collector.accept()
        assert not collect()["objects"]

    original_textures = material.texture_resources
    replacement = resources.register_texture_source(
        "Replacement", str(Path(ke.__file__).parent / "assets/textures/awesomeface.png"),
        srgb=False,
    )
    for textures in ({"base_color_texture": replacement}, {}, original_textures):
        material.texture_resources = textures
        changed = collect()["objects"]
        assert {item["id"] for item in changed} == set(ids.values())
        for item in changed:
            assert {slot: source["id"] for slot, source in item["textures"].items()} == textures
        # Until accepted, material changes must remain available for retry.
        assert {item["id"] for item in collect()["objects"]} == set(ids.values())
        collector.accept()
        assert not collect()["objects"]

    second.set_display_color(ke.Vec3(0.2, 0.4, 0.6))
    color_update = collect()["objects"]
    assert [item["id"] for item in color_update] == [ids["/group/second"]]
    np.testing.assert_allclose(color_update[0]["color"], [0.2, 0.4, 0.6, 1])
    collector.accept()
    assert not collect()["objects"]

    second.set_local_translation(ke.Vec3(2, 0, 0))
    changed = collect()
    assert [item["id"] for item in changed["objects"]] == [ids["/group/second"]]
    # A failed consumer does not accept: the same change must be retried.
    retried = collect()
    assert [item["id"] for item in retried["objects"]] == [ids["/group/second"]]
    np.testing.assert_array_equal(changed["objects"][0]["world"], retried["objects"][0]["world"])
    collector.accept()

    scene.get_prim_at_path("/group").set_local_translation(ke.Vec3(1, 0, 0))
    assert len(collect()["objects"]) == 2
    collector.accept()
    material.metallic = 0.8
    updated = collect()["objects"]
    assert len(updated) == 2 and all(abs(item["material"]["metallic"] - 0.8) < 1e-6 for item in updated)
    collector.accept()

    parent = scene.get_prim_at_path("/group")
    parent.set_active(False)
    scene.define_prim("/unrelated", ke.scene.PrimType.XFORM)
    inactive = collect()
    assert len(inactive["objects"]) == 2 and not inactive["removed_objects"]
    assert all(not item["visible"] for item in inactive["objects"])
    collector.accept()
    parent.set_active(True)
    assert {item["id"] for item in collect()["objects"]} == set(ids.values())
    collector.accept()

    scene.remove_prim("/group/second")
    recreated = add_cube("/group/second")
    replaced = collect()
    assert replaced["removed_objects"] == [ids["/group/second"]]
    assert len(replaced["objects"]) == 1
    assert replaced["objects"][0]["id"] not in ids.values()
    collector.accept()

    recreated.remove_render_component()
    removed = collect()
    assert len(removed["removed_objects"]) == 1 and not removed["objects"]
    collector.accept()
    recreated.add_render_component()
    assert len(collect()["objects"]) == 1
    collector.accept()

    mesh = ke.geometry.create_cube_data(0.5)
    handle = resources.register_mesh("Replacement", mesh)
    recreated.set_mesh_data(mesh)
    recreated.get_mesh_component().resource_handle = handle
    update = collect()
    assert len(update["objects"]) == 1 and update["objects"][0]["mesh_id"] == handle
    collector.accept()

    scene.get_prim_at_path("/light").set_visible(False)
    assert collect()["lights"] == []
    collector.accept()
    assert collect()["lights"] is None
    scene.get_prim_at_path("/light").set_visible(True)
    assert len(collect()["lights"]) == 1
    print("PASS: unchanged delivery, local/parent/material updates, retry, identity, resources, and lights")
    check_app_retry()


if __name__ == "__main__":
    main()
