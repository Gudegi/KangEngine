"""Validate static input identity, ownership and rejection without a GPU context."""

import gc
import numpy as np
import kangengine as ke
from kangengine._core import _ke


def main():
    scene = ke.scene.create_backend(ke.scene.BackendType.NATIVE)
    resources = ke.scene.SceneResourceManager(scene)
    camera = scene.define_prim("/camera", ke.scene.PrimType.CAMERA)
    camera.add_camera_component()
    mesh = ke.geometry.create_cube_data()
    mesh_id = resources.register_mesh("Cube", mesh)
    texture_id = resources.register_texture_source("Image", "unused.png", srgb=True)
    material = ke.material.PBRMaterial()
    material.base_color = ke.Vec4(0.2, 0.4, 0.6, 1)
    material.emissive_color = ke.Vec3(0.1, 0.2, 0.3)
    material.emissive_strength = 2
    material.texture_resources = {"base_color_texture": texture_id}
    material_id = resources.register_material("Surface", material.native)

    def add():
        prim = scene.define_prim("/cube", ke.scene.PrimType.MESH)
        prim.set_mesh_data(mesh)
        prim.set_local_translation(ke.Vec3(1, 2, 3))
        prim.get_mesh_component().resource_handle = mesh_id
        prim.set_material(resources.material(material_id))
        prim.add_render_component()
        return prim

    prim = add()
    del material
    gc.collect()
    collector = _ke._RenderSceneDataCollector()

    def collect():
        return collector.collect(scene, resources, "/camera")["objects"][0]

    initial = collect()
    expected_world = np.eye(4, dtype=np.float32)
    expected_world[:3, 3] = [1, 2, 3]
    assert initial["world"].dtype == np.float32
    assert initial["world"].flags.c_contiguous
    np.testing.assert_array_equal(initial["world"], expected_world)
    assert isinstance(initial["material"]["base_color"], list)
    assert isinstance(initial["material"]["emissive"], list)
    np.testing.assert_allclose(initial["material"]["base_color"], [0.2, 0.4, 0.6, 1])
    np.testing.assert_allclose(initial["material"]["emissive"], [0.2, 0.4, 0.6])
    assert initial["material_id"] == material_id
    assert initial["textures"]["base_color_texture"]["srgb"]
    assert resources.usage_count(texture_id) == 1
    prim.set_display_color(ke.Vec3(0.3, 0.6, 0.9))
    tinted = collect()
    np.testing.assert_allclose(tinted["color"], [0.3, 0.6, 0.9, 1])
    assert tinted["id"] == initial["id"]
    assert tinted["material"] == initial["material"]
    prim.get_render_component().display_color = ke.Vec4(0.9, 0.6, 0.3, 0.5)
    np.testing.assert_allclose(collect()["color"], [0.9, 0.6, 0.3, 0.5])
    prim.set_visible(False)
    assert not collect()["visible"]
    prim.set_active(False)
    prim.set_visible(True)
    assert not collect()["visible"] and collect()["id"] == initial["id"]
    prim.set_active(True)
    assert collect()["visible"] and collect()["id"] == initial["id"]
    scene.remove_prim("/cube")
    add()
    assert collect()["id"] != initial["id"]
    resources.clear()
    assert resources.register_mesh("New", mesh) > texture_id
    try:
        collect()
    except ValueError:
        pass
    else:
        raise AssertionError("retired resource handles resolved after clear")
    print("PASS: static render source identity, material lifetime and retired handles")


if __name__ == "__main__":
    main()
