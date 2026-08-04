"""Validate Python App helpers use the C++ SceneResourceManager."""

from __future__ import annotations

import kangengine as ke
from kangengine.app import App


class _DummyApp:
    pass


def main():
    app = _DummyApp()
    app.backend = ke.scene.create_backend(ke.scene.BackendType.NATIVE)
    app.resources = ke.scene.SceneResourceManager(app.backend)
    app._resource_handles_by_object_id = {}
    app._resource_counter = 0
    app._remember_resource_handle = App._remember_resource_handle.__get__(app)
    app._existing_resource_handle = App._existing_resource_handle.__get__(app)
    app._next_resource_name = App._next_resource_name.__get__(app)
    app._register_material_resource = App._register_material_resource.__get__(app)

    material = ke.material.PhongMaterial()
    handle = app._register_material_resource(material)
    prim = app.resources.resource_prim(handle)
    if prim is None or prim.get_type() != ke.scene.PrimType.RESOURCE:
        raise AssertionError("_register_material_resource did not mirror a prim")
    component = prim.get_resource_component()
    if component is None:
        raise AssertionError("Resource prim has no ResourceComponent")
    if component.type != ke.scene.ResourceType.MATERIAL:
        raise AssertionError("ResourceComponent type mismatch")
    if component.handle != handle:
        raise AssertionError("ResourceComponent handle mismatch")

    same = app._register_material_resource(material)
    if same != handle:
        raise AssertionError("registering same material should return same handle")

    shader = ke.scene.ShaderSourceResource()
    shader.stage = ke.render.ShaderType.FRAGMENT
    shader.language = ke.scene.ShaderLanguage.GLSL
    shader.source = "#version 410 core\nvoid main() {}"
    shader_handle = app.resources.register_shader_source(
        "Smoke Fragment", shader, "memory://smoke.fs"
    )
    shader_prim = app.resources.resource_prim(shader_handle)
    if shader_prim is None or not shader_prim.get_path().startswith(
        "/.Resources/ShaderSources/"
    ):
        raise AssertionError("shader source resource was not mirrored")

    print("PASS: Python resource manager smoke completed")


if __name__ == "__main__":
    main()
