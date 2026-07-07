"""Validate LightComponent lifecycle without requiring a GL window."""

from __future__ import annotations

import math

import kangengine as ke


def _close_vec3(actual, expected, eps=1.0e-5):
    vals = [float(actual.x), float(actual.y), float(actual.z)]
    for i, (a, b) in enumerate(zip(vals, expected)):
        if abs(a - b) > eps:
            raise AssertionError(f"vec3 value {i} mismatch: {a} != {b}")


def main():
    scene = ke.scene.create_backend(ke.scene.BackendType.Native)

    point_prim = scene.define_prim("/lights/test_point", ke.scene.PrimType.Light)
    point = ke.PointLight()
    point.position = ke.vec3(1.0, 2.0, 3.0)
    point.color = ke.vec3(0.25, 0.5, 1.0)
    point.intensity = 2.5
    point.range = 12.0
    point_prim.set_point_light(point)

    component = point_prim.get_light_component()
    if component is None or not component.attached:
        raise AssertionError("set_point_light did not attach LightComponent")
    if component.type != ke.scene.LightType.Point:
        raise AssertionError("unexpected LightComponent type")
    if component.owner is not point_prim:
        raise AssertionError("LightComponent owner mismatch")
    if "LightComponent" not in repr(component):
        raise AssertionError("LightComponent repr missing type name")

    resolved_point = component.point_light()
    _close_vec3(resolved_point.position, [1.0, 2.0, 3.0])
    _close_vec3(point_prim.get_point_light().position, [1.0, 2.0, 3.0])

    version = component.version
    point.range = 5.0
    component.set_point_light(point)
    if component.version <= version:
        raise AssertionError("LightComponent version did not advance")

    spot_prim = scene.define_prim("/lights/test_spot", ke.scene.PrimType.Light)
    spot = ke.SpotLight()
    spot.position = ke.vec3(-1.0, 0.0, 2.0)
    spot.direction = ke.vec3(0.0, -1.0, 0.0)
    spot.color = ke.vec3(1.0, 0.5, 0.25)
    spot.intensity = 1.5
    spot.range = 8.0
    spot.inner_cone_angle = math.radians(10.0)
    spot.outer_cone_angle = math.radians(25.0)
    spot_prim.set_spot_light(spot)
    spot_component = spot_prim.get_light_component()
    if spot_component.type != ke.scene.LightType.Spot:
        raise AssertionError("unexpected spot LightComponent type")
    _close_vec3(spot_component.spot_light().position, [-1.0, 0.0, 2.0])

    directional_prim = scene.define_prim(
        "/lights/test_directional", ke.scene.PrimType.Light
    )
    directional = ke.DirectionalLight()
    directional.direction = ke.vec3(0.0, 0.0, -1.0)
    directional.color = ke.vec3(1.0, 1.0, 0.75)
    directional.intensity = 0.8
    directional.ambient = ke.vec3(0.1, 0.1, 0.12)
    directional_prim.set_directional_light(directional)
    directional_component = directional_prim.get_light_component()
    if directional_component.type != ke.scene.LightType.Directional:
        raise AssertionError("unexpected directional LightComponent type")
    _close_vec3(directional_component.directional_light().direction, [0.0, 0.0, -1.0])

    if not point_prim.remove_light_component():
        raise AssertionError("remove_light_component returned false")
    if point_prim.get_light_component() is not None:
        raise AssertionError("LightComponent still attached after removal")
    try:
        component.point_light()
    except RuntimeError:
        pass
    else:
        raise AssertionError("detached LightComponent remained usable")

    mesh_prim = scene.define_prim("/not_a_light", ke.scene.PrimType.Xform)
    try:
        mesh_prim.add_light_component()
    except RuntimeError:
        pass
    else:
        raise AssertionError("non-Light prim accepted LightComponent")

    child = scene.define_prim("/lights/subtree/child", ke.scene.PrimType.Light)
    child_component = child.add_light_component()
    if not scene.remove_prim("/lights/subtree"):
        raise AssertionError("failed to remove light subtree")
    if child_component.attached:
        raise AssertionError("LightComponent stayed attached after subtree removal")

    print("PASS: LightComponent smoke completed")


if __name__ == "__main__":
    main()
