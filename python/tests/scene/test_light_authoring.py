"""Light edits preserve local authoring under transformed scene hierarchies."""

import math

import numpy as np
import pytest

import kangengine as ke


def xyz(value):
    return [value.x, value.y, value.z]


@pytest.mark.parametrize("kind", ["directional", "point", "spot"])
@pytest.mark.parametrize("matrix_authored", [False, True])
def test_local_light_round_trip(kind, matrix_authored):
    scene = ke.scene.create_backend(ke.scene.BackendType.NATIVE)
    parent = scene.define_prim("/lights/rig", ke.scene.PrimType.XFORM)
    parent.set_local_translation(ke.Vec3(4, 5, 6))
    parent.set_local_rotation_axis_angle(ke.Vec3(0, 0, 1), math.pi / 2)
    parent.set_local_scale(ke.Vec3(2, 3, 4))
    prim = scene.define_prim("/lights/rig/light", ke.scene.PrimType.LIGHT)
    value_type = {
        "directional": ke.scene.DirectionalLight,
        "point": ke.scene.PointLight,
        "spot": ke.scene.SpotLight,
    }[kind]
    light = value_type()
    if kind != "directional":
        light.position = ke.Vec3(1, 2, 3)
    if kind != "point":
        light.direction = ke.Vec3(0, -1, 0)
    setter = getattr(prim, f"set_{kind}_light")
    getter = getattr(prim, f"get_{kind}_light")
    setter(light)
    prim.set_local_rotation_axis_angle(ke.Vec3(1, 0, 0), math.pi / 4)
    if matrix_authored:
        # Distinct from the prim's previous TRS attributes: emission edits must
        # not switch the transform back to those stale attributes.
        prim.set_local_matrix(parent.compute_local_matrix())
    local_before = np.asarray(prim.compute_local_matrix()).copy()
    world_before = np.asarray(prim.compute_world_matrix()).copy()
    position_before = xyz(prim.get_local_translation())
    component_getter = getattr(prim.get_light_component(), f"{kind}_light")

    for read in (getter, component_getter):
        for _ in range(3):
            edited = read()
            if kind != "directional":
                np.testing.assert_allclose(xyz(edited.position), position_before)
            if kind != "point":
                np.testing.assert_allclose(xyz(edited.direction), [0, -1, 0])
            edited.intensity += 1
            setter(edited)
            np.testing.assert_allclose(np.asarray(prim.compute_local_matrix()), local_before)
            np.testing.assert_allclose(np.asarray(prim.compute_world_matrix()), world_before)

    # Parent edits affect world placement, never local getter values.
    parent.set_local_translation(ke.Vec3(8, 9, 10))
    assert not np.allclose(np.asarray(prim.compute_world_matrix()), world_before)
    if kind != "directional":
        np.testing.assert_allclose(xyz(getter().position), position_before)
    if kind != "point":
        np.testing.assert_allclose(xyz(getter().direction), [0, -1, 0])

    # Prim's attribute-backed path must obey the same local contract.
    assert prim.remove_light_component()
    if kind != "directional":
        np.testing.assert_allclose(xyz(getter().position), position_before)
    if kind != "point":
        np.testing.assert_allclose(xyz(getter().direction), [0, -1, 0])


@pytest.mark.parametrize("kind", ["point", "spot"])
def test_position_tracks_prim_edits(kind):
    scene = ke.scene.create_backend(ke.scene.BackendType.NATIVE)
    parent = scene.define_prim("/lights/rig", ke.scene.PrimType.XFORM)
    parent.set_local_translation(ke.Vec3(10, 0, 0))
    prim = scene.define_prim("/lights/rig/light", ke.scene.PrimType.LIGHT)
    light = ke.scene.PointLight() if kind == "point" else ke.scene.SpotLight()
    setter = getattr(prim, f"set_{kind}_light")
    getter = getattr(prim, f"get_{kind}_light")
    setter(light)
    prim.set_local_translation(ke.Vec3(1, 2, 3))
    np.testing.assert_allclose(xyz(getter().position), [1, 2, 3])
    light = getter()
    light.position = ke.Vec3(4, 5, 6)
    setter(light)
    np.testing.assert_allclose(xyz(prim.get_local_translation()), [4, 5, 6])
    np.testing.assert_allclose(xyz(prim.get_world_translation()), [14, 5, 6])
