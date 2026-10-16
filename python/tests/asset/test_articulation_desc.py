"""Shared visual descriptions retain geometry and material ownership."""

import gc
import weakref

import numpy as np
import pytest

import kangengine as ke


def test_rigid_and_articulation_share_context_free_visual_parts():
    mesh = ke.geometry.create_box_data(1.0, 1.0, 1.0)
    material = ke.material.PBRMaterial()
    material.base_color = ke.Vec4(0.8, 0.2, 0.1, 0.6)
    material.roughness = 0.35
    material.texture_paths = {"base_color_texture": "textures/paint.png"}
    visual = ke.asset.VisualGeomDesc(
        mesh_data=mesh,
        material=material,
        position=[0.1, 0.2, 0.3],
        rgba=[1.0, 1.0, 1.0, 1.0],
    )
    tree = ke.animation.SkeletonTree(
        ["body"], [-1], np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0, 0.0]])
    )
    articulated = ke.asset.ArticulationDesc(
        tree,
        joints={},
        collision_geoms={},
        inertials={},
        visual_geoms=[visual],
        asset_dir="shared_assets",
    )
    rigid = ke.asset.RigidDesc(visual_geoms=[visual], asset_dir="shared_assets")

    for description in (rigid, articulated):
        part = description.visual_geoms[0]
        assert part.mesh_data is mesh
        assert part.material is material.native
        assert part.material.base_color_texture is None
        np.testing.assert_allclose(
            [part.pos.x, part.pos.y, part.pos.z], [0.1, 0.2, 0.3]
        )
        assert description.asset_dir == "shared_assets"
    material.roughness = 0.42
    assert rigid.visual_geoms[0].material.roughness == pytest.approx(0.42)
    assert articulated.visual_geoms[0].material.roughness == pytest.approx(0.42)
    assert rigid.visual_geoms[0].material.texture_paths == {
        "base_color_texture": "textures/paint.png"
    }


def test_visual_material_setter_accepts_public_and_native_materials():
    material = ke.material.PhongMaterial()
    material.diffuse = ke.Vec3(0.4, 0.5, 0.6)
    material.texture_paths = {"diffuse_map": "paint.png"}
    visual = ke.asset.VisualGeomDesc(material=material)
    assert visual.material is material.native
    assert visual.material.diffuse_map is None
    np.testing.assert_allclose(visual.rgba, [1.0, 1.0, 1.0, 1.0])

    replacement = ke.material.PBRMaterial()
    visual.material = replacement
    assert visual.material is replacement.native
    visual.material = material.native
    assert visual.material is material.native


def test_material_survives_python_owners_and_articulation_template_copy():
    material = ke.material.PBRMaterial()
    material.metallic = 0.37
    material.texture_paths = {"normal_texture": "not_loaded_without_an_app.png"}
    wrapper_reference = weakref.ref(material)
    native = material.native
    visual = ke.asset.VisualGeomDesc(material=material)
    tree = ke.animation.SkeletonTree(
        ["body"], [-1], np.zeros((1, 3)), np.array([[1.0, 0.0, 0.0, 0.0]])
    )
    source = ke.asset.ArticulationDesc(
        tree, joints={}, collision_geoms={}, inertials={}, visual_geoms=[visual]
    )
    rigid = ke.asset.RigidDesc.from_articulation(source)
    template = ke.physics.ArticulationTemplate.create(source)
    del material, native, visual, source
    gc.collect()

    assert wrapper_reference() is None
    copied = template.source_data.visual_geoms[0].material
    assert copied is rigid.visual_geoms[0].material
    assert copied.metallic == pytest.approx(0.37)
    assert copied.normal_texture is None
    assert copied.texture_paths == {"normal_texture": "not_loaded_without_an_app.png"}


def test_multibody_articulation_cannot_be_converted_to_one_rigid():
    tree = ke.animation.SkeletonTree(
        ["base", "child"],
        [-1, 0],
        np.zeros((2, 3)),
        np.array([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]]),
    )
    description = ke.asset.ArticulationDesc(
        tree, joints={}, collision_geoms={}, inertials={}
    )
    with pytest.raises(ValueError):
        ke.asset.RigidDesc.from_articulation(description)
