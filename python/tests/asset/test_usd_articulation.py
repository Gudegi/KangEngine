"""USD physics import without a Python USD dependency or GPU."""
from pathlib import Path

import numpy as np
import pytest

from kangengine import asset


USD = '''#usda 1.0
(
    defaultPrim = "Robot"
    metersPerUnit = 1
    upAxis = "Z"
)
def Xform "Prototype" {
    def Mesh "mesh" (prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]) {
        point3f[] points = [(0,0,0), (1,0,0), (0,1,0), (0,0,1)]
        int[] faceVertexCounts = [3,3,3,3]
        int[] faceVertexIndices = [0,2,1, 0,1,3, 0,3,2, 1,2,3]
        token physics:approximation = "convexHull"
        bool physics:collisionEnabled = true
    }
}
def Xform "Robot" {
    def Xform "base" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        float physics:mass = 2
        float3 physics:diagonalInertia = (1,2,3)
        point3f physics:centerOfMass = (0.1,0.2,0.3)
        quatf physics:principalAxes = (1,0,0,0)
    }
    def Xform "tip" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {
        float physics:mass = 1
        point3f physics:centerOfMass = (0,0,0)
        quatf physics:principalAxes = (1,0,0,0)
        float3 physics:diagonalInertia = (0.1,0.2,0.3)
        double3 xformOp:translate = (10,20,30)
        uniform token[] xformOpOrder = ["xformOp:translate"]
        def Xform "geometry" (
            prepend references = </Prototype>
            instanceable = true
        ) {
            double3 xformOp:translate = (2,0,0)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }
    }
    def PhysicsRevoluteJoint "hinge" {
        rel physics:body0 = </Robot/base>
        rel physics:body1 = </Robot/tip>
        point3f physics:localPos0 = (0,0,3)
        point3f physics:localPos1 = (0,0,0)
        quatf physics:localRot0 = (1,0,0,0)
        quatf physics:localRot1 = (0.70710678,0.70710678,0,0)
        uniform token physics:axis = "Y"
        float physics:lowerLimit = -90
        float physics:upperLimit = 45
        float drive:angular:physics:stiffness = 20
        float drive:angular:physics:damping = 2
        float drive:angular:physics:maxForce = 30
    }
}
'''


def load(tmp_path: Path, text=USD):
    path = tmp_path / "robot.usda"
    path.write_text(text)
    try:
        return asset.USDLoader.parse_articulation(str(path))
    except RuntimeError as error:
        if "USD support not compiled" in str(error):
            pytest.skip(str(error))
        raise


def xyz(v):
    return np.array([v.x, v.y, v.z])


def with_material(shader, *, mdl=False, textures=""):
    surface = "mdl:surface" if mdl else "surface"
    text = USD.replace(
        '"PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"',
        '"PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"',
    ).replace(
        'bool physics:collisionEnabled = true',
        '''bool physics:collisionEnabled = true
        rel material:binding = </Paint>
        texCoord2f[] primvars:st = [(0,0), (1,0), (0,1), (1,1)] (
            interpolation = "vertex"
        )''',
    )
    return text + f'''
def Material "Paint" {{
    token outputs:{surface}.connect = </Paint/Shader.outputs:out>
    def Shader "Shader" {{
        {shader}
        token outputs:out
    }}
    {textures}
}}
'''


@pytest.mark.parametrize("mdl", [False, True])
@pytest.mark.parametrize("authored_factors", [False, True])
def test_visual_material_preserves_authored_color(tmp_path, mdl, authored_factors):
    surface = "mdl:surface" if mdl else "surface"
    shader = (
        'uniform token info:implementationSource = "sourceAsset"\n'
        'uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@\n'
        'uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"\n'
        'color3f inputs:diffuse_color_constant = (0.7, 0.2, 0.1)\n'
        'float inputs:metallic_constant = 0.3\n'
        'float inputs:reflection_roughness_constant = 0.6\n'
        if mdl else
        'uniform token info:id = "UsdPreviewSurface"\n'
        'color3f inputs:diffuseColor = (0.7, 0.2, 0.1)\n'
        'float inputs:metallic = 0.3\n'
        'float inputs:roughness = 0.6\n'
    )
    if not authored_factors:
        shader = "\n".join(line for line in shader.splitlines() if not line.startswith("float inputs:"))
    text = USD.replace(
        '"PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"',
        '"PhysicsCollisionAPI", "PhysicsMeshCollisionAPI", "MaterialBindingAPI"',
    ).replace(
        'bool physics:collisionEnabled = true',
        'bool physics:collisionEnabled = true\nrel material:binding = </Paint>',
    )
    text += f'''\ndef Material "Paint" {{
        token outputs:{surface}.connect = </Paint/Shader.outputs:out>
        def Shader "Shader" {{
            {shader}
            token outputs:out
        }}
    }}\n'''
    geom = load(tmp_path, text).articulation.visual_geoms[0]
    np.testing.assert_allclose(geom.rgba, [1, 1, 1, 1])
    np.testing.assert_allclose(xyz(geom.material.base_color), [.7, .2, .1])
    assert geom.material.metallic == pytest.approx(.3 if authored_factors else 0.)
    assert geom.material.roughness == pytest.approx(.6 if authored_factors else .5)


@pytest.mark.parametrize("color_space", ["auto", "sRGB", "raw"])
def test_omnipbr_direct_texture_replaces_constant_color(tmp_path, color_space):
    texture = tmp_path / "paint.png"
    texture.touch()
    text = with_material(f'''
        uniform token info:implementationSource = "sourceAsset"
        uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
        uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
        color3f inputs:diffuse_color_constant = (0.2,0.2,0.2)
        color3f inputs:diffuse_tint = (0.8,0.6,0.4)
        asset inputs:diffuse_texture = @paint.png@ (colorSpace = "{color_space}")
    ''', mdl=True)
    result = load(tmp_path, text)
    geom = result.articulation.visual_geoms[0]
    material = geom.material
    assert material.texture_paths == {"base_color_texture": str(texture)}
    assert material.texture_srgb["base_color_texture"] == (color_space != "raw")
    assert material.base_color_texture is None
    np.testing.assert_allclose(xyz(material.base_color), [.8, .6, .4])
    assert np.asarray(geom.mesh_data.uvs).max() == 1
    assert not any("texture" in message.lower() for message in result.diagnostics.warnings)

    # Static scene imports use the same material reader.
    scene = asset.USDLoader.parse(str(tmp_path / "robot.usda"))
    assert all(mesh.diffuse_texture_path == str(texture) for mesh in scene.meshes)


@pytest.mark.parametrize("missing_file", [False, True])
def test_omnipbr_default_color_and_missing_texture(tmp_path, missing_file):
    texture_input = 'asset inputs:diffuse_texture = @missing.png@' if missing_file else ""
    text = with_material(f'''
        uniform token info:implementationSource = "sourceAsset"
        uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
        uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
        {texture_input}
    ''', mdl=True)
    result = load(tmp_path, text)
    material = result.articulation.visual_geoms[0].material
    np.testing.assert_allclose(xyz(material.base_color), [.2, .2, .2])
    assert material.texture_paths == {}
    assert any("texture file missing" in msg for msg in result.diagnostics.warnings) == missing_file


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("color_space", ["sRGB", "raw"])
def test_preview_surface_articulation_texture_slots(tmp_path, packed, color_space):
    texture = tmp_path / "paint.png"
    texture.touch()
    roughness, metallic = ("g", "b") if packed else ("r", "r")
    text = with_material(f'''
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor.connect = </Paint/Texture.outputs:rgb>
        color3f inputs:emissiveColor.connect = </Paint/Texture.outputs:rgb>
        normal3f inputs:normal.connect = </Paint/Normal.outputs:rgb>
        float inputs:roughness.connect = </Paint/Texture.outputs:{roughness}>
        float inputs:metallic.connect = </Paint/Texture.outputs:{metallic}>
    ''', textures=f'''
    def Shader "Texture" {{
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @paint.png@
        token inputs:sourceColorSpace = "{color_space}"
        float4 inputs:scale = (0.2,0.4,0.6,1)
        float3 outputs:rgb
        float outputs:r
        float outputs:g
        float outputs:b
    }}
    def Shader "Normal" {{
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @paint.png@
        float4 inputs:scale = (2,2,2,1)
        float4 inputs:bias = (-1,-1,-1,0)
        float3 outputs:rgb
    }}
    ''')
    material = load(tmp_path, text).articulation.visual_geoms[0].material
    slots = {"base_color_texture", "emissive_texture", "normal_texture"}
    slots |= {"metallic_roughness_texture"} if packed else {"metallic_texture", "roughness_texture"}
    assert material.texture_paths == {slot: str(texture) for slot in slots}
    assert material.texture_srgb == {
        "base_color_texture": color_space != "raw",
        "emissive_texture": color_space != "raw",
    }
    np.testing.assert_allclose(xyz(material.base_color), [.2, .4, .6])
    assert material.roughness == pytest.approx(.4 if packed else .2)
    assert material.metallic == pytest.approx(.6 if packed else .2)


def test_joint_frames_mass_units_and_instance_mesh(tmp_path):
    result = load(tmp_path)
    data = result.articulation
    assert data.skeleton_tree.node_names() == ["base", "tip"]
    np.testing.assert_allclose(xyz(data.skeleton_tree.local_translation(1)), [0, 0, 3])
    joint = data.joints[1][0]
    np.testing.assert_allclose(xyz(joint.axis), [0, 0, 1], atol=1e-6)
    assert joint.lo_limit == pytest.approx(-np.pi / 2)
    assert joint.hi_limit == pytest.approx(np.pi / 4)
    assert (joint.kp, joint.kd, joint.effort_limit) == (20, 2, 30)
    assert data.inertials[0].mass == 2
    np.testing.assert_allclose(xyz(data.inertials[0].com), [.1, .2, .3])
    assert len(data.visual_geoms) == 1
    hull = data.collision_geoms[1][0]
    vertices = np.asarray(hull.mesh_data.vertices)
    np.testing.assert_allclose(vertices.min(axis=0), [2, 0, 0])
    np.testing.assert_allclose(vertices.max(axis=0), [3, 1, 1])
    assert result.diagnostics.warnings


@pytest.mark.parametrize("old,new,error", [
    ('metersPerUnit = 1', 'metersPerUnit = 0.01', 'meter units'),
    ('point3f physics:localPos1 = (0,0,0)', 'point3f physics:localPos1 = (1,0,0)', 'child joint anchor'),
    ('PhysicsRevoluteJoint', 'PhysicsSphericalJoint', 'Unsupported USD joint'),
    ('"convexHull"', '"convexDecomposition"', 'convexHull'),
    ('float physics:mass = 2', 'float physics:mass = 0', 'positive USD mass'),
])
def test_unsupported_physics_fails_explicitly(tmp_path, old, new, error):
    with pytest.raises(RuntimeError, match=error):
        load(tmp_path, USD.replace(old, new))


def test_prismatic_uses_meters_and_fixed_joint_has_no_dof(tmp_path):
    result = load(tmp_path, USD.replace("PhysicsRevoluteJoint", "PhysicsPrismaticJoint"))
    joint = result.articulation.joints[1][0]
    assert joint.lo_limit == -90
    assert joint.hi_limit == 45
    fixed = load(tmp_path, USD.replace("PhysicsRevoluteJoint", "PhysicsFixedJoint"))
    assert not fixed.articulation.joints
    assert fixed.articulation.skeleton_tree.parent_index(1) == 0


def test_world_cache_and_visual_bridge_share_usd_descriptor(tmp_path):
    from kangengine import sim, visual

    load(tmp_path)
    path = str(tmp_path / "robot.usda")
    world = sim.KangSimWorld(num_envs=1, add_ground=False)
    try:
        data = world.load_usd(path)
        assert world.load_usd(path) is data
        selected = world.load_usd(path, prim_path="/Robot")
        assert selected is not data
        assert world.get_articulation_asset_cache_size() == 2
        bridge = visual.ArticulationVisualAsset.from_data(data)
        assert bridge.num_bodies() == 2
    finally:
        world.release()
