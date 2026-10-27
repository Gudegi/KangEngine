"""USD render snapshots preserve display semantics without importing physics."""

from pathlib import Path

import numpy as np
import pytest

import kangengine as ke
from kangengine import asset


TRIANGLE = """
    point3f[] points = [(0,0,0), (1,0,0), (0,1,0)]
    int[] faceVertexCounts = [3]
    int[] faceVertexIndices = [0,1,2]
"""


def parse(tmp_path: Path, body: str, *, metadata: str = "", scale: float = 1):
    path = tmp_path / "scene.usda"
    path.write_text(f"#usda 1.0\n{metadata}\n{body}")
    try:
        return asset.USDLoader.parse(usd_path=str(path), scale=scale)
    except RuntimeError as exc:
        if "USD support not compiled" in str(exc):
            pytest.skip(str(exc))
        raise


def xyz(vector):
    return [vector.x, vector.y, vector.z]


def test_visibility_purpose_instances_and_collision_visuals(tmp_path):
    result = parse(tmp_path, f'''
def Xform "Hidden" {{
    token visibility = "invisible"
    def Mesh "Mesh" {{ {TRIANGLE} }}
}}
def Xform "Guide" {{
    uniform token purpose = "guide"
    def Mesh "Mesh" {{ {TRIANGLE} }}
}}
def Mesh "Proxy" {{ uniform token purpose = "proxy" {TRIANGLE} }}
def Mesh "Inactive" (active = false) {{ {TRIANGLE} }}
def Mesh "RenderableCollider" (prepend apiSchemas = ["PhysicsCollisionAPI"]) {{
    uniform token purpose = "render"
    bool doubleSided = true
    {TRIANGLE}
}}
def Xform "Prototype" {{
    def Mesh "Mesh" {{ {TRIANGLE} }}
}}
def Xform "Instance" (
    prepend references = </Prototype>
    instanceable = true
) {{
    double3 xformOp:translate = (10,20,30)
    uniform token[] xformOpOrder = ["xformOp:translate"]
}}
''')
    meshes = {mesh.prim_path: mesh for mesh in result.meshes}
    assert set(meshes) == {"/RenderableCollider", "/Prototype/Mesh", "/Instance/Mesh"}
    assert meshes["/RenderableCollider"].double_sided
    lower, upper = meshes["/Instance/Mesh"].bounds
    np.testing.assert_allclose(xyz(lower), [10, 20, 30])
    np.testing.assert_allclose(xyz(upper), [11, 21, 30])


def test_constant_preview_surface_and_stage_camera(tmp_path):
    result = parse(tmp_path, f'''
def Material "Paint" {{
    token outputs:surface.connect = </Paint/Surface.outputs:surface>
    def Shader "Surface" {{
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (0.1,0.3,0.8)
        color3f inputs:emissiveColor = (1,2,3)
        float inputs:metallic = 0.7
        float inputs:roughness = 0.25
        float inputs:opacity = 0.6
        float inputs:opacityThreshold = 0.2
        token outputs:surface
    }}
}}
def Mesh "Mesh" (prepend apiSchemas = ["MaterialBindingAPI"]) {{
    rel material:binding = </Paint>
    {TRIANGLE}
}}
def Xform "Rig" {{
    double3 xformOp:translate = (1,2,3)
    float3 xformOp:rotateXYZ = (90,0,0)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:rotateXYZ"]
    def Camera "Interior" {{
        float focalLength = 20
        float verticalAperture = 20
        float2 clippingRange = (0.02,150)
    }}
}}
def Camera "Placeholder" {{}}
''', metadata='(upAxis = "Z"\n metersPerUnit = 1)', scale=2)
    assert result.up_axis == ke.UpAxis.Z
    assert result.meters_per_unit == 1
    mesh = result.meshes[0]
    assert mesh.material_path == "/Paint"
    np.testing.assert_allclose(mesh.base_color, [0.1, 0.3, 0.8, 0.6])
    np.testing.assert_allclose(mesh.emissive_color, [1, 2, 3])
    assert mesh.metallic == pytest.approx(0.7)
    assert mesh.roughness == pytest.approx(0.25)
    assert mesh.opacity_threshold == pytest.approx(0.2)
    assert mesh.diffuse_texture_path == ""
    assert result.diagnostics.warnings == []
    assert len(result.cameras) == 1
    camera = result.cameras[0]
    np.testing.assert_allclose(xyz(camera.position), [2, 4, 6])
    np.testing.assert_allclose(xyz(camera.target), [2, 6, 6])
    assert camera.vertical_fov == pytest.approx(53.1301)
    assert camera.near_plane == pytest.approx(0.04)
    assert camera.far_plane == pytest.approx(300)


@pytest.mark.parametrize("metallic_connected", [True, False])
def test_packed_texture_channels_and_factors(tmp_path, metallic_connected):
    # Parsing resolves assets without decoding the image on the CPU.
    texture = tmp_path / "packed.png"
    texture.touch()
    metal = 'float inputs:metallic.connect = </Paint/Texture.outputs:b>' if metallic_connected else 'float inputs:metallic = 0'
    result = parse(tmp_path, f'''
def Material "Paint" {{
    token outputs:surface.connect = </Paint/Surface.outputs:surface>
    def Shader "Surface" {{
        uniform token info:id = "UsdPreviewSurface"
        float inputs:roughness.connect = </Paint/Texture.outputs:g>
        {metal}
        color3f inputs:diffuseColor.connect = </Paint/Texture.outputs:rgb>
        token outputs:surface
    }}
    def Shader "Texture" {{
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @packed.png@
        token inputs:sourceColorSpace = "raw"
        float4 inputs:scale = (0.2,0.4,0.6,1)
        float outputs:g
        float outputs:b
        float3 outputs:rgb
    }}
}}
def Mesh "Mesh" (prepend apiSchemas = ["MaterialBindingAPI"]) {{
    rel material:binding = </Paint>
    {TRIANGLE}
}}
''')
    mesh = result.meshes[0]
    assert Path(mesh.metallic_roughness_texture_path) == texture
    assert mesh.roughness == pytest.approx(0.4)
    assert mesh.metallic == pytest.approx(0.6 if metallic_connected else 0)
    np.testing.assert_allclose(mesh.base_color, [0.2, 0.4, 0.6, 1])
    assert not mesh.diffuse_texture_srgb
    assert result.diagnostics.warnings == []


@pytest.mark.parametrize("scalar_channel", ["r", "rgb"])
def test_nodegraph_texture_forwarding(tmp_path, scalar_channel):
    texture = tmp_path / "color.png"
    texture.touch()
    result = parse(tmp_path, f'''
def Material "Paint" {{
    token outputs:surface.connect = </Paint/Surface.outputs:surface>
    def Shader "Surface" {{
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor.connect = </Paint/Maps.outputs:color>
        float inputs:roughness.connect = </Paint/Maps.outputs:roughness>
        float inputs:metallic.connect = </Paint/Maps.outputs:roughness>
        float inputs:opacity.connect = </Paint/Maps.outputs:roughness>
        normal3f inputs:normal.connect = </Paint/Maps/Normal.outputs:rgb>
        token outputs:surface
    }}
    def NodeGraph "Maps" {{
        color3f outputs:color.connect = </Paint/Maps/Nested.outputs:color>
        float outputs:roughness.connect = </Paint/Maps/Nested/Texture.outputs:{scalar_channel}>
        def Shader "Normal" {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @color.png@
            float4 inputs:scale = (2,2,2,1)
            float4 inputs:bias = (-1,-1,-1,0)
            float3 outputs:rgb
        }}
        def NodeGraph "Nested" {{
            color3f outputs:color.connect = </Paint/Maps/Nested/Texture.outputs:rgb>
            def Shader "Texture" {{
                uniform token info:id = "UsdUVTexture"
                asset inputs:file = @color.png@
                float4 inputs:scale = (0.2,0.4,0.6,1)
                color3f outputs:rgb
                float outputs:r
            }}
        }}
    }}
}}
def Mesh "Mesh" (prepend apiSchemas = ["MaterialBindingAPI"]) {{
    rel material:binding = </Paint>
    {TRIANGLE}
}}
''')
    mesh = result.meshes[0]
    assert Path(mesh.diffuse_texture_path) == texture
    assert Path(mesh.roughness_texture_path) == texture
    assert Path(mesh.metallic_texture_path) == texture
    assert Path(mesh.normal_texture_path) == texture
    assert Path(mesh.opacity_texture_path) == texture
    np.testing.assert_allclose(mesh.base_color, [0.2, 0.4, 0.6, 0.2])
    assert mesh.roughness == pytest.approx(0.2)
    assert mesh.metallic == pytest.approx(0.2)
    assert result.diagnostics.warnings == []


def test_missing_texture_diagnostic_once_per_material(tmp_path):
    result = parse(tmp_path, f'''
def Material "Paint" {{
    token outputs:surface.connect = </Paint/Surface.outputs:surface>
    def Shader "Surface" {{
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor.connect = </Paint/Texture.outputs:rgb>
        token outputs:surface
    }}
    def Shader "Texture" {{
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @missing.png@
        float3 outputs:rgb
    }}
}}
def Xform "Parent" (prepend apiSchemas = ["MaterialBindingAPI"]) {{
    rel material:binding = </Paint>
    def Mesh "A" {{ {TRIANGLE} }}
    def Mesh "B" {{ {TRIANGLE} }}
}}
''')
    assert len(result.meshes) == 2
    assert all(mesh.diffuse_texture_path == "" for mesh in result.meshes)
    assert len(result.diagnostics.warnings) == 1
    assert "missing.png" in result.diagnostics.warnings[0]


def test_subset_material_identity(tmp_path):
    result = parse(tmp_path, '''
def Material "Paint" {
    token outputs:surface.connect = </Paint/Surface.outputs:surface>
    def Shader "Surface" {
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor = (0.2,0.4,0.6)
        token outputs:surface
    }
}
def Mesh "Mesh" {
    point3f[] points = [(0,0,0), (1,0,0), (0,1,0), (1,1,0)]
    int[] faceVertexCounts = [3,3]
    int[] faceVertexIndices = [0,1,2, 1,3,2]
    def GeomSubset "Painted" (prepend apiSchemas = ["MaterialBindingAPI"]) {
        uniform token elementType = "face"
        uniform token familyName = "materialBind"
        int[] indices = [0]
        rel material:binding = </Paint>
    }
}
''')
    meshes = {mesh.prim_path: mesh for mesh in result.meshes}
    assert set(meshes) == {"/Mesh/Painted", "/Mesh"}
    assert meshes["/Mesh/Painted"].material_path == "/Paint"
    np.testing.assert_allclose(meshes["/Mesh/Painted"].base_color[:3], [0.2, 0.4, 0.6])
    assert all(mesh.index_count == 3 for mesh in meshes.values())


def test_light_authored_values_transform_visibility_and_camera_exposure(tmp_path):
    result = parse(tmp_path, '''
def Xform "Rig" {
    double3 xformOp:translate = (1,2,3)
    float3 xformOp:scale = (2,3,4)
    uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    def RectLight "Panel" {
        float inputs:width = 2
        float inputs:height = 1
        color3f inputs:color = (0.5,0.8,1)
        float inputs:intensity = 3000
        float inputs:exposure = 2
        bool inputs:normalize = true
        bool inputs:enableColorTemperature = true
        float inputs:colorTemperature = 4200
        float inputs:diffuse = 0.75
        float inputs:specular = 0.25
    }
}
def Xform "Hidden" {
    token visibility = "invisible"
    def DomeLight "Sky" {
        float inputs:intensity = 700
    }
}
def Camera "Camera" {
    float focalLength = 20
    float exposure = 1
    float exposure:time = 0.02
    float exposure:iso = 100
    float exposure:fStop = 5
    float exposure:responsivity = 1.1
}
''', scale=2)
    assert len(result.lights) == 2
    panel, sky = result.lights
    assert panel.type == ke.scene.LightType.RECT
    assert sky.type == ke.scene.LightType.DOME
    assert panel.visible and not sky.visible
    np.testing.assert_allclose(np.asarray(panel.world_transform), [
        [4, 0, 0, 2], [0, 6, 0, 4], [0, 0, 8, 6], [0, 0, 0, 1],
    ])
    assert panel.rect.width == 2 and panel.rect.height == 1
    assert panel.rect.intensity == 3000 and panel.rect.exposure == 2
    assert panel.rect.normalize and panel.rect.enable_color_temperature
    assert panel.rect.color_temperature == 4200
    assert panel.rect.diffuse == 0.75 and panel.rect.specular == 0.25
    assert sky.dome.intensity == 700
    assert result.cameras[0].linear_exposure_scale == pytest.approx(1.1 * .02 * 2 / 25)
    assert result.diagnostics.warnings == []


def test_unsupported_light_is_diagnosed(tmp_path):
    result = parse(tmp_path, 'def SphereLight "Bulb" {}')
    assert result.lights == []
    assert any("Unsupported USD light type: /Bulb" in message for message in result.diagnostics.warnings)


def test_light_texture_metadata_is_retained_with_diagnostics(tmp_path):
    result = parse(tmp_path, '''
def RectLight "Panel" {
    asset inputs:texture:file = @panel.png@
}
def DomeLight "Sky" {
    asset inputs:texture:file = @environment.hdr@
    token inputs:texture:format = "latlong"
}
''')
    assert result.lights[0].rect.texture_path == str(tmp_path / "panel.png")
    assert result.lights[1].dome.texture_path == str(tmp_path / "environment.hdr")
    assert result.lights[1].dome.texture_format == "latlong"
    assert len(result.diagnostics.warnings) == 2
    assert all("retained but not rendered" in message for message in result.diagnostics.warnings)


def test_rect_component_preserves_local_authoring_and_rejects_invalid_values():
    scene = ke.scene.create_backend(ke.scene.BackendType.NATIVE)
    parent = scene.define_prim("/lights/rig", ke.scene.PrimType.XFORM)
    parent.set_local_translation(ke.Vec3(1, 2, 3))
    parent.set_local_scale(ke.Vec3(2, 3, 4))
    prim = scene.define_prim("/lights/rig/panel", ke.scene.PrimType.LIGHT)
    prim.set_local_translation(ke.Vec3(4, 5, 6))
    light = ke.scene.RectLight()
    light.width, light.height = 2, 3
    light.intensity, light.exposure = 3000, -1
    light.normalize = True
    prim.set_rect_light(light)
    authored = prim.get_rect_light()
    assert authored.width == 2 and authored.height == 3
    assert authored.intensity == 3000 and authored.exposure == -1
    assert authored.normalize
    assert prim.get_light_type() == ke.scene.LightType.RECT
    np.testing.assert_allclose(xyz(prim.get_local_translation()), [4, 5, 6])
    light.width = 0
    with pytest.raises(ValueError, match="dimensions"):
        prim.set_rect_light(light)
    assert prim.get_rect_light().width == 2
    light.width, light.intensity = 1, float("nan")
    with pytest.raises(ValueError, match="emission"):
        prim.set_rect_light(light)
