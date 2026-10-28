"""Render USD articulation textures and verify App-owned texture sharing."""

import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

import kangengine as ke
from kangengine.sim import GridCloner, KangSimWorld
from kangengine.visual.sim import SimWorldVisualizer


def _asset(root, *, mdl, raw):
    pixels = np.zeros((32, 32, 3), dtype=np.uint8)
    pixels[:, :16] = (220, 20, 20)
    pixels[:, 16:] = (20, 220, 20)
    Image.fromarray(pixels).save(root / "paint.png")
    color_space = "raw" if raw else "sRGB"
    surface = "mdl:surface" if mdl else "surface"
    shader = f'''
        uniform token info:implementationSource = "sourceAsset"
        uniform asset info:mdl:sourceAsset = @OmniPBR.mdl@
        uniform token info:mdl:sourceAsset:subIdentifier = "OmniPBR"
        asset inputs:diffuse_texture = @paint.png@ (colorSpace = "{color_space}")
    ''' if mdl else '''
        uniform token info:id = "UsdPreviewSurface"
        color3f inputs:diffuseColor.connect = </Paint/Texture.outputs:rgb>
    '''
    path = root / "robot.usda"
    path.write_text(f'''#usda 1.0
(
    defaultPrim = "Robot"
    metersPerUnit = 1
    upAxis = "Z"
)
def Material "Paint" {{
    token outputs:{surface}.connect = </Paint/Shader.outputs:out>
    def Shader "Shader" {{
        {shader}
        token outputs:out
    }}
    def Shader "Texture" {{
        uniform token info:id = "UsdUVTexture"
        asset inputs:file = @paint.png@
        token inputs:sourceColorSpace = "{color_space}"
        float3 outputs:rgb
    }}
}}
def Xform "Robot" {{
    def Xform "body" (prepend apiSchemas = ["PhysicsRigidBodyAPI", "PhysicsMassAPI"]) {{
        float physics:mass = 1
        float3 physics:diagonalInertia = (1,1,1)
        def Mesh "panel" (prepend apiSchemas = ["MaterialBindingAPI"]) {{
            rel material:binding = </Paint>
            point3f[] points = [(-1,0,0), (1,0,0), (1,0,2), (-1,0,2)]
            int[] faceVertexCounts = [4]
            int[] faceVertexIndices = [0,1,2,3]
            texCoord2f[] primvars:st = [(0,0), (1,0), (1,1), (0,1)] (
                interpolation = "vertex"
            )
        }}
    }}
}}
''')
    return ke.asset.USDLoader.parse_articulation(usd_path=str(path)).articulation


def main():
    app = ke.App()
    app.initialize(width=640, height=320, hide_ui=True, up_axis=ke.UpAxis.Z, headless=True)
    try:
        app.set_camera_view(position=(0.0, -8.0, 2.5), target=(0.0, 0.0, 1.0))
        app.set_light_direction(ke.Vec3(0.0, 1.0, -0.3))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for mdl in (False, True):
                frames = []
                for raw in (False, True):
                    data = _asset(root, mdl=mdl, raw=raw)
                    source = data.visual_geoms[0].material
                    world = KangSimWorld(num_envs=1, sim_device="cpu", state_device="cpu")
                    viewer = SimWorldVisualizer(app, world)
                    try:
                        cloner = GridCloner(world)
                        for obj_id, x in enumerate((-1.2, 1.2)):
                            batch = cloner.add_articulation(
                                data, obj_id=obj_id, initial_root_pos=(x, 0.0, 0.0)
                            )
                            viewer.add(batch, path=f"/robot_{obj_id}")

                        materials = [
                            app.scene.get_prim_at_path(f"/robot_{i}/body/visual_0").get_material()
                            for i in range(2)
                        ]
                        texture = app.load_texture(root / "paint.png", srgb=not raw)
                        assert all(mat.base_color_texture is texture for mat in materials)
                        assert source.base_color_texture is None
                        assert materials[0] is not materials[1]

                        app.draw_frame()
                        frame = np.asarray(app.read_rgb_pixels(True)).copy()
                        red, green = frame[..., 0].astype(float), frame[..., 1].astype(float)
                        assert np.count_nonzero((red > 1.5 * green) & (red > 40)) > 200
                        assert np.count_nonzero((green > 1.5 * red) & (green > 40)) > 200
                        frames.append(frame)
                        output = Path(tempfile.gettempdir()) / f"usd-articulation-{'mdl' if mdl else 'preview'}-{'raw' if raw else 'srgb'}.png"
                        Image.fromarray(frame).save(output)
                        print(f"PASS: {output}")
                    finally:
                        viewer.release()
                        world.release()

                assert np.count_nonzero(np.any(frames[0] != frames[1], axis=-1)) > 200
    finally:
        app.close()


if __name__ == "__main__":
    main()
