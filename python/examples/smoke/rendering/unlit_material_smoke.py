"""Check shared unlit shading, live material edits and per-batch alpha settings."""

from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

import kangengine as ke


def main():
    legacy = ke.material.ColorMaterial(style=ke.material.ColorStyle.UNLIT)
    assert legacy.unlit and legacy.native.unlit
    legacy.unlit = False
    assert legacy.style == ke.material.ColorStyle.UNTEXTURED
    legacy.set_style(ke.material.ColorStyle.UNLIT)
    assert legacy.unlit
    legacy.set_style(ke.material.ColorStyle.TEXTURED)
    assert not legacy.unlit

    app = ke.App()
    app.initialize(width=192, height=192, hide_ui=True, headless=True, up_axis=ke.UpAxis.Y)
    try:
        app.set_camera_view([0, 0, 3], [0, 0, 0])
        light = app.scene.get_prim_at_path("/lights/default_directional")
        vertices = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32)
        faces = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
        normals = np.tile([0, 0, 1], (4, 1))

        def mesh_data(colors=None):
            mesh = ke.scene.MeshData.from_arrays(vertices, faces, normals=normals, colors=colors)
            mesh.uvs = [ke.Vec2(0, 0), ke.Vec2(1, 0), ke.Vec2(1, 1), ke.Vec2(0, 1)]
            return mesh

        def add(material, *, colors=None, skinned=False):
            mesh = mesh_data(colors)
            if skinned:
                skin = ke.scene.SkinnedMeshData(
                    mesh=mesh, bone_indices=np.zeros((4, 4), dtype=np.int32),
                    bone_weights=np.tile([1, 0, 0, 0], (4, 1)), bone_node_indices=[0],
                    inverse_bind_matrices=np.eye(4, dtype=np.float32)[None],
                )
                prim = app.scene.define_prim("/surface", ke.scene.PrimType.MESH)
                prim.set_mesh_data(mesh)
                prim.set_display_color_alpha(ke.Vec4(1, 1, 1, 1))
                surface = app.add_skinned_mesh(prim, material, skin)
                surface.update_skinning(np.eye(4, dtype=np.float32)[None])
            else:
                surface = app.scene.add_mesh("/surface", mesh, material, color=ke.Vec4(1, 1, 1, 1))
            surface.set_casts_shadow(False)
            return surface

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pixels = np.array([
                [[220, 70, 30, 255], [40, 180, 90, 255]],
                [[50, 70, 220, 255], [210, 170, 30, 255]],
            ], dtype=np.uint8)
            textures = []
            for index, data in enumerate((pixels, pixels[:, ::-1].copy())):
                path = root / f"color_{index}.png"
                Image.fromarray(data).save(path)
                textures.append(app.load_texture(path=path, srgb=True))

            def capture():
                app.draw_frame()
                path = root / "frame.png"
                assert app.write_pixels_png(path=str(path), flip_y=True)
                return np.asarray(Image.open(path).convert("RGB"), dtype=float)

            materials = [app.create_color_material(style=ke.material.ColorStyle.TEXTURED),
                         app.create_phong_material(), app.create_pbr_material()]
            for material in materials:
                assert not material.unlit
                assert material.set_unlit(enabled=True) is material

            def set_texture(index, surface, texture):
                if index == 0:
                    surface.set_texture(texture)
                elif index == 1:
                    materials[index].set_diffuse_map(texture)
                else:
                    materials[index].set_base_color_texture(texture)

            for skinned in (False, True):
                for colors in (None, np.array([[1, .2, .2], [.2, 1, .2], [.2, .2, 1], [1, 1, 1]])):
                    reference = []
                    for index, material in enumerate(materials):
                        surface = add(material, colors=colors, skinned=skinned)
                        frames = []
                        for texture in (textures[0], textures[1], None):
                            set_texture(index, surface, texture)
                            frames.append(capture())
                        assert np.mean(np.abs(frames[0] - frames[1])) > 1
                        if index == 0:
                            reference = frames
                        else:
                            np.testing.assert_allclose(frames, reference, atol=1)
                        set_texture(index, surface, textures[0])
                        light.set_visible(False)
                        np.testing.assert_allclose(capture(), frames[0], atol=1)
                        light.set_visible(True)
                        material.unlit = False
                        assert np.mean(np.abs(capture() - frames[0])) > 1
                        material.unlit = True
                        np.testing.assert_allclose(capture(), frames[0], atol=1)

                        # Live material factors must match the same instance tint.
                        tint = ke.Vec3(.3, .6, .8)
                        surface.set_base_color(ke.Vec4(tint.x, tint.y, tint.z, 1))
                        tinted = capture()
                        if index > 0:
                            surface.set_base_color(ke.Vec4(1, 1, 1, 1))
                            if index == 1:
                                material.diffuse = tint
                            else:
                                material.base_color = ke.Vec4(tint.x, tint.y, tint.z, 1)
                            np.testing.assert_allclose(capture(), tinted, atol=1)
                            if index == 1:
                                material.diffuse = ke.Vec3(1, 1, 1)
                            else:
                                material.base_color = ke.Vec4(1, 1, 1, 1)
                        app.scene.remove_prim("/surface")

            # Separate red opacity overrides a zero-alpha base texture.
            opacity = []
            for name, rgba in (("base", [200, 100, 50, 0]),
                               ("opaque", [255, 0, 0, 0]), ("clear", [0, 255, 255, 255])):
                path = root / f"{name}.png"
                Image.fromarray(np.full((2, 2, 4), rgba, dtype=np.uint8)).save(path)
                opacity.append(app.load_texture(path=path))
            for index in (1, 2):
                material = materials[index]
                for skinned in (False, True):
                    surface = add(material, skinned=skinned)
                    surface.set_visible(False)
                    hidden = capture()
                    surface.set_visible(True)
                    set_texture(index, surface, opacity[0])
                    set_opacity = material.set_alpha_map if index == 1 else material.set_opacity_texture
                    for mode in (ke.render.AlphaMode.MASK, ke.render.AlphaMode.BLEND):
                        surface.set_alpha_mode(mode, cutoff=0.5)
                        set_opacity(opacity[1])
                        assert np.mean(np.abs(capture() - hidden)) > 1
                        for texture in (opacity[2], None):
                            set_opacity(texture)
                            np.testing.assert_allclose(capture(), hidden, atol=1)
                    app.scene.remove_prim("/surface")
                set_texture(index, None, None)

            # Shared materials must not share alpha parameters across batches.
            for material in materials[1:]:
                left = app.scene.add_mesh("/left", mesh_data(), material, color=ke.Vec4(1, 1, 1, .5))
                right = app.scene.add_mesh("/right", mesh_data(), material, color=ke.Vec4(1, 1, 1, .5))
                left.set_local_translation(ke.Vec3(-.7, 0, 0))
                right.set_local_translation(ke.Vec3(.7, 0, 0))
                for surface in (left, right):
                    surface.set_casts_shadow(False)
                    surface.set_local_scale(ke.Vec3(.5, .5, .5))
                left.set_alpha_mode(ke.render.AlphaMode.MASK, cutoff=.2)
                right.set_alpha_mode(ke.render.AlphaMode.MASK, cutoff=.8)
                masked = capture()
                right.set_visible(False)
                np.testing.assert_allclose(capture(), masked, atol=1)
                right.set_visible(True)
                right.set_alpha_mode(ke.render.AlphaMode.MASK, cutoff=.2)
                assert np.mean(np.abs(capture() - masked)) > 1
                app.scene.remove_prim("/left")
                app.scene.remove_prim("/right")
        print("PASS: common unlit colors/textures, skinning, live edits, opacity and per-batch alpha")
    finally:
        app.close()


if __name__ == "__main__":
    main()
