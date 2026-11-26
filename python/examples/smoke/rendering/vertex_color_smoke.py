"""Check mesh RGB, white fallback, unlit lighting independence and lit families."""

from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

import kangengine as ke


class ColorSmokeApp(ke.App):
    pending_creation = None

    def pre_render(self):
        if self.pending_creation is not None:
            create = self.pending_creation
            self.pending_creation = None
            create()


def main():
    assert ke.material.VertexColorMaterial is ke.material.ColorMaterial
    assert ke.material.VertexColorStyle is ke.material.ColorStyle
    assert ke.material.NativeVertexColorMaterial is ke.material.NativeColorMaterial
    assert ke.App.create_vertex_color_material is ke.App.create_color_material
    vertices = np.array([[-1, -1, 0], [1, -1, 0], [0, 1, 0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int32)
    normals = np.tile([0, 0, 1], (3, 1))
    colors = np.eye(3, dtype=np.float32)
    for attribute in ("colors", "normals"):
        try:
            ke.scene.MeshData.from_arrays(vertices, faces, **{attribute: np.ones((2, 3))})
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted mismatched {attribute}")

    app = ColorSmokeApp()
    app.initialize(width=192, height=192, hide_ui=True, headless=True, up_axis=ke.UpAxis.Y)
    try:
        app.set_camera_view([0, 0, 3], [0, 0, 0])
        material = app.create_color_material(style=ke.material.ColorStyle.UNLIT)
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "frame.png"

            def capture():
                app.draw_frame()
                assert app.write_pixels_png(path=str(image_path), flip_y=True)
                return np.array(Image.open(image_path).convert("RGB"), dtype=float)

            def add(colors=None, chosen_material=material, *, skinned=False):
                mesh = ke.scene.MeshData.from_arrays(vertices, faces, normals=normals, colors=colors)
                if skinned:
                    skin = ke.scene.SkinnedMeshData(
                        mesh=mesh,
                        bone_indices=np.zeros((3, 4), dtype=np.int32),
                        bone_weights=np.tile([1, 0, 0, 0], (3, 1)),
                        bone_node_indices=[0],
                        inverse_bind_matrices=np.eye(4, dtype=np.float32)[None],
                    )
                    prim = app.scene.define_prim("/triangle", ke.scene.PrimType.MESH)
                    prim.set_mesh_data(mesh)
                    prim.set_display_color_alpha(ke.Vec4(1, 1, 1, 1))
                    surface = app.add_skinned_mesh(prim, chosen_material, skin)
                    surface.update_skinning(np.eye(4, dtype=np.float32)[None])
                else:
                    surface = app.scene.add_mesh("/triangle", mesh, chosen_material, color=ke.Vec4(1, 1, 1, 1))
                surface.set_double_sided(True)
                surface.set_casts_shadow(False)
                assert surface.prim.get_mesh_component().resource_handle != 0
                return surface

            surface = add(colors)
            colored = capture()
            channel_spread = colored.max(axis=2) - colored.min(axis=2)
            assert np.count_nonzero(channel_spread > 40) > 500, "vertex RGB did not reach the image"
            light = app.scene.get_prim_at_path("/lights/default_directional")
            light.set_visible(False)
            np.testing.assert_allclose(capture(), colored, atol=1)
            light.set_visible(True)
            surface.set_base_color(ke.Vec4(.25, .25, .25, 1))
            tinted = capture()
            assert np.mean(colored - tinted) > 3, "instance tint was lost"
            surface.set_visible(False)
            assert np.mean(np.abs(capture() - tinted)) > 3
            app.scene.remove_prim("/triangle")

            add()
            fallback = capture()
            app.scene.remove_prim("/triangle")
            add(np.ones((3, 3)))
            np.testing.assert_allclose(capture(), fallback, atol=1)
            app.scene.remove_prim("/triangle")

            for lit_material in (app.create_color_material(), app.create_phong_material(),
                                 app.create_pbr_material()):
                add(colors, lit_material)
                lit = capture()
                assert np.count_nonzero(lit.max(axis=2) - lit.min(axis=2) > 25) > 200
                app.scene.remove_prim("/triangle")

            # Color/no-color layouts must work across skinning and blend variants.
            for chosen in (material, app.create_color_material(),
                           app.create_phong_material(), app.create_pbr_material()):
                for skinned in (False, True):
                    for alpha in (1.0, 0.5):
                        frames = []
                        for rgb in (None, np.ones((3, 3)), colors):
                            surface = add(rgb, chosen, skinned=skinned)
                            surface.set_double_sided(False)
                            surface.set_base_color(ke.Vec4(1, 1, 1, alpha))
                            frames.append(capture())
                            app.scene.remove_prim("/triangle")
                        np.testing.assert_allclose(frames[0], frames[1], atol=1)
                        assert np.mean(np.abs(frames[2] - frames[1])) > 1

            receiver_mesh = ke.scene.MeshData.from_arrays(
                np.array([[-4, -4, -1], [4, -4, -1], [4, 4, -1], [-4, 4, -1]]),
                np.array([[0, 1, 2], [0, 2, 3]]),
                normals=np.tile([0, 0, 1], (4, 1)),
            )
            app.scene.add_mesh("/receiver", receiver_mesh, app.create_phong_material())
            sun = ke.scene.DirectionalLight()
            sun.direction = ke.Vec3(0.8, 0, -1)
            light.set_directional_light(sun)

            # Cutout affects the surface and shadow; selection stays identifiable.
            for skinned in (False, True):
                for rgb in (None, colors):
                    surface = add(rgb, skinned=skinned)
                    surface.set_casts_shadow(True)
                    surface.set_visible(False)
                    hidden = capture()
                    surface.set_visible(True)
                    surface.set_base_color(ke.Vec4(1, 1, 1, 0.25))
                    surface.set_alpha_mode(ke.render.AlphaMode.MASK, cutoff=0.5)
                    np.testing.assert_allclose(capture(), hidden, atol=1)
                    app.select_prim(prim=surface.prim)
                    selected_hidden = capture()
                    assert np.count_nonzero(np.max(np.abs(selected_hidden - hidden), axis=2) > 10) > 20
                    surface.set_base_color(ke.Vec4(1, 1, 1, 0))
                    np.testing.assert_allclose(capture(), selected_hidden, atol=1)

                    # A foreground object must not hide the selection outline.
                    occluder = app.scene.add_mesh(
                        "/occluder", receiver_mesh, material,
                    )
                    occluder.set_local_translation(ke.Vec3(0, 0, 2))
                    app.clear_selection()
                    occluded = capture()
                    app.select_prim(prim=surface.prim)
                    assert np.count_nonzero(np.max(np.abs(capture() - occluded), axis=2) > 10) > 20
                    app.scene.remove_prim("/occluder")
                    assert app.get_selection().prim == surface.prim
                    np.testing.assert_allclose(capture(), selected_hidden, atol=1)
                    app.clear_selection()
                    surface.set_base_color(ke.Vec4(1, 1, 1, 0.25))
                    np.testing.assert_allclose(capture(), hidden, atol=1)
                    app.select_prim(prim=surface.prim)
                    surface.set_alpha_mode(ke.render.AlphaMode.MASK, cutoff=0.1)
                    outlined = capture()
                    app.clear_selection()
                    visible = capture()
                    assert np.count_nonzero(np.max(np.abs(outlined - visible), axis=2) > 10) > 20, (
                        "missing outline", skinned, rgb is not None
                    )
                    surface.set_casts_shadow(False)
                    unshadowed = capture()
                    assert np.count_nonzero(np.mean(unshadowed - visible, axis=2) > 10) > 20
                    surface.set_casts_shadow(True)
                    app.select_prim(prim=surface.prim)
                    surface.set_alpha_mode(ke.render.AlphaMode.MASK, cutoff=0.5)
                    np.testing.assert_allclose(capture(), selected_hidden, atol=1)
                    app.clear_selection()
                    np.testing.assert_allclose(capture(), hidden, atol=1)
                    for mode in (ke.render.AlphaMode.BLEND, ke.render.AlphaMode.OPAQUE):
                        surface.set_alpha_mode(mode, cutoff=0.5)
                        assert np.mean(np.abs(capture() - hidden)) > 1
                    app.scene.remove_prim("/triangle")
            app.scene.remove_prim("/receiver")

            # Measure allocations inside a profiled frame, after warming both variants.
            renderer = app.get_renderer()
            renderer.set_profiler_enabled(enabled=True)
            allocations = []
            for rgb in (None, np.ones((3, 3))):
                app.pending_creation = lambda: add(rgb)
                capture()
                counters = renderer.latest_frame_profile().counters
                allocations.append((counters.buffer_allocations, counters.buffer_allocated_bytes))
                app.scene.remove_prim("/triangle")
            assert allocations[1][0] - allocations[0][0] == 1, allocations
            assert allocations[1][1] - allocations[0][1] == colors.nbytes, allocations
            renderer.set_profiler_enabled(enabled=False)
        print("PASS: RGB interpolation, unlit lighting independence, tint, visibility, white fallback, lit families")
        print("PASS: color material aliases, skinned/blend variants, no fallback color allocation")
        print("PASS: unlit cutout/shadows, transparent selection outline and alpha-mode transitions")
    finally:
        app.close()


if __name__ == "__main__":
    main()
