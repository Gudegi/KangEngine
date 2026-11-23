"""Check shared PBR edits on both objects through App or SceneRenderer."""

import argparse
from pathlib import Path
import tempfile

import numpy as np
from PIL import Image

import kangengine as ke
from kangengine._core import _ke


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "ovrtx"), default="native")
    parser.add_argument("--scene-renderer", action="store_true")
    args = parser.parse_args()
    app = None
    renderer = None
    with tempfile.TemporaryDirectory() as directory:
        directory = Path(directory)
        for name, color in (("red", (230, 20, 20)), ("blue", (20, 20, 230))):
            Image.new("RGB", (8, 8), color).save(directory / f"{name}.png")
        try:
            if args.scene_renderer:
                scene = ke.scene.create_backend(ke.scene.BackendType.NATIVE)
                resources = ke.scene.SceneResourceManager(scene)
                material = ke.material.PBRMaterial()
                material.roughness = 0.8
                resources.register_material("Shared", material.native)
            else:
                app = ke.App()
                app.initialize(width=256, height=192, headless=True, hide_ui=True,
                               render_backend=args.backend)
                app.set_frame_capture_active(True)
                scene, resources = app.scene.native, app.resources
                material = app.create_pbr_material(roughness=0.8)
            camera = scene.define_prim("/camera", ke.scene.PrimType.CAMERA)
            camera.set_local_translation(ke.Vec3(0, 0, 5))
            camera.add_camera_component().set_perspective(45, 0.1, 100)
            light = ke.scene.DirectionalLight()
            light.direction = ke.Vec3(0, 0, -1)
            light.intensity = 3
            light.ambient = ke.Vec3(0, 0, 0)
            if app is not None:
                app.set_active_scene_camera(camera)
                scene.get_prim_at_path("/lights/default_directional").set_directional_light(light)
            else:
                scene.define_prim("/light", ke.scene.PrimType.LIGHT).set_directional_light(light)
            mesh = ke.geometry.create_sphere_data(radius=0.7, num_longitudes=32, num_latitudes=16)
            if app is not None:
                surfaces = [app.scene.add_mesh(path, mesh, material) for path in ("/left", "/right")]
                prims = [surface.prim for surface in surfaces]
                textures = [app.load_texture(directory / f"{name}.png", srgb=True)
                            for name in ("red", "blue")]
            else:
                mesh_id = resources.register_mesh("Sphere", mesh)
                prims = [scene.define_prim(path, ke.scene.PrimType.MESH) for path in ("/left", "/right")]
                for prim in prims:
                    prim.set_mesh_data(mesh)
                    prim.get_mesh_component().resource_handle = mesh_id
                    prim.set_material(material.native)
                    prim.add_render_component()
                textures = [resources.register_texture_source(
                    name, str(directory / f"{name}.png"), srgb=True,
                ) for name in ("red", "blue")]
            for prim, x in zip(prims, (-1, 1)):
                prim.set_local_translation(ke.Vec3(x, 0, 0))
            collector = _ke._RenderSceneDataCollector()
            first = collector.collect(scene, resources, "/camera")
            ids = {item["id"] for item in first["objects"]}
            assert len(ids) == 2
            collector.accept()
            if args.scene_renderer:
                renderer = ke.render.SceneRenderer(
                    scene, resources, camera_path="/camera",
                    backend=args.backend, width=256, height=192,
                )

            def capture():
                if renderer is not None:
                    return renderer.render()
                app.draw_frame()
                return app.read_rgb_pixels().copy()

            def check_delivery():
                frame = collector.collect(scene, resources, "/camera")
                assert {item["id"] for item in frame["objects"]} == ids
                collector.accept()
                assert not collector.collect(scene, resources, "/camera")["objects"]

            def set_texture(texture):
                if app is not None:
                    material.base_color_texture = texture
                else:
                    material.texture_resources = (
                        {} if texture is None else {"base_color_texture": texture}
                    )

            def means(image):
                return np.array([image[72:120, x:x + 32].mean(axis=(0, 1)) for x in (64, 160)])

            capture()
            for color, dominant in ((ke.Vec4(1, 0.03, 0.03, 1), 0),
                                    (ke.Vec4(0.03, 0.03, 1, 1), 2)):
                material.base_color = color
                check_delivery()
                rgb = means(capture())
                assert np.all(rgb[:, dominant] > rgb[:, 2 - dominant] + 20), rgb

            material.base_color = ke.Vec4(1, 1, 1, 1)
            for texture, dominant in zip(textures, (0, 2)):
                set_texture(texture)
                check_delivery()
                rgb = means(capture())
                assert np.all(rgb[:, dominant] > rgb[:, 2 - dominant] + 20), rgb
            set_texture(None)
            check_delivery()
            rgb = means(capture())
            assert np.all(np.abs(rgb[:, 0] - rgb[:, 2]) < 10), rgb

            material.metallic = 0.8
            material.roughness = 0.1
            check_delivery()
            glossy = capture().astype(float)
            material.roughness = 0.9
            check_delivery()
            rough = capture().astype(float)
            for x in (48, 144):
                assert np.abs(glossy[56:136, x:x + 64] - rough[56:136, x:x + 64]).mean() > 1

            material.emissive_color = ke.Vec3(0, 0.5, 0)
            material.emissive_strength = 3
            check_delivery()
            emissive = means(capture())
            assert np.all(emissive[:, 1] > means(rough)[:, 1] + 10), emissive
            if args.backend == "native" and app is not None:
                assert app.get_scene_render_system().shares_batch(
                    surfaces[0].component, surfaces[1].component)
            route = "SceneRenderer" if renderer is not None else "App"
            print(f"PASS: {args.backend} {route} shared color, textures, "
                  "roughness, emission, and idle delivery")
        finally:
            if renderer is not None:
                renderer.close()
            if app is not None:
                app.close()


if __name__ == "__main__":
    main()
