"""Render the same authored scene with Native or optional ovrtx."""

from pathlib import Path

from PIL import Image
import kangengine as ke
from kangengine.utils._example_args import render_example_parser


def build_scene(image_path):
    scene = ke.scene.create_backend(ke.scene.BackendType.NATIVE)
    resources = ke.scene.SceneResourceManager(scene)
    material = ke.material.PBRMaterial()
    material.roughness = 0.8
    material.texture_resources = {"base_color_texture": resources.register_texture_source(
        "Color", str(image_path), srgb=True)}
    resources.register_material("Surface", material.native)

    parent = scene.define_prim("/group", ke.scene.PrimType.XFORM)
    parent.set_local_translation(ke.Vec3(0.25, 0, 0))
    mesh = ke.geometry.create_cube_data(1.5)
    mesh_id = resources.register_mesh("Cube", mesh)

    def add_cube(path="/group/cube"):
        prim = scene.define_prim(path, ke.scene.PrimType.MESH)
        prim.set_mesh_data(mesh)
        prim.get_mesh_component().resource_handle = mesh_id
        prim.set_material(material.native)
        prim.add_render_component()
        return prim

    add_cube()
    camera = scene.define_prim("/camera", ke.scene.PrimType.CAMERA)
    camera.set_local_translation(ke.Vec3(0, 0, 5))
    camera.add_camera_component().set_perspective(45, 0.1, 100)
    light = ke.scene.DirectionalLight()
    light.direction = ke.Vec3(0, 0, -1)
    light.intensity = 3
    light.ambient = ke.Vec3(0, 0, 0)
    scene.define_prim("/light", ke.scene.PrimType.LIGHT).set_directional_light(light)
    return scene, resources, material, add_cube


def main():
    parser = render_example_parser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("tmp/scene-renderer.png"))
    args = parser.parse_args()
    image_path = Path(ke.__file__).parent / "assets/textures/awesomeface.png"
    scene, resources, material, _ = build_scene(image_path)
    with ke.render.SceneRenderer(scene, resources, camera_path="/camera", renderer=args.renderer) as renderer:
        pixels = renderer.render()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
