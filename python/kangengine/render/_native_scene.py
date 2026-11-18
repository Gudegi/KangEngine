"""Native renderer resources for evaluated static scene inputs."""

from ..app import App
from .. import scene


class NativeSceneAdapter:
    def __init__(self, width, height):
        self._app = App()
        self._objects = {}
        self._materials = {}
        self._textures = {}
        try:
            self._app.initialize(width=width, height=height, hide_ui=True, headless=True)
            self._app.set_frame_capture_active(True)
            self._camera = self._app.scene.define_prim("/camera", scene.PrimType.CAMERA)
            self._camera.add_camera_component()
            self._app.set_active_scene_camera(self._camera)
        except BaseException:
            self._app.close()
            raise

    def _texture(self, source):
        sampler = source["sampler"]
        key = (source["id"], source["path"], source["srgb"], source["flip"],
               sampler.wrap_u, sampler.wrap_v, sampler.min_filter, sampler.mag_filter)
        if key not in self._textures:
            texture = self._app.get_renderer().device().create_texture(
                source["path"], source["flip"], sampler, srgb=source["srgb"])
            self._app._remember_textures(texture)
            self._textures[key] = texture
        return self._textures[key]

    def render(self, frame):
        ids = {item["id"] for item in frame["objects"]}
        for instance_id in self._objects.keys() - ids:
            view, _, _ = self._objects.pop(instance_id)
            view.remove()
        for item in frame["objects"]:
            instance_id = item["id"]
            previous = self._objects.get(instance_id)
            if previous is not None and previous[2] != item["mesh_id"]:
                previous[0].remove()
                previous = None
            if previous is None:
                material = self._material(item["material_id"])
                view = self._app.scene.add_mesh(
                    f"/instances/i{instance_id}", item["mesh"], material)
                self._objects[instance_id] = view, material, item["mesh_id"]
            else:
                view, material, _ = previous
                replacement = self._material(item["material_id"])
                if replacement is not material:
                    view.set_material(replacement)
                    material = replacement
                    self._objects[instance_id] = view, material, item["mesh_id"]
            value = item["material"]
            material.base_color = value["base_color"]
            material.metallic = value["metallic"]
            material.roughness = value["roughness"]
            material.emissive_color = value["emissive"]
            material.emissive_strength = 1.0
            texture = item["textures"].get("base_color_texture")
            material.base_color_texture = self._texture(texture) if texture else None
            # Prim's Python matrix setter consumes contiguous GLM column storage.
            view.set_world_matrix(item["world"].T.copy())
            view.set_base_color(item["color"])
            view.set_visible(item["visible"])
            view.set_double_sided(item["double_sided"])

        camera = frame["camera"]
        self._camera.set_world_matrix(camera["world"].T.copy())
        self._camera.get_camera_component().set_perspective(
            camera["vertical_fov"], camera["near"], camera["far"])
        light = frame["lights"][0] if frame["lights"] else None
        self._app.set_light_intensity(light["intensity"] if light else 0)
        self._app.set_light_ambient([0, 0, 0])
        if light:
            self._app.set_light_direction(light["direction"])
            self._app.set_light_color(light["color"])
        self._app.draw_frame()
        return self._app.read_rgb_pixels()

    def _material(self, resource_id):
        if resource_id not in self._materials:
            self._materials[resource_id] = self._app.create_pbr_material()
        return self._materials[resource_id]

    def close(self):
        self._objects.clear()
        self._textures.clear()
        self._materials.clear()
        self._app.close()
