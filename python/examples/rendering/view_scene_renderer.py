"""Inspect materials, lights, deformation, skinning, instances, and picking."""

import argparse
import math
from pathlib import Path

import numpy as np
import kangengine as ke


def rgba(preset):
    color = ke.ColorLibrary.get(preset)
    return (color.r, color.g, color.b, color.a)


class SceneRendererApp(ke.App):
    hdri_path = None

    def setup(self):
        self.animate = True
        self.angle = 0.0
        self.controls_docked = False
        self.samples_dirty = True
        self.cuda_enabled = False
        self.cuda_instances = None
        self.cuda_status = "CUDA instances are off."
        self.last_pick = "Click an object to inspect its surface hit."
        self.sample_groups = {}
        self.set_fixed_update_hz(update_hz=60)
        self.set_ui_layout_mode(mode=ke.UILayoutMode.EDITOR)
        self.set_camera_view(position=[18, 16, 24], target=[0, 1, 0])
        light = ke.scene.DirectionalLight()
        light.direction = ke.Vec3(-0.5, -1, -0.4)
        light.intensity = 3
        light.ambient = ke.Vec3(0, 0, 0)
        self.scene.get_prim_at_path("/lights/default_directional").set_directional_light(light)
        self.environment_enabled = True
        self.environment: ke.scene.Prim = self.scene.define_prim(
            path="/lights/environment", prim_type=ke.scene.PrimType.LIGHT,
        )
        self.environment_light: ke.scene.DomeLight = ke.scene.DomeLight()
        self.environment_light.intensity = 0.3
        if self.hdri_path is not None:
            self.environment_light.texture_path = str(self.hdri_path)
            self.environment_light.texture_format = "latlong"
        self.environment.set_dome_light(light=self.environment_light)

        # Shared geometry and materials keep repeated objects comparable.
        cube: ke.scene.MeshData = ke.geometry.create_cube_data(scale=1.0)
        sphere: ke.scene.MeshData = ke.geometry.create_sphere_data(
            radius=0.5, num_longitudes=48, num_latitudes=24,
        )
        floor_material: ke.material.PBRMaterial = self.create_pbr_material(
            base_color=rgba(ke.ColorType.LIGHT_GRAY), roughness=0.7,
        )
        wall_material: ke.material.PBRMaterial = self.create_pbr_material(
            base_color=rgba(ke.ColorType.PASTEL_BLUE), roughness=0.85,
        )
        floor = self.scene.add_mesh(path="/room/floor", mesh_data=cube, material=floor_material)
        floor.set_local_scale(ke.Vec3(24, 0.2, 20))
        floor.set_local_translation(ke.Vec3(0, -0.1, 0))
        wall = self.scene.add_mesh(path="/room/back", mesh_data=cube, material=wall_material)
        wall.set_local_scale(ke.Vec3(24, 4, 0.2))
        wall.set_local_translation(ke.Vec3(0, 2, -8.5))

        # Left -> right: roughness; front -> back: metallic factor.
        for row, metallic in enumerate((0.0, 0.5, 1.0)):
            for column, roughness in enumerate((0.05, 0.2, 0.4, 0.65, 1.0)):
                material: ke.material.PBRMaterial = self.create_pbr_material(
                    base_color=rgba(ke.ColorType.PASTEL_ORANGE),
                    metallic=metallic, roughness=roughness,
                )
                surface = self.scene.add_mesh(
                    path=f"/materials/row_{row}/sphere_{column}",
                    mesh_data=sphere, material=material,
                )
                surface.set_local_translation(ke.Vec3((column - 2) * 1.5, 0.5, 1 - row * 1.7))

        texture = self.load_texture(
            Path(ke.__file__).parent / "assets/textures/awesomeface.png", srgb=True,
        )
        textured: ke.material.PBRMaterial = self.create_pbr_material(
            base_color=rgba(ke.ColorType.WHITE), roughness=0.6, base_color_texture=texture,
        )
        for index in range(3):
            surface = self.scene.add_mesh(
                path=f"/textures/cube_{index}", mesh_data=cube, material=textured,
            )
            surface.set_local_translation(ke.Vec3(-3.8, 0.6 + index * 1.1, -6.5))
            surface.set_local_rotation_axis_angle(axis=ke.Vec3(0, 1, 0), angle_radians=index * 0.3)

        self.rotor: ke.scene.Prim = self.scene.define_prim("/moving", ke.scene.PrimType.XFORM)
        self.rotor.set_local_translation(ke.Vec3(0, 1.1, 3.5))
        arm = self.scene.add_mesh(path="/moving/arm", mesh_data=cube, material=wall_material)
        arm.set_local_scale(ke.Vec3(4, 0.2, 0.3))
        for index, x in enumerate((-2.0, 2.0)):
            surface = self.scene.add_mesh(
                path=f"/moving/cube_{index}", mesh_data=cube, material=textured,
            )
            surface.set_local_translation(ke.Vec3(x, 0, 0))

        self.setup_motion_samples()
        self.setup_material_samples()
        self.setup_light_samples()
        self.setup_selection_sample(cube, sphere, wall_material)

    def setup_motion_samples(self):
        self.sample_groups["Skinning"] = self.scene.define_prim("/samples/skin", ke.scene.PrimType.XFORM)
        mesh = ke.geometry.create_sphere_data(radius=1, num_longitudes=24, num_latitudes=24)
        positions = np.asarray(mesh.vertices, dtype=np.float32) * [0.4, 1.6, 0.4] + [0, 1.6, 0]
        normals = np.asarray(mesh.normals, dtype=np.float32) / [0.4, 1.6, 0.4]
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        tree = ke.animation.SkeletonTree(
            ["root", "tip"], [-1, 0], [[0, 0, 0], [0, 1.6, 0]], [[1, 0, 0, 0]] * 2,
        )
        weights = np.zeros((len(positions), 4), dtype=np.float32)
        weights[:, 1] = np.clip((positions[:, 1] - 0.8) / 1.6, 0, 1)
        weights[:, 0] = 1 - weights[:, 1]
        inverse_bind = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
        inverse_bind[1, 1, 3] = -1.6
        asset = ke.visual.SkinnedSurfaceAsset(
            tree, positions.astype(np.float32), normals.astype(np.float32),
            np.asarray(mesh.indices, dtype=np.int32),
            np.tile([0, 1, 0, 0], (len(positions), 1)).astype(np.int32),
            weights, np.array([0, 1], dtype=np.int32), inverse_bind,
        )
        material = self.create_pbr_material(roughness=0.65)
        first: ke.visual.SkinnedSurface = ke.visual.SkinnedSurface.create(
            self, "/samples/skin/first", asset, material=material,
        )
        self.skins = [first, first.create_instance("/samples/skin/second"),
                      first.create_instance("/samples/skin/third")]
        for idx, (surface, preset) in enumerate(zip(
            self.skins, (ke.ColorType.ROYAL_BLUE, ke.ColorType.ORANGE, ke.ColorType.SEA_GREEN),
        )):
            surface.set_color(rgba(preset))
            surface.prims[0].set_local_translation(ke.Vec3(-9 + idx * 2, 0, -5))

        self.sample_groups["Deformation"] = self.scene.define_prim("/samples/deform", ke.scene.PrimType.XFORM)
        self.deform_mesh = ke.geometry.create_sphere_data(radius=1, num_longitudes=32, num_latitudes=24)
        self.deform_positions = np.asarray(self.deform_mesh.vertices, dtype=np.float32)
        self.deform_normals = np.asarray(self.deform_mesh.normals, dtype=np.float32)
        material = self.create_pbr_material(base_color=rgba(ke.ColorType.ROYAL_BLUE), roughness=0.4)
        self.deformed = self.scene.add_mesh("/samples/deform/moving", self.deform_mesh, material)
        self.deformed.set_local_translation(ke.Vec3(-9, 1.3, 0))
        # Native geometry updates affect a batch; keep the reference in a separate batch.
        reference_material = self.create_pbr_material(
            base_color=rgba(ke.ColorType.ROYAL_BLUE), roughness=0.4,
        )
        reference = self.scene.add_mesh("/samples/deform/source", self.deform_mesh, reference_material)
        reference.set_local_translation(ke.Vec3(-6.5, 1.3, 0))

        self.sample_groups["CPU instances"] = self.scene.define_prim("/samples/cpu", ke.scene.PrimType.XFORM)
        self.instance_mesh = ke.geometry.create_cube_data(scale=0.4)
        self.instance_material = self.create_pbr_material(base_color=rgba(ke.ColorType.ORANGE))
        self.cpu_instances = self.scene.add_mesh(
            "/samples/cpu/cubes", self.instance_mesh, self.instance_material,
            transform_source=ke.render.TransformSource.EXTERNAL_BUFFER,
        )
        self.cpu_transforms = np.tile(np.eye(4, dtype=np.float32), (36, 1, 1))
        rows, columns = np.divmod(np.arange(36), 6)
        # ExternalBuffer matrices store columns; translation occupies row 3 in NumPy.
        self.cpu_transforms[:, 3, 0] = -10 + columns * 0.7
        self.cpu_transforms[:, 3, 2] = 3 + rows * 0.7
        self.instance_phases = (rows + columns).astype(np.float32) * 0.6

    def setup_material_samples(self):
        self.sample_groups["Materials"] = self.scene.define_prim("/samples/materials", ke.scene.PrimType.XFORM)
        rows, columns = np.indices((64, 64))
        checks = (rows // 8 + columns // 8) % 2 == 0
        pixels = np.empty((64, 64, 4), dtype=np.uint8)
        pixels[checks] = [240, 130, 30, 255]
        pixels[~checks] = [30, 90, 220, 255]
        self.memory_texture = self.create_texture(pixels, srgb=True)
        color = self.create_color_material(style=ke.material.ColorStyle.TEXTURED)
        phong = self.create_phong_material(diffuse=[1, 1, 1], shininess=64)
        pbr = self.create_pbr_material(roughness=0.4)
        self.sample_materials = [color, phong, pbr]
        self.materials_unlit = False
        self.memory_enabled = True
        self.set_sample_textures()
        cube = ke.geometry.create_cube_data(scale=1.3)
        for idx, material in enumerate(self.sample_materials):
            surface = self.scene.add_mesh(f"/samples/materials/model_{idx}", cube, material)
            surface.set_local_translation(ke.Vec3(6 + idx * 2, 0.8, -5))

        self.sample_groups["Mask and normals"] = self.scene.define_prim("/samples/maps", ke.scene.PrimType.XFORM)
        pixels[~checks, 3] = 0
        self.mask_material = self.create_pbr_material(
            base_color_texture=self.create_texture(pixels, srgb=True), roughness=0.8,
        )
        self.masked = self.scene.add_mesh("/samples/maps/mask", cube, self.mask_material)
        self.masked.set_local_translation(ke.Vec3(6.5, 1, 0))
        self.mask_enabled = True
        self.masked.set_alpha_mode(ke.render.AlphaMode.MASK, 0.5)
        self.normal_material = self.create_pbr_material(roughness=0.45)
        sphere = ke.geometry.create_sphere_data(radius=0.9, num_longitudes=48, num_latitudes=24)
        self.normal_surface = self.scene.add_mesh("/samples/maps/normal", sphere, self.normal_material)
        self.normal_surface.set_local_translation(ke.Vec3(9.5, 1, 0))
        normal = np.zeros((64, 64, 3), dtype=np.float32)
        normal[..., 0] = 0.7 * np.sin(columns * math.tau / 16)
        normal[..., 1] = 0.7 * np.sin(rows * math.tau / 16)
        normal[..., 2] = 1
        normal /= np.linalg.norm(normal, axis=2, keepdims=True)
        self.normal_texture = self.create_texture(((normal + 1) * 127.5).astype(np.uint8), srgb=False)
        self.normal_enabled = True
        self.normal_material.normal_texture = self.normal_texture

    def set_sample_textures(self):
        texture = self.memory_texture if self.memory_enabled else None
        self.sample_materials[0].base_color_texture = texture
        self.sample_materials[1].diffuse_map = texture
        self.sample_materials[2].base_color_texture = texture

    def setup_light_samples(self):
        self.local_lights = {}
        for idx, (name, light_type) in enumerate((
            ("point", ke.scene.PointLight), ("spot", ke.scene.SpotLight), ("rect", ke.scene.RectLight),
        )):
            light = light_type()
            light.intensity = 40
            prim = self.scene.define_prim(f"/lights/samples/{name}", ke.scene.PrimType.LIGHT)
            if name == "rect":
                prim.set_local_translation(ke.Vec3(-3 + idx * 3, 3, -6))
                prim.set_local_rotation_axis_angle(ke.Vec3(1, 0, 0), -math.pi / 2)
                light.width = light.height = 1.5
            else:
                light.position = ke.Vec3(-3 + idx * 3, 3, -6)
                if name == "spot":
                    light.direction = ke.Vec3(0, -1, 0)
            getattr(prim, f"set_{name}_light")(light)
            prim.set_visible(False)
            self.local_lights[name] = prim

    def setup_selection_sample(self, cube, sphere, material):
        self.sample_groups["Picking"] = self.scene.define_prim("/samples/picking", ke.scene.PrimType.XFORM)
        blocker = self.scene.add_mesh("/samples/picking/occluder", cube, material)
        blocker.set_local_scale(ke.Vec3(2.6, 2.6, 0.3))
        blocker.set_local_translation(ke.Vec3(0, 1.3, 6.5))
        self.hidden_target = self.scene.add_mesh("/samples/picking/behind", sphere, material)
        self.hidden_target.set_local_translation(ke.Vec3(0, 1.3, 5.5))

    def set_cuda_enabled(self, enabled):
        if enabled and self.cuda_instances is None:
            import torch
            try:
                from cuda.bindings import runtime
            except ImportError:
                self.cuda_status = "Install cuda-bindings to enable this sample."
                return
            if not torch.cuda.is_available():
                self.cuda_status = "Torch CUDA is unavailable."
                return
            # Independent external streams need separate batches.
            material = self.create_pbr_material(base_color=rgba(ke.ColorType.SEA_GREEN))
            self.cuda_instances = self.scene.add_mesh(
                "/samples/cuda/cubes", self.instance_mesh, material,
                transform_source=ke.render.TransformSource.EXTERNAL_BUFFER,
            )
            self.cuda_transforms = torch.tensor(self.cpu_transforms, device="cuda")
            self.cuda_transforms[:, 3, 0] += 16
            self.cuda_phases = torch.tensor(self.instance_phases, device="cuda")
            self.cuda_ready = torch.cuda.Event()
            self.cuda_version = 0
        self.cuda_enabled = enabled
        if self.cuda_instances is not None:
            self.cuda_instances.set_visible(enabled)
        self.cuda_status = "CUDA transforms: 36 instances, no CPU readback." if enabled else "CUDA instances are off."
        self.samples_dirty = True

    def update_motion_samples(self):
        self.rotor.set_local_rotation_axis_angle(ke.Vec3(0, 1, 0), self.angle)
        for idx, surface in enumerate(self.skins):
            angle = 0.8 * math.sin(self.angle + idx * math.tau / 3)
            c, s = math.cos(angle), math.sin(angle)
            matrices = np.tile(np.eye(4, dtype=np.float32), (2, 1, 1))
            matrices[1, :2, :2] = [[c, -s], [s, c]]
            matrices[1, :3, 3] = [1.6 * s, 1.6 * (1 - c), 0]
            surface.update_joint_matrices(matrices)
        scale = np.array([1 + 0.25 * math.sin(self.angle), 1 - 0.4 * math.sin(self.angle), 1])
        normals = self.deform_normals / scale
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        self.deformed.update_geometry(self.deform_positions * scale, normals)
        self.cpu_transforms[:, 3, 1] = 0.65 + 0.4 * np.sin(self.angle + self.instance_phases)
        self.get_scene_render_system().update_instances(self.cpu_instances.component, self.cpu_transforms)
        if self.cuda_enabled:
            import torch
            from kangengine.utils.sim_buffer import SimBuffer, to_external_transform_desc

            self.cuda_transforms[:, 3, 1] = 0.65 + 0.4 * torch.sin(self.angle + self.cuda_phases)
            stream = torch.cuda.current_stream(self.cuda_transforms.device)
            self.cuda_ready.record(stream)
            self.cuda_version += 1
            buffer = SimBuffer(
                self.cuda_transforms, sim_device=str(self.cuda_transforms.device),
                memory_type="cuda_device", device_id=self.cuda_transforms.device.index,
                version=self.cuda_version, stream_handle=stream.cuda_stream,
                ready_event_handle=self.cuda_ready.cuda_event,
            )
            descriptor, self.cuda_buffer = to_external_transform_desc(buffer)
            self.get_scene_render_system().set_external_buffer(self.cuda_instances.component, descriptor)
        self.samples_dirty = False

    def on_ray_picked(self, result):
        self.last_pick = (f"{result.prim.get_path()} | instance {result.instance_index} | {result.distance:.2f} m"
                          if result.hit and result.prim is not None else "No surface hit.")

    def fixed_update(self, fixed_dt):
        if self.animate:
            self.angle = (self.angle + fixed_dt * 0.6) % math.tau
            self.samples_dirty = True

    def pre_render(self):
        if not self.controls_docked:
            self.controls_docked = ke.imgui.dock_window("Scene Controls", "Inspector")
        if ke.imgui.begin("Scene Controls"):
            _, self.animate = ke.imgui.checkbox("Animate", self.animate)
            changed, self.angle = ke.imgui.slider_float("Pose phase", self.angle, 0, math.tau)
            if changed:
                self.animate = False
                self.samples_dirty = True
            if ke.imgui.button("Overview"):
                self.set_camera_view(position=[18, 16, 24], target=[0, 1, 0])
            if ke.imgui.collapsing_header("Samples"):
                targets = {"Skinning": [-7, 1.5, -5], "Deformation": [-8, 1, 0],
                           "CPU instances": [-8, 0.5, 5], "Materials": [8, 1, -5],
                           "Mask and normals": [8, 1, 0], "Picking": [0, 1.3, 5.5]}
                for name, prim in self.sample_groups.items():
                    changed, visible = ke.imgui.checkbox(f"{name}##sample_visibility", prim.is_visible())
                    if changed:
                        prim.set_visible(visible)
                    ke.imgui.same_line()
                    if ke.imgui.button(f"Focus##{name}"):
                        target = targets[name]
                        position = [target[0] + 1, target[1] + 3, target[2] + 8]
                        if name == "Picking":
                            position = [0, 1.3, 12]
                        self.set_camera_view(position=position, target=target)
                ke.imgui.text_wrapped("Skin clones share one source; each has its own pose.")
                ke.imgui.text_wrapped("Deformation: left changes; right keeps the source shape.")
                if self.render_backend == "ovrtx":
                    changed, enabled = ke.imgui.checkbox("CUDA instances", self.cuda_enabled)
                    if changed:
                        self.set_cuda_enabled(enabled)
                    if self.cuda_enabled and ke.imgui.button("Focus CUDA"):
                        self.set_camera_view(position=[8, 5, 13], target=[8, 0.5, 5])
                    ke.imgui.text_wrapped(self.cuda_status)
                else:
                    ke.imgui.text_wrapped("CUDA sample: launch with --backend ovrtx.")
            if ke.imgui.collapsing_header("Materials"):
                ke.imgui.text("Left to right: Color, Phong, PBR.")
                changed, self.materials_unlit = ke.imgui.checkbox("Unlit", self.materials_unlit)
                if changed:
                    for material in self.sample_materials:
                        material.unlit = self.materials_unlit
                changed, self.memory_enabled = ke.imgui.checkbox("Memory texture", self.memory_enabled)
                if changed:
                    self.set_sample_textures()
                changed, self.mask_enabled = ke.imgui.checkbox("Alpha mask", self.mask_enabled)
                if changed:
                    self.masked.set_alpha_mode(ke.render.AlphaMode.MASK if self.mask_enabled else ke.render.AlphaMode.OPAQUE)
                changed, self.normal_enabled = ke.imgui.checkbox("Normal map", self.normal_enabled)
                if changed:
                    self.normal_material.normal_texture = self.normal_texture if self.normal_enabled else None
                ke.imgui.text_wrapped("OVRTX alpha blending is not supported.")
            if ke.imgui.collapsing_header("Picking"):
                ke.imgui.text_wrapped(self.last_pick)
                if ke.imgui.button("Select behind occluder"):
                    self.sample_groups["Picking"].set_visible(True)
                    self.set_camera_view(position=[0, 1.3, 12], target=[0, 1.3, 5.5])
                    self.select_prim(self.hidden_target.prim)
                ke.imgui.text_wrapped("Click a moving skin or an instance to test its current pose.")
            self.draw_light_controls()
        ke.imgui.end()
        if self.samples_dirty:
            self.update_motion_samples()

    def draw_light_controls(self):
        if ke.imgui.collapsing_header("Lighting"):
            directional = self.scene.get_prim_at_path("/lights/default_directional")
            changed, enabled = ke.imgui.checkbox("Directional", directional.is_visible())
            if changed:
                directional.set_visible(enabled)
            changed, self.environment_enabled = ke.imgui.checkbox("Environment", self.environment_enabled)
            if changed:
                self.environment.set_visible(visible=self.environment_enabled)
            changed, intensity = ke.imgui.slider_float(
                "Environment intensity", self.environment_light.intensity, 0.0, 2.0,
            )
            if changed:
                self.environment_light.intensity = intensity
                self.environment.set_dome_light(light=self.environment_light)
            ke.imgui.text("Spheres: roughness increases left to right.")
            ke.imgui.text("Metallic increases front to back.")
            for name, prim in self.local_lights.items():
                changed, enabled = ke.imgui.checkbox(name.title(), prim.is_visible())
                if changed:
                    prim.set_visible(enabled)
            if ke.imgui.button("Focus local lights"):
                self.set_camera_view(position=[0, 7, 1], target=[0, 0, -6])
            ke.imgui.text_wrapped("Turn off Directional and Environment to isolate local lights.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("native", "ovrtx"), default="native")
    parser.add_argument("--hdri-path", type=Path, help="Latlong environment image for ovrtx")
    args = parser.parse_args()
    if args.hdri_path is not None and args.backend != "ovrtx":
        parser.error("--hdri-path requires --backend ovrtx; Native supports uniform dome lighting")
    app = SceneRendererApp()
    app.hdri_path = args.hdri_path
    try:
        app.initialize(width=1280, height=800, up_axis=ke.UpAxis.Y, render_backend=args.backend)
        app.start()
    finally:
        app.close()


if __name__ == "__main__":
    main()
