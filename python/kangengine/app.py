"""Python-friendly base App for KangEngine examples and tools.

The native C++ App still owns the main loop, camera controls, renderer, ImGui
frame lifecycle, and scene rendering.  This wrapper adds default lifecycle
hooks and small input helpers so examples can inherit from `kangengine.App`
without talking directly to the pybind class.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from ._core import _ke

keys = _ke.keys
scene = _ke.scene
NativeApp = _ke.App


def _safe_scene_segment(value, fallback: str = "item") -> str:
    text = str(value or fallback)
    safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in text)
    return safe or fallback


def _texture_path_candidate(path) -> Path:
    """Return an existing texture path, accepting OBJ/MTL Windows separators."""
    candidate = Path(path).expanduser()
    if candidate.exists():
        return candidate
    text = str(path)
    if "\\" in text:
        normalized = Path(text.replace("\\", "/")).expanduser()
        if normalized.exists():
            return normalized
    return candidate


class RenderablePrimView:
    """User-facing view for one scene prim and its renderer resources."""

    def __init__(self, app: "App", prim, component):
        self._app = app
        self.prim = prim
        self.component = component

    @property
    def _render_system(self):
        return self._app.get_scene_render_system()

    @property
    def path(self) -> str:
        return self.prim.get_path()

    def set_visible(self, visible: bool):
        self.component.visible = bool(visible)
        return self

    def set_double_sided(self, enabled: bool = True):
        self._render_system.set_double_sided(self.component, bool(enabled))
        return self

    def set_casts_shadow(self, enabled: bool = True):
        self._render_system.set_casts_shadow(self.component, bool(enabled))
        return self

    def set_alpha_mode(self, mode, cutoff: float = 0.5):
        """Choose opaque, cutout-mask, or blended alpha rendering."""
        self._render_system.set_alpha_mode(self.component, mode, float(cutoff))
        return self

    def set_base_color(self, color):
        """Set the per-instance base-color multiplier for this renderable."""
        self.prim.set_display_color_alpha(color)
        return self

    def get_base_color(self):
        """Return the per-instance base-color multiplier."""
        return self.prim.get_display_color_alpha()

    def set_texture(self, texture, role_or_slot=_ke.TextureRole.BaseColor):
        self._render_system.set_texture(self.component, texture, role_or_slot)
        return self

    def set_material(self, material):
        """Replace this renderable's material and move it to the right batch."""
        self._render_system.set_material(self.component, material)
        return self

    def set_transform_buffer(
        self,
        transforms,
        *,
        sim_device=None,
        sync_policy=None,
    ):
        """Set the ``[N, 4, 4]`` buffer of an ExternalBuffer renderable."""
        from .utils.sim_buffer import to_external_transform_desc

        descriptor, _ = to_external_transform_desc(
            transforms,
            sim_device=sim_device,
            dtype="float32",
            name=f"{self.path}:transforms",
            sync_policy=(
                _ke.ExternalSyncPolicy.NONE
                if sync_policy is None
                else sync_policy
            ),
        )

        self._render_system.set_external_buffer(self.component, descriptor)
        return self

    def update_geometry(self, positions, normals=None):
        """Update dynamic vertex positions and optional normals."""
        self._render_system.update_geometry(self.component, positions, normals)
        return self

    def update_skinning(self, bone_matrices):
        """Update skinned bone matrices for this renderable."""
        self._render_system.update_skinning(self.component, bone_matrices)
        return self

    def remove(self):
        return self._app.remove_prim(self.prim)


class DebugPrimitiveView(RenderablePrimView):
    """Component-backed instanced debug primitive view."""

    def update_instances(self, transforms, colors=None):
        """Update raw instance transforms and optional colors."""
        self._render_system.update_instances(self.component, transforms, colors)
        return self

    def update_lines(self, starts, ends, colors=None):
        """Update this view as instanced debug lines."""
        scene.DebugDraw.update_component_lines(
            self._app,
            self.component,
            starts,
            ends,
            colors,
        )
        return self

    def update_arrows(self, starts, ends, colors=None):
        """Update this view as instanced debug arrows."""
        scene.DebugDraw.update_component_arrows(
            self._app,
            self.component,
            starts,
            ends,
            colors,
        )
        return self


class ObjImportView:
    """Result returned by SceneContext.add_obj()."""

    def __init__(self, root, views, info):
        self.root = root
        self.views = views
        self.info = info

    def __iter__(self):
        return iter(self.views)

    def __len__(self):
        return len(self.views)


class SceneContext:
    """Scene-facing facade connected to the owning App renderer.

    Use this for common add/remove workflows. It keeps renderer handles inside
    the app-facing layer while preserving access to the underlying scene backend
    for lower-level operations.
    """

    def __init__(self, app: "App"):
        self._app = app

    @property
    def native(self):
        """Return the native SceneBackend escape hatch.

        Prefer SceneContext helpers such as add_mesh(), add_ground(), and
        define_prim() for authored scene objects. Use native only when a C++ or
        pybind API explicitly requires SceneBackend.
        """
        return self._app.get_native_scene()

    def define_prim(self, path: str, prim_type):
        return self.native.define_prim(path, prim_type)

    def get_prim_at_path(self, path: str):
        return self.native.get_prim_at_path(path)

    def get_root_prim(self):
        return self.native.get_root_prim()

    def add_renderable(
        self,
        prim,
        material,
        transform_source=None,
    ):
        """Register a scene prim as renderable through RenderComponent.

        This is the preferred public path for authored scene objects. It
        returns a RenderablePrimView facade instead of exposing the native
        renderer handle. Pass a Material for material-first rendering; raw
        Shader objects are still accepted as a compatibility path and are
        wrapped internally by the native SceneRenderSystem.
        """
        if transform_source is None:
            transform_source = _ke.TransformSource.SceneGraph
        self._app._add_renderable(material, prim, transform_source)
        component = prim.get_render_component()
        if component is None:
            raise RuntimeError(f"failed to register renderable prim: {prim.get_path()}")
        return RenderablePrimView(self._app, prim, component)

    def add_mesh(
        self,
        path: str,
        mesh_data,
        material,
        color=None,
        transform_source=None,
        uri=None,
    ):
        prim = self.define_prim(path, scene.PrimType.Mesh)
        mesh_handle = self._app._register_mesh_resource(
            mesh_data,
            display_name=Path(path).name or "Mesh",
            uri=uri or f"scene://mesh{path}",
        )
        mesh_component = prim.get_mesh_component()
        if mesh_component is None:
            mesh_component = prim.add_mesh_component()
        # Keep the render prim on the fast path by caching the shared mesh
        # directly. The resource handle remains the editor/resource identity,
        # but renderer registration no longer depends on traversing /.Resources.
        mesh_component.mesh_data = mesh_data
        mesh_component.resource_handle = mesh_handle
        if color is not None:
            prim.set_display_color_alpha(color)
        return self.add_renderable(prim, material, transform_source)

    def add_obj(
        self,
        path: str,
        obj_path,
        *,
        transform_source=None,
        double_sided=False,
    ):
        """Load an OBJ/MTL file and add material-subset mesh prims.

        Multi-material OBJ files become one Xform root with one renderable child
        per material subset. Single-material OBJ files still use the same code
        path, which keeps resource registration and material creation
        consistent.
        """
        obj_path = Path(obj_path)
        info = _ke.asset.load_obj_with_materials(str(obj_path))
        root = self.define_prim(path, scene.PrimType.Xform)

        subsets = list(info.subsets)
        if not subsets:
            subsets = [
                SimpleNamespace(
                    name=obj_path.stem,
                    material_index=int(info.primary_material_index),
                    mesh_data=info.mesh_data,
                )
            ]

        views = []
        used_child_names = set()
        for index, subset in enumerate(subsets):
            material = self._app._create_obj_subset_material(info, subset)
            base_child_name = _safe_scene_segment(
                subset.name if subset.name != "default" else f"subset_{index}",
                f"subset_{index}",
            )
            child_name = base_child_name
            suffix = 1
            while child_name in used_child_names:
                child_name = f"{base_child_name}_{suffix}"
                suffix += 1
            used_child_names.add(child_name)
            child_path = f"{path.rstrip('/')}/{child_name}"
            view = self.add_mesh(
                child_path,
                subset.mesh_data,
                material,
                color=self._app._obj_subset_display_color(info, subset),
                transform_source=transform_source,
                uri=f"{obj_path}#{subset.name or index}",
            )
            if double_sided:
                view.set_double_sided(True)
            if self._app._obj_subset_has_alpha_texture(info, subset):
                view.set_alpha_mode(_ke.AlphaMode.Mask)
            elif self._app._obj_subset_alpha(info, subset) < 1.0:
                view.set_alpha_mode(_ke.AlphaMode.Blend)
            views.append(view)

        return ObjImportView(root, views, info)

    def add_ground(self, path: str = "/ground", scale: float = 20.0, shader=None):
        if shader is None:
            shader = self._app.create_asset_shader("common.vs", "checkerboard.fs")
            self._app.configure_checker_shader(shader)
        return self.add_mesh(
            path,
            _ke.geometry.create_plane_data(float(scale), self._app.up_axis),
            shader,
        )

    def log_lines(
        self,
        path: str,
        shader,
        starts,
        ends,
        colors=None,
        radius: float = 0.005,
        segments: int = 8,
    ):
        component = scene.DebugDraw.log_component_lines(
            self._app,
            shader,
            path,
            starts,
            ends,
            colors,
            float(radius),
            int(segments),
        )
        if component is None or component.owner is None:
            raise RuntimeError(f"failed to register debug line prim: {path}")
        return DebugPrimitiveView(self._app, component.owner, component)

    def log_arrows(
        self,
        path: str,
        shader,
        starts,
        ends,
        colors=None,
        radius: float = 0.02,
        segments: int = 12,
    ):
        component = scene.DebugDraw.log_component_arrows(
            self._app,
            shader,
            path,
            starts,
            ends,
            colors,
            float(radius),
            int(segments),
        )
        if component is None or component.owner is None:
            raise RuntimeError(f"failed to register debug arrow prim: {path}")
        return DebugPrimitiveView(self._app, component.owner, component)

    def remove_prim(self, path_or_prim):
        return self._app.remove_prim(path_or_prim)


class App(NativeApp):
    """Base class for Python KangEngine apps.

    This mirrors the C++ App lifecycle:

    - `setup()` runs once before the loop starts.
    - `preRender()` runs before scene rendering each frame.
    - `render()` runs while the ImGui frame is active.
    - `postRender()` runs after UI rendering and buffer swap setup.

    The C++ implementation still handles the default camera controls:
    WASD/mouse navigation, H to hide UI, Escape to close, framebuffer updates,
    and the built-in scene/performance panels.
    """

    def __init__(self):
        super().__init__()
        self._previous_key_states = {}
        self.width = 1920
        self.height = 1080
        self.hide_ui = False
        # Lazily populated after the native graphics device/context exists.
        # Pure compute/headless apps can leave this as None.
        self.shaders = None
        self.materials = []
        self.textures = []
        self.up_axis = _ke.UpAxis.Y
        self.graphics_backend_type = _ke.BackendType.OpenGL
        self.scene_backend_type = scene.BackendType.Native
        self.headless = False
        self.scene = SceneContext(self)
        self.resources = self.get_scene_resource_manager()
        self._resource_handles_by_object_id = {}
        self._resource_counter = 0
        self._textures_by_uri = {}

    def get_native_scene(self):
        """Return the native SceneBackend escape hatch.

        Public Python code should prefer self.scene for authored scene workflows.
        Use this only when a C++ or pybind API explicitly requires SceneBackend.
        """
        return super().get_scene()

    def get_scene(self):
        """Return the Python-friendly SceneContext facade.

        This mirrors the ``app.scene`` property. Use get_native_scene() or
        scene.native only when a C++ or pybind API explicitly requires the
        native SceneBackend.
        """
        return self.scene

    def package_asset_path(self, *parts: str) -> str:
        return str(Path(_ke.__file__).resolve().parent / "assets" / Path(*parts))

    ############# Shaders ###########################################
    def _bind_common_ubos(self, *shaders):
        for shader in shaders:
            shader.use()
            shader.set_uniform_block_binding("cameraUBO", 0)
            shader.set_uniform_block_binding("lightUBO", 1)
            shader.set_uniform_block_binding("shadowUBO", 2)
        return shaders[0] if len(shaders) == 1 else shaders

    def create_asset_shader(self, vertex_shader: str, fragment_shader: str):
        renderer = self.get_renderer()
        device = renderer.device() if renderer is not None else None
        if device is None:
            raise RuntimeError(
                "shader creation requires an initialized graphics device; "
                "call this after initialize(), usually from setup()."
            )
        shader = device.create_shader_from_file(
            self.package_asset_path("shaders", vertex_shader),
            self.package_asset_path("shaders", fragment_shader),
        )
        self._bind_common_ubos(shader)
        if str(fragment_shader).endswith("checkerboard.fs"):
            self.set_background_shader(shader)
        self._register_shader_resource(
            shader,
            f"{vertex_shader}+{fragment_shader}",
            f"asset://shaders/{vertex_shader}+{fragment_shader}",
        )
        return shader

    def set_texture_uniform(self, shader, unit: int = 0, name: str = "uTexture"):
        shader.use()
        shader.set_int(name, int(unit))
        return shader

    def configure_checker_shader(
        self,
        shader,
        color1=None,
        color2=None,
    ):
        if color1 is None:
            color1 = _ke.vec4(1.0, 1.0, 1.0, 1.0)
        if color2 is None:
            preset = _ke.ColorLibrary.get(_ke.ColorType.DARK_GREEN)
            color2 = _ke.vec4(preset.r, preset.g, preset.b, preset.a)
        shader.use()
        shader.set_vec4("checkerColor1", color1)
        shader.set_vec4("checkerColor2", color2)
        return shader

    def create_standard_shaders(self, *, force: bool = False):
        """Create or return the cached standard shader bundle.

        This intentionally stays lazy: pure compute/headless apps do not pay for
        shader creation, while offscreen render/headless apps can call this
        after initialize() once a graphics context exists.
        """
        if self.shaders is not None and not force:
            return self.shaders

        self.shaders = SimpleNamespace(
            common=self.create_asset_shader("common.vs", "common.fs"),
            common_texture=self.create_asset_shader("common.vs", "commonTex.fs"),
            phong=self.create_asset_shader("common.vs", "phong.fs"),
            pbr=self.create_asset_shader("common.vs", "pbr_forward.fs"),
            common_debug=self.create_asset_shader("common.vs", "debug_checker.fs"),
            skinned=self.create_asset_shader("skinned_mesh.vs", "common.fs"),
            skinned_texture=self.create_asset_shader(
                "skinned_mesh.vs",
                "commonTex.fs",
            ),
            skinned_phong=self.create_asset_shader("skinned_mesh.vs", "phong.fs"),
            skinned_pbr=self.create_asset_shader(
                "skinned_mesh.vs",
                "pbr_forward.fs",
            ),
            skinned_debug=self.create_asset_shader(
                "skinned_mesh.vs",
                "debug_checker.fs",
            ),
            ground=self.create_asset_shader("common.vs", "checkerboard.fs"),
        )
        self.set_texture_uniform(self.shaders.common_texture)
        self.set_texture_uniform(self.shaders.skinned_texture)
        self.configure_checker_shader(self.shaders.ground)
        return self.shaders

    #################################################################

    ############# Helpers ###########################################
    def _remember_resource_handle(self, obj, handle):
        if obj is not None:
            # C++ resource entries for materials/textures/shaders are
            # intentionally non-owning. Holding the Python wrapper here keeps
            # those objects alive and prevents stale id() reuse from returning
            # an unrelated handle.
            self._resource_handles_by_object_id[id(obj)] = (obj, handle)
        return handle

    def _existing_resource_handle(self, obj):
        if obj is None:
            return None
        record = self._resource_handles_by_object_id.get(id(obj))
        if record is None:
            return None
        recorded_obj, handle = record
        return handle if recorded_obj is obj else None

    def _next_resource_name(self, obj, prefix: str):
        self._resource_counter += 1
        return f"{prefix}_{type(obj).__name__}_{self._resource_counter}"

    def _register_shader_resource(self, shader, display_name, uri):
        existing = self._existing_resource_handle(shader)
        if existing is not None:
            return existing
        handle = self.resources.register_shader(str(display_name), shader, str(uri))
        return self._remember_resource_handle(shader, handle)

    def _register_material_resource(self, material):
        existing = self._existing_resource_handle(material)
        if existing is not None:
            return existing
        name = self._next_resource_name(material, "Material")
        handle = self.resources.register_material(
            name,
            material,
            f"python://resource/Material/{name}",
        )
        return self._remember_resource_handle(material, handle)

    def _register_texture_resource(self, texture, display_name=None, uri=None):
        existing = self._existing_resource_handle(texture)
        if existing is not None:
            return existing
        name = str(display_name) if display_name else self._next_resource_name(texture, "Texture")
        handle = self.resources.register_texture(
            name,
            texture,
            str(uri) if uri is not None else f"python://resource/Texture/{name}",
        )
        return self._remember_resource_handle(texture, handle)

    def _register_mesh_resource(self, mesh_data, display_name=None, uri=None):
        existing = self._existing_resource_handle(mesh_data)
        if existing is not None:
            return existing
        name = str(display_name) if display_name else self._next_resource_name(mesh_data, "Mesh")
        handle = self.resources.register_mesh(
            name,
            mesh_data,
            str(uri) if uri is not None else f"python://resource/Mesh/{name}",
        )
        return self._remember_resource_handle(mesh_data, handle)

    def _remember_material(self, material):
        """Keep Python-created native materials alive for renderer users."""
        self.materials.append(material)
        self._register_material_resource(material)
        return material

    def create_vertex_color_material(self, *, shader=None):
        """Create and retain a material for display/vertex-color shaders."""
        if shader is None:
            shader = self.create_standard_shaders().common
        return self._remember_material(_ke.VertexColorMaterial(shader))

    def _remember_textures(self, *textures):
        """Keep Python-created native textures alive while materials use them."""
        for texture in textures:
            if texture is not None and texture not in self.textures:
                self.textures.append(texture)
                self._register_texture_resource(texture)

    def load_texture(self, path, *, flip: bool = True):
        """Load and retain a GPU texture, cached by normalized path."""
        texture_path = str(_texture_path_candidate(path).resolve())
        cached = self._textures_by_uri.get(texture_path)
        if cached is not None:
            return cached
        device = self.get_renderer().device()
        texture = device.create_texture(texture_path, bool(flip))
        self.textures.append(texture)
        self._register_texture_resource(
            texture,
            display_name=Path(texture_path).name,
            uri=texture_path,
        )
        self._textures_by_uri[texture_path] = texture
        return texture

    def create_phong_material(
        self,
        preset=None,
        *,
        shader=None,
        ambient=None,
        diffuse=None,
        specular=None,
        shininess=None,
        diffuse_map=None,
        specular_map=None,
        alpha_map=None,
        normal_map=None,
    ):
        """Create and retain a Phong material instance.

        Each call returns a distinct material, so two meshes can share the same
        shader while carrying different colors/textures and batching keys.
        """
        if shader is None:
            shader = self.create_standard_shaders().phong
        material = _ke.PhongMaterial(shader)
        if preset is not None:
            material.load_from_preset(preset)
        if ambient is not None:
            material.ambient = ambient
        if diffuse is not None:
            material.diffuse = diffuse
        if specular is not None:
            material.specular = specular
        if shininess is not None:
            material.shininess = float(shininess)
        if diffuse_map is not None:
            material.diffuse_map = diffuse_map
        if specular_map is not None:
            material.specular_map = specular_map
        if alpha_map is not None:
            material.alpha_map = alpha_map
        if normal_map is not None:
            material.normal_map = normal_map
        self._remember_textures(diffuse_map, specular_map, alpha_map, normal_map)
        return self._remember_material(material)

    def create_pbr_material(
        self,
        preset=None,
        *,
        shader=None,
        base_color=None,
        metallic=None,
        roughness=None,
        emissive_color=None,
        emissive_strength=None,
        base_color_texture=None,
        normal_texture=None,
        metallic_roughness_texture=None,
        metallic_texture=None,
        roughness_texture=None,
        ao_texture=None,
        orm_texture=None,
        emissive_texture=None,
    ):
        """Create and retain a PBR material instance.

        Material identity is intentionally per instance: sharing one material
        shares its parameters, while separate materials can use the same shader
        with different factors/textures.
        """
        if shader is None:
            shader = self.create_standard_shaders().pbr
        material = _ke.PBRMaterial(shader)
        if preset is not None:
            material.load_from_preset(preset)
        if base_color is not None:
            material.base_color = base_color
        if metallic is not None:
            material.metallic = float(metallic)
        if roughness is not None:
            material.roughness = float(roughness)
        if emissive_color is not None:
            material.emissive_color = emissive_color
        if emissive_strength is not None:
            material.emissive_strength = float(emissive_strength)
        if base_color_texture is not None:
            material.base_color_texture = base_color_texture
        if normal_texture is not None:
            material.normal_texture = normal_texture
        if metallic_roughness_texture is not None:
            material.metallic_roughness_texture = metallic_roughness_texture
        if metallic_texture is not None:
            material.metallic_texture = metallic_texture
        if roughness_texture is not None:
            material.roughness_texture = roughness_texture
        if ao_texture is not None:
            material.ao_texture = ao_texture
        if orm_texture is not None:
            material.orm_texture = orm_texture
        if emissive_texture is not None:
            material.emissive_texture = emissive_texture
        self._remember_textures(
            base_color_texture,
            normal_texture,
            metallic_roughness_texture,
            metallic_texture,
            roughness_texture,
            ao_texture,
            orm_texture,
            emissive_texture,
        )
        return self._remember_material(material)

    def _obj_material_info(self, info, material_index):
        if material_index is None or int(material_index) < 0:
            return None
        materials = list(info.materials)
        index = int(material_index)
        return materials[index] if index < len(materials) else None

    def _obj_subset_alpha(self, info, subset):
        material = self._obj_material_info(info, subset.material_index)
        if material is None:
            return 1.0
        return float(material.diffuse_color[3])

    def _obj_subset_has_alpha_texture(self, info, subset):
        material = self._obj_material_info(info, subset.material_index)
        return bool(material is not None and material.has_alpha_texture)

    def _obj_subset_display_color(self, info, subset):
        material = self._obj_material_info(info, subset.material_index)
        alpha = 1.0 if material is None else float(material.diffuse_color[3])
        return _ke.vec4(1.0, 1.0, 1.0, alpha)

    def _create_obj_subset_material(self, info, subset):
        material_info = self._obj_material_info(info, subset.material_index)
        if material_info is None:
            return self.create_phong_material(
                shader=self.create_standard_shaders().phong,
                diffuse=_ke.vec3(1.0, 1.0, 1.0),
                specular=_ke.vec3(0.05, 0.05, 0.05),
                shininess=16.0,
            )

        diffuse_map = None
        specular_map = None
        alpha_map = None
        normal_map = None
        if material_info.has_diffuse_texture:
            diffuse_path = _texture_path_candidate(material_info.diffuse_texture_path)
            if diffuse_path.exists():
                diffuse_map = self.load_texture(diffuse_path, flip=True)
        if material_info.has_specular_texture:
            specular_path = _texture_path_candidate(
                material_info.specular_texture_path
            )
            if specular_path.exists():
                specular_map = self.load_texture(specular_path, flip=True)
        if material_info.has_alpha_texture:
            alpha_path = _texture_path_candidate(material_info.alpha_texture_path)
            if alpha_path.exists():
                alpha_map = self.load_texture(alpha_path, flip=True)
        if material_info.has_normal_texture:
            normal_path = _texture_path_candidate(material_info.normal_texture_path)
            if normal_path.exists():
                normal_map = self.load_texture(normal_path, flip=True)

        return self.create_phong_material(
            shader=self.create_standard_shaders().phong,
            ambient=_ke.vec3(
                float(material_info.ambient_color[0]),
                float(material_info.ambient_color[1]),
                float(material_info.ambient_color[2]),
            ),
            diffuse=_ke.vec3(
                float(material_info.diffuse_color[0]),
                float(material_info.diffuse_color[1]),
                float(material_info.diffuse_color[2]),
            ),
            specular=_ke.vec3(
                float(material_info.specular_color[0]),
                float(material_info.specular_color[1]),
                float(material_info.specular_color[2]),
            ),
            shininess=float(material_info.shininess),
            diffuse_map=diffuse_map,
            specular_map=specular_map,
            alpha_map=alpha_map,
            normal_map=normal_map,
        )

    def _add_renderable(self, material_or_shader, prim, transform_source=None):
        """Low-level renderer handle path used by internal bridges."""
        if transform_source is None:
            transform_source = _ke.TransformSource.SceneGraph
        return super().add_renderable(material_or_shader, prim, transform_source)

    def _add_skinned_renderable(
        self,
        material_or_shader,
        prim,
        skinned_mesh_data,
        transform_source=None,
    ):
        """Low-level skinned renderer handle path used by internal bridges."""
        if transform_source is None:
            transform_source = _ke.TransformSource.SceneGraph
        return super().add_skinned_renderable(
            material_or_shader,
            prim,
            skinned_mesh_data,
            transform_source,
        )

    def add_ground(self, path: str = "/ground", scale: float = 20.0, shader=None):
        return self.scene.add_ground(path, scale, shader)

    def add_mesh(
        self,
        path: str,
        mesh_data,
        material=None,
        color=None,
        *,
        shader=None,
        uri=None,
    ):
        if material is None:
            material = shader
        if material is None:
            raise ValueError("add_mesh requires a material or compatibility shader")
        return self.scene.add_mesh(path, mesh_data, material, color=color, uri=uri)

    def add_obj(self, path: str, obj_path, **kwargs):
        return self.scene.add_obj(path, obj_path, **kwargs)

    def add_skinned_mesh(
        self,
        prim,
        material,
        skinned_mesh_data,
        transform_source=None,
    ):
        """Register a skinned mesh prim and return a RenderablePrimView.

        Prefer a Material. Shader input remains accepted for compatibility.
        """
        if transform_source is None:
            transform_source = _ke.TransformSource.SceneGraph
        self._add_skinned_renderable(
            material,
            prim,
            skinned_mesh_data,
            transform_source,
        )
        component = prim.get_render_component()
        if component is None:
            raise RuntimeError(
                f"failed to register skinned renderable prim: {prim.get_path()}"
            )
        return RenderablePrimView(self, prim, component)

    def as_vec3(self, value):
        if isinstance(value, _ke.vec3):
            return value
        if hasattr(value, "tolist"):
            value = value.tolist()
        if len(value) != 3:
            raise ValueError("vec3 value expected 3 elements")
        return _ke.vec3(float(value[0]), float(value[1]), float(value[2]))

    def set_camera_view(self, position, target):
        camera = self.get_camera()
        camera.set_camera_pos(self.as_vec3(position))
        camera.set_target_pos(self.as_vec3(target))
        return camera

    def set_render_hz(self, hz: float):
        return super().set_render_hz(float(hz))
    
    #################################################################

    def setup(self):
        pass

    def preRender(self):
        pass

    def render(self):
        pass

    def postRender(self):
        pass

    def cleanup(self):
        pass

    def start(self):
        try:
            return super().start()
        finally:
            self.cleanup()

    def initialize(
        self,
        width=1920,
        height=1080,
        hide_ui=False,
        up_axis=None,
        graphics_backend_type=None,
        scene_backend_type=None,
        headless=False,
    ):
        self.initial_width = width
        self.initial_height = height
        self.hide_ui = hide_ui
        self.headless = headless

        self.up_axis = _ke.UpAxis.Y if up_axis is None else up_axis
        if graphics_backend_type is None:
            graphics_backend_type = _ke.BackendType.OpenGL
        if scene_backend_type is None:
            scene_backend_type = scene.BackendType.Native
        self.graphics_backend_type = graphics_backend_type
        self.scene_backend_type = scene_backend_type

        result = super().initialize(
            self.initial_width,
            self.initial_height,
            self.hide_ui,
            self.up_axis,
            self.graphics_backend_type,
            self.scene_backend_type,
            self.headless,
        )
        # Native initialize() recreates the SceneBackend. Keep the resource
        # registry mirrored into the live scene used by ScenePanel/rendering.
        self.resources.bind_scene(self.get_native_scene())
        return result

    def is_key_down(self, key):
        return self.is_key_pressed(key)

    def was_key_pressed(self, key):
        down = self.is_key_pressed(key)
        was_down = self._previous_key_states.get(key, False)
        self._previous_key_states[key] = down
        return down and not was_down

    def was_key_released(self, key):
        down = self.is_key_pressed(key)
        was_down = self._previous_key_states.get(key, False)
        self._previous_key_states[key] = down
        return not down and was_down

    def should_close_shortcut_pressed(self):
        return self.is_key_pressed(keys.ESCAPE)
