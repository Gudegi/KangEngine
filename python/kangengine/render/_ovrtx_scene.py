"""ovstage resources and synchronous RGB output for static scene inputs."""

from contextlib import contextmanager
import math

import numpy as np
import ovrtx
import ovstage
from .._core import _ke


def _tuples(values):
    return ", ".join(str(tuple(float(x) for x in np.asarray(value))) for value in values)


def _texture_key(source):
    sampler = source["sampler"]
    return (source["id"], source["path"], source["srgb"], source["flip"],
            int(sampler.wrap_u), int(sampler.wrap_v))


class OvrtxSceneAdapter:
    def __init__(self, width, height):
        self._resolution = (width, height)
        self._renderer = ovrtx.Renderer(config=ovrtx.RendererConfig(keep_system_alive=False))
        self._stage = None
        self._attached = False
        self._references = {}
        self._geometry = {}
        self._values = {}
        self._ordinal = 1
        try:
            self._stage = ovstage.Stage("kangengine.scene-renderer")
            self._renderer.attach_ovstage(self._stage)
            self._attached = True
            # Scene cameras have no physical exposure controls; use unit exposure.
            bootstrap = f'''#usda 1.0
(
    metersPerUnit = 1
    upAxis = "Y"
)
def Xform "World" {{
    def Camera "Camera" {{
        matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        float verticalAperture = 24
        float horizontalAperture = {24 * width / height}
        float focalLength = 24
        float2 clippingRange = (0.1, 100)
        float exposure = 0
        float exposure:fStop = 1
        float exposure:iso = 100
        float exposure:responsivity = 1
        float exposure:time = 1
    }}
    def DistantLight "Light" {{
        matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
        uniform token[] xformOpOrder = ["xformOp:transform"]
        float inputs:intensity = 1
        color3f inputs:color = (1,1,1)
    }}
}}
def Scope "Render" {{
    def RenderProduct "Camera" {{
        rel camera = </World/Camera>
        int2 resolution = ({width}, {height})
        token omni:rtx:rendermode = "PathTracing"
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished"]
        int omni:rtx:pathtracing:spp = 4
        int omni:rtx:pathtracing:totalSpp = 16
        rel orderedVars = </Render/Camera/LdrColor>
        def RenderVar "LdrColor" {{
            string sourceName = "LdrColor"
        }}
    }}
}}
'''
            ovstage.population.open_usd_from_string(self._stage, bootstrap, ordinal=1)
            self._stage.advance_write_floor(1, ovstage.Scope.ALL).wait()
        except BaseException:
            self.close()
            raise

    @contextmanager
    def _query(self, path):
        with ovstage.PathDictionary(self._stage) as paths:
            path_list = paths.create_path_list_from_strings([path])
            try:
                with self._stage.query_from_path_list(path_list) as query:
                    yield query
            finally:
                paths.destroy_path_list(path_list)

    def _write(self, path, attribute, value, *, dtype=np.float32, semantic=ovstage.AttributeSemantic.NONE):
        values = np.asarray([value], dtype=dtype)
        key = path, attribute
        if key in self._values and np.array_equal(self._values[key], values):
            return
        with self._query(path) as query:
            self._stage.write_attribute(query, attribute, ordinal=self._ordinal,
                                        tensors=values, is_array=False, semantic=semantic).wait()
        self._values[key] = values

    def _matrix(self, path, matrix):
        self._write(path, "omni:xform", matrix.T, dtype=np.float64,
                    semantic=ovstage.AttributeSemantic.MATRIX)

    def _mesh_layer(self, item):
        mesh_id = item["mesh_id"]
        if mesh_id not in self._geometry:
            mesh = item["mesh"]
            self._geometry[mesh_id] = f'''
    point3f[] points = [{_tuples(mesh.vertices)}]
    normal3f[] normals = [{_tuples(mesh.normals)}] (interpolation = "vertex")
    texCoord2f[] primvars:st = [{_tuples(mesh.uvs)}] (interpolation = "vertex")
    int[] faceVertexCounts = {[3] * (len(mesh.indices) // 3)}
    int[] faceVertexIndices = {list(mesh.indices)}
'''
        texture = item["textures"].get("base_color_texture")
        color_input = "color3f inputs:diffuseColor = (1,1,1)"
        texture_shader = ""
        if texture:
            path = texture["path"]
            wraps = {0: "repeat", 1: "clamp", 2: "mirror"}
            sampler = texture["sampler"]
            color_input = "color3f inputs:diffuseColor.connect = </Mesh/Material/Texture.outputs:rgb>"
            texture_shader = f'''
        def Shader "UV" {{
            uniform token info:id = "UsdPrimvarReader_float2"
            token inputs:varname = "st"
            float2 outputs:result
        }}
        def Shader "Texture" {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @{path}@
            token inputs:sourceColorSpace = "{'sRGB' if texture['srgb'] else 'raw'}"
            token inputs:wrapS = "{wraps[int(sampler.wrap_u)]}"
            token inputs:wrapT = "{wraps[int(sampler.wrap_v)]}"
            float2 inputs:st.connect = </Mesh/Material/UV.outputs:result>
            float4 inputs:scale = (1,1,1,1)
            float3 outputs:rgb
        }}
'''
        return f'''#usda 1.0
(defaultPrim = "Mesh")
def Mesh "Mesh" (prepend apiSchemas = ["MaterialBindingAPI"]) {{
{self._geometry[mesh_id]}
    uniform token subdivisionScheme = "none"
    uniform bool doubleSided = {'true' if item['double_sided'] else 'false'}
    token visibility = "inherited"
    matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
    uniform token[] xformOpOrder = ["xformOp:transform"]
    rel material:binding = </Mesh/Material>
    def Material "Material" {{
        token outputs:surface.connect = </Mesh/Material/Shader.outputs:surface>
        def Shader "Shader" {{
            uniform token info:id = "UsdPreviewSurface"
            {color_input}
            float inputs:metallic = 0
            float inputs:roughness = 0.5
            color3f inputs:emissiveColor = (0,0,0)
            token outputs:surface
        }}
{texture_shader}
    }}
}}
'''

    def render(self, frame):
        # Validate all asset paths before changing stage registrations.
        for item in frame["objects"]:
            for texture in item["textures"].values():
                if texture["pixels"] is not None:
                    raise ValueError("ovrtx does not yet support memory-image textures")
                if not texture["flip"]:
                    raise ValueError("ovrtx currently requires flip=True texture sources")
                sampler = texture["sampler"]
                if (sampler.min_filter != _ke.TextureFilter.LINEAR_MIPMAP_LINEAR or
                        sampler.mag_filter != _ke.TextureFilter.LINEAR):
                    raise ValueError("ovrtx currently requires linear mipmapped filtering")
                if "@" in texture["path"] or "\n" in texture["path"]:
                    raise ValueError("ovrtx texture asset paths cannot contain @ or a newline")

        self._ordinal += 1
        objects = {item["id"]: item for item in frame["objects"]}
        signatures = {key: (item["mesh_id"], item["double_sided"],
                           tuple((slot, _texture_key(source)) for slot, source in item["textures"].items()))
                      for key, item in objects.items()}
        surviving = {key for key, (_, signature) in self._references.items()
                     if key in signatures and signature == signatures[key]}
        changed = set(self._references) != surviving or set(objects) != surviving
        for key in self._references.keys() - surviving:
            handle, _ = self._references.pop(key)
            ovstage.population.remove_usd(self._stage, handle)
            prefix = f"/World/i{key}"
            self._values = {k: v for k, v in self._values.items()
                            if k[0] != prefix and not k[0].startswith(prefix + "/")}
        for key in objects.keys() - surviving:
            handle = ovstage.population.add_usd_reference_from_string(
                self._stage, self._mesh_layer(objects[key]), f"/World/i{key}")
            self._references[key] = handle, signatures[key]
        if changed:
            ovstage.population.apply_usd_changes(self._stage, ordinal=self._ordinal)

        self._ordinal += 1
        for key, item in objects.items():
            path = f"/World/i{key}"
            self._matrix(path, item["world"])
            with ovstage.PathDictionary(self._stage) as paths:
                token = paths.intern_token("inherited" if item["visible"] else "invisible")
                self._write(path, "visibility", token.raw_id, dtype=np.uint64,
                            semantic=ovstage.AttributeSemantic.TOKEN_ID)
            value = item["material"]
            color = np.asarray(value["base_color"]) * item["color"]
            if item["textures"]:
                self._write(path + "/Material/Texture", "inputs:scale", color)
            else:
                self._write(path + "/Material/Shader", "inputs:diffuseColor", color[:3],
                            semantic=ovstage.AttributeSemantic.COLOR)
            for attribute, data in (("metallic", value["metallic"]),
                                    ("roughness", value["roughness"]),
                                    ("emissiveColor", value["emissive"])):
                self._write(path + "/Material/Shader", "inputs:" + attribute, data,
                            semantic=(ovstage.AttributeSemantic.COLOR if attribute == "emissiveColor"
                                      else ovstage.AttributeSemantic.NONE))

        camera = frame["camera"]
        self._write("/Render/Camera", "resolution", self._resolution, dtype=np.int32)
        self._write("/World/Camera", "horizontalAperture", 24 * self._resolution[0] / self._resolution[1])
        self._matrix("/World/Camera", camera["world"])
        self._write("/World/Camera", "focalLength", 12 / math.tan(math.radians(camera["vertical_fov"]) / 2))
        self._write("/World/Camera", "clippingRange", [camera["near"], camera["far"]])
        light = frame["lights"][0] if frame["lights"] else None
        self._write("/World/Light", "inputs:intensity", light["intensity"] if light else 0)
        if light:
            # Rendering population stores light color as VECTOR, not COLOR.
            self._write("/World/Light", "inputs:color", light["color"],
                        semantic=ovstage.AttributeSemantic.VECTOR)
            z = -np.asarray(light["direction"])
            up = np.array([0, 1, 0]) if abs(z[1]) < 0.99 else np.array([1, 0, 0])
            x = np.cross(up, z)
            x /= np.linalg.norm(x)
            world = np.eye(4)
            world[:3, :3] = np.column_stack((x, np.cross(z, x), z))
            self._matrix("/World/Light", world)
        self._stage.compute_hierarchy(self._ordinal, self._ordinal)
        self._stage.advance_write_floor(self._ordinal, ovstage.Scope.ALL).wait()

        # The pinned SDK needs a rebuild when no previous mesh survives.
        if self._ordinal > 3 and objects and not surviving:
            self._attached = False
            self._renderer.detach_ovstage()
            self._renderer.attach_ovstage(self._stage)
            self._attached = True
        products = self._renderer.step(render_products={"/Render/Camera"},
                                       delta_time=1 / 60, ordinal=self._ordinal)
        frame = products["/Render/Camera"].frames[-1]
        mapped = frame.render_vars["/Render/Camera/LdrColor"].map(device=ovrtx.Device.CPU)
        try:
            view = np.from_dlpack(mapped)
            pixels = view[..., :3].copy()
            del view
        finally:
            mapped.unmap()
        return pixels

    def resize(self, width, height):
        self._resolution = (width, height)

    def close(self):
        try:
            if self._attached:
                self._attached = False
                self._renderer.detach_ovstage()
        finally:
            if self._stage is not None:
                self._stage.destroy()
            self._renderer.destroy()


class OvrtxAppAdapter(OvrtxSceneAdapter):
    """Validate App snapshots before passing them to ovrtx."""

    def __init__(self, width, height):
        super().__init__(width, height)
        self._texture_paths = {}
        self._validated_meshes = set()

    def render(self, frame):
        from .scene_renderer import _validate_frame

        return super().render(_validate_frame(frame, self._texture_paths, self._validated_meshes))
