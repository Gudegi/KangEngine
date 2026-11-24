"""ovstage scene updates and CPU/CUDA output images."""

from contextlib import ExitStack, contextmanager
from itertools import chain
import math
import os
from pathlib import Path
import sys
import warnings

if sys.platform == "darwin":
    raise RuntimeError(
        "ovrtx is not supported on macOS; use the native render backend."
    )

# Local scene assets do not require the Omniverse Hub cache service.
os.environ.setdefault("OMNICLIENT_HUB_MODE", "disabled")

import numpy as np
import ovrtx
import ovstage
from .._core import _ke


def _tuples(values):
    return ", ".join(
        str(tuple(float(x) for x in np.asarray(value))) for value in values
    )


def _texture_key(source):
    sampler = source["sampler"]
    return (
        source["id"],
        source["path"],
        source["srgb"],
        source["flip"],
        int(sampler.wrap_u),
        int(sampler.wrap_v),
    )


def _material_key(item):
    return (item["material"]["model"], item["material"]["style"],
            item["material"]["unlit"], item["alpha_mode"],
            tuple((slot, _texture_key(source)) for slot, source in item["textures"].items()))


def _base_color_texture(item):
    material = item["material"]
    # Untextured ColorMaterial ignores this texture unless unlit.
    if material["model"] == "color" and material["style"] != 1 and not material["unlit"]:
        return None
    return item["textures"].get("base_color_texture", item["textures"].get("diffuse_map"))


class OvrtxSceneAdapter:
    def __init__(self, width, height):
        self._resolution = (width, height)
        ovrtx.library_telemetry_opt_out()
        self._renderer = ovrtx.Renderer(
            config=ovrtx.RendererConfig(keep_system_alive=False)
        )
        self._renderer.set_selection_group_styles({
            1: ovrtx.SelectionGroupStyle(outline_color=(1, 0.6, 0, 1), fill_color=(0, 0, 0, 0)),
        })
        self._stage = None
        self._attached = False
        self._references = {}
        self._objects = {}
        self._batches = {}
        self._batch_revisions = {}
        self._skin_layers = {}
        self._skin_revisions = {}
        self._textures = {}
        self._texture_pixels = {}
        self.pick_result = None
        self._selection_paths = set()
        self._batch_queries = {}
        self._batch_query_stack = ExitStack()
        self._light_values = []
        self._lights = {}
        self._warned_ranges = set()
        self._geometry = {}
        self._deformations = {}
        self._values = {}
        self._ordinal = 1
        self._output = None
        self._products = None
        self._mapped = None
        try:
            self._stage = ovstage.Stage("kangengine.scene-renderer")
            self._renderer.attach_ovstage(self._stage)
            self._attached = True
            # Scene cameras have no physical exposure controls; use unit exposure.
            bootstrap = f"""#usda 1.0
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

}}
def Scope "Render" {{
    def RenderProduct "Camera" {{
        rel camera = </World/Camera>
        int2 resolution = ({width}, {height})
        uint[] deviceIds = [0]
        token omni:rtx:rendermode = "RealTimePathTracing" # "PathTracing" # "Minimal"
        rel orderedVars = </Render/Camera/LdrColor>
        def RenderVar "LdrColor" {{
            string sourceName = "LdrColor"
        }}
    }}
}}
"""
            ovstage.population.open_usd_from_string(self._stage, bootstrap, ordinal=1)
            self._stage.advance_write_floor(1, ovstage.Scope.ALL).wait()
        except BaseException:
            self.close()
            raise

    @contextmanager
    def _query(self, path):
        with ovstage.PathDictionary(self._stage) as paths:
            path_list = paths.create_path_list_from_strings(
                [path] if isinstance(path, str) else path
            )
            try:
                with self._stage.query_from_path_list(path_list) as query:
                    yield query
            finally:
                paths.destroy_path_list(path_list)

    def _write(
        self,
        path,
        attribute,
        value,
        *,
        dtype=np.float32,
        semantic=ovstage.AttributeSemantic.NONE,
    ):
        values = np.asarray([value], dtype=dtype)
        key = path, attribute
        if key in self._values and np.array_equal(self._values[key], values):
            return
        with self._query(path) as query:
            self._stage.write_attribute(
                query,
                attribute,
                ordinal=self._ordinal,
                tensors=values,
                is_array=False,
                semantic=semantic,
            ).wait()
        self._values[key] = values

    def _matrix(self, path, matrix):
        self._write(
            path,
            "omni:xform",
            matrix.T,
            dtype=np.float64,
            semantic=ovstage.AttributeSemantic.MATRIX,
        )

    def _write_geometry(self, path, geometry):
        if self._deformations.get(path) is geometry:
            return
        with self._query(path) as query:
            for attribute, values, semantic in (
                ("points", geometry.positions, ovstage.AttributeSemantic.POINT),
                ("normals", geometry.normals, ovstage.AttributeSemantic.NORMAL),
            ):
                if not len(values):
                    continue
                # Array attributes require explicit three-lane element types.
                tensor = ovstage.make_dltensor(
                    values, shape=[len(values)],
                    dtype=ovstage.DLDataType(ovstage.DLDataTypeCode.kDLFloat, 32, 3),
                )
                self._stage.write_attribute(
                    query, attribute, ordinal=self._ordinal,
                    tensors=tensor, is_array=True, semantic=semantic,
                ).wait()
        self._deformations[path] = geometry

    def _mesh_layer(self, item, *, instance_count=None):
        mesh_id = item["mesh_id"]
        if mesh_id not in self._geometry:
            mesh = item["mesh"]
            self._geometry[mesh_id] = f"""
    point3f[] points = [{_tuples(mesh.vertices)}]
    normal3f[] normals = [{_tuples(mesh.normals)}] (interpolation = "vertex")
    texCoord2f[] primvars:st = [{_tuples(mesh.uvs)}] (interpolation = "vertex")
    int[] faceVertexCounts = {[3] * (len(mesh.indices) // 3)}
    int[] faceVertexIndices = {list(mesh.indices)}
"""
        skin = item.get("skin")
        root = "/Mesh" if instance_count is None and skin is None else "/Batch"
        material_layer = self._material_layer(item, root)
        mesh_attributes = f"""
{self._geometry[mesh_id]}
    uniform token subdivisionScheme = "none"
    uniform bool doubleSided = {"true" if item["double_sided"] else "false"}
    rel material:binding = <{root}/Material>
"""
        transform_attributes = """
    matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
    uniform token[] xformOpOrder = ["xformOp:transform"]
"""
        if skin is not None:
            return self._skin_layer(skin, mesh_attributes, transform_attributes, material_layer)
        if instance_count is None:
            return f'''#usda 1.0
(defaultPrim = "Mesh")
def Mesh "Mesh" (prepend apiSchemas = ["MaterialBindingAPI"]) {{
{mesh_attributes}
{transform_attributes}
    token visibility = "inherited"
{material_layer}
}}
'''

        # USD instances share a prototype; only root transforms vary.
        instances = "\n".join(f'''
    def Xform "i{idx}" (
        prepend references = </Batch/Prototype>
        instanceable = true
    ) {{
{transform_attributes}
    }}
''' for idx in range(instance_count))
        return f'''#usda 1.0
(defaultPrim = "Batch")
def Xform "Batch" {{
    token visibility = "inherited"
{material_layer}
    class Xform "Prototype" {{
        def Mesh "Mesh" (prepend apiSchemas = ["MaterialBindingAPI"]) {{
{mesh_attributes}
        }}
    }}
{instances}
}}
'''

    def _material_layer(self, item, root):
        textures = item["textures"]
        color = _base_color_texture(item)
        normal = textures.get("normal_texture", textures.get("normal_map"))
        alpha = textures.get("alpha_map")
        shaders = ''
        if textures:
            shaders = '''
        def Shader "UV" {
            uniform token info:id = "UsdPrimvarReader_float2"
            token inputs:varname = "st"
            float2 outputs:result
        }
'''
        for name, source in (("Texture", color), ("Normal", normal), ("Alpha", alpha)):
            if source is None:
                continue
            path = source["path"] if source["pixels"] is None else f'dynamic://kangengine/t{source["id"]}'
            wraps = {0: "repeat", 1: "clamp", 2: "mirror"}
            sampler = source["sampler"]
            shaders += f'''
        def Shader "{name}" {{
            uniform token info:id = "UsdUVTexture"
            asset inputs:file = @{path}@
            token inputs:sourceColorSpace = "{"sRGB" if source["srgb"] else "raw"}"
            token inputs:wrapS = "{wraps[int(sampler.wrap_u)]}"
            token inputs:wrapT = "{wraps[int(sampler.wrap_v)]}"
            float2 inputs:st.connect = <{root}/Material/UV.outputs:result>
            float4 inputs:scale = {"(2,2,2,1)" if name == "Normal" else "(1,1,1,1)"}
            float4 inputs:bias = {"(-1,-1,-1,0)" if name == "Normal" else "(0,0,0,0)"}
            float3 outputs:rgb
            float outputs:r
            float outputs:a
        }}
'''
        unlit = item["material"]["unlit"]
        channel = "emissiveColor" if unlit else "diffuseColor"
        connections = ""
        if color:
            connections += f"color3f inputs:{channel}.connect = <{root}/Material/Texture.outputs:rgb>\n"
        if normal and not unlit:
            connections += f"normal3f inputs:normal.connect = <{root}/Material/Normal.outputs:rgb>\n"
        if item["alpha_mode"] and (alpha or color):
            source = "Alpha.outputs:r" if alpha else "Texture.outputs:a"
            connections += f"float inputs:opacity.connect = <{root}/Material/{source}>\n"
        return f'''
    def Material "Material" {{
        token outputs:surface.connect = <{root}/Material/Shader.outputs:surface>
        def Shader "Shader" {{
            uniform token info:id = "UsdPreviewSurface"
            color3f inputs:diffuseColor = (0,0,0)
            color3f inputs:emissiveColor = (0,0,0)
            float inputs:metallic = 0
            float inputs:roughness = 0.5
            int inputs:useSpecularWorkflow = 0
            color3f inputs:specularColor = (0,0,0)
            float inputs:opacity = 1
            float inputs:opacityThreshold = 0
            {connections}
            token outputs:surface
        }}
{shaders}
    }}
'''

    def _sync_textures(self, items):
        sources = {source["id"]: source for item in items
                   for source in item["textures"].values() if source["pixels"] is not None}
        removed = self._textures.keys() - sources.keys()
        added = sources.keys() - self._textures.keys()
        for key in removed:
            ovstage.population.remove_usd(self._stage, self._textures.pop(key))
            self._texture_pixels.pop(key, None)
        for key in added:
            source = sources[key]
            pixels = source["pixels"]
            height, width, channels = pixels.shape
            # Runtime textures use RGBA; scene resources retain the original pixels.
            rgba = np.full((height, width, 4), 255, dtype=np.uint8)
            rgba[..., :3] = pixels[..., :3] if channels >= 3 else pixels[..., :1]
            if channels in (2, 4):
                rgba[..., 3] = pixels[..., -1]
            # Runtime-image rows follow file-image order in OVRTX.
            self._texture_pixels[key] = np.ascontiguousarray(rgba if source["flip"] else rgba[::-1])
            layer = f'''#usda 1.0
(defaultPrim = "Texture")
def OmniRuntimeTexture "Texture" {{
    string texture:uri = "dynamic://kangengine/t{key}"
    token texture:dimension = "2d"
    int3 texture:extent = ({width}, {height}, 1)
    token texture:format = "rgba8unorm"
    token texture:colorSpace = "{"sRGB" if source["srgb"] else "linear"}"
    uchar[] texture:data = []
}}
'''
            self._textures[key] = ovstage.population.add_usd_reference_from_string(
                self._stage, layer, f"/Textures/t{key}",
            )
        return bool(removed or added)

    def _skin_layer(self, skin, mesh_attributes, transform_attributes, material_layer):
        if skin not in self._skin_layers:
            ids = np.asarray(skin.indices)
            weights = np.where(ids >= 0, skin.weights, 0).copy()
            ids = np.where(ids >= 0, ids, skin.bone_count).copy()
            unweighted = weights.sum(axis=1) == 0
            ids[unweighted, 0] = skin.bone_count
            weights[unweighted, 0] = 1
            self._skin_layers[skin] = f'''
        int[] primvars:skel:jointIndices = {ids.ravel().tolist()} (elementSize = 4
            interpolation = "vertex")
        float[] primvars:skel:jointWeights = {weights.ravel().tolist()} (elementSize = 4
            interpolation = "vertex")
'''
        # Independent root joints consume final skin matrices, with identity binds.
        identity = "((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))"
        count = skin.bone_count + 1
        joints = ", ".join(f'"j{idx}"' for idx in range(count))
        matrices = ", ".join([identity] * count)
        return f'''#usda 1.0
(defaultPrim = "Batch")
def SkelRoot "Batch" {{
{transform_attributes}
    token visibility = "inherited"
{material_layer}
    def Skeleton "Skeleton" {{
        uniform token[] joints = [{joints}]
        uniform matrix4d[] bindTransforms = [{matrices}]
        uniform matrix4d[] restTransforms = [{matrices}]
    }}
    def Mesh "Mesh" (prepend apiSchemas = ["MaterialBindingAPI", "SkelBindingAPI"]) {{
{mesh_attributes}
        rel skel:skeleton = </Batch/Skeleton>
        matrix4d primvars:skel:geomBindTransform = {identity}
{self._skin_layers[skin]}
    }}
}}
'''

    def _write_skin(self, key, item):
        path = item["path"]
        if item["geometry"] is not None:
            self._write_geometry(path + "/Mesh", item["geometry"])
        if self._skin_revisions.get(key) == item["skin_revision"]:
            return
        matrices = np.empty((len(item["joints"]) + 1, 4, 4), dtype=np.float64)
        matrices[:-1] = item["joints"].swapaxes(1, 2)
        matrices[-1] = np.eye(4)
        tensor = ovstage.make_dltensor(
            matrices, shape=[len(matrices)],
            dtype=ovstage.DLDataType(ovstage.DLDataTypeCode.kDLFloat, 64, 16),
        )
        with self._query(path + "/Skeleton") as query:
            self._stage.write_attribute(
                query, "restTransforms", ordinal=self._ordinal, tensors=tensor,
                is_array=True, semantic=ovstage.AttributeSemantic.MATRIX,
            ).wait()
        self._skin_revisions[key] = item["skin_revision"]

    def _sync_batches(self, batches):
        signatures = {
            key: (item["mesh_id"], item["double_sided"], item["count"], item["skin"],
                  _material_key(item))
            for key, item in batches.items()
        }
        replacements = {
            key for key, signature in signatures.items()
            if key not in self._batches or self._batches[key][1] != signature
        }
        removed = self._batches.keys() - batches.keys()
        surviving = len(self._batches.keys() - removed - replacements)
        for key in removed | replacements:
            if key in self._batches:
                handle, _ = self._batches.pop(key)
                ovstage.population.remove_usd(self._stage, handle)
        for key in replacements:
            item = batches[key]
            layer = self._mesh_layer(item, instance_count=item["count"])
            handle = ovstage.population.add_usd_reference_from_string(
                self._stage, layer, item["path"],
            )
            self._batches[key] = handle, signatures[key]
        return bool(removed or replacements), surviving

    def _write_batch_transforms(self, key, item):
        external = item["external"]
        versioned = external is None or external.sync_policy == _ke.ExternalSyncPolicy.VERSIONED
        if versioned and self._batch_revisions.get(key) == item["revision"]:
            return
        if key not in self._batch_queries:
            paths = [f'{item["path"]}/i{idx}' for idx in range(item["count"])]
            self._batch_queries[key] = self._batch_query_stack.enter_context(self._query(paths))

        if external is not None:
            self._write_cuda_transforms(self._batch_queries[key], external)
            self._batch_revisions[key] = item["revision"]
            return

        # Preserve complete matrices, including shear; USD uses row vectors.
        matrices = np.ascontiguousarray(item["transforms"].swapaxes(1, 2), dtype=np.float64)
        self._stage.write_attribute(
            self._batch_queries[key], "omni:xform", ordinal=self._ordinal,
            tensors=matrices, is_array=False, semantic=ovstage.AttributeSemantic.MATRIX,
        ).wait()
        self._batch_revisions[key] = item["revision"]

    def _write_cuda_transforms(self, query, external):
        import torch
        from cuda.bindings import runtime

        view = external.view
        with torch.cuda.device(view.device_id):
            stream = torch.cuda.current_stream()
            if view.ready_event_handle:
                status, = runtime.cudaStreamWaitEvent(stream.cuda_stream, view.ready_event_handle, 0)
                if status != runtime.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"CUDA transform readiness failed: {status}")
            else:
                stream.wait_stream(torch.cuda.ExternalStream(view.stream_handle, device=view.device_id))
            # Column-major GLM bytes already match USD row-vector matrices.
            # Conversion stays on the GPU; ovstage owns a copy after wait().
            matrices = view.torch()[:external.count].to(dtype=torch.float64)
            self._stage.write_attribute(
                query, "omni:xform", ordinal=self._ordinal,
                tensors=ovstage.make_dltensor(matrices), is_array=False,
                semantic=ovstage.AttributeSemantic.MATRIX,
                cuda_stream=stream.cuda_stream,
                placement=ovstage.Placement.from_dlpack_device(matrices),
            ).wait()

    def _write_material(self, path, item):
        value = item["material"]
        color = np.asarray(value["base_color"]) * item["color"]
        textures = item["textures"]
        textured = _base_color_texture(item) is not None
        shader = path + "/Material/Shader"
        unlit = value["unlit"]
        if textured:
            self._write(path + "/Material/Texture", "inputs:scale", color)
        else:
            self._write(shader, "inputs:emissiveColor" if unlit else "inputs:diffuseColor", color[:3],
                        semantic=ovstage.AttributeSemantic.COLOR)
        phong = value["model"] == "phong"
        roughness = math.sqrt(2 / (max(0, value["shininess"]) + 2)) if phong else value["roughness"]
        for attribute, data in (
            ("metallic", 0 if unlit else value["metallic"]),
            ("roughness", roughness),
            ("specularColor", [0, 0, 0] if unlit else value["specular"]),
        ):
            self._write(
                shader, "inputs:" + attribute, data,
                semantic=(ovstage.AttributeSemantic.COLOR if attribute == "specularColor"
                          else ovstage.AttributeSemantic.NONE),
            )
        self._write(shader, "inputs:useSpecularWorkflow", int(phong or unlit), dtype=np.int32)
        if not unlit:
            self._write(shader, "inputs:emissiveColor", value["emissive"], semantic=ovstage.AttributeSemantic.COLOR)
        if not item["alpha_mode"] or not (textured or "alpha_map" in textures):
            self._write(shader, "inputs:opacity", color[3] if item["alpha_mode"] else 1)
        if "alpha_map" in textures:
            self._write(path + "/Material/Alpha", "inputs:scale", [color[3]] * 4)
        self._write(shader, "inputs:opacityThreshold", item["alpha_cutoff"] if item["alpha_mode"] == 1 else 0)

    def _write_visibility(self, path, visible):
        with ovstage.PathDictionary(self._stage) as paths:
            token = paths.intern_token("inherited" if visible else "invisible")
            self._write(path, "visibility", token.raw_id, dtype=np.uint64,
                        semantic=ovstage.AttributeSemantic.TOKEN_ID)

    def _prepare_interaction(self, frame, objects, batches, structural_change):
        selection = frame.get("selection")
        selected = set()
        if selection and selection["hit"]:
            batch = batches.get(selection["handle"])
            if batch is not None:
                if batch["skin"] is not None:
                    selected.add(batch["path"] + "/Mesh")
                elif 0 <= selection["instance"] < batch["count"]:
                    selected.add(f'{batch["path"]}/i{selection["instance"]}/Mesh')
            elif selection["id"] in objects:
                selected.add(f'/World/i{selection["id"]}')
        if selected != self._selection_paths or structural_change:
            if self._selection_paths:
                self._renderer.set_selection_outline_group_strings(sorted(self._selection_paths), 0)
            if selected:
                self._renderer.set_selection_outline_group_strings(sorted(selected), 1)
            self._selection_paths = selected

        self.pick_result = None
        pick = frame.get("pick")
        if pick is None:
            return
        paths, allowed = [], []
        for key, item in objects.items():
            paths.append(f"/World/i{key}")
            allowed.append(item["pickable"] and item["visible"])
        for item in batches.values():
            mesh_paths = ([item["path"] + "/Mesh"] if item["skin"] is not None else
                          [f'{item["path"]}/i{idx}/Mesh' for idx in range(item["count"])])
            paths.extend(mesh_paths)
            allowed.extend([item["pickable"] and item["visible"]] * len(mesh_paths))
        if paths:
            self._renderer.set_pickable_strings(paths, allowed)
        width, height = self._resolution
        x = min(width - 1, max(0, int((pick["ndc"][0] + 1) * 0.5 * width)))
        y = min(height - 1, max(0, int((1 - pick["ndc"][1]) * 0.5 * height)))
        self._renderer.enqueue_pick_query(
            "/Render/Camera", x / width, y / height, (x + 1) / width, (y + 1) / height,
            flags=ovrtx.OVRTX_PICK_FLAG_INCLUDE_TRACKED_INFO,
        )

    def _read_pick(self, product, objects, batches):
        output = product.render_vars.get(ovrtx.OVRTX_RENDER_VAR_PICK_HIT)
        if output is None:
            raise RuntimeError("ovrtx returned no result for the queued pick")
        with output.map(device=ovrtx.Device.CPU) as mapped:
            magic = int(np.from_dlpack(mapped.params["magic"]))
            version = int(np.from_dlpack(mapped.params["version"]))
            if magic != ovrtx.OVRTX_PICK_HIT_MAGIC or version != ovrtx.OVRTX_PICK_HIT_VERSION:
                raise RuntimeError("Unsupported ovrtx pick result schema")
            if not int(np.from_dlpack(mapped.params["hitCount"])):
                return
            path_id = int(np.from_dlpack(mapped["primPath"]).reshape(-1)[0])
            path = self._renderer.resolve_prim_path_id(ovrtx.PrimPath(path_id))
            position = np.from_dlpack(mapped["worldPositionM"]).reshape(-1, 3)[0].tolist()
        parts = path.strip("/").split("/")
        if len(parts) >= 2 and parts[0] == "World" and parts[1].startswith("i"):
            key = int(parts[1][1:])
            item = objects.get(key)
            if item is not None and item["pickable"] and item["visible"]:
                self.pick_result = dict(handle=0xFFFFFFFF, id=key, instance=0,
                                        external=False, position=position)
        elif len(parts) >= 3 and parts[0] == "Batches" and parts[1].startswith("b"):
            key = int(parts[1][1:])
            item = batches.get(key)
            if item is None or not item["pickable"] or not item["visible"]:
                return
            skinned = item["skin"] is not None
            instance = 0 if skinned else int(parts[2][1:])
            self.pick_result = dict(handle=key, id=item["instance_ids"][0] if skinned else 0,
                                    instance=instance, external=not skinned, position=position)

    def _light_layer(self, item):
        kind, light = item["type"], item["light"]
        texture = light.texture_path if kind in ("rect", "dome") else ""
        if texture:
            texture = str(Path(texture).resolve(strict=True))
            if "@" in texture or "\n" in texture:
                raise ValueError("Light texture paths cannot contain @ or a newline")
        attributes = ""
        schema = {"directional": "DistantLight", "point": "SphereLight",
                  "spot": "SphereLight", "rect": "RectLight", "dome": "DomeLight"}[kind]
        shaping = ' (prepend apiSchemas = ["ShapingAPI"])' if kind == "spot" else ""
        if kind in ("point", "spot"):
            attributes += "    bool treatAsPoint = true\n    float inputs:radius = 0.5\n"
        if kind == "spot":
            attributes += "    float inputs:shaping:cone:angle = 30\n    float inputs:shaping:cone:softness = 0\n"
        if kind in ("rect", "dome"):
            attributes += f"    asset inputs:texture:file = @{texture}@\n"
        if kind == "rect":
            attributes += "    float inputs:width = 1\n    float inputs:height = 1\n"
        if kind == "dome":
            if light.texture_format not in (
                "automatic", "latlong", "mirroredBall", "angular", "cubeMapVerticalCross",
            ):
                raise ValueError(f"Unsupported dome texture format: {light.texture_format}")
            attributes += f'    token inputs:texture:format = "{light.texture_format}"\n'
        return f'''#usda 1.0
(defaultPrim = "Light")
def {schema} "Light"{shaping} {{
    matrix4d xformOp:transform = ((1,0,0,0),(0,1,0,0),(0,0,1,0),(0,0,0,1))
    uniform token[] xformOpOrder = ["xformOp:transform"]
    color3f inputs:color = (1,1,1)
    float inputs:intensity = 1
    float inputs:exposure = 0
    bool inputs:normalize = false
    bool inputs:enableColorTemperature = false
    float inputs:colorTemperature = 6500
    float inputs:diffuse = 1
    float inputs:specular = 1
{attributes}
}}
'''

    def _sync_lights(self, lights):
        sources = {item["path"]: item for item in lights}
        signatures = {
            path: (item["type"],
                   item["light"].texture_path if item["type"] in ("rect", "dome") else "",
                   item["light"].texture_format if item["type"] == "dome" else "")
            for path, item in sources.items()
        }
        # Validate all replacements before removing accepted light resources.
        layers = {}
        for path, item in sources.items():
            kind, light = item["type"], item["light"]
            if kind == "directional" and any((light.ambient.x, light.ambient.y, light.ambient.z)):
                raise ValueError("ovrtx requires directional ambient=[0, 0, 0]; use a DomeLight")
            if kind in ("point", "spot") and kind not in self._warned_ranges:
                warnings.warn(
                    f"ovrtx {kind} lights use inverse-square falloff; Native range cutoff is not supported",
                    stacklevel=3,
                )
                self._warned_ranges.add(kind)
            if path not in self._lights or self._lights[path][1] != signatures[path]:
                layers[path] = self._light_layer(item)
        removed = self._lights.keys() - sources.keys()
        changed = bool(removed or layers)
        for path in removed | layers.keys():
            if path in self._lights:
                handle, _ = self._lights.pop(path)
                ovstage.population.remove_usd(self._stage, handle)
        for path, layer in layers.items():
            handle = ovstage.population.add_usd_reference_from_string(
                self._stage, layer, "/Lights" + path,
            )
            self._lights[path] = handle, signatures[path]
        return changed

    def _write_light(self, item):
        path = "/Lights" + item["path"]
        kind, light = item["type"], item["light"]
        world = item["world"]
        if kind in ("directional", "spot"):
            direction = light.direction
            z = -(world[:3, :3] @ np.array([direction.x, direction.y, direction.z]))
            length = np.linalg.norm(z)
            z = z / length if length > 1e-4 else np.array([0.0, 0.0, 1.0])
            up = np.array([0, 1, 0]) if abs(z[1]) < 0.99 else np.array([1, 0, 0])
            x = np.cross(up, z)
            x /= np.linalg.norm(x)
            oriented = np.eye(4)
            oriented[:3, :3] = np.column_stack((x, np.cross(z, x), z))
            oriented[:3, 3] = world[:3, 3]
            world = oriented
        self._matrix(path, world)
        color = light.color
        self._write(path, "inputs:color", [color.x, color.y, color.z],
                    semantic=ovstage.AttributeSemantic.VECTOR)
        self._write(path, "inputs:intensity", light.intensity)
        if kind in ("point", "spot"):
            self._write(path, "inputs:normalize", True, dtype=np.bool_)
        if kind in ("rect", "dome"):
            for attribute, value in (
                ("exposure", light.exposure), ("colorTemperature", light.color_temperature),
                ("diffuse", light.diffuse), ("specular", light.specular),
            ):
                self._write(path, "inputs:" + attribute, value)
            self._write(path, "inputs:normalize", light.normalize, dtype=np.bool_)
            self._write(path, "inputs:enableColorTemperature",
                        light.enable_color_temperature, dtype=np.bool_)
        if kind == "rect":
            self._write(path, "inputs:width", light.width)
            self._write(path, "inputs:height", light.height)
        if kind == "spot":
            # UsdLux stores half-angles in degrees and the edge width as a fraction.
            outer = light.outer_cone_angle
            softness = 1 - light.inner_cone_angle / outer if outer > 0 else 0
            self._write(path, "inputs:shaping:cone:angle", math.degrees(outer))
            self._write(path, "inputs:shaping:cone:softness", softness)

    def _render_output(self, frame):
        self._release_output()
        # Validate all asset paths before changing stage registrations.
        for item in chain(frame["objects"], frame.get("batches", ())):
            for texture in item["textures"].values():
                if texture["pixels"] is None and not texture["flip"]:
                    raise ValueError(
                        "ovrtx currently requires flip=True texture sources"
                    )
                sampler = texture["sampler"]
                if (
                    sampler.min_filter != _ke.TextureFilter.LINEAR_MIPMAP_LINEAR
                    or sampler.mag_filter != _ke.TextureFilter.LINEAR
                ):
                    raise ValueError(
                        "ovrtx currently requires linear mipmapped filtering"
                    )
                if "@" in texture["path"] or "\n" in texture["path"]:
                    raise ValueError(
                        "ovrtx texture asset paths cannot contain @ or a newline"
                    )

        self._ordinal += 1
        batches = {item["id"]: item for item in frame.get("batches", ())
                   if item["skin"] is not None or item["count"]}
        lights = frame["lights"] if frame["lights"] is not None else self._light_values
        lights_changed = self._sync_lights(lights) if frame["lights"] is not None else False
        updates = {item["id"]: item for item in frame["objects"]}
        if frame["incremental"]:
            objects = self._objects.copy() if updates or frame["removed_objects"] else self._objects
            removed = self._references.keys() & set(frame["removed_objects"])
        else:
            objects = {}
            removed = self._references.keys() - updates.keys()
        for key in frame["removed_objects"]:
            objects.pop(key, None)
        objects.update(updates)
        batches_changed, surviving_batches = self._sync_batches(batches)
        textures_changed = False
        if updates or frame["removed_objects"] or batches_changed or not frame["incremental"]:
            textures_changed = self._sync_textures(chain(objects.values(), batches.values()))
        replacements = {}
        for key, item in updates.items():
            signature = (
                item["mesh_id"], item["double_sided"],
                _material_key(item),
            )
            if key not in self._references or self._references[key][1] != signature:
                replacements[key] = signature
        surviving_count = len(self._references) - len(removed | (replacements.keys() & self._references.keys()))
        changed = bool(removed or replacements)
        for key in removed | replacements.keys():
            self._deformations.pop(f"/World/i{key}", None)
            if key in self._references:
                handle, _ = self._references.pop(key)
                ovstage.population.remove_usd(self._stage, handle)
        for key, signature in replacements.items():
            handle = ovstage.population.add_usd_reference_from_string(
                self._stage, self._mesh_layer(objects[key]), f"/World/i{key}",
            )
            self._references[key] = handle, signature
        structural_change = changed or lights_changed or batches_changed or textures_changed
        if structural_change:
            self._batch_query_stack.close()
            self._batch_queries.clear()
            ovstage.population.apply_usd_changes(self._stage, ordinal=self._ordinal)
            # Population can restore USD defaults on surviving objects too.
            self._values.clear()
            self._deformations.clear()
            self._batch_revisions.clear()
            self._skin_revisions.clear()

        self._ordinal += 1
        if structural_change:
            for key, pixels in self._texture_pixels.items():
                with self._query(f"/Textures/t{key}") as query:
                    self._stage.write_attribute(
                        query, "texture:data", ordinal=self._ordinal,
                        tensors=pixels.reshape(-1), is_array=True,
                    ).wait()
        if structural_change or frame["lights"] is not None:
            for item in lights:
                self._write_light(item)
        for key, item in (objects if structural_change else updates).items():
            path = f"/World/i{key}"
            if item["geometry"] is not None:
                self._write_geometry(path, item["geometry"])
            self._matrix(path, item["world"])
            self._write_visibility(path, item["visible"])
            self._write_material(path, item)
        for key, item in batches.items():
            if item["skin"] is None:
                self._write_batch_transforms(key, item)
            else:
                self._write_skin(key, item)
                self._matrix(item["path"], item["world"])
            self._write_visibility(item["path"], item["visible"])
            self._write_material(item["path"], item)

        camera = frame["camera"]
        self._write("/Render/Camera", "resolution", self._resolution, dtype=np.int32)
        self._write(
            "/World/Camera",
            "horizontalAperture",
            24 * self._resolution[0] / self._resolution[1],
        )
        self._matrix("/World/Camera", camera["world"])
        self._write(
            "/World/Camera",
            "focalLength",
            12 / math.tan(math.radians(camera["vertical_fov"]) / 2),
        )
        self._write("/World/Camera", "clippingRange", [camera["near"], camera["far"]])
        self._stage.compute_hierarchy(self._ordinal, self._ordinal)
        self._stage.advance_write_floor(self._ordinal, ovstage.Scope.ALL).wait()

        # The pinned SDK needs a rebuild when no previous mesh survives.
        if (self._ordinal > 3 and (objects or batches)
                and not (surviving_count or surviving_batches)):
            self._attached = False
            self._renderer.detach_ovstage()
            self._renderer.attach_ovstage(self._stage)
            self._attached = True
        self._prepare_interaction(frame, objects, batches, structural_change)
        # The result set owns the native output handles until presentation ends.
        self._products = self._renderer.step(
            render_products={"/Render/Camera"}, delta_time=1 / 60, ordinal=self._ordinal
        )
        product = self._products["/Render/Camera"].frames[-1]
        self._output = product.render_vars["/Render/Camera/LdrColor"]
        if frame.get("pick") is not None:
            self._read_pick(product, objects, batches)
        self._objects = objects
        self._light_values = lights
        if changed or batches_changed:
            live_meshes = {item["mesh_id"] for item in chain(objects.values(), batches.values())}
            self._geometry = {key: value for key, value in self._geometry.items() if key in live_meshes}
            live_skins = {item["skin"] for item in batches.values()}
            self._skin_layers = {key: value for key, value in self._skin_layers.items() if key in live_skins}

    def render(self, frame):
        self._render_output(frame)
        return self.read_image()

    def read_image(self):
        """Download ovrtx output for CPU rendering results or explicit capture."""
        if self._mapped is not None:
            self._mapped.unmap()
            self._mapped = None
        mapped = self._output.map(device=ovrtx.Device.CPU)
        try:
            view = np.from_dlpack(mapped)
            pixels = view[..., :3].copy()
            del view
        finally:
            mapped.unmap()
        return pixels

    def resize(self, width, height):
        if self._resolution == (width, height):
            return
        self._release_output()
        self._resolution = (width, height)
        # Resume progressively instead of rerunning first-frame quality passes.
        self._renderer.clear_history(render_products={"/Render/Camera"})

    def _release_output(self):
        if self._mapped is not None:
            self._mapped.unmap()
            self._mapped = None
        self._output = None
        self._products = None

    def close(self):
        try:
            self._batch_query_stack.close()
            self._release_output()
            if self._attached:
                self._attached = False
                self._renderer.detach_ovstage()
        finally:
            if self._stage is not None:
                self._stage.destroy()
            self._renderer.destroy()


class OvrtxAppAdapter(OvrtxSceneAdapter):
    """Validate App scene updates before rendering with ovrtx."""

    def __init__(self, width, height):
        super().__init__(width, height)
        self._texture_paths = {}
        self._validated_meshes = set()

    def render(self, frame):
        from .scene_renderer import _validate_frame

        return super().render(
            _validate_frame(frame, self._texture_paths, self._validated_meshes)
        )

    def render_gpu(self, frame):
        """Expose ovrtx CUDA image handles for App's OpenGL presenter."""
        from .scene_renderer import _validate_frame

        self._render_output(
            _validate_frame(frame, self._texture_paths, self._validated_meshes)
        )
        # The presenter waits on the event and completes its GPU copy before
        # this mapping is released by readback, resize, close, or the next frame.
        self._mapped = self._output.map(device=ovrtx.Device.CUDA_ARRAY, sync_stream=0)
        tensor = self._mapped[next(iter(self._mapped))]
        if tensor.shape != (self._resolution[1], self._resolution[0], 4):
            raise RuntimeError(
                f"ovrtx presentation requires RGBA8 output, got {tensor.shape}"
            )
        return {
            "array": tensor.cuda_array,
            "device": tensor.__dlpack_device__()[1],
            "ready_event": self._mapped.wait_event or 0,
        }
