from __future__ import annotations

from enum import Enum
from typing import Any, overload

class BackendType(Enum):
    OpenGL: BackendType
    Vulkan: BackendType
    WebGPU: BackendType

class TextureRole(Enum):
    BaseColor: TextureRole
    Diffuse: TextureRole
    Normal: TextureRole
    MetallicRoughness: TextureRole
    AmbientOcclusion: TextureRole
    Emissive: TextureRole
    Metallic: TextureRole
    Roughness: TextureRole
    OcclusionRoughnessMetallic: TextureRole

class AlphaMode(Enum):
    Opaque: AlphaMode
    Mask: AlphaMode
    Blend: AlphaMode

class ToneMapMode(Enum):
    Off: ToneMapMode
    Reinhard: ToneMapMode
    Exponential: ToneMapMode
    AcesNarkowicz: ToneMapMode
    AcesFitted: ToneMapMode

class TextAlignment(Enum):
    Left: TextAlignment
    Center: TextAlignment
    Right: TextAlignment

class TextDepthMode(Enum):
    DepthTested: TextDepthMode
    Overlay: TextDepthMode

class ScreenAnchor(Enum):
    TopLeft: ScreenAnchor
    TopCenter: ScreenAnchor
    TopRight: ScreenAnchor
    CenterLeft: ScreenAnchor
    Center: ScreenAnchor
    CenterRight: ScreenAnchor
    BottomLeft: ScreenAnchor
    BottomCenter: ScreenAnchor
    BottomRight: ScreenAnchor

class TransformSource(Enum):
    SceneGraph: TransformSource
    ExternalBuffer: TransformSource

class ExternalBufferFormat(Enum):
    MAT4: ExternalBufferFormat
    POSITION_ROTATION: ExternalBufferFormat
    POSITION_ROTATION_SCALE: ExternalBufferFormat
    CUSTOM: ExternalBufferFormat

class ExternalSyncPolicy(Enum):
    NONE: ExternalSyncPolicy
    VERSIONED: ExternalSyncPolicy
    FENCE: ExternalSyncPolicy
    EVENT: ExternalSyncPolicy

class ExternalBufferDesc:
    view: Any
    format: ExternalBufferFormat
    count: int
    stride_bytes: int
    sync_policy: ExternalSyncPolicy
    def __init__(
        self,
        *,
        view: Any = ...,
        format: ExternalBufferFormat = ExternalBufferFormat.MAT4,
        count: int = 0,
        stride_bytes: int = 0,
        sync_policy: ExternalSyncPolicy = ExternalSyncPolicy.NONE,
    ) -> None: ...
    def __repr__(self) -> str: ...

class TextureWrap(Enum):
    Repeat: TextureWrap
    ClampToEdge: TextureWrap
    MirroredRepeat: TextureWrap

class TextureFilter(Enum):
    Nearest: TextureFilter
    Linear: TextureFilter
    LinearMipmapLinear: TextureFilter

class SamplerDesc:
    wrap_u: TextureWrap
    wrap_v: TextureWrap
    min_filter: TextureFilter
    mag_filter: TextureFilter
    def __init__(
        self,
        *,
        wrap_u: TextureWrap = TextureWrap.Repeat,
        wrap_v: TextureWrap = TextureWrap.Repeat,
        min_filter: TextureFilter = TextureFilter.LinearMipmapLinear,
        mag_filter: TextureFilter = TextureFilter.Linear,
    ) -> None: ...
    def __repr__(self) -> str: ...

class Shader:
    def use(self) -> None: ...
    def bind(self) -> None: ...
    def unbind(self) -> None: ...
    def set_bool(self, name: str, value: bool) -> None: ...
    def set_int(self, name: str, value: int) -> None: ...
    def set_float(self, name: str, value: float) -> None: ...
    @overload
    def set_color(self, name: str, value: Any) -> None: ...
    @overload
    def set_color(self, name: str, r: float, g: float, b: float, a: float) -> None: ...
    @overload
    def set_vec2(self, name: str, value: Any) -> None: ...
    @overload
    def set_vec2(self, name: str, x: float, y: float) -> None: ...
    @overload
    def set_vec3(self, name: str, value: Any) -> None: ...
    @overload
    def set_vec3(self, name: str, x: float, y: float, z: float) -> None: ...
    @overload
    def set_vec4(self, name: str, value: Any) -> None: ...
    @overload
    def set_vec4(self, name: str, x: float, y: float, z: float, w: float) -> None: ...
    def set_mat2(self, name: str, value: Any) -> None: ...
    def set_mat3(self, name: str, value: Any) -> None: ...
    def set_mat4(self, name: str, value: Any) -> None: ...
    def set_uniform_block_binding(
        self, block_name: str, binding_point: int
    ) -> None: ...

class Texture:
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    def bind(self, slot: int = 0) -> None: ...
    def unbind(self) -> None: ...
    def get_width(self) -> int: ...
    def get_height(self) -> int: ...

class GraphicsDevice:
    def create_shader(self, vertex_source: str, fragment_source: str) -> Shader: ...
    def create_shader_from_file(self, vert_path: str, frag_path: str) -> Shader: ...
    def create_texture(
        self,
        path: str,
        flip: bool = False,
        sampler: SamplerDesc = ...,
    ) -> Texture: ...

class Renderer:
    def device(self) -> GraphicsDevice: ...
    def set_point_lights(self, lights: Any) -> None: ...
    def point_lights(self) -> Any: ...
    def set_spot_lights(self, lights: Any) -> None: ...
    def spot_lights(self) -> Any: ...
    def sync_scene_lights(self, scene: Any) -> None: ...
    def update_renderable_transforms(
        self, handle: int, transforms: Any, colors: Any | None = None
    ) -> None: ...
    def set_renderable_colors(self, handle: int, colors: Any) -> None: ...
    def set_renderable_external_buffer(
        self, handle: int, descriptor: ExternalBufferDesc
    ) -> None: ...
    def map_renderable_cuda_transform_buffers(
        self,
        handles: Any,
        count: int,
        device_id: int,
        stream_handle: int = 0,
    ) -> Any: ...
    def unmap_renderable_cuda_transform_buffers(
        self, handles: Any, device_id: int, stream_handle: int = 0
    ) -> None: ...
    def set_renderable_double_sided(
        self, handle: int, double_sided: bool = True
    ) -> None: ...
    def set_renderable_casts_shadow(
        self, handle: int, casts_shadow: bool = True
    ) -> None: ...
    def set_renderable_alpha_mode(
        self, handle: int, mode: AlphaMode, cutoff: float = 0.5
    ) -> None: ...
    @overload
    def set_renderable_texture(
        self, handle: int, texture: Texture, role: TextureRole
    ) -> None: ...
    @overload
    def set_renderable_texture(
        self, handle: int, texture: Texture, slot: int = 0
    ) -> None: ...
    def update_renderable_geometry(
        self, handle: int, positions: Any, normals: Any | None = None
    ) -> None: ...
    def update_renderable_skinning_matrices(
        self, handle: int, bone_matrices: Any
    ) -> None: ...

__all__: list[str]
