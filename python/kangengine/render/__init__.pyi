"""Typed low-level rendering API. Array-like arguments must expose a contiguous buffer."""

from __future__ import annotations

from enum import Enum
from typing import Any, Callable, overload

class BackendType(Enum):
    """Graphics backend selected by the application."""
    OPENGL: BackendType
    VULKAN: BackendType
    WEBGPU: BackendType

class TextureRole(Enum):
    """Semantic texture slot used by built-in materials."""
    BASE_COLOR: TextureRole
    DIFFUSE: TextureRole
    NORMAL: TextureRole
    METALLIC_ROUGHNESS: TextureRole
    AMBIENT_OCCLUSION: TextureRole
    EMISSIVE: TextureRole
    METALLIC: TextureRole
    ROUGHNESS: TextureRole
    OCCLUSION_ROUGHNESS_METALLIC: TextureRole

class AlphaMode(Enum):
    """Fragment alpha policy."""
    OPAQUE: AlphaMode
    MASK: AlphaMode
    BLEND: AlphaMode

class ToneMapMode(Enum):
    """Final-output tone mapping operator."""
    OFF: ToneMapMode
    REINHARD: ToneMapMode
    EXPONENTIAL: ToneMapMode
    ACES_NARKOWICZ: ToneMapMode
    ACES_FITTED: ToneMapMode

class TransformSource(Enum):
    """Owner of a renderable's instance transforms."""
    SCENE_GRAPH: TransformSource
    EXTERNAL_BUFFER: TransformSource

class TextAlignment(Enum):
    """Horizontal text alignment."""
    LEFT: TextAlignment
    CENTER: TextAlignment
    RIGHT: TextAlignment

class TextDepthMode(Enum):
    """Text depth-testing policy."""
    DEPTH_TESTED: TextDepthMode
    OVERLAY: TextDepthMode

class ScreenAnchor(Enum):
    """Anchor used for screen-space placement."""
    TOP_LEFT: ScreenAnchor
    TOP_CENTER: ScreenAnchor
    TOP_RIGHT: ScreenAnchor
    CENTER_LEFT: ScreenAnchor
    CENTER: ScreenAnchor
    CENTER_RIGHT: ScreenAnchor
    BOTTOM_LEFT: ScreenAnchor
    BOTTOM_CENTER: ScreenAnchor
    BOTTOM_RIGHT: ScreenAnchor

class ExternalBufferFormat(Enum):
    """Memory layout of externally owned transform elements."""
    MAT4: ExternalBufferFormat
    POSITION_ROTATION: ExternalBufferFormat
    POSITION_ROTATION_SCALE: ExternalBufferFormat
    CUSTOM: ExternalBufferFormat

class ExternalSyncPolicy(Enum):
    """Synchronization policy for an external producer."""
    NONE: ExternalSyncPolicy
    VERSIONED: ExternalSyncPolicy
    FENCE: ExternalSyncPolicy
    EVENT: ExternalSyncPolicy

class ExternalBufferDesc:
    """Describe transform storage owned outside the renderer."""
    view: Any
    format: ExternalBufferFormat
    count: int
    stride_bytes: int
    sync_policy: ExternalSyncPolicy
    def __init__(self, *, view: Any = ..., format: ExternalBufferFormat = ExternalBufferFormat.MAT4, count: int = 0, stride_bytes: int = 0, sync_policy: ExternalSyncPolicy = ExternalSyncPolicy.NONE) -> None: ...

class TextureWrap(Enum):
    """Texture-coordinate wrapping mode."""
    REPEAT: TextureWrap
    CLAMP_TO_EDGE: TextureWrap
    MIRRORED_REPEAT: TextureWrap

class TextureFilter(Enum):
    """Texture sampling filter."""
    NEAREST: TextureFilter
    LINEAR: TextureFilter
    LINEAR_MIPMAP_LINEAR: TextureFilter

class SamplerDesc:
    """Backend-neutral texture sampler settings."""
    wrap_u: TextureWrap
    wrap_v: TextureWrap
    min_filter: TextureFilter
    mag_filter: TextureFilter
    def __init__(self, *, wrap_u: TextureWrap = TextureWrap.REPEAT, wrap_v: TextureWrap = TextureWrap.REPEAT, min_filter: TextureFilter = TextureFilter.LINEAR_MIPMAP_LINEAR, mag_filter: TextureFilter = TextureFilter.LINEAR) -> None: ...

class ShaderType(Enum):
    """Programmable shader stage."""
    VERTEX: ShaderType
    FRAGMENT: ShaderType
    GEOMETRY: ShaderType
    COMPUTE: ShaderType

class ShaderStage:
    """One source stage in a pipeline shader description."""
    source: str
    stage: ShaderType
    entry_point: str
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, source: str, stage: ShaderType, entry_point: str = "main") -> None: ...

class ShaderDesc:
    """Shader stages and diagnostic name used to build a pipeline."""
    stages: list[ShaderStage]
    name: str
    def __init__(self) -> None: ...

class BufferUsage(Enum):
    """Bit flags describing GPU buffer use."""
    NONE: BufferUsage
    VERTEX: BufferUsage
    INDEX: BufferUsage
    UNIFORM: BufferUsage
    COPY_SRC: BufferUsage
    COPY_DST: BufferUsage
    def __or__(self, other: BufferUsage) -> BufferUsage:
        """Combine buffer usage flags."""
        ...

class VertexFormat(Enum):
    """Format of one vertex attribute."""
    FLOAT32: VertexFormat
    FLOAT32_X2: VertexFormat
    FLOAT32_X3: VertexFormat
    FLOAT32_X4: VertexFormat
    SINT32_X4: VertexFormat

class VertexStepMode(Enum):
    """Frequency at which a vertex buffer advances."""
    VERTEX: VertexStepMode
    INSTANCE: VertexStepMode

class VertexAttributeDesc:
    """Map one vertex attribute to a shader location."""
    format: VertexFormat
    offset: int
    shader_location: int
    @overload
    def __init__(self) -> None: ...
    @overload
    def __init__(self, format: VertexFormat, offset: int, shader_location: int) -> None: ...

class VertexBufferLayout:
    """Describe the stride and attributes of one vertex-buffer slot."""
    array_stride: int
    step_mode: VertexStepMode
    attributes: list[VertexAttributeDesc]
    def __init__(self) -> None: ...

class PrimitiveTopology(Enum):
    """Primitive assembly topology."""
    POINT_LIST: PrimitiveTopology
    LINE_LIST: PrimitiveTopology
    LINE_STRIP: PrimitiveTopology
    TRIANGLE_LIST: PrimitiveTopology
    TRIANGLE_STRIP: PrimitiveTopology

class CullMode(Enum):
    """Triangle face-culling mode."""
    NONE: CullMode
    FRONT: CullMode
    BACK: CullMode

class CompareFunction(Enum):
    """Depth or stencil comparison operator."""
    NEVER: CompareFunction
    LESS: CompareFunction
    LESS_EQUAL: CompareFunction
    GREATER: CompareFunction
    GREATER_EQUAL: CompareFunction
    EQUAL: CompareFunction
    NOT_EQUAL: CompareFunction
    ALWAYS: CompareFunction

class IndexFormat(Enum):
    """Index-buffer element format."""
    UINT16: IndexFormat
    UINT32: IndexFormat

class SceneHookBlendMode(Enum):
    """Blend preset for a scene-hook pipeline."""
    OPAQUE: SceneHookBlendMode
    ALPHA: SceneHookBlendMode
    ADDITIVE: SceneHookBlendMode

class SceneHookPipelineDesc:
    """Describe a custom graphics pipeline inserted into scene rendering."""
    shader: ShaderDesc
    vertex_buffers: list[VertexBufferLayout]
    topology: PrimitiveTopology
    cull_mode: CullMode
    blend: SceneHookBlendMode
    depth_test: bool
    depth_write: bool
    use_scene_frame_bindings: bool
    depth_compare: CompareFunction
    label: str
    def __init__(self) -> None: ...

class RenderHookPhase(Enum):
    """Point at which a scene command-recording callback runs."""
    AFTER_OPAQUE: RenderHookPhase
    AFTER_TRANSPARENT: RenderHookPhase

class Texture:
    """GPU texture created by a graphics device."""
    @property
    def width(self) -> int:
        """Return the width in pixels."""
        ...
    @property
    def height(self) -> int:
        """Return the height in pixels."""
        ...
    def bind(self, slot: int = 0) -> None:
        """Bind the texture to a backend texture unit."""
        ...
    def unbind(self) -> None:
        """Unbind the texture from the active context."""
        ...
    def get_width(self) -> int:
        """Return the width in pixels."""
        ...
    def get_height(self) -> int:
        """Return the height in pixels."""
        ...

class Buffer:
    """GPU buffer created by a graphics device."""
    @property
    def size(self) -> int:
        """Return the allocation size in bytes."""
        ...
    @property
    def usage(self) -> BufferUsage:
        """Return the buffer usage flags."""
        ...
    def set_data(self, data: Any, offset: int = 0) -> None:
        """Upload a contiguous Python buffer at a byte offset."""
        ...

class GraphicsPipeline:
    """Opaque compiled graphics pipeline. Keep it alive while hooks use it."""

class RenderPassEncoder:
    """Record draw state and commands into the current scene render pass."""
    def set_viewport(self, x: float, y: float, width: float, height: float, min_depth: float = 0.0, max_depth: float = 1.0) -> None:
        """Set the viewport and depth range."""
        ...
    def set_pipeline(self, pipeline: GraphicsPipeline) -> None:
        """Select the graphics pipeline for subsequent draws."""
        ...
    def set_vertex_buffer(self, slot: int, buffer: Buffer, offset: int = 0) -> None:
        """Bind a vertex buffer to a pipeline slot."""
        ...
    def set_index_buffer(self, buffer: Buffer, format: IndexFormat, offset: int = 0) -> None:
        """Bind an index buffer and its element format."""
        ...
    def draw(self, vertex_count: int, instance_count: int = 1, first_vertex: int = 0, first_instance: int = 0) -> None:
        """Record a non-indexed draw."""
        ...
    def draw_indexed(self, index_count: int, instance_count: int = 1, first_index: int = 0, base_vertex: int = 0, first_instance: int = 0) -> None:
        """Record an indexed draw."""
        ...

class RenderHookContext:
    """Borrowed scene-pass state passed to a render-hook callback."""
    @property
    def pass_encoder(self) -> RenderPassEncoder:
        """Return the borrowed command encoder; do not retain it."""
        ...
    @property
    def width(self) -> int:
        """Return the render-target width."""
        ...
    @property
    def height(self) -> int:
        """Return the render-target height."""
        ...

class GraphicsDevice:
    """Factory for backend graphics resources."""
    def create_texture(self, path: str, flip: bool = False, sampler: SamplerDesc = ...) -> Texture:
        """Load a texture from an image file."""
        ...
    def create_buffer(self, data: Any, usage: BufferUsage, label: str = "") -> Buffer:
        """Create a GPU buffer initialized from a contiguous Python buffer."""
        ...

class ProfilerCapabilities:
    """Backend timing support; pass timing does not imply arbitrary timestamps."""
    @property
    def cpu_scopes(self) -> bool:
        """Whether CPU scopes are supported."""
        ...
    @property
    def gpu_timestamps(self) -> bool:
        """Whether any GPU timestamp path is available."""
        ...
    @property
    def pass_timestamps(self) -> bool:
        """Whether render-pass begin/end timestamps are supported."""
        ...
    @property
    def command_timestamps(self) -> bool:
        """Whether arbitrary command timestamps are supported."""
        ...
    @property
    def external_timestamps(self) -> bool:
        """Whether native render scopes outside command buffers are supported."""
        ...
    @property
    def in_pass_timestamps(self) -> bool:
        """Whether arbitrary in-pass timestamps are supported."""
        ...
    @property
    def debug_groups(self) -> bool:
        """Whether backend pass debug labels are supported."""
        ...
    @property
    def timestamp_capacity(self) -> int:
        """Maximum timestamp slots; each measured pass uses two slots."""
        ...

class ProfileTimingDomain(Enum):
    """Clock domain of a profiler sample."""
    CPU: ProfileTimingDomain
    GPU: ProfileTimingDomain

class ProfileSampleStatus(Enum):
    """Availability or failure reason for a timing sample."""
    PENDING: ProfileSampleStatus
    READY: ProfileSampleStatus
    UNSUPPORTED: ProfileSampleStatus
    CAPACITY_EXCEEDED: ProfileSampleStatus
    DROPPED: ProfileSampleStatus
    INVALID: ProfileSampleStatus
    DEVICE_LOST: ProfileSampleStatus

class ProfileSample:
    """Immutable inclusive timing sample with explicit parent identity."""
    @property
    def sample_id(self) -> int:
        """Frame-local sample identity."""
        ...
    @property
    def parent_sample_id(self) -> int | None:
        """Parent scope identity, or None for a root scope."""
        ...
    @property
    def path(self) -> str:
        """Semantic scope path."""
        ...
    @property
    def domain(self) -> ProfileTimingDomain:
        """CPU or GPU timing domain."""
        ...
    @property
    def duration_ms(self) -> float | None:
        """Inclusive duration, or None when unavailable."""
        ...
    @property
    def status(self) -> ProfileSampleStatus:
        """Availability or failure reason."""
        ...
    @property
    def available(self) -> bool:
        """Whether a valid duration is present."""
        ...

class RenderCounters:
    """Submitted RHI workload and requested upload bytes for one frame."""
    @property
    def draw_calls(self) -> int:
        """Executed RHI draws, including indexed draws."""
        ...
    @property
    def indexed_draw_calls(self) -> int:
        """Indexed subset of draw_calls."""
        ...
    @property
    def instances(self) -> int:
        """Sum of submitted instances across all passes."""
        ...
    @property
    def triangles(self) -> int:
        """Submitted triangle count including repeated passes."""
        ...
    @property
    def buffer_upload_bytes(self) -> int:
        """Requested CPU-to-buffer payload bytes."""
        ...
    @property
    def texture_upload_bytes(self) -> int:
        """Requested CPU-to-texture payload bytes."""
        ...
    @property
    def external_buffer_bytes(self) -> int:
        """Requested external device copy payload bytes."""
        ...
    @property
    def buffer_allocations(self) -> int:
        """Buffer creation or reallocation count."""
        ...
    @property
    def buffer_allocated_bytes(self) -> int:
        """Requested buffer allocation capacity in bytes."""
        ...

class FrameProfile:
    """Immutable revision of a frame snapshot with explicit GPU result status."""
    @property
    def capture_id(self) -> int:
        """Capture identity, incremented when capture restarts."""
        ...
    @property
    def frame_index(self) -> int:
        """Original App frame index."""
        ...
    @property
    def revision(self) -> int:
        """Snapshot revision for this capture and frame."""
        ...
    @property
    def finalized(self) -> bool:
        """Whether all timing results have a terminal status."""
        ...
    @property
    def backend(self) -> BackendType:
        """Backend that rendered the captured frame."""
        ...
    @property
    def gpu_latency_frames(self) -> int | None:
        """GPU result delay, or None when GPU timing is unavailable."""
        ...
    @property
    def dropped_samples(self) -> int:
        """Scopes omitted because sample capacity was exhausted."""
        ...
    @property
    def counters(self) -> RenderCounters:
        """Immutable workload counters."""
        ...
    @property
    def samples(self) -> tuple[ProfileSample, ...]:
        """Immutable sequence of timing samples."""
        ...
    @property
    def metadata(self) -> dict[str, str]:
        """Copy of settings, backend information and coverage metadata."""
        ...

class ScopeProfileSummary:
    """Inclusive per-frame scope statistics; absent and incomplete frames are excluded."""
    @property
    def path(self) -> str:
        """Scope path grouped within its timing domain."""
        ...
    @property
    def domain(self) -> ProfileTimingDomain:
        """CPU or GPU timing domain."""
        ...
    @property
    def sample_count(self) -> int:
        """Number of observed samples, including unavailable results."""
        ...
    @property
    def ready_frames(self) -> int:
        """Frames with a complete valid sum for this scope."""
        ...
    @property
    def pending_frames(self) -> int:
        """Frames awaiting results with no terminal failures."""
        ...
    @property
    def unavailable_frames(self) -> int:
        """Frames excluded by failed samples or capture overflow."""
        ...
    @property
    def mean_ms(self) -> float | None:
        """Mean of complete per-frame sums, or None."""
        ...
    @property
    def median_ms(self) -> float | None:
        """Median of complete per-frame sums, or None."""
        ...
    @property
    def p95_ms(self) -> float | None:
        """Nearest-rank 95th percentile of complete per-frame sums."""
        ...
    @property
    def max_ms(self) -> float | None:
        """Maximum complete per-frame sum, or None."""
        ...

class ProfileSummary:
    """Immutable summary of the latest retained capture, recomputed from current revisions."""
    @property
    def capture_id(self) -> int:
        """Latest retained capture ID, or zero when empty."""
        ...
    @property
    def frame_count(self) -> int:
        """Number of retained frames in the selected capture window."""
        ...
    @property
    def first_frame_index(self) -> int | None:
        """First frame index in the window, or None."""
        ...
    @property
    def last_frame_index(self) -> int | None:
        """Last frame index in the window, or None."""
        ...
    @property
    def scopes(self) -> tuple[ScopeProfileSummary, ...]:
        """Statistics grouped by domain and scope path."""
        ...

class Renderer:
    """Facade for custom pipelines and renderable resource updates."""
    def set_profiler_enabled(self, enabled: bool) -> None:
        """Enable or disable CPU/GPU pass capture at the next frame boundary."""
        ...
    @property
    def profiler_enabled(self) -> bool:
        """Requested profiling state, applied at the next frame boundary."""
        ...
    @property
    def profiler_capabilities(self) -> ProfilerCapabilities:
        """Return timing capabilities of the current backend."""
        ...
    def latest_frame_profile(self) -> FrameProfile | None:
        """Return the latest captured CPU frame, or None before capture."""
        ...
    def frame_profile_history(self) -> tuple[FrameProfile, ...]:
        """Return at most 240 immutable snapshots, oldest first."""
        ...
    def profile_summary(self, *, max_frames: int = 240) -> ProfileSummary:
        """Summarize the latest capture; max_frames must be between 1 and 240."""
        ...
    def export_profile_json(self, path: str, *, max_frames: int = 240) -> None:
        """Write the last max_frames snapshots and latest-capture summary without GPU waits."""
        ...
    def device(self) -> GraphicsDevice:
        """Return the renderer-owned graphics device."""
        ...
    def create_scene_hook_pipeline(self, desc: SceneHookPipelineDesc) -> GraphicsPipeline:
        """Create a pipeline compatible with the scene render targets."""
        ...
    def add_render_hook(self, phase: RenderHookPhase, callback: Callable[[RenderHookContext], None]) -> int:
        """Register a command-recording callback and return its handle."""
        ...
    def remove_render_hook(self, handle: int) -> bool:
        """Remove a previously registered render hook."""
        ...
    def set_point_lights(self, lights: Any) -> None:
        """Replace the renderer's point-light list."""
        ...
    def point_lights(self) -> Any:
        """Return the stored point lights."""
        ...
    def set_spot_lights(self, lights: Any) -> None:
        """Replace the renderer's spot-light list."""
        ...
    def spot_lights(self) -> Any:
        """Return the stored spot lights."""
        ...
    def sync_scene_lights(self, scene: Any) -> None:
        """Synchronize renderer lights from scene light prims."""
        ...
    def update_renderable_transforms(self, handle: int, transforms: Any, colors: Any | None = None) -> None:
        """Replace instance transforms and optional colors."""
        ...
    def set_renderable_colors(self, handle: int, colors: Any) -> None:
        """Set per-instance colors."""
        ...
    def set_renderable_external_buffer(self, handle: int, descriptor: ExternalBufferDesc) -> None:
        """Attach externally owned transform storage."""
        ...
    def map_renderable_cuda_transform_buffers(self, handles: Any, count: int, device_id: int, stream_handle: int = 0) -> Any:
        """Map transform buffers for direct CUDA writes."""
        ...
    def unmap_renderable_cuda_transform_buffers(self, handles: Any, device_id: int, stream_handle: int = 0) -> None:
        """Unmap transform buffers after CUDA writes."""
        ...
    def set_renderable_double_sided(self, handle: int, enabled: bool = True) -> None:
        """Set whether both triangle faces are rendered."""
        ...
    def set_renderable_casts_shadow(self, handle: int, enabled: bool = True) -> None:
        """Set whether the renderable casts shadows."""
        ...
    def set_renderable_alpha_mode(self, handle: int, mode: AlphaMode, cutoff: float = 0.5) -> None:
        """Set alpha handling and mask cutoff."""
        ...
    @overload
    def set_renderable_texture(self, handle: int, texture: Texture, role: TextureRole) -> None:
        """Attach a texture by semantic material role."""
        ...
    @overload
    def set_renderable_texture(self, handle: int, texture: Texture, slot: int = 0) -> None: ...
    def update_renderable_geometry(self, handle: int, positions: Any, normals: Any | None = None) -> None:
        """Update dynamic positions and optional normals."""
        ...
    def update_renderable_skinning_matrices(self, handle: int, bone_matrices: Any) -> None:
        """Update matrices for a skinned renderable."""
        ...

__all__: list[str]
