"""Torch-backed simulation buffers and native interop helpers."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SimBuffer:
    """Simulation buffer backed by one CPU or CUDA Torch tensor."""

    data: torch.Tensor
    sim_device: str = "cpu"
    memory_type: str = "cpu_host"
    device_id: int = -1
    version: int = 0
    lifetime: str = "shared_owner"
    stream_handle: int = 0
    ready_event_handle: int = 0
    owner: object | None = None

    def __post_init__(self):
        if not torch.is_tensor(self.data):
            raise TypeError("SimBuffer data must be a torch.Tensor")

    @property
    def is_cpu(self) -> bool:
        return self.data.device.type == "cpu"

    @property
    def is_cuda(self) -> bool:
        return self.data.device.type == "cuda"

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    @property
    def strides(self):
        return tuple(int(stride) for stride in self.data.stride())

    @property
    def cuda_array_interface(self):
        if not self.is_cuda:
            return None
        return self.__cuda_array_interface__

    @property
    def __cuda_array_interface__(self):
        if not self.is_cuda:
            raise AttributeError(
                "__cuda_array_interface__ is only available for CUDA buffers"
            )

        result = {
            "version": 3,
            "shape": tuple(int(dim) for dim in self.data.shape),
            "typestr": _cuda_array_typestr(self.data.dtype),
            "data": (int(self.data.data_ptr()), False),
        }
        if self.data.stride():
            itemsize = self.data.element_size()
            result["strides"] = tuple(
                int(stride * itemsize) for stride in self.data.stride()
            )
        if self.stream_handle:
            result["stream"] = int(self.stream_handle)
        return result


def _normalize_torch_dtype(dtype) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype

    name = getattr(dtype, "name", None) or getattr(dtype, "__name__", None)
    if name is None:
        name = str(dtype).rsplit(".", 1)[-1]
    name = str(name).lower()

    mapping = {
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
        "bool_": torch.bool,
    }
    for optional_name in ("uint32", "uint64"):
        optional_dtype = getattr(torch, optional_name, None)
        if optional_dtype is not None:
            mapping[optional_name] = optional_dtype

    try:
        return mapping[name]
    except KeyError as exc:
        raise TypeError(f"unsupported simulation dtype: {dtype!r}") from exc


def _cuda_array_typestr(dtype: torch.dtype) -> str:
    mapping = {
        torch.float32: "<f4",
        torch.float64: "<f8",
        torch.int32: "<i4",
        torch.int64: "<i8",
        torch.uint8: "|u1",
        torch.bool: "|b1",
    }
    optional_uint32 = getattr(torch, "uint32", None)
    optional_uint64 = getattr(torch, "uint64", None)
    if optional_uint32 is not None:
        mapping[optional_uint32] = "<u4"
    if optional_uint64 is not None:
        mapping[optional_uint64] = "<u8"
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise TypeError(
            f"dtype {dtype!r} is not supported by __cuda_array_interface__"
        ) from exc


def _sim_device(value, sim_device) -> torch.device:
    if sim_device is not None:
        device = torch.device(str(sim_device).strip().lower())
    elif torch.is_tensor(value):
        device = value.device
    else:
        device = torch.device("cpu")
    if device.type not in ("cpu", "cuda"):
        raise ValueError(
            "sim_device must be 'cpu', 'cuda', or 'cuda:<ordinal>' "
            f"(got {sim_device!r})"
        )
    return device


def as_sim_buffer(value, *, shape=None, sim_device=None, dtype=torch.float32):
    """Return a Torch-backed simulation buffer on CPU or CUDA."""
    device = _sim_device(value, sim_device)
    tensor = torch.as_tensor(
        value,
        dtype=_normalize_torch_dtype(dtype),
        device=device,
    )
    if shape is not None:
        tensor = tensor.reshape(shape)

    is_cuda = device.type == "cuda"
    stream_handle = (
        int(torch.cuda.current_stream(device).cuda_stream) if is_cuda else 0
    )
    return SimBuffer(
        tensor,
        str(device),
        memory_type="cuda_device" if is_cuda else "cpu_host",
        device_id=(device.index if device.index is not None else 0) if is_cuda else -1,
        lifetime="shared_owner",
        stream_handle=stream_handle,
        owner=tensor,
    )


def _memory_type_enum(memory_type: str):
    from .._core import _ke

    mapping = {
        "cpu_host": _ke.SimMemoryType.CPU_HOST,
        "cpu_pinned": _ke.SimMemoryType.CPU_PINNED,
        "cuda_device": _ke.SimMemoryType.CUDA_DEVICE,
        "opengl_buffer": _ke.SimMemoryType.OPENGL_BUFFER,
        "vulkan_buffer": _ke.SimMemoryType.VULKAN_BUFFER,
        "webgpu_buffer": _ke.SimMemoryType.WEBGPU_BUFFER,
        "external": _ke.SimMemoryType.EXTERNAL,
    }
    return mapping[memory_type]


def _lifetime_enum(lifetime: str):
    from .._core import _ke

    mapping = {
        "borrowed": _ke.SimLifetimePolicy.BORROWED,
        "shared_owner": _ke.SimLifetimePolicy.SHARED_OWNER,
        "external_owner": _ke.SimLifetimePolicy.EXTERNAL_OWNER,
    }
    return mapping[lifetime]


def _dtype_enum(dtype: torch.dtype):
    from .._core import _ke

    mapping = {
        torch.float32: _ke.SimDType.FLOAT32,
        torch.float64: _ke.SimDType.FLOAT64,
        torch.int32: _ke.SimDType.INT32,
        torch.int64: _ke.SimDType.INT64,
        torch.uint8: _ke.SimDType.UINT8,
        torch.bool: _ke.SimDType.BOOL,
    }
    optional_uint32 = getattr(torch, "uint32", None)
    optional_uint64 = getattr(torch, "uint64", None)
    if optional_uint32 is not None:
        mapping[optional_uint32] = _ke.SimDType.UINT32
    if optional_uint64 is not None:
        mapping[optional_uint64] = _ke.SimDType.UINT64
    return mapping.get(dtype, _ke.SimDType.UNKNOWN)


def to_gpu_array_view(
    value,
    *,
    shape=None,
    sim_device=None,
    dtype=torch.float32,
    name="",
):
    """Convert an array-like object into a native metadata-only buffer view."""
    from .._core import _ke

    buffer = (
        value
        if isinstance(value, SimBuffer)
        else as_sim_buffer(
            value,
            shape=shape,
            sim_device=sim_device,
            dtype=dtype,
        )
    )

    view = _ke.GpuArrayView()
    view.ptr = int(buffer.data.data_ptr())
    view.memory_type = _memory_type_enum(buffer.memory_type)
    view.dtype = _dtype_enum(buffer.dtype)
    view.lifetime = _lifetime_enum(buffer.lifetime)
    view.device_id = buffer.device_id
    view.shape = [int(dim) for dim in buffer.shape]
    view.strides = [int(stride) for stride in buffer.strides]
    view.version = int(buffer.version)
    view.stream_handle = int(buffer.stream_handle)
    view.ready_event_handle = int(buffer.ready_event_handle)
    view.name = name
    view.set_owner(buffer.owner if buffer.owner is not None else buffer.data)
    return view


def to_external_transform_desc(
    value,
    *,
    shape=None,
    sim_device=None,
    dtype=torch.float32,
    name="",
    version=None,
    sync_policy=None,
):
    """Build an ExternalBufferDesc for float32 column-major [N, 4, 4] transforms."""
    from .._core import _ke

    buffer = (
        value
        if isinstance(value, SimBuffer)
        else as_sim_buffer(
            value,
            shape=shape,
            sim_device=sim_device,
            dtype=dtype,
        )
    )
    if len(buffer.shape) != 3 or tuple(
        int(dim) for dim in buffer.shape[1:]
    ) != (4, 4):
        raise ValueError("transform buffer must have shape [N, 4, 4]")

    view = to_gpu_array_view(buffer, name=name)
    if version is not None:
        view.version = int(version)

    desc = _ke.ExternalBufferDesc()
    desc.view = view
    desc.format = _ke.ExternalBufferFormat.MAT4
    desc.count = int(buffer.shape[0])
    desc.stride_bytes = 0
    desc.sync_policy = (
        _ke.ExternalSyncPolicy.VERSIONED
        if sync_policy is None
        else sync_policy
    )
    return desc, buffer
