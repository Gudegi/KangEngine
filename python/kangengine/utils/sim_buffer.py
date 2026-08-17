"""NumPy/Torch simulation buffers and native interop helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Type, TypeAlias

import numpy as np
import torch

from .. import render as render_api

if TYPE_CHECKING:
    from .._core import _ke


BufferData: TypeAlias = np.ndarray | torch.Tensor
BufferDType: TypeAlias = np.dtype | torch.dtype | Type[np.generic] | str
BufferShape: TypeAlias = tuple[int, ...] | list[int] | torch.Size
SimDevice: TypeAlias = str | torch.device
MemoryType: TypeAlias = Literal[
    "cpu_host",
    "cpu_pinned",
    "cuda_device",
    "opengl_buffer",
    "vulkan_buffer",
    "webgpu_buffer",
    "external",
]
LifetimePolicy: TypeAlias = Literal["borrowed", "shared_owner", "external_owner"]


@dataclass(frozen=True)
class SimBuffer:
    """Simulation buffer backed by a CPU NumPy array or Torch tensor."""

    data: BufferData
    sim_device: str = "cpu"
    memory_type: MemoryType = "cpu_host"
    device_id: int = -1
    version: int = 0
    lifetime: LifetimePolicy = "shared_owner"
    stream_handle: int = 0
    ready_event_handle: int = 0
    owner: BufferData | None = None

    def __post_init__(self) -> None:
        if not torch.is_tensor(self.data) and not isinstance(self.data, np.ndarray):
            raise TypeError("SimBuffer data must be a NumPy array or torch.Tensor")
        if isinstance(self.data, np.ndarray) and not self.data.flags.c_contiguous:
            raise ValueError("CPU NumPy SimBuffer data must be C-contiguous")

    @property
    def is_cpu(self) -> bool:
        return isinstance(self.data, np.ndarray) or self.data.device.type == "cpu"

    @property
    def is_cuda(self) -> bool:
        return torch.is_tensor(self.data) and self.data.device.type == "cuda"

    @property
    def shape(self) -> tuple[int, ...] | torch.Size:
        return self.data.shape

    @property
    def dtype(self) -> np.dtype | torch.dtype:
        return self.data.dtype

    @property
    def strides(self) -> tuple[int, ...]:
        if isinstance(self.data, np.ndarray):
            return tuple(
                int(stride // self.data.itemsize) for stride in self.data.strides
            )
        return tuple(int(stride) for stride in self.data.stride())

    @property
    def cuda_array_interface(
        self,
    ) -> dict[str, int | str | tuple[int, ...] | tuple[int, bool]] | None:
        if not self.is_cuda:
            return None
        return self.__cuda_array_interface__

    @property
    def __cuda_array_interface__(
        self,
    ) -> dict[str, int | str | tuple[int, ...] | tuple[int, bool]]:
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


def _normalize_torch_dtype(dtype: BufferDType) -> torch.dtype:
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


def _sim_device(value: BufferData, sim_device: SimDevice | None) -> torch.device:
    if sim_device is not None:
        device = torch.device(str(sim_device).strip().lower())
    elif torch.is_tensor(value):
        device = value.device
    else:
        device = torch.device("cpu")
    if device.type not in ("cpu", "cuda"):
        raise ValueError(
            f"sim_device must be 'cpu', 'cuda', or 'cuda:<ordinal>' (got {sim_device!r})"
        )
    return device


def as_sim_buffer(
    value: BufferData,
    *,
    shape: BufferShape | None = None,
    sim_device: SimDevice | None = None,
    dtype: BufferDType = torch.float32,
) -> SimBuffer:
    """Return a NumPy-backed CPU or Torch-backed CUDA simulation buffer."""
    device = _sim_device(value, sim_device)
    if device.type == "cpu" and isinstance(value, np.ndarray):
        numpy_dtype = np.dtype(
            str(_normalize_torch_dtype(dtype)).removeprefix("torch.")
        )
        array = np.asarray(value, dtype=numpy_dtype)
        if shape is not None:
            array = array.reshape(shape)
        array = np.ascontiguousarray(array)
        return SimBuffer(array, "cpu", owner=array)

    tensor = torch.as_tensor(
        value,
        dtype=_normalize_torch_dtype(dtype),
        device=device,
    )
    if shape is not None:
        tensor = tensor.reshape(shape)

    is_cuda = device.type == "cuda"
    stream_handle = int(torch.cuda.current_stream(device).cuda_stream) if is_cuda else 0
    return SimBuffer(
        tensor,
        str(device),
        memory_type="cuda_device" if is_cuda else "cpu_host",
        device_id=(device.index if device.index is not None else 0) if is_cuda else -1,
        lifetime="shared_owner",
        stream_handle=stream_handle,
        owner=tensor,
    )


def _memory_type_enum(memory_type: MemoryType) -> _ke.SimMemoryType:
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


def _lifetime_enum(lifetime: LifetimePolicy) -> _ke.SimLifetimePolicy:
    from .._core import _ke

    mapping = {
        "borrowed": _ke.SimLifetimePolicy.BORROWED,
        "shared_owner": _ke.SimLifetimePolicy.SHARED_OWNER,
        "external_owner": _ke.SimLifetimePolicy.EXTERNAL_OWNER,
    }
    return mapping[lifetime]


def _dtype_enum(dtype: np.dtype | torch.dtype) -> _ke.SimDType:
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
    result = mapping.get(dtype)
    if result is not None:
        return result
    numpy_mapping = {
        np.dtype(np.float32): _ke.SimDType.FLOAT32,
        np.dtype(np.float64): _ke.SimDType.FLOAT64,
        np.dtype(np.int32): _ke.SimDType.INT32,
        np.dtype(np.int64): _ke.SimDType.INT64,
        np.dtype(np.uint8): _ke.SimDType.UINT8,
        np.dtype(np.bool_): _ke.SimDType.BOOL,
        np.dtype(np.uint32): _ke.SimDType.UINT32,
        np.dtype(np.uint64): _ke.SimDType.UINT64,
    }
    return numpy_mapping.get(np.dtype(dtype), _ke.SimDType.UNKNOWN)


def to_gpu_array_view(
    value: BufferData | SimBuffer,
    *,
    shape: BufferShape | None = None,
    sim_device: SimDevice | None = None,
    dtype: BufferDType = torch.float32,
    name: str = "",
) -> _ke.GpuArrayView:
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
    view.ptr = (
        int(buffer.data.__array_interface__["data"][0])
        if isinstance(buffer.data, np.ndarray)
        else int(buffer.data.data_ptr())
    )
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
    value: BufferData | SimBuffer,
    *,
    shape: BufferShape | None = None,
    sim_device: SimDevice | None = None,
    dtype: BufferDType = torch.float32,
    name: str = "",
    version: int | None = None,
    sync_policy: render_api.ExternalSyncPolicy | None = None,
) -> tuple[render_api.ExternalBufferDesc, SimBuffer]:
    """Build an ExternalBufferDesc for float32 column-major [N, 4, 4] transforms."""
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
    if len(buffer.shape) != 3 or tuple(int(dim) for dim in buffer.shape[1:]) != (4, 4):
        raise ValueError("transform buffer must have shape [N, 4, 4]")

    view = to_gpu_array_view(buffer, name=name)
    if version is not None:
        view.version = int(version)

    desc = render_api.ExternalBufferDesc()
    desc.view = view
    desc.format = render_api.ExternalBufferFormat.MAT4
    desc.count = int(buffer.shape[0])
    desc.stride_bytes = 0
    desc.sync_policy = (
        render_api.ExternalSyncPolicy.VERSIONED if sync_policy is None else sync_policy
    )
    return desc, buffer
