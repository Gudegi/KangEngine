///
/// Simulation buffer Python bindings
///

#include "sim/gpu_array_view.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace {

const char* simDTypeTypestr(KE::Sim::SimDType dtype) {
    switch (dtype) {
    case KE::Sim::SimDType::Float32:
        return "<f4";
    case KE::Sim::SimDType::Float64:
        return "<f8";
    case KE::Sim::SimDType::Int32:
        return "<i4";
    case KE::Sim::SimDType::UInt32:
        return "<u4";
    case KE::Sim::SimDType::Int64:
        return "<i8";
    case KE::Sim::SimDType::UInt64:
        return "<u8";
    case KE::Sim::SimDType::UInt8:
        return "|u1";
    case KE::Sim::SimDType::Bool:
        return "|b1";
    case KE::Sim::SimDType::Unknown:
        return "";
    }
    return "";
}

py::tuple intVectorTuple(const std::vector<int64_t>& values) {
    py::tuple result(values.size());
    for (size_t i = 0; i < values.size(); ++i)
        result[i] = values[i];
    return result;
}

py::dict cudaArrayInterface(const KE::Sim::GpuArrayView& view) {
    if (!view.isCuda())
        throw std::runtime_error(
            "__cuda_array_interface__ is only valid for CUDA buffers");
    if (view.data == nullptr)
        throw std::runtime_error(
            "__cuda_array_interface__ requires a non-null data pointer");

    const char* typestr = simDTypeTypestr(view.dtype);
    if (typestr[0] == '\0')
        throw std::runtime_error(
            "__cuda_array_interface__ requires a known dtype");

    py::dict result;
    result["version"] = 3;
    result["shape"] = intVectorTuple(view.shape);
    result["typestr"] = typestr;
    result["data"] = py::make_tuple(
        reinterpret_cast<uintptr_t>(view.data), false);

    if (!view.strides.empty()) {
        py::tuple strides(view.strides.size());
        size_t itemSize = KE::Sim::simDTypeSize(view.dtype);
        for (size_t i = 0; i < view.strides.size(); ++i)
            strides[i] = static_cast<int64_t>(
                view.strides[i] * static_cast<int64_t>(itemSize));
        result["strides"] = strides;
    }
    if (view.streamHandle != 0)
        result["stream"] = view.streamHandle;
    return result;
}

} // namespace

void bind_sim(py::module& m) {
    using namespace KE::Sim;

    py::enum_<SimMemoryType>(m, "SimMemoryType")
        .value("CPU_HOST", SimMemoryType::CpuHost)
        .value("CPU_PINNED", SimMemoryType::CpuPinned)
        .value("CUDA_DEVICE", SimMemoryType::CudaDevice)
        .value("OPENGL_BUFFER", SimMemoryType::OpenGLBuffer)
        .value("VULKAN_BUFFER", SimMemoryType::VulkanBuffer)
        .value("WEBGPU_BUFFER", SimMemoryType::WebGPUBuffer)
        .value("EXTERNAL", SimMemoryType::External);

    py::enum_<SimDType>(m, "SimDType")
        .value("FLOAT32", SimDType::Float32)
        .value("FLOAT64", SimDType::Float64)
        .value("INT32", SimDType::Int32)
        .value("UINT32", SimDType::UInt32)
        .value("INT64", SimDType::Int64)
        .value("UINT64", SimDType::UInt64)
        .value("UINT8", SimDType::UInt8)
        .value("BOOL", SimDType::Bool)
        .value("UNKNOWN", SimDType::Unknown);

    py::enum_<SimLifetimePolicy>(m, "SimLifetimePolicy")
        .value("BORROWED", SimLifetimePolicy::Borrowed)
        .value("SHARED_OWNER", SimLifetimePolicy::SharedOwner)
        .value("EXTERNAL_OWNER", SimLifetimePolicy::ExternalOwner);

    py::class_<GpuArrayView>(
        m, "GpuArrayView",
        "Metadata view for CPU/GPU simulation buffers. The producer owns "
        "memory lifetime and synchronization.")
        .def(py::init<>())
        .def_property(
            "ptr",
            [](const GpuArrayView& self) {
                return reinterpret_cast<uintptr_t>(self.data);
            },
            [](GpuArrayView& self, uintptr_t ptr) {
                self.data = reinterpret_cast<void*>(ptr);
            },
            "Raw memory pointer encoded as an integer.")
        .def_readwrite("memory_type", &GpuArrayView::memoryType)
        .def_readwrite("dtype", &GpuArrayView::dtype)
        .def_readwrite("lifetime", &GpuArrayView::lifetime)
        .def_readwrite("device_id", &GpuArrayView::deviceId)
        .def_readwrite("shape", &GpuArrayView::shape)
        .def_readwrite("strides", &GpuArrayView::strides)
        .def_readwrite("version", &GpuArrayView::version)
        .def_readwrite("stream_handle", &GpuArrayView::streamHandle)
        .def_readwrite("ready_event_handle", &GpuArrayView::readyEventHandle)
        .def_readwrite("name", &GpuArrayView::name)
        .def_property_readonly("empty", &GpuArrayView::empty)
        .def_property_readonly("has_owner", &GpuArrayView::hasOwner)
        .def_property_readonly("is_cuda", &GpuArrayView::isCuda)
        .def_property_readonly("is_cpu", &GpuArrayView::isCpu)
        .def_property_readonly("numel", &GpuArrayView::numel)
        .def_property_readonly("byte_size", &GpuArrayView::byteSize)
        .def_property_readonly("__cuda_array_interface__",
                               &cudaArrayInterface);
}
