///
/// Simulation buffer Python bindings
///

#include "sim/gpu_array_view.hpp"
#include "py_array_view.hpp"

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
    result["data"] =
        py::make_tuple(reinterpret_cast<uintptr_t>(view.data), false);

    if (!view.strides.empty()) {
        py::tuple strides(view.strides.size());
        size_t itemSize = KE::Sim::simDTypeSize(view.dtype);
        for (size_t i = 0; i < view.strides.size(); ++i)
            strides[i] = static_cast<int64_t>(view.strides[i] *
                                              static_cast<int64_t>(itemSize));
        result["strides"] = strides;
    }
    if (view.streamHandle != 0)
        result["stream"] = view.streamHandle;
    return result;
}

py::object gpuArrayViewTorch(const KE::Sim::GpuArrayView& view) {
    if (!view.isCuda())
        throw std::runtime_error("GpuArrayView.torch() requires a CUDA buffer");

    KE::Sim::GpuArrayView tensorView = view;
    if (tensorView.dtype == KE::Sim::SimDType::UInt32)
        tensorView.dtype = KE::Sim::SimDType::Int32;
    else if (tensorView.dtype == KE::Sim::SimDType::UInt64)
        tensorView.dtype = KE::Sim::SimDType::Int64;

    py::object object = py::cast(tensorView);
    py::object asTensor = py::module_::import("torch").attr("as_tensor");
    std::string device =
        view.deviceId >= 0 ? "cuda:" + std::to_string(view.deviceId) : "cuda";
    return asTensor(py::arg("data") = object, py::arg("device") = device);
}

} // namespace

void bind_sim_buffer(py::module& m) {
    using namespace KE::Sim;
    py::enum_<SimMemoryType>(m, "SimMemoryType")
        .value("CPU_HOST", SimMemoryType::CPUHost)
        .value("CPU_PINNED", SimMemoryType::CPUPinned)
        .value("CUDA_DEVICE", SimMemoryType::CUDADevice)
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
        .def(
            "set_owner",
            [](GpuArrayView& self, py::object owner) {
                if (owner.is_none()) {
                    self.owner.reset();
                    return;
                }
                PyObject* retained = owner.ptr();
                Py_INCREF(retained);
                self.owner = std::shared_ptr<void>(retained, [](void* value) {
                    if (!value || !Py_IsInitialized())
                        return;
                    py::gil_scoped_acquire gil;
                    Py_DECREF(reinterpret_cast<PyObject*>(value));
                });
            },
            py::arg("owner"),
            "Keep a Python buffer owner alive with this metadata view.")
        .def_property_readonly("empty", &GpuArrayView::empty)
        .def_property_readonly("has_owner", &GpuArrayView::hasOwner)
        .def_property_readonly("is_cuda", &GpuArrayView::isCuda)
        .def_property_readonly("is_cpu", &GpuArrayView::isCpu)
        .def_property_readonly("numel", &GpuArrayView::numel)
        .def_property_readonly("byte_size", &GpuArrayView::byteSize)
        .def_property_readonly("__cuda_array_interface__", &cudaArrayInterface)
        .def("torch", &gpuArrayViewTorch,
             "Return a zero-copy Torch CUDA tensor view of this buffer.");
}
