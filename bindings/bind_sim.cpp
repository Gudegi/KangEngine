///
/// Simulation buffer Python bindings
///

#include "sim/gpu_array_view.hpp"
#include "physics/sim_model.hpp"
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
        .def_property_readonly("empty", &GpuArrayView::empty)
        .def_property_readonly("has_owner", &GpuArrayView::hasOwner)
        .def_property_readonly("is_cuda", &GpuArrayView::isCuda)
        .def_property_readonly("is_cpu", &GpuArrayView::isCpu)
        .def_property_readonly("numel", &GpuArrayView::numel)
        .def_property_readonly("byte_size", &GpuArrayView::byteSize)
        .def_property_readonly("__cuda_array_interface__",
                               &cudaArrayInterface);

    py::class_<KE::SimModel>(
        m, "SimModel",
        "Low-level C++ simulation topology used by renderer batch paths.")
        .def(py::init<>())
        .def("set_body_renderables", &KE::SimModel::setBodyRenderables)
        .def("add_shape",
             [](KE::SimModel& self, int bodyId, KE::RenderableHandle renderable,
                py::sequence localPos, py::sequence localRot,
                const std::string& name) {
                 return self.addShape(bodyId, renderable,
                                      vec3FromSequence(localPos),
                                      quatFromXYZWSequence(localRot), name);
             },
             py::arg("body_id"), py::arg("renderable"),
             py::arg("local_pos") = py::make_tuple(0.0f, 0.0f, 0.0f),
             py::arg("local_rot") = py::make_tuple(0.0f, 0.0f, 0.0f, 1.0f),
             py::arg("name") = "")
        .def("add_object_boundary", &KE::SimModel::addObjectBoundary,
             py::arg("body_start"), py::arg("body_count"),
             py::arg("name") = "")
        .def_property_readonly("body_count", &KE::SimModel::bodyCount)
        .def_property_readonly("shape_count", &KE::SimModel::shapeCount)
        .def("is_valid", &KE::SimModel::isValid)
        .def_readwrite("body_names", &KE::SimModel::bodyNames)
        .def_readwrite("shape_names", &KE::SimModel::shapeNames)
        .def_readwrite("object_names", &KE::SimModel::objectNames);

    py::class_<KE::SimState>(
        m, "SimState",
        "Low-level C++ simulation state used by renderer batch paths.")
        .def(py::init<>())
        .def("resize", &KE::SimState::resize)
        .def("body_index", &KE::SimState::bodyIndex)
        .def("set_body_transform",
             [](KE::SimState& self, int envId, int bodyId, py::sequence pos,
                py::sequence rot) {
                 self.setBodyTransform(envId, bodyId, vec3FromSequence(pos),
                                       quatFromXYZWSequence(rot));
             },
             py::arg("env_id"), py::arg("body_id"), py::arg("pos"),
             py::arg("rot"))
        .def("get_body_pos",
             [](const KE::SimState& self, int envId, int bodyId) {
                 return vec3Tuple(self.bodyPos[static_cast<size_t>(
                     self.bodyIndex(envId, bodyId))]);
             })
        .def("get_body_rot",
             [](const KE::SimState& self, int envId, int bodyId) {
                 return quatXYZWTuple(self.bodyRot[static_cast<size_t>(
                     self.bodyIndex(envId, bodyId))]);
             })
        .def("body_matrix",
             [](const KE::SimState& self, int envId, int bodyId) {
                 return mat4Tuple(self.bodyMatrix(envId, bodyId));
             })
        .def_readonly("num_envs", &KE::SimState::numEnvs)
        .def_readonly("num_bodies", &KE::SimState::numBodies);

    py::class_<KE::SimVisualBatch>(
        m, "SimVisualBatch",
        "Low-level C++ SimState-to-renderable transform batch.")
        .def(py::init<>())
        .def("clear", &KE::SimVisualBatch::clear)
        .def("set_model",
             [](KE::SimVisualBatch& self, const KE::SimModel& model) {
                 self.setModel(&model);
             })
        .def("prepare_from_state", &KE::SimVisualBatch::prepareFromState)
        .def_property_readonly("renderable_count",
                               &KE::SimVisualBatch::renderableCount)
        .def("renderable", &KE::SimVisualBatch::renderable)
        .def("transforms",
             [](const KE::SimVisualBatch& self, int shapeId) {
                 return mat4List(self.transforms(shapeId));
             })
        .def("external_transform_desc",
             &KE::SimVisualBatch::externalTransformDesc,
             py::arg("shape_id"), py::arg("version"),
             py::arg("name") = "sim_visual_batch_transforms");
}
