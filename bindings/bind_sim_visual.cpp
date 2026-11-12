///
/// Simulation visual synchronization, transform batches, and CUDA conversion
///

#include "sim/gpu_array_view.hpp"
#include "physics/sim_model.hpp"
#include "py_array_view.hpp"
#ifdef KANGENGINE_USE_PHYSX
#include "bridge/physics_bridge.hpp"
#include "physics/articulation.hpp"
#include "engine/scene/scene_backend.hpp"
#include "engine/scene/native/prim.hpp"
#include "bridge/articulation_visual_bridge.hpp"
#endif
#ifdef KANGENGINE_USE_CUDA
#include "sim/gpu_transform_kernels.hpp"
#endif

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

void bind_sim_visual(py::module& m) {
    using namespace KE::Sim;

#ifdef KANGENGINE_USE_CUDA
    m.def("indexed_rigid_state_to_mat4_cuda",
          &launchIndexedRigidStateToMat4CUDA, py::arg("rigid_state"),
          py::arg("rigid_rows"), py::arg("transforms"),
          "Gather selected rigid state rows into CUDA Mat4 transforms.");
    m.def("rigid_state_to_mapped_mat4_cuda", &launchRigidStateToMappedMat4CUDA,
          py::arg("rigid_state"), py::arg("rigid_rows"),
          py::arg("local_transforms"), py::arg("mapped_transforms"),
          "Compose selected rigid poses with host column-major [P, 16] local "
          "matrices directly into mapped graphics buffers. Inputs produced on "
          "another CUDA stream require ready events.");
    m.def("articulation_link_state_to_mat4_cuda",
          &launchArticulationLinkStateToMat4CUDA,
          py::arg("articulation_link_state"), py::arg("articulation_rows"),
          py::arg("link_indices"), py::arg("transforms"), py::arg("link_count"),
          "Gather articulation link state into link-major CUDA Mat4 "
          "transforms.");
    m.def("articulation_link_state_to_mapped_mat4_cuda",
          &launchArticulationLinkStateToMappedMat4CUDA,
          py::arg("articulation_link_state"), py::arg("articulation_rows"),
          py::arg("link_indices"), py::arg("mapped_transforms"),
          "Write articulation link transforms directly into mapped graphics "
          "buffers.");
#endif

    py::class_<KE::SimModel>(
        m, "SimModel",
        "Low-level C++ simulation topology for engine-owned renderer batch "
        "paths. This API intentionally deals in internal renderable handles.")
        .def(py::init<>())
        .def("set_body_renderables", &KE::SimModel::setBodyRenderables,
             py::arg("renderable_handles"),
             "Compatibility helper for one shape per body. Expects internal "
             "renderable handles from low-level renderer paths.")
        .def(
            "add_shape",
            [](KE::SimModel& self, int bodyId, KE::RenderableHandle renderable,
               py::sequence localPos, py::sequence localRot,
               const std::string& name, py::sequence localScale) {
                return self.addShape(
                    bodyId, renderable, vec3FromSequence(localPos),
                    quatFromXYZWSequence(localRot), name,
                    vec3FromSequence(localScale, "local_scale"));
            },
            py::arg("body_id"), py::arg("renderable_handle"),
            py::arg("local_pos") = py::make_tuple(0.0f, 0.0f, 0.0f),
            py::arg("local_quat_xyzw") = py::make_tuple(0.0f, 0.0f, 0.0f, 1.0f),
            py::arg("name") = "", py::kw_only(),
            py::arg("local_scale") = py::make_tuple(1.0f, 1.0f, 1.0f),
            "Add one shape-to-renderable mapping for native visual batches. "
            "Regular scene objects should be authored through app.scene "
            "instead.")
        .def("add_object_boundary", &KE::SimModel::addObjectBoundary,
             py::arg("body_start"), py::arg("body_count"), py::arg("name") = "")
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
        .def(
            "set_body_transform",
            [](KE::SimState& self, int envId, int bodyId, py::sequence pos,
               py::sequence rot) {
                self.setBodyTransform(envId, bodyId, vec3FromSequence(pos),
                                      quatFromXYZWSequence(rot));
            },
            py::arg("env_id"), py::arg("body_id"), py::arg("pos"),
            py::arg("quat_xyzw"))
        .def("get_body_pos",
             [](const KE::SimState& self, int envId, int bodyId) {
                 return vec3Tuple(self.bodyPos[static_cast<size_t>(
                     self.bodyIndex(envId, bodyId))]);
             })
        .def("get_body_quat_xyzw",
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
        "Low-level C++ SimState-to-renderable transform batch. This is the "
        "native backend for ExternalBuffer visual sync, not the high-level "
        "viewer wrapper.")
        .def(py::init<>())
        .def("clear", &KE::SimVisualBatch::clear)
        .def("set_model",
             [](KE::SimVisualBatch& self, const KE::SimModel& model) {
                 self.setModel(&model);
             })
        .def("prepare_from_state", &KE::SimVisualBatch::prepareFromState)
        .def_property_readonly("renderable_count",
                               &KE::SimVisualBatch::renderableCount)
        .def("renderable", &KE::SimVisualBatch::renderable, py::arg("shape_id"),
             "Return the internal renderable handle for a native batch shape.")
        .def("transforms",
             [](const KE::SimVisualBatch& self, int shapeId) {
                 return mat4List(self.transforms(shapeId));
             })
        .def("external_transform_desc",
             &KE::SimVisualBatch::externalTransformDesc, py::arg("shape_id"),
             py::arg("version"),
             py::arg("name") = "sim_visual_batch_transforms");

#ifdef KANGENGINE_USE_PHYSX
    using KE::Bridge::PhysicsBridge;
    auto physics = m.attr("physics").cast<py::module>();

    // PhysX-to-SceneGraph synchronization and collision visuals.
    py::class_<PhysicsBridge>(
        physics, "PhysicsBridge",
        "Syncs PhysX articulation state into KangEngine scene/render visuals.")
        .def(py::init<>(), "Create a scene-graph physics bridge.")
        .def("add", &PhysicsBridge::add, py::arg("artic"),
             py::arg("skel_bridge"), py::keep_alive<1, 2>(),
             py::keep_alive<1, 3>(),
             "Connect an articulation to an ArticulationVisualBridge.")
        .def("sync", &PhysicsBridge::sync,
             "Copy latest physics transforms into connected visuals.")
        .def("set_collision_visible", &PhysicsBridge::setCollisionVisible,
             py::arg("visible"), "Show or hide collision visual prims.")
        .def(
            "add_collision_visuals",
            [](PhysicsBridge& self, const KE::Articulation& artic,
               KE::Scene::SceneBackend* scene, const std::string& basePath,
               bool visibleByDefault) {
                return self.addCollisionVisuals(artic, scene, basePath,
                                                visibleByDefault);
            },
            py::arg("artic"), py::arg("scene"), py::arg("path") = "/collision",
            py::arg("visible_by_default") = false,
            py::return_value_policy::reference, py::keep_alive<1, 2>(),
            py::keep_alive<1, 3>(),
            "Create scene prims that visualize articulation collision shapes.");
#endif
}
