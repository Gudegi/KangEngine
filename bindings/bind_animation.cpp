///
/// Animation System Python Bindings
/// Robot, SkeletonTree, SkeletonState
///

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "bridge/skeleton_bridge.hpp"
#include "animation/skeleton_math.hpp"
#include "animation/skeleton_state.hpp"
#include "animation/skeleton_tree.hpp"
#include "engine/scene/scene_backend.hpp"
#include "engine/scene/native/prim.hpp"

namespace py = pybind11;
using namespace KE;
using namespace KE::Animation;
using namespace KE::Bridge;

void bind_animation(py::module& m) {
    py::module anim = m.def_submodule("animation", "Skeleton animation system");

    // SkeletonTree (read-only after construction)
    py::class_<SkeletonTree, std::shared_ptr<SkeletonTree>>(anim,
                                                            "SkeletonTree")
        .def("num_joints", &SkeletonTree::numJoints)
        .def("node_name", &SkeletonTree::nodeName)
        .def("parent_index", &SkeletonTree::parentIndex)
        .def("local_translation",
             [](const SkeletonTree& self, int i) {
                 return toGlm(self.localTranslation(i));
             })
        .def("node_names", &SkeletonTree::nodeNames)
        .def("parent_indices", &SkeletonTree::parentIndices)
        .def("print", &SkeletonTree::print);

    // Transform (FK result)
    py::class_<Transform>(anim, "Transform")
        .def_property_readonly(
            "rotation", [](const Transform& t) { return toGlm(t.rotation); })
        .def_property_readonly("translation", [](const Transform& t) {
            return toGlm(t.translation);
        });

    // SkeletonState
    py::class_<SkeletonState>(anim, "SkeletonState")
        .def("num_joints", &SkeletonState::numJoints)
        .def("is_local", &SkeletonState::isLocal)
        .def("compute_global_transforms",
             &SkeletonState::computeGlobalTransforms)
        .def("compute_global_positions",
             [](const SkeletonState& self) {
                 auto positions = self.computeGlobalPositions();
                 std::vector<glm::vec3> result;
                 result.reserve(positions.size());
                 for (const auto& p : positions) {
                     result.push_back(toGlm(p));
                 }
                 return result;
             })
        .def("rotation", [](const SkeletonState& self,
                            int i) { return toGlm(self.rotation(i)); })
        .def("set_rotation",
             [](SkeletonState& self, int i, const glm::quat& q) {
                 self.setRotation(i, fromGlm(q));
             })
        .def("root_translation",
             [](const SkeletonState& self) {
                 return toGlm(self.rootTranslation());
             })
        .def("set_root_translation",
             [](SkeletonState& self, const glm::vec3& t) {
                 self.setRootTranslation(fromGlm(t));
             })
        .def("print_global_positions", &SkeletonState::printGlobalPositions);

    // SkeletonBridge
    py::class_<SkeletonBridge>(anim, "SkeletonBridge")
        .def_static(
            "from_mjcf",
            [](const std::string& mjcfPath, Scene::SceneBackend* scene,
               const std::string& primBasePath, float scale,
               const std::string& order) {
                return SkeletonBridge::fromMJCF(mjcfPath, scene, primBasePath,
                                                scale, order);
            },
            py::arg("mjcf_path"), py::arg("scene"),
            py::arg("prim_base_path") = "/robot", py::arg("scale") = 1.0f,
            py::arg("order") = "DFS")
        .def("apply_pose", &SkeletonBridge::applyPose)
        .def("set_joint_rotation",
             [](SkeletonBridge& self, int idx, const glm::quat& q) {
                 self.setJointRotation(idx, fromGlm(q));
             })
        .def("set_root_translation",
             [](SkeletonBridge& self, const glm::vec3& t) {
                 self.setRootTranslation(fromGlm(t));
             })
        .def("reset_to_zero_pose", &SkeletonBridge::resetToZeroPose)
        .def("skeleton",
             [](SkeletonBridge& self) -> const SkeletonTree& {
                 return self.fk().skeleton();
             },
             py::return_value_policy::reference_internal)
        .def("state",
             [](SkeletonBridge& self) -> SkeletonState& {
                 return self.fk().state();
             },
             py::return_value_policy::reference_internal)
        .def("body_prim", &SkeletonBridge::bodyPrim,
             py::return_value_policy::reference)
        .def("body_prims", &SkeletonBridge::bodyPrims,
             py::return_value_policy::reference_internal)
        .def("num_bodies", &SkeletonBridge::numBodies);
}
