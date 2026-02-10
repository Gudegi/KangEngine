///
/// Animation System Python Bindings
/// Robot, SkeletonTree, SkeletonState
///

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>

#include "animation/robot.hpp"
#include "animation/skeleton_math.hpp"
#include "animation/skeleton_state.hpp"
#include "animation/skeleton_tree.hpp"
#include "scene/scene_backend.hpp"
#include "scene/native/prim.hpp"
#include "app/app.hpp"

namespace py = pybind11;
using namespace KE;
using namespace KE::Animation;

void bind_animation(py::module& m) {
    py::module anim =
        m.def_submodule("animation", "Skeleton animation system");

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
        .def_property_readonly("rotation",
                               [](const Transform& t) {
                                   return toGlm(t.rotation);
                               })
        .def_property_readonly("translation",
                               [](const Transform& t) {
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
        .def("rotation",
             [](const SkeletonState& self, int i) {
                 return toGlm(self.rotation(i));
             })
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
        .def("print_global_positions",
             &SkeletonState::printGlobalPositions);

    // Robot
    py::class_<Robot>(anim, "Robot")
        .def_static(
            "from_mjcf",
            [](const std::string& mjcfPath,
               Scene::SceneBackend* scene, Backend::Shader* shader,
               App* app, const std::string& primBasePath,
               float scale) {
                return Robot::fromMJCF(mjcfPath, scene, shader, app,
                                       primBasePath, scale);
            },
            py::arg("mjcf_path"), py::arg("scene"),
            py::arg("shader"), py::arg("app"),
            py::arg("prim_base_path") = "/robot",
            py::arg("scale") = 1.0f)
        .def("apply_pose", &Robot::applyPose)
        .def("set_joint_rotation",
             [](Robot& self, int idx, const glm::quat& q) {
                 self.setJointRotation(idx, fromGlm(q));
             })
        .def("set_root_translation",
             [](Robot& self, const glm::vec3& t) {
                 self.setRootTranslation(fromGlm(t));
             })
        .def("reset_to_zero_pose", &Robot::resetToZeroPose)
        .def("skeleton", &Robot::skeleton,
             py::return_value_policy::reference_internal)
        .def("state",
             py::overload_cast<>(&Robot::state),
             py::return_value_policy::reference_internal)
        .def("body_prim", &Robot::bodyPrim,
             py::return_value_policy::reference)
        .def("num_bodies", &Robot::numBodies);
}
