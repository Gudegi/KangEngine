///
/// Animation System Python Bindings
/// MJCFLoader, CharacterData, SkeletonTree, SkeletonState, SkeletonBridge
///

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <glm/glm.hpp>
#include <glm/gtc/quaternion.hpp>
#include "py_array_view.hpp"

#include "animation/character_description.hpp"
#include "animation/mjcf_loader.hpp"
#include "animation/skeleton_math.hpp"
#include "animation/skeleton_state.hpp"
#include "animation/skeleton_tree.hpp"
#include "bridge/skeleton_bridge.hpp"
#include "engine/scene/native/prim.hpp"
#include "engine/scene/scene_backend.hpp"

namespace py = pybind11;
using namespace KE;
using namespace KE::Animation;
using namespace KE::Bridge;

namespace {

Eigen::Vector3f eigenVec3FromArray(const FloatArray& array, const char* name) {
    auto view = vec3ArrayView(array, name);
    if (view.count != 1)
        throw py::value_error(std::string(name) + " expected shape [3]");
    return Eigen::Vector3f(view.data[0], view.data[1], view.data[2]);
}

Eigen::Quaternionf eigenQuatXyzwFromArray(const FloatArray& array,
                                          const char* name) {
    auto view = vec4ArrayView(array, name);
    if (view.count != 1)
        throw py::value_error(std::string(name) + " expected shape [4]");
    return Eigen::Quaternionf(view.data[3], view.data[0], view.data[1],
                              view.data[2]);
}

std::vector<Eigen::Quaternionf>
eigenQuatXyzwArray(const FloatArray& array, const char* name) {
    auto view = vec4ArrayView(array, name);
    std::vector<Eigen::Quaternionf> result;
    result.reserve(view.count);
    for (size_t i = 0; i < view.count; ++i) {
        const float* q = view.data + i * 4;
        result.emplace_back(q[3], q[0], q[1], q[2]);
    }
    return result;
}

} // namespace

void bind_animation(py::module& m) {
    py::module anim = m.def_submodule("animation", "Skeleton animation system");

    // Joint (from character_description.hpp)
    py::class_<Joint>(anim, "Joint")
        .def_readonly("name", &Joint::name)
        .def_readonly("lo_limit", &Joint::loLimit)
        .def_readonly("hi_limit", &Joint::hiLimit)
        .def_property_readonly("axis",
                               [](const Joint& j) { return toGlm(j.axis); });

    // MeshInfo
    py::class_<MeshInfo>(anim, "MeshInfo")
        .def_readonly("body_name", &MeshInfo::bodyName)
        .def_readonly("mesh_file", &MeshInfo::meshFile)
        .def_readonly("body_index", &MeshInfo::bodyIndex);

    py::enum_<CollisionGeom::Type>(anim, "CollisionGeomType")
        .value("Capsule", CollisionGeom::Type::Capsule)
        .value("Cylinder", CollisionGeom::Type::Cylinder)
        .value("Sphere", CollisionGeom::Type::Sphere)
        .value("Box", CollisionGeom::Type::Box)
        .export_values();

    py::class_<CollisionGeom>(anim, "CollisionGeom")
        .def_readonly("type", &CollisionGeom::type)
        .def_property_readonly("pos",
                               [](const CollisionGeom& g) {
                                   return toGlm(g.pos);
                               })
        .def_property_readonly("quat",
                               [](const CollisionGeom& g) {
                                   return toGlm(g.quat);
                               })
        .def_property_readonly(
            "size",
            [](const CollisionGeom& g) {
                return std::vector<float>{g.size[0], g.size[1], g.size[2]};
            })
        .def_readonly("has_from_to", &CollisionGeom::hasFromTo)
        .def_property_readonly("from_pos",
                               [](const CollisionGeom& g) {
                                   return toGlm(g.from);
                               })
        .def_property_readonly("to_pos",
                               [](const CollisionGeom& g) {
                                   return toGlm(g.to);
                               })
        .def_readonly("friction", &CollisionGeom::friction)
        .def_readonly("condim", &CollisionGeom::condim)
        .def_readonly("margin", &CollisionGeom::margin);

    // CharacterData — aggregate returned by MJCFLoader::load()
    py::class_<CharacterData>(anim, "CharacterData")
        .def_readonly("skeleton_tree", &CharacterData::skeletonTree)
        .def_readonly("mesh_infos", &CharacterData::meshInfos)
        .def_readonly("mesh_dir", &CharacterData::meshDir)
        // joints: return dict[int, list[Joint]]
        .def_property_readonly("joints", [](const CharacterData& d) {
            py::dict result;
            for (const auto& [idx, jvec] : d.joints)
                result[py::int_(idx)] = jvec;
            return result;
        })
        .def_property_readonly("collision_geoms", [](const CharacterData& d) {
            py::dict result;
            for (const auto& [idx, geoms] : d.collisionGeoms)
                result[py::int_(idx)] = geoms;
            return result;
        });

    // MJCFLoader
    py::class_<MJCFLoader>(anim, "MJCFLoader")
        .def_static("load", &MJCFLoader::load, py::arg("mjcf_path"),
                    py::arg("scale") = 1.0f, py::arg("order") = "DFS");

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
        .def_static(
            "from_rotation_and_root_translation",
            [](std::shared_ptr<SkeletonTree> tree, const FloatArray& rotations,
               const FloatArray& rootTranslation, bool isLocal) {
                auto rot = eigenQuatXyzwArray(rotations, "rotations");
                if (static_cast<int>(rot.size()) != tree->numJoints()) {
                    throw py::value_error(
                        "rotations must have shape [num_joints, 4]");
                }
                return SkeletonState::fromRotationAndRootTranslation(
                    tree, rot,
                    eigenVec3FromArray(rootTranslation, "root_translation"),
                    isLocal);
            },
            py::arg("tree"), py::arg("rotations"),
            py::arg("root_translation"), py::arg("is_local") = true)
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
             [](SkeletonState& self, int i, const FloatArray& q) {
                 self.setRotation(i, eigenQuatXyzwFromArray(q, "rotation"));
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
             [](SkeletonState& self, const FloatArray& t) {
                 self.setRootTranslation(
                     eigenVec3FromArray(t, "root_translation"));
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
               const std::string& order,
               const std::string& meshAssetBasePath) {
                return SkeletonBridge::fromMJCF(mjcfPath, scene, primBasePath,
                                                scale, order,
                                                meshAssetBasePath);
            },
            py::arg("mjcf_path"), py::arg("scene"),
            py::arg("prim_base_path") = "/robot", py::arg("scale") = 1.0f,
            py::arg("order") = "DFS",
            py::arg("mesh_asset_base_path") = "")
        .def("apply_pose", &SkeletonBridge::applyPose)
        .def("set_joint_rotation",
             [](SkeletonBridge& self, int idx, const FloatArray& q) {
                 self.setJointRotation(idx,
                                       eigenQuatXyzwFromArray(q, "rotation"));
             })
        .def("set_joint_rotation",
             [](SkeletonBridge& self, int idx, const glm::quat& q) {
                 self.setJointRotation(idx, fromGlm(q));
             })
        .def("set_root_translation",
             [](SkeletonBridge& self, const FloatArray& t) {
                 self.setRootTranslation(
                     eigenVec3FromArray(t, "root_translation"));
             })
        .def("set_root_translation",
             [](SkeletonBridge& self, const glm::vec3& t) {
                 self.setRootTranslation(fromGlm(t));
             })
        .def("reset_to_zero_pose", &SkeletonBridge::resetToZeroPose)
        .def(
            "skeleton",
            [](SkeletonBridge& self) -> const SkeletonTree& {
                return self.fk().skeleton();
            },
            py::return_value_policy::reference_internal)
        .def(
            "state",
            [](SkeletonBridge& self) -> SkeletonState& {
                return self.fk().state();
            },
            py::return_value_policy::reference_internal)
        .def("body_prim", &SkeletonBridge::bodyPrim,
             py::return_value_policy::reference)
        .def("body_prims", &SkeletonBridge::bodyPrims,
             py::return_value_policy::reference_internal)
        .def("num_bodies", &SkeletonBridge::numBodies);

    py::class_<SkeletonBridgeAsset>(anim, "SkeletonBridgeAsset")
        .def_static("from_mjcf", &SkeletonBridgeAsset::fromMJCF,
                    py::arg("mjcf_path"), py::arg("scale") = 1.0f,
                    py::arg("order") = "DFS")
        .def("define_mesh_assets", &SkeletonBridgeAsset::defineMeshAssets,
             py::arg("scene"), py::arg("mesh_asset_base_path"))
        .def("instantiate", &SkeletonBridgeAsset::instantiate,
             py::arg("scene"), py::arg("prim_base_path") = "/robot",
             py::arg("mesh_asset_base_path") = "")
        .def("num_bodies", &SkeletonBridgeAsset::numBodies);
}
