#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "py_array_view.hpp"
#include "animation/skeleton_math.hpp"
#include "bridge/articulation_visual_bridge.hpp"
#include "bridge/skeletal_visual_bridge.hpp"
#include "engine/core/app/app.hpp"

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

Eigen::Quaternionf eigenQuatWxyzFromArray(const FloatArray& array,
                                          const char* name) {
    auto view = vec4ArrayView(array, name);
    if (view.count != 1)
        throw py::value_error(std::string(name) + " expected shape [4]");
    return Eigen::Quaternionf(view.data[0], view.data[1], view.data[2],
                              view.data[3]);
}

} // namespace

void bind_animation_visual(py::module& m) {
    py::module anim = m.attr("animation");

    // ArticulationVisual / ArticulationVisualAsset
    py::class_<ArticulationVisualBridge>(
        anim, "ArticulationVisual",
        "Viewer-side articulated rigid-link visual object.")
        .def_static(
            "from_mjcf",
            [](const std::string& mjcfPath, Scene::SceneBackend* scene,
               const std::string& primBasePath, float scale,
               const std::string& order, const std::string& meshAssetBasePath,
               const std::string& targetCoordinateSystem) {
                return ArticulationVisualBridge::fromMJCF(
                    mjcfPath, scene, primBasePath, scale, order,
                    meshAssetBasePath,
                    Utils::coordinateSystemFromString(targetCoordinateSystem));
            },
            py::arg("mjcf_path"), py::arg("scene"), py::arg("path") = "/robot",
            py::arg("scale") = 1.0f, py::arg("order") = "DFS",
            py::arg("mesh_asset_base_path") = "",
            py::arg("target_coordinate_system") = "z_up_x_forward",
            "Create an articulation visual and scene prims from an MJCF file.")
        .def_static(
            "from_urdf",
            [](const std::string& urdfPath, Scene::SceneBackend* scene,
               const std::string& primBasePath, float scale,
               const std::string& order, const std::string& meshAssetBasePath,
               const std::string& targetCoordinateSystem) {
                return ArticulationVisualBridge::fromURDF(
                    urdfPath, scene, primBasePath, scale, order,
                    meshAssetBasePath,
                    Utils::coordinateSystemFromString(targetCoordinateSystem));
            },
            py::arg("urdf_path"), py::arg("scene"), py::arg("path") = "/robot",
            py::arg("scale") = 1.0f, py::arg("order") = "DFS",
            py::arg("mesh_asset_base_path") = "",
            py::arg("target_coordinate_system") = "z_up_x_forward",
            "Create an articulation visual and scene prims from a URDF file.")
        .def("apply_pose", &ArticulationVisualBridge::applyPose,
             "Apply the current skeleton/body state to scene prims.")
        .def(
            "set_joint_quat_wxyz",
            [](ArticulationVisualBridge& self, int idx, const FloatArray& q) {
                self.setJointRotation(idx,
                                      eigenQuatWxyzFromArray(q, "quat_wxyz"));
            },
            py::arg("index"), py::arg("quat_wxyz"),
            "Set the indexed node's quaternion from WXYZ values in the state "
            "frame.")
        .def(
            "set_joint_quat_wxyz",
            [](ArticulationVisualBridge& self, int idx, const glm::quat& q) {
                self.setJointRotation(idx, fromGlm(q));
            },
            py::arg("index"), py::arg("quat_wxyz"),
            "Set a joint rotation from a quaternion.")
        .def(
            "set_root_translation",
            [](ArticulationVisualBridge& self, const FloatArray& t) {
                self.setRootTranslation(
                    eigenVec3FromArray(t, "root_translation"));
            },
            py::arg("translation"), "Set the root translation from an array.")
        .def(
            "set_root_translation",
            [](ArticulationVisualBridge& self, const glm::vec3& t) {
                self.setRootTranslation(fromGlm(t));
            },
            py::arg("translation"), "Set the root translation from a vec3.")
        .def("reset_to_zero_pose", &ArticulationVisualBridge::resetToZeroPose,
             "Reset all joint rotations and root translation to zero pose.")
        .def("set_pickable", &ArticulationVisualBridge::setPickable,
             py::arg("pickable"),
             "Set whether rendered geometry can be picked.")
        .def(
            "skeleton",
            [](ArticulationVisualBridge& self) -> const SkeletonTree& {
                return self.fk().skeleton();
            },
            py::return_value_policy::reference_internal,
            "Return the bridged skeleton tree.")
        .def(
            "state",
            [](ArticulationVisualBridge& self) -> SkeletonState& {
                return self.fk().state();
            },
            py::return_value_policy::reference_internal,
            "Return the mutable current skeleton state.")
        .def("body_prim", &ArticulationVisualBridge::bodyPrim, py::arg("index"),
             py::return_value_policy::reference,
             "Return the scene prim for a body index.")
        .def("body_prims", &ArticulationVisualBridge::bodyPrims,
             py::return_value_policy::reference_internal,
             "Return all body scene prims.")
        .def("render_prims", &ArticulationVisualBridge::renderPrims,
             py::return_value_policy::reference_internal,
             "Return actual renderable mesh prims.")
        .def("render_prim_body_indices",
             &ArticulationVisualBridge::renderPrimBodyIndices,
             py::return_value_policy::reference_internal,
             "Return body index for each render prim.")
        .def("render_prim_visual_indices",
             &ArticulationVisualBridge::renderPrimVisualIndices,
             py::return_value_policy::reference_internal,
             "Return descriptor visual indices, or -1 for collision fallback.")
        .def("num_bodies", &ArticulationVisualBridge::numBodies,
             "Return the number of bridged bodies.");

    py::class_<ArticulationVisualBridgeAsset>(
        anim, "ArticulationVisualAsset",
        "Reusable articulated rigid-link visual asset that can instantiate "
        "scene prims.")
        .def_static("from_data", &ArticulationVisualBridgeAsset::fromData,
                    py::arg("data"), py::arg("scale") = 1.0f,
                    py::arg("asset_path") = "",
                    "Build reusable bridge asset data from a parsed "
                    "articulation description.")
        .def_static(
            "from_mjcf",
            [](const std::string& path, float scale, const std::string& order,
               const std::string& targetCoordinateSystem) {
                return ArticulationVisualBridgeAsset::fromMJCF(
                    path, scale, order,
                    Utils::coordinateSystemFromString(targetCoordinateSystem));
            },
            py::arg("mjcf_path"), py::arg("scale") = 1.0f,
            py::arg("order") = "DFS",
            py::arg("target_coordinate_system") = "z_up_x_forward",
            "Load reusable bridge asset data from an MJCF file.")
        .def_static(
            "from_urdf",
            [](const std::string& path, float scale, const std::string& order,
               const std::string& targetCoordinateSystem) {
                return ArticulationVisualBridgeAsset::fromURDF(
                    path, scale, order,
                    Utils::coordinateSystemFromString(targetCoordinateSystem));
            },
            py::arg("urdf_path"), py::arg("scale") = 1.0f,
            py::arg("order") = "DFS",
            py::arg("target_coordinate_system") = "z_up_x_forward",
            "Load reusable bridge asset data from a URDF file.")
        .def("define_mesh_assets",
             &ArticulationVisualBridgeAsset::defineMeshAssets, py::arg("scene"),
             py::arg("mesh_asset_base_path"),
             py::arg("split_visual_geoms") = false,
             "Define shared mesh asset prims in a scene.")
        .def("instantiate", &ArticulationVisualBridgeAsset::instantiate,
             py::arg("scene"), py::arg("path") = "/robot",
             py::arg("mesh_asset_base_path") = "",
             py::arg("split_visual_geoms") = false,
             py::arg("hierarchical") = false,
             "Instantiate this asset; hierarchical preserves source-node "
             "parenting.")
        .def("num_bodies", &ArticulationVisualBridgeAsset::numBodies,
             "Return the number of bodies in this asset.");

    py::class_<SkeletalVisualConfig>(
        anim, "SkeletalVisualConfig",
        "Style settings for SkeletalVisual line/joint rendering.")
        .def(py::init([](const std::optional<glm::vec4>& boneColor,
                         const std::optional<glm::vec4>& jointColor,
                         float boneRadius, float jointRadius, int segments,
                         bool visible, bool showJoints) {
                 SkeletalVisualConfig config;
                 if (boneColor)
                     config.boneColor = *boneColor;
                 if (jointColor)
                     config.jointColor = *jointColor;
                 config.boneRadius = boneRadius;
                 config.jointRadius = jointRadius;
                 config.segments = segments;
                 config.visible = visible;
                 config.showJoints = showJoints;
                 return config;
             }),
             py::kw_only(), py::arg("bone_color") = py::none(),
             py::arg("joint_color") = py::none(),
             py::arg("bone_radius") = SkeletalVisualConfig{}.boneRadius,
             py::arg("joint_radius") = SkeletalVisualConfig{}.jointRadius,
             py::arg("segments") = SkeletalVisualConfig{}.segments,
             py::arg("visible") = SkeletalVisualConfig{}.visible,
             py::arg("show_joints") = SkeletalVisualConfig{}.showJoints,
             "Create skeleton visual settings from keyword fields.")
        .def_readwrite("bone_color", &SkeletalVisualConfig::boneColor,
                       "RGBA color for bones.")
        .def_readwrite("joint_color", &SkeletalVisualConfig::jointColor,
                       "RGBA color for joints.")
        .def_readwrite("bone_radius", &SkeletalVisualConfig::boneRadius,
                       "Radius used for bone line geometry.")
        .def_readwrite("joint_radius", &SkeletalVisualConfig::jointRadius,
                       "Radius used for joint point geometry.")
        .def_readwrite("segments", &SkeletalVisualConfig::segments,
                       "Segment count for generated round geometry.")
        .def_readwrite("visible", &SkeletalVisualConfig::visible,
                       "Initial visibility state.")
        .def_readwrite("show_joints", &SkeletalVisualConfig::showJoints,
                       "Whether joint markers should be visible.")
        .def("__repr__", [](const SkeletalVisualConfig& config) {
            return py::str(
                       "SkeletalVisualConfig(bone_color=({:g}, {:g}, {:g}, "
                       "{:g}), joint_color=({:g}, {:g}, {:g}, {:g}), "
                       "bone_radius={:g}, joint_radius={:g}, segments={!r}, "
                       "visible={!r}, show_joints={!r})")
                .attr("format")(
                    config.boneColor.x, config.boneColor.y, config.boneColor.z,
                    config.boneColor.w, config.jointColor.x,
                    config.jointColor.y, config.jointColor.z,
                    config.jointColor.w, config.boneRadius, config.jointRadius,
                    config.segments, config.visible, config.showJoints);
        });

    py::class_<SkeletalVisualBridge>(
        anim, "SkeletalVisual",
        "Instanced line/point renderer for skeleton poses and motion clips.")
        .def(py::init<>(), "Create an empty skeletal visual.")
        .def_static(
            "define",
            [](App* app, Material* material, const std::string& basePath,
               const SkeletonState& state, const SkeletalVisualConfig& config) {
                return SkeletalVisualBridge::define(app, material, basePath,
                                                    state, config);
            },
            py::arg("app"), py::arg("material"), py::arg("path"),
            py::arg("state"), py::arg("config") = SkeletalVisualConfig{},
            "Create skeleton visuals from a SkeletonState.")
        .def_static(
            "define",
            [](App* app, Material* material, const std::string& basePath,
               const SkeletonMotion& motion, float time, bool loop,
               const SkeletalVisualConfig& config) {
                return SkeletalVisualBridge::define(app, material, basePath,
                                                    motion, time, loop, config);
            },
            py::arg("app"), py::arg("material"), py::arg("path"),
            py::arg("motion"), py::arg("time") = 0.0f, py::arg("loop") = true,
            py::arg("config") = SkeletalVisualConfig{},
            "Create skeleton visuals by sampling a SkeletonMotion.")
        .def("apply_state", &SkeletalVisualBridge::applyState, py::arg("state"),
             "Update visuals from a SkeletonState.")
        .def("apply_motion", &SkeletalVisualBridge::applyMotion,
             py::arg("motion"), py::arg("time"), py::arg("loop") = true,
             "Update visuals by sampling a SkeletonMotion.")
        .def("set_visible", &SkeletalVisualBridge::setVisible,
             py::arg("visible"), "Set visibility for all skeleton visuals.")
        .def("set_show_joints", &SkeletalVisualBridge::setShowJoints,
             py::arg("show_joints"), "Show or hide joint markers.")
        .def("set_pickable", &SkeletalVisualBridge::setPickable,
             py::arg("pickable"),
             "Set whether skeleton geometry can be picked.")
        .def("set_bone_pickable", &SkeletalVisualBridge::setBonePickable,
             py::arg("pickable"), "Set whether bone geometry can be picked.")
        .def("set_joint_pickable", &SkeletalVisualBridge::setJointPickable,
             py::arg("pickable"), "Set whether joint markers can be picked.")
        .def("remove", &SkeletalVisualBridge::remove,
             "Remove all authored skeleton visual scene objects.")
        .def("bone_handle", &SkeletalVisualBridge::boneHandle,
             "Return the renderable handle used for bones.")
        .def("joint_handle", &SkeletalVisualBridge::jointHandle,
             "Return the renderable handle used for joints.");
}
